import sys
import time
from pathlib import Path
import argparse
from typing import Optional
import struct

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
)
from PyQt6.QtGui import QFont
import pyqtgraph.opengl as gl
from zstandard import ZstdDecompressor

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn


class PoseCNN(nn.Module):
    """
    CNN for 2-channel fixed-size radar density grids -> pose class logits.
    Channel 0 = Y-Z view, channel 1 = X-Z view (see prepare_dataset.py).
    Architecture must exactly match train_pose_classifier.py, since we're
    loading its saved weights.
    """

    def __init__(self, num_classes: int, grid_size: int, in_channels: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        pooled_size = grid_size // 8
        flat_features = 128 * pooled_size * pooled_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(flat_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class PoseClassifier:
    """
    Wraps the trained checkpoint: holds the model plus the exact grid_size /
    view bounds used at training time, so live rasterization matches what
    the model was trained on.
    """

    def __init__(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.label_names = list(ckpt["label_names"])
        self.grid_size = int(ckpt["grid_size"])
        self.in_channels = int(ckpt.get("in_channels", 2))
        self.yz_lo = np.asarray(ckpt["grid_bounds_yz_lo"], dtype=np.float64)
        self.yz_hi = np.asarray(ckpt["grid_bounds_yz_hi"], dtype=np.float64)
        self.xz_lo = np.asarray(ckpt["grid_bounds_xz_lo"], dtype=np.float64)
        self.xz_hi = np.asarray(ckpt["grid_bounds_xz_hi"], dtype=np.float64)

        self.model = PoseCNN(num_classes=len(self.label_names), grid_size=self.grid_size,
                              in_channels=self.in_channels)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def rasterize(self, points_2d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """
        Histogram a (N, 2) point set into a grid_size x grid_size density map
        using the SAVED training-time bounds, so grid cells mean the same
        physical thing as during training.
        """
        hist, _, _ = np.histogram2d(
            points_2d[:, 0], points_2d[:, 1],
            bins=self.grid_size,
            range=[[lo[0], hi[0]], [lo[1], hi[1]]],
        )
        if hist.sum() > 0:
            hist = hist / hist.sum()
        return hist.astype(np.float32)

    def predict(self, xyz_points: np.ndarray):
        """
        Builds the 2-channel (Y-Z, X-Z) input from a raw (N, 3) X-Y-Z point
        set and returns (predicted_label: str, confidence: float, all_probs: dict).
        """
        yz = self.rasterize(xyz_points[:, [1, 2]], self.yz_lo, self.yz_hi)
        xz = self.rasterize(xyz_points[:, [0, 2]], self.xz_lo, self.xz_hi)
        grid = np.stack([yz, xz], axis=0)  # (2, H, W)
        tensor = torch.from_numpy(grid).unsqueeze(0).float()  # (1, 2, H, W)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        pred_label = self.label_names[pred_idx]
        confidence = float(probs[pred_idx].item())
        all_probs = {name: float(p) for name, p in zip(self.label_names, probs)}
        return pred_label, confidence, all_probs


class RadarPlotter(QMainWindow):
    """
    This class plots the camera data along with the radar scan, plus the
    live pose prediction when a classifier is attached.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Plotter")
        self.resize(1600, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)

        # Prediction banner sits above the two views
        self.prediction_label = QLabel("Pose: —")
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.prediction_label.setStyleSheet("padding: 6px;")
        outer_layout.addWidget(self.prediction_label)

        display_widget = QWidget()
        display_layout = QHBoxLayout(display_widget)
        outer_layout.addWidget(display_widget)

        # Settings for the axes and grid
        axis_size = 12
        tick_spacing = 1

        self.view1 = gl.GLViewWidget()
        self.view1.setCameraPosition(distance=45, elevation=20, azimuth=45)
        display_layout.addWidget(self.view1)

        # Add synchronized grid
        grid1 = gl.GLGridItem()
        grid1.setSize(x=axis_size * 2, y=axis_size * 2)
        grid1.setSpacing(x=tick_spacing, y=tick_spacing)
        self.view1.addItem(grid1)

        self.scatter1 = gl.GLScatterPlotItem(
            size=2,
            pxMode=True,
        )
        self.view1.addItem(self.scatter1)
        self.add_3d_axes_with_ticks(self.view1, size=axis_size, spacing=tick_spacing)

        self.view2 = gl.GLViewWidget()
        self.view2.setCameraPosition(distance=5, elevation=0, azimuth=0)
        display_layout.addWidget(self.view2)

        self.scatter2 = gl.GLScatterPlotItem(
            size=2,
            pxMode=True,
        )
        self.view2.addItem(self.scatter2)
        self.add_basic_2d_axes(self.view2, size=axis_size)

    def add_basic_2d_axes(self, view, size):
        """
        Same as the following but only uses Y, Z axes without ticks or labels
        """
        pos = []
        pos.extend([[0, -size, 0], [0, size, 0]])
        pos.extend([[0, 0, -size], [0, 0, size]])

        lines_item = gl.GLLinePlotItem(
            pos=np.array(pos, dtype=float),
            color=(1,0,0,1), # red
            mode="lines",
            width=0.8,
        )
        view.addItem(lines_item)

    def add_3d_axes_with_ticks(self, view, size, spacing):
        """
        Creates custom X, Y, and Z axes stretching from -size to +size,
        adding perpendicular tick marks and numeric labels at set intervals.
        """
        pos = []
        colors = []

        c_x = [1, 0.3, 0.3, 0.5]  # Red
        c_y = [0.3, 1, 0.3, 0.5]  # Green
        c_z = [0.3, 0.3, 1, 0.5]  # Blue

        # 1. Main Axis Lines
        pos.extend([[-size, 0, 0], [size, 0, 0]])
        colors.extend([c_x, c_x])

        pos.extend([[0, -size, 0], [0, size, 0]])
        colors.extend([c_y, c_y])

        pos.extend([[0, 0, -size], [0, 0, size]])
        colors.extend([c_z, c_z])

        tick_font = QFont("Arial", 8)
        title_font = QFont("Arial", 11, QFont.Weight.Bold)
        tick_len = size * 0.03

        for val in range(-size, size + 1, spacing):
            if val == 0:
                continue
            pos.extend([[val, -tick_len, 0], [val, tick_len, 0]])
            colors.extend([c_x, c_x])
            view.addItem(
                gl.GLTextItem(
                    pos=[val, -tick_len * 3, 0], text=str(val), font=tick_font
                )
            )

            pos.extend([[-tick_len, val, 0], [tick_len, val, 0]])
            colors.extend([c_y, c_y])
            view.addItem(
                gl.GLTextItem(
                    pos=[-tick_len * 3, val, 0], text=str(val), font=tick_font
                )
            )

            pos.extend([[-tick_len, 0, val], [tick_len, 0, val]])
            colors.extend([c_z, c_z])
            view.addItem(
                gl.GLTextItem(
                    pos=[-tick_len * 3, 0, val], text=str(val), font=tick_font
                )
            )

        lines_item = gl.GLLinePlotItem(
            pos=np.array(pos, dtype=float),
            color=np.array(colors, dtype=float),
            mode="lines",
            width=0.8,
        )
        view.addItem(lines_item)

        view.addItem(gl.GLTextItem(pos=[size + 1, 0, 0], text="X (m)", font=title_font))
        view.addItem(gl.GLTextItem(pos=[0, size + 1, 0], text="Y (m)", font=title_font))
        view.addItem(gl.GLTextItem(pos=[0, 0, size + 1], text="Z (m)", font=title_font))

    def update_data(
        self, /, *, data1: Optional[NDArray] = None, data2: Optional[NDArray] = None
    ):
        """
        Updates both plots with new data.
        """
        if data1 is not None:
            self.scatter1.setData(pos=data1[:, :3], size=3)

        if data2 is not None:
            self.scatter2.setData(pos=data2[:, :3], size=3)

    def update_prediction(self, label: str, confidence: float):
        """
        Updates the pose prediction banner.
        """
        self.prediction_label.setText(f"Pose: {label}  ({confidence * 100:.1f}%)")


class ReaderParserError(Exception):
    def __init__(self, reason):
        super().__init__(reason)


class DatReader:
    def __init__(self, path: Path):
        self._file_path = path

    def nextFrame(self):
        """
        Parses a ZSTD compressed binary file containing PointCloud messages
        """
        with self._file_path.open("rb") as f:
            dctx = ZstdDecompressor()

            SOF_DELIMITER = b"::"
            EOF_DELIMITER = b";;"
            HEADER_META_SIZE = len(SOF_DELIMITER) + struct.calcsize("<IBI")

            with dctx.stream_reader(f) as reader:
                while True:
                    header = reader.read(HEADER_META_SIZE)
                    if header is None or not header:
                        break
                    elif len(header) < HEADER_META_SIZE:
                        raise EOFError("Unexpected EOF while reading metadata.")
                    try:
                        _, timestamp_us, message_type, payload_length = struct.unpack(
                            "<2sIBI", header
                        )
                    except struct.error as e:
                        raise ReaderParserError(f"Invalid Parse Syntax: {e}")

                    raw_payload = reader.read(payload_length)
                    point_cloud_np = np.frombuffer(raw_payload, dtype=np.int16) / 1000
                    point_cloud_np = point_cloud_np.reshape((-1, 3))

                    footer = reader.read(len(EOF_DELIMITER))
                    if footer != EOF_DELIMITER:
                        raise ValueError(
                            f"Stream corrupted: Expected {EOF_DELIMITER}, got {footer}"
                        )

                    yield {
                        "timestamp_us": timestamp_us,
                        "message_type": message_type,
                        "point_cloud": point_cloud_np,
                    }

            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A Simple Script to read recorded files and Visualize them",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-f",
        "--file",
        metavar="<FILE>",
        type=Path,
        dest="path",
        required=True,
        help="path to file that ends with .bag prefix",
    )

    parser.add_argument(
        "-m",
        "--model",
        metavar="<MODEL.pt>",
        type=Path,
        dest="model_path",
        default=None,
        help="path to a trained pose_classifier.pt checkpoint; if provided, "
             "runs live pose classification on each radar buffer flush",
    )

    return parser.parse_args()


# Defines static boundaries around the movement area
def filter_data(data: NDArray):
    x_bound = (-10, 10)
    y_bound = (-10, 10)
    z_bound = (-10, 10)
    mask = (
        (data[:, 0] >= x_bound[0]) & (data[:, 0] <= x_bound[1]) &
        (data[:, 1] >= y_bound[0]) & (data[:, 1] <= y_bound[1]) &
        (data[:, 2] >= z_bound[0]) & (data[:, 2] <= z_bound[1])
    )

    return data[mask]


def center_data(
        points: np.ndarray
) -> np.ndarray:
    """
    Centres the average data to the point (0,0,1)
    """
    p = np.asarray(points, dtype=np.float64)
    if p.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return p - p.mean(axis=0)


def main() -> int:
    args = parse_args()
    path = args.path

    app = QApplication(sys.argv)

    classifier = None
    if args.model_path is not None:
        classifier = PoseClassifier(args.model_path)
        print(f"Loaded pose classifier from {args.model_path} "
              f"(classes: {classifier.label_names}, in_channels: {classifier.in_channels})")

    dat_reader = DatReader(path)
    plotter = RadarPlotter()
    plotter.show()

    def accumulated_data_plot():
        nonlocal dat_reader
        nonlocal plotter
        current_tick_us = 0
        # Set number of points to accumulate
        max_points = 100
        pending = []
        pending_count = 0
        overflow = np.empty((0, 3), dtype=np.float64)

        # Sets fps limit
        max_render_fps = 30.0
        min_render_interval = 1.0 / max_render_fps
        last_render_time = 0.0

        try:
            for d in dat_reader.nextFrame():
                msg_type = d["message_type"]
                msg = d["point_cloud"]
                new_tick_us = d["timestamp_us"]

                if msg_type == 2:
                    data = center_data(filter_data(msg))
                    pending.append(data)
                    pending_count += len(data)

                    saved_points = len(overflow) + pending_count
                    if saved_points >= max_points:
                        current_points = (
                            np.concatenate([overflow] + pending, axis=0) if pending else overflow
                        )
                        points = current_points.copy()

                        if classifier is not None:
                            label, confidence, _ = classifier.predict(points)
                            plotter.update_prediction(label, confidence)

                        now = time.time()
                        if now - last_render_time >= min_render_interval:
                            points[:, 0] = 0
                            plotter.update_data(data2=points)
                            QApplication.processEvents()
                            last_render_time = now

                        overflow = current_points[max_points:]
                        pending = []
                        pending_count = 0
                elif msg_type == 1:
                    plotter.update_data(data1=msg)

                diff_tick_us = new_tick_us - current_tick_us
                current_tick_us = new_tick_us
                time.sleep(diff_tick_us / 1e6)
            print("Finished")
        except KeyboardInterrupt:
            print("Interupted")
            app.quit()
            return
        return

    QTimer.singleShot(50, accumulated_data_plot)

    try:
        print("Running")
        app.exec()
    except KeyboardInterrupt:
        print("Interupted")
        app.quit()
        pass
    return 0


if __name__ == "__main__":
    main()
