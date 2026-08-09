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


def normalize_and_features(points: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    mask_f = mask.float()
    n_real = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)

    point_norms = points.norm(dim=2)
    point_norms = point_norms.masked_fill(~mask, 0.0)
    raw_scale = point_norms.max(dim=1).values.clamp(min=eps)

    normalized_points = points / raw_scale.view(-1, 1, 1)
    normalized_points = normalized_points * mask_f.unsqueeze(-1)

    mean = (normalized_points * mask_f.unsqueeze(-1)).sum(dim=1) / n_real
    centered = (normalized_points - mean.unsqueeze(1)) * mask_f.unsqueeze(-1)

    var = (centered ** 2).sum(dim=1) / n_real
    std = torch.sqrt(var + eps)

    third_moment = (centered ** 3).sum(dim=1) / n_real
    skew = third_moment / (std ** 3 + eps)

    aux = torch.stack([raw_scale, std[:, 0], std[:, 1], skew[:, 2]], dim=1)
    return normalized_points, aux


class PointCloudNet(nn.Module):
    AUX_FEATURES = 4

    def __init__(self, num_classes: int):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.aux_norm = nn.BatchNorm1d(self.AUX_FEATURES)
        self.classifier = nn.Sequential(
            nn.Linear(256 + self.AUX_FEATURES, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized_points, aux = normalize_and_features(points, mask)

        x = normalized_points.transpose(1, 2)
        x = self.shared_mlp(x)

        mask_expanded = mask.unsqueeze(1).expand(-1, x.size(1), -1)
        x = x.masked_fill(~mask_expanded, float("-inf"))

        pooled, _ = x.max(dim=2)
        pooled = torch.nan_to_num(pooled, neginf=0.0)

        aux = self.aux_norm(aux)

        fused = torch.cat([pooled, aux], dim=1)
        return self.classifier(fused)


class PointCloudClassifier:
    def __init__(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.label_names = list(ckpt["label_names"])

        self.model = PointCloudNet(num_classes=len(self.label_names))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(self, xyz_points: np.ndarray):
        points_t = torch.from_numpy(xyz_points.astype(np.float32)).unsqueeze(0)
        mask_t = torch.ones(1, xyz_points.shape[0], dtype=torch.bool)
        with torch.no_grad():
            logits = self.model(points_t, mask_t)
            probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        pred_label = self.label_names[pred_idx]
        confidence = float(probs[pred_idx].item())
        all_probs = {name: float(p) for name, p in zip(self.label_names, probs)}
        return pred_label, confidence, all_probs


class RadarPlotter(QMainWindow):
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
        if data1 is not None:
            self.scatter1.setData(pos=data1[:, :3], size=3)

        if data2 is not None:
            self.scatter2.setData(pos=data2[:, :3], size=3)

    def update_prediction(self, label: str, confidence: float):
        self.prediction_label.setText(f"Pose: {label}  ({confidence * 100:.1f}%)")


class ReaderParserError(Exception):
    def __init__(self, reason):
        super().__init__(reason)


class DatReader:
    def __init__(self, path: Path):
        self._file_path = path

    def nextFrame(self):
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
        help="path to a trained pointcloud_classifier.pt checkpoint; if provided, "
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
        classifier = PointCloudClassifier(args.model_path)
        print(f"Loaded point-cloud pose classifier from {args.model_path} "
              f"(classes: {classifier.label_names})")

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
