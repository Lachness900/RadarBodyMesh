import sys
import time
from pathlib import Path
import argparse
from typing import Optional
import struct
import threading

from PyQt6.QtCore import Qt
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
    def __init__(self, num_classes: int, grid_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class PoseClassifier:
    def __init__(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.label_names = list(ckpt["label_names"])
        self.grid_size = int(ckpt["grid_size"])
        self.lo = np.asarray(ckpt["grid_bounds_lo"], dtype=np.float64)
        self.hi = np.asarray(ckpt["grid_bounds_hi"], dtype=np.float64)

        self.model = PoseCNN(num_classes=len(self.label_names), grid_size=self.grid_size)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def rasterize(self, yz_points: np.ndarray) -> np.ndarray:
        hist, _, _ = np.histogram2d(
            yz_points[:, 0], yz_points[:, 1],
            bins=self.grid_size,
            range=[[self.lo[0], self.hi[0]], [self.lo[1], self.hi[1]]],
        )
        if hist.sum() > 0:
            hist = hist / hist.sum()
        return hist.astype(np.float32)

    def predict(self, yz_points: np.ndarray):
        grid = self.rasterize(yz_points)
        tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        with torch.no_grad():
            logits = self.model(tensor)
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
            # color=(1.0, 0.2, 0.2, 0.8),
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

        # Fonts for ticks and titles
        tick_font = QFont("Arial", 8)
        title_font = QFont("Arial", 11, QFont.Weight.Bold)
        tick_len = size * 0.03  # Length of the tick marks

        for val in range(-size, size + 1, spacing):
            if val == 0:
                continue
            # X-axis
            pos.extend([[val, -tick_len, 0], [val, tick_len, 0]])
            colors.extend([c_x, c_x])
            view.addItem(
                gl.GLTextItem(
                    pos=[val, -tick_len * 3, 0], text=str(val), font=tick_font
                )
            )

            # Y-axis
            pos.extend([[-tick_len, val, 0], [tick_len, val, 0]])
            colors.extend([c_y, c_y])
            view.addItem(
                gl.GLTextItem(
                    pos=[-tick_len * 3, val, 0], text=str(val), font=tick_font
                )
            )

            # Z-axis
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

        # 3. Add Axis Titles at the extremities
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

        pass

    def nextFrame(self):
        with self._file_path.open("rb") as f:
            dctx = ZstdDecompressor()

            SOF_DELIMITER = b"::"
            EOF_DELIMITER = b";;"
            HEADER_META_SIZE = len(SOF_DELIMITER) + struct.calcsize("<IBI")

            with dctx.stream_reader(f) as reader:
                while True:
                    header = reader.read(HEADER_META_SIZE)
                    # print(header)
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
                    # print(timestamp_us, message_type, payload_length)

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
        pass

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
    )

    parser.add_argument(
        "-m",
        "--model",
        metavar="<MODEL.pt>",
        type=Path,
        dest="model_path",
        default=None,
    )

    return parser.parse_args()

# Defines static boundaries around the movement area
def filter_data(data: NDArray):
    x_bound = (2, 4)
    y_bound = (-1, 2)
    z_bound = (-1.5, 1.5)

    mask = (
        (data[:, 0] >= x_bound[0]) & (data[:, 0] <= x_bound[1]) &
        (data[:, 1] >= y_bound[0]) & (data[:, 1] <= y_bound[1]) &
        (data[:, 2] >= z_bound[0]) & (data[:, 2] <= z_bound[1])
    )

    return data[mask]

def center_data(
        points: np.ndarray
) -> np.ndarray:
    radar_points = np.asarray(points, dtype=np.float64).copy()
    if radar_points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    total_point = [0,0,0]
    for point in points:
        total_point[0] += point[0]
        total_point[1] += point[1]
        total_point[2] += point[2]
    radar_points[:, 0] = radar_points[:, 0] - total_point[0]/len(points)
    radar_points[:, 1] = radar_points[:, 1] - total_point[1]/len(points)
    radar_points[:, 2] = radar_points[:, 2] - total_point[2]/len(points)
    return radar_points

def main() -> int:
    args = parse_args()
    path = args.path

    exit_event = threading.Event()
    app = QApplication(sys.argv)

    classifier = None
    if args.model_path is not None:
        classifier = PoseClassifier(args.model_path)
        print(f"Loaded pose classifier from {args.model_path} "
              f"(classes: {classifier.label_names})")

    dat_reader = DatReader(path)
    plotter = RadarPlotter()
    plotter.show()

    def accumulated_data_plot():
        nonlocal dat_reader
        nonlocal plotter
        current_tick_us = 0
        current_points = np.array([])
        # Set number of points to accumulate
        max_points = 100
        # Base image
        try:
            for d in dat_reader.nextFrame():
                msg_type = d["message_type"]
                msg = d["point_cloud"]
                new_tick_us = d["timestamp_us"]

                if msg_type == 2:
                    data = center_data(filter_data(msg))
                    saved_points = len(current_points) + len(data)
                    current_points = np.append(current_points, data).reshape(-1, 3)
                    if saved_points >= max_points:
                        points = current_points

                        # Run live pose classification on this flush, using
                        # the same Y-Z points (before X gets zeroed below)
                        # that training-time extraction sampled from.
                        if classifier is not None:
                            yz_points = points[:, 1:3]
                            label, confidence, _ = classifier.predict(yz_points)
                            plotter.update_prediction(label, confidence)

                        points[:, 0] = 0
                        plotter.update_data(data2=points)
                        current_points = current_points[max_points:]
                        QApplication.processEvents()               
                elif msg_type == 1:
                    plotter.update_data(data1=msg)


                diff_tick_us = new_tick_us - current_tick_us
                current_tick_us = new_tick_us
                time.sleep(diff_tick_us / 1e6)
            print("Finished")   
            exit_event.set()                 
        except KeyboardInterrupt:
            print("Interupted")
            app.quit()
            exit_event.set()
            return
        return

    plot_thread = threading.Thread(target=accumulated_data_plot, args=())
    plot_thread.start()
    try:
        print("Running")
        app.exec()
        exit_event.wait()
    except KeyboardInterrupt:
        print("Interupted")
        app.quit()
        pass
    plot_thread.join()
    return 0


if __name__ == "__main__":
    main()
