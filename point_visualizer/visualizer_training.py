#!/usr/bin/env python

import sys
import time
from pathlib import Path
import argparse
from typing import Optional
import struct

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from PyQt6.QtGui import QFont, QImage
import pyqtgraph.opengl as gl
from zstandard import ZstdDecompressor
import pyvirtualcam

import numpy as np
from numpy.typing import NDArray


class RadarPlotter(QMainWindow):
    """
    This class plots the camera data along with the radar scan
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Plotter")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        display_layout = QHBoxLayout(central_widget)

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
        """
        Updates both plots with new data.
        """
        if data1 is not None:
            self.scatter1.setData(pos=data1[:, :3], size=3)

        if data2 is not None:
            self.scatter2.setData(pos=data2[:, :3], size=3)

            
        


class ReaderParserError(Exception):
    def __init__(self, reason):
        super().__init__(reason)


class DatReader:
    def __init__(self, path: Path):
        self._file_path = path

        pass

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
        help="path to file that ends with .bag prefix",
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

def accumulated_data_plot(dat_reader, plotter, size):
    frames = []
    current_points = np.array([])
    imgHeight, imgWidth = size
    # Set number of points to accumulate
    max_points = 100
    # Base image
    for d in dat_reader.nextFrame():
        msg_type = d["message_type"]
        msg = d["point_cloud"]

        if msg_type == 2:
            data = filter_data(msg)
            saved_points = len(current_points) + len(data)
            current_points = np.append(current_points, data).reshape(-1, 3)
            if saved_points >= max_points:
                points = current_points
                points[:, 0] = 0
                plotter.update_data(data2=points)
                current_points = current_points[:saved_points - max_points]

                QApplication.processEvents()
                pixels = plotter.view2.grab()
                if pixels.isNull():
                    return
                img = pixels.toImage().convertToFormat(QImage.Format.Format_RGB888)
                ptr = img.bits()
                ptr.setsize(img.sizeInBytes())
                arr = np.frombuffer(ptr, np.uint8).reshape((imgHeight, img.bytesPerLine()))
                arr = arr[:, :imgWidth * 3]
                frames.append(arr.reshape((imgHeight, imgWidth, 3)))  
    return frames        

def stream_data(frames, size):
    with pyvirtualcam.Camera(width=size[1], height=size[0], fps=1) as cam:
        print(f"Using: {cam.device} on {cam.backend}")
        for frame in frames:
            cam.send(frame)
            cam.sleep_until_next_frame()

def main() -> int:
    args = parse_args()
    path = args.path

    app = QApplication(sys.argv)

    plotter = RadarPlotter()
    plotter.resize(1600, 800)
    plotter.show()
    pixels = plotter.view2.grab()         
    img = pixels.toImage().convertToFormat(QImage.Format.Format_RGB888)
    size = (img.height(), img.width())
    dat_reader = DatReader(path)
    frames = accumulated_data_plot(dat_reader, plotter, size)
    print("accumulated frames,", len(frames))
    stream_data(frames, size)
    print("streamed")
    return 0


if __name__ == "__main__":
    main()
