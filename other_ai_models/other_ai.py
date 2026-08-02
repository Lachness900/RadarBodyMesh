from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

from zstandard import ZstdDecompressor

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import struct

BASE_PATH = 'other_ai_models/data'


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
    def accumulateFrame(self):
        returned_points = np.array([])
        current_points = np.array([])
        max_points = 100
        frame_num = 0
        for d in self.nextFrame():
            msg_type = d["message_type"]
            msg = d["point_cloud"]

            if msg_type == 2:
                data = center_data(filter_data(msg))
                current_points = append_recent_points(current_points, data, limit=max_points)
                if len(current_points) == max_points:
                    returned_points = np.append(returned_points, current_points)
                    frame_num += 1
                    print("Saved frame", frame_num, current_points.shape)
                    yield current_points
            elif msg_type == 1:
                pass

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

def append_recent_points(
    current: np.ndarray,
    new_points: np.ndarray,
    limit: int,
) -> np.ndarray:
    points = np.asarray(new_points, dtype=np.float64)
    if points.size == 0:
        return current
    points = points[:, :3]
    if current.size == 0:
        return points[-limit:]
    return np.concatenate([current, points], axis=0)[-limit:]

def main() -> int:
    base_path = Path(BASE_PATH)
    poses = [d for d in base_path.iterdir()]
    samples = []
    labels = []
    for pose in poses:
        files = [file for file in pose.iterdir()]
        pose_name = pose.name
        print(f"Reading {pose_name} files")
        for file in files:
            print("Opening", file.name)
            dat_reader = DatReader(file)
            for frame in dat_reader.accumulateFrame():
                samples.append(frame.flatten())
                labels.append(pose_name)
    X = np.array(samples)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    models = [
        {
            "name": "K-Nearest Neighbour 5",
            "model": make_pipeline(
                        StandardScaler(),
                        KNeighborsClassifier(n_neighbors=5)
                    )
        },
        {
            "name": "K-Nearest Neighbour 10",
            "model": make_pipeline(
                        StandardScaler(),
                        KNeighborsClassifier(n_neighbors=10)
                    )
        },
        {
            "name": "K-Nearest Neighbour 15",
            "model": make_pipeline(
                        StandardScaler(),
                        KNeighborsClassifier(n_neighbors=15)
                    )
        },
        {
            "name": "Decision Tree",
            "model": make_pipeline(
                        StandardScaler(),
                        DecisionTreeClassifier()
                    )
        },
        {
            "name": "Random Forest",
            "model": make_pipeline(
                        StandardScaler(),
                        RandomForestClassifier(random_state=45)
                    )
        },
        {
            "name": "Support Vector Machine",
            "model": make_pipeline(
                        StandardScaler(),
                        SVC()
                    )
        },
        {
            "name": "Logistic Regression",
            "model": make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=1000)
                    )
        },
        {
            "name": "MLP Classifier",
            "model": make_pipeline(
                        StandardScaler(),
                        MLPClassifier(max_iter=500)
                    )
        },
        {
            "name": "Gaussian NB",
            "model": make_pipeline(
                        StandardScaler(),
                        GaussianNB()
                    )
        },
    ]
    for model in models:
        model_pipeline = model["model"]
        model_name = model["name"]
        print("Training", model_name)
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2%}")
        matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix: \n", matrix)
        print("  ")

if __name__ == "__main__":
    main()
