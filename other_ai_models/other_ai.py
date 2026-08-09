import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from joblib import dump
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from zstandard import ZstdDecompressor

BASE_PATH = 'other_ai_models/run_2'
OUTPUT_PATH = 'other_ai_models/Simple_models'
TEST_PATH = 'other_ai_models/test_data'
MAX_POINTS = 100
MAX_UPPER_POINTS = 60
MAX_LOWER_POINTS = 40
Y_SPLIT = 0.7
HISTORY_FACTOR = 0.8
EXPONENTIAL_FACTOR = 0.2

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
    # Change size to (0, 4) if using exponential
    def accumulateFrame(self):
        current_points = np.empty((0, 3), dtype=np.float64)
        for d in self.nextFrame():
            msg_type = d["message_type"]
            msg = d["point_cloud"]
            if msg_type == 2:
                current_points = process_data(current_points, msg)
                if len(current_points) >= MAX_POINTS:
                    yield current_points[:MAX_POINTS]
                    current_points = current_points[MAX_POINTS:]
            elif msg_type == 1:
                pass

## Simple filtering
def process_data(current_data: NDArray, data: NDArray):
    return append_recent_points(current_data, center_data(filter_data(data)), limit=MAX_POINTS)

# Split filtering
# def process_data(current_data: NDArray, data: NDArray):
#     upper_points, lower_points = split_points(current_data)
#     upper_data, lower_data = split_points(center_data(filter_data(data)))
#     upper_points = append_recent_points(upper_points, upper_data, MAX_UPPER_POINTS)
#     lower_points = append_recent_points(lower_points, lower_data, MAX_LOWER_POINTS)
#     if upper_points.shape == (0,):
#         return lower_points
#     if lower_points.shape == (0,):
#         return upper_points
#     return np.concatenate((upper_points, lower_points), axis=0) 

## Exponentially weighted filtering
# def process_data(current_data: NDArray, data: NDArray):
#     data = center_data(filter_data(data))
#     data = add_strength_column(data)
#     data = update_strength_column(current_data.reshape(-1, 4), data)
#     return data

def add_strength_column(data, init_value: float = 1):
    if data.shape[1] == 4:
        return data
    return np.hstack((data, np.full(np.size(data, axis=0), init_value).reshape(-1, 1)))

def update_strength_column(current_data, data, factor = HISTORY_FACTOR):
    data = np.concatenate((current_data,data))
    data[:,3] *= factor * (1 - np.exp(- EXPONENTIAL_FACTOR**2 *(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)))
    if np.size(data) <= MAX_POINTS:
        return data
    indicies = np.argsort(data[:,3])[-MAX_POINTS:][::-1]
    return data[indicies]

def split_points(
        points: np.ndarray,
        y_split: float = Y_SPLIT
) -> tuple[NDArray, NDArray]:
    upper_points = np.array([point for point in points if point[2] > y_split])
    lower_points = np.array([point for point in points if point[2] < y_split])
    return upper_points, lower_points

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
    limit: int = MAX_POINTS,
) -> np.ndarray:
    points = np.asarray(new_points, dtype=np.float64)

    if points.size == 0:
        return current
    elif current.size == 0:
        return points[-limit:]
    
    return np.concatenate([current, points], axis=0)

def get_data(base_path):
    poses = [d for d in base_path.iterdir()]
    file_num = len([file for pose in poses for file in pose.iterdir()])
    file_count = 0
    samples = []
    labels = []
    for pose in poses:
        files = [file for file in pose.iterdir()]
        pose_name = pose.name
        print(f"Reading {pose_name} files")
        for file in files:
            print("Opening", file.name)
            dat_reader = DatReader(file)
            for frame in enumerate(dat_reader.accumulateFrame(), 0):
                samples.append(frame[1].flatten())
                labels.append(pose_name)
                print("Saved frame", frame[0], f"({file_count}/{file_num})")
            file_count += 1
    return np.array(samples), np.array(labels)

def train_model(model, X, y, X_test, y_test):
    model_pipeline = model["model"]
    model_name = model["name"]
    model_pipeline.fit(X, y)
    evaluate_model(model_pipeline, model_name, X_test, y_test)
    dump(model_pipeline, f"{OUTPUT_PATH}\\{model_name}.joblib")

def evaluate_model(model, model_name, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("Training", model_name)
    print(f"Accuracy: {accuracy:.2%}")
    print("Confusion Matrix: \n", matrix)
    print("  ")

def main() -> int:
    base_path = Path(BASE_PATH)
    test_path = Path(TEST_PATH)
    X, y = get_data(base_path)
    X_test, y_test = get_data(test_path)


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
            "name": "Linear Support Vector Machine",
            "model": make_pipeline(
                        StandardScaler(),
                        LinearSVC()
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
    print("Training")
    model_num = len(models)
    with ThreadPoolExecutor(max_workers=model_num) as executor:
        for i in range(model_num):
            executor.submit(train_model, model=models[i], X=X, y=y, X_test=X_test, y_test=y_test)

if __name__ == "__main__":
    main()
