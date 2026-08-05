import argparse
import struct
from pathlib import Path

import numpy as np
from zstandard import ZstdDecompressor

SOF_DELIMITER = b"::"
EOF_DELIMITER = b";;"
HEADER_META_SIZE = len(SOF_DELIMITER) + struct.calcsize("<IBI")

# Must match visualizer.py's accumulated_data_plot() exactly
MAX_UPPER_POINTS = 60
MAX_LOWER_POINTS = 40
Y_SPLIT = -0.7  # threshold on the Z coordinate, despite the name (matches visualizer.py)


def filter_data(data: np.ndarray) -> np.ndarray:
    x_bound = (2, 4)
    y_bound = (-1, 2)
    z_bound = (-1.5, 1.5)
    mask = (
        (data[:, 0] >= x_bound[0]) & (data[:, 0] <= x_bound[1]) &
        (data[:, 1] >= y_bound[0]) & (data[:, 1] <= y_bound[1]) &
        (data[:, 2] >= z_bound[0]) & (data[:, 2] <= z_bound[1])
    )
    return data[mask]


def center_data(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return p - p.mean(axis=0)


def split_points(points: np.ndarray, y_split: float):
    """Same upper/lower split as visualizer.py (splits on Z, i.e. point[2])."""
    upper = [pt for pt in points if pt[2] >= y_split]
    lower = [pt for pt in points if pt[2] < y_split]
    return upper, lower


def append_recent_points(current: np.ndarray, new_points, limit: int) -> np.ndarray:
    """Same FIFO sliding-window append as visualizer.py."""
    points = np.asarray(new_points, dtype=np.float64)
    if points.size == 0:
        return current
    points = points[:, :3]
    if current.size == 0:
        return points[-limit:]
    return np.concatenate([current, points], axis=0)[-limit:]


def iter_frames(path: Path):
    with path.open("rb") as f:
        dctx = ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            while True:
                header = reader.read(HEADER_META_SIZE)
                if not header:
                    break
                if len(header) < HEADER_META_SIZE:
                    raise EOFError(f"Unexpected EOF while reading metadata in {path.name}.")
                _, timestamp_us, message_type, payload_length = struct.unpack("<2sIBI", header)

                raw_payload = reader.read(payload_length)
                point_cloud_np = np.frombuffer(raw_payload, dtype=np.int16) / 1000
                point_cloud_np = point_cloud_np.reshape((-1, 3))

                footer = reader.read(len(EOF_DELIMITER))
                if footer != EOF_DELIMITER:
                    raise ValueError(f"Stream corrupted in {path.name}: expected {EOF_DELIMITER}, got {footer}")

                yield {
                    "timestamp_us": timestamp_us,
                    "message_type": message_type,
                    "point_cloud": point_cloud_np,
                }


def extract_flush_samples(path: Path):
    """
    Replays the exact sliding-window buffer logic from the threaded
    visualizer.py's accumulated_data_plot() for radar (message_type == 2)
    frames only. Yields one sample per display-update trigger:
        (frame_timestamp_us, xyz_points)
    where xyz_points is (N, 3), N <= MAX_UPPER_POINTS + MAX_LOWER_POINTS.
    """
    upper_points = np.empty((0, 3), dtype=np.float64)
    lower_points = np.empty((0, 3), dtype=np.float64)

    for d in iter_frames(path):
        if d["message_type"] != 2:
            continue

        data = center_data(filter_data(d["point_cloud"]))
        ts = d["timestamp_us"]

        upper_data, lower_data = split_points(data, Y_SPLIT)
        lower_size = len(lower_points) + len(lower_data)
        upper_size = len(upper_points) + len(upper_data)
        upper_points = append_recent_points(upper_points, upper_data, MAX_UPPER_POINTS)
        lower_points = append_recent_points(lower_points, lower_data, MAX_LOWER_POINTS)

        if lower_size > MAX_LOWER_POINTS or upper_size > MAX_UPPER_POINTS:
            xyz_points = np.concatenate([upper_points, lower_points], axis=0)  # (N, 3): X, Y, Z
            yield ts, xyz_points


def main():
    parser = argparse.ArgumentParser(description="Batch-extract radar training samples (upper/lower buffer, 3D) from a labeled-folder dataset")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="root folder containing one subfolder per label")
    parser.add_argument("--out", type=Path, required=True, help="output combined .npz path")
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Dataset dir not found: {args.dataset_dir}")

    label_dirs = sorted([p for p in args.dataset_dir.iterdir() if p.is_dir()])
    if not label_dirs:
        raise SystemExit(f"No label subfolders found under {args.dataset_dir}")

    all_samples, all_labels, all_file_ids, all_timestamps = [], [], [], []

    for label_dir in label_dirs:
        label = label_dir.name
        dat_files = sorted(label_dir.glob("*.dat"))
        if not dat_files:
            print(f"WARNING: no .dat files found in {label_dir}")
            continue

        label_sample_count = 0
        for dat_path in dat_files:
            try:
                for ts, xyz in extract_flush_samples(dat_path):
                    all_samples.append(xyz.astype(np.float32))
                    all_labels.append(label)
                    all_file_ids.append(dat_path.name)
                    all_timestamps.append(ts)
                    label_sample_count += 1
            except (EOFError, ValueError) as e:
                print(f"ERROR parsing {dat_path}: {e} -- skipping this file")

        print(f"{label:20s}  {len(dat_files)} files  ->  {label_sample_count} samples")

    print(f"\nTotal: {len(all_samples)} samples across {len(label_dirs)} labels")

    np.savez(
        args.out,
        samples=np.array(all_samples, dtype=object),
        labels=np.array(all_labels, dtype=object),
        file_ids=np.array(all_file_ids, dtype=object),
        timestamps=np.array(all_timestamps, dtype=np.int64),
    )
    print(f"Saved combined dataset to {args.out}")


if __name__ == "__main__":
    main()
