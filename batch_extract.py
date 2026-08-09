"""
Batch extracts mmWave radar training samples from a folder per label dataset
of .dat captures, using the logic from visualizer.py's accumulated_data_plot()

Expected input layout:
    dataset/
        t_pose/
            cam_radar_t_pose_1_1.dat
            cam_radar_t_pose_1_2.dat
            ...
        standing_pose/
            ...
        warrior_1_pose/
        warrior_2_pose/
        angle_pose/
        other/              <- misc / negative-class captures
            ...

Each subfolder name becomes the label for every .dat file inside it.

Output: a single .npz with:
    samples    - object array of (N, 3) float32 arrays (variable N, X-Y-Z points)
    labels     - object array of str, one per sample
    file_ids   - object array of str, source filename per sample (for traceability)
    timestamps - int64 array, flush timestamp (micro seconds) per sample

Usage:
    python batch_extract.py --dataset dataset/ --out combined_dataset.npz
"""

import argparse
import struct
from pathlib import Path

import numpy as np
from zstandard import ZstdDecompressor

SOF_DELIMITER = b"::"
EOF_DELIMITER = b";;"
HEADER_META_SIZE = len(SOF_DELIMITER) + struct.calcsize("<IBI")
MAX_POINTS = 100


def filter_data(data: np.ndarray) -> np.ndarray:
    x_bound = (-10, 10)
    y_bound = (-10, 10)
    z_bound = (-10, 10)
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
    pending = []
    pending_count = 0

    overflow = np.empty((0, 3), dtype=np.float64)
    last_ts = None

    for d in iter_frames(path):
        if d["message_type"] != 2:
            continue

        data = center_data(filter_data(d["point_cloud"]))
        last_ts = d["timestamp_us"]

        pending.append(data)
        pending_count += len(data)

        saved_points = len(overflow) + pending_count
        if saved_points >= MAX_POINTS:
            current_points = (
                np.concatenate([overflow] + pending, axis=0) if pending else overflow
            )
            xyz_points = current_points[:, :3].copy()
            yield last_ts, xyz_points

            overflow = current_points[MAX_POINTS:]
            pending = []
            pending_count = 0


def main():
    parser = argparse.ArgumentParser(description="Batch-extract radar training samples from a labeled-folder dataset")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="root folder containing one subfolder per label")
    parser.add_argument("--out", type=Path, required=True, help="output combined .npz path")
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Dataset dir not found: {args.dataset_dir}")

    label_dirs = sorted([p for p in args.dataset_dir.iterdir() if p.is_dir()])
    if not label_dirs:
        raise SystemExit(f"No label subfolders found under {args.dataset_dir}")

    all_samples, all_labels, all_file_ids, all_timestamps = [], [], [], []
    per_label_counts = {}
    per_label_files = {}

    for label_dir in label_dirs:
        label = label_dir.name
        dat_files = sorted(label_dir.glob("*.dat"))
        if not dat_files:
            print(f"WARNING: no .dat files found in {label_dir}")
            continue

        per_label_files[label] = len(dat_files)
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

        per_label_counts[label] = label_sample_count
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
