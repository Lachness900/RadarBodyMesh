"""
Takes combined_dataset.npz (from batch_extract.py) and produces a
model-ready dataset for the POINT-CLOUD-NATIVE model (no rasterization).

Unlike prepare_dataset.py, this does NOT rasterize samples into a fixed-size
grid -- point-cloud architectures (see train_pointcloud_classifier.py) take
the variable-length (N, 3) X-Y-Z points directly, so there's no grid_size or
grid_bounds to compute or store. This also sidesteps every grid-bound/
clipping issue this project ran into (test.dat's Z-offset clipping problem
would simply not occur here, since there's no fixed rasterization window for
points to be clipped by AFTER filter_data()).

Steps:
  1. Optional subsampling (same as prepare_dataset.py) to reduce redundancy
     between nearby samples from the same sliding-window buffer.
  2. Splits into train/val by FILE (not by sample), stratified per label.

NOTE: this does NOT apply scale normalization. Samples are saved at raw physical scale (already
zero-centered by center_data() at extraction time, but not scale-normalized).
Scale normalization now happens INSIDE PointCloudNet.forward() instead,
because doing it here would destroy each sample's raw scale before the
model ever saw it -- and raw scale turned out to be a real, validated
signal for distinguishing e.g. "squat" from "standing_pose" (a crouched
body has smaller extent from its own centroid than an upright one).
Normalizing inside the model lets it use raw scale as an explicit feature
AND still see scale-invariant shape for the point-cloud branch itself.

Output: prepared_pointcloud_dataset.npz with:
    samples_train, samples_val   - object arrays of (N, 3) float32 arrays (variable N)
    y_train, y_val               - int64 class indices
    label_names                  - sorted list of label strings (class index order)

Usage:
    python prepare_pointcloud_dataset.py --in combined_dataset.npz --out prepared_pointcloud_dataset.npz --val_frac 0.2
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def subsample_per_file(samples, file_ids, timestamps, stride):
    if stride <= 1:
        return np.arange(len(samples))

    file_to_indices = defaultdict(list)
    for idx, fid in enumerate(file_ids):
        file_to_indices[fid].append(idx)

    keep = []
    for fid, idxs in file_to_indices.items():
        idxs_sorted = sorted(idxs, key=lambda i: timestamps[i])
        keep.extend(idxs_sorted[::stride])

    return np.array(sorted(keep), dtype=np.int64)


def file_level_split(file_ids, labels, val_frac, seed=0):
    rng = np.random.default_rng(seed)

    file_to_label = {}
    file_to_indices = defaultdict(list)
    for idx, (fid, lbl) in enumerate(zip(file_ids, labels)):
        file_to_label[fid] = lbl
        file_to_indices[fid].append(idx)

    label_to_files = defaultdict(list)
    for fid, lbl in file_to_label.items():
        label_to_files[lbl].append(fid)

    train_idx, val_idx = [], []
    for lbl, files in label_to_files.items():
        files = sorted(files)
        rng.shuffle(files)
        n_val = max(1, round(len(files) * val_frac)) if len(files) > 1 else 0
        val_files = set(files[:n_val])
        for fid in files:
            target = val_idx if fid in val_files else train_idx
            target.extend(file_to_indices[fid])

    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Prepare a point-cloud-native (unrasterized) dataset for training")
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--subsample_stride", type=int, default=1,
                         help="Keep only every Nth sample per file (by timestamp order). "
                              "1 = keep all (default).")
    args = parser.parse_args()

    d = np.load(args.in_path, allow_pickle=True)
    samples, labels, file_ids, timestamps = d["samples"], d["labels"], d["file_ids"], d["timestamps"]

    if samples[0].shape[1] < 3:
        raise SystemExit(
            f"combined_dataset.npz samples only have {samples[0].shape[1]} column(s) -- "
            "expected 3 (X, Y, Z). Re-run batch_extract.py on the raw .dat files first."
        )

    if args.subsample_stride > 1:
        keep_idx = subsample_per_file(samples, file_ids, timestamps, args.subsample_stride)
        print(f"=== Subsampling: stride={args.subsample_stride} ===")
        print(f"  {len(samples)} samples -> {len(keep_idx)} samples "
              f"({len(keep_idx)/len(samples)*100:.1f}% kept)")
        samples, labels, file_ids = samples[keep_idx], labels[keep_idx], file_ids[keep_idx]

    print("\n=== Class balance ===")
    unique_labels = sorted(set(labels.tolist()))
    for lbl in unique_labels:
        mask = labels == lbl
        n_samples = mask.sum()
        n_files = len(set(file_ids[mask].tolist()))
        print(f"  {lbl:20s} {n_samples:5d} samples  from {n_files} file(s)")
    print(f"  {'TOTAL':20s} {len(labels):5d} samples  from {len(set(file_ids.tolist()))} file(s)")

    label_names = unique_labels
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    print(f"\n=== Splitting by file (val_frac={args.val_frac}) ===")
    train_idx, val_idx = file_level_split(file_ids, labels, args.val_frac)
    print(f"  train: {len(train_idx)} samples")
    print(f"  val:   {len(val_idx)} samples")

    train_files = set(file_ids[train_idx].tolist())
    val_files = set(file_ids[val_idx].tolist())
    overlap = train_files & val_files
    assert not overlap, f"File leakage between train/val: {overlap}"
    print("  no file overlap between train/val (verified)")

    np.savez(
        args.out,
        samples_train=np.array([samples[i].astype(np.float32) for i in train_idx], dtype=object),
        y_train=y[train_idx],
        samples_val=np.array([samples[i].astype(np.float32) for i in val_idx], dtype=object),
        y_val=y[val_idx],
        label_names=np.array(label_names, dtype=object),
    )
    print(f"\nSaved prepared point-cloud dataset to {args.out}")


if __name__ == "__main__":
    main()
