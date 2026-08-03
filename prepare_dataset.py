"""
Takes combined_dataset.npz (from batch_extract.py) and produces a
model-ready dataset:

  1. Prints class balance (samples per label, files per label).
  2. Rasterizes each variable-length (N, 3) X-Y-Z point sample into a
     fixed-size 2-CHANNEL 2D image:
       channel 0 = Y-Z density grid (lateral x vertical -- the original view)
       channel 1 = X-Z density grid (depth x vertical -- NEW)
     Bounds for each view are computed separately across the WHOLE dataset
     (not per-sample) so grid cells mean the same physical thing for every
     sample. X is included because some pose pairs (e.g. t_pose vs
     warrior_2_pose) are nearly indistinguishable in Y-Z alone -- they mainly
     differ in forward/backward (X) arm extension, which Y-Z can't see.
  3. Splits into train/val by FILE (not by sample) so samples from the same
     recording never appear in both sets, stratified per label where
     possible.

Output: prepared_dataset.npz with:
    X_train, y_train, X_val, y_val   - grids: (num_samples, 2, grid_size, grid_size)
    label_names                       - sorted list of label strings (class index order)
    grid_bounds_yz_lo/hi              - Y-Z view bounds used for rasterization
    grid_bounds_xz_lo/hi              - X-Z view bounds used for rasterization

Usage:
    python prepare_dataset.py --in combined_dataset.npz --out prepared_dataset.npz --grid_size 32 --val_frac 0.2
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def compute_global_bounds(samples, col_a, col_b, percentile=1.0):
    """
    Global bounds for a given pair of columns (e.g. Y,Z or X,Z) across the
    whole dataset, using a percentile clip to avoid a few outlier points
    blowing up the grid extent.
    """
    all_points = np.concatenate(samples, axis=0)  # (total_points, 3): X, Y, Z
    pair = all_points[:, [col_a, col_b]]
    lo = np.percentile(pair, percentile, axis=0)
    hi = np.percentile(pair, 100 - percentile, axis=0)
    return lo, hi


def rasterize(points_2d, grid_size, lo, hi):
    """
    Histogram2d a (N, 2) point set into a grid_size x grid_size density map
    using shared dataset-wide bounds. Normalized by point count so grid
    values are a density (0-1 range) rather than raw counts, keeping scale
    consistent regardless of how many points a given sample happened to have.
    """
    hist, _, _ = np.histogram2d(
        points_2d[:, 0], points_2d[:, 1],
        bins=grid_size,
        range=[[lo[0], hi[0]], [lo[1], hi[1]]],
    )
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist.astype(np.float32)


def rasterize_two_view(points_xyz, grid_size, yz_lo, yz_hi, xz_lo, xz_hi):
    """
    Builds a 2-channel image for one sample:
        channel 0: Y-Z view (columns 1, 2)
        channel 1: X-Z view (columns 0, 2)
    """
    yz = rasterize(points_xyz[:, [1, 2]], grid_size, yz_lo, yz_hi)
    xz = rasterize(points_xyz[:, [0, 2]], grid_size, xz_lo, xz_hi)
    return np.stack([yz, xz], axis=0)  # (2, grid_size, grid_size)


def subsample_per_file(samples, file_ids, timestamps, stride):
    """
    Keeps only every `stride`-th sample WITHIN each source file (ordered by
    timestamp), to cut down redundancy between nearby samples from the same
    file/recording. stride=1 keeps everything (no change in behavior).
    """
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
    """
    Groups sample indices by source file, then splits FILES (not samples)
    into train/val, stratified per label so each label's files are split
    proportionally.
    """
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
    parser = argparse.ArgumentParser(description="Prepare rasterized, file-split dataset for training")
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--subsample_stride", type=int, default=1,
                         help="Keep only every Nth sample per file (by timestamp order) to "
                              "reduce redundancy between nearby samples. 1 = keep all "
                              "(default, no change). E.g. 5 keeps ~20%% of samples per file.")
    args = parser.parse_args()

    d = np.load(args.in_path, allow_pickle=True)
    samples, labels, file_ids, timestamps = d["samples"], d["labels"], d["file_ids"], d["timestamps"]

    if samples[0].shape[1] < 3:
        raise SystemExit(
            "combined_dataset.npz samples only have "
            f"{samples[0].shape[1]} column(s) -- expected 3 (X, Y, Z). "
            "This file was extracted with an older batch_extract.py that drops X. "
            "Re-run batch_extract.py (updated to keep X) on the raw .dat files first."
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

    print(f"\n=== Rasterizing to 2 x {args.grid_size}x{args.grid_size} grids (Y-Z, X-Z) ===")
    yz_lo, yz_hi = compute_global_bounds(samples, col_a=1, col_b=2)
    xz_lo, xz_hi = compute_global_bounds(samples, col_a=0, col_b=2)
    print(f"Y-Z bounds (1st-99th pct): lo={yz_lo}, hi={yz_hi}")
    print(f"X-Z bounds (1st-99th pct): lo={xz_lo}, hi={xz_hi}")

    grids = np.stack([
        rasterize_two_view(s, args.grid_size, yz_lo, yz_hi, xz_lo, xz_hi)
        for s in samples
    ])  # (num_samples, 2, grid_size, grid_size)

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
        X_train=grids[train_idx], y_train=y[train_idx],
        X_val=grids[val_idx], y_val=y[val_idx],
        label_names=np.array(label_names, dtype=object),
        grid_bounds_yz_lo=yz_lo, grid_bounds_yz_hi=yz_hi,
        grid_bounds_xz_lo=xz_lo, grid_bounds_xz_hi=xz_hi,
    )
    print(f"\nSaved prepared dataset to {args.out}")


if __name__ == "__main__":
    main()