"""
Evaluates a trained pose_classifier.pt checkpoint against a held-out test
dataset.

Run this script against that file and your trained checkpoint:
           python evaluate_test_dataset.py --data combined_test.npz --model pose_classifier.pt

Output:
    - Printed accuracy, per-class precision/recall, and confusion matrix
      (same format as train_pose_classifier.py's evaluate())
    - confusion_matrix.png: a heatmap plot
    - metrics.txt: the same printed report, saved to disk

Usage:
    python evaluate_test_dataset.py --data combined_test.npz --model pose_classifier.pt --out_dir eval_results
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class PoseCNN(nn.Module):
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


def rasterize(points_2d: np.ndarray, grid_size: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        points_2d[:, 0], points_2d[:, 1],
        bins=grid_size,
        range=[[lo[0], hi[0]], [lo[1], hi[1]]],
    )
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist.astype(np.float32)


def rasterize_two_view(points_xyz, grid_size, yz_lo, yz_hi, xz_lo, xz_hi):
    yz = rasterize(points_xyz[:, [1, 2]], grid_size, yz_lo, yz_hi)
    xz = rasterize(points_xyz[:, [0, 2]], grid_size, xz_lo, xz_hi)
    return np.stack([yz, xz], axis=0)  # (2, grid_size, grid_size)


def plot_confusion_matrix(confusion: np.ndarray, label_names: list[str], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = len(label_names)
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized = confusion / row_sums  # row-normalized (recall-style) for coloring

    fig, ax = plt.subplots(figsize=(1.4 * num_classes + 2, 1.4 * num_classes + 2))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized) -- Test Dataset")

    for i in range(num_classes):
        for j in range(num_classes):
            count = confusion[i, j]
            frac = normalized[i, j]
            text_color = "white" if frac > 0.6 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Fraction of true class (recall)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained pose classifier against a held-out test dataset")
    parser.add_argument("--data", type=Path, required=True,
                         help="combined_dataset.npz produced by running batch_extract.py on your test_dataset folder")
    parser.add_argument("--model", type=Path, required=True, help="trained pose_classifier.pt checkpoint")
    parser.add_argument("--out_dir", type=Path, default=Path("eval_results"),
                         help="directory to save confusion_matrix.png and metrics.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint ---
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    label_names = list(ckpt["label_names"])
    grid_size = int(ckpt["grid_size"])
    in_channels = int(ckpt.get("in_channels", 2))
    yz_lo = np.asarray(ckpt["grid_bounds_yz_lo"], dtype=np.float64)
    yz_hi = np.asarray(ckpt["grid_bounds_yz_hi"], dtype=np.float64)
    xz_lo = np.asarray(ckpt["grid_bounds_xz_lo"], dtype=np.float64)
    xz_hi = np.asarray(ckpt["grid_bounds_xz_hi"], dtype=np.float64)
    num_classes = len(label_names)

    model = PoseCNN(num_classes=num_classes, grid_size=grid_size, in_channels=in_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded model: classes={label_names}, grid_size={grid_size}, in_channels={in_channels}")

    d = np.load(args.data, allow_pickle=True)
    samples, labels, file_ids = d["samples"], d["labels"], d["file_ids"]
    print(f"Loaded test data: {len(samples)} samples from {len(set(file_ids.tolist()))} file(s)")

    test_label_set = set(labels.tolist())
    unknown_labels = test_label_set - set(label_names)
    if unknown_labels:
        print(f"WARNING: test data contains label(s) not in the trained model: {unknown_labels} "
              f"-- these samples will be EXCLUDED from evaluation.")
        keep_mask = np.array([lbl in label_names for lbl in labels])
        samples, labels, file_ids = samples[keep_mask], labels[keep_mask], file_ids[keep_mask]
        print(f"  {len(samples)} samples remain after exclusion.")

    print("\n=== Test set class balance ===")
    for lbl in label_names:
        mask = labels == lbl
        n_files = len(set(file_ids[mask].tolist())) if mask.sum() else 0
        print(f"  {lbl:20s} {mask.sum():5d} samples  from {n_files} file(s)")

    print(f"\nRasterizing {len(samples)} samples using the model's saved bounds...")
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    grids = np.stack([
        rasterize_two_view(s, grid_size, yz_lo, yz_hi, xz_lo, xz_hi)
        for s in samples
    ])
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    ds = TensorDataset(torch.from_numpy(grids).float(), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    correct, total = 0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            for t, p in zip(yb.numpy(), preds.numpy()):
                confusion[t, p] += 1
    acc = correct / total if total else 0.0

    report_lines = []
    report_lines.append(f"Test accuracy: {acc:.4f} ({correct}/{total})")
    report_lines.append("")
    report_lines.append("Per-class precision/recall:")
    for i, name in enumerate(label_names):
        tp = confusion[i, i]
        support = confusion[i, :].sum()
        pred_total = confusion[:, i].sum()
        recall = tp / support if support else 0.0
        precision = tp / pred_total if pred_total else 0.0
        report_lines.append(f"  {name:20s} precision={precision:.3f}  recall={recall:.3f}  support={support}")

    report_lines.append("")
    report_lines.append("Confusion matrix (rows=true, cols=predicted):")
    header = "               " + "".join(f"{n[:10]:>12s}" for n in label_names)
    report_lines.append(header)
    for i, name in enumerate(label_names):
        row = "".join(f"{confusion[i, j]:>12d}" for j in range(num_classes))
        report_lines.append(f"  {name[:13]:13s}{row}")

    report = "\n".join(report_lines)
    print("\n" + report)

    metrics_path = args.out_dir / "metrics.txt"
    metrics_path.write_text(report + "\n")
    print(f"\nSaved metrics to {metrics_path}")

    cm_path = args.out_dir / "confusion_matrix.png"
    plot_confusion_matrix(confusion, label_names, cm_path)
    print(f"Saved confusion matrix plot to {cm_path}")


if __name__ == "__main__":
    main()
