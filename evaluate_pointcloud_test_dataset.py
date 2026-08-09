"""
Evaluates a trained pointcloud_classifier.pt checkpoint against a held-out
test dataset. This is the point-cloud-model counterpart to
evaluate_test_dataset.py (which only works with the rasterized CNN) --
they are NOT interchangeable, since the two models take different inputs
and store different things in their checkpoints.

Workflow:
    1. Run batch_extract.py on your test_dataset folder (same folder-per-label
       layout as training) to get a combined_dataset.npz:
           python batch_extract.py --dataset_dir test_dataset/ --out combined_test.npz
    2. Run this script against that file and your trained checkpoint:
           python evaluate_pointcloud_test_dataset.py --data combined_test.npz --model pointcloud_classifier.pt

No rasterization happens here (there's no grid to rasterize into), and no
scale normalization happens here either -- PointCloudNet normalizes
internally (see train_pointcloud_classifier.py's normalize_and_features()),
so this script passes raw (centered, extraction-time-only) points straight
to the model, same as training does. This is IMPORTANT: earlier versions of
this pipeline normalized scale at data-prep time, which destroyed raw_scale
before the model ever saw it. Do not add normalization here -- it would
silently double-normalize (or mismatch) relative to what the model expects.

Output:
    - Printed accuracy, per-class precision/recall, and confusion matrix
    - confusion_matrix.png: a heatmap plot
    - metrics.txt: the same printed report, saved to disk

Usage:
    python evaluate_pointcloud_test_dataset.py --data combined_test.npz --model pointcloud_classifier.pt --out_dir eval_results
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_pointcloud_classifier import (
    PointCloudNet,
    PointCloudDataset,
    collate_variable_length,
)


def plot_confusion_matrix(confusion: np.ndarray, label_names: list[str], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = len(label_names)
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized = confusion / row_sums

    fig, ax = plt.subplots(figsize=(1.4 * num_classes + 2, 1.4 * num_classes + 2))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized) -- Point-Cloud Model, Test Dataset")

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
    parser = argparse.ArgumentParser(description="Evaluate a trained point-cloud pose classifier against a held-out test dataset")
    parser.add_argument("--data", type=Path, required=True,
                         help="combined_dataset.npz produced by running batch_extract.py on your test_dataset folder")
    parser.add_argument("--model", type=Path, required=True, help="trained pointcloud_classifier.pt checkpoint")
    parser.add_argument("--out_dir", type=Path, default=Path("eval_results_pointcloud"),
                         help="directory to save confusion_matrix.png and metrics.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    label_names = list(ckpt["label_names"])
    num_classes = len(label_names)

    model = PointCloudNet(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded model: classes={label_names}")

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

    samples = np.array([s.astype(np.float32) for s in samples], dtype=object)

    label_to_idx = {l: i for i, l in enumerate(label_names)}
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    ds = PointCloudDataset(samples, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_variable_length)

    correct, total = 0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for points, mask, yb in loader:
            logits = model(points, mask)
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
