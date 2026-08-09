"""
Trains a point-cloud-native CNN pose classifier directly on the variable-
length (N, 3) X-Y-Z point sets produced by prepare_pointcloud_dataset.py --
no rasterization into a fixed grid.

Input: prepared_pointcloud_dataset.npz with samples_train, y_train,
       samples_val, y_val, label_names (samples are raw, centered-but-not-
       scale-normalized points -- normalization now happens inside the
       model, see normalize_and_features())
Output: trained model weights (.pt) + printed classification report / confusion matrix

Usage:
    python train_pointcloud_classifier.py --in prepared_pointcloud_dataset.npz --out pointcloud_classifier.pt --epochs 240
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PointCloudDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        points = torch.from_numpy(np.asarray(self.samples[idx], dtype=np.float32))
        label = int(self.labels[idx])
        return points, label


def collate_variable_length(batch):
    points_list, labels = zip(*batch)
    batch_size = len(points_list)
    max_n = max(p.shape[0] for p in points_list)
    max_n = max(max_n, 1)

    padded = torch.zeros((batch_size, max_n, 3), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_n), dtype=torch.bool)
    for i, p in enumerate(points_list):
        n = p.shape[0]
        if n > 0:
            padded[i, :n] = p
            mask[i, :n] = True

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, mask, labels


def normalize_and_features(points: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    mask_f = mask.float()
    n_real = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)

    point_norms = points.norm(dim=2)
    point_norms = point_norms.masked_fill(~mask, 0.0)
    raw_scale = point_norms.max(dim=1).values.clamp(min=eps)

    normalized_points = points / raw_scale.view(-1, 1, 1)
    normalized_points = normalized_points * mask_f.unsqueeze(-1)

    mean = (normalized_points * mask_f.unsqueeze(-1)).sum(dim=1) / n_real
    centered = (normalized_points - mean.unsqueeze(1)) * mask_f.unsqueeze(-1)

    var = (centered ** 2).sum(dim=1) / n_real
    std = torch.sqrt(var + eps)

    third_moment = (centered ** 3).sum(dim=1) / n_real
    skew = third_moment / (std ** 3 + eps)

    aux = torch.stack([raw_scale, std[:, 0], std[:, 1], skew[:, 2]], dim=1)
    return normalized_points, aux


class PointCloudNet(nn.Module):
    AUX_FEATURES = 4

    def __init__(self, num_classes: int):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.aux_norm = nn.BatchNorm1d(self.AUX_FEATURES)
        self.classifier = nn.Sequential(
            nn.Linear(256 + self.AUX_FEATURES, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized_points, aux = normalize_and_features(points, mask)

        x = normalized_points.transpose(1, 2)
        x = self.shared_mlp(x)

        mask_expanded = mask.unsqueeze(1).expand(-1, x.size(1), -1)
        x = x.masked_fill(~mask_expanded, float("-inf"))

        pooled, _ = x.max(dim=2)
        pooled = torch.nan_to_num(pooled, neginf=0.0)

        aux = self.aux_norm(aux)

        fused = torch.cat([pooled, aux], dim=1)
        return self.classifier(fused)


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def augment_points(points: torch.Tensor, mask: torch.Tensor,
                    jitter_std: float = 0.02, drop_prob: float = 0.1) -> tuple:
    jitter = torch.randn_like(points) * jitter_std
    points = points + jitter * mask.unsqueeze(-1)

    drop = torch.rand(mask.shape, device=mask.device) < drop_prob
    new_mask = mask & ~drop
    all_dropped = new_mask.sum(dim=1) == 0
    if all_dropped.any():
        new_mask[all_dropped] = mask[all_dropped]

    return points, new_mask


def evaluate(model, loader, device, num_classes, label_names):
    model.eval()
    correct, total = 0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for points, mask, yb in loader:
            points, mask, yb = points.to(device), mask.to(device), yb.to(device)
            logits = model(points, mask)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            for t, p in zip(yb.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1
    acc = correct / total if total else 0.0

    print(f"\nValidation accuracy: {acc:.4f} ({correct}/{total})")
    print("\nPer-class precision/recall:")
    for i, name in enumerate(label_names):
        tp = confusion[i, i]
        support = confusion[i, :].sum()
        pred_total = confusion[:, i].sum()
        recall = tp / support if support else 0.0
        precision = tp / pred_total if pred_total else 0.0
        print(f"  {name:20s} precision={precision:.3f}  recall={recall:.3f}  support={support}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "               " + "".join(f"{n[:10]:>12s}" for n in label_names)
    print(header)
    for i, name in enumerate(label_names):
        row = "".join(f"{confusion[i, j]:>12d}" for j in range(num_classes))
        print(f"  {name[:13]:13s}{row}")

    return acc


def main():
    parser = argparse.ArgumentParser(description="Train a point-cloud-native pose classifier")
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_augment", action="store_true",
                         help="Disable the default jitter + point-dropout training augmentation.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.in_path, allow_pickle=True)
    samples_train, y_train = d["samples_train"], d["y_train"]
    samples_val, y_val = d["samples_val"], d["y_val"]
    label_names = [str(x) for x in d["label_names"]]
    num_classes = len(label_names)

    sizes = [len(s) for s in samples_train]
    print(f"Train: {len(samples_train)} samples (points/sample: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.1f}), Val: {len(samples_val)} samples")
    print(f"Classes: {label_names} (scale normalization + aux features computed inside the model)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = PointCloudDataset(samples_train, y_train)
    val_ds = PointCloudDataset(samples_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_variable_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_variable_length)

    model = PointCloudNet(num_classes=num_classes).to(device)

    class_weights = compute_class_weights(y_train, num_classes).to(device)
    print(f"Class weights (inverse-frequency, for imbalance): "
          f"{dict(zip(label_names, class_weights.cpu().numpy().round(2)))}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for points, mask, yb in train_loader:
            points, mask, yb = points.to(device), mask.to(device), yb.to(device)
            if not args.no_augment:
                points, mask = augment_points(points, mask)
            optimizer.zero_grad()
            logits = model(points, mask)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            running_correct += (logits.argmax(dim=1) == yb).sum().item()
            running_total += yb.size(0)
        scheduler.step()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for points, mask, yb in val_loader:
                points, mask, yb = points.to(device), mask.to(device), yb.to(device)
                preds = model(points, mask).argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)
        val_acc = val_correct / val_total if val_total else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    print(f"\nBest validation accuracy during training: {best_val_acc:.4f}")
    model.load_state_dict(best_state)

    evaluate(model, val_loader, device, num_classes, label_names)

    torch.save({
        "model_state_dict": best_state,
        "label_names": label_names,
    }, args.out)
    print(f"\nSaved best model to {args.out}")


if __name__ == "__main__":
    main()
