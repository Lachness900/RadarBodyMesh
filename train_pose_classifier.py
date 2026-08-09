"""
Trains a small CNN pose classifier on the rasterized radar grids produced by
prepare_dataset.py.

Input: prepared_dataset.npz with X_train, y_train, X_val, y_val, label_names
Output: trained model weights (.pt) + printed classification report / confusion matrix

Usage:
    python train_pose_classifier.py --in prepared_dataset.npz --out pose_classifier.pt --epochs 40
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path

import numpy as np
from joblib import dump
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

OUTPUT_PATH = 'other_ai_models/Simple_models'

class PoseCNN(nn.Module):
    """
    CNN for 2-channel fixed-size radar density grids -> pose class logits.
    Channel 0 = Y-Z view, channel 1 = X-Z view (see prepare_dataset.py).
    Using two views instead of one reintroduces the depth (X) axis, which
    Y-Z alone discards -- some pose pairs (e.g. t_pose vs warrior_2_pose)
    are nearly indistinguishable in Y-Z since they mainly differ in
    forward/backward arm extension, an X-axis difference.

    This does NOT end in global average pooling (nn.AdaptiveAvgPool2d(1)).
    Global average pooling collapses the whole spatial map down to one value
    per channel, discarding WHERE density is concentrated and keeping only
    HOW MUCH total density existed. Instead, this pools down to a small
    spatial map (grid_size/8 per side, e.g. 4x4 for a 32x32 input) and
    flattens THAT into the classifier head, preserving coarse spatial layout.
    """

    def __init__(self, num_classes: int, grid_size: int, in_channels: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # grid_size -> grid_size/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> grid_size/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> grid_size/8
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


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights so the underrepresented 'other' class
    doesn't get ignored by the loss."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1  # avoid div-by-zero for any empty class
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def augment_batch(xb: torch.Tensor, max_shift: int = 2, noise_std: float = 0.15) -> torch.Tensor:
    """
    Lightweight augmentation applied to TRAINING batches only:
      - random translation of up to +/- max_shift pixels per sample (per axis),
        using zero-padding so nothing wraps around -- this discourages the
        model from memorizing the EXACT pixel position of a session's point
        cloud (e.g. a slightly different centering/registration offset),
        which is exactly the kind of session-specific artifact it can
        otherwise latch onto given how repetitive sliding-window samples are.
      - small MULTIPLICATIVE jitter applied only to already-occupied cells
        (not additive noise across the whole grid). These grids are sparse
        density maps (typically <10% of cells nonzero, each cell holding a
        small fraction like 0.01-0.04); additive Gaussian noise across ALL
        cells -- including the ~90%+ that are legitimately empty -- injects
        noise mass that can far exceed the real signal mass (measured at
        ~88% of total post-noise mass in this dataset), since a radar frame
        with no detection somewhere is real information, not missing data.
        Multiplicative jitter on occupied cells only avoids inventing density
        where there was none.
    Renormalizes each channel independently after shifting so it still sums
    to ~1.0 (a density map), matching the raw (un-augmented) data's scale.
    """
    b, c, h, w = xb.shape
    shifts_y = torch.randint(-max_shift, max_shift + 1, (b,))
    shifts_x = torch.randint(-max_shift, max_shift + 1, (b,))
    out = torch.zeros_like(xb)
    for i in range(b):
        sy, sx = int(shifts_y[i]), int(shifts_x[i])
        src_y0, src_y1 = max(0, -sy), h - max(0, sy)
        src_x0, src_x1 = max(0, -sx), w - max(0, sx)
        dst_y0, dst_y1 = max(0, sy), h - max(0, -sy)
        dst_x0, dst_x1 = max(0, sx), w - max(0, -sx)
        out[i, :, dst_y0:dst_y1, dst_x0:dst_x1] = xb[i, :, src_y0:src_y1, src_x0:src_x1]

    occupied = out > 0
    jitter = 1.0 + torch.randn_like(out) * noise_std
    out = torch.where(occupied, out * jitter, out).clamp(min=0)

    sums = out.sum(dim=(2, 3), keepdim=True)
    sums[sums == 0] = 1
    out = out / sums
    return out


def evaluate(model, loader, device, num_classes, label_names):
    model.eval()
    correct, total = 0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
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

def evaluate_other_models(X, y, X_test, y_test):
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
    for model in models:
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

def main():
    parser = argparse.ArgumentParser(description="Train a CNN pose classifier on rasterized radar grids")
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                         help="L2 regularization strength, to fight the overfitting caused by "
                              "highly-correlated sliding-window samples within each file.")
    parser.add_argument("--no_augment", action="store_true",
                         help="Disable the default random-shift + noise training augmentation.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.in_path, allow_pickle=True)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    label_names = [str(x) for x in d["label_names"]]
    num_classes = len(label_names)
    in_channels = X_train.shape[1]
    grid_size = X_train.shape[2]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, classes: {label_names}, "
          f"in_channels: {in_channels}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PoseCNN(num_classes=num_classes, grid_size=grid_size, in_channels=in_channels).to(device)

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
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if not args.no_augment:
                xb = augment_batch(xb)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            running_correct += (logits.argmax(dim=1) == yb).sum().item()
            running_total += xb.size(0)
        scheduler.step()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)
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
        "grid_size": grid_size,
        "in_channels": in_channels,
        "grid_bounds_yz_lo": d["grid_bounds_yz_lo"],
        "grid_bounds_yz_hi": d["grid_bounds_yz_hi"],
        "grid_bounds_xz_lo": d["grid_bounds_xz_lo"],
        "grid_bounds_xz_hi": d["grid_bounds_xz_hi"],
    }, args.out)
    print(f"\nSaved best model to {args.out}")

    evaluate_other_models(X=X_train, y=y_train, X_test=X_val, y_test=y_val)


if __name__ == "__main__":
    main()