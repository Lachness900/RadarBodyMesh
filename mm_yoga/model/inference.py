"""Pose-classifier and mock prediction helpers used by the replay backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
import pickle
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

import joblib

DEFAULT_POSE_LABELS = [
    "standing_pose",
    "t_pose",
    "squat",
    "angle_pose",
]


@dataclass
class PredictionResult:
    """Pose prediction result returned by any backend predictor."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class PoseCNN(nn.Module):
    """Current checkpoint architecture: two input planes and 128-dim embedding."""

    def __init__(
        self,
        num_classes: int,
        grid_size: int = 32,
        in_channels: int = 2,
    ):
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
        final_spatial = grid_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * final_spatial * final_spatial, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class LegacyPoseCNN(nn.Module):
    """Older single-plane checkpoint architecture kept for compatibility."""

    def __init__(self, num_classes: int, grid_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def normalize_and_features(
    points: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the normalization and auxiliary features used during training."""

    mask_f = mask.float()
    n_real = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)

    point_norms = points.norm(dim=2).masked_fill(~mask, 0.0)
    raw_scale = point_norms.max(dim=1).values.clamp(min=eps)

    normalized_points = points / raw_scale.view(-1, 1, 1)
    normalized_points = normalized_points * mask_f.unsqueeze(-1)

    mean = (normalized_points * mask_f.unsqueeze(-1)).sum(dim=1) / n_real
    centered = (normalized_points - mean.unsqueeze(1)) * mask_f.unsqueeze(-1)
    variance = (centered**2).sum(dim=1) / n_real
    std = torch.sqrt(variance + eps)
    third_moment = (centered**3).sum(dim=1) / n_real
    skew = third_moment / (std**3 + eps)

    aux = torch.stack([raw_scale, std[:, 0], std[:, 1], skew[:, 2]], dim=1)
    return normalized_points, aux


class PointCloudNet(nn.Module):
    """Simplified PointNet-style classifier used by the final checkpoints."""

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
        features = self.shared_mlp(normalized_points.transpose(1, 2))

        mask_expanded = mask.unsqueeze(1).expand(-1, features.size(1), -1)
        features = features.masked_fill(~mask_expanded, float("-inf"))
        pooled = torch.nan_to_num(features.max(dim=2).values, neginf=0.0)

        fused = torch.cat([pooled, self.aux_norm(aux)], dim=1)
        return self.classifier(fused)


class SklearnPoseClassifier:
    """Load a legacy scikit-learn point classifier."""

    def __init__(self, path: Path):
        self.model = joblib.load(path)
        self.labels = list(self.model.classes_)

    def predict(self, points: np.ndarray[np.floating]) -> PredictionResult:
        points = points[:, 0:3].flatten()[:300].reshape(1, -1)
        probabilities = self.model.predict_proba(points)[0]
        return _result_from_probabilities(self.labels, probabilities)


class CNNPoseClassifier:
    """Load and run the CNN checkpoint trained from Y-Z radar histograms."""

    def __init__(self, path: Path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.labels = list(ckpt["label_names"])
        self.grid_size = int(ckpt["grid_size"])
        self.in_channels = int(ckpt.get("in_channels", 1))
        # Newer checkpoints store separate XZ/YZ bounds and two input planes;
        # older ones only stored the YZ pair used by the single-plane model.
        self.yz_lo = np.asarray(
            ckpt.get("grid_bounds_yz_lo", ckpt.get("grid_bounds_lo")),
            dtype=np.float64,
        )
        self.yz_hi = np.asarray(
            ckpt.get("grid_bounds_yz_hi", ckpt.get("grid_bounds_hi")),
            dtype=np.float64,
        )
        self.xz_lo = np.asarray(
            ckpt.get("grid_bounds_xz_lo", self.yz_lo),
            dtype=np.float64,
        )
        self.xz_hi = np.asarray(
            ckpt.get("grid_bounds_xz_hi", self.yz_hi),
            dtype=np.float64,
        )

        state = ckpt["model_state_dict"]
        if "features.14.weight" in state:
            self.model = PoseCNN(
                num_classes=len(self.labels),
                grid_size=self.grid_size,
                in_channels=self.in_channels,
            )
        else:
            self.model = LegacyPoseCNN(
                num_classes=len(self.labels),
                grid_size=self.grid_size,
            )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _rasterize(
        self,
        plane_points: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        """Normalize one 2D plane into a density grid for the CNN."""

        hist, _, _ = np.histogram2d(
            plane_points[:, 0],
            plane_points[:, 1],
            bins=self.grid_size,
            range=[[lo[0], hi[0]], [lo[1], hi[1]]],
        )
        if hist.sum() > 0:
            hist = hist / hist.sum()
        return hist.astype(np.float32)

    def rasterize(self, points: NDArray[np.floating]) -> np.ndarray:
        """Rasterize xyz points into the checkpoint's input planes."""

        if self.in_channels == 2:
            xz = self._rasterize(points[:, [0, 2]], self.xz_lo, self.xz_hi)
            yz = self._rasterize(points[:, [1, 2]], self.yz_lo, self.yz_hi)
            return np.stack([xz, yz], axis=0)
        yz = self._rasterize(points[:, [1, 2]], self.yz_lo, self.yz_hi)
        return yz

    def predict(self, points: NDArray[np.floating]) -> PredictionResult:
        """Predict from variable-length xyz points using the checkpoint planes."""

        grid = self.rasterize(points)
        tensor = torch.from_numpy(grid).unsqueeze(0).float()  # (1, C, H, W)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        return _result_from_probabilities(self.labels, probs)


class PointCloudPoseClassifier:
    """Load a PointNet-style checkpoint and predict from raw centred XYZ points."""

    def __init__(self, path: Path, *, checkpoint: Optional[dict] = None):
        ckpt = checkpoint or torch.load(path, map_location="cpu", weights_only=True)
        self.labels = list(ckpt["label_names"])
        self.model = PointCloudNet(num_classes=len(self.labels))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(self, points: NDArray[np.floating]) -> PredictionResult:
        """Run one variable-length sample without rasterizing or zero-padding it."""

        xyz = np.asarray(points, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] < 3 or len(xyz) == 0:
            raise ValueError("point-cloud inference requires a non-empty N x 3 array")

        points_t = torch.from_numpy(np.ascontiguousarray(xyz[:, :3])).unsqueeze(0)
        mask_t = torch.ones((1, len(xyz)), dtype=torch.bool)
        with torch.inference_mode():
            logits = self.model(points_t, mask_t)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return _result_from_probabilities(self.labels, probabilities)


class MockPosePredictor:
    """Deterministic fallback predictor for UI/backend smoke tests.

    This is not a trained yoga model. It only produces stable-looking
    probabilities so the replay backend and frontend contract can be developed
    before the team connects the final model.
    """

    def __init__(self, labels: Sequence[str] = DEFAULT_POSE_LABELS) -> None:
        """Store the pose label order used by generated probabilities."""

        self.labels = list(labels)

    def predict(self, features: NDArray[np.floating]) -> PredictionResult:
        """Return deterministic pseudo-probabilities for a feature tensor."""

        # Use a simple numeric signature from the incoming point features so the
        # demo changes over time while remaining deterministic for the same data.
        signature = float(np.nan_to_num(features).sum())
        raw = np.array(
            [
                1.0 + 0.25 * math.sin(signature + index * 2.0)
                for index, _ in enumerate(self.labels)
            ],
            dtype=np.float64,
        )
        probabilities = raw / float(np.sum(raw))
        return _result_from_probabilities(self.labels, probabilities)


def _result_from_probabilities(
    labels: Sequence[str], probabilities: NDArray[np.floating]
) -> PredictionResult:
    """Package probability vectors into the backend prediction shape."""

    best_index = int(np.argmax(probabilities))
    probs = {
        label: float(probabilities[index])
        for index, label in enumerate(labels)
    }
    return PredictionResult(
        label=labels[best_index],
        confidence=float(probabilities[best_index]),
        probabilities=probs,
    )


def load_predictor(
    path: Optional[Union[str, Path]] = None,
) -> PointCloudPoseClassifier | CNNPoseClassifier | SklearnPoseClassifier:
    """Load a trained predictor and detect the PyTorch architecture from its state."""

    if path is None:
        raise FileNotFoundError("No model checkpoint is configured")
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    file_type = model_path.suffix.lower()
    if file_type == ".pt":
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            # The repository's older raster checkpoint stores NumPy objects that
            # PyTorch's restricted loader rejects. It remains a trusted, tracked
            # compatibility artifact; new point-cloud checkpoints load safely.
            return CNNPoseClassifier(model_path)

        state = checkpoint.get("model_state_dict", {})
        if "shared_mlp.0.weight" in state:
            return PointCloudPoseClassifier(model_path, checkpoint=checkpoint)
        return CNNPoseClassifier(model_path)
    if file_type == ".joblib":
        return SklearnPoseClassifier(model_path)
    raise ValueError(f"Unsupported model checkpoint type: {model_path.suffix}")
