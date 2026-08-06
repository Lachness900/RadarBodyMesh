"""Pose-classifier and mock prediction helpers used by the replay backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
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

## Should be joblib path
class SklearnPoseClassifier:
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
    *,
    labels: Sequence[str] = DEFAULT_POSE_LABELS,
) -> CNNPoseClassifier | SklearnPoseClassifier:
    """Load a trained checkpoint, falling back to mock output if unavailable."""

    model_path = Path(path)

    assert(model_path.exists())

    file_type = model_path.name.split('.')[-1]
    if file_type == 'pt':
        return CNNPoseClassifier(model_path)
    elif file_type == 'joblib':
        return SklearnPoseClassifier(model_path)
    else:
        raise Exception("No model found")
