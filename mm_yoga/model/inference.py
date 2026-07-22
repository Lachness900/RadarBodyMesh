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

DEFAULT_POSE_LABELS = [
    "t_pose",
    "standing_pose",
    "warrior_1_pose",
    "warrior_2_pose",
    "angle_pose",
    "other",
]


@dataclass(frozen=True)
class PredictionResult:
    """Pose prediction result returned by any backend predictor."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class PoseCNN(nn.Module):
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


class PoseClassifier:
    """Load and run the CNN checkpoint trained from Y-Z radar histograms."""

    def __init__(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.labels = list(ckpt["label_names"])
        self.grid_size = int(ckpt["grid_size"])
        self.lo = np.asarray(ckpt["grid_bounds_lo"], dtype=np.float64)
        self.hi = np.asarray(ckpt["grid_bounds_hi"], dtype=np.float64)

        self.model = PoseCNN(num_classes=len(self.labels), grid_size=self.grid_size)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def rasterize(self, yz_points: np.ndarray) -> np.ndarray:
        hist, _, _ = np.histogram2d(
            yz_points[:, 0], yz_points[:, 1],
            bins=self.grid_size,
            range=[[self.lo[0], self.hi[0]], [self.lo[1], self.hi[1]]],
        )
        if hist.sum() > 0:
            hist = hist / hist.sum()
        return hist.astype(np.float32)

    def predict(self, points: NDArray[np.floating]) -> PredictionResult:
        """Predict from variable-length xyz points using their Y-Z projection."""

        yz_points = points[:, 1:3]
        grid = self.rasterize(yz_points)
        tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
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
) -> MockPosePredictor | PoseClassifier:
    """Load a trained checkpoint, falling back to mock output if unavailable."""

    if path is None:
        return MockPosePredictor(labels)

    model_path = Path(path)
    if not model_path.exists():
        return MockPosePredictor(labels)

    return PoseClassifier(model_path)
