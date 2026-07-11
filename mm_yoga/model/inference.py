"""Prediction helpers used by the replay backend.

There is intentionally no trained model implementation in this stage. The
backend exposes a small predictor interface and a deterministic mock predictor
so the frontend and replay pipeline can be tested before the final trained
model exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

DEFAULT_POSE_LABELS = [
    "t_pose",
    "straight_pose",
    "warrior_pose",
    "angle_pose",
    "other_pose",
]


@dataclass(frozen=True)
class PredictionResult:
    """Pose prediction result returned by any backend predictor."""

    label: str
    confidence: float
    probabilities: dict[str, float]


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
) -> MockPosePredictor:
    """Load the active predictor for the backend.

    For now, no real model loader is implemented. If no path is provided, the
    deterministic mock predictor is returned. When the final model is ready,
    this function is the narrow place where the team can connect it.
    """

    if path is None:
        return MockPosePredictor(labels)

    model_path = Path(path)
    if not model_path.exists():
        return MockPosePredictor(labels)

    raise NotImplementedError(
        f"Model loading is not implemented yet. Requested model file: {model_path}"
    )
