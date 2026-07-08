"""Prediction interface used by the mmYoga backend."""

from .inference import (
    DEFAULT_POSE_LABELS,
    MockPosePredictor,
    PredictionResult,
    load_predictor,
)

__all__ = [
    "DEFAULT_POSE_LABELS",
    "MockPosePredictor",
    "PredictionResult",
    "load_predictor",
]
