"""Prediction interface used by the mmYoga backend."""

from .inference import (
    DEFAULT_POSE_LABELS,
    CNNPoseClassifier,
    MockPosePredictor,
    PointCloudNet,
    PointCloudPoseClassifier,
    PredictionResult,
    SklearnPoseClassifier,
    load_predictor,
    normalize_and_features,
)

__all__ = [
    "DEFAULT_POSE_LABELS",
    "CNNPoseClassifier",
    "MockPosePredictor",
    "PointCloudNet",
    "PointCloudPoseClassifier",
    "PredictionResult",
    "SklearnPoseClassifier",
    "load_predictor",
    "normalize_and_features",
]
