"""Shared radar-frame processing for replay and live input sources."""

from __future__ import annotations

from collections import deque
import os
import time
from typing import Optional

import numpy as np

from mm_yoga.backend.schemas import MetricsPayload, PredictionMessage, PredictionPayload
from mm_yoga.data.preprocess import (
    DEFAULT_RADAR_BOUNDS,
    Bounds3D,
    filter_points,
    points_to_features,
    to_frontend_points,
)
from mm_yoga.model.inference import (
    CNNPoseClassifier,
    MockPosePredictor,
    PredictionResult,
    SklearnPoseClassifier,
)

CLASSIFIER_RADAR_BOUNDS: Bounds3D = ((1.0, 4.0), (-1.0, 2.0), (-1.5, 1.5))
CLASSIFIER_BATCH_POINTS = 100
DEFAULT_PREDICTION_INTERVAL_MS = float(
    os.getenv("MMYOGA_PREDICTION_INTERVAL_MS", "500")
)


def build_message(
    *,
    timestamp_ms: float,
    source: str,
    points: np.ndarray,
    prediction: PredictionResult,
    fps: float,
    latency_ms: float,
    display_points: Optional[np.ndarray] = None,
    point_sets: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, object]:
    """Build the JSON payload consumed by the frontend dashboard."""

    json_point_sets = (
        {name: to_frontend_points(point_array) for name, point_array in point_sets.items()}
        if point_sets is not None
        else None
    )
    return PredictionMessage(
        timestamp_ms=timestamp_ms,
        source=source,
        prediction=PredictionPayload(
            label=prediction.label,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
        ),
        points=to_frontend_points(display_points if display_points is not None else points),
        point_sets=json_point_sets,
        metrics=MetricsPayload(fps=fps, latency_ms=latency_ms),
    ).to_dict()


def predict_points(
    predictor: MockPosePredictor | CNNPoseClassifier | SklearnPoseClassifier,
    points: np.ndarray,
) -> tuple[PredictionResult, float]:
    """Run one prediction and return its measured inference latency."""

    started = time.perf_counter()
    model_input = (
        points
        if isinstance(predictor, (CNNPoseClassifier, SklearnPoseClassifier))
        else points_to_features(points, max_points=CLASSIFIER_BATCH_POINTS)
    )
    prediction = predictor.predict(model_input)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return prediction, elapsed_ms


def projected_radar_points(points: np.ndarray) -> np.ndarray:
    """Flatten x for the visualizer-compatible projected radar view."""

    radar_points = np.asarray(points, dtype=np.float64).copy()
    if radar_points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    radar_points[:, 0] = 0.0
    return radar_points


def center_points(points: np.ndarray, *, target_z: float) -> np.ndarray:
    """Center xyz around ``(0, 0, target_z)`` without mutating the input."""

    radar_points = np.asarray(points, dtype=np.float64).copy()
    if radar_points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    radar_points = radar_points[:, :3]
    radar_points -= radar_points.mean(axis=0)
    radar_points[:, 2] += target_z
    return radar_points


def append_recent_points(
    current: np.ndarray,
    new_points: np.ndarray,
    *,
    limit: int,
) -> np.ndarray:
    """Append xyz points while keeping a bounded recent history."""

    points = np.asarray(new_points, dtype=np.float64)
    if points.size == 0:
        return current
    points = points[:, :3]
    if current.size == 0:
        return points[-limit:]
    return np.concatenate([current, points], axis=0)[-limit:]


def _smooth_prediction(
    previous: PredictionResult | None,
    current: PredictionResult,
) -> PredictionResult:
    """Blend prediction probabilities using the dashboard's existing 80/20 rule."""

    if previous is None:
        probabilities = dict(current.probabilities)
    else:
        probabilities = {
            label: current.probabilities.get(label, 0.0) * 0.2
            + previous.probabilities.get(label, 0.0) * 0.8
            for label in current.probabilities
        }
    label = max(probabilities, key=probabilities.get)
    return PredictionResult(
        label=label,
        confidence=probabilities[label],
        probabilities=probabilities,
    )


class RadarStreamProcessor:
    """Turn successive raw radar frames into model-backed dashboard messages.

    Replay and Live deliberately use this same stateful processor so their ROI,
    centering, accumulation, model batches, and displayed point sets stay equal.
    """

    def __init__(
        self,
        *,
        predictor: MockPosePredictor | CNNPoseClassifier | SklearnPoseClassifier,
        source: str,
        prediction_interval_ms: float = DEFAULT_PREDICTION_INTERVAL_MS,
    ) -> None:
        self.predictor = predictor
        self.source = source
        self.prediction_interval_ms = max(0.0, prediction_interval_ms)
        self.raw_history = np.empty((0, 3), dtype=np.float64)
        self.display_history = np.empty((0, 3), dtype=np.float64)
        self.classifier_pending = np.empty((0, 3), dtype=np.float64)
        self.current_prediction: PredictionResult | None = None
        self.last_prediction: PredictionResult | None = None
        self.last_inference_latency_ms = 0.0
        self._last_prediction_timestamp_ms: float | None = None
        self.previous_timestamp_ms: float | None = None
        self.frame_intervals_ms: deque[float] = deque(maxlen=10)
        with open("log.txt", "w"):
            pass

    def process_frame(
        self,
        *,
        timestamp_ms: float,
        points: np.ndarray,
    ) -> dict[str, object] | None:
        """Consume one raw xyz frame and emit after the first real model batch."""

        raw_points = np.asarray(points, dtype=np.float64)
        display_points = center_points(
            filter_points(raw_points, bounds=DEFAULT_RADAR_BOUNDS),
            target_z=1.0,
        )
        classifier_points = center_points(
            filter_points(raw_points, bounds=CLASSIFIER_RADAR_BOUNDS),
            target_z=0.0,
        )
        self.raw_history = append_recent_points(self.raw_history, raw_points, limit=256)
        self.display_history = append_recent_points(
            self.display_history,
            display_points,
            limit=128,
        )
        if len(classifier_points):
            self.classifier_pending = np.concatenate(
                [self.classifier_pending, classifier_points],
                axis=0,
            )

        if len(self.classifier_pending) >= CLASSIFIER_BATCH_POINTS:
            # This matches the trained visualizer: classify all accumulated real
            # points, then consume exactly one 100-point batch from the buffer.
            # Prediction cadence is throttled so the pose result stays stable
            # while the point cloud keeps streaming at full frame rate.
            should_predict = (
                self.prediction_interval_ms <= 0
                or self._last_prediction_timestamp_ms is None
                or timestamp_ms - self._last_prediction_timestamp_ms
                >= self.prediction_interval_ms
            )
            if should_predict:
                self.current_prediction, self.last_inference_latency_ms = predict_points(
                    self.predictor,
                    self.classifier_pending,
                )
                self._last_prediction_timestamp_ms = timestamp_ms
                self.classifier_pending = self.classifier_pending[
                    CLASSIFIER_BATCH_POINTS:
                ]

        fps = self._update_fps(timestamp_ms)
        if self.current_prediction is None:
            return None
        self.last_prediction = _smooth_prediction(
            self.last_prediction,
            self.current_prediction,
        )
        with open("log.txt", "a") as file:
            file.write(f"\n{self.last_prediction.label}")

        projected = projected_radar_points(self.display_history[-100:])
        return build_message(
            timestamp_ms=timestamp_ms,
            source=self.source,
            points=self.display_history,
            prediction=self.last_prediction,
            display_points=projected if len(projected) else self.raw_history,
            point_sets={
                "projected_radar": projected,
                "filtered_radar": self.display_history,
                "raw_radar": self.raw_history,
            },
            fps=fps,
            latency_ms=self.last_inference_latency_ms,
        )

    def _update_fps(self, timestamp_ms: float) -> float:
        """Estimate radar FPS from a smoothed window of source timestamps."""

        if self.previous_timestamp_ms is None:
            fps = 0.0
        else:
            interval_ms = max(0.001, timestamp_ms - self.previous_timestamp_ms)
            self.frame_intervals_ms.append(interval_ms)
            fps = 1000.0 / (sum(self.frame_intervals_ms) / len(self.frame_intervals_ms))
        self.previous_timestamp_ms = timestamp_ms
        return fps
