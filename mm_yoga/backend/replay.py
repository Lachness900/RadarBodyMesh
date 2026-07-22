"""Replay helpers that convert recordings into frontend prediction messages."""

from __future__ import annotations

import asyncio
from collections import deque
import time
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Union

import numpy as np

from mm_yoga.backend.schemas import MetricsPayload, PredictionMessage, PredictionPayload
from mm_yoga.data.parser import RADAR_MESSAGE, DatFrameReader
from mm_yoga.data.preprocess import (
    DEFAULT_RADAR_BOUNDS,
    Bounds3D,
    filter_points,
    points_to_features,
    to_frontend_points,
)
from mm_yoga.model.inference import (
    MockPosePredictor,
    PoseClassifier,
    PredictionResult,
)

CLASSIFIER_RADAR_BOUNDS: Bounds3D = ((2.0, 4.0), (-1.0, 2.0), (-1.5, 1.5))
CLASSIFIER_BATCH_POINTS = 100


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
    """Build the JSON payload consumed by the frontend dashboard.

    Inference is performed before this function so replay frames can retain the
    most recent batch prediction. ``points`` and ``display_points`` are used
    only for the dashboard point-cloud views.
    """

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


def _predict_points(
    predictor: MockPosePredictor | PoseClassifier,
    points: np.ndarray,
) -> tuple[PredictionResult, float]:
    """Run one prediction and return its measured inference latency.

    The CNN rasterizes a variable number of real xyz points directly. The mock
    predictor keeps the fixed feature tensor used by the frontend smoke test.
    """

    started = time.perf_counter()
    model_input = (
        points
        if isinstance(predictor, PoseClassifier)
        else points_to_features(points, max_points=CLASSIFIER_BATCH_POINTS)
    )
    prediction = predictor.predict(model_input)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return prediction, elapsed_ms


def mock_message(timestamp_ms: float = 0.0, *, interval_s: float = 0.1) -> dict[str, object]:
    """Build one mock dashboard message for REST smoke tests and fallbacks."""

    theta = np.linspace(0, 2 * np.pi, 80)
    radius = 0.4 + 0.1 * np.sin(theta * 3 + timestamp_ms / 500.0)
    x = 2.8 + radius * np.cos(theta)
    y = 0.2 + radius * np.sin(theta)
    z = 0.7 + 0.4 * np.sin(theta * 2 + timestamp_ms / 800.0)
    points = np.stack([x, y, z], axis=1)
    prediction, latency_ms = _predict_points(MockPosePredictor(), points)
    return build_message(
        timestamp_ms=timestamp_ms,
        source="mock",
        points=points,
        prediction=prediction,
        point_sets={
            "projected_radar": _projected_radar_points(points),
            "filtered_radar": points,
            "raw_radar": points,
        },
        fps=1.0 / interval_s,
        latency_ms=latency_ms,
    )


def _projected_radar_points(points: np.ndarray) -> np.ndarray:
    """Match the current Python visualizer's radar display convention.

    The team's Python visualizer filters radar points to a rough subject ROI,
    accumulates a short history, then sets x to 0 before plotting. Keeping this
    as a named display set makes the web view explainable without changing the
    raw replay data.
    """

    radar_points = np.asarray(points, dtype=np.float64).copy()
    if radar_points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    radar_points[:, 0] = 0.0
    return radar_points


def _center_points(points: np.ndarray, *, target_z: float) -> np.ndarray:
    """Center xyz around ``(0, 0, target_z)`` without mutating the input."""

    radar_points = np.asarray(points, dtype=np.float64).copy()
    if radar_points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    radar_points = radar_points[:, :3]
    radar_points -= radar_points.mean(axis=0)
    radar_points[:, 2] += target_z
    return radar_points


def _append_recent_points(
    current: np.ndarray,
    new_points: np.ndarray,
    *,
    limit: int,
) -> np.ndarray:
    """Keep a rolling window of recent xyz points.

    Individual radar frames can be sparse, especially after ROI filtering. The
    dashboard is easier to read when it sees a short history instead of a single
    frame that may contain only a few points.
    """

    points = np.asarray(new_points, dtype=np.float64)
    if points.size == 0:
        return current
    points = points[:, :3]
    if current.size == 0:
        return points[-limit:]
    return np.concatenate([current, points], axis=0)[-limit:]


def iter_replay_messages(
    replay_file: Union[str, Path],
    *,
    predictor,
    max_frames: Optional[int] = None,
) -> Iterator[dict[str, object]]:
    """Yield dashboard messages from a recorded radar replay.

    This is synchronous on purpose: REST endpoints can grab one message with
    ``max_frames=1``, while the WebSocket wrapper below adds async sleep timing.
    """

    previous_timestamp_us: Optional[int] = None
    emitted = 0
    # Display history and classifier input stay separate on purpose:
    # - raw_history keeps the original replay xyz points for inspection.
    # - display_history is centered at z=1 for the focused dashboard view.
    # - classifier_pending exactly follows the trained visualizer pipeline:
    #   its wider ROI is centered at the origin and consumed in 100-point batches.
    raw_history = np.empty((0, 3), dtype=np.float64)
    display_history = np.empty((0, 3), dtype=np.float64)
    classifier_pending = np.empty((0, 3), dtype=np.float64)
    last_prediction: Optional[PredictionResult] = None
    last_inference_latency_ms = 0.0
    frame_intervals_us: deque[int] = deque(maxlen=10)
    reader = DatFrameReader(replay_file)
    for frame in reader.iter_frames(message_types={RADAR_MESSAGE}):
        display_points = _center_points(
            filter_points(frame.points, bounds=DEFAULT_RADAR_BOUNDS),
            target_z=1.0,
        )
        classifier_points = _center_points(
            filter_points(frame.points, bounds=CLASSIFIER_RADAR_BOUNDS),
            target_z=0.0,
        )
        raw_history = _append_recent_points(raw_history, frame.points, limit=256)
        display_history = _append_recent_points(
            display_history,
            display_points,
            limit=128,
        )
        if len(classifier_points):
            classifier_pending = np.concatenate(
                [classifier_pending, classifier_points],
                axis=0,
            )

        if len(classifier_pending) >= CLASSIFIER_BATCH_POINTS:
            # Match visualizer_with_classifier.py: classify all points currently
            # accumulated, then consume exactly one 100-point batch.
            last_prediction, last_inference_latency_ms = _predict_points(
                predictor,
                classifier_pending,
            )
            classifier_pending = classifier_pending[CLASSIFIER_BATCH_POINTS:]

        projected_radar = _projected_radar_points(display_history[-100:])
        if previous_timestamp_us is None:
            fps = 0.0
        else:
            # Smooth recorded radar cadence over recent frames. This is the
            # sensor FPS stored in the replay, not browser rendering throughput.
            diff_us = max(1, frame.timestamp_us - previous_timestamp_us)
            frame_intervals_us.append(diff_us)
            fps = 1_000_000.0 / (sum(frame_intervals_us) / len(frame_intervals_us))
        previous_timestamp_us = frame.timestamp_us

        # Do not invent a padded prediction while the first real batch warms up.
        if last_prediction is None:
            continue

        yield build_message(
            timestamp_ms=frame.timestamp_ms,
            source="replay",
            points=display_history,
            prediction=last_prediction,
            display_points=projected_radar if len(projected_radar) else raw_history,
            point_sets={
                "projected_radar": projected_radar,
                "filtered_radar": display_history,
                "raw_radar": raw_history,
            },
            fps=fps,
            latency_ms=last_inference_latency_ms,
        )
        emitted += 1
        if max_frames is not None and emitted >= max_frames:
            break


async def async_replay_messages(
    replay_file: Union[str, Path],
    *,
    predictor,
    playback_speed: float = 1.0,
) -> AsyncIterator[dict[str, object]]:
    """Async replay stream that approximates the original recording cadence."""

    previous_timestamp_ms: Optional[float] = None
    speed = max(playback_speed, 0.1)
    for message in iter_replay_messages(replay_file, predictor=predictor):
        timestamp_ms = float(message["timestamp_ms"])
        if previous_timestamp_ms is not None:
            # Preserve the recording cadence, but cap each sleep so a long gap in
            # the file does not make the demo look frozen.
            delay_s = max(0.0, (timestamp_ms - previous_timestamp_ms) / 1000.0 / speed)
            await asyncio.sleep(min(delay_s, 0.25))
        previous_timestamp_ms = timestamp_ms
        yield message


async def async_mock_messages(*, interval_s: float = 0.1) -> AsyncIterator[dict[str, object]]:
    """Fallback stream used only when no replay file is available."""

    timestamp_ms = 0.0
    while True:
        # Keep the mock stream visibly alive even on machines without replay data.
        yield mock_message(timestamp_ms=timestamp_ms, interval_s=interval_s)
        timestamp_ms += interval_s * 1000.0
        await asyncio.sleep(interval_s)
