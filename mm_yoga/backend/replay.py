"""Replay helpers that convert recordings into frontend prediction messages."""

from __future__ import annotations

import asyncio
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
from mm_yoga.model.inference import MockPosePredictor


def build_message(
    *,
    timestamp_ms: float,
    source: str,
    points: np.ndarray,
    predictor,
    fps: float,
    latency_ms: float,
    display_points: Optional[np.ndarray] = None,
    point_sets: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, object]:
    """Build the JSON payload consumed by the frontend dashboard.

    ``points`` are the predictor input points. ``display_points`` can be
    different: for the MVP we show raw replay points in the UI while the
    predictor receives the filtered/accumulated radar points.
    """

    started = time.perf_counter()
    features = points_to_features(points)
    prediction = predictor.predict(features)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
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
        metrics=MetricsPayload(fps=fps, latency_ms=latency_ms + elapsed_ms),
    ).to_dict()


def mock_message(timestamp_ms: float = 0.0, *, interval_s: float = 0.1) -> dict[str, object]:
    """Build one mock dashboard message for REST smoke tests and fallbacks."""

    theta = np.linspace(0, 2 * np.pi, 80)
    radius = 0.4 + 0.1 * np.sin(theta * 3 + timestamp_ms / 500.0)
    x = 2.8 + radius * np.cos(theta)
    y = 0.2 + radius * np.sin(theta)
    z = 0.7 + 0.4 * np.sin(theta * 2 + timestamp_ms / 800.0)
    points = np.stack([x, y, z], axis=1)
    return build_message(
        timestamp_ms=timestamp_ms,
        source="mock",
        points=points,
        point_sets={
            "projected_radar": _projected_radar_points(points),
            "filtered_radar": points,
            "raw_radar": points,
        },
        predictor=MockPosePredictor(),
        fps=1.0 / interval_s,
        latency_ms=0.0,
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

def _center_points(
        points: np.ndarray
) -> np.ndarray:
    """
    Centres the average data to the point (0,0,1)
    """
    
    radar_points = np.asarray(points, dtype=np.float64).copy()
    if radar_points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    total_point = [0,0,0]
    for point in points:
        total_point[0] += point[0]
        total_point[1] += point[1]
        total_point[2] += point[2]
    radar_points[:, 0] = radar_points[:, 0] - total_point[0]/len(points)
    radar_points[:, 1] = radar_points[:, 1] - total_point[1]/len(points)
    radar_points[:, 2] = radar_points[:, 2] - total_point[2]/len(points) + 1
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
    # Three histories are maintained on purpose:
    # - raw_history keeps the original replay xyz points for inspection.
    # - filtered_history keeps the visualizer ROI points for model/focused views.
    # - projected_radar is derived from filtered_history with x flattened to 0,
    #   matching the current Python visualizer display convention.
    raw_history = np.empty((0, 3), dtype=np.float64)
    model_history = np.empty((0, 3), dtype=np.float64)
    reader = DatFrameReader(replay_file)
    for frame in reader.iter_frames(message_types={RADAR_MESSAGE}):
        filtered = _center_points(filter_points(frame.points, bounds=DEFAULT_RADAR_BOUNDS))
        raw_history = _append_recent_points(raw_history, frame.points, limit=256)
        model_history = _append_recent_points(model_history, filtered, limit=128)
        projected_radar = _projected_radar_points(model_history[-100:])
        if previous_timestamp_us is None:
            fps = 0.0
        else:
            # Timestamps in the recording are microseconds from the beginning of
            # the capture, so frame-to-frame delta gives replay FPS.
            diff_us = max(1, frame.timestamp_us - previous_timestamp_us)
            fps = 1_000_000.0 / diff_us
        previous_timestamp_us = frame.timestamp_us

        yield build_message(
            timestamp_ms=frame.timestamp_ms,
            source="replay",
            points=model_history,
            display_points=projected_radar if len(projected_radar) else raw_history,
            point_sets={
                "projected_radar": projected_radar,
                "filtered_radar": model_history,
                "raw_radar": raw_history,
            },
            predictor=predictor,
            fps=fps,
            latency_ms=0.0,
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
