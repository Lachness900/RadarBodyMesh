"""Replay helpers that convert recordings into frontend prediction messages."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Union

import numpy as np

from mm_yoga.backend.stream import (
    RadarStreamProcessor,
    build_message,
    predict_points,
    projected_radar_points,
)
from mm_yoga.data.parser import RADAR_MESSAGE, DatFrameReader
from mm_yoga.model.inference import MockPosePredictor


def mock_message(timestamp_ms: float = 0.0, *, interval_s: float = 0.1) -> dict[str, object]:
    """Build one mock dashboard message for REST smoke tests and fallbacks."""

    theta = np.linspace(0, 2 * np.pi, 80)
    radius = 0.4 + 0.1 * np.sin(theta * 3 + timestamp_ms / 500.0)
    x = 2.8 + radius * np.cos(theta)
    y = 0.2 + radius * np.sin(theta)
    z = 0.7 + 0.4 * np.sin(theta * 2 + timestamp_ms / 800.0)
    points = np.stack([x, y, z], axis=1)
    prediction, latency_ms = predict_points(MockPosePredictor(), points)
    return build_message(
        timestamp_ms=timestamp_ms,
        source="mock",
        points=points,
        prediction=prediction,
        point_sets={
            "projected_radar": projected_radar_points(points),
            "filtered_radar": points,
            "raw_radar": points,
        },
        fps=1.0 / interval_s,
        latency_ms=latency_ms,
    )


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

    emitted = 0
    processor = RadarStreamProcessor(predictor=predictor, source="replay")
    reader = DatFrameReader(replay_file)
    for frame in reader.iter_frames(message_types={RADAR_MESSAGE}):
        message = processor.process_frame(
            timestamp_ms=frame.timestamp_ms,
            points=frame.points,
        )
        if message is None:
            continue
        yield message
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
