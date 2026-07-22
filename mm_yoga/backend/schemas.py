"""JSON message builders for the frontend WebSocket contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class PredictionPayload:
    """Pose prediction section of a dashboard message."""

    label: str
    confidence: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class MetricsPayload:
    """Recorded radar FPS and latest backend model-inference latency."""

    fps: float
    latency_ms: float


@dataclass(frozen=True)
class PredictionMessage:
    """Complete WebSocket/REST payload sent from backend to frontend.

    The frontend intentionally receives JSON-ready dictionaries rather than raw
    numpy arrays so it never needs to know how replay files are parsed.
    """

    timestamp_ms: float
    source: str
    prediction: PredictionPayload
    points: list[dict[str, float]]
    point_sets: Optional[dict[str, list[dict[str, float]]]]
    metrics: MetricsPayload

    def to_dict(self) -> dict[str, object]:
        """Convert nested dataclasses into a JSON-serializable dictionary."""

        return asdict(self)
