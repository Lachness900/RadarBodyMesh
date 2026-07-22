"""Preprocessing utilities used by the replay backend."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

Bound = Tuple[float, float]
Bounds3D = Tuple[Bound, Bound, Bound]

DEFAULT_RADAR_BOUNDS: Bounds3D = ((2.0, 4.0), (-1.0, 2.0), (-1.0, 1.0))
DEFAULT_MAX_POINTS = 128


def filter_points(
    points: NDArray[np.floating],
    *,
    bounds: Optional[Bounds3D] = DEFAULT_RADAR_BOUNDS,
) -> NDArray[np.float64]:
    """Keep points inside an axis-aligned 3D box.

    The default bounds are inherited from the existing visualizer's rough subject
    area. They are useful for a first replay MVP, but should be re-tuned once the
    team records the final labelled yoga dataset.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be a 2D array with at least x, y, z columns")
    if bounds is None:
        return points[:, :3]

    (x_bound, y_bound, z_bound) = bounds
    xyz = points[:, :3]
    mask = (
        (xyz[:, 0] >= x_bound[0])
        & (xyz[:, 0] <= x_bound[1])
        & (xyz[:, 1] >= y_bound[0])
        & (xyz[:, 1] <= y_bound[1])
        & (xyz[:, 2] >= z_bound[0])
        & (xyz[:, 2] <= z_bound[1])
    )
    return xyz[mask]


def points_to_features(
    points: NDArray[np.floating],
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> NDArray[np.float32]:
    """Convert variable-length point clouds to fixed ``max_points x 6`` features.

    Feature columns are ``x, y, z, intensity, velocity, range``. Raw replay
    files only contain xyz, so missing intensity and velocity are filled with 0.
    This fixed shape is used by the deterministic mock predictor. The trained
    CNN consumes variable-length xyz batches directly before Y-Z rasterization.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((max_points, 6), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be a 2D array with at least x, y, z columns")

    xyz = points[:, :3]
    intensity = points[:, 3:4] if points.shape[1] > 3 else np.zeros((len(points), 1))
    velocity = points[:, 4:5] if points.shape[1] > 4 else np.zeros((len(points), 1))
    range_col = (
        points[:, 5:6]
        if points.shape[1] > 5
        else np.linalg.norm(xyz, axis=1, keepdims=True)
    )
    features = np.concatenate([xyz, intensity, velocity, range_col], axis=1)

    # Pad or truncate to keep a stable tensor shape for model code.
    fixed = np.zeros((max_points, 6), dtype=np.float32)
    copy_count = min(max_points, len(features))
    fixed[:copy_count] = features[:copy_count].astype(np.float32)
    return fixed


def to_frontend_points(
    points_or_features: NDArray[np.floating],
    *,
    limit: int = 256,
) -> list[dict[str, float]]:
    """Convert point arrays to compact JSON-friendly point dictionaries.

    FastAPI can serialize plain dict/list values directly. Keeping this
    conversion in the backend means the React app only deals with simple JSON.
    """

    points = np.asarray(points_or_features, dtype=np.float64)
    if points.size == 0:
        return []
    rows = points[:limit]
    result: list[dict[str, float]] = []
    for row in rows:
        if row.shape[0] < 3:
            continue
        result.append(
            {
                "x": float(row[0]),
                "y": float(row[1]),
                "z": float(row[2]),
                "intensity": float(row[3]) if row.shape[0] > 3 else 0.0,
                "velocity": float(row[4]) if row.shape[0] > 4 else 0.0,
            }
        )
    return result
