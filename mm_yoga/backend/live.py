'''forward streaming data from udp socket to websocket'''

from __future__ import annotations
import asyncio
from collections import deque
from typing import AsyncIterator, Iterator, Optional, Union
import numpy as np
from mm_yoga.backend.replay import (
    _append_recent_points,
    _center_points,
    _predict_points,
    _projected_radar_points,
    build_message
)
from mm_yoga.model.inference import (
    MockPosePredictor,
    CNNPoseClassifier,
    SklearnPoseClassifier,
    PredictionResult,
)
from mm_yoga.data.preprocess import (
    DEFAULT_RADAR_BOUNDS,
    Bounds3D,
    filter_points,
    points_to_features,
    to_frontend_points,
)
CLASSIFIER_RADAR_BOUNDS: Bounds3D = ((2.0, 4.0), (-1.0, 2.0), (-1.5, 1.5))
CLASSIFIER_BATCH_POINTS = 100

#from replay import 
from point_visualizer.decoder import UDPStreamReader

message_q = deque()

async def async_live_messages(
    *,
    predictor, 
    host: str = "239.255.0.1",
    port: int = 4200,
) -> AsyncIterator[dict[str, object]]:
    # send data to websocket backend api
    print("starting live stream")
    try:
        async for message in iter_live_messages(predictor=predictor):
            yield message
    except asyncio.CancelledError:
        print("live stream cancelled")
        return


async def iter_live_messages(
    *,
    predictor,
) -> AsyncIterator[dict[str, object]]:
    """Yield dashboard messages from a udp socket."""

    previous_timestamp_us: Optional[int] = None

    # Display history and classifier input stay separate on purpose:
    # - raw_history keeps the original replay xyz points for inspection.
    # - display_history is centered at z=1 for the focused dashboard view.
    # - classifier_pending exactly follows the trained visualizer pipeline:
    #   its wider ROI is centered at the origin and consumed in 100-point batches.
    raw_history = np.empty((0, 3), dtype=np.float64)
    display_history = np.empty((0, 3), dtype=np.float64)
    classifier_pending = np.empty((0, 3), dtype=np.float64)
    last_prediction: PredictionResult | None = None
    prediction: PredictionResult | None = None
    last_inference_latency_ms = 0.0
    frame_intervals_us: deque[int] = deque(maxlen=10)
    # read from udp socket
    reader = UDPStreamReader()
    
    for timestamp_us, pointCloud in reader.frames():
        print("Radar frame:", timestamp_us, pointCloud.shape)
        display_points = _center_points(
            filter_points(pointCloud, bounds=DEFAULT_RADAR_BOUNDS),
            target_z=1.0,
        )
        classifier_points = _center_points(
            filter_points(pointCloud, bounds=CLASSIFIER_RADAR_BOUNDS),
            target_z=0.0,
        )
        raw_history = _append_recent_points(raw_history, pointCloud, limit=256)
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

        #print("classifier points:",len(classifier_pending))
        if len(classifier_pending) >= CLASSIFIER_BATCH_POINTS:
            # Match visualizer_with_classifier.py: classify all points currently
            # accumulated, then consume exactly one 100-point batch.
            prediction, last_inference_latency_ms = _predict_points(
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
            diff_us = max(1, timestamp_us - previous_timestamp_us)
            frame_intervals_us.append(diff_us)
            fps = 1_000_000.0 / (sum(frame_intervals_us) / len(frame_intervals_us))
        previous_timestamp_us = timestamp_us

        # Do not invent a padded prediction while the first real batch warms up.
        if prediction is None:
            continue
        elif last_prediction is None:
            last_prediction = prediction
        else:
            new_confidence = 0
            for label in prediction.probabilities.keys():
                last_prediction.probabilities[label] = prediction.probabilities[label]*0.2 + last_prediction.probabilities[label]*0.8
                if last_prediction.probabilities[label] > new_confidence:
                    last_prediction.label = label
                    last_prediction.confidence = last_prediction.probabilities[label]
                    new_confidence = last_prediction.confidence
        
        yield build_message(
            timestamp_ms=timestamp_us / 1000,
            source="live",
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