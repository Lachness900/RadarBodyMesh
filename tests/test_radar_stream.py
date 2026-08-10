import unittest

import numpy as np

from mm_yoga.backend.stream import RadarStreamProcessor
from mm_yoga.model.inference import PredictionResult


class CapturingPredictor:
    def __init__(self):
        self.inputs = []

    def predict(self, points):
        self.inputs.append(np.asarray(points).copy())
        if len(self.inputs) == 1:
            probabilities = {"standing_pose": 1.0, "t_pose": 0.0}
        else:
            probabilities = {"standing_pose": 0.0, "t_pose": 1.0}
        label = max(probabilities, key=probabilities.get)
        return PredictionResult(label, probabilities[label], probabilities)


def make_frame(point_count, offset):
    axis = np.linspace(-0.5, 0.5, point_count)
    return np.stack(
        [axis + offset, np.sin(axis) + offset, np.cos(axis) + offset],
        axis=1,
    )


class RadarStreamTests(unittest.TestCase):
    def test_stream_matches_visualizer_flush_and_uses_raw_predictions(self):
        predictor = CapturingPredictor()
        processor = RadarStreamProcessor(predictor=predictor, source="replay")

        self.assertIsNone(
            processor.process_frame(timestamp_ms=0.0, points=make_frame(60, 1.0))
        )
        first_message = processor.process_frame(
            timestamp_ms=100.0,
            points=make_frame(50, 2.0),
        )

        self.assertEqual(len(predictor.inputs), 1)
        self.assertEqual(predictor.inputs[0].shape, (110, 3))
        np.testing.assert_allclose(predictor.inputs[0][:60].mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(predictor.inputs[0][60:].mean(axis=0), 0.0, atol=1e-12)
        self.assertEqual(first_message["prediction"]["label"], "standing_pose")

        second_message = processor.process_frame(
            timestamp_ms=200.0,
            points=make_frame(90, 3.0),
        )
        self.assertEqual(len(predictor.inputs), 2)
        self.assertEqual(predictor.inputs[1].shape, (100, 3))
        self.assertEqual(second_message["prediction"]["label"], "t_pose")
        self.assertEqual(second_message["prediction"]["probabilities"]["t_pose"], 1.0)


if __name__ == "__main__":
    unittest.main()
