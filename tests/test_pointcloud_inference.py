from pathlib import Path
import unittest

import numpy as np

from mm_yoga.model.inference import PointCloudPoseClassifier, load_predictor


ROOT = Path(__file__).resolve().parents[1]
FINAL_CHECKPOINT = ROOT / "pointcloud_classifier240.pt"


class PointCloudInferenceTests(unittest.TestCase):
    def test_final_checkpoint_matches_visualizer_reference_probabilities(self):
        predictor = load_predictor(FINAL_CHECKPOINT)
        self.assertIsInstance(predictor, PointCloudPoseClassifier)

        axis = np.linspace(-1.0, 1.0, 137, dtype=np.float32)
        points = np.stack(
            [axis * 0.4, np.sin(axis * 2.0) * 0.6, np.cos(axis * 3.0) * 0.8],
            axis=1,
        )
        points -= points.mean(axis=0)

        prediction = predictor.predict(points)
        expected = {
            "angle_pose": 0.0000823385562398471,
            "squat": 0.000000007550651659471441,
            "standing_pose": 0.8075569868087769,
            "t_pose": 0.19236071407794952,
        }

        self.assertEqual(prediction.label, "standing_pose")
        self.assertEqual(set(prediction.probabilities), set(expected))
        for label, probability in expected.items():
            self.assertAlmostEqual(
                prediction.probabilities[label],
                probability,
                places=6,
            )
        self.assertAlmostEqual(sum(prediction.probabilities.values()), 1.0, places=6)

    def test_checkpoint_accepts_variable_length_point_sets(self):
        predictor = load_predictor(FINAL_CHECKPOINT)
        rng = np.random.default_rng(6733)

        for point_count in (1, 37, 100, 143):
            prediction = predictor.predict(rng.normal(size=(point_count, 3)))
            probabilities = np.asarray(list(prediction.probabilities.values()))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
