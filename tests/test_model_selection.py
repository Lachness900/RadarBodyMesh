import asyncio
import unittest

from mm_yoga.backend.app import (
    _available_model_options,
    _default_model_option,
    _selected_model,
    sources,
)
from mm_yoga.backend.live import LiveRadarService


class ModelSelectionTests(unittest.TestCase):
    def test_sources_exposes_only_server_approved_model_ids(self):
        options = _available_model_options()
        payload = sources()

        self.assertGreaterEqual(len(options), 2)
        self.assertEqual(
            [model["id"] for model in payload["models"]],
            [option.id for option in options],
        )
        self.assertEqual(payload["default_model"], _default_model_option().id)
        self.assertTrue(all("path" not in model for model in payload["models"]))

    def test_unknown_model_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown or unavailable model"):
            _selected_model("not-a-checkpoint")

    def test_live_model_change_replaces_processor_and_clears_old_message(self):
        service = LiveRadarService()
        service._thread = _AliveThread()
        service._model_id = "pointcloud_120"
        service._latest_message = {"model_id": "pointcloud_120"}
        loop = asyncio.new_event_loop()
        predictor = object()
        try:
            service.start(
                loop=loop,
                predictor=predictor,
                model_id="pointcloud_240",
            )
        finally:
            loop.close()

        self.assertEqual(service._model_id, "pointcloud_240")
        self.assertIs(service._processor.predictor, predictor)
        self.assertEqual(service._processor.model_id, "pointcloud_240")
        self.assertIsNone(service.latest_message)


class _AliveThread:
    def is_alive(self):
        return True


if __name__ == "__main__":
    unittest.main()
