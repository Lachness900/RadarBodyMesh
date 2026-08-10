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

    def test_joining_live_does_not_replace_global_model(self):
        service = LiveRadarService()
        service._thread = _AliveThread()
        service._model_id = "pointcloud_120"
        current_predictor = object()
        service._processor = _ProcessorStub(current_predictor, "pointcloud_120")
        service._latest_message = {"model_id": "pointcloud_120"}
        loop = asyncio.new_event_loop()
        try:
            service.start(
                loop=loop,
                predictor=object(),
                model_id="pointcloud_240",
            )
        finally:
            loop.close()

        self.assertEqual(service._model_id, "pointcloud_120")
        self.assertEqual(service.status()["model_id"], "pointcloud_120")
        self.assertIs(service._processor.predictor, current_predictor)
        self.assertEqual(service._processor.model_id, "pointcloud_120")
        self.assertEqual(service.latest_message, {"model_id": "pointcloud_120"})

    def test_live_model_configuration_is_global_before_receiver_starts(self):
        service = LiveRadarService()
        predictor = object()

        changed = service.configure_model(
            predictor=predictor,
            model_id="pointcloud_120",
        )

        self.assertTrue(changed)
        self.assertEqual(service.model_id, "pointcloud_120")
        self.assertIs(service._processor.predictor, predictor)
        self.assertFalse(
            service.configure_model(
                predictor=predictor,
                model_id="pointcloud_120",
            )
        )

    def test_restarting_live_receiver_resets_same_model_processor(self):
        service = LiveRadarService()
        predictor = object()
        service.configure_model(
            predictor=predictor,
            model_id="pointcloud_120",
        )
        previous_processor = service._processor
        with service._lock:
            service._configure_model_locked(
                predictor=predictor,
                model_id="pointcloud_120",
                force_reset=True,
            )

        self.assertIsNot(service._processor, previous_processor)


class _AliveThread:
    def is_alive(self):
        return True


class _ProcessorStub:
    def __init__(self, predictor, model_id):
        self.predictor = predictor
        self.model_id = model_id


if __name__ == "__main__":
    unittest.main()
