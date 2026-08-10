"""Shared UDP radar service for live model inference and WebSocket clients."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import AsyncIterator

import numpy as np

from mm_yoga.backend.stream import RadarStreamProcessor
from mm_yoga.model.inference import (
    CNNPoseClassifier,
    MockPosePredictor,
    PointCloudPoseClassifier,
    SklearnPoseClassifier,
)
from point_visualizer.decoder import UDPStreamReader


class LiveRadarUnavailable(RuntimeError):
    """Raised when the backend cannot start or continue a live UDP stream."""


class LiveRadarService:
    """Own one UDP reader and broadcast processed frames to WebSocket clients."""

    def __init__(
        self,
        *,
        host: str = "239.255.0.1",
        port: int = 4200,
        interface: str = "127.0.0.1",
        receive_timeout_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.interface = interface
        self.receive_timeout_s = receive_timeout_s
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader: UDPStreamReader | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._processor: RadarStreamProcessor | None = None
        self._subscribers: set[asyncio.Queue[object]] = set()
        self._latest_message: dict[str, object] | None = None
        self._last_frame_monotonic: float | None = None
        self._running = False
        self._error: str | None = None
        self._timestamp_origin_us: int | None = None
        self._last_timestamp_us: int | None = None
        self._timestamp_wrap_offset_us = 0

    def status(self) -> dict[str, object]:
        """Return JSON-ready receiver and recent-frame state for the API."""

        with self._lock:
            last_frame = self._last_frame_monotonic
            running = self._running
            error = self._error
        frame_age_s = None if last_frame is None else max(0.0, time.monotonic() - last_frame)
        return {
            "supported": True,
            "running": running,
            "receiving": running and frame_age_s is not None and frame_age_s < 2.5,
            "endpoint": f"{self.host}:{self.port}",
            "interface": self.interface,
            "last_frame_age_s": frame_age_s,
            "error": error,
        }

    def start(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        predictor: (
            MockPosePredictor
            | PointCloudPoseClassifier
            | CNNPoseClassifier
            | SklearnPoseClassifier
        ),
    ) -> None:
        """Start the UDP receiver thread if it is not already running."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._loop = loop
            self._processor = RadarStreamProcessor(predictor=predictor, source="live")
            self._latest_message = None
            self._last_frame_monotonic = None
            self._error = None
            self._timestamp_origin_us = None
            self._last_timestamp_us = None
            self._timestamp_wrap_offset_us = 0
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_udp,
                name="mm-pose-live-radar",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def stop(self) -> None:
        """Stop receiving and release the UDP socket during app shutdown."""

        self._stop_event.set()
        with self._lock:
            reader = self._reader
            thread = self._thread
        if reader is not None:
            reader.close()
        if (
            thread is not None
            and thread.is_alive()
            and threading.current_thread() is not thread
        ):
            thread.join(timeout=2.0)

    @property
    def latest_message(self) -> dict[str, object] | None:
        """Return the latest message produced after model warm-up."""

        with self._lock:
            return self._latest_message

    async def messages(self) -> AsyncIterator[dict[str, object]]:
        """Yield the newest live message to one WebSocket client."""

        subscriber: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
        self._subscribers.add(subscriber)
        try:
            latest = self.latest_message
            if latest is not None:
                subscriber.put_nowait(latest)
            while True:
                try:
                    item = await asyncio.wait_for(subscriber.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    with self._lock:
                        running = self._running
                        error = self._error
                        thread_alive = self._thread is not None and self._thread.is_alive()
                    if error and not running and not thread_alive:
                        raise LiveRadarUnavailable(error)
                    continue
                if isinstance(item, LiveRadarUnavailable):
                    raise item
                yield item  # type: ignore[misc]
        finally:
            self._subscribers.discard(subscriber)

    def _run_udp(self) -> None:
        """Receive and process UDP frames away from FastAPI's event loop."""

        reader: UDPStreamReader | None = None
        failure: str | None = None
        try:
            reader = UDPStreamReader(
                host=self.host,
                port=self.port,
                interface=self.interface,
            )
            with self._lock:
                self._reader = reader
                self._running = True

            for timestamp_us, points in reader.frames(timeout_s=self.receive_timeout_s):
                if self._stop_event.is_set():
                    break
                self._process_frame(timestamp_us, points)
        except TimeoutError:
            if not self._stop_event.is_set():
                failure = (
                    f"No Live Radar UDP packets received on {self.host}:{self.port} "
                    f"for {self.receive_timeout_s:.0f} seconds."
                )
        except Exception as exc:
            if not self._stop_event.is_set():
                failure = f"Live Radar receiver stopped: {exc}"
        finally:
            if reader is not None:
                reader.close()
            with self._lock:
                self._reader = None
                self._running = False
                if failure is not None:
                    self._error = failure
            if failure is not None:
                self._schedule(self._publish_error, failure)

    def _process_frame(self, timestamp_us: int, points: np.ndarray) -> None:
        """Run shared preprocessing/inference for one decoded UDP frame."""

        processor = self._processor
        if processor is None:
            return
        timestamp_ms = self._elapsed_timestamp_ms(timestamp_us)
        message = processor.process_frame(timestamp_ms=timestamp_ms, points=points)
        with self._lock:
            self._last_frame_monotonic = time.monotonic()
            self._error = None
        if message is not None:
            self._schedule(self._publish_message, message)

    def _elapsed_timestamp_ms(self, timestamp_us: int) -> float:
        """Convert the bridge's wrapping uint32 clock to elapsed milliseconds."""

        if (
            self._last_timestamp_us is not None
            and timestamp_us < self._last_timestamp_us
            and self._last_timestamp_us - timestamp_us > 2**31
        ):
            self._timestamp_wrap_offset_us += 2**32
        unwrapped_us = self._timestamp_wrap_offset_us + timestamp_us
        if self._timestamp_origin_us is None:
            self._timestamp_origin_us = unwrapped_us
        self._last_timestamp_us = timestamp_us
        return max(0.0, (unwrapped_us - self._timestamp_origin_us) / 1000.0)

    def _schedule(self, callback, *args) -> None:
        """Schedule one publisher callback safely on FastAPI's event loop."""

        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            pass

    def _publish_message(self, message: dict[str, object]) -> None:
        """Keep and fan out only the newest processed live message."""

        with self._lock:
            self._latest_message = message
        self._publish(message)

    def _publish_error(self, reason: str) -> None:
        """Wake waiting clients so the frontend can show an unavailable state."""

        self._publish(LiveRadarUnavailable(reason))

    def _publish(self, item: object) -> None:
        for subscriber in tuple(self._subscribers):
            if subscriber.full():
                try:
                    subscriber.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            subscriber.put_nowait(item)
