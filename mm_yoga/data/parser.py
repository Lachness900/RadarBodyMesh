"""Streaming parser for synchronized depth-camera/mmWave point-cloud recordings.

The recording format is documented in
``OneDrive/DepthCam_Radar_Cloud_Combined/FORMAT.md``. Files are Zstandard
compressed byte streams. The current packet format is:

    b"::" + uint32 timestamp_us + uint8 message_type
    + uint32 payload_size + int16 xyz payload + b";;"

The first sample recordings used an older packet format without
``payload_size``. This parser supports both so old smoke-test recordings and
new labelled recordings can use the same backend.

Coordinates are stored as millimetres and converted back to metres here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable, Iterator, Optional, Sequence, Set, Union

import numpy as np
from numpy.typing import NDArray
import zstandard as zstd

HEADER = b"::"
FOOTER = b";;"
LEGACY_METADATA_FORMAT = "<2sIB"
LEGACY_METADATA_SIZE = struct.calcsize(LEGACY_METADATA_FORMAT)
SIZED_METADATA_FORMAT = "<2sIBI"
SIZED_METADATA_SIZE = struct.calcsize(SIZED_METADATA_FORMAT)
# Backwards-compatible aliases for older import sites.
METADATA_FORMAT = LEGACY_METADATA_FORMAT
METADATA_SIZE = LEGACY_METADATA_SIZE
BYTES_PER_POINT = np.dtype(np.int16).itemsize * 3
# FORMAT.md describes payloads up to roughly 2 MB. Keep headroom for camera
# frames while still rejecting obviously bogus legacy payload bytes interpreted
# as a new-format length.
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024

DEPTH_CAMERA_MESSAGE = 1
RADAR_MESSAGE = 2


@dataclass(frozen=True)
class PointCloudFrame:
    """A single decoded point-cloud frame from the custom ``.dat`` format.

    ``points`` is always an ``N x 3`` xyz array in metres. ``message_type``
    tells callers whether the frame came from the depth camera or the radar.
    """

    timestamp_us: int
    message_type: int
    points: NDArray[np.float64]

    @property
    def timestamp_ms(self) -> float:
        """Timestamp converted from microseconds to milliseconds."""

        return self.timestamp_us / 1000.0

    @property
    def source(self) -> str:
        """Human-readable frame source used by logs and frontend messages."""

        if self.message_type == DEPTH_CAMERA_MESSAGE:
            return "depth_camera"
        if self.message_type == RADAR_MESSAGE:
            return "radar"
        return "unknown"


class DatFrameReader:
    """Read custom Zstandard-compressed point-cloud ``.dat`` recordings.

    This small wrapper keeps call sites readable. The actual parsing is done by
    ``iter_dat_frames`` so tests and tools can use either API.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        chunk_size: int = 1024 * 1024,
        strict: bool = False,
    ) -> None:
        """Store parser options without opening the file yet."""

        self.path = Path(path)
        self.chunk_size = chunk_size
        self.strict = strict

    def iter_frames(
        self, message_types: Optional[Iterable[int]] = None
    ) -> Iterator[PointCloudFrame]:
        """Yield decoded frames, optionally filtering by message type."""

        yield from iter_dat_frames(
            self.path,
            message_types=message_types,
            chunk_size=self.chunk_size,
            strict=self.strict,
        )

    def __iter__(self) -> Iterator[PointCloudFrame]:
        """Allow ``for frame in DatFrameReader(path)`` style usage."""

        return self.iter_frames()


def iter_dat_frames(
    path: Union[str, Path],
    *,
    message_types: Optional[Iterable[int]] = None,
    chunk_size: int = 1024 * 1024,
    strict: bool = False,
) -> Iterator[PointCloudFrame]:
    """Stream frames from a compressed recording without loading it all at once."""

    file_path = Path(path)
    allowed_types: Optional[Set[int]] = (
        set(message_types) if message_types is not None else None
    )
    dctx = zstd.ZstdDecompressor()
    # The .dat file can be hundreds of MB, so keep only a rolling decompressed
    # buffer instead of materializing the entire recording in memory.
    buffer = b""

    with file_path.open("rb") as file_obj:
        with dctx.stream_reader(file_obj) as reader:
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break

                buffer += chunk
                while True:
                    # Packets can start in the middle of a decompressed chunk.
                    # Search the rolling buffer for the next full header/footer pair.
                    header_index = buffer.find(HEADER)
                    if header_index == -1:
                        # Keep a tiny suffix so a header split across two chunks
                        # (for example b":" then b":") can still be found later.
                        buffer = buffer[-(METADATA_SIZE - 1) :]
                        break

                    if len(buffer) < header_index + LEGACY_METADATA_SIZE:
                        # We found a header but not enough bytes for metadata yet.
                        buffer = buffer[header_index:]
                        break

                    buffer = buffer[header_index:]
                    sized_packet_length = _sized_packet_length(buffer)
                    if sized_packet_length is not None:
                        if len(buffer) < sized_packet_length:
                            break
                        packet = buffer[:sized_packet_length]
                        if packet[-len(FOOTER) :] == FOOTER:
                            buffer = buffer[sized_packet_length:]
                            frame = parse_packet(packet, strict=strict)
                            if frame is None:
                                continue
                            if allowed_types is not None and frame.message_type not in allowed_types:
                                continue
                            yield frame
                            continue
                        # If the claimed payload size does not lead to a footer,
                        # treat this as a legacy packet candidate below.

                    footer_index = buffer.find(FOOTER, LEGACY_METADATA_SIZE)
                    if footer_index == -1:
                        # Preserve the partial packet and continue after reading
                        # the next decompressed chunk.
                        break

                    packet = buffer[: footer_index + len(FOOTER)]
                    buffer = buffer[footer_index + len(FOOTER) :]
                    frame = parse_packet(packet, strict=strict)
                    if frame is None:
                        continue
                    if allowed_types is not None and frame.message_type not in allowed_types:
                        continue
                    yield frame


def _sized_packet_length(buffer: bytes) -> Optional[int]:
    """Return total new-format packet length when the header is plausible."""

    if len(buffer) < SIZED_METADATA_SIZE:
        return None
    try:
        header, _timestamp_us, _message_type, payload_size = struct.unpack(
            SIZED_METADATA_FORMAT, buffer[:SIZED_METADATA_SIZE]
        )
    except struct.error:
        return None
    if header != HEADER:
        return None
    if payload_size > MAX_PAYLOAD_BYTES:
        return None
    if payload_size % BYTES_PER_POINT != 0:
        return None
    return SIZED_METADATA_SIZE + payload_size + len(FOOTER)


def parse_packet(packet: bytes, *, strict: bool = False) -> Optional[PointCloudFrame]:
    """Parse one raw packet into a :class:`PointCloudFrame`."""

    sized_packet_length = _sized_packet_length(packet)
    if sized_packet_length == len(packet) and packet[-len(FOOTER) :] == FOOTER:
        return _parse_sized_packet(packet, strict=strict)
    return _parse_legacy_packet(packet, strict=strict)


def _parse_sized_packet(packet: bytes, *, strict: bool = False) -> Optional[PointCloudFrame]:
    """Parse the current size-prefixed packet format."""

    if len(packet) < SIZED_METADATA_SIZE + len(FOOTER):
        if strict:
            raise ValueError("Packet is shorter than metadata and footer.")
        return None

    try:
        header, timestamp_us, message_type, payload_size = struct.unpack(
            SIZED_METADATA_FORMAT, packet[:SIZED_METADATA_SIZE]
        )
    except struct.error:
        if strict:
            raise
        return None

    if header != HEADER or packet[-len(FOOTER) :] != FOOTER:
        if strict:
            raise ValueError("Packet delimiters are invalid.")
        return None

    payload = packet[SIZED_METADATA_SIZE : -len(FOOTER)]
    if len(payload) != payload_size:
        if strict:
            raise ValueError(
                f"Declared payload size {payload_size} does not match {len(payload)} bytes."
            )
        return None
    points = _payload_to_points(payload, strict=strict)
    if points is None:
        return None
    return PointCloudFrame(
        timestamp_us=int(timestamp_us),
        message_type=int(message_type),
        points=points,
    )


def _parse_legacy_packet(packet: bytes, *, strict: bool = False) -> Optional[PointCloudFrame]:
    """Parse the original packet format that had no payload-size field."""

    if len(packet) < LEGACY_METADATA_SIZE + len(FOOTER):
        if strict:
            raise ValueError("Packet is shorter than metadata and footer.")
        return None

    try:
        header, timestamp_us, message_type = struct.unpack(
            LEGACY_METADATA_FORMAT, packet[:LEGACY_METADATA_SIZE]
        )
    except struct.error:
        if strict:
            raise
        return None

    if header != HEADER or packet[-len(FOOTER) :] != FOOTER:
        if strict:
            raise ValueError("Packet delimiters are invalid.")
        return None

    payload = packet[LEGACY_METADATA_SIZE : -len(FOOTER)]
    points = _payload_to_points(payload, strict=strict)
    if points is None:
        return None
    return PointCloudFrame(
        timestamp_us=int(timestamp_us),
        message_type=int(message_type),
        points=points,
    )


def _payload_to_points(payload: bytes, *, strict: bool = False) -> Optional[NDArray[np.float64]]:
    """Convert an int16 millimetre xyz payload into metres."""

    if len(payload) % BYTES_PER_POINT != 0:
        if strict:
            raise ValueError(
                f"Point-cloud payload length {len(payload)} is not divisible by 6."
            )
        return None

    if not payload:
        return np.empty((0, 3), dtype=np.float64)

    # Convert back to metres so downstream code can use physical units.
    points = np.frombuffer(payload, dtype=np.int16).astype(np.float64) / 1000.0
    return points.reshape((-1, 3))


def read_preview(
    path: Union[str, Path], *, limit: int = 5, message_types: Optional[Sequence[int]] = None
) -> list[PointCloudFrame]:
    """Read a small number of frames for smoke tests and debugging.

    Unlike ``list(iter_dat_frames(path))``, this stops early and is safe to use
    with large replay files during quick format checks.
    """

    frames: list[PointCloudFrame] = []
    for frame in iter_dat_frames(path, message_types=message_types):
        frames.append(frame)
        if len(frames) >= limit:
            break
    return frames
