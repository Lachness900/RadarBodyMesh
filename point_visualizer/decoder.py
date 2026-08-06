#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import queue
from selectors import DefaultSelector, EVENT_READ
import socket
import struct
import threading

import numpy as np

# if sys.version_info < (3, 14):
#     from compression.zstd import ZstdDecompressor
# else:

from zstandard import ZstdDecompressor


UDP_HEADER = struct.Struct("<2sIH")
UDP_SOF = b"::"
UDP_EOF = b";;"
MAX_UDP_PACKET_SIZE = 65_535


def decode_udp_packet(packet: bytes) -> tuple[int, np.ndarray] | None:
    """Decode one complete UDP datagram from ``PointCloudUDPBridge``.

    The wire format is intentionally kept identical to the ROS2 bridge:
    ``::`` + uint32 timestamp + uint16 payload length + int16 xyz + ``;;``.
    Malformed or truncated datagrams are dropped instead of entering the model.
    """

    minimum_size = UDP_HEADER.size + len(UDP_EOF)
    if len(packet) < minimum_size:
        return None

    sof, timestamp_us, payload_len = UDP_HEADER.unpack_from(packet)
    if sof != UDP_SOF or payload_len % 6 != 0:
        return None

    expected_size = UDP_HEADER.size + payload_len + len(UDP_EOF)
    if len(packet) != expected_size or packet[-len(UDP_EOF) :] != UDP_EOF:
        return None

    payload = packet[UDP_HEADER.size : UDP_HEADER.size + payload_len]
    points = np.frombuffer(payload, dtype="<i2").reshape((-1, 3))
    return timestamp_us, points.astype(np.float32) / 1000.0


class UDPStreamReader:
    """Receive complete point-cloud UDP datagrams on a background thread."""

    def __init__(
        self,
        host: str = "239.255.0.1",
        port: int = 4200,
        interface: str = "127.0.0.1",
    ):
        self.host = host
        self.port = port
        self.interface = interface
        self.udp_sock_ = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
            socket.IPPROTO_UDP,
        )
        self.udp_sock_.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_sock_.bind(("", port))
        self.udp_sock_.setblocking(False)

        # Join the same multicast group used by PointCloudUDPBridge.
        membership = struct.pack(
            "=4s4s",
            socket.inet_aton(host),
            socket.inet_aton(interface),
        )
        self.udp_sock_.setsockopt(
            socket.IPPROTO_IP,
            socket.IP_ADD_MEMBERSHIP,
            membership,
        )

        self.sel_ = DefaultSelector()
        self.sel_.register(self.udp_sock_, EVENT_READ, self.__read_sock)
        self.q: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(5)
        self.working_ = True
        self._close_lock = threading.Lock()
        self.sock_listener_thrd = threading.Thread(
            target=self.__poll_loop,
            name="mm-yoga-udp-reader",
            daemon=True,
        )
        self.sock_listener_thrd.start()
        print(f"Listening to UDP IP {host}:{port} on {interface} ...")

    def __enter__(self) -> "UDPStreamReader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __poll_loop(self) -> None:
        """Run the OS selector until ``close`` releases the socket."""

        while self.working_:
            try:
                events = self.sel_.select(timeout=0.1)
            except (OSError, ValueError):
                break
            for key, mask in events:
                key.data(key.fileobj, mask)

    def close(self) -> None:
        """Stop receiving and release the selector, socket, and listener thread."""

        with self._close_lock:
            if not self.working_:
                return
            self.working_ = False
            try:
                self.sel_.unregister(self.udp_sock_)
            except (KeyError, ValueError):
                pass
            self.sel_.close()
            self.udp_sock_.close()
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put_nowait(None)

        if (
            self.sock_listener_thrd.is_alive()
            and threading.current_thread() is not self.sock_listener_thrd
        ):
            self.sock_listener_thrd.join(timeout=1.0)

    def __read_sock(self, sock: socket.socket, _mask: int) -> None:
        try:
            packet = sock.recv(MAX_UDP_PACKET_SIZE)
        except (BlockingIOError, OSError):
            return

        decoded = decode_udp_packet(packet)
        if decoded is None:
            return
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put_nowait(decoded)

    def frames(self, *, timeout_s: float = 2.0):
        """Yield decoded frames, raising when no radar datagram arrives in time."""

        while self.working_ or not self.q.empty():
            try:
                item = self.q.get(timeout=timeout_s)
            except queue.Empty as exc:
                raise TimeoutError("No UDP radar data received") from exc
            if item is None:
                break
            yield item


class ReaderParserError(Exception):
    def __init__(self, reason):
        super().__init__(reason)


class DatReader:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"{str(path)} does not exist")
        
        self._file_path = path
        
        pass

    def frame(self):
        """
        Parses a ZSTD compressed binary file containing PointCloud messages
        """
        with self._file_path.open('rb') as f:
            dctx = ZstdDecompressor()

            SOF_DELIMITER = b'::'
            EOF_DELIMITER = b';;'
            HEADER_META_SIZE = len(SOF_DELIMITER) + struct.calcsize("<IBI")

            with dctx.stream_reader(f) as reader:
                while True:
                    header = reader.read(HEADER_META_SIZE)
                    # print(header)
                    if header is None or not header:
                        break
                    elif len(header) < HEADER_META_SIZE:
                        print(header)
                        raise EOFError("Unexpected EOF while reading metadata.")
                    try:
                        _, timestamp_us, message_type, payload_length = struct.unpack(
                            '<2sIBI', 
                            header
                        )
                    except struct.error as e:
                        raise ReaderParserError(f"Invalid Parse Syntax: {e}")
                    # print(timestamp_us, message_type, payload_length)
                    
                    raw_payload = reader.read(payload_length)
                    point_cloud_np = np.frombuffer(raw_payload, dtype=np.int16) / 1000
                    point_cloud_np = point_cloud_np.reshape((-1, 3))
                    
                    footer = reader.read(len(EOF_DELIMITER))
                    if footer != EOF_DELIMITER:
                        raise ValueError(f"Stream corrupted: Expected {EOF_DELIMITER}, got {footer}")
                    
                    yield {
                        'timestamp_us': timestamp_us,
                        'message_type': message_type,
                        'point_cloud': point_cloud_np
                    }
            
            return None
        pass


if __name__ == "__main__":
    with UDPStreamReader() as udp_reader:
        for timestamp, pc_data in udp_reader.frames():
            print(timestamp, pc_data.shape)
