#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path
import socket
import struct
import threading
import signal
from typing import Tuple
from selectors import DefaultSelector, EVENT_READ
import queue


import numpy as np

if sys.version_info < (3, 14):
    from compression.zstd import ZstdDecompressor
else:
    from zstandard import ZstdDecompressor


class UDPStreamReader:
  def __init__(self, host: str = "239.255.0.1", port: int = 4200):

    self.udp_sock_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    self.udp_sock_.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    self.udp_sock_.bind(('239.255.0.1', port))
    self.udp_sock_.setblocking(False)

    # tell kernel to listen for packets from this host
    mreq = struct.pack("4sl", socket.inet_aton(host), socket.INADDR_ANY)
    self.udp_sock_.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    self.sel_ = DefaultSelector()
    self.sel_.register(self.udp_sock_, EVENT_READ, self.__read_sock)

    self.q:queue.Queue[Tuple] = queue.Queue(5)

    self.header = struct.Struct("<2sIH")
    self.SOF = b"::"
    self.EOF = b";;"

    self.working_ = True
    self.sock_listener_thrd = threading.Thread(target=self.__poll_loop)
    self.sock_listener_thrd.start()

    signal.signal(signal.SIGINT, self.__sigint_handler)

    print(f"Listening to UDP IP {host}:{port} ...")

  def __sigint_handler(self, sig, frame):
    self.__shutdown()

  def __poll_loop(self):
    """Background thread that runs the OS selector."""
    while self.working_:
      events = self.sel_.select(timeout=0.1)
      for key, mask in events:
        callback = key.data
        callback(key.fileobj, mask)

  def __shutdown(self):
    self.working_ = False
    self.sel_.unregister(self.udp_sock_)
    self.q.shutdown()
    self.udp_sock_.close()
    self.sock_listener_thrd.join()

  def __read_sock(self, sock: socket.socket,  mask):
    packet = sock.recv(1024)

    # print(len(packet))

    if not packet:
      self.__shutdown()
    else:
      if len(packet) < self.header.size + len(self.EOF):
        return
  
      prefix_bytes = packet[:self.header.size]
      sof_delim, timestamp_us, payload_len = self.header.unpack(prefix_bytes)
      # print(self.header.unpack(prefix_bytes))
  
      if sof_delim != self.SOF:
        print(f"Warning: Invalid header {sof_delim}")
        return
  
  
      expected_packet_size = self.header.size + payload_len + 2
      payload = packet[self.header.size : self.header.size + payload_len]
      eof_delim = packet[self.header.size + payload_len : expected_packet_size]
  
      if eof_delim != self.EOF:
        print(f"Warning: Invalid footer {eof_delim}")
        return
  
  
      pointcloud = np.frombuffer(payload, dtype=np.int16)
      pointcloud = pointcloud.reshape((-1, 3)).astype(np.float32) / 1000

      t = (timestamp_us, pointcloud)
      if self.q.qsize() >= 5:
        self.q.get_nowait()

      self.q.put_nowait(t)

  def frames(self):
    while True:
      try:
        yield self.q.get()
      except queue.Empty:
        continue
      except queue.ShutDown:
        break

    return 


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
   udp_reader = UDPStreamReader()

   for timestamp, pc_data in udp_reader.frames():
      print(timestamp, pc_data.shape)