import socket
import struct
import numpy as np
from typing import Union, Optional
from pathlib import Path
import zstandard as zstd
from zstandard import ZstdDecompressor

DEFAULT_ADDR = ""
DEFAULT_PORT = 5005

RADAR_FRAME_DIRECTORY = "radar_frames"
RADAR_EXAMPLE_FILE = r"raw.bin"
CLEANED_EXAMPLE_FILE = r"train.dat"
NEW_EXAMPLE_FILE = "cam_radar_1783260788477740516.dat"


reciever = (DEFAULT_ADDR, DEFAULT_PORT)


# IPv6 UDP server setup (for recieving radar packets)
def udp_radar_reciever(port: int = DEFAULT_PORT):
    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)

    sock.bind(reciever)
    print(f"IPv6 UDP server listening on {reciever[0]}:{port}")

    while True:
        try:
            data, addr = sock.recvfrom(4096)
            client_ip, client_port = addr[0], addr[1]
            print(f"Received {len(data)} bytes from [{client_ip}]:{client_port}")
        except KeyboardInterrupt:
            print("\nServer shutting down.")
            break
        except Exception as e:
            print(f"Error: {e}")

    sock.close()


# For manually reading a data file into the server
def import_data_from_desktop(file_name: Union[Path, str]):
    if isinstance(file_name, str):
        file = Path(file_name).resolve()
    elif isinstance(file_name, Path):
        file = file_name.resolve()
    else:
        raise RuntimeError("Invalid Argument")

    if not file.exists() or file.suffix != ".dat":
        raise FileNotFoundError("The provided file does not match it's intended usage")

    HEADER = b"::"
    FOOTER = b";;"
    METADATA_SIZE = 7

    time = int(file.stem.strip("abcdefghijklmnopqrstuvwxyz. _"))
    with file.open("rb") as f:
        decoder = ZstdDecompressor()
        decompressed_data = b"".join(decoder.read_to_iter(f.read()))

    cursor = 0
    data_len = len(decompressed_data)

    while cursor < data_len:
        # Find the next header start
        header_idx = decompressed_data.find(HEADER, cursor)
        if header_idx == -1:
            break  # EOF

        # Find the corresponding footer
        footer_idx = decompressed_data.find(FOOTER, header_idx)
        if footer_idx == -1:
            print(f"Warning: Found header at {header_idx} but no matching footer.")
            break

        # Extract Metadata (2 + 4 + 1)
        try:
            # throw away the header
            _, timestamp_us, message_type = struct.unpack(
                "<2sIB", decompressed_data[header_idx : header_idx + METADATA_SIZE]
            )
        except struct.error as e:
            print(f"Error parsing metadata block at index {header_idx}: {e}")
            break

        payload_start = header_idx + METADATA_SIZE
        payload_end = footer_idx
        pc_payload = decompressed_data[payload_start:payload_end]

        point_cloud_np = np.frombuffer(pc_payload, dtype=np.int16) / 1000
        point_cloud_np = point_cloud_np.reshape((-1, 3))

        # skip footer
        cursor = footer_idx + 2
        yield time, timestamp_us, message_type, point_cloud_np


if __name__ == "__main__":
    try:
        for time, duration, mes_type, points in import_data_from_desktop(
            NEW_EXAMPLE_FILE
        ):
            print(duration, mes_type, points.shape)

    except KeyboardInterrupt:
        pass
    # udp_radar_reciever(DEFAULT_PORT)
