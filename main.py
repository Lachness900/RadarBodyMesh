import socket
import struct
import numpy as np
import zstandard as zstd

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
def import_data_from_desktop(file_name: str):
    time = int(file_name.strip('abcdefghijklmnopqrstuvwxyz. _'))
    with zstd.open(file_name) as f:
        data = f.read(7)
        duration = struct.unpack('<I',data[2:6])*1000
        message_type = data[6]

        data = f.read()
    length = len(data)
    print(length)
    points = np.frombuffer(data[:length - (length % 6)], dtype=np.uint16).reshape(-1, 3) 
    return time, duration[0], message_type, points

if __name__ == "__main__":
    time, duration, mes_type, points = import_data_from_desktop(NEW_EXAMPLE_FILE)
    print(time, duration, mes_type, points)
    # udp_radar_reciever(DEFAULT_PORT)
