import socket
import struct

DEFAULT_ADDR = ""
DEFAULT_PORT = 5005

RADAR_FRAME_DIRECTORY = "radar_frames"
RADAR_EXAMPLE_FILE = r"raw.bin"
CLEANED_EXAMPLE_FILE = r"train.dat"
CHUNK_SIZE = 8
CHUNK_LIMIT = 20


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

# For manually reading a raw data file into the server
def import_raw_data_from_desktop():
    with open(RADAR_EXAMPLE_FILE, "rb") as f:
        data = f.read(CHUNK_SIZE*CHUNK_LIMIT)
        for i in range(CHUNK_LIMIT):
            print(struct.unpack('<4h', data[CHUNK_SIZE*i:CHUNK_SIZE*(i+1)]))


# For manually reading a data file into the server
def import_data_from_desktop():
    with open(CLEANED_EXAMPLE_FILE, "rb") as f:
        data = f.read(CHUNK_SIZE*CHUNK_LIMIT)
        for i in range(CHUNK_LIMIT):
            print(struct.unpack('<4h', data[CHUNK_SIZE*i:CHUNK_SIZE*(i+1)]))

if __name__ == "__main__":
    import_data_from_desktop()
    #udp_radar_reciever(DEFAULT_PORT)
