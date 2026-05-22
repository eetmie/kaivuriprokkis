import random
import time

from modules.udp_socket import UDPSocket


HOST = "127.0.0.1"
PORT = 5005
VALUE_COUNT = 3


def main():
    server = UDPSocket(local_id=1, max_age_seconds=1.0, data_format="f")
    server.setup(
        HOST,
        PORT,
        num_inputs=0,
        num_outputs=VALUE_COUNT,
        is_server=True,
    )

    if not server.handshake(timeout=30.0):
        server.close()
        return

    try:
        while True:
            values = [random.uniform(0.0, 100.0) for _ in range(VALUE_COUNT)]
            server.send(values)
            print(f"sent: {values}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("server stopped")
    finally:
        server.close()


if __name__ == "__main__":
    main()
