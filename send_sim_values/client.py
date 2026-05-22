import time

from modules.udp_socket import UDPSocket


HOST = "127.0.0.1"
PORT = 5005
VALUE_COUNT = 3


def main():
    client = UDPSocket(local_id=2, max_age_seconds=2.0, data_format="f")
    client.setup(
        HOST,
        PORT,
        num_inputs=VALUE_COUNT,
        num_outputs=0,
        is_server=False,
    )

    if not client.handshake(timeout=30.0):
        client.close()
        return

    client.start_receiving()

    try:
        while True:
            values = client.get_latest()
            if values is None:
                print("waiting for data...")
            else:
                rounded = [round(value, 2) for value in values]
                print(f"received: {rounded}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("client stopped")
    finally:
        client.close()


if __name__ == "__main__":
    main()
