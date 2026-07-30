import os
import socket
import struct
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from kaivuri_bringup.project_paths import add_project_import_path, resolve_project_root


_HANDSHAKE_SIZE = 69
_INPUT_FORMAT = "<8bH"
_PACKET_FORMAT = "<I8bH"


class MotionPlatformUdpReceiver:
    """Receive MotionPlatform/main.py packets: 8 int8 axes + uint16 button mask."""

    def __init__(self, *, local_id: int, max_age_seconds: float) -> None:
        self.local_id = int(local_id)
        self.max_age_seconds = float(max_age_seconds)
        self.socket: Optional[socket.socket] = None
        self.remote_addr = None
        self.latest_data = None
        self.latest_timestamp = 0.0
        self.data_lock = threading.Lock()
        self.recv_thread = None
        self.running = False

    def setup(self, host: str, port: int) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(1.0)
        self.socket.bind((host, port))
        print(f"UDP Server listening on {host}:{port}")

    def handshake(self, timeout: float) -> bool:
        if self.socket is None:
            return False

        max_age_ms = int(self.max_age_seconds * 1000)
        nominal_rate_c_hz = 0
        outputs_fmt = b""
        inputs_fmt = _INPUT_FORMAT.encode("ascii")
        our_info = struct.pack(
            "<BHH32s32s",
            self.local_id,
            max_age_ms,
            nominal_rate_c_hz,
            outputs_fmt.ljust(32, b"\x00"),
            inputs_fmt.ljust(32, b"\x00"),
        )

        print("Waiting for handshake from client...")
        self.socket.settimeout(timeout)
        try:
            data, addr = self.socket.recvfrom(_HANDSHAKE_SIZE)
            self.remote_addr = addr
            self.socket.sendto(our_info, self.remote_addr)
        except socket.timeout:
            print("Handshake timeout!")
            return False

        if len(data) != _HANDSHAKE_SIZE:
            print(f"Handshake packet wrong size: expected {_HANDSHAKE_SIZE}, got {len(data)}")
            return False

        remote_id, remote_max_age_ms, remote_rate_c_hz, raw_out, raw_in = struct.unpack("<BHH32s32s", data)
        remote_out_fmt = raw_out.rstrip(b"\x00").decode("ascii")
        remote_in_fmt = raw_in.rstrip(b"\x00").decode("ascii")
        if remote_in_fmt != "":
            print(f"Mismatch: MotionPlatform expects inputs '{remote_in_fmt}', we send ''")
            return False
        if remote_out_fmt != _INPUT_FORMAT:
            print(f"Mismatch: MotionPlatform sends outputs '{remote_out_fmt}', we expect '{_INPUT_FORMAT}'")
            return False

        remote_rate_hz = remote_rate_c_hz / 100.0 if remote_rate_c_hz > 0 else 0.0
        rate_msg = f", rate: {remote_rate_hz:.2f}Hz" if remote_rate_hz > 0.0 else ""
        print(
            f"Handshake OK with MotionPlatform device ID {remote_id} "
            f"(max_age: {remote_max_age_ms}ms, in: '{_INPUT_FORMAT}', out: ''{rate_msg})"
        )
        self.socket.settimeout(1.0)
        return True

    def start_receiving(self) -> None:
        if self.recv_thread and self.recv_thread.is_alive():
            return
        self.running = True
        self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.recv_thread.start()
        print("Started receive thread")

    def get_latest(self):
        with self.data_lock:
            if self.latest_data is None:
                return None
            if time.monotonic() - self.latest_timestamp > self.max_age_seconds:
                return None
            return list(self.latest_data)

    def _receive_loop(self) -> None:
        if self.socket is None:
            return
        expected_size = struct.calcsize(_PACKET_FORMAT)
        while self.running:
            try:
                data, addr = self.socket.recvfrom(expected_size)
                if self.remote_addr is None:
                    self.remote_addr = addr
                if len(data) != expected_size:
                    print(f"Wrong packet size: expected {expected_size}, got {len(data)}")
                    continue
                unpacked = struct.unpack(_PACKET_FORMAT, data)
                values = list(unpacked[1:])
                arrival_time = time.monotonic()
                with self.data_lock:
                    self.latest_data = values
                    self.latest_timestamp = arrival_time
            except socket.timeout:
                continue
            except Exception as exc:
                if self.running:
                    print(f"Receive error: {exc}")

    def close(self) -> None:
        self.running = False
        if self.recv_thread:
            self.recv_thread.join(timeout=2.0)
        if self.socket:
            self.socket.close()
            self.socket = None


class UdpJoystickValuesNode(Node):
    """Receive real joystick values over UDP and publish them as ROS 2 floats."""

    def __init__(self) -> None:
        super().__init__("udp_joystick_values_node")
        self.declare_parameter("project_root", os.environ.get("KAIVURI_PROJECT_ROOT", "~/kaivuriprokkis"))
        self.declare_parameter("host", "10.214.33.132")
        self.declare_parameter("port", 8080)
        self.declare_parameter("joystick_publish_rate", 100.0)
        self.declare_parameter("max_age_seconds", 0.5)
        self.declare_parameter("topic", "joystick_values")
        self.declare_parameter("local_id", 2)

        project_root = resolve_project_root(str(self.get_parameter("project_root").value))
        add_project_import_path(project_root)

        self._udp: Optional[MotionPlatformUdpReceiver] = None
        self._publisher = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("topic").value),
            10,
        )

        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        rate_hz = max(1.0, float(self.get_parameter("joystick_publish_rate").value))
        max_age_seconds = float(self.get_parameter("max_age_seconds").value)
        local_id = int(self.get_parameter("local_id").value)

        self._udp = MotionPlatformUdpReceiver(local_id=local_id, max_age_seconds=max_age_seconds)
        self._udp.setup(host, port)

        if not self._udp.handshake(timeout=30.0):
            self.get_logger().error(f"UDP handshake failed with {host}:{port}")
            self._udp.close()
            self._udp = None
            return

        self._udp.start_receiving()
        self.create_timer(1.0 / rate_hz, self._publish_latest)
        self.get_logger().info(
            f"Publishing MotionPlatform joystick values from {host}:{port}; data=[axis0..axis7, button_mask]"
        )

    def _publish_latest(self) -> None:
        if self._udp is None:
            return

        values = self._udp.get_latest()
        if values is None:
            self.get_logger().warning("No fresh UDP joystick values available", throttle_duration_sec=2.0)
            return

        msg = Float32MultiArray()
        msg.data = [float(value) for value in values]
        self._publisher.publish(msg)

    def destroy_node(self) -> bool:
        if self._udp is not None:
            self._udp.close()
            self._udp = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UdpJoystickValuesNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
