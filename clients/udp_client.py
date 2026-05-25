"""
UDP client for excavator GUI — handles connect, send/receive threads, Hz tracking.

Extracted from client_gui.py to isolate networking from UI logic.
"""

import threading
import time
from typing import Callable, Optional

from modules.udp_socket import UDPSocket
from modules.control_protocol import (
    COMMAND_PACKET_SIZE,
    TELEMETRY_PACKET_SIZE,
    ControlCommand,
    decode_telemetry_message,
    encode_command_message,
    RobotTelemetry,
)


class ExcavatorUDPClient:
    """Manages UDP connection, send/receive threads, and Hz tracking."""

    HZ_UPDATE_INTERVAL = 0.1  # seconds

    def __init__(self, local_id: int = 3, max_age_seconds: float = 0.5):
        self._local_id = local_id
        self._max_age = max_age_seconds
        self._udp: Optional[UDPSocket] = None
        self._connected = False
        self._streaming = False
        self._send_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._thread_running = False

        # Hz tracking
        self._send_hz_actual = 0.0
        self._send_count_window = 0
        self._send_time_window_start = 0.0

        # Callbacks
        self._on_telemetry: Optional[Callable[[RobotTelemetry], None]] = None
        self._on_hz_update: Optional[Callable[[float], None]] = None
        self._get_command: Optional[Callable[[], ControlCommand]] = None

        # Config
        self._send_rate_hz = 20.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    def get_send_hz(self) -> float:
        return self._send_hz_actual

    def connect(self, host: str, port: int, rate_hz: float) -> bool:
        """Connect to the server. Returns True on success."""
        if self._connected:
            return True
        self._send_rate_hz = rate_hz
        try:
            self._udp = UDPSocket(
                local_id=self._local_id,
                max_age_seconds=self._max_age,
                nominal_rate_hz=rate_hz,
            )
            self._udp.setup(host, port,
                            inputs=f'{TELEMETRY_PACKET_SIZE}b',
                            outputs=f'{COMMAND_PACKET_SIZE}b',
                            is_server=False)
            if not self._udp.handshake(timeout=10.0):
                self._udp.close()
                self._udp = None
                return False
        except Exception:
            self._udp = None
            return False
        self._connected = True
        return True

    def disconnect(self):
        """Stop streaming and close connection."""
        self.stop_streaming()
        if self._udp:
            self._udp.close()
        self._udp = None
        self._connected = False

    def start_streaming(
        self,
        get_command: Callable[[], ControlCommand],
        on_telemetry: Callable[[RobotTelemetry], None],
        on_hz_update: Optional[Callable[[float], None]] = None,
    ):
        """Start send + receive threads.

        Args:
            get_command: Called each send tick to get the current command to send.
            on_telemetry: Called when a new telemetry packet arrives.
            on_hz_update: Called periodically with the actual send Hz.
        """
        if not self._connected or not self._udp or self._streaming:
            return
        self._get_command = get_command
        self._on_telemetry = on_telemetry
        self._on_hz_update = on_hz_update
        self._streaming = True
        self._thread_running = True
        self._send_time_window_start = time.perf_counter()
        self._send_count_window = 0
        self._udp.start_receiving()

        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._send_thread.start()
        self._recv_thread.start()

    def stop_streaming(self):
        """Stop send/receive threads."""
        if not self._streaming:
            return
        self._streaming = False
        self._thread_running = False
        if self._udp:
            self._udp.stop_receiving()
        if self._send_thread and self._send_thread.is_alive():
            self._send_thread.join(timeout=1.0)
        self._send_hz_actual = 0.0

    def _send_loop(self):
        rate = max(1.0, self._send_rate_hz)
        period = 1.0 / rate
        next_t = time.perf_counter()
        self._send_time_window_start = time.perf_counter()
        self._send_count_window = 0

        while self._thread_running:
            try:
                if self._udp and self._get_command:
                    command = self._get_command()
                    payload = encode_command_message(command)
                    self._udp.send(payload)
                    self._send_count_window += 1

                    now = time.perf_counter()
                    if now - self._send_time_window_start >= self.HZ_UPDATE_INTERVAL:
                        self._send_hz_actual = self._send_count_window / (now - self._send_time_window_start)
                        self._send_time_window_start = now
                        self._send_count_window = 0
                        if self._on_hz_update:
                            self._on_hz_update(self._send_hz_actual)

                next_t += period
                remaining = next_t - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_t = time.perf_counter()
            except Exception as e:
                print(f"Send error: {e}")
                time.sleep(0.1)

    def _receive_loop(self):
        while self._thread_running:
            try:
                if not self._udp:
                    time.sleep(0.05)
                    continue
                data = self._udp.get_latest()
                if data and len(data) == TELEMETRY_PACKET_SIZE:
                    telemetry = decode_telemetry_message(data)
                    if self._on_telemetry:
                        self._on_telemetry(telemetry)
                time.sleep(0.01)
            except Exception as e:
                print(f"Receive error: {e}")
                time.sleep(0.1)
