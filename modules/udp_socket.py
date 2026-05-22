import hashlib
import hmac
import socket
import struct
import threading
import time
from typing import Optional, List

HMAC_DIGEST_SIZE = 32  # SHA-256
_HANDSHAKE_SIZE = 69   # BHH32s32s
_ENDIAN_CHARS = frozenset('@=<>!')


def _parse_fmt(fmt: str) -> tuple:
    """Return (endian, data_part) from a struct format string. Default endian is '<'."""
    if fmt and fmt[0] in _ENDIAN_CHARS:
        return fmt[0], fmt[1:]
    return '<', fmt


class UDPSocket:
    """
    UDP socket with heartbeat safety mechanism.
    - Adds timestamp to each packet
    - Returns None if data is too old
    - Configurable timeout for safety

    inputs/outputs are struct format strings, e.g. '10b', '<8b10?', '>12f'.
    Endianness prefix is optional (defaults to '<'). Use '' for zero elements.
    """

    def __init__(self, local_id=0, max_age_seconds=0.5, hmac_key: Optional[str] = None,
                 nominal_rate_hz: Optional[float] = None):
        self.socket = None
        self.remote_addr = None
        self.local_id = local_id
        self.inputs_fmt = ''
        self.outputs_fmt = ''
        self._num_inputs = 0
        self._num_outputs = 0
        self.max_age_seconds = max_age_seconds
        self.nominal_rate_hz = float(nominal_rate_hz) if nominal_rate_hz is not None else None
        self.remote_nominal_rate_hz: Optional[float] = None

        # For receiving data
        self.latest_data = None
        self.latest_timestamp = 0.0
        self.data_lock = threading.Lock()
        self.recv_thread = None
        self.running = False

        # Statistics
        self.packets_received = 0
        self.packets_expired = 0
        self.last_packet_time = 0.0

        # Pre-computed format strings (filled after setup)
        self.send_format = None
        self.recv_format = None

        # HMAC authentication (optional)
        self._hmac_key = hmac_key.encode('utf-8') if isinstance(hmac_key, str) else hmac_key
        self.packets_rejected = 0

    def set_hmac(self, key: str):
        """Set HMAC key for packet authentication. Must be called on both sides with the same key."""
        self._hmac_key = key.encode('utf-8') if isinstance(key, str) else key

    def setup(self, host, port, inputs: str = '', outputs: str = '', is_server=False):
        """Set up UDP socket.

        inputs:  struct format for data received (e.g. '10b', '<8b10?'). '' = receive-nothing.
        outputs: struct format for data sent     (e.g. '10b', '<12f').  '' = send-nothing.
        Endianness prefix is optional — defaults to '<' if omitted.
        """
        for name, fmt in (('inputs', inputs), ('outputs', outputs)):
            if len(fmt) > 32:
                raise ValueError(f"{name} format exceeds 32-char handshake limit: '{fmt}'")
            try:
                endian, data = _parse_fmt(fmt)
                struct.calcsize(endian + data)
            except struct.error as e:
                raise ValueError(f"Invalid {name} format '{fmt}': {e}") from e

        in_endian, in_data = _parse_fmt(inputs)
        out_endian, out_data = _parse_fmt(outputs)

        # Normalize to always include endianness prefix (stored and transmitted)
        self.inputs_fmt = (in_endian + in_data) if inputs else ''
        self.outputs_fmt = (out_endian + out_data) if outputs else ''

        self._num_inputs = len(struct.unpack(self.inputs_fmt, bytes(struct.calcsize(self.inputs_fmt)))) if inputs else 0
        self._num_outputs = len(struct.unpack(self.outputs_fmt, bytes(struct.calcsize(self.outputs_fmt)))) if outputs else 0

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(1.0)

        if is_server:
            self.socket.bind((host, port))
            print(f"UDP Server listening on {host}:{port}")
        else:
            self.remote_addr = (host, port)
            print(f"UDP Client ready to send to {host}:{port}")

        # Format: endian + timestamp (uint32) + data elements
        self.recv_format = (in_endian + 'I' + in_data) if inputs else (in_endian + 'I')
        self.send_format = (out_endian + 'I' + out_data) if outputs else (out_endian + 'I')

        return True

    def set_nominal_rate_hz(self, nominal_rate_hz: Optional[float]):
        """Set local nominal application send/update rate advertised in handshake."""
        self.nominal_rate_hz = float(nominal_rate_hz) if nominal_rate_hz is not None else None

    def get_handshake_info(self) -> dict:
        """Return local/remote handshake metadata."""
        return {
            'local_id': self.local_id,
            'local_nominal_rate_hz': self.nominal_rate_hz,
            'remote_addr': self.remote_addr,
            'remote_nominal_rate_hz': self.remote_nominal_rate_hz,
            'max_age_seconds': self.max_age_seconds,
            'inputs_fmt': self.inputs_fmt,
            'outputs_fmt': self.outputs_fmt,
        }

    # ================================
    # Conversion helpers (int8 <-> float)
    # ================================
    @staticmethod
    def ints_to_floats(values: List[int]) -> List[float]:
        """Convert signed int8 [-128..127] to floats in [-1.0..1.0] with clamping."""
        floats: List[float] = []
        for v in values:
            try:
                iv = int(v)
            except Exception:
                iv = 0
            iv = max(-128, min(127, iv))
            floats.append(max(-1.0, min(1.0, iv / 127.0)))
        return floats

    @staticmethod
    def floats_to_ints(values: List[float]) -> List[int]:
        """Convert floats in [-1.0..1.0] to signed int8 [-128..127]."""
        ints: List[int] = []
        for v in values:
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            fv = max(-1.0, min(1.0, fv))
            iv = int(round(fv * 127.0))
            ints.append(max(-128, min(127, iv)))
        return ints

    def handshake(self, timeout=5.0):
        """Perform handshake exchanging format strings and rate info.

        Packet layout (69 bytes): [local_id:B][max_age_ms:H][nominal_rate_cHz:H][out_fmt:32s][in_fmt:32s]
        """
        max_age_ms = int(self.max_age_seconds * 1000)
        nominal_rate_c_hz = max(0, min(65535, int(round(self.nominal_rate_hz * 100.0)))) if self.nominal_rate_hz else 0
        out_fmt_bytes = self.outputs_fmt.encode('ascii').ljust(32, b'\x00')
        in_fmt_bytes = self.inputs_fmt.encode('ascii').ljust(32, b'\x00')
        our_info = struct.pack('<BHH32s32s', self.local_id, max_age_ms, nominal_rate_c_hz, out_fmt_bytes, in_fmt_bytes)

        if self.remote_addr:  # Client mode
            self.socket.sendto(our_info, self.remote_addr)
            self.socket.settimeout(timeout)
            try:
                data, addr = self.socket.recvfrom(_HANDSHAKE_SIZE)
                self.remote_addr = addr
            except socket.timeout:
                print("Handshake timeout!")
                return False
        else:  # Server mode
            print("Waiting for handshake...")
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

        remote_id, remote_max_age_ms, remote_rate_c_hz, raw_out, raw_in = struct.unpack('<BHH32s32s', data)
        remote_out_fmt = raw_out.rstrip(b'\x00').decode('ascii')
        remote_in_fmt = raw_in.rstrip(b'\x00').decode('ascii')
        self.remote_nominal_rate_hz = remote_rate_c_hz / 100.0 if remote_rate_c_hz > 0 else None

        if remote_in_fmt != self.outputs_fmt:
            print(f"Mismatch: They expect inputs '{remote_in_fmt}', we send '{self.outputs_fmt}'")
            return False
        if remote_out_fmt != self.inputs_fmt:
            print(f"Mismatch: They send outputs '{remote_out_fmt}', we expect '{self.inputs_fmt}'")
            return False

        rate_msg = f", rate: {self.remote_nominal_rate_hz:.2f}Hz" if self.remote_nominal_rate_hz is not None else ""
        print(f"Handshake OK with device ID {remote_id} (max_age: {remote_max_age_ms}ms, "
              f"in: '{self.inputs_fmt}', out: '{self.outputs_fmt}'{rate_msg})")
        self.socket.settimeout(1.0)
        return True

    def send(self, values):
        """Send values with timestamp. Values must match the outputs format."""
        if not self.remote_addr:
            print("No remote address set!")
            return False

        if len(values) != self._num_outputs:
            print(f"Expected {self._num_outputs} values, got {len(values)}")
            return False

        timestamp_ms = int(time.time() * 1000) & 0xFFFFFFFF
        data = struct.pack(self.send_format, timestamp_ms, *values)
        if self._hmac_key:
            data += hmac.new(self._hmac_key, data, hashlib.sha256).digest()
        self.socket.sendto(data, self.remote_addr)
        return True

    def get_latest(self) -> Optional[List]:
        """Get latest received data if fresh, else None."""
        with self.data_lock:
            if self.latest_data is None:
                return None

            age = time.monotonic() - self.latest_timestamp
            if age > self.max_age_seconds:
                self.packets_expired += 1
                return None

            return self.latest_data.copy()

    def get_connection_stats(self) -> dict:
        """Get connection statistics for monitoring."""
        with self.data_lock:
            current_time = time.monotonic()
            age = current_time - self.latest_timestamp if self.latest_timestamp > 0 else float('inf')
            time_since_last = current_time - self.last_packet_time if self.last_packet_time > 0 else float('inf')

            return {
                'packets_received': self.packets_received,
                'packets_expired': self.packets_expired,
                'packets_rejected': self.packets_rejected,
                'data_age_seconds': age,
                'time_since_last_packet': time_since_last,
                'is_connected': age < self.max_age_seconds,
                'has_data': self.latest_data is not None
            }

    def start_receiving(self):
        """Start the receive thread."""
        if not self.recv_thread or not self.recv_thread.is_alive():
            self.running = True
            self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.recv_thread.start()
            print("Started receive thread")

    def stop_receiving(self):
        """Stop the receive thread."""
        self.running = False
        if self.recv_thread:
            self.recv_thread.join(timeout=2.0)
            print("Stopped receive thread")

    def _receive_loop(self):
        """Background thread to continuously receive data with timestamps."""
        payload_size = struct.calcsize(self.recv_format)
        expected_size = payload_size + (HMAC_DIGEST_SIZE if self._hmac_key else 0)

        while self.running:
            try:
                data, addr = self.socket.recvfrom(expected_size)

                if not self.remote_addr:
                    self.remote_addr = addr

                if len(data) == expected_size:
                    if self._hmac_key:
                        payload = data[:payload_size]
                        received_mac = data[payload_size:]
                        expected_mac = hmac.new(self._hmac_key, payload, hashlib.sha256).digest()
                        if not hmac.compare_digest(received_mac, expected_mac):
                            self.packets_rejected += 1
                            continue
                        data = payload

                    unpacked = struct.unpack(self.recv_format, data)
                    values = list(unpacked[1:])

                    # Use arrival time — simpler and more reliable than handling 32-bit ms wraparound
                    arrival_time = time.monotonic()

                    with self.data_lock:
                        self.latest_data = values
                        self.latest_timestamp = arrival_time
                        self.packets_received += 1
                        self.last_packet_time = arrival_time

                else:
                    print(f"Wrong packet size: expected {expected_size}, got {len(data)}")

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")

    def close(self):
        """Clean shutdown."""
        self.stop_receiving()
        if self.socket:
            self.socket.close()
            print("Socket closed")
