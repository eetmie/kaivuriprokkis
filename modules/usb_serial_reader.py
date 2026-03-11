import serial
import time
import logging
import serial.tools.list_ports
import struct
import math


class USBSerialReader:
    """Read IMU quaternion data from Pico over USB serial (binary protocol)."""

    MAX_SENSORS = 4
    FLOATS_PER_SENSOR = 7  # w, x, y, z, gx, gy, gz

    # Binary protocol frame versions
    FRAME_VERSION_DATA = 1
    FRAME_VERSION_CTRL = 2
    FRAME_VERSION_DESC = 3

    # Control frame message types (version=2)
    MSG_TYPE_CFG_OK = 0x01
    MSG_TYPE_CFG_WAIT = 0x02
    MSG_TYPE_ZERO_ACK = 0x03
    MSG_TYPE_ERROR = 0x04

    # Tuned AHRS defaults (extensively tested)
    DEFAULT_SAMPLE_RATE = 200
    DEFAULT_GYRO_DPS = 250
    DEFAULT_GAIN = 5.0
    DEFAULT_ACCEL_REJ = 20.0
    DEFAULT_RECOVERY_S = 0.5
    DEFAULT_OFFSET_S = 0.5

    def __init__(self, baud_rate=115200, timeout=1.0, simulation_mode=False,
                 log_level: str = "INFO", port: str = None, debug: bool = False,
                 verify_checksum: bool = True, heartbeat_timeout: float = 3.0):
        self.logger = logging.getLogger(f"{__name__}.USBSerialReader")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self.port = port
        self._requested_port = port  # remember original for reconnect
        self.simulation_mode = simulation_mode
        self.debug = debug
        self.verify_checksum = verify_checksum
        self.heartbeat_timeout = heartbeat_timeout
        self.sim_time = 0.0
        self.num_sensors = 0  # discovered at runtime from first frame
        self.last_timestamp_us = None
        self._imu_descriptors = []  # list of {"bus": int, "addr": int}
        self._target_sps = 0       # target sample rate reported by firmware
        self._last_data_time = time.time()
        self._connect_time = time.time()
        self._bin_buf = bytearray()
        self._text_buf = bytearray()
        self._last_config = None
        self._checksum_failures = 0
        self._header_failures = 0
        self._reconnect_count = 0

        # SPS (samples per second) tracking
        self._frame_count = 0
        self._sps_time = time.time()
        self._sps = 0.0

        if not simulation_mode:
            self.connect()
        else:
            self.num_sensors = 3
            self._target_sps = self.DEFAULT_SAMPLE_RATE
            self._imu_descriptors = [
                {"bus": 0, "addr": 0x6A, "label": "I2C0:0x6A"},
                {"bus": 0, "addr": 0x6B, "label": "I2C0:0x6B"},
                {"bus": 1, "addr": 0x6A, "label": "I2C1:0x6A"},
            ]
            self.logger.info("Simulation mode: generating synthetic IMU data")

    # ------------------------------------------------------------------
    # Status API
    # ------------------------------------------------------------------

    def status(self):
        """Return current connection status as a dict.

        Keys:
            connected (bool): Whether the serial port is open.
            port (str|None): Current serial port name.
            num_sensors (int): Number of IMUs detected in last frame.
            imu_descriptors (list[dict]): Per-IMU info from firmware.
                Each dict: {"bus": int, "addr": int, "label": str}.
                Empty until descriptor frame is received.
            target_sps (int): Target sample rate reported by firmware.
            sps (float): Actual received frames per second.
            last_timestamp_us (int|None): Last sensor timestamp in microseconds.
            checksum_failures (int): Cumulative checksum mismatch count.
            header_failures (int): Cumulative invalid header count.
            reconnect_count (int): Number of automatic reconnections.
            uptime_s (float): Seconds since initial connection.
            simulation_mode (bool): Whether running in simulation mode.
        """
        if self.simulation_mode:
            connected = True
        else:
            connected = self.ser is not None and self.ser.is_open
        return {
            "connected": connected,
            "port": self.port,
            "num_sensors": self.num_sensors,
            "imu_descriptors": list(self._imu_descriptors),
            "target_sps": self._target_sps,
            "sps": self._sps,
            "last_timestamp_us": self.last_timestamp_us,
            "checksum_failures": self._checksum_failures,
            "header_failures": self._header_failures,
            "reconnect_count": self._reconnect_count,
            "uptime_s": time.time() - self._connect_time,
            "simulation_mode": self.simulation_mode,
        }

    @property
    def connected(self):
        """True if the serial port is open (or in simulation mode)."""
        if self.simulation_mode:
            return True
        return self.ser is not None and self.ser.is_open

    @property
    def sps(self):
        """Current frames-per-second rate."""
        return self._sps

    @property
    def target_sps(self):
        """Target sample rate reported by firmware (0 if not yet received)."""
        return self._target_sps

    @property
    def imu_descriptors(self):
        """List of IMU descriptors from firmware.

        Each entry is a dict with:
            bus (int): I2C bus number (0 or 1)
            addr (int): I2C address (e.g. 0x6A, 0x6B)
            label (str): Human-readable label (e.g. "I2C0:0x6A")
        """
        return list(self._imu_descriptors)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def find_pico_port(self):
        """Try to find a likely XIAO RP2040 / Pico port automatically."""
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.logger.error("No serial ports found.")
            return None

        keywords = ("XIAO", "RP2040", "Pico", "USB Serial", "CDC", "ACM")
        vidpid_markers = ("VID:PID=2E8A:", "VID:PID=2886:")

        for port in ports:
            desc = port.description or ""
            hwid = port.hwid or ""
            if any(k in desc for k in keywords) or any(m in hwid for m in vidpid_markers):
                self.logger.info(f"Found device on port: {port.device}")
                return port.device

        self.logger.warning(f"No descriptor match; defaulting to {ports[0].device}")
        return ports[0].device

    def connect(self):
        """Connect to the serial port."""
        if not self.port:
            self.port = self.find_pico_port()
        if not self.port:
            raise serial.SerialException("No serial port found")

        self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self._bin_buf.clear()
        self._last_data_time = time.time()
        self._connect_time = time.time()
        self.logger.info(f"Connected to {self.port} at {self.baud_rate} baud")
        time.sleep(0.1)

    def reconnect(self):
        """Close and re-establish connection, then re-send config."""
        self._reconnect_count += 1
        self.logger.info(f"Reconnecting (attempt #{self._reconnect_count})...")

        # Close existing connection
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

        # Re-detect port (may have changed after Pico reset)
        self.port = self._requested_port
        time.sleep(0.5)

        try:
            self.connect()
        except serial.SerialException as e:
            self.logger.warning(f"Reconnect failed: {e}")
            return False

        # Re-send config and wait for handshake
        if self._last_config:
            self.ser.write(self._last_config.encode("utf-8"))
            if self.wait_for_cfg_ok(timeout_s=3.0, resend_s=0.5):
                self.logger.info("Reconnected and handshake OK")
                return True
            else:
                self.logger.warning("Reconnected but no CFG_OK — Pico may be streaming already")
                return True  # still usable, just no handshake
        return True

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def send_config(
        self,
        sample_rate=None,
        gyro_dps=None,
        gain=None,
        accel_rejection=None,
        recovery_s=None,
        offset_s=None,
    ):
        """Send configuration to Pico. Uses tuned defaults for omitted values."""
        if not self.ser or not self.ser.is_open:
            self.logger.warning("Not connected - cannot send config")
            return False

        sr = sample_rate if sample_rate is not None else self.DEFAULT_SAMPLE_RATE
        gyro = gyro_dps if gyro_dps is not None else self.DEFAULT_GYRO_DPS
        g = gain if gain is not None else self.DEFAULT_GAIN
        ar = accel_rejection if accel_rejection is not None else self.DEFAULT_ACCEL_REJ
        rs = recovery_s if recovery_s is not None else self.DEFAULT_RECOVERY_S
        os_ = offset_s if offset_s is not None else self.DEFAULT_OFFSET_S

        config = (
            f"SR={sr}|"
            f"GYRO_DPS={gyro}|"
            f"GAIN={g}|"
            f"ACC_REJ={ar}|"
            f"RECOV_S={rs}|"
            f"OFFSET_S={os_}|\n"
        )
        self._last_config = config
        self.ser.write(config.encode("utf-8"))
        self.logger.info(f"Sent config: {config.strip()}")
        return True

    def wait_for_cfg_ok(self, timeout_s=5.0, resend_s=0.5):
        """Wait for binary CFG_OK from Pico, resending config periodically."""
        if not self.ser or not self.ser.is_open:
            self.logger.warning("Not connected - cannot wait for CFG_OK")
            return False

        start = time.time()
        last_send = 0.0

        while True:
            now = time.time()
            if timeout_s is not None and (now - start) >= timeout_s:
                return False

            if resend_s is not None and (now - last_send) >= resend_s:
                # Resend last config if available
                if self._last_config:
                    self.ser.write(self._last_config.encode("utf-8"))
                last_send = now

            if self.ser.in_waiting > 0:
                chunk = self.ser.read(self.ser.in_waiting)
                if self.debug and chunk:
                    self.logger.info("RX(wait): %r", chunk)
                self._bin_buf.extend(chunk)

                # Scan buffer for CFG_OK and descriptor frames
                found_cfg_ok = False
                idx = 0
                while idx <= len(self._bin_buf) - 6:
                    if self._bin_buf[idx] == 0xAA and self._bin_buf[idx + 1] == 0x55:
                        version = self._bin_buf[idx + 2]
                        if version == self.FRAME_VERSION_CTRL:
                            msg_type = self._bin_buf[idx + 3]
                            expected_cs = struct.unpack_from("<H", self._bin_buf, idx + 4)[0]
                            calc_cs = (version + msg_type) & 0xFFFF
                            if calc_cs == expected_cs and msg_type == self.MSG_TYPE_CFG_OK:
                                found_cfg_ok = True
                                self._bin_buf = bytearray(self._bin_buf[idx + 6:])
                                break
                            idx += 6
                        elif version == self.FRAME_VERSION_DESC:
                            # Parse descriptor if complete
                            dc = self._bin_buf[idx + 3]
                            df_len = 4 + 2 + dc * 2 + 2
                            if idx + df_len <= len(self._bin_buf) and 1 <= dc <= self.MAX_SENSORS:
                                cs_off = idx + df_len - 2
                                exp_cs = struct.unpack_from("<H", self._bin_buf, cs_off)[0]
                                cal_cs = sum(self._bin_buf[idx + 2:cs_off]) & 0xFFFF
                                if cal_cs == exp_cs:
                                    self._target_sps = struct.unpack_from("<H", self._bin_buf, idx + 4)[0]
                                    self._imu_descriptors = []
                                    for di in range(dc):
                                        bus = self._bin_buf[idx + 6 + di * 2]
                                        addr = self._bin_buf[idx + 6 + di * 2 + 1]
                                        self._imu_descriptors.append({
                                            "bus": bus, "addr": addr,
                                            "label": f"I2C{bus}:0x{addr:02X}",
                                        })
                                    self.num_sensors = dc
                                    self.logger.info(
                                        "IMU descriptor: %d sensor(s) @ %d Hz — %s",
                                        dc, self._target_sps,
                                        ", ".join(d["label"] for d in self._imu_descriptors),
                                    )
                                idx += df_len
                            else:
                                idx += 1
                        else:
                            idx += 1
                    else:
                        idx += 1

                if found_cfg_ok:
                    self._last_data_time = time.time()
                    return True

                # Keep buffer bounded
                if len(self._bin_buf) > 4096:
                    del self._bin_buf[:-1024]

            time.sleep(0.01)

    def send_zero(self):
        """Request re-zero (sets current pitch as zero reference)."""
        if not self.ser or not self.ser.is_open:
            self.logger.warning("Not connected - cannot send ZERO")
            return False

        self.ser.write(b"CMD=ZERO\n")
        self.logger.info("Sent ZERO command")
        return True

    # ------------------------------------------------------------------
    # Data reading
    # ------------------------------------------------------------------

    def generate_simulation_data(self):
        """Generate synthetic IMU data for testing."""
        self.sim_time += 0.0083  # ~120Hz
        sim_count = self.num_sensors or 3

        # Simulate rotation from 0 to 45 degrees over 5 seconds
        angle = min(45.0, (self.sim_time / 5.0) * 45.0) * math.pi / 180.0
        angular_velocity = (45.0 * math.pi / 180.0) / 5.0 if self.sim_time < 5.0 else 0.0

        data = []
        for i in range(sim_count):
            # Each IMU at slightly different angle
            offset = i * 30.0 * math.pi / 180.0
            a = angle + offset
            w = math.cos(a / 2)
            y = math.sin(a / 2)
            data.append([w, 0.0, y, 0.0, 0.0, angular_velocity * 180.0 / math.pi, 0.0])

        self.last_timestamp_us = int(self.sim_time * 1e6)
        return data

    def _check_heartbeat(self):
        """Check if data stream is alive. Returns True if OK, False if timed out."""
        if self.heartbeat_timeout <= 0:
            return True
        elapsed = time.time() - self._last_data_time
        if elapsed > self.heartbeat_timeout:
            self.logger.warning(f"No data for {elapsed:.1f}s — connection may be lost")
            return False
        return True

    def _update_sps(self):
        """Update samples-per-second counter."""
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._sps_time
        if elapsed >= 1.0:
            self._sps = self._frame_count / elapsed
            self._frame_count = 0
            self._sps_time = now

    def _read_binary_frame(self):
        """Parse binary IMU frame from serial buffer."""
        if not self.ser or not self.ser.is_open:
            return None

        # Read available data into buffer
        try:
            if self.ser.in_waiting > 0:
                self._bin_buf.extend(self.ser.read(self.ser.in_waiting))
        except (serial.SerialException, OSError):
            return None

        # Frame: 0xAA 0x55 | ver(1) | count(1) | ts_us(4) | payload | checksum(2)
        while True:
            if len(self._bin_buf) < 4:
                return None

            # Find sync bytes
            sync = self._bin_buf.find(b"\xAA\x55")
            if sync < 0:
                # Keep a trailing 0xAA in case sync splits across reads
                if self._bin_buf.endswith(b"\xAA"):
                    self._bin_buf[:] = b"\xAA"
                else:
                    self._bin_buf.clear()
                return None
            if sync > 0:
                del self._bin_buf[:sync]
                if len(self._bin_buf) < 4:
                    return None

            # Parse header
            if len(self._bin_buf) < 8:
                return None

            version = self._bin_buf[2]

            # Handle control frames (version=2) - skip them silently
            if version == self.FRAME_VERSION_CTRL:
                if len(self._bin_buf) < 6:
                    return None
                del self._bin_buf[:6]
                continue

            # Handle IMU descriptor frame (version=3)
            if version == self.FRAME_VERSION_DESC:
                if len(self._bin_buf) < 8:
                    return None
                desc_count = self._bin_buf[3]
                if desc_count < 1 or desc_count > self.MAX_SENSORS:
                    del self._bin_buf[:1]
                    continue
                # Frame: sync(2) + ver(1) + count(1) + sps(2) + N*2 + checksum(2)
                desc_frame_len = 4 + 2 + desc_count * 2 + 2
                if len(self._bin_buf) < desc_frame_len:
                    return None
                # Verify checksum
                cs_off = desc_frame_len - 2
                expected_cs = struct.unpack_from("<H", self._bin_buf, cs_off)[0]
                calc_cs = sum(self._bin_buf[2:cs_off]) & 0xFFFF
                if calc_cs == expected_cs:
                    self._target_sps = struct.unpack_from("<H", self._bin_buf, 4)[0]
                    self._imu_descriptors = []
                    for i in range(desc_count):
                        bus = self._bin_buf[6 + i * 2]
                        addr = self._bin_buf[6 + i * 2 + 1]
                        self._imu_descriptors.append({
                            "bus": bus,
                            "addr": addr,
                            "label": f"I2C{bus}:0x{addr:02X}",
                        })
                    self.num_sensors = desc_count
                    self.logger.info(
                        "IMU descriptor: %d sensor(s) @ %d Hz — %s",
                        desc_count, self._target_sps,
                        ", ".join(d["label"] for d in self._imu_descriptors),
                    )
                del self._bin_buf[:desc_frame_len]
                continue

            sensor_count = self._bin_buf[3]

            if version != self.FRAME_VERSION_DATA or sensor_count < 1 or sensor_count > self.MAX_SENSORS:
                # Drop one byte and rescan for sync
                self._header_failures += 1
                del self._bin_buf[0:1]
                continue

            # Calculate frame length
            payload_len = 4 + sensor_count * self.FLOATS_PER_SENSOR * 4
            frame_len = 2 + 1 + 1 + payload_len + 2

            if len(self._bin_buf) < frame_len:
                return None

            # Verify checksum
            checksum_offset = 2 + 1 + 1 + payload_len
            expected = struct.unpack_from("<H", self._bin_buf, checksum_offset)[0]
            calc = sum(self._bin_buf[2:checksum_offset]) & 0xFFFF

            if calc != expected:
                self._checksum_failures += 1
                if self.debug and (self._checksum_failures % 50 == 0):
                    self.logger.warning(
                        "Checksum mismatch (count=%d): calc=0x%04X expected=0x%04X",
                        self._checksum_failures,
                        calc,
                        expected,
                    )
                if not self.verify_checksum:
                    break
                # Drop one byte and rescan for sync
                del self._bin_buf[0:1]
                continue

            # Parse timestamp
            ts = struct.unpack_from("<I", self._bin_buf, 4)[0]
            self.last_timestamp_us = int(ts)

            # Parse sensor data
            imu_data = []
            offset = 8
            for _ in range(sensor_count):
                vals = struct.unpack_from("<fffffff", self._bin_buf, offset)
                imu_data.append(list(vals))
                offset += 28

            del self._bin_buf[:frame_len]
            self._last_data_time = time.time()
            self._update_sps()
            if self.num_sensors != sensor_count:
                self.num_sensors = sensor_count
                self.logger.info(f"Receiving data from {sensor_count} IMU(s)")
            return imu_data

    def read_imus(self, auto_reconnect=True):
        """
        Read IMU data (returns latest frame, discarding any backlog).

        Args:
            auto_reconnect: If True, automatically reconnect on timeout/error.

        Returns:
            List of N arrays, each containing [w, x, y, z, gx, gy, gz]
            (quaternion + gyro in dps), or None if no data available.
        """
        if self.simulation_mode:
            return self.generate_simulation_data()

        if self.debug and self.ser and self.ser.in_waiting > 0:
            peek = self.ser.read(self.ser.in_waiting)
            if peek:
                self.logger.info("RX(data): %r", peek)
            # Put bytes back into binary buffer for parsing
            self._bin_buf.extend(peek)

        # Drain buffer and return only the LATEST frame to avoid latency
        latest_data = None
        while True:
            data = self._read_binary_frame()
            if data is None:
                break
            latest_data = data

        # Heartbeat check: reconnect if no data for too long
        if latest_data is None and auto_reconnect and not self._check_heartbeat():
            self.reconnect()

        return latest_data

    def quaternion_to_pitch(self, w, x, y, z):
        """Extract pitch angle (Y-axis rotation) from quaternion in degrees."""
        # Y-twist: 2*atan2(y, w)
        pitch = 2.0 * math.atan2(y, w) * 180.0 / math.pi
        if pitch <= -180.0:
            pitch += 360.0
        elif pitch > 180.0:
            pitch -= 360.0
        return pitch

    def iter_stream(self, sleep_s=0.001):
        """Generator that yields parsed IMU data as it arrives."""
        try:
            while True:
                data = self.read_imus()
                if data is not None:
                    yield data
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            self.close()

    def close(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.logger.info("Connection closed")

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read 1-4 IMU quaternions over USB serial")
    parser.add_argument("--port", help="Serial port (auto-detect if not specified)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--sr", type=int, default=USBSerialReader.DEFAULT_SAMPLE_RATE,
                        help="Sample rate Hz")
    parser.add_argument("--gyro-dps", type=int, default=USBSerialReader.DEFAULT_GYRO_DPS,
                        help="Gyro range (125/250/500/1000/2000)")
    parser.add_argument("--sim", action="store_true", help="Simulation mode")
    parser.add_argument("--debug", action="store_true", help="Print raw serial data")
    parser.add_argument("--no-checksum", action="store_true",
                        help="Skip checksum verification (debug only)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip waiting for CFG_OK (useful if Pico is already streaming)")
    parser.add_argument("--heartbeat", type=float, default=3.0,
                        help="Reconnect if no data for this many seconds (0=disable)")
    args = parser.parse_args()

    reader = USBSerialReader(
        baud_rate=args.baud,
        port=args.port,
        simulation_mode=args.sim,
        debug=args.debug,
        verify_checksum=not args.no_checksum,
        heartbeat_timeout=args.heartbeat,
    )

    if not args.sim:
        reader.send_config(
            sample_rate=args.sr,
            gyro_dps=args.gyro_dps,
        )
        if not args.no_wait:
            if not reader.wait_for_cfg_ok(timeout_s=5.0, resend_s=0.5):
                reader.logger.warning("CFG_OK not received; continuing anyway")

    # Wait briefly for descriptor frame to arrive with first data
    print(f"Connecting... (Ctrl+C to stop)")
    desc_shown = False
    t0 = time.time()
    n = 0

    try:
        while True:
            data = reader.read_imus()
            if data is not None:
                # Show descriptor info once when first received
                if not desc_shown and reader.imu_descriptors:
                    desc_shown = True
                    descs = reader.imu_descriptors
                    print(f"\n--- IMU Descriptor ---")
                    print(f"  Sensors:    {reader.num_sensors}")
                    print(f"  Target SPS: {reader.target_sps} Hz")
                    for i, d in enumerate(descs):
                        print(f"  [{i}] {d['label']}  (bus={d['bus']}, addr=0x{d['addr']:02X})")
                    print(f"----------------------\n")

                n += 1
                if time.time() - t0 >= 1.0:
                    st = reader.status()
                    # Build per-IMU pitch readout with labels
                    descs = reader.imu_descriptors
                    pitches = []
                    for i, imu in enumerate(data):
                        w, x, y, z = imu[0:4]
                        pitch = reader.quaternion_to_pitch(w, x, y, z)
                        label = descs[i]["label"] if i < len(descs) else f"IMU{i}"
                        pitches.append(f"{label}:{pitch:+6.1f}")
                    sps_str = f"{st['sps']:.0f}"
                    if st['target_sps']:
                        sps_str += f"/{st['target_sps']}"
                    ts = reader.last_timestamp_us
                    print(f"SPS={sps_str:>7s} Hz  ts={ts:10d}  {' | '.join(pitches)}")
                    n = 0
                    t0 = time.time()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
