#!/usr/bin/env python3
"""
Comprehensive test for usb_serial_reader.py

Tests all functions using simulation mode, plus hardware tests if device connected.
Speed test capped at 200 Hz.
"""

import sys
import time
import traceback
from typing import Callable, List
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, '/home/joel/masi/modules')


class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", duration_ms: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms


def run_test(name: str, test_fn: Callable, expect_exception: type = None) -> TestResult:
    """Run a single test."""
    start = time.perf_counter()
    try:
        test_fn()
        duration = (time.perf_counter() - start) * 1000
        if expect_exception:
            return TestResult(name, False, f"Expected {expect_exception.__name__} but none raised", duration)
        return TestResult(name, True, "OK", duration)
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        if expect_exception and isinstance(e, expect_exception):
            return TestResult(name, True, f"Correctly raised {expect_exception.__name__}", duration)
        elif expect_exception:
            return TestResult(name, False, f"Expected {expect_exception.__name__}, got {type(e).__name__}: {e}", duration)
        else:
            return TestResult(name, False, f"{type(e).__name__}: {e}", duration)


def print_result(result: TestResult):
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {result.name} ({result.duration_ms:.1f}ms)")
    if result.message and result.message != "OK":
        for line in result.message.split('\n')[:2]:
            print(f"         {line}")


def main():
    print("=" * 70)
    print("USB Serial Reader Comprehensive Test")
    print("=" * 70)
    print()

    results: List[TestResult] = []

    # =========================================================================
    # PART 1: Import and Basic Tests
    # =========================================================================
    print("=== PART 1: Import and Simulation Mode ===\n")

    def test_import():
        from usb_serial_reader import USBSerialReader
        assert USBSerialReader is not None

    results.append(run_test("Import USBSerialReader", test_import))
    print_result(results[-1])

    from usb_serial_reader import USBSerialReader
    import serial

    # --- Simulation mode tests ---
    reader_sim = None

    def test_simulation_init():
        nonlocal reader_sim
        reader_sim = USBSerialReader(simulation_mode=True, log_level="ERROR")
        assert reader_sim is not None
        assert reader_sim.simulation_mode == True
        assert reader_sim.ser is None  # No serial in simulation

    results.append(run_test("Simulation mode initialization", test_simulation_init))
    print_result(results[-1])

    def test_simulation_read_imus():
        data = reader_sim.read_imus()
        assert data is not None
        assert len(data) == 3  # 3 simulated IMUs
        for imu in data:
            assert len(imu) == 7  # w, x, y, z, gx, gy, gz

    results.append(run_test("Simulation read_imus()", test_simulation_read_imus))
    print_result(results[-1])

    def test_simulation_quaternion_valid():
        data = reader_sim.read_imus()
        for imu in data:
            w, x, y, z = imu[0:4]
            # Quaternion should be normalized (magnitude ~1)
            mag = (w*w + x*x + y*y + z*z) ** 0.5
            assert 0.99 < mag < 1.01, f"Quaternion magnitude {mag} not normalized"

    results.append(run_test("Simulation quaternion normalization", test_simulation_quaternion_valid))
    print_result(results[-1])

    def test_generate_simulation_data():
        # Test multiple calls advance simulation time
        reader_sim.sim_time = 0.0
        data1 = reader_sim.generate_simulation_data()
        data2 = reader_sim.generate_simulation_data()
        # Quaternions should differ as sim_time advances
        assert data1[0][0] != data2[0][0] or data1[0][2] != data2[0][2]

    results.append(run_test("generate_simulation_data() time advance", test_generate_simulation_data))
    print_result(results[-1])

    def test_set_log_level():
        reader_sim.set_log_level("DEBUG")
        assert reader_sim.logger.level == 10
        reader_sim.set_log_level("ERROR")
        assert reader_sim.logger.level == 40

    results.append(run_test("set_log_level()", test_set_log_level))
    print_result(results[-1])

    def test_close_simulation():
        reader_sim.close()  # Should not crash even with no serial

    results.append(run_test("close() in simulation mode", test_close_simulation))
    print_result(results[-1])

    # =========================================================================
    # PART 2: CSV Parsing Tests
    # =========================================================================
    print("\n=== PART 2: CSV Line Parsing ===\n")

    reader_parse = USBSerialReader(simulation_mode=True, log_level="ERROR")

    def test_parse_single_imu_no_timestamp():
        # 7 values: w, x, y, z, gx, gy, gz
        line = "1.0,0.0,0.0,0.0,0.1,0.2,0.3"
        data = reader_parse.parse_imu_line(line)
        assert data is not None
        assert len(data) == 1
        assert data[0] == [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3]
        assert reader_parse.last_timestamp_us is None

    results.append(run_test("Parse single IMU (no timestamp)", test_parse_single_imu_no_timestamp))
    print_result(results[-1])

    def test_parse_single_imu_with_timestamp():
        # timestamp + 7 values
        line = "123456,1.0,0.0,0.0,0.0,0.1,0.2,0.3"
        data = reader_parse.parse_imu_line(line)
        assert data is not None
        assert len(data) == 1
        assert reader_parse.last_timestamp_us == 123456

    results.append(run_test("Parse single IMU (with timestamp)", test_parse_single_imu_with_timestamp))
    print_result(results[-1])

    def test_parse_three_imus_with_timestamp():
        # timestamp + 3 * 7 values = 22 values
        line = "999999," + ",".join(["1.0,0.0,0.0,0.0,0.0,0.0,0.0"] * 3)
        data = reader_parse.parse_imu_line(line)
        assert data is not None
        assert len(data) == 3
        assert reader_parse.last_timestamp_us == 999999

    results.append(run_test("Parse 3 IMUs (with timestamp)", test_parse_three_imus_with_timestamp))
    print_result(results[-1])

    def test_parse_empty_line():
        data = reader_parse.parse_imu_line("")
        assert data is None

    results.append(run_test("Parse empty line", test_parse_empty_line))
    print_result(results[-1])

    def test_parse_invalid_values():
        line = "1.0,abc,0.0,0.0,0.0,0.0,0.0"
        data = reader_parse.parse_imu_line(line)
        assert data is None

    results.append(run_test("Parse invalid values (abc)", test_parse_invalid_values))
    print_result(results[-1])

    def test_parse_wrong_count():
        # 5 values - not divisible by 7
        line = "1.0,0.0,0.0,0.0,0.0"
        data = reader_parse.parse_imu_line(line)
        assert data is None

    results.append(run_test("Parse wrong value count", test_parse_wrong_count))
    print_result(results[-1])

    def test_parse_whitespace():
        line = "  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  "
        data = reader_parse.parse_imu_line(line)
        assert data is not None
        assert len(data) == 1

    results.append(run_test("Parse with whitespace", test_parse_whitespace))
    print_result(results[-1])

    def test_parse_negative_values():
        line = "-0.707,0.707,0.0,0.0,-1.5,2.3,-0.8"
        data = reader_parse.parse_imu_line(line)
        assert data is not None
        assert data[0][0] == -0.707
        assert data[0][4] == -1.5

    results.append(run_test("Parse negative values", test_parse_negative_values))
    print_result(results[-1])

    def test_parse_scientific_notation():
        line = "1.0e0,1e-5,0.0,0.0,1.5e2,-2.3e-1,0.0"
        data = reader_parse.parse_imu_line(line)
        assert data is not None
        assert abs(data[0][1] - 1e-5) < 1e-10

    results.append(run_test("Parse scientific notation", test_parse_scientific_notation))
    print_result(results[-1])

    # =========================================================================
    # PART 3: Binary Parsing Tests
    # =========================================================================
    print("\n=== PART 3: Binary Frame Parsing ===\n")

    import struct

    def build_binary_frame(sensor_count: int, timestamp: int, imu_data: list) -> bytes:
        """Build a valid binary frame for testing."""
        # Frame: 0xAA 0x55 | ver(1) | count(1) | ts_us(4) | payload | checksum(2)
        header = b'\xAA\x55'
        version = 1
        payload = struct.pack('<I', timestamp)  # timestamp
        for imu in imu_data:
            payload += struct.pack('<fffffff', *imu)
        checksum_data = bytes([version, sensor_count]) + payload
        checksum = sum(checksum_data) & 0xFFFF
        frame = header + bytes([version, sensor_count]) + payload + struct.pack('<H', checksum)
        return frame

    def test_binary_frame_valid():
        # Create a fresh reader with mock serial to bypass is_open check
        reader_bin = USBSerialReader(simulation_mode=True, data_format="BIN", log_level="ERROR")
        reader_bin.ser = MagicMock()
        reader_bin.ser.is_open = True
        reader_bin.ser.in_waiting = 0

        imu_data = [[1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3]]
        frame = build_binary_frame(1, 123456, imu_data)
        reader_bin._bin_buf = bytearray(frame)
        result = reader_bin._read_imus_binary()
        assert result is not None, "Expected valid result"
        assert len(result) == 1
        assert abs(result[0][0] - 1.0) < 0.001
        assert reader_bin.last_timestamp_us == 123456

    results.append(run_test("Binary frame valid (1 IMU)", test_binary_frame_valid))
    print_result(results[-1])

    def test_binary_frame_3_imus():
        reader_bin = USBSerialReader(simulation_mode=True, data_format="BIN", log_level="ERROR")
        reader_bin.ser = MagicMock()
        reader_bin.ser.is_open = True
        reader_bin.ser.in_waiting = 0

        imu_data = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.707, 0.707, 0.0, 0.0, 1.0, 2.0, 3.0],
            [0.5, 0.5, 0.5, 0.5, -1.0, -2.0, -3.0],
        ]
        frame = build_binary_frame(3, 999999, imu_data)
        reader_bin._bin_buf = bytearray(frame)
        result = reader_bin._read_imus_binary()
        assert result is not None, "Expected valid result"
        assert len(result) == 3
        assert abs(result[1][0] - 0.707) < 0.001

    results.append(run_test("Binary frame valid (3 IMUs)", test_binary_frame_3_imus))
    print_result(results[-1])

    def test_binary_frame_bad_sync():
        reader_bin = USBSerialReader(simulation_mode=True, data_format="BIN", log_level="ERROR")
        reader_bin.ser = MagicMock()
        reader_bin.ser.is_open = True
        reader_bin.ser.in_waiting = 0
        reader_bin._bin_buf = bytearray(b'\x00\x00\x00\x00\x00')
        result = reader_bin._read_imus_binary()
        assert result is None

    results.append(run_test("Binary frame bad sync bytes", test_binary_frame_bad_sync))
    print_result(results[-1])

    def test_binary_frame_bad_checksum():
        reader_bin = USBSerialReader(simulation_mode=True, data_format="BIN", log_level="ERROR")
        reader_bin.ser = MagicMock()
        reader_bin.ser.is_open = True
        reader_bin.ser.in_waiting = 0
        imu_data = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        frame = bytearray(build_binary_frame(1, 123456, imu_data))
        frame[-1] ^= 0xFF  # Corrupt checksum
        reader_bin._bin_buf = frame
        result = reader_bin._read_imus_binary()
        assert result is None

    results.append(run_test("Binary frame bad checksum", test_binary_frame_bad_checksum))
    print_result(results[-1])

    def test_binary_frame_incomplete():
        reader_bin = USBSerialReader(simulation_mode=True, data_format="BIN", log_level="ERROR")
        reader_bin.ser = MagicMock()
        reader_bin.ser.is_open = True
        reader_bin.ser.in_waiting = 0
        imu_data = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        frame = build_binary_frame(1, 123456, imu_data)
        reader_bin._bin_buf = bytearray(frame[:10])  # Truncate
        result = reader_bin._read_imus_binary()
        assert result is None

    results.append(run_test("Binary frame incomplete", test_binary_frame_incomplete))
    print_result(results[-1])

    def test_binary_frame_invalid_version():
        reader_bin = USBSerialReader(simulation_mode=True, data_format="BIN", log_level="ERROR")
        reader_bin.ser = MagicMock()
        reader_bin.ser.is_open = True
        reader_bin.ser.in_waiting = 0
        imu_data = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        frame = bytearray(build_binary_frame(1, 123456, imu_data))
        frame[2] = 99  # Invalid version
        reader_bin._bin_buf = frame
        result = reader_bin._read_imus_binary()
        assert result is None

    results.append(run_test("Binary frame invalid version", test_binary_frame_invalid_version))
    print_result(results[-1])

    # =========================================================================
    # PART 4: Speed Test (Simulation Mode, capped at 200 Hz)
    # =========================================================================
    print("\n=== PART 4: Speed Test (Simulation, max 200 Hz) ===\n")

    def test_simulation_speed():
        reader_speed = USBSerialReader(simulation_mode=True, log_level="ERROR")
        iterations = 0
        start = time.perf_counter()
        duration = 2.0

        while time.perf_counter() - start < duration:
            data = reader_speed.read_imus()
            if data is not None:
                iterations += 1
            # Cap at 200 Hz
            time.sleep(1.0 / 200.0)

        elapsed = time.perf_counter() - start
        rate = iterations / elapsed
        assert rate > 150, f"Rate {rate:.1f} Hz too slow"
        assert rate < 210, f"Rate {rate:.1f} Hz exceeds 200 Hz cap"
        print(f"         Achieved {rate:.1f} Hz over {elapsed:.1f}s")

    results.append(run_test("Simulation speed (200 Hz cap)", test_simulation_speed))
    print_result(results[-1])

    def test_simulation_jitter():
        reader_jitter = USBSerialReader(simulation_mode=True, log_level="ERROR")
        intervals = []
        last_time = time.perf_counter()
        target_interval = 1.0 / 200.0

        for _ in range(400):  # 2 seconds at 200 Hz
            data = reader_jitter.read_imus()
            now = time.perf_counter()
            intervals.append(now - last_time)
            last_time = now
            # Sleep to target 200 Hz
            sleep_time = target_interval - (time.perf_counter() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)

        import statistics
        intervals_ms = [i * 1000 for i in intervals[10:]]  # Skip warmup
        mean = statistics.mean(intervals_ms)
        std = statistics.stdev(intervals_ms)
        print(f"         Mean: {mean:.2f}ms, StdDev: {std:.3f}ms")
        assert std < 2.0, f"Jitter too high: {std:.3f}ms"

    results.append(run_test("Simulation jitter at 200 Hz", test_simulation_jitter))
    print_result(results[-1])

    # =========================================================================
    # PART 5: Robustness / Edge Cases
    # =========================================================================
    print("\n=== PART 5: Robustness / Edge Cases ===\n")

    def test_close_twice():
        reader = USBSerialReader(simulation_mode=True, log_level="ERROR")
        reader.close()
        reader.close()  # Should not crash

    results.append(run_test("close() called twice", test_close_twice))
    print_result(results[-1])

    def test_read_after_close():
        reader = USBSerialReader(simulation_mode=True, log_level="ERROR")
        reader.close()
        data = reader.read_imus()  # Simulation mode still works
        assert data is not None

    results.append(run_test("read_imus() after close (simulation)", test_read_after_close))
    print_result(results[-1])

    def test_invalid_log_level():
        reader = USBSerialReader(simulation_mode=True, log_level="INVALID")
        # Should default to INFO
        assert reader.logger.level == 20  # INFO

    results.append(run_test("Invalid log_level defaults to INFO", test_invalid_log_level))
    print_result(results[-1])

    def test_data_format_case_insensitive():
        reader1 = USBSerialReader(simulation_mode=True, data_format="csv", log_level="ERROR")
        assert reader1.data_format == "CSV"
        reader2 = USBSerialReader(simulation_mode=True, data_format="bin", log_level="ERROR")
        assert reader2.data_format == "BIN"

    results.append(run_test("data_format case insensitive", test_data_format_case_insensitive))
    print_result(results[-1])

    def test_data_format_none():
        reader = USBSerialReader(simulation_mode=True, data_format=None, log_level="ERROR")
        assert reader.data_format == "CSV"  # Default

    results.append(run_test("data_format=None defaults to CSV", test_data_format_none))
    print_result(results[-1])

    # --- Mock serial tests for methods requiring hardware ---
    def test_send_handshake_no_serial():
        reader = USBSerialReader(simulation_mode=True, log_level="ERROR")
        result = reader.send_handshake_config()
        assert result == False  # No serial connection

    results.append(run_test("send_handshake_config() no serial", test_send_handshake_no_serial))
    print_result(results[-1])

    def test_send_zero_no_serial():
        reader = USBSerialReader(simulation_mode=True, log_level="ERROR")
        result = reader.send_zero()
        assert result == False

    results.append(run_test("send_zero() no serial", test_send_zero_no_serial))
    print_result(results[-1])

    def test_find_pico_port_no_ports():
        reader = USBSerialReader(simulation_mode=True, log_level="ERROR")
        with patch('serial.tools.list_ports.comports', return_value=[]):
            port = reader.find_pico_port()
            assert port is None

    results.append(run_test("find_pico_port() no ports", test_find_pico_port_no_ports))
    print_result(results[-1])

    def test_find_pico_port_generic():
        reader = USBSerialReader(simulation_mode=True, log_level="ERROR")
        mock_port = Mock()
        mock_port.device = "/dev/ttyUSB0"
        mock_port.description = "Generic USB"
        mock_port.hwid = "USB VID:PID=1234:5678"
        with patch('serial.tools.list_ports.comports', return_value=[mock_port]):
            port = reader.find_pico_port()
            assert port == "/dev/ttyUSB0"  # Falls back to first port

    results.append(run_test("find_pico_port() generic fallback", test_find_pico_port_generic))
    print_result(results[-1])

    def test_find_pico_port_xiao():
        reader = USBSerialReader(simulation_mode=True, log_level="ERROR")
        mock_port = Mock()
        mock_port.device = "/dev/ttyACM0"
        mock_port.description = "XIAO RP2040"
        mock_port.hwid = "USB VID:PID=2E8A:000A"
        with patch('serial.tools.list_ports.comports', return_value=[mock_port]):
            port = reader.find_pico_port()
            assert port == "/dev/ttyACM0"

    results.append(run_test("find_pico_port() XIAO detected", test_find_pico_port_xiao))
    print_result(results[-1])

    # --- Mocked serial connection tests ---
    def test_connect_with_mock():
        with patch('serial.Serial') as mock_serial:
            mock_ser = MagicMock()
            mock_serial.return_value = mock_ser
            reader = USBSerialReader(simulation_mode=False, port="/dev/ttyTEST", log_level="ERROR")
            assert reader.ser is not None
            mock_serial.assert_called_once()

    results.append(run_test("connect() with mock serial", test_connect_with_mock))
    print_result(results[-1])

    def test_send_handshake_mock():
        with patch('serial.Serial') as mock_serial:
            mock_ser = MagicMock()
            mock_ser.is_open = True
            mock_serial.return_value = mock_ser
            reader = USBSerialReader(simulation_mode=False, port="/dev/ttyTEST", log_level="ERROR")
            result = reader.send_handshake_config(sensor_count=3, sample_rate=120)
            assert result == True
            mock_ser.write.assert_called_once()

    results.append(run_test("send_handshake_config() with mock", test_send_handshake_mock))
    print_result(results[-1])

    def test_send_zero_mock():
        with patch('serial.Serial') as mock_serial:
            mock_ser = MagicMock()
            mock_ser.is_open = True
            mock_serial.return_value = mock_ser
            reader = USBSerialReader(simulation_mode=False, port="/dev/ttyTEST", log_level="ERROR")
            result = reader.send_zero()
            assert result == True
            mock_ser.write.assert_called_with(b"CMD=ZERO\n")

    results.append(run_test("send_zero() with mock", test_send_zero_mock))
    print_result(results[-1])

    def test_handshake_qmode_values():
        with patch('serial.Serial') as mock_serial:
            mock_ser = MagicMock()
            mock_ser.is_open = True
            mock_serial.return_value = mock_ser
            reader = USBSerialReader(simulation_mode=False, port="/dev/ttyTEST", log_level="ERROR")

            # Test valid qmodes
            for qmode in ["FULL", "PITCH", "full", "pitch", "0", "1"]:
                result = reader.send_handshake_config(qmode=qmode)
                assert result == True

            # Invalid qmode should default to FULL
            result = reader.send_handshake_config(qmode="INVALID")
            assert result == True
            # Just verify write was called multiple times
            assert mock_ser.write.call_count >= 7

    results.append(run_test("send_handshake_config() qmode values", test_handshake_qmode_values))
    print_result(results[-1])

    # =========================================================================
    # PART 6: Hardware Test (if device available)
    # =========================================================================
    print("\n=== PART 6: Hardware Test (if available) ===\n")

    hardware_available = False
    reader_hw = None

    def test_hardware_connect():
        nonlocal hardware_available, reader_hw
        try:
            reader_hw = USBSerialReader(simulation_mode=False, log_level="WARNING", timeout=0.5)
            hardware_available = True
            assert reader_hw.ser is not None
            assert reader_hw.ser.is_open
        except Exception as e:
            raise Exception(f"No hardware: {e}")

    results.append(run_test("Hardware connection", test_hardware_connect))
    print_result(results[-1])

    if hardware_available and reader_hw:
        def test_hardware_handshake():
            result = reader_hw.send_handshake_config(sensor_count=3, sample_rate=120, qmode="FULL")
            assert result == True

        results.append(run_test("Hardware send_handshake_config()", test_hardware_handshake))
        print_result(results[-1])

        def test_hardware_read():
            time.sleep(0.3)  # Wait for data stream to start
            # Try multiple reads with longer timeout
            data = None
            for _ in range(100):
                data = reader_hw.read_imus()
                if data is not None:
                    break
                time.sleep(0.01)
            if data is None:
                raise Exception("No IMU data received after 100 attempts")
            assert len(data) >= 1, f"Expected at least 1 IMU, got {len(data)}"
            assert len(data[0]) == 7, f"Expected 7 values per IMU, got {len(data[0])}"

        results.append(run_test("Hardware read_imus()", test_hardware_read))
        print_result(results[-1])

        def test_hardware_speed():
            iterations = 0
            start = time.perf_counter()
            duration = 2.0

            while time.perf_counter() - start < duration:
                data = reader_hw.read_imus()
                if data is not None:
                    iterations += 1
                time.sleep(0.001)  # Don't spin too hard

            elapsed = time.perf_counter() - start
            rate = iterations / elapsed
            print(f"         Hardware rate: {rate:.1f} Hz")
            assert rate > 50, f"Rate {rate:.1f} Hz seems low"

        results.append(run_test("Hardware speed test", test_hardware_speed))
        print_result(results[-1])

        def test_hardware_send_zero():
            result = reader_hw.send_zero()
            assert result == True

        results.append(run_test("Hardware send_zero()", test_hardware_send_zero))
        print_result(results[-1])

        reader_hw.close()
    else:
        print("  [SKIP] Hardware not available - skipping hardware tests")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration_ms for r in results)

    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Total time: {total_time:.1f}ms")
    print()

    if failed > 0:
        print("FAILED TESTS:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message[:70]}")
        return 1
    else:
        print("All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
