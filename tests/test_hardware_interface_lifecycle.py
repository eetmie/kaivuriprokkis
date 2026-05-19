import logging
import importlib
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

sys.modules.pop("modules.hardware_interface", None)
_HW_MODULE = importlib.import_module("modules.hardware_interface")

HardwareInterface = _HW_MODULE.HardwareInterface
ReadyState = _HW_MODULE.ReadyState
_ImuSnapshot = _HW_MODULE._ImuSnapshot


class _FakeThread:
    def __init__(self):
        self.join_calls = []

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class _FakeUsbReader:
    def __init__(self, order_log):
        self.order_log = order_log

    def stop_background_reader(self):
        self.order_log.append("stop")

    def close(self):
        self.order_log.append("close")


class _FakeTracker:
    def record_sample(self):
        pass


class _FakeAdc:
    def __init__(self, hw):
        self.hw = hw
        self.mid_scan_snapshot = None

    def read_channel(self, board, channel):
        if channel == 0:
            self.mid_scan_snapshot = self.hw.get_latest_adc_readings()
            return 1.25
        self.hw._stop_event.set()
        return 2.5


class _FakeSmbus:
    def __init__(self, calls):
        self.calls = calls

    def write_byte_data(self, addr, reg, val):
        self.calls.append(("reg", addr, reg, val))

    def write_i2c_block_data(self, addr, reg, data):
        self.calls.append(("block", addr, reg, len(data)))

    def close(self):
        self.calls.append(("close",))


class HardwareInterfaceLifecycleTests(unittest.TestCase):
    def _make_hw_stub(self):
        hw = object.__new__(HardwareInterface)
        hw.logger = logging.getLogger("test.hardware_interface_lifecycle")
        hw._enable_adc = True
        hw.pwm_controller = None
        hw._stop_event = threading.Event()
        hw._imu_lock = threading.Lock()
        hw._adc_lock = threading.Lock()
        hw._imu_state = ReadyState.READY
        hw._adc_state = ReadyState.READY
        hw._base_imu_index = 0
        hw.latest_imu_data = ["imu"]
        hw.latest_imu_by_role = {"base": "quat"}
        hw.latest_imu_raw_quat = ["raw"]
        hw.latest_imu_corrected_quat = ["corr"]
        hw.latest_base_imu_quat = "base_quat"
        hw.latest_base_imu_gyro = [0.0, 0.0, 0.0]
        hw.latest_imu_gyro = [[0.0, 0.0, 0.0]]
        hw._imu_last_device_ts = 123
        hw._imu_snapshot = _ImuSnapshot(
            imu_data=["imu"],
            imu_by_role={"base": "quat"},
            base_imu_quat=[1.0, 0.0, 0.0, 0.0],
            base_imu_gyro=[0.0, 0.0, 0.0],
            imu_gyro=[[0.0, 0.0, 0.0]],
            raw_quat=[[1.0, 0.0, 0.0, 0.0]],
            corrected_quat=[[1.0, 0.0, 0.0, 0.0]],
            device_ts=456,
        )
        hw._latest_adc_readings = {"a": 1.0}
        hw._latest_adc_timestamp = 99.0
        hw.imu_thread = _FakeThread()
        hw.adc_thread = _FakeThread()
        return hw

    def test_shutdown_stops_usb_reader_before_close_and_clears_snapshots(self):
        hw = self._make_hw_stub()
        order_log = []
        hw.usb_reader = _FakeUsbReader(order_log)

        HardwareInterface.shutdown(hw)

        self.assertTrue(hw._stop_event.is_set())
        self.assertEqual(order_log, ["stop", "close"])
        self.assertIsNone(hw._imu_snapshot)
        self.assertIsNone(hw.latest_imu_data)
        self.assertEqual(hw._latest_adc_readings, {})
        self.assertIsNone(hw._latest_adc_timestamp)
        self.assertEqual(hw._imu_state, ReadyState.PENDING)
        self.assertEqual(hw._adc_state, ReadyState.PENDING)

    def test_read_base_imu_returns_none_when_not_ready_even_if_snapshot_exists(self):
        hw = self._make_hw_stub()
        hw._imu_state = ReadyState.PENDING

        self.assertIsNone(HardwareInterface.read_base_imu(hw))

    def test_adc_reader_publishes_whole_snapshot_at_once(self):
        hw = object.__new__(HardwareInterface)
        hw.logger = logging.getLogger("test.hardware_interface_adc")
        hw._enable_adc = True
        hw._stop_event = threading.Event()
        hw._adc_lock = threading.Lock()
        hw._adc_state = ReadyState.READY
        hw._adc_expected_hz = 1000.0
        hw._adc_channel_plan = [
            {"board": "b0", "channel": 0, "name": "first"},
            {"board": "b0", "channel": 1, "name": "second"},
        ]
        hw._latest_adc_readings = {"first": 10.0, "second": 20.0}
        hw._latest_adc_timestamp = 1.0
        hw._adc_tracker = _FakeTracker()
        hw._adc_rt_priority = 0
        hw._rt_lock_memory = False
        hw._adc_cpu_core = None
        hw.adc = _FakeAdc(hw)

        HardwareInterface._adc_reader_thread(hw)

        self.assertEqual(hw.adc.mid_scan_snapshot, {"first": 10.0, "second": 20.0})
        self.assertEqual(hw._latest_adc_readings, {"first": 1.25, "second": 2.5})
        self.assertIsNotNone(hw._latest_adc_timestamp)

    def test_read_base_imu_returns_snapshot_gyro_without_debug_flag(self):
        hw = self._make_hw_stub()
        hw._debug_telemetry_enabled = False

        payload = HardwareInterface.read_base_imu(hw)

        self.assertIsNotNone(payload)
        self.assertIn('gyro', payload)
        self.assertEqual(list(payload['gyro']), [0.0, 0.0, 0.0])

    def test_try_read_imu_gyro_returns_latest_gyro_without_debug_flag(self):
        hw = self._make_hw_stub()
        hw._debug_telemetry_enabled = False

        payload = HardwareInterface.try_read_imu_gyro(hw)

        self.assertIsNotNone(payload)
        self.assertEqual(payload['device_timestamp_us'], 123)
        self.assertEqual(payload['gyro'], [[0.0, 0.0, 0.0]])

    def test_all_disabled_hardware_does_not_load_imu_config(self):
        with patch.object(_HW_MODULE, "_load_control_config", side_effect=AssertionError("IMU config loaded")):
            hw = HardwareInterface(
                enable_pwm=False,
                enable_imu=False,
                enable_adc=False,
                start_imu_reader=False,
                start_adc_reader=False,
                log_level="CRITICAL",
            )

        try:
            self.assertTrue(hw.is_hardware_ready())
            self.assertEqual(hw.get_status()["pwm_state"], ReadyState.READY)
            self.assertEqual(hw.get_status()["imu_state"], ReadyState.READY)
            self.assertEqual(hw.get_status()["adc_state"], ReadyState.READY)
            self.assertEqual(hw.get_pwm_channel_names(), [])
            self.assertEqual(hw.get_latest_adc_snapshot(), {"readings": {}, "timestamp": None})
            self.assertIsNone(hw.read_base_imu())
        finally:
            hw.shutdown()

    def test_jetson_pwm_config_uses_selected_bus_with_imu_adc_disabled(self):
        calls = []

        def open_bus(bus_number):
            calls.append(("bus", bus_number))
            return _FakeSmbus(calls)

        with patch("modules.PCA9685_controller._open_smbus", side_effect=open_bus):
            hw = HardwareInterface(
                config_file=str(ROOT_DIR / "configuration_files" / "servo_config_jetson.yaml"),
                enable_pwm=True,
                enable_imu=False,
                enable_adc=False,
                start_imu_reader=False,
                start_adc_reader=False,
                input_rate_threshold=0,
                stale_timeout_s=0.0,
                cleanup_disable_osc=False,
                pwm_i2c_bus=7,
                pwm_i2c_addr=0x40,
                log_level="CRITICAL",
            )

        try:
            self.assertTrue(hw.is_hardware_ready())
            self.assertEqual(calls[0], ("bus", 7))
            self.assertEqual(hw.get_status()["imu_state"], ReadyState.READY)
            self.assertEqual(hw.get_status()["adc_state"], ReadyState.READY)
            self.assertEqual(
                hw.get_pwm_channel_names(include_pump=True),
                ["trackL", "trackR", "slew", "lift", "tilt", "scoop", "extra1", "extra2", "pump"],
            )
            self.assertTrue(hw.send_named_pwm_commands({
                "trackL": 0.0,
                "trackR": 0.0,
                "slew": 0.0,
                "lift": 0.0,
                "tilt": 0.0,
                "scoop": 0.0,
            }))
            self.assertIn(("block", 0x40, 0x06, 32), calls)
        finally:
            hw.shutdown()


if __name__ == "__main__":
    unittest.main()
