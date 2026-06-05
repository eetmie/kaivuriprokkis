import inspect
import logging
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.udp_socket import UDPSocket  # noqa: E402
from tools.pid_tuner import robot as pid_tuner_robot  # noqa: E402


class _FakeUDPSocket:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = dict(kwargs)
        self.setup_kwargs = None
        self.sent = []
        self.closed = False
        _FakeUDPSocket.instances.append(self)

    def setup(self, host, port, inputs='', outputs='', is_server=False):
        self.setup_kwargs = {
            "host": host,
            "port": port,
            "inputs": inputs,
            "outputs": outputs,
            "is_server": is_server,
        }
        return True

    def handshake(self, timeout=5.0):
        return True

    def start_receiving(self):
        pass

    def get_latest(self):
        return [2.0, 10.0, 3.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5]

    def send(self, values):
        self.sent.append(list(values))
        raise KeyboardInterrupt

    def close(self):
        self.closed = True


class _FakeHardware:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.ready_checks = 0
        self.pump_log = []
        self.command_log = []
        self.reset_log = []
        self.shutdown_calls = 0
        self.reload_calls = 0
        _FakeHardware.instances.append(self)

    def is_hardware_ready(self):
        self.ready_checks += 1
        return True

    def read_all_imu_quaternions(self):
        return [np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for _ in range(4)]

    def set_pump_enabled(self, enabled):
        self.pump_log.append(bool(enabled))
        return True

    def send_named_pwm_commands(self, commands, *, unset_to_zero=None, command_ts=None):
        self.command_log.append((dict(commands), unset_to_zero, command_ts))
        return True

    def reset(self, reset_pump=False):
        self.reset_log.append(bool(reset_pump))

    def reload_config(self):
        self.reload_calls += 1
        return True

    def shutdown(self):
        self.shutdown_calls += 1


class PIDTunerRobotTests(unittest.TestCase):
    def setUp(self):
        self._logging_disabled = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(self._logging_disabled)

    def test_helpers_clamp_joint_and_wrap_angle_error(self):
        self.assertEqual(pid_tuner_robot._safe_joint_id(-10.0), 0)
        self.assertEqual(pid_tuner_robot._safe_joint_id(2.49), 2)
        self.assertEqual(pid_tuner_robot._safe_joint_id(99.0), 3)

        error = pid_tuner_robot._angle_error(math.radians(179.0), math.radians(-179.0))
        self.assertAlmostEqual(math.degrees(error), -2.0, places=5)

    def test_read_joint_angles_uses_current_imu_pipeline(self):
        class Hardware:
            def read_all_imu_quaternions(self):
                return [[1.0, 0.0, 0.0, 0.0]] * 4

        expected = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        with patch.object(pid_tuner_robot, "canonical_joint_angles_from_imus",
                          return_value=expected) as canonical:
            angles = pid_tuner_robot._read_joint_angles(Hardware(), object())

        np.testing.assert_allclose(angles, expected)
        self.assertEqual(canonical.call_args.args[0].shape, (4, 4))

    def test_robot_tuner_uses_current_hardware_and_udp_apis(self):
        import importlib
        hardware_interface_module = importlib.import_module("modules.hardware_interface")
        HardwareInterfaceClass = hardware_interface_module.HardwareInterface
        hw_sig = inspect.signature(HardwareInterfaceClass)
        for name in (
            "config_file",
            "pump_auto_mode",
            "cleanup_disable_osc",
            "enable_adc",
            "start_adc_reader",
            "log_level",
        ):
            self.assertIn(name, hw_sig.parameters)

        for method in (
            "read_all_imu_quaternions",
            "is_hardware_ready",
            "reload_config",
            "set_pump_enabled",
            "send_named_pwm_commands",
            "reset",
            "shutdown",
        ):
            self.assertTrue(hasattr(HardwareInterfaceClass, method), method)

        for method in ("setup", "handshake", "start_receiving", "get_latest", "send", "close"):
            self.assertTrue(hasattr(UDPSocket, method), method)

    def test_main_single_command_dry_run_sends_selected_pwm_and_cleans_up(self):
        _FakeUDPSocket.instances = []
        _FakeHardware.instances = []

        with patch.object(pid_tuner_robot, "UDPSocket", _FakeUDPSocket), \
                patch.object(pid_tuner_robot, "HardwareInterface", _FakeHardware), \
                patch.object(pid_tuner_robot, "load_excavator_robot_config",
                             return_value=object()), \
                patch.object(
                    pid_tuner_robot,
                    "canonical_joint_angles_from_imus",
                    return_value=np.asarray([0.0, 0.0, math.radians(5.0), 0.0],
                                            dtype=np.float32),
                ), \
                patch.object(sys, "argv", [
                    "robot.py",
                    "--host", "127.0.0.1",
                    "--port", "8090",
                    "--rate", "50",
                    "--log-level", "ERROR",
                ]):
            rc = pid_tuner_robot.main_joint()

        self.assertEqual(rc, 0)
        self.assertEqual(len(_FakeUDPSocket.instances), 1)
        self.assertEqual(len(_FakeHardware.instances), 1)

        sock = _FakeUDPSocket.instances[0]
        self.assertEqual(sock.setup_kwargs["inputs"],
                         f'{pid_tuner_robot.COMMAND_SIZE}f')
        self.assertEqual(sock.setup_kwargs["outputs"],
                         f'{pid_tuner_robot.TELEMETRY_SIZE}f')
        self.assertTrue(sock.setup_kwargs["is_server"])
        self.assertTrue(sock.closed)

        hardware = _FakeHardware.instances[0]
        self.assertEqual(hardware.kwargs["config_file"],
                         "configuration_files/profiles/rpi/servo_config.yaml")
        self.assertEqual(hardware.kwargs["control_config_file"],
                         "configuration_files/profiles/rpi/control_config.yaml")
        self.assertFalse(hardware.kwargs["pump_auto_mode"])
        self.assertFalse(hardware.kwargs["enable_adc"])
        self.assertFalse(hardware.kwargs["start_adc_reader"])
        self.assertEqual(hardware.pump_log, [True])
        self.assertEqual(len(hardware.command_log), 1)
        commands, unset_to_zero, command_ts = hardware.command_log[0]
        self.assertEqual(set(commands), {"arm"})
        self.assertGreater(commands["arm"], 0.0)
        self.assertLessEqual(commands["arm"], 0.5)
        self.assertTrue(unset_to_zero)
        self.assertIsNone(command_ts)
        self.assertIn(True, hardware.reset_log)
        self.assertEqual(hardware.shutdown_calls, 1)

        telemetry = sock.sent[0]
        self.assertEqual(len(telemetry), pid_tuner_robot.TELEMETRY_SIZE)
        self.assertEqual(int(telemetry[0]), 2)
        self.assertAlmostEqual(telemetry[3], 5.0, places=5)
        self.assertAlmostEqual(telemetry[5], 10.0, places=5)
        self.assertAlmostEqual(telemetry[7], 3.0, places=5)
        self.assertAlmostEqual(telemetry[11], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
