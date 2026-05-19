import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import simple_drive  # noqa: E402


class _Args:
    robot = "jetson"
    config_file = None
    pwm_i2c_bus = None
    pwm_i2c_addr = None
    enable_imu = False
    no_imu = False


class _FakeHardware:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.pwm_controller = None
        self.shutdown_calls = 0
        _FakeHardware.instances.append(self)

    def is_hardware_ready(self):
        return True

    def shutdown(self):
        self.shutdown_calls += 1


class _FakeSocket:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, *args, **kwargs):
        pass

    def handshake(self, timeout=0.0):
        return False


class SimpleDriveProfileTests(unittest.TestCase):
    def test_jetson_profile_maps_pwm_names_and_disables_imu(self):
        profile = simple_drive.resolve_robot_profile(_Args())

        self.assertEqual(profile["config_file"], "configuration_files/servo_config_jetson.yaml")
        self.assertEqual(profile["pwm_i2c_bus"], 7)
        self.assertEqual(profile["pwm_i2c_addr"], 0x40)
        self.assertFalse(profile["enable_imu"])
        self.assertEqual(
            simple_drive.map_pwm_commands(
                {
                    "rotate": 1.0,
                    "lift_boom": 0.5,
                    "tilt_boom": -0.5,
                    "scoop": -1.0,
                    "trackL": 0.25,
                    "trackR": -0.25,
                },
                profile["command_names"],
            ),
            {
                "slew": 1.0,
                "lift": 0.5,
                "tilt": -0.5,
                "scoop": -1.0,
                "trackL": 0.25,
                "trackR": -0.25,
            },
        )

    def test_jetson_main_uses_hardware_interface_with_hard_disabled_imu_adc(self):
        _FakeHardware.instances = []

        with patch.object(sys, "argv", ["simple_drive.py", "--robot", "jetson"]), \
                patch.object(simple_drive, "UDPSocket", _FakeSocket), \
                patch("modules.hardware_interface.HardwareInterface", _FakeHardware):
            with self.assertRaises(SystemExit):
                simple_drive.main()

        self.assertEqual(len(_FakeHardware.instances), 1)
        kwargs = _FakeHardware.instances[0].kwargs
        self.assertEqual(kwargs["config_file"], "configuration_files/servo_config_jetson.yaml")
        self.assertEqual(kwargs["pwm_i2c_bus"], 7)
        self.assertEqual(kwargs["pwm_i2c_addr"], 0x40)
        self.assertTrue(kwargs["enable_pwm"])
        self.assertFalse(kwargs["enable_imu"])
        self.assertFalse(kwargs["enable_adc"])
        self.assertFalse(kwargs["start_imu_reader"])
        self.assertFalse(kwargs["start_adc_reader"])
        self.assertGreaterEqual(_FakeHardware.instances[0].shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
