"""
simple_drive profile / wiring tests.

Known test gaps
---------------
- --enable-imu override end-to-end through main(): when IMU is enabled main()
  creates ExcavatorController which calls hardware._check_faults() and several
  other methods, making _FakeHardware too thin. Would need a heavier controller
  stub or a dedicated integration fixture. Covered at profile-resolution level
  (test_enable_imu_overrides_jetson_profile) but NOT wired all the way to the
  HardwareInterface kwargs through main().

- DirectPWMWriter.flush() smbus block chunking: the 32-byte SMBus write limit
  splits the I2C buffer for channel ranges wider than 8 channels. No unit test
  exercises a multi-chunk write with a mock SMBus; currently only hit
  incidentally by the hardware smoke tests on real hardware.

- ADCPi.__detect_raspberry_pi_bus(): hostname-based I2C bus auto-detection
  branches on platform.uname(). Not tested; low-risk since it is only a
  fallback when the explicit bus is not set.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import simple_drive  # noqa: E402


def _args(robot="rpi", *, config_file=None, pwm_i2c_bus=None, pwm_i2c_addr=None,
          enable_imu=False, no_imu=False):
    """Build a minimal args namespace for resolve_robot_profile."""
    class _A:
        pass
    a = _A()
    a.robot = robot
    a.config_file = config_file
    a.pwm_i2c_bus = pwm_i2c_bus
    a.pwm_i2c_addr = pwm_i2c_addr
    a.enable_imu = enable_imu
    a.no_imu = no_imu
    return a


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


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------

class ProfileResolutionTests(unittest.TestCase):

    def test_rpi_profile_defaults(self):
        profile = simple_drive.resolve_robot_profile(_args("rpi"))
        self.assertEqual(profile["config_file"], "configuration_files/servo_config_200.yaml")
        self.assertEqual(profile["pwm_i2c_bus"], 1)
        self.assertEqual(profile["pwm_i2c_addr"], 0x40)
        self.assertTrue(profile["enable_imu"])

    def test_jetson_profile_defaults(self):
        profile = simple_drive.resolve_robot_profile(_args("jetson"))
        self.assertEqual(profile["config_file"], "configuration_files/servo_config_jetson.yaml")
        self.assertEqual(profile["pwm_i2c_bus"], 7)
        self.assertEqual(profile["pwm_i2c_addr"], 0x40)
        self.assertFalse(profile["enable_imu"])

    def test_cli_bus_override_applies_to_rpi(self):
        profile = simple_drive.resolve_robot_profile(_args("rpi", pwm_i2c_bus=3))
        self.assertEqual(profile["pwm_i2c_bus"], 3)

    def test_cli_bus_override_applies_to_jetson(self):
        profile = simple_drive.resolve_robot_profile(_args("jetson", pwm_i2c_bus=1))
        self.assertEqual(profile["pwm_i2c_bus"], 1)

    def test_cli_addr_override(self):
        profile = simple_drive.resolve_robot_profile(_args("rpi", pwm_i2c_addr=0x41))
        self.assertEqual(profile["pwm_i2c_addr"], 0x41)

    def test_cli_config_override(self):
        profile = simple_drive.resolve_robot_profile(
            _args("rpi", config_file="configuration_files/servo_config_200_linear.yaml")
        )
        self.assertEqual(profile["config_file"], "configuration_files/servo_config_200_linear.yaml")

    def test_no_imu_overrides_rpi_profile(self):
        # rpi enables IMU by default; --no-imu should force it off
        profile = simple_drive.resolve_robot_profile(_args("rpi", no_imu=True))
        self.assertFalse(profile["enable_imu"])

    def test_enable_imu_overrides_jetson_profile(self):
        # jetson disables IMU by default; --enable-imu should force it on
        profile = simple_drive.resolve_robot_profile(_args("jetson", enable_imu=True))
        self.assertTrue(profile["enable_imu"])


# ---------------------------------------------------------------------------
# Command name mapping
# ---------------------------------------------------------------------------

class CommandMappingTests(unittest.TestCase):

    def test_rpi_identity_mapping(self):
        profile = simple_drive.resolve_robot_profile(_args("rpi"))
        result = simple_drive.map_pwm_commands(
            {"rotate": 1.0, "lift_boom": 0.5, "tilt_boom": -0.5, "scoop": -1.0},
            profile["command_names"],
        )
        # rpi names are pass-through
        self.assertEqual(result, {"rotate": 1.0, "lift_boom": 0.5, "tilt_boom": -0.5, "scoop": -1.0})

    def test_jetson_remaps_names(self):
        profile = simple_drive.resolve_robot_profile(_args("jetson"))
        result = simple_drive.map_pwm_commands(
            {"rotate": 1.0, "lift_boom": 0.5, "tilt_boom": -0.5, "scoop": -1.0,
             "trackL": 0.25, "trackR": -0.25},
            profile["command_names"],
        )
        self.assertEqual(result, {"slew": 1.0, "lift": 0.5, "tilt": -0.5, "scoop": -1.0,
                                  "trackL": 0.25, "trackR": -0.25})


# ---------------------------------------------------------------------------
# main() wiring — verifies HardwareInterface receives the right bus number
# ---------------------------------------------------------------------------

class MainWiringTests(unittest.TestCase):

    def _run_main(self, argv):
        _FakeHardware.instances = []
        with patch.object(sys, "argv", argv), \
                patch.object(simple_drive, "UDPSocket", _FakeSocket), \
                patch("modules.hardware_interface.HardwareInterface", _FakeHardware):
            with self.assertRaises(SystemExit):
                simple_drive.main()
        self.assertEqual(len(_FakeHardware.instances), 1)
        return _FakeHardware.instances[0].kwargs

    def test_rpi_main_uses_bus_1(self):
        kwargs = self._run_main(["simple_drive.py", "--robot", "rpi", "--no-imu"])
        self.assertEqual(kwargs["pwm_i2c_bus"], 1)
        self.assertEqual(kwargs["config_file"], "configuration_files/servo_config_200.yaml")
        self.assertTrue(kwargs["enable_pwm"])

    def test_jetson_main_uses_bus_7(self):
        kwargs = self._run_main(["simple_drive.py", "--robot", "jetson"])
        self.assertEqual(kwargs["pwm_i2c_bus"], 7)
        self.assertEqual(kwargs["config_file"], "configuration_files/servo_config_jetson.yaml")
        self.assertFalse(kwargs["enable_imu"])
        self.assertFalse(kwargs["enable_adc"])
        self.assertGreaterEqual(_FakeHardware.instances[0].shutdown_calls, 1)

    def test_cli_bus_override_reaches_hardware_interface(self):
        kwargs = self._run_main(["simple_drive.py", "--robot", "jetson", "--pwm-i2c-bus", "3"])
        self.assertEqual(kwargs["pwm_i2c_bus"], 3)

    def test_cli_no_imu_override_reaches_hardware_interface(self):
        # rpi enables IMU by default; --no-imu must flow through to HardwareInterface
        kwargs = self._run_main(["simple_drive.py", "--robot", "rpi", "--no-imu"])
        self.assertFalse(kwargs["enable_imu"])


if __name__ == "__main__":
    unittest.main()
