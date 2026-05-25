"""
simple_drive profile / wiring tests.

Known test gaps
---------------
- IMU end-to-end through main(): when IMU is enabled main() creates
  ExcavatorController which calls hardware._check_faults() and several other
  methods, making _FakeHardware too thin. Covered at profile-resolution level
  but NOT wired all the way to HardwareInterface kwargs through main().

- DirectPWMWriter.flush() smbus block chunking: the 32-byte SMBus write limit
  splits the I2C buffer for channel ranges wider than 8 channels. No unit test
  exercises a multi-chunk write with a mock SMBus; currently only hit
  incidentally by the hardware smoke tests on real hardware.

- modules.board.detect(): hostname/device-tree auto-detection branches not unit-tested;
  low-risk since explicit --robot overrides are available.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import simple_drive  # noqa: E402
import modules.board as board_module  # noqa: E402


def _args(robot="rpi", *, config_file=None, control_config_file=None, pwm_i2c_bus=None, pwm_i2c_addr=None,
          disable_imu=False):
    """Build a minimal args namespace for resolve_robot_profile."""
    class _A:
        pass
    a = _A()
    a.robot = robot
    a.config_file = config_file
    a.control_config_file = control_config_file
    a.pwm_i2c_bus = pwm_i2c_bus
    a.pwm_i2c_addr = pwm_i2c_addr
    a.disable_imu = disable_imu
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
        self.assertEqual(profile["config_file"], "configuration_files/profiles/rpi/servo_config.yaml")
        self.assertEqual(profile["servo_config_file"], "configuration_files/profiles/rpi/servo_config.yaml")
        self.assertEqual(profile["control_config_file"], "configuration_files/profiles/rpi/control_config.yaml")
        self.assertEqual(profile["profile_name"], "rpi")
        self.assertEqual(profile["board"], "rpi")
        self.assertEqual(profile["profile_dir"], "configuration_files/profiles/rpi")
        self.assertEqual(profile["pwm_i2c_bus"], 1)
        self.assertEqual(profile["pwm_i2c_addr"], 0x40)
        self.assertTrue(profile["enable_imu"])

    def test_jetson_profile_defaults(self):
        profile = simple_drive.resolve_robot_profile(_args("jetson"))
        self.assertEqual(profile["config_file"], "configuration_files/profiles/jetson/servo_config.yaml")
        self.assertEqual(profile["servo_config_file"], "configuration_files/profiles/jetson/servo_config.yaml")
        self.assertEqual(profile["control_config_file"], "configuration_files/profiles/jetson/control_config.yaml")
        self.assertEqual(profile["profile_name"], "jetson")
        self.assertEqual(profile["board"], "jetson")
        self.assertEqual(profile["profile_dir"], "configuration_files/profiles/jetson")
        self.assertEqual(profile["pwm_i2c_bus"], 7)
        self.assertEqual(profile["pwm_i2c_addr"], 0x40)
        self.assertTrue(profile["enable_imu"])

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
            _args("rpi", config_file="configuration_files/old/servo_config_200_linear.yaml")
        )
        self.assertEqual(profile["config_file"], "configuration_files/old/servo_config_200_linear.yaml")
        self.assertEqual(profile["servo_config_file"], "configuration_files/old/servo_config_200_linear.yaml")

    def test_cli_control_config_override(self):
        profile = simple_drive.resolve_robot_profile(
            _args("rpi", control_config_file="configuration_files/profiles/rpi/control_config.yaml")
        )
        self.assertEqual(profile["control_config_file"], "configuration_files/profiles/rpi/control_config.yaml")

    def test_auto_detects_board_profile_only(self):
        with patch("modules.board.detect", return_value="jetson"):
            profile = simple_drive.resolve_robot_profile(_args("auto"))
        self.assertEqual(profile["profile_name"], "jetson")
        self.assertEqual(profile["board"], "jetson")
        self.assertEqual(profile["servo_config_file"], "configuration_files/profiles/jetson/servo_config.yaml")

    def test_explicit_robot_profile_uses_board_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_dir = root / "configuration_files" / "profiles" / "robot13"
            profile_dir.mkdir(parents=True)
            (profile_dir / "servo_config.yaml").write_text("pwm_frequency: 100\nCHANNEL_CONFIGS: {}\n", encoding="utf-8")
            (profile_dir / "control_config.yaml").write_text("robot: {}\n", encoding="utf-8")
            (profile_dir / "profile.yaml").write_text(
                "\n".join([
                    "name: robot13",
                    "board: jetson",
                    "servo_config_file: servo_config.yaml",
                    "control_config_file: control_config.yaml",
                    "enable_imu: true",
                    "enable_adc: false",
                    "",
                ]),
                encoding="utf-8",
            )

            with patch.object(board_module, "ROOT_DIR", root), \
                    patch.object(board_module, "PROFILES_DIR", profile_dir.parent):
                profile = board_module.resolve_profile("robot13")

        self.assertEqual(profile["profile_name"], "robot13")
        self.assertEqual(profile["board"], "jetson")
        self.assertEqual(profile["pwm_i2c_bus"], 7)
        self.assertEqual(profile["servo_config_file"], "configuration_files/profiles/robot13/servo_config.yaml")
        self.assertEqual(profile["control_config_file"], "configuration_files/profiles/robot13/control_config.yaml")

    def test_disable_imu_overrides_rpi_profile(self):
        # both profiles enable IMU by default; --disable-imu should force it off
        profile = simple_drive.resolve_robot_profile(_args("rpi", disable_imu=True))
        self.assertFalse(profile["enable_imu"])

    def test_disable_imu_overrides_jetson_profile(self):
        profile = simple_drive.resolve_robot_profile(_args("jetson", disable_imu=True))
        self.assertFalse(profile["enable_imu"])


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
        kwargs = self._run_main(["simple_drive.py", "--robot", "rpi", "--disable-imu"])
        self.assertEqual(kwargs["pwm_i2c_bus"], 1)
        self.assertEqual(kwargs["config_file"], "configuration_files/profiles/rpi/servo_config.yaml")
        self.assertEqual(kwargs["control_config_file"], "configuration_files/profiles/rpi/control_config.yaml")
        self.assertTrue(kwargs["enable_pwm"])

    def test_jetson_main_uses_bus_7(self):
        kwargs = self._run_main(["simple_drive.py", "--robot", "jetson", "--disable-imu"])
        self.assertEqual(kwargs["pwm_i2c_bus"], 7)
        self.assertEqual(kwargs["config_file"], "configuration_files/profiles/jetson/servo_config.yaml")
        self.assertEqual(kwargs["control_config_file"], "configuration_files/profiles/jetson/control_config.yaml")
        self.assertFalse(kwargs["enable_imu"])
        self.assertFalse(kwargs["enable_adc"])
        self.assertGreaterEqual(_FakeHardware.instances[0].shutdown_calls, 1)

    def test_disable_arg_disables_toggleable_channels_for_jetson(self):
        kwargs = self._run_main(["simple_drive.py", "--robot", "jetson", "--disable-imu", "--disable"])
        self.assertFalse(kwargs["toggle_channels"])

    def test_disable_arg_disables_toggleable_channels_for_rpi(self):
        kwargs = self._run_main(["simple_drive.py", "--robot", "rpi", "--disable-imu", "--disable"])
        self.assertEqual(kwargs["config_file"], "configuration_files/profiles/rpi/servo_config.yaml")
        self.assertEqual(kwargs["pwm_i2c_bus"], 1)
        self.assertFalse(kwargs["enable_imu"])
        self.assertFalse(kwargs["toggle_channels"])

    def test_cli_bus_override_reaches_hardware_interface(self):
        kwargs = self._run_main(["simple_drive.py", "--robot", "jetson", "--pwm-i2c-bus", "3", "--disable-imu"])
        self.assertEqual(kwargs["pwm_i2c_bus"], 3)

    def test_cli_disable_imu_reaches_hardware_interface(self):
        # both profiles enable IMU by default; --disable-imu must flow through to HardwareInterface
        kwargs = self._run_main(["simple_drive.py", "--robot", "rpi", "--disable-imu"])
        self.assertFalse(kwargs["enable_imu"])


if __name__ == "__main__":
    unittest.main()
