"""simple_drive profile / wiring tests.

simple_drive is now open-loop only and takes four args (--robot, --ip,
--enable-slew, --enable-tracks). The per-run config overrides it used to
carry (--config-file, --pwm-i2c-bus, --disable-imu, ...) are gone, so profile
resolution is a straight delegation to modules.board.resolve_profile and is
covered there; what remains worth testing here is that the resolved profile
actually reaches HardwareInterface.

Known test gaps
---------------
- IMU end-to-end through main(): when IMU is enabled main() creates
  ExcavatorController which calls hardware._check_faults() and several other
  methods, making _FakeHardware too thin. The wiring tests below force the
  IMU off at the profile level to take the PWMOnlyController branch, so the
  ExcavatorController path through main() is NOT covered.

- DirectPWMWriter.flush() smbus block chunking: the 32-byte SMBus write limit
  splits the I2C buffer for channel ranges wider than 8 channels. No unit test
  exercises a multi-chunk write with a mock SMBus; currently only hit
  incidentally by the hardware smoke tests on real hardware.

- modules.board.detect() branches are covered in test_board_detect.py.
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


def _args(robot="rpi"):
    """Build a minimal args namespace for resolve_robot_profile."""
    class _A:
        pass
    a = _A()
    a.robot = robot
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
        self.assertEqual(profile["servo_config_file"], "configuration_files/profiles/rpi/servo_config.yaml")
        self.assertEqual(profile["control_config_file"], "configuration_files/profiles/rpi/control_config.yaml")
        self.assertEqual(profile["profile_name"], "rpi")
        self.assertEqual(profile["board"], "rpi")
        self.assertEqual(profile["pwm_i2c_bus"], 1)
        self.assertEqual(profile["pwm_i2c_addr"], 0x40)
        self.assertTrue(profile["enable_imu"])

    def test_jetson_profile_defaults(self):
        profile = simple_drive.resolve_robot_profile(_args("jetson"))
        self.assertEqual(profile["servo_config_file"], "configuration_files/profiles/jetson/servo_config.yaml")
        self.assertEqual(profile["control_config_file"], "configuration_files/profiles/jetson/control_config.yaml")
        self.assertEqual(profile["profile_name"], "jetson")
        self.assertEqual(profile["board"], "jetson")
        self.assertEqual(profile["pwm_i2c_bus"], 7)
        self.assertEqual(profile["pwm_i2c_addr"], 0x40)
        self.assertTrue(profile["enable_imu"])

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


# ---------------------------------------------------------------------------
# main() wiring — verifies HardwareInterface receives the right bus number
# ---------------------------------------------------------------------------

class MainWiringTests(unittest.TestCase):
    """main() must hand HardwareInterface exactly what the profile resolved.

    --ip is passed so main() takes the UDP branch, where _FakeSocket refuses
    the handshake and main() exits before it needs a real pad or real I2C.
    """

    def _run_main(self, robot):
        _FakeHardware.instances = []
        # Real profile for this board, IMU forced off so main() takes the
        # PWMOnlyController branch instead of constructing ExcavatorController
        # against _FakeHardware.
        profile = dict(board_module.resolve_profile(robot))
        profile["enable_imu"] = False

        argv = ["simple_drive.py", "--robot", robot, "--ip", "0.0.0.0:8080"]
        with patch.object(sys, "argv", argv), \
                patch.object(simple_drive, "_resolve_board_profile", return_value=profile), \
                patch.object(simple_drive, "UDPSocket", _FakeSocket), \
                patch("modules.hardware_interface.HardwareInterface", _FakeHardware):
            with self.assertRaises(SystemExit):
                simple_drive.main()
        self.assertEqual(len(_FakeHardware.instances), 1)
        return _FakeHardware.instances[0].kwargs

    def test_rpi_main_uses_bus_1(self):
        kwargs = self._run_main("rpi")
        self.assertEqual(kwargs["pwm_i2c_bus"], 1)
        self.assertEqual(kwargs["config_file"], "configuration_files/profiles/rpi/servo_config.yaml")
        self.assertEqual(kwargs["control_config_file"], "configuration_files/profiles/rpi/control_config.yaml")
        self.assertTrue(kwargs["enable_pwm"])

    def test_jetson_main_uses_bus_7(self):
        kwargs = self._run_main("jetson")
        self.assertEqual(kwargs["pwm_i2c_bus"], 7)
        self.assertEqual(kwargs["config_file"], "configuration_files/profiles/jetson/servo_config.yaml")
        self.assertEqual(kwargs["control_config_file"], "configuration_files/profiles/jetson/control_config.yaml")
        self.assertFalse(kwargs["enable_adc"])
        self.assertGreaterEqual(_FakeHardware.instances[0].shutdown_calls, 1)

    def test_pump_is_fixed_not_auto(self):
        # simple_drive is open-loop only; the pump is static and button X
        # toggles it. Auto-pump lives in control_prototype/drive_compensated.py.
        kwargs = self._run_main("jetson")
        self.assertFalse(kwargs["pump_auto_mode"])

    def test_toggleable_channels_stay_enabled(self):
        kwargs = self._run_main("rpi")
        self.assertTrue(kwargs["toggle_channels"])


# ---------------------------------------------------------------------------
# CLI surface — the cleanup is the point, so guard it
# ---------------------------------------------------------------------------

class ArgSurfaceTests(unittest.TestCase):

    def _parse(self, argv):
        with patch.object(sys, "argv", ["simple_drive.py", *argv]):
            return simple_drive._parse_args()

    def test_defaults_to_local_gamepad(self):
        args = self._parse([])
        self.assertIsNone(args.ip)
        self.assertIsInstance(simple_drive.make_input_source(args),
                              simple_drive.LocalGamepadInput)

    def test_ip_selects_udp(self):
        args = self._parse(["--ip", "0.0.0.0:8080"])
        self.assertIsInstance(simple_drive.make_input_source(args),
                              simple_drive.UDPInput)

    def test_slew_and_tracks_are_off_by_default(self):
        args = self._parse([])
        self.assertFalse(args.enable_slew)
        self.assertFalse(args.enable_tracks)

    def test_arg_surface_stays_small(self):
        # The whole point of the cleanup. If an arg is added here on purpose,
        # update this set — it exists so knobs do not creep back in one at a
        # time. Per-run tuning belongs in the profile's control_config.yaml,
        # and compensation belongs in control_prototype/drive_compensated.py.
        args = self._parse([])
        self.assertEqual(
            set(vars(args)),
            {"robot", "ip", "enable_slew", "enable_tracks"},
        )


if __name__ == "__main__":
    unittest.main()
