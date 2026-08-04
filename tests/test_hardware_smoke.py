"""Hardware smoke tests — real PCA9685 using the selected robot profile.

Safe to run on the bench: every test path ends with a pump-stop + PWM reset.
If the PWM controller cannot be reached (no I2C hardware) the whole suite is
skipped.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def _resolve_hardware_profile() -> dict:
    """Resolve board/profile defaults exactly like runtime scripts.

    Override with KAIVURI_TEST_ROBOT=rpi|jetson|auto, KAIVURI_TEST_I2C_BUS,
    or KAIVURI_TEST_I2C_ADDR when testing a non-default wiring.
    """
    try:
        from modules.board import resolve_profile  # noqa: E402

        profile = resolve_profile(os.environ.get("KAIVURI_TEST_ROBOT", "auto"))
    except Exception as exc:
        # Covers both an unimportable resolver and auto-detection failing off-board
        # (dev laptop / container without /proc/device-tree). Either way there is
        # no PCA9685 to talk to, so skip rather than error out at collection.
        raise unittest.SkipTest(f"Board profile unavailable: {exc}") from exc
    if "KAIVURI_TEST_I2C_BUS" in os.environ:
        profile["pwm_i2c_bus"] = int(os.environ["KAIVURI_TEST_I2C_BUS"], 0)
    if "KAIVURI_TEST_I2C_ADDR" in os.environ:
        profile["pwm_i2c_addr"] = int(os.environ["KAIVURI_TEST_I2C_ADDR"], 0)
    return profile


_PROFILE = _resolve_hardware_profile()


def _pwm_available() -> bool:
    """Return True if the profile-selected PCA9685 is reachable."""
    try:
        import smbus2  # type: ignore
    except Exception:
        return False
    try:
        bus = smbus2.SMBus(int(_PROFILE["pwm_i2c_bus"]))
        try:
            bus.read_byte(int(_PROFILE["pwm_i2c_addr"]))
            return True
        except Exception:
            return False
        finally:
            try:
                bus.close()
            except Exception:
                pass
    except Exception:
        return False


def _load_hardware_interface():
    try:
        from modules.hardware_interface import HardwareInterface  # noqa: E402
        return HardwareInterface
    except Exception as exc:
        raise unittest.SkipTest(f"Hardware interface unavailable: {exc}") from exc


@unittest.skipUnless(
    _pwm_available(),
    f"PCA9685 not reachable on I2C bus {_PROFILE['pwm_i2c_bus']} "
    f"addr 0x{int(_PROFILE['pwm_i2c_addr']):02X}",
)
class HardwareSmokeTests(unittest.TestCase):
    """Hardware safety sanity — every test resets the PWM layer afterwards."""

    @classmethod
    def setUpClass(cls):
        hardware_interface = _load_hardware_interface()
        cls.hw = hardware_interface(
            enable_pwm=True,
            enable_imu=False,
            enable_adc=False,
            log_level="WARNING",
            pump_auto_mode=False,
            config_file=_PROFILE["servo_config_file"],
            control_config_file=_PROFILE["control_config_file"],
            pwm_i2c_bus=int(_PROFILE["pwm_i2c_bus"]),
            pwm_i2c_addr=int(_PROFILE["pwm_i2c_addr"]),
        )

    @classmethod
    def tearDownClass(cls):
        try:
            cls.hw.reset(reset_pump=True)
        finally:
            cls.hw.shutdown()

    def tearDown(self):
        # Leave the system safe after every test.
        self.hw.reset(reset_pump=True)

    def test_hardware_reports_ready_with_pwm_only(self):
        self.assertTrue(self.hw.is_hardware_ready())

    def test_expected_pwm_channel_names_exist(self):
        names = set(self.hw.get_pwm_channel_names(include_pump=True))
        for required in ("slew", "boom", "arm", "bucket", "pump"):
            self.assertIn(required, names, msg=f"missing channel: {required}")

    def test_zero_named_pwm_commands_accepted(self):
        ok = self.hw.send_named_pwm_commands({
            "slew": 0.0,
            "boom": 0.0,
            "arm": 0.0,
            "bucket": 0.0,
        })
        self.assertTrue(ok)

    def test_pump_enable_disable_cycle(self):
        self.assertTrue(self.hw.set_pump_enabled(False, flush=True))
        self.assertTrue(self.hw.set_pump_enabled(True, flush=True))
        # Always end with pump off for safety.
        self.assertTrue(self.hw.set_pump_enabled(False, flush=True))

    def test_reset_pump_clears_pump_output(self):
        # Enable pump, then hard-reset with pump flag — pump command path must
        # accept this without error (real safety path on fault).
        self.hw.set_pump_enabled(True, flush=True)
        self.hw.reset(reset_pump=True)

    def test_status_fields_populated(self):
        status = self.hw.get_status()
        for key in ("pwm_state", "imu_state", "adc_state"):
            self.assertIn(key, status)
        # PWM must be READY; IMU/ADC are disabled so mark READY by convention.
        self.assertEqual(status["pwm_state"], "ready")


if __name__ == "__main__":
    unittest.main()
