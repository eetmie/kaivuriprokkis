"""Hardware smoke tests — real PCA9685 on bus 1, IMU/ADC disabled.

Safe to run on the bench: every test path ends with a pump-stop + PWM reset.
If the PWM controller cannot be reached (no I2C hardware) the whole suite is
skipped.
"""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

def _pwm_available() -> bool:
    """Return True if a PCA9685 is reachable on bus 1 (0x40)."""
    try:
        import smbus2  # type: ignore
    except Exception:
        try:
            import smbus  # type: ignore  # noqa: F401
            smbus2 = __import__("smbus")
        except Exception:
            return False
    try:
        bus = smbus2.SMBus(1)
        try:
            bus.read_byte(0x40)
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


@unittest.skipUnless(_pwm_available(), "PCA9685 not reachable on I2C bus 1")
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
        for required in ("lift_boom", "tilt_boom", "scoop", "rotate", "pump"):
            self.assertIn(required, names, msg=f"missing channel: {required}")

    def test_zero_named_pwm_commands_accepted(self):
        ok = self.hw.send_named_pwm_commands({
            "rotate": 0.0,
            "lift_boom": 0.0,
            "tilt_boom": 0.0,
            "scoop": 0.0,
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
