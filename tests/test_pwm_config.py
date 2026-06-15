import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.pwm.config import ChannelConfig  # noqa: E402


class PWMConfigTests(unittest.TestCase):
    def test_inactive_feature_defaults_are_zero_or_false(self):
        cfg = ChannelConfig(
            output_channel=0,
            pulse_min=1000,
            pulse_max=2000,
            direction=1,
        )

        self.assertEqual(cfg.center, 1500.0)
        self.assertEqual(cfg.deadzone, 0.0)
        self.assertFalse(cfg.affects_pump)
        self.assertFalse(cfg.toggleable)
        self.assertEqual(cfg.deadband_us_pos, 0.0)
        self.assertEqual(cfg.deadband_us_neg, 0.0)
        self.assertFalse(cfg.dither_enable)
        self.assertEqual(cfg.dither_amp_us, 0.0)
        self.assertEqual(cfg.dither_hz, 0.0)
        self.assertFalse(cfg.dither_taper)
        self.assertEqual(cfg.dither_taper_us, 0.0)
        self.assertFalse(cfg.ramp_enable)
        self.assertEqual(cfg.ramp_limit, 0.0)
        self.assertFalse(cfg.ramp_skip_deadband)
        self.assertEqual(cfg.gamma, 1.0)


if __name__ == "__main__":
    unittest.main()
