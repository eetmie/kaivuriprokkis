"""Unit tests for modules.direct_controller.DirectController.

The class is a thin wrapper around the hardware PWM bus, so the tests
focus on three behaviours:

* commands stored by ``give_commands`` are pushed by ``send_pending``
* empty/cleared commands trigger an idempotent zero (single
  ``hardware.reset(reset_pump=False)`` call until new commands arrive)
* ``emergency_stop`` clears state and slams outputs
"""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.direct_controller import DirectController  # noqa: E402


class _FakeHardware:
    def __init__(self):
        self.pwm_log = []
        self.reset_log = []

    def send_named_pwm_commands(self, commands):
        self.pwm_log.append(dict(commands))

    def reset(self, reset_pump=False):
        self.reset_log.append(bool(reset_pump))


class DirectControllerTests(unittest.TestCase):
    def setUp(self):
        self.hardware = _FakeHardware()
        self.direct = DirectController(self.hardware)

    def test_send_without_commands_zeroes_once(self):
        self.direct.send_pending()
        self.assertEqual(self.hardware.reset_log, [False])
        # Second send with still-no-commands must be idempotent.
        self.direct.send_pending()
        self.assertEqual(self.hardware.reset_log, [False])
        self.assertEqual(self.hardware.pwm_log, [])

    def test_give_then_send_pushes_to_hardware(self):
        cmds = {"slew": 0.1, "boom": -0.5, "arm": 0.0, "bucket": 0.3}
        self.direct.give_commands(cmds)
        self.direct.send_pending()
        self.assertEqual(len(self.hardware.pwm_log), 1)
        self.assertEqual(self.hardware.pwm_log[-1], cmds)
        self.assertEqual(self.hardware.reset_log, [])

    def test_give_commands_copies_dict(self):
        cmds = {"slew": 0.1, "boom": 0.0, "arm": 0.0, "bucket": 0.0}
        self.direct.give_commands(cmds)
        cmds["slew"] = 99.9  # mutate after handoff
        self.direct.send_pending()
        self.assertEqual(self.hardware.pwm_log[-1]["slew"], 0.1)

    def test_clear_zeroes_on_next_send(self):
        self.direct.give_commands({"slew": 0.4})
        self.direct.send_pending()
        self.assertEqual(len(self.hardware.pwm_log), 1)

        self.direct.clear()
        self.direct.send_pending()
        # First send-after-clear zeroes outputs.
        self.assertEqual(self.hardware.reset_log, [False])
        # Second send-after-clear is idempotent.
        self.direct.send_pending()
        self.assertEqual(self.hardware.reset_log, [False])

    def test_new_commands_after_zero_resume_pushes(self):
        self.direct.send_pending()  # zeros once
        self.direct.give_commands({"slew": 0.7})
        self.direct.send_pending()
        self.assertEqual(self.hardware.pwm_log[-1], {"slew": 0.7})
        # Subsequent empty send must zero again.
        self.direct.clear()
        self.direct.send_pending()
        self.assertEqual(self.hardware.reset_log, [False, False])

    def test_emergency_stop_clears_and_resets_pump_by_default(self):
        self.direct.give_commands({"slew": 0.5})
        self.direct.emergency_stop()
        self.assertEqual(self.hardware.reset_log, [True])
        # Next send_pending must not re-push the cleared dict.
        self.direct.send_pending()
        self.assertEqual(self.hardware.pwm_log, [])

    def test_emergency_stop_can_skip_pump_reset(self):
        self.direct.emergency_stop(reset_pump=False)
        self.assertEqual(self.hardware.reset_log, [False])


if __name__ == "__main__":
    unittest.main()
