import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.control_protocol import ControlMode  # noqa: E402
from tools import stress_full_stack  # noqa: E402


class StressFullStackHelpersTests(unittest.TestCase):
    def test_mixed_mode_alternates_by_block_duration(self):
        self.assertEqual(
            stress_full_stack._select_command_mode("mixed", sequence=0, target_hz=100.0, mixed_block_s=1.0),
            ControlMode.IK,
        )
        self.assertEqual(
            stress_full_stack._select_command_mode("mixed", sequence=99, target_hz=100.0, mixed_block_s=1.0),
            ControlMode.IK,
        )
        self.assertEqual(
            stress_full_stack._select_command_mode("mixed", sequence=100, target_hz=100.0, mixed_block_s=1.0),
            ControlMode.DIRECT,
        )
        self.assertEqual(
            stress_full_stack._select_command_mode("mixed", sequence=200, target_hz=100.0, mixed_block_s=1.0),
            ControlMode.IK,
        )

    def test_explicit_modes_do_not_depend_on_sequence(self):
        self.assertEqual(
            stress_full_stack._select_command_mode("ik", sequence=10_000, target_hz=200.0, mixed_block_s=0.5),
            ControlMode.IK,
        )
        self.assertEqual(
            stress_full_stack._select_command_mode("direct", sequence=0, target_hz=100.0, mixed_block_s=1.0),
            ControlMode.DIRECT,
        )

    def test_synthetic_joystick_state_is_bounded_and_armed(self):
        state = stress_full_stack._synthetic_controller_state(sequence=123, target_hz=200.0, amplitude=0.25)

        self.assertTrue(state["LeftBumper"])
        for key in ("LeftJoystickX", "LeftJoystickY", "RightJoystickX", "RightJoystickY"):
            self.assertLessEqual(abs(state[key]), 0.25)

    def test_synthetic_joystick_amplitude_is_clamped(self):
        state = stress_full_stack._synthetic_controller_state(sequence=123, target_hz=100.0, amplitude=5.0)

        for key in ("LeftJoystickX", "LeftJoystickY", "RightJoystickX", "RightJoystickY"):
            self.assertLessEqual(abs(state[key]), 1.0)


if __name__ == "__main__":
    unittest.main()
