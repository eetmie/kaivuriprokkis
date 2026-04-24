import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.control_protocol import ControlCommand
from modules.robot_service import RobotService


class _FakeController:
    def __init__(self):
        self.pause_calls = 0
        self.resume_calls = 0
        self.give_pose_calls = []

    def get_pose(self):
        return [0.0, 0.0, 0.0], 0.0

    def pause(self):
        self.pause_calls += 1

    def resume(self):
        self.resume_calls += 1

    def give_pose(self, position, rotation_y_deg):
        self.give_pose_calls.append((tuple(float(v) for v in position), float(rotation_y_deg)))


class _FakeHardware:
    def __init__(self):
        self.pump_history = []

    def set_pump_enabled(self, enabled: bool, flush: bool = True):
        self.pump_history.append((bool(enabled), bool(flush)))
        return True


class RobotServicePumpToggleTests(unittest.TestCase):
    @patch("modules.robot_service.diff_ik.load_excavator_robot_config", return_value=object())
    def test_pause_and_resume_toggle_pump(self, _mock_robot_cfg):
        controller = _FakeController()
        hardware = _FakeHardware()
        service = RobotService(controller, hardware)

        service.submit_command(ControlCommand(sequence=1, pause=True))
        self.assertTrue(service.state.paused)
        self.assertEqual(controller.pause_calls, 1)
        self.assertEqual(hardware.pump_history[-1], (False, True))

        service.submit_command(ControlCommand(sequence=2, resume=True))
        self.assertFalse(service.state.paused)
        self.assertEqual(controller.resume_calls, 1)
        self.assertEqual(hardware.pump_history[-1], (True, True))


if __name__ == "__main__":
    unittest.main()
