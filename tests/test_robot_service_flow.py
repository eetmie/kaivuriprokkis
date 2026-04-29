"""RobotService command-flow tests — mode switches, radial-composed poses, pump toggles.

These tests stub the controller/hardware so they exercise the RobotService
state machine without hitting any real hardware.
"""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.control_protocol import (  # noqa: E402
    ControlCommand,
    ControlMode,
    DirectCommand,
    PoseTarget,
    decode_command_message,
    encode_command_message,
)
from modules.excavator_target_state import ExcavatorTargetState  # noqa: E402
from modules.reachability import ReachabilityResult  # noqa: E402
from modules.robot_service import RobotService  # noqa: E402


class _FakeController:
    def __init__(self, initial_pose=((0.5, 0.0, 0.0), 0.0)):
        self.initial_pose = initial_pose
        self.pose_log = []
        self.next_pose_result = None
        self.direct_log = []
        self.pause_calls = 0
        self.resume_calls = 0
        self.enter_direct_calls = 0
        self.exit_direct_calls = 0
        self.perf_resets = 0

    def get_pose(self):
        position, rotation = self.initial_pose
        return np.asarray(position, dtype=np.float32), float(rotation)

    def pause(self):
        self.pause_calls += 1

    def resume(self):
        self.resume_calls += 1

    def enter_direct_mode(self):
        self.enter_direct_calls += 1

    def exit_direct_mode(self):
        self.exit_direct_calls += 1

    def give_pose(self, position, rotation_y_deg):
        self.pose_log.append((tuple(float(v) for v in position), float(rotation_y_deg)))
        return self.next_pose_result

    def give_direct_commands(self, commands):
        self.direct_log.append(dict(commands))

    def reset_performance_stats(self):
        self.perf_resets += 1

    def start(self):
        pass

    def stop(self):
        pass

    def get_joint_angles(self):
        return np.zeros(4, dtype=np.float32)

    def get_fk_quaternions(self):
        return None

    def get_condition_number(self):
        return 1.0

    def get_performance_stats(self):
        return {}


class _FakeHardware:
    def __init__(self):
        self.pump_log = []
        self.reset_log = []
        self.reload_log = 0
        self._ready = True

    def set_pump_enabled(self, enabled, flush=True):
        self.pump_log.append((bool(enabled), bool(flush)))
        return True

    def reset(self, reset_pump=False):
        self.reset_log.append(bool(reset_pump))

    def reload_config(self):
        self.reload_log += 1
        return True

    def is_hardware_ready(self):
        return self._ready

    def read_base_imu(self):
        return None


class ServiceSubmitCommandTests(unittest.TestCase):
    """RobotService.submit_command routes commands to controller/hardware correctly."""

    def _make_service(self, initial_pose=((0.5, 0.1, -0.05), 7.0)):
        controller = _FakeController(initial_pose=initial_pose)
        hardware = _FakeHardware()
        with patch("modules.robot_service.load_excavator_robot_config",
                   return_value=object()):
            service = RobotService(controller, hardware)
        return service, controller, hardware

    def test_initial_target_pose_seeded_from_controller(self):
        service, _, _ = self._make_service(initial_pose=((0.5, 0.1, -0.05), 7.0))
        self.assertAlmostEqual(service.state.target_pose.x, 0.5, places=6)
        self.assertAlmostEqual(service.state.target_pose.y, 0.1, places=6)
        self.assertAlmostEqual(service.state.target_pose.z, -0.05, places=6)
        self.assertAlmostEqual(service.state.target_pose.rot_y_deg, 7.0, places=6)

    def test_ik_pose_command_reaches_controller(self):
        service, controller, _ = self._make_service()
        cmd = ControlCommand(sequence=10, mode=ControlMode.IK,
                             pose=PoseTarget(0.4, 0.2, -0.1, 5.0))
        service.submit_command(cmd)
        self.assertEqual(len(controller.pose_log), 1)
        pos, rot = controller.pose_log[-1]
        self.assertAlmostEqual(pos[0], 0.4, places=5)
        self.assertAlmostEqual(pos[1], 0.2, places=5)
        self.assertAlmostEqual(pos[2], -0.1, places=5)
        self.assertAlmostEqual(rot, 5.0, places=5)

    def test_rejected_ik_pose_does_not_replace_last_service_target(self):
        service, controller, _ = self._make_service(initial_pose=((0.5, 0.1, -0.05), 7.0))
        previous = service.state.target_pose
        rejected = ReachabilityResult(
            reachable=False,
            closest_position=np.asarray([0.5, 0.1, -0.05], dtype=np.float32),
            pos_error_m=0.25,
            iters=80,
            final_cond_number=12.0,
        )
        controller.next_pose_result = rejected

        result = service.submit_command(ControlCommand(
            sequence=12,
            mode=ControlMode.IK,
            pose=PoseTarget(9.0, 9.0, 9.0, 90.0),
        ))

        self.assertIs(result, rejected)
        self.assertEqual(len(controller.pose_log), 1)
        self.assertAlmostEqual(service.state.target_pose.x, previous.x, places=6)
        self.assertAlmostEqual(service.state.target_pose.y, previous.y, places=6)
        self.assertAlmostEqual(service.state.target_pose.z, previous.z, places=6)
        self.assertAlmostEqual(service.state.target_pose.rot_y_deg, previous.rot_y_deg, places=6)

    def test_radial_composed_pose_arrives_as_cartesian(self):
        """When the client composes a radial target, the controller sees Cartesian."""
        service, controller, _ = self._make_service(initial_pose=((0.5, 0.0, 0.0), 0.0))
        target_state = ExcavatorTargetState()
        target_state.set_radial_pose(0.5, 45.0, -0.05, 10.0)
        cmd = ControlCommand(
            sequence=11, mode=ControlMode.IK,
            pose=PoseTarget(*target_state.compose_cartesian_pose()),
        )
        service.submit_command(cmd)

        pos, rot = controller.pose_log[-1]
        expected_x = 0.5 * math.cos(math.radians(45.0))
        expected_y = 0.5 * math.sin(math.radians(45.0))
        self.assertAlmostEqual(pos[0], expected_x, places=5)
        self.assertAlmostEqual(pos[1], expected_y, places=5)
        self.assertAlmostEqual(pos[2], -0.05, places=6)
        self.assertAlmostEqual(rot, 10.0, places=6)

    def test_mode_switch_ik_to_direct_and_back(self):
        service, controller, _ = self._make_service()
        service.submit_command(ControlCommand(
            sequence=1, mode=ControlMode.DIRECT,
            direct=DirectCommand(0.1, 0.2, 0.3, 0.4),
        ))
        self.assertEqual(controller.enter_direct_calls, 1)
        self.assertEqual(len(controller.direct_log), 1)
        self.assertEqual(controller.direct_log[-1],
                         {"rotate": 0.1, "lift_boom": 0.2, "tilt_boom": 0.3, "scoop": 0.4})

        service.submit_command(ControlCommand(
            sequence=2, mode=ControlMode.IK,
            pose=PoseTarget(0.5, 0.0, 0.0, 0.0),
        ))
        self.assertEqual(controller.exit_direct_calls, 1)

    def test_direct_commands_blocked_while_paused(self):
        service, controller, _ = self._make_service()
        service.submit_command(ControlCommand(sequence=1, pause=True))
        service.submit_command(ControlCommand(
            sequence=2, mode=ControlMode.DIRECT,
            direct=DirectCommand(0.5, 0.5, 0.5, 0.5),
        ))
        self.assertEqual(len(controller.direct_log), 0)

    def test_ik_commands_blocked_while_paused(self):
        service, controller, _ = self._make_service()
        service.submit_command(ControlCommand(sequence=1, pause=True))
        baseline = len(controller.pose_log)
        service.submit_command(ControlCommand(
            sequence=2, mode=ControlMode.IK,
            pose=PoseTarget(0.4, 0.0, 0.0, 0.0),
        ))
        self.assertEqual(len(controller.pose_log), baseline)

    def test_pause_toggles_pump_off_and_resume_toggles_on(self):
        service, controller, hardware = self._make_service()

        service.submit_command(ControlCommand(sequence=1, pause=True))
        self.assertTrue(service.state.paused)
        self.assertEqual(controller.pause_calls, 1)
        self.assertEqual(hardware.pump_log[-1], (False, True))

        service.submit_command(ControlCommand(sequence=2, resume=True))
        self.assertFalse(service.state.paused)
        self.assertEqual(controller.resume_calls, 1)
        self.assertEqual(hardware.pump_log[-1], (True, True))

    def test_reload_config_flag_triggers_hardware_reload(self):
        service, _, hardware = self._make_service()
        service.submit_command(ControlCommand(sequence=1, reload_config=True))
        self.assertEqual(hardware.reload_log, 1)


class ProtocolEncodeDecodeTests(unittest.TestCase):
    """Round-trip the wire protocol so radial-composed poses survive encoding."""

    def test_round_trip_ik_pose_command(self):
        cmd = ControlCommand(
            sequence=42,
            mode=ControlMode.IK,
            pose=PoseTarget(0.425, -0.128, -0.07, 12.5),
        )
        wire = encode_command_message(cmd)
        decoded = decode_command_message(wire)
        self.assertEqual(decoded.sequence, 42)
        self.assertEqual(decoded.mode, ControlMode.IK)
        self.assertAlmostEqual(decoded.pose.x, 0.425, places=5)
        self.assertAlmostEqual(decoded.pose.y, -0.128, places=5)
        self.assertAlmostEqual(decoded.pose.z, -0.07, places=5)
        self.assertAlmostEqual(decoded.pose.rot_y_deg, 12.5, places=3)

    def test_round_trip_direct_command_with_flags(self):
        cmd = ControlCommand(
            sequence=7,
            mode=ControlMode.DIRECT,
            direct=DirectCommand(-0.1, 0.2, -0.3, 0.4),
            pause=True,
            reload_config=True,
        )
        wire = encode_command_message(cmd)
        decoded = decode_command_message(wire)
        self.assertEqual(decoded.mode, ControlMode.DIRECT)
        self.assertAlmostEqual(decoded.direct.slew, -0.1, places=5)
        self.assertAlmostEqual(decoded.direct.bucket, 0.4, places=5)
        self.assertTrue(decoded.pause)
        self.assertTrue(decoded.reload_config)
        self.assertFalse(decoded.resume)


if __name__ == "__main__":
    unittest.main()
