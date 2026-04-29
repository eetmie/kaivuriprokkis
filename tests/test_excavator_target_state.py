import math
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from clients.input_handler import InputHandler  # type: ignore
    _HAS_INPUT_HANDLER = True
except Exception:
    InputHandler = None  # type: ignore
    _HAS_INPUT_HANDLER = False
from modules.excavator_target_state import ExcavatorTargetState, IKControlSpace


WORKSPACE_LIMITS = {
    "x_min": 0.350,
    "x_max": 0.680,
    "y_min": -0.150,
    "y_max": 0.150,
    "z_min": -0.300,
    "z_max": 0.150,
}


class ExcavatorTargetStateTests(unittest.TestCase):
    def assertPoseAlmostEqual(self, left, right, places=6):
        self.assertEqual(len(left), len(right))
        for lval, rval in zip(left, right):
            self.assertAlmostEqual(lval, rval, places=places)

    def test_cartesian_pose_updates_radial_view(self):
        state = ExcavatorTargetState.from_cartesian_pose(0.4, 0.3, -0.1, 12.5)

        self.assertAlmostEqual(state.radius, 0.5, places=6)
        self.assertAlmostEqual(state.slew_yaw_deg, math.degrees(math.atan2(0.3, 0.4)), places=6)
        self.assertAlmostEqual(state.rot_y_deg, 12.5, places=6)

    def test_radial_round_trip_does_not_move_target(self):
        state = ExcavatorTargetState.from_cartesian_pose(0.55, -0.12, -0.08, -7.0)
        original_pose = state.compose_cartesian_pose()

        state.set_radial_pose(*state.compose_radial_pose())

        self.assertPoseAlmostEqual(state.compose_cartesian_pose(), original_pose)

    def test_clamp_cartesian_resyncs_radial_state(self):
        state = ExcavatorTargetState.from_cartesian_pose(0.6, 0.0, 0.0, 0.0)
        state.apply_radial_delta(0.2, 90.0, -0.5, 60.0)
        state.clamp_cartesian(WORKSPACE_LIMITS)

        pose = state.compose_cartesian_pose()
        self.assertPoseAlmostEqual(pose, (0.35, 0.15, -0.3, 45.0))
        self.assertAlmostEqual(state.radius, math.hypot(0.35, 0.15), places=6)
        self.assertAlmostEqual(state.slew_yaw_deg, math.degrees(math.atan2(0.15, 0.35)), places=6)


@unittest.skipUnless(_HAS_INPUT_HANDLER, "clients.input_handler not available in this checkout")
class InputHandlerTests(unittest.TestCase):
    def test_radial_keyboard_mapping(self):
        handler = InputHandler()

        dr, d_slew, dz, drot = handler.tick_ik(
            {"w", "d", "q", "r"},
            controller_state=None,
            step_pos=0.01,
            step_rot=2.0,
            control_space=IKControlSpace.RADIAL,
        )

        self.assertAlmostEqual(dr, 0.01, places=6)
        self.assertAlmostEqual(d_slew, math.degrees(0.01 / 0.6), places=6)
        self.assertAlmostEqual(dz, -0.01, places=6)
        self.assertAlmostEqual(drot, 2.0, places=6)

    def test_cartesian_keyboard_mapping_unchanged(self):
        handler = InputHandler()

        dx, dy, dz, drot = handler.tick_ik(
            {"a", "e", "w", "f"},
            controller_state=None,
            step_pos=0.01,
            step_rot=2.0,
            control_space=IKControlSpace.CARTESIAN,
        )

        self.assertEqual((dx, dy, dz, drot), (-0.01, 0.01, 0.01, -2.0))


if __name__ == "__main__":
    unittest.main()
