"""Tests for the radial/cartesian IK target-state layer.

Exercises ExcavatorTargetState invariants (radius/slew/z representation,
round-trips, clamping) and the InputHandler radial key/gamepad mapping.
"""

import math
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from clients.input_handler import InputHandler  # type: ignore  # noqa: E402
    _HAS_INPUT_HANDLER = True
except Exception:  # client side stripped from this checkout
    InputHandler = None  # type: ignore
    _HAS_INPUT_HANDLER = False
from modules.excavator_target_state import ExcavatorTargetState, IKControlSpace  # noqa: E402


WORKSPACE_LIMITS = {
    "x_min": 0.350,
    "x_max": 0.680,
    "y_min": -0.150,
    "y_max": 0.150,
    "z_min": -0.300,
    "z_max": 0.150,
}


class TargetStateConversionTests(unittest.TestCase):
    def _assert_pose_close(self, left, right, places=6):
        self.assertEqual(len(left), len(right))
        for lval, rval in zip(left, right):
            self.assertAlmostEqual(lval, rval, places=places)

    def test_cartesian_updates_radial_view(self):
        state = ExcavatorTargetState.from_cartesian_pose(0.4, 0.3, -0.1, 12.5)
        self.assertAlmostEqual(state.radius, 0.5, places=6)
        self.assertAlmostEqual(state.slew_yaw_deg, math.degrees(math.atan2(0.3, 0.4)), places=6)
        self.assertAlmostEqual(state.rot_y_deg, 12.5, places=6)

    def test_radial_updates_cartesian_view(self):
        state = ExcavatorTargetState()
        state.set_radial_pose(0.5, 30.0, -0.1, 7.5)
        self.assertAlmostEqual(state.cartesian_x, 0.5 * math.cos(math.radians(30.0)), places=6)
        self.assertAlmostEqual(state.cartesian_y, 0.5 * math.sin(math.radians(30.0)), places=6)
        self.assertAlmostEqual(state.z, -0.1, places=6)
        self.assertAlmostEqual(state.rot_y_deg, 7.5, places=6)

    def test_radial_round_trip_is_idempotent(self):
        for x, y, z, rot in [
            (0.55, -0.12, -0.08, -7.0),
            (0.30, 0.00, 0.10, 0.0),
            (0.40, 0.40, -0.20, 30.0),
            (-0.20, 0.35, 0.05, -15.0),
        ]:
            state = ExcavatorTargetState.from_cartesian_pose(x, y, z, rot)
            original = state.compose_cartesian_pose()
            state.set_radial_pose(*state.compose_radial_pose())
            self._assert_pose_close(state.compose_cartesian_pose(), original)

    def test_radius_zero_collapses_to_origin(self):
        state = ExcavatorTargetState()
        state.set_cartesian_pose(0.0, 0.0, 0.2, 0.0)
        self.assertAlmostEqual(state.radius, 0.0, places=9)
        self.assertAlmostEqual(state.slew_yaw_deg, 0.0, places=9)  # defined as 0 for r≈0

    def test_negative_radius_is_clamped_to_zero(self):
        state = ExcavatorTargetState()
        state.set_radial_pose(-0.2, 30.0, 0.0, 0.0)
        self.assertAlmostEqual(state.radius, 0.0, places=9)
        self.assertAlmostEqual(state.cartesian_x, 0.0, places=6)
        self.assertAlmostEqual(state.cartesian_y, 0.0, places=6)

    def test_slew_wraps_to_normalized_range(self):
        state = ExcavatorTargetState()
        state.set_radial_pose(0.5, 540.0, 0.0, 0.0)  # 540 -> wrap to 180
        self.assertAlmostEqual(state.slew_yaw_deg, 180.0, places=3)
        # -540 -> wrap to +180 per the implementation's stable representation
        state.set_radial_pose(0.5, -540.0, 0.0, 0.0)
        self.assertAlmostEqual(abs(state.slew_yaw_deg), 180.0, places=3)

    def test_slew_wrap_consistent_with_cartesian(self):
        """After yaw wrap, the cartesian view must match the wrapped angle."""
        state = ExcavatorTargetState()
        state.set_radial_pose(0.5, 540.0, 0.0, 0.0)  # wraps to 180
        self.assertAlmostEqual(state.cartesian_x, -0.5, places=5)
        self.assertAlmostEqual(state.cartesian_y, 0.0, places=5)

    def test_radial_set_clears_y_when_yaw_zero(self):
        state = ExcavatorTargetState.from_cartesian_pose(0.5, 0.3, 0.0, 0.0)
        state.set_radial_pose(0.5, 0.0, 0.0, 0.0)
        # +y component should now be zero (yaw=0 means cart along +x).
        self.assertAlmostEqual(state.cartesian_y, 0.0, places=6)
        self.assertAlmostEqual(state.cartesian_x, 0.5, places=6)

    def test_negative_x_yaw_resolves_to_180(self):
        state = ExcavatorTargetState.from_cartesian_pose(-0.4, 0.0, 0.0, 0.0)
        # atan2(0, -0.4) = pi -> 180 deg
        self.assertAlmostEqual(state.slew_yaw_deg, 180.0, places=3)
        self.assertAlmostEqual(state.radius, 0.4, places=6)

    def test_zero_delta_is_noop(self):
        state = ExcavatorTargetState.from_cartesian_pose(0.4, 0.1, -0.05, 5.0)
        before = state.compose_cartesian_pose()
        state.apply_radial_delta(0.0, 0.0, 0.0, 0.0)
        state.apply_cartesian_delta(0.0, 0.0, 0.0, 0.0)
        after = state.compose_cartesian_pose()
        for a, b in zip(before, after):
            self.assertAlmostEqual(a, b, places=9)


class TargetStateDeltaTests(unittest.TestCase):
    def test_pure_slew_delta_preserves_radius(self):
        state = ExcavatorTargetState.from_cartesian_pose(0.5, 0.0, 0.1, 0.0)
        for step in range(18):
            state.apply_radial_delta(0.0, 5.0, 0.0, 0.0)
            self.assertAlmostEqual(state.radius, 0.5, places=5,
                                   msg=f"radius drifted at step {step}: {state.radius}")
            self.assertAlmostEqual(state.z, 0.1, places=5)

    def test_pure_radius_delta_preserves_slew(self):
        state = ExcavatorTargetState()
        state.set_radial_pose(0.5, 45.0, -0.1, 0.0)
        initial_slew = state.slew_yaw_deg
        state.apply_radial_delta(0.1, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(state.slew_yaw_deg, initial_slew, places=5)
        self.assertAlmostEqual(state.radius, 0.6, places=5)

    def test_cartesian_delta_keeps_radial_consistent(self):
        state = ExcavatorTargetState.from_cartesian_pose(0.5, 0.0, 0.1, 0.0)
        state.apply_cartesian_delta(0.0, 0.1, 0.0, 0.0)
        self.assertAlmostEqual(state.cartesian_x, 0.5, places=6)
        self.assertAlmostEqual(state.cartesian_y, 0.1, places=6)
        self.assertAlmostEqual(state.radius, math.hypot(0.5, 0.1), places=6)
        self.assertAlmostEqual(state.slew_yaw_deg, math.degrees(math.atan2(0.1, 0.5)), places=6)

    def test_large_slew_delta_accumulates_monotonically(self):
        state = ExcavatorTargetState()
        state.set_radial_pose(0.4, 0.0, 0.0, 0.0)
        yaws = []
        for _ in range(30):
            state.apply_radial_delta(0.0, 10.0, 0.0, 0.0)
            yaws.append(state.slew_yaw_deg)
        # Unwrap to be monotonically increasing.
        unwrapped = yaws[:1]
        for y in yaws[1:]:
            prev = unwrapped[-1]
            while y < prev - 180.0:
                y += 360.0
            while y > prev + 180.0:
                y -= 360.0
            unwrapped.append(y)
        for a, b in zip(unwrapped, unwrapped[1:]):
            self.assertGreater(b, a - 1e-6, msg="slew should be monotonically non-decreasing")
        # 30 * 10 degrees starting from 0 ≈ 300 degrees.
        self.assertAlmostEqual(unwrapped[-1], 300.0, places=2)

    def test_clamp_cartesian_resyncs_radial_state(self):
        state = ExcavatorTargetState.from_cartesian_pose(0.6, 0.0, 0.0, 0.0)
        state.apply_radial_delta(0.2, 90.0, -0.5, 60.0)  # forces out-of-workspace values
        state.clamp_cartesian(WORKSPACE_LIMITS)

        pose = state.compose_cartesian_pose()
        self.assertAlmostEqual(pose[0], 0.35, places=6)
        self.assertAlmostEqual(pose[1], 0.15, places=6)
        self.assertAlmostEqual(pose[2], -0.3, places=6)
        self.assertAlmostEqual(pose[3], 45.0, places=6)
        self.assertAlmostEqual(state.radius, math.hypot(0.35, 0.15), places=6)
        self.assertAlmostEqual(state.slew_yaw_deg, math.degrees(math.atan2(0.15, 0.35)), places=6)

    def test_clamp_cartesian_at_negative_y_min(self):
        """Negative Y should clamp to y_min and the radial view must follow."""
        state = ExcavatorTargetState.from_cartesian_pose(0.4, -0.5, 0.0, 0.0)
        state.clamp_cartesian(WORKSPACE_LIMITS)
        self.assertAlmostEqual(state.cartesian_y, -0.15, places=6)
        self.assertAlmostEqual(state.slew_yaw_deg,
                               math.degrees(math.atan2(-0.15, 0.4)), places=6)

    def test_clamp_cartesian_keeps_in_range_pose_intact(self):
        """A pose strictly inside the workspace should be untouched by clamp."""
        state = ExcavatorTargetState.from_cartesian_pose(0.4, 0.1, -0.05, 10.0)
        before = state.compose_cartesian_pose()
        state.clamp_cartesian(WORKSPACE_LIMITS)
        after = state.compose_cartesian_pose()
        for a, b in zip(before, after):
            self.assertAlmostEqual(a, b, places=6)

    def test_rot_y_clamp_respects_custom_limits(self):
        state = ExcavatorTargetState.from_cartesian_pose(0.4, 0.0, 0.0, 90.0)
        state.clamp_cartesian(WORKSPACE_LIMITS, min_rot_y_deg=-30.0, max_rot_y_deg=30.0)
        self.assertAlmostEqual(state.rot_y_deg, 30.0, places=6)

    def test_copy_is_deep_enough_for_independent_edits(self):
        a = ExcavatorTargetState.from_cartesian_pose(0.4, 0.0, 0.0, 0.0)
        b = a.copy()
        b.apply_radial_delta(0.1, 30.0, 0.0, 10.0)
        self.assertAlmostEqual(a.radius, 0.4, places=6)
        self.assertNotAlmostEqual(b.radius, 0.4, places=3)


@unittest.skipUnless(_HAS_INPUT_HANDLER, "clients.input_handler not available in this checkout")
class InputHandlerRadialTests(unittest.TestCase):
    def test_radial_keys_map_to_expected_deltas(self):
        handler = InputHandler()

        dr, d_slew, dz, drot = handler.tick_ik(
            {"w", "d", "q", "r"},
            controller_state=None,
            step_pos=0.01,
            step_rot=2.0,
            control_space=IKControlSpace.RADIAL,
        )
        # w → +dr, d → +slew, q → -dz, r → +rot; slew step preserves arc length.
        self.assertAlmostEqual(dr, 0.01, places=6)
        self.assertAlmostEqual(d_slew, math.degrees(0.01 / 0.6), places=6)
        self.assertAlmostEqual(dz, -0.01, places=6)
        self.assertAlmostEqual(drot, 2.0, places=6)

    def test_radial_s_key_produces_negative_radius_delta(self):
        handler = InputHandler()
        dr, d_slew, dz, drot = handler.tick_ik(
            {"s"},
            controller_state=None,
            step_pos=0.01,
            step_rot=2.0,
            control_space=IKControlSpace.RADIAL,
        )
        self.assertEqual((dr, d_slew, dz, drot), (-0.01, 0.0, 0.0, 0.0))

    def test_radial_shift_multiplies_step_sizes(self):
        handler = InputHandler()
        dr, d_slew, dz, drot = handler.tick_ik(
            {"w", "shift"},
            controller_state=None,
            step_pos=0.01,
            step_rot=2.0,
            control_space=IKControlSpace.RADIAL,
        )
        # fast multiplier is 5x
        self.assertAlmostEqual(dr, 0.05, places=6)
        self.assertEqual(d_slew, 0.0)

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

    def test_radial_gamepad_requires_left_bumper(self):
        handler = InputHandler()
        # Left bumper NOT held: gamepad axes should be ignored.
        controller_state = {
            "LeftBumper": False,
            "LeftJoystickX": 1.0,
            "LeftJoystickY": 1.0,
            "RightJoystickX": 0.0,
            "RightJoystickY": 0.0,
            "LeftTrigger": 0.0,
            "RightTrigger": 0.0,
        }
        dr, d_slew, dz, drot = handler.tick_ik(
            set(),
            controller_state=controller_state,
            step_pos=0.01,
            step_rot=2.0,
            control_space=IKControlSpace.RADIAL,
        )
        self.assertEqual((dr, d_slew, dz, drot), (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
