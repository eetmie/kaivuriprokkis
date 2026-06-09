"""Radial target-state ↔ IK solver integration tests.

The radial layer composes into Cartesian, which the controller + IK then
solve. These tests drive the full radial delta sequence through IK and
check that the end-effector actually tracks the radial trajectory.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.ik import (  # noqa: E402
    get_state,
    quat_from_axis_angle,
    solve_ik_step,
)
from modules.excavator_target_state import ExcavatorTargetState  # noqa: E402

from tests.common_kinematics import (  # noqa: E402
    fk_from_angles,
    load_robot_config,
    make_ik_config,
)


def _solve_toward(ik_cfg, rc, angles, target_pos, target_rot_y_deg=0.0,
                  max_iters=300, pos_tol=2e-3, dt=0.005):
    """Drive ``solve_ik_step`` (position-only) toward target_pos.

    These radial-integration tests historically used position-mode IK; the
    target orientation argument is accepted but not enforced here.
    """
    del target_rot_y_deg  # unused: kept for back-compat with old call sites
    target = np.asarray(target_pos, dtype=np.float32)

    last_pos = None
    for _ in range(max_iters):
        state = get_state(angles, rc)
        last_pos = np.asarray(state.ee_position, dtype=np.float32)
        if np.linalg.norm(target - last_pos) < pos_tol:
            return angles, last_pos, True
        result = solve_ik_step(angles, target, None, rc, ik_cfg, dt=dt, state=state)
        angles = np.asarray(result.q_next_rad, dtype=np.float32)

    pos, _ = fk_from_angles(angles, rc)
    return angles, pos, np.linalg.norm(target - pos) < pos_tol


class RadialTargetIKTests(unittest.TestCase):
    """Each radial-delta applied through ExcavatorTargetState must be
    reachable by the IK solver (within tolerance)."""

    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def _make_ik(self):
        return make_ik_config(
            method="dls",
            enable_velocity_limiting=False,
            ik_params={"k_val": 0.5, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )

    def _nominal_start_angles(self):
        # Near a realistic mid-workspace posture.
        return np.array([0.0, math.radians(-45.0), math.radians(30.0), 0.0], dtype=np.float32)

    def test_pure_radial_outward_reaches_target(self):
        """Incrementing radius while holding slew/z should extend the EE along +X."""
        ik = self._make_ik()
        angles = self._nominal_start_angles()
        start_pos, _ = fk_from_angles(angles, self.rc)

        state = ExcavatorTargetState.from_cartesian_pose(
            float(start_pos[0]), float(start_pos[1]), float(start_pos[2]), 0.0
        )
        initial_y = state.cartesian_y

        for step in range(5):
            state.apply_radial_delta(0.015, 0.0, 0.0, 0.0)
            target = np.array(state.compose_cartesian_position(), dtype=np.float32)
            angles, reached, ok = _solve_toward(ik, self.rc, angles, target,
                                                pos_tol=3e-3, max_iters=500)
            self.assertTrue(ok, msg=f"radial step {step}: err={np.linalg.norm(target-reached):.4f}")

        # Starting at slew=0 so Y should never drift away from its initial value.
        final_pos, _ = fk_from_angles(angles, self.rc)
        self.assertAlmostEqual(float(final_pos[1]), initial_y, delta=3e-3)

    def test_pure_radial_inward_reaches_target(self):
        """Decrementing radius should retract toward the base along the same bearing."""
        ik = self._make_ik()
        angles = self._nominal_start_angles()
        start_pos, _ = fk_from_angles(angles, self.rc)

        state = ExcavatorTargetState.from_cartesian_pose(
            float(start_pos[0]), float(start_pos[1]), float(start_pos[2]), 0.0
        )

        prev_radius = state.radius
        for step in range(3):
            state.apply_radial_delta(-0.01, 0.0, 0.0, 0.0)
            target = np.array(state.compose_cartesian_position(), dtype=np.float32)
            angles, reached, ok = _solve_toward(ik, self.rc, angles, target,
                                                pos_tol=3e-3, max_iters=500)
            self.assertTrue(ok, msg=f"radial inward step {step}")
            self.assertLess(state.radius, prev_radius)
            prev_radius = state.radius

    def test_small_slew_sweep_keeps_radius_constant(self):
        """Small radial slew sweeps should track an arc with nearly constant radius."""
        ik = self._make_ik()
        angles = self._nominal_start_angles()
        start_pos, _ = fk_from_angles(angles, self.rc)

        state = ExcavatorTargetState.from_cartesian_pose(
            float(start_pos[0]), float(start_pos[1]), float(start_pos[2]), 0.0
        )
        start_radius = state.radius

        # Sweep +/- 30 degrees in 5-degree steps — well inside the region where
        # the analytic Jacobian stays accurate for the IK.
        seen_yaws = []
        for step in range(6):
            state.apply_radial_delta(0.0, 5.0, 0.0, 0.0)
            target = np.array(state.compose_cartesian_position(), dtype=np.float32)
            angles, reached, ok = _solve_toward(ik, self.rc, angles, target,
                                                pos_tol=3e-3, max_iters=500)
            self.assertTrue(ok, msg=f"slew sweep step {step}: err={np.linalg.norm(target-reached):.4f}")

            actual_radius = math.hypot(float(reached[0]), float(reached[1]))
            self.assertAlmostEqual(actual_radius, start_radius, delta=5e-3,
                                   msg=f"radius drift at step {step}: {actual_radius} vs {start_radius}")
            seen_yaws.append(math.degrees(math.atan2(float(reached[1]), float(reached[0]))))

        # Yaws must be (approximately) monotonic.
        for a, b in zip(seen_yaws, seen_yaws[1:]):
            self.assertGreater(b, a - 0.5, msg=f"yaw not monotonic: {seen_yaws}")

    def test_full_360_slew_sweep_reaches_every_radial_target(self):
        """Radial IK should stay stable through a full slew rotation."""
        ik = self._make_ik()
        angles = self._nominal_start_angles()
        start_pos, _ = fk_from_angles(angles, self.rc)

        state = ExcavatorTargetState.from_cartesian_pose(
            float(start_pos[0]), float(start_pos[1]), float(start_pos[2]), 0.0
        )
        start_radius = state.radius
        start_z = state.z

        for yaw_deg in range(0, 361, 30):
            state.set_radial_pose(start_radius, float(yaw_deg), start_z, 0.0)
            target = np.array(state.compose_cartesian_position(), dtype=np.float32)
            angles, reached, ok = _solve_toward(
                ik, self.rc, angles, target,
                pos_tol=4e-3, max_iters=700,
            )
            self.assertTrue(
                ok,
                msg=f"yaw {yaw_deg}: err={np.linalg.norm(target-reached):.4f} target={target} reached={reached}",
            )

            actual_radius = math.hypot(float(reached[0]), float(reached[1]))
            self.assertAlmostEqual(actual_radius, start_radius, delta=6e-3)

    def test_radial_z_change_moves_vertical_only(self):
        """Pure z change in radial space should map to a pure Cartesian Z move."""
        ik = self._make_ik()
        angles = self._nominal_start_angles()
        start_pos, _ = fk_from_angles(angles, self.rc)
        state = ExcavatorTargetState.from_cartesian_pose(
            float(start_pos[0]), float(start_pos[1]), float(start_pos[2]), 0.0
        )

        state.apply_radial_delta(0.0, 0.0, -0.05, 0.0)
        target = np.array(state.compose_cartesian_position(), dtype=np.float32)
        self.assertAlmostEqual(target[0], start_pos[0], places=6)
        self.assertAlmostEqual(target[1], start_pos[1], places=6)
        self.assertAlmostEqual(target[2], start_pos[2] - 0.05, places=6)

        angles, reached, ok = _solve_toward(ik, self.rc, angles, target,
                                            pos_tol=3e-3, max_iters=500)
        self.assertTrue(ok)

    def test_alternating_cartesian_and_radial_stay_consistent(self):
        """Interleaving cartesian and radial deltas must not silently drift state."""
        state = ExcavatorTargetState.from_cartesian_pose(0.5, 0.0, 0.0, 0.0)
        # cartesian +y → then radial +r (along current bearing) → cartesian -y back
        state.apply_cartesian_delta(0.0, 0.1, 0.0, 0.0)
        self.assertAlmostEqual(state.cartesian_y, 0.1, places=6)

        state.apply_radial_delta(0.1, 0.0, 0.0, 0.0)
        # Radial outward along current bearing; radius increased by 0.1
        expected_radius = math.hypot(0.5, 0.1) + 0.1
        self.assertAlmostEqual(state.radius, expected_radius, places=5)

    def test_radial_then_cartesian_starts_from_updated_bearing(self):
        """A cartesian delta after a radial slew uses updated bearing (not original)."""
        state = ExcavatorTargetState.from_cartesian_pose(0.5, 0.0, 0.0, 0.0)
        state.apply_radial_delta(0.0, 90.0, 0.0, 0.0)  # slew to face +Y
        self.assertAlmostEqual(state.cartesian_y, 0.5, places=5)
        self.assertAlmostEqual(state.cartesian_x, 0.0, places=5)

        state.apply_cartesian_delta(0.1, 0.0, 0.0, 0.0)  # +x in world
        # Cartesian delta is in world frame, independent of slew.
        self.assertAlmostEqual(state.cartesian_x, 0.1, places=6)
        self.assertAlmostEqual(state.cartesian_y, 0.5, places=6)


class RadialTargetEdgeCaseTests(unittest.TestCase):
    """Edge cases that caused surprising behavior in prior iterations."""

    def test_zero_radius_preserves_slew_yaw_when_reset_via_cartesian(self):
        """When shrinking radius through origin, slew_yaw_deg snaps to 0 per implementation."""
        state = ExcavatorTargetState()
        state.set_radial_pose(0.5, 30.0, 0.0, 0.0)
        state.set_cartesian_pose(0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(state.radius, 0.0, places=9)
        self.assertAlmostEqual(state.slew_yaw_deg, 0.0, places=9)

    def test_delta_accumulation_equivalent_to_single_large_delta(self):
        """Ten small radial slew deltas must equal one large slew delta."""
        stateA = ExcavatorTargetState.from_cartesian_pose(0.4, 0.0, 0.0, 0.0)
        stateB = ExcavatorTargetState.from_cartesian_pose(0.4, 0.0, 0.0, 0.0)
        for _ in range(10):
            stateA.apply_radial_delta(0.0, 3.0, 0.0, 0.0)
        stateB.apply_radial_delta(0.0, 30.0, 0.0, 0.0)

        self.assertAlmostEqual(stateA.radius, stateB.radius, places=6)
        self.assertAlmostEqual(stateA.cartesian_x, stateB.cartesian_x, places=6)
        self.assertAlmostEqual(stateA.cartesian_y, stateB.cartesian_y, places=6)
        self.assertAlmostEqual(stateA.slew_yaw_deg, stateB.slew_yaw_deg, places=6)


class MixedRadialIntegrationTests(unittest.TestCase):
    """Realistic operator inputs: combined radius/slew/z deltas through IK."""

    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def _make_ik(self):
        return make_ik_config(
            method="dls",
            enable_velocity_limiting=False,
            ik_params={"k_val": 0.5, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )

    def test_combined_radius_slew_z_delta_reaches_composed_target(self):
        ik = self._make_ik()
        angles = np.array([0.0, math.radians(-45), math.radians(30), 0.0],
                          dtype=np.float32)
        start_pos, _ = fk_from_angles(angles, self.rc)

        state = ExcavatorTargetState.from_cartesian_pose(
            float(start_pos[0]), float(start_pos[1]), float(start_pos[2]), 0.0
        )

        # Operator: extend a bit, swing right, drop down — all in one go.
        state.apply_radial_delta(0.02, 10.0, -0.02, 0.0)
        target = np.array(state.compose_cartesian_position(), dtype=np.float32)
        angles, reached, ok = _solve_toward(ik, self.rc, angles, target,
                                            pos_tol=3e-3, max_iters=600)
        self.assertTrue(ok, msg=f"combined delta unreachable: err="
                                f"{np.linalg.norm(target-reached):.4f}")

        # Cartesian view must agree with the radial view we asked for.
        self.assertAlmostEqual(math.hypot(float(reached[0]), float(reached[1])),
                               state.radius, delta=5e-3)
        self.assertAlmostEqual(float(reached[2]), state.z, delta=4e-3)

    def test_radial_state_radius_can_grow_through_workspace(self):
        """Outward radius accumulation along multiple steps stays self-consistent."""
        state = ExcavatorTargetState.from_cartesian_pose(0.40, 0.0, 0.0, 0.0)
        radii = [state.radius]
        for _ in range(8):
            state.apply_radial_delta(0.01, 0.0, 0.0, 0.0)
            radii.append(state.radius)
            # cartesian_y must still be zero (slew unchanged).
            self.assertAlmostEqual(state.cartesian_y, 0.0, places=6)
        # Strictly monotonic.
        for a, b in zip(radii, radii[1:]):
            self.assertGreater(b, a)
        self.assertAlmostEqual(radii[-1], 0.40 + 0.01 * 8, places=5)


if __name__ == "__main__":
    unittest.main()
