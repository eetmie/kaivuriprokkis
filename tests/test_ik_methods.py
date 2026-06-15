"""Inverse kinematics convergence tests across all solver methods.

For each of pinv, svd, trans, dls we confirm that iteratively applying
``solve_ik_step`` drives the end-effector toward a Cartesian target. We
also exercise joint-limit avoidance, velocity limiting, the body-frame
transform, and pose-mode convergence across the full slew range.
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
    compute_pose_error,
    extract_axis_rotation,
    get_state,
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    solve_ik_step,
)

from tests.common_kinematics import (  # noqa: E402
    fk_from_angles,
    load_robot_config,
    make_ik_config,
    simulate_ik,
)


POSITION_TOL_M = 3e-3


def _nominal_start_angles() -> np.ndarray:
    return np.array(
        [0.0, math.radians(-45.0), math.radians(30.0), 0.0], dtype=np.float32
    )


class IKConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def _run_method(self, method: str, offset, max_iters: int = 800):
        ik_params = {
            "k_val": 0.5 if method != "trans" else 0.05,
            "min_singular_value": 1e-4,
            "lambda_val": 0.02,
        }
        cfg = make_ik_config(method=method, ik_params=ik_params,
                             enable_velocity_limiting=False)
        start = _nominal_start_angles()
        start_pos, _ = fk_from_angles(start, self.rc)
        target = start_pos + np.asarray(offset, dtype=np.float32)
        result = simulate_ik(self.rc, cfg, start, target,
                             max_iters=max_iters, pos_tol=POSITION_TOL_M, dt=0.005)
        return result, target

    def test_pinv_converges(self):
        result, target = self._run_method("pinv", [0.02, 0.01, -0.01])
        self.assertTrue(result["converged"],
                        msg=f"pinv final err={result['pos_error']:.4f} target={target}")

    def test_svd_converges(self):
        result, _ = self._run_method("svd", [0.02, 0.01, -0.01])
        self.assertTrue(result["converged"],
                        msg=f"svd final err={result['pos_error']:.4f}")

    def test_transpose_converges(self):
        # Jacobian-transpose with smaller gain needs more iterations.
        result, _ = self._run_method("trans", [0.015, 0.005, -0.01], max_iters=4000)
        self.assertTrue(result["converged"],
                        msg=f"trans final err={result['pos_error']:.4f}")

    def test_dls_converges(self):
        result, _ = self._run_method("dls", [0.02, 0.01, -0.015])
        self.assertTrue(result["converged"],
                        msg=f"dls final err={result['pos_error']:.4f}")


class IKFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def test_velocity_limit_bounds_per_iteration_delta(self):
        """Per-cycle joint delta must not exceed ``max_joint_velocity_rad_per_step``."""
        max_rad_per_iter = [0.02, 0.02, 0.02, 0.02]
        cfg = make_ik_config(
            method="dls",
            enable_velocity_limiting=True,
            max_joint_velocities=max_rad_per_iter,
            ik_params={"k_val": 1.0, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )

        angles = _nominal_start_angles()
        start_pos, _ = fk_from_angles(angles, self.rc)
        target = start_pos + np.array([0.4, 0.3, -0.3], dtype=np.float32)
        target_quat = quat_from_axis_angle(np.array([0, 1, 0], np.float32), np.float32(0.0))

        result = solve_ik_step(angles, target, target_quat, self.rc, cfg, dt=0.005)
        delta = np.abs(result.dq_rad)
        for i, limit in enumerate(max_rad_per_iter):
            self.assertLessEqual(float(delta[i]), limit + 1e-5,
                                 msg=f"joint {i} delta {delta[i]} exceeds limit {limit}")

    def test_joint_limit_avoidance_repels_from_upper_bound(self):
        """Near the upper limit, the repulsion term should push the joint downward."""
        joint_limits = [
            (-math.pi, math.pi),
            (-0.5, 0.2),
            (-math.pi, math.pi),
            (-math.pi, math.pi),
        ]
        cfg = make_ik_config(
            method="dls", joint_limits=joint_limits,
            enable_joint_limit_avoidance=True,
            enable_velocity_limiting=False,
            ik_params={"k_val": 0.0, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )

        # Zero IK task (k_val=0 kills task contribution) so only repulsion drives delta.
        angles = np.array([0.0, 0.19, 0.0, 0.0], dtype=np.float32)
        pos, quat = fk_from_angles(angles, self.rc)
        result = solve_ik_step(angles, pos, quat, self.rc, cfg, dt=0.005)
        self.assertLess(float(result.q_next_rad[1]), float(angles[1]),
                        msg=f"joint-limit avoidance did not push inward: "
                            f"{result.q_next_rad[1]} vs {angles[1]}")

    def test_joint_limit_avoidance_at_lower_boundary_pushes_outward(self):
        joint_limits = [
            (-math.pi, math.pi),
            (-0.5, 0.2),
            (-math.pi, math.pi),
            (-math.pi, math.pi),
        ]
        cfg = make_ik_config(
            method="dls",
            joint_limits=joint_limits,
            enable_joint_limit_avoidance=True,
            enable_velocity_limiting=False,
            ik_params={"k_val": 0.0, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )
        angles = np.array([0.0, -0.49, 0.0, 0.0], dtype=np.float32)
        pos, quat = fk_from_angles(angles, self.rc)
        result = solve_ik_step(angles, pos, quat, self.rc, cfg, dt=0.005)
        self.assertGreater(float(result.q_next_rad[1]), float(angles[1]),
                           msg="lower-limit avoidance must push joint upward")

    def test_unbounded_joint_passes_through_clamp(self):
        """``None`` sentinel for a joint must skip both clamp and avoidance."""
        joint_limits = [
            None,                # slew unbounded
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
        ]
        cfg = make_ik_config(
            method="dls",
            enable_velocity_limiting=False,
            joint_limits=joint_limits,
            enable_joint_limit_avoidance=True,
            ik_params={"k_val": 0.5, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )

        angles = np.array([3.5, 0.0, 0.0, 0.0], dtype=np.float32)  # slew past +π
        pos, quat = fk_from_angles(angles, self.rc)
        target = pos + np.array([0.01, 0.0, 0.0], dtype=np.float32)
        result = solve_ik_step(angles, target, quat, self.rc, cfg, dt=0.005)

        self.assertTrue(np.isfinite(float(result.q_next_rad[0])))
        for i in (1, 2, 3):
            self.assertGreaterEqual(float(result.q_next_rad[i]), -1.0 - 1e-6)
            self.assertLessEqual(float(result.q_next_rad[i]), 1.0 + 1e-6)

    def test_q_next_clamped_to_joint_limits(self):
        """A massive position request must still be clamped to per-joint limits."""
        joint_limits = [(-0.10, 0.10), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        cfg = make_ik_config(
            method="dls",
            enable_velocity_limiting=False,
            joint_limits=joint_limits,
            enable_joint_limit_avoidance=False,
            ik_params={"k_val": 5.0, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )

        angles = np.zeros(self.rc.num_joints, dtype=np.float32)
        pos, quat = fk_from_angles(angles, self.rc)
        target = pos + np.array([1.0, 1.0, 0.0], dtype=np.float32)
        result = solve_ik_step(angles, target, quat, self.rc, cfg, dt=0.005)
        for i, (q_min, q_max) in enumerate(joint_limits):
            self.assertGreaterEqual(float(result.q_next_rad[i]), q_min - 1e-6)
            self.assertLessEqual(float(result.q_next_rad[i]), q_max + 1e-6)

    def test_condition_number_populated_each_call(self):
        cfg = make_ik_config(method="dls")
        angles = np.array(
            [0.0, math.radians(-30.0), math.radians(20.0), 0.0], dtype=np.float32
        )
        pos, quat = fk_from_angles(angles, self.rc)
        result = solve_ik_step(angles, pos, quat, self.rc, cfg, dt=0.005)
        self.assertTrue(np.isfinite(result.state.condition_number))
        self.assertGreater(result.state.condition_number, 0.0)

    def test_state_argument_is_respected(self):
        """If a precomputed state is passed it should be propagated into IKResult.state."""
        cfg = make_ik_config(method="dls")
        angles = np.array([0.1, -0.3, 0.4, -0.2], dtype=np.float32)
        state = get_state(angles, self.rc)
        pos = state.ee_position
        result = solve_ik_step(angles, pos, state.ee_orientation, self.rc, cfg,
                               dt=0.005, state=state)
        self.assertIs(result.state, state)

    def test_singular_gate_rejects(self):
        """At a deeply singular pose with a low cond threshold, the solver flags rejection."""
        cfg = make_ik_config(
            method="dls",
            condition_number_threshold=10.0,
            ik_params={"k_val": 0.5, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )
        # Fully-extended pose is singular.
        angles = np.zeros(self.rc.num_joints, dtype=np.float32)
        pos, quat = fk_from_angles(angles, self.rc)
        result = solve_ik_step(angles, pos + np.array([0.01, 0, 0], np.float32),
                               quat, self.rc, cfg, dt=0.005)
        self.assertTrue(result.rejected)
        self.assertIsNotNone(result.reason)
        # When rejected, dq is zero and q_next == q (after clamping, which is a no-op here).
        self.assertTrue(np.allclose(result.dq_rad, 0.0))


class IKPoseModeTests(unittest.TestCase):
    """Pose-mode IK matches the production controller's target-quat recipe.

    The controller composes target_quat as ``slew_quat * pitch_quat`` so that
    yaw error cancels by construction. Pitch is body-frame (around the
    upper-body Y axis after the slew rotation).
    """

    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def _make_pose_cfg(self):
        return make_ik_config(
            method="dls",
            enable_velocity_limiting=False,
            enable_adaptive_damping=False,
            enable_joint_limit_avoidance=False,
            ik_params={"k_val": 0.5, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )

    def _make_target_quat(self, slew_rad, total_pitch_rad):
        slew_q = quat_from_axis_angle(np.array([0, 0, 1], np.float32),
                                      np.float32(slew_rad))
        pitch_q = quat_from_axis_angle(np.array([0, 1, 0], np.float32),
                                       np.float32(total_pitch_rad))
        return quat_normalize(quat_multiply(slew_q, pitch_q))

    def _solve_to_pose(self, cfg, start_angles, target_pos, target_quat,
                       max_iters=600, pos_tol=3e-3, ang_tol=math.radians(1.0),
                       dt=0.005):
        angles = np.asarray(start_angles, dtype=np.float32).copy()
        for _ in range(max_iters):
            state = get_state(angles, self.rc)
            pe, ae = compute_pose_error(state.ee_position, state.ee_orientation,
                                        target_pos, target_quat)
            if np.linalg.norm(pe) < pos_tol and np.linalg.norm(ae) < ang_tol:
                return angles, True
            result = solve_ik_step(angles, target_pos, target_quat, self.rc, cfg,
                                   dt=dt, state=state)
            angles = np.asarray(result.q_next_rad, dtype=np.float32)
        return angles, False

    def _start_angles(self, slew_rad):
        return np.array(
            [slew_rad, math.radians(-45), math.radians(30), math.radians(-15)],
            dtype=np.float32,
        )

    def _target_for_pitch_offset(self, slew_rad, pitch_offset_rad):
        start = self._start_angles(slew_rad)
        pos, q = fk_from_angles(start, self.rc)
        slew_q = quat_from_axis_angle(np.array([0, 0, 1], np.float32),
                                      np.float32(slew_rad))
        body_q = quat_normalize(quat_multiply(quat_conjugate(slew_q), q))
        cur_pitch = extract_axis_rotation(body_q, np.array([0, 1, 0], np.float32))
        target_quat = self._make_target_quat(slew_rad, cur_pitch + pitch_offset_rad)
        return start, pos, np.asarray(target_quat, np.float32)

    def test_pose_mode_converges_at_zero_slew(self):
        cfg = self._make_pose_cfg()
        start, pos0, target_quat = self._target_for_pitch_offset(0.0, math.radians(5.0))
        _, ok = self._solve_to_pose(cfg, start, pos0.astype(np.float32), target_quat)
        self.assertTrue(ok)

    def test_pose_mode_converges_at_quarter_slew(self):
        cfg = self._make_pose_cfg()
        start, pos0, target_quat = self._target_for_pitch_offset(
            math.radians(45.0), math.radians(5.0)
        )
        _, ok = self._solve_to_pose(cfg, start, pos0.astype(np.float32), target_quat,
                                    max_iters=800)
        self.assertTrue(ok)

    def test_pose_mode_converges_at_pi_over_two_slew(self):
        cfg = self._make_pose_cfg()
        start, pos0, target_quat = self._target_for_pitch_offset(
            math.pi / 2, math.radians(5.0)
        )
        _, ok = self._solve_to_pose(cfg, start, pos0.astype(np.float32), target_quat,
                                    max_iters=800)
        self.assertTrue(ok)

    def test_pose_mode_converges_at_full_slew_sweep(self):
        for slew_deg in (-150, -90, -45, 0, 45, 90, 135, 180):
            cfg = self._make_pose_cfg()
            start, pos0, target_quat = self._target_for_pitch_offset(
                math.radians(slew_deg), math.radians(4.0)
            )
            _, ok = self._solve_to_pose(cfg, start, pos0.astype(np.float32),
                                        target_quat, max_iters=800)
            self.assertTrue(ok, msg=f"pose IK failed at slew={slew_deg}deg")


if __name__ == "__main__":
    unittest.main()
