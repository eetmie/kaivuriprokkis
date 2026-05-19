"""Inverse kinematics convergence tests across all solver methods.

For each of pinv, svd, trans, dls we confirm that iteratively applying
IKController.compute() drives the end-effector toward a Cartesian target.
We also exercise joint-limit avoidance, velocity limiting, and the
base-frame transform flag.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.differential_ik import IKController  # noqa: E402
from modules.excavator_ik_utils import compute_relative_joint_angles  # noqa: E402
from modules.quaternion_math import (  # noqa: E402
    quat_from_axis_angle, quat_multiply, quat_normalize,
)

from tests.common_kinematics import (  # noqa: E402
    build_absolute_quaternions,
    fk_from_angles,
    load_robot_config,
    make_ik_config,
    simulate_ik,
)


# Tolerance used for near-pose convergence in deep iteration simulations.
POSITION_TOL_M = 3e-3


class IKConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def _nominal_start_angles(self) -> np.ndarray:
        return np.array([0.0, math.radians(-45.0), math.radians(30.0), 0.0], dtype=np.float32)

    def _run_method(self, method: str, offset: np.ndarray, max_iters: int = 800):
        ik_params = {
            "k_val": 0.5 if method != "trans" else 0.05,
            "min_singular_value": 1e-4,
            "lambda_val": 0.02,
        }
        cfg = make_ik_config(method=method, ik_params=ik_params,
                             enable_velocity_limiting=False)
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)

        start = self._nominal_start_angles()
        start_pos, _ = fk_from_angles(start, self.rc)
        target = start_pos + np.asarray(offset, dtype=np.float32)
        result = simulate_ik(self.rc, ik, start, target,
                             max_iters=max_iters, pos_tol=POSITION_TOL_M, dt=0.005)
        return result, target

    def test_pinv_converges(self):
        result, target = self._run_method("pinv", [0.02, 0.01, -0.01])
        self.assertTrue(result["converged"],
                        msg=f"pinv final err={result['pos_error']:.4f} pos={result['final_pos']} target={target}")

    def test_svd_converges(self):
        result, target = self._run_method("svd", [0.02, 0.01, -0.01])
        self.assertTrue(result["converged"],
                        msg=f"svd final err={result['pos_error']:.4f}")

    def test_transpose_converges(self):
        # Jacobian-transpose with smaller gain needs more iterations.
        result, target = self._run_method("trans", [0.015, 0.005, -0.01], max_iters=4000)
        self.assertTrue(result["converged"],
                        msg=f"trans final err={result['pos_error']:.4f}")

    def test_dls_converges(self):
        result, target = self._run_method("dls", [0.02, 0.01, -0.015])
        self.assertTrue(result["converged"],
                        msg=f"dls final err={result['pos_error']:.4f}")


class IKFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def test_velocity_limit_bounds_per_iteration_delta(self):
        """Per-cycle joint delta must not exceed the configured max_joint_velocities."""
        max_rad_per_iter = [0.02, 0.02, 0.02, 0.02]
        cfg = make_ik_config(method="dls",
                             enable_velocity_limiting=True,
                             max_joint_velocities=max_rad_per_iter,
                             ik_params={"k_val": 1.0, "min_singular_value": 1e-4, "lambda_val": 0.02})
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)

        angles = np.array([0.0, math.radians(-45.0), math.radians(30.0), 0.0], dtype=np.float32)
        start_pos, _ = fk_from_angles(angles, self.rc)
        # Large target to saturate the velocity limiter.
        target = start_pos + np.array([0.4, 0.3, -0.3], dtype=np.float32)

        from modules.quaternion_math import quat_from_axis_angle
        ik.ee_pos_des = target.astype(np.float32)
        ik.ee_quat_des = quat_from_axis_angle(np.array([0, 1, 0], np.float32), np.float32(0.0))

        quats = build_absolute_quaternions(angles, self.rc)
        rel = compute_relative_joint_angles(quats, self.rc)
        pos, quat = fk_from_angles(angles, self.rc)
        new_angles = ik.compute(
            ee_pos=pos, ee_quat=quat,
            joint_angles=rel, joint_quats=quats, dt=0.005,
        )
        delta = np.abs(np.asarray(new_angles, dtype=np.float32) - angles)
        for i, limit in enumerate(max_rad_per_iter):
            self.assertLessEqual(float(delta[i]), limit + 1e-5,
                                 msg=f"joint {i} delta {delta[i]} exceeds limit {limit}")

    def test_joint_limit_avoidance_repels_near_boundary(self):
        """Near an upper joint limit, the repulsion term should push the solver inward."""
        # Tight limits for the boom (joint 1) so we can start near the upper bound.
        joint_limits = [
            (-math.pi, math.pi),
            (-0.5, 0.2),
            (-math.pi, math.pi),
            (-math.pi, math.pi),
        ]
        cfg = make_ik_config(method="dls", joint_limits=joint_limits,
                             enable_joint_limit_avoidance=True,
                             enable_velocity_limiting=False,
                             ik_params={"k_val": 0.0, "min_singular_value": 1e-4, "lambda_val": 0.02})
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)

        # Zero IK task (k_val=0 kills task contribution) so only repulsion drives the delta.
        angles = np.array([0.0, 0.19, 0.0, 0.0], dtype=np.float32)  # just under 0.2 upper limit
        quats = build_absolute_quaternions(angles, self.rc)
        rel = compute_relative_joint_angles(quats, self.rc)
        pos, quat = fk_from_angles(angles, self.rc)

        ik.ee_pos_des = pos.astype(np.float32)
        from modules.quaternion_math import quat_from_axis_angle
        ik.ee_quat_des = quat_from_axis_angle(np.array([0, 1, 0], np.float32), np.float32(0.0))

        new_angles = ik.compute(
            ee_pos=pos, ee_quat=quat,
            joint_angles=rel, joint_quats=quats, dt=0.005,
        )
        # Joint 1 should be pushed downward (away from upper bound).
        self.assertLess(float(new_angles[1]), float(angles[1]),
                        msg=f"joint-limit avoidance did not push inward: {new_angles[1]} vs {angles[1]}")

    def test_frame_transform_position_still_stable_near_zero_slew(self):
        """Near zero slew, position IK should converge stably."""
        cfg = make_ik_config(method="dls",
                             enable_velocity_limiting=False,
                             ik_params={"k_val": 0.5, "min_singular_value": 1e-4, "lambda_val": 0.02})
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)
        start = np.array([0.0, math.radians(-45.0), math.radians(30.0), 0.0], dtype=np.float32)
        start_pos, _ = fk_from_angles(start, self.rc)
        target = start_pos + np.array([0.015, 0.0, -0.01], dtype=np.float32)
        result = simulate_ik(self.rc, ik, start, target,
                             max_iters=400, pos_tol=POSITION_TOL_M, dt=0.005)
        self.assertTrue(result["converged"],
                        msg=f"position IK did not converge: err={result['pos_error']:.4f}")

    def test_auto_detected_controllable_dofs_pose_mode(self):
        """Auto-detection includes yaw (slew=Z) and pitch (boom/arm/bucket=Y), excludes roll (no X joints)."""
        cfg = make_ik_config(method="dls", command_type="pose",
                             enable_velocity_limiting=False,
                             ik_params={"k_val": 0.5, "min_singular_value": 1e-4, "lambda_val": 0.02})
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)

        # Slew (Z) → yaw present; boom/arm/bucket (Y) → pitch present; no X joint → roll absent.
        self.assertIn(5, ik.controllable_dofs, msg="yaw should be auto-detected (slew is Z-axis joint)")
        self.assertIn(4, ik.controllable_dofs, msg="pitch should be auto-detected")
        self.assertNotIn(3, ik.controllable_dofs, msg="roll should be absent (no X-axis joint)")
        self.assertIn(0, ik.controllable_dofs)
        self.assertIn(1, ik.controllable_dofs)
        self.assertIn(2, ik.controllable_dofs)

    def test_condition_number_updated_each_iteration(self):
        cfg = make_ik_config(method="dls")
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)

        angles = np.array([0.0, math.radians(-30.0), math.radians(20.0), 0.0], dtype=np.float32)
        quats = build_absolute_quaternions(angles, self.rc)
        rel = compute_relative_joint_angles(quats, self.rc)
        pos, quat = fk_from_angles(angles, self.rc)
        ik.ee_pos_des = pos.astype(np.float32)
        from modules.quaternion_math import quat_from_axis_angle
        ik.ee_quat_des = quat_from_axis_angle(np.array([0, 1, 0], np.float32), np.float32(0.0))

        _ = ik.compute(ee_pos=pos, ee_quat=quat, joint_angles=rel, joint_quats=quats, dt=0.005)
        self.assertTrue(np.isfinite(ik.last_condition_number))
        self.assertGreater(ik.last_condition_number, 0.0)

    def test_unbounded_joint_limit_passes_target_through(self):
        """(-inf, +inf) sentinel for a joint must skip both clamp and avoidance."""
        joint_limits = [
            (float("-inf"), float("inf")),  # slew: unbounded
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
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)

        # Drive a target that needs slew far past +π. We don't expect the IK to
        # actually run all the way around (FK position objective doesn't push
        # there) — we just need the path to be NaN/clamp-free for an unbounded
        # joint, and the bounded joints still respected.
        angles = np.array([3.5, 0.0, 0.0, 0.0], dtype=np.float32)  # slew already past π
        pos, quat = fk_from_angles(angles, self.rc)
        ik.ee_pos_des = (pos + np.array([0.01, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
        from modules.quaternion_math import quat_from_axis_angle as q_aa
        ik.ee_quat_des = q_aa(np.array([0, 1, 0], np.float32), np.float32(0.0))

        quats = build_absolute_quaternions(angles, self.rc)
        rel = compute_relative_joint_angles(quats, self.rc)
        new_angles = ik.compute(
            ee_pos=pos, ee_quat=quat, joint_angles=rel, joint_quats=quats, dt=0.005,
        )
        # Slew unbounded → must not be NaN, must not be silently snapped to ±π.
        self.assertTrue(np.isfinite(new_angles[0]))
        # Bounded joints must still respect their limits.
        for i in (1, 2, 3):
            self.assertGreaterEqual(float(new_angles[i]), -1.0 - 1e-6)
            self.assertLessEqual(float(new_angles[i]), 1.0 + 1e-6)

    def test_target_joint_angles_always_within_configured_limits(self):
        """A massive position request must still be clamped to per-joint limits."""
        joint_limits = [(-0.10, 0.10), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        cfg = make_ik_config(
            method="dls",
            enable_velocity_limiting=False,
            joint_limits=joint_limits,
            enable_joint_limit_avoidance=False,
            ik_params={"k_val": 5.0, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)

        angles = np.zeros(4, dtype=np.float32)
        pos, quat = fk_from_angles(angles, self.rc)
        # 1m offset is well past the workspace.
        target = pos + np.array([1.0, 1.0, 0.0], dtype=np.float32)
        ik.ee_pos_des = target.astype(np.float32)
        from modules.quaternion_math import quat_from_axis_angle
        ik.ee_quat_des = quat_from_axis_angle(np.array([0, 1, 0], np.float32), np.float32(0.0))

        quats = build_absolute_quaternions(angles, self.rc)
        rel = compute_relative_joint_angles(quats, self.rc)
        new_angles = ik.compute(
            ee_pos=pos, ee_quat=quat, joint_angles=rel, joint_quats=quats, dt=0.005,
        )
        for i, (q_min, q_max) in enumerate(joint_limits):
            self.assertGreaterEqual(float(new_angles[i]), q_min - 1e-6,
                                    msg=f"joint {i} below limit: {new_angles[i]} < {q_min}")
            self.assertLessEqual(float(new_angles[i]), q_max + 1e-6,
                                 msg=f"joint {i} above limit: {new_angles[i]} > {q_max}")

    def test_joint_limit_avoidance_at_lower_boundary_pushes_outward(self):
        """Near the lower limit, repulsion should push the joint upward."""
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
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)

        # Just above the lower limit (-0.5 + tiny).
        angles = np.array([0.0, -0.49, 0.0, 0.0], dtype=np.float32)
        pos, quat = fk_from_angles(angles, self.rc)
        ik.ee_pos_des = pos.astype(np.float32)
        from modules.quaternion_math import quat_from_axis_angle
        ik.ee_quat_des = quat_from_axis_angle(np.array([0, 1, 0], np.float32), np.float32(0.0))

        quats = build_absolute_quaternions(angles, self.rc)
        rel = compute_relative_joint_angles(quats, self.rc)
        new_angles = ik.compute(
            ee_pos=pos, ee_quat=quat, joint_angles=rel, joint_quats=quats, dt=0.005,
        )
        self.assertGreater(float(new_angles[1]), float(angles[1]),
                           msg="lower-limit avoidance must push joint upward")

    def test_velocity_mode_integrates_over_dt(self):
        """In velocity_mode, the per-cycle delta should scale with dt for the same desired_vel."""
        cfg = make_ik_config(
            method="pinv",
            command_type="position",
            velocity_mode=True,
            enable_velocity_limiting=False,
            enable_joint_limit_avoidance=False,
            enable_adaptive_damping=False,
            ik_params={"k_val": 1.0, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)
        ik.velocity_error_gain = 0.0  # Pure feedforward velocity, no error feedback.

        angles = np.array([0.0, math.radians(-45.0), math.radians(30.0), 0.0], dtype=np.float32)
        pos, quat = fk_from_angles(angles, self.rc)
        # Set target = current so error is exactly zero — only feedforward contributes.
        ik.ee_pos_des = pos.astype(np.float32)
        from modules.quaternion_math import quat_from_axis_angle
        ik.ee_quat_des = quat_from_axis_angle(np.array([0, 1, 0], np.float32), np.float32(0.0))

        desired_vel = np.array([0.05, 0.0, 0.0], dtype=np.float32)  # 5cm/s along +x
        quats = build_absolute_quaternions(angles, self.rc)
        rel = compute_relative_joint_angles(quats, self.rc)

        new_a = ik.compute(ee_pos=pos, ee_quat=quat, joint_angles=rel,
                           joint_quats=quats, desired_ee_velocity=desired_vel, dt=0.01)
        new_b = ik.compute(ee_pos=pos, ee_quat=quat, joint_angles=rel,
                           joint_quats=quats, desired_ee_velocity=desired_vel, dt=0.02)
        delta_a = np.asarray(new_a) - angles
        delta_b = np.asarray(new_b) - angles
        # Doubling dt should double the integrated joint delta to leading order.
        for i in range(4):
            if abs(float(delta_a[i])) < 1e-8:
                self.assertAlmostEqual(float(delta_b[i]), 0.0, delta=1e-8)
                continue
            self.assertAlmostEqual(float(delta_b[i]) / (float(delta_a[i]) + 1e-12),
                                   2.0, delta=0.05,
                                   msg=f"joint {i}: dt scaling broken {delta_a[i]} -> {delta_b[i]}")

    def test_position_only_ik_at_pi_over_two_slew(self):
        """Position-only IK must converge at slew=π/2 (frame transform consistency)."""
        cfg = make_ik_config(
            method="dls", command_type="position",
            enable_velocity_limiting=False,
            ik_params={"k_val": 0.5, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )
        ik = IKController(cfg, self.rc, verbose=False, default_dt=0.005)
        start = np.array([math.pi / 2, math.radians(-45), math.radians(30), 0.0],
                         dtype=np.float32)
        pos0, _ = fk_from_angles(start, self.rc)
        target = pos0 + np.array([0.02, 0.01, -0.01], dtype=np.float32)
        result = simulate_ik(self.rc, ik, start, target,
                             max_iters=500, pos_tol=POSITION_TOL_M, dt=0.005)
        self.assertTrue(result["converged"],
                        msg=f"position IK at slew=π/2 stalled: err={result['pos_error']:.4f}")


class IKPoseModeTests(unittest.TestCase):
    """Pose-mode IK has to match the production controller's target-quat recipe.

    The controller composes target_quat as ``slew_quat * pitch_quat`` so that
    yaw error cancels by construction. Pitch is therefore body-frame (around
    the upper-body Y axis after the slew rotation).
    """

    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def _make_pose_ik(self):
        cfg = make_ik_config(
            method="dls", command_type="pose",
            enable_velocity_limiting=False,
            enable_adaptive_damping=False,
            enable_joint_limit_avoidance=False,
            ik_params={"k_val": 0.5, "min_singular_value": 1e-4, "lambda_val": 0.02},
        )
        return IKController(cfg, self.rc, verbose=False, default_dt=0.005)

    def _make_target_quat(self, slew_rad: float, total_pitch_rad: float):
        from modules.quaternion_math import quat_from_axis_angle as q_aa
        slew_q = q_aa(np.array([0, 0, 1], np.float32), np.float32(slew_rad))
        pitch_q = q_aa(np.array([0, 1, 0], np.float32), np.float32(total_pitch_rad))
        return quat_normalize(quat_multiply(slew_q, pitch_q))

    def _solve_to_pose(self, ik, start_angles, target_pos, target_quat,
                       max_iters=500, pos_tol=3e-3, ang_tol=math.radians(1.0),
                       dt=0.005):
        from modules.quaternion_math import compute_pose_error
        ik.ee_pos_des = target_pos.astype(np.float32)
        ik.ee_quat_des = target_quat.astype(np.float32)
        angles = np.asarray(start_angles, dtype=np.float32).copy()
        for _ in range(max_iters):
            pos, quat = fk_from_angles(angles, self.rc)
            pe, ae = compute_pose_error(pos, quat, target_pos, target_quat)
            if np.linalg.norm(pe) < pos_tol and np.linalg.norm(ae) < ang_tol:
                return angles, True
            quats = build_absolute_quaternions(angles, self.rc)
            rel = compute_relative_joint_angles(quats, self.rc)
            new_angles = ik.compute(
                ee_pos=pos, ee_quat=quat,
                joint_angles=rel, joint_quats=quats, dt=dt,
            )
            angles = np.asarray(new_angles, dtype=np.float32)
        return angles, False

    def _start_angles(self, slew_rad):
        return np.array(
            [slew_rad, math.radians(-45), math.radians(30), math.radians(-15)],
            dtype=np.float32,
        )

    def test_pose_mode_converges_at_zero_slew(self):
        ik = self._make_pose_ik()
        start = self._start_angles(0.0)
        pos0, _ = fk_from_angles(start, self.rc)

        # Target: same position, +5° body pitch.
        from modules.differential_ik import extract_axis_rotation
        from modules.quaternion_math import quat_conjugate
        _, q0 = fk_from_angles(start, self.rc)
        slew_q = quat_from_axis_angle(np.array([0, 0, 1], np.float32),
                                      np.float32(0.0))
        body_q = quat_normalize(quat_multiply(quat_conjugate(slew_q), q0))
        cur_pitch = extract_axis_rotation(body_q, np.array([0, 1, 0], np.float32))
        target_quat = self._make_target_quat(0.0, cur_pitch + math.radians(5.0))

        _, ok = self._solve_to_pose(ik, start, pos0.astype(np.float32),
                                    np.asarray(target_quat, np.float32))
        self.assertTrue(ok, msg="pose IK at slew=0 must converge")

    def test_pose_mode_converges_at_quarter_slew(self):
        ik = self._make_pose_ik()
        slew = math.radians(45.0)
        start = self._start_angles(slew)
        pos0, q0 = fk_from_angles(start, self.rc)
        from modules.differential_ik import extract_axis_rotation
        from modules.quaternion_math import quat_conjugate
        slew_q = quat_from_axis_angle(np.array([0, 0, 1], np.float32),
                                      np.float32(slew))
        body_q = quat_normalize(quat_multiply(quat_conjugate(slew_q), q0))
        cur_pitch = extract_axis_rotation(body_q, np.array([0, 1, 0], np.float32))
        target_quat = self._make_target_quat(slew, cur_pitch + math.radians(5.0))

        _, ok = self._solve_to_pose(ik, start, pos0.astype(np.float32),
                                    np.asarray(target_quat, np.float32),
                                    max_iters=800)
        self.assertTrue(ok, msg="pose IK at slew=π/4 must converge")

    def test_pose_mode_converges_at_pi_over_two_slew(self):
        """At slew=π/2 the orientation error is computed in world frame as a
        pure roll (since the controller composes target as ``slew_quat * pitch_quat``)
        which the controllable-DOF reduction would drop on a roll-less excavator
        chain. ``IKController.compute`` rotates the orientation error into base
        frame to match the body-frame Jacobian, so the IK still sees the pitch
        error and can drive the bucket joint.
        """
        ik = self._make_pose_ik()
        slew = math.pi / 2
        start = self._start_angles(slew)
        pos0, q0 = fk_from_angles(start, self.rc)
        from modules.differential_ik import extract_axis_rotation
        from modules.quaternion_math import quat_conjugate
        slew_q = quat_from_axis_angle(np.array([0, 0, 1], np.float32),
                                      np.float32(slew))
        body_q = quat_normalize(quat_multiply(quat_conjugate(slew_q), q0))
        cur_pitch = extract_axis_rotation(body_q, np.array([0, 1, 0], np.float32))
        target_quat = self._make_target_quat(slew, cur_pitch + math.radians(5.0))

        _, ok = self._solve_to_pose(ik, start, pos0.astype(np.float32),
                                    np.asarray(target_quat, np.float32),
                                    max_iters=800)
        self.assertTrue(ok, msg="pose IK at slew=π/2 must converge")

    def test_pose_mode_converges_at_full_slew_sweep(self):
        """The fix has to hold across the full slew range, not just at the singular angle."""
        for slew_deg in (-150, -90, -45, 0, 45, 90, 135, 180):
            ik = self._make_pose_ik()
            slew = math.radians(slew_deg)
            start = self._start_angles(slew)
            pos0, q0 = fk_from_angles(start, self.rc)
            from modules.differential_ik import extract_axis_rotation
            from modules.quaternion_math import quat_conjugate
            slew_q = quat_from_axis_angle(np.array([0, 0, 1], np.float32),
                                          np.float32(slew))
            body_q = quat_normalize(quat_multiply(quat_conjugate(slew_q), q0))
            cur_pitch = extract_axis_rotation(body_q, np.array([0, 1, 0], np.float32))
            target_quat = self._make_target_quat(slew, cur_pitch + math.radians(4.0))
            _, ok = self._solve_to_pose(ik, start, pos0.astype(np.float32),
                                        np.asarray(target_quat, np.float32),
                                        max_iters=800)
            self.assertTrue(ok, msg=f"pose IK failed at slew={slew_deg}deg")


if __name__ == "__main__":
    unittest.main()
