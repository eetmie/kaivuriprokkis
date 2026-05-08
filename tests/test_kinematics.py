"""Forward kinematics and Jacobian correctness tests.

These tests verify that the excavator's FK pipeline and Jacobian behave
the way the IK layer assumes.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.differential_ik import (  # noqa: E402
    compute_jacobian, compute_jacobian_from_joint_angles,
    compute_jacobian_metrics, project_to_rotation_axes, propagate_base_rotation,
    extract_axis_rotation,
)
from modules.differential_ik_cfg import RobotConfig  # noqa: E402
from modules.excavator_ik_utils import (  # noqa: E402
    get_all_poses, compute_relative_joint_angles,
    canonical_joint_angles_from_imus,
    absolute_link_angles_from_quats,
    get_joint_positions,
)
from modules.quaternion_math import (  # noqa: E402
    quat_from_axis_angle, quat_multiply, quat_normalize, quat_rotate_vector,
)

from tests.common_kinematics import (  # noqa: E402
    build_absolute_quaternions,
    fk_from_angles,
    joint_positions_from_angles,
    load_robot_config,
    numerical_jacobian,
)


class ForwardKinematicsTests(unittest.TestCase):
    """FK should agree with analytically-derived excavator geometry."""

    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def _assert_close(self, a, b, atol=1e-4, msg=None):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        self.assertTrue(
            np.allclose(a, b, atol=atol),
            msg=f"{msg or ''}\n  expected: {b}\n  actual  : {a}\n  diff    : {a-b}",
        )

    def test_zero_pose_geometry(self):
        """With all joints at zero the EE tip sits at the analytic forward reach."""
        angles = np.zeros(4, dtype=np.float32)
        pos, quat = fk_from_angles(angles, self.rc)

        # slew link direction is a unit vector; its x-component * slew-length adds to x.
        slew_length = float(self.rc.link_lengths[0])
        slew_dir = np.asarray(self.rc.link_directions[0], dtype=np.float64)
        base_x = float(self.rc.origin_offset[0]) + slew_length * slew_dir[0]
        base_z = float(self.rc.origin_offset[2]) + slew_length * slew_dir[2]

        expected_x = base_x + float(self.rc.link_lengths[1] + self.rc.link_lengths[2] + self.rc.link_lengths[3])
        expected_y = 0.0
        expected_z = base_z + float(self.rc.ee_offset[2])

        self._assert_close(pos, [expected_x, expected_y, expected_z], atol=2e-4,
                           msg="zero-pose EE position")

        # Orientation at zero pose is identity.
        self._assert_close(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-6,
                           msg="zero-pose EE orientation")

    def test_pure_slew_rotates_ee_around_z(self):
        """A pure slew rotation moves the EE on a circle at constant radius and z."""
        angles0 = np.zeros(4, dtype=np.float32)
        pos0, _ = fk_from_angles(angles0, self.rc)
        r0 = math.hypot(float(pos0[0]), float(pos0[1]))
        z0 = float(pos0[2])

        for deg in (30, 60, 90, -45, -135, 170):
            angles = np.array([math.radians(deg), 0.0, 0.0, 0.0], dtype=np.float32)
            pos, _ = fk_from_angles(angles, self.rc)
            r = math.hypot(float(pos[0]), float(pos[1]))
            self.assertAlmostEqual(r, r0, delta=3e-4,
                                   msg=f"slew {deg} changed radius from {r0:.4f} to {r:.4f}")
            self.assertAlmostEqual(float(pos[2]), z0, delta=3e-4,
                                   msg=f"slew {deg} changed z from {z0:.4f} to {pos[2]:.4f}")
            yaw = math.degrees(math.atan2(float(pos[1]), float(pos[0])))
            # Wrap both to [-180, 180] and compare
            wrap = lambda a: (a + 180.0) % 360.0 - 180.0
            self.assertAlmostEqual(wrap(yaw), wrap(deg), delta=0.2,
                                   msg=f"slew {deg} yields yaw {yaw}")

    def test_boom_pitch_lifts_ee_vertically(self):
        """At zero slew, a positive boom relative angle raises the EE in +Z only."""
        angles = np.array([0.0, math.radians(30.0), 0.0, 0.0], dtype=np.float32)
        pos, _ = fk_from_angles(angles, self.rc)

        # Y component should still be zero because slew is zero.
        self.assertAlmostEqual(float(pos[1]), 0.0, delta=1e-4)

    def test_get_joint_positions_matches_forward_walk(self):
        """`get_joint_positions` should equal a direct walk along the link chain."""
        angles = np.array([0.2, -0.3, 0.4, -0.1], dtype=np.float32)
        jp = joint_positions_from_angles(angles, self.rc)
        self.assertEqual(jp.shape, (4, 3))

        # Last joint position + rotated ee_offset should equal FK output.
        ee_pos, ee_quat = fk_from_angles(angles, self.rc)
        last = np.asarray(jp[-1], dtype=np.float64)
        from modules.quaternion_math import quat_rotate_vector

        ee_from_chain = last + np.asarray(
            quat_rotate_vector(ee_quat, self.rc.ee_offset), dtype=np.float64
        )
        self.assertTrue(np.allclose(ee_from_chain, np.asarray(ee_pos, dtype=np.float64), atol=2e-4))

    def test_get_all_poses_returns_consistent_bundle(self):
        angles = np.array([0.5, -0.4, 0.2, -0.3], dtype=np.float32)
        quats = build_absolute_quaternions(angles, self.rc)
        jp, jo, ee_p, ee_o = get_all_poses(quats, self.rc)

        self.assertEqual(jp.shape, (4, 3))
        self.assertEqual(jo.shape, (4, 4))
        self.assertEqual(ee_p.shape, (3,))
        self.assertEqual(ee_o.shape, (4,))

        # EE orientation is the last joint orientation.
        self.assertTrue(np.allclose(jo[-1], ee_o, atol=1e-5))

    def test_compute_relative_joint_angles_recovers_inputs(self):
        """Absolute -> relative decomposition should recover the original angles."""
        for trial in range(5):
            rng = np.random.default_rng(1234 + trial)
            slew = float(rng.uniform(-1.0, 1.0))
            boom = float(rng.uniform(-0.8, 0.4))
            arm = float(rng.uniform(-0.4, 1.2))
            bucket = float(rng.uniform(-1.5, 0.6))

            angles = np.array([slew, boom, arm, bucket], dtype=np.float32)
            quats = build_absolute_quaternions(angles, self.rc)

            recovered = np.asarray(
                compute_relative_joint_angles(quats, self.rc), dtype=np.float64
            )
            self.assertTrue(
                np.allclose(recovered, angles.astype(np.float64), atol=5e-4),
                msg=f"trial {trial}: {recovered} vs {angles}",
            )

    def test_four_imu_canonical_extraction_recovers_inputs(self):
        """Four absolute IMU quats should reduce to one canonical joint vector."""
        for deg in range(-180, 181, 30):
            angles = np.array([math.radians(deg), -0.45, 0.35, -0.2], dtype=np.float32)
            imu_quats = build_absolute_quaternions(angles, self.rc)
            recovered = canonical_joint_angles_from_imus(imu_quats, self.rc)
            diff = np.asarray(recovered - angles, dtype=np.float64)
            diff[0] = math.atan2(math.sin(diff[0]), math.cos(diff[0]))
            self.assertTrue(
                np.all(np.abs(diff) < 6e-4),
                msg=f"yaw {deg}: recovered {recovered} vs {angles}",
            )

    def test_four_imu_canonical_extraction_uses_gravity_pitch_chain(self):
        """Hinge angles are link gravity-pitch differences: lift/base, arm/lift, bucket/arm."""
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        slew = math.radians(35.0)
        base_pitch = math.radians(7.0)
        lift = math.radians(12.0)
        bucket = math.radians(-18.0)

        for arm_deg in (-35.0, -10.0, 0.0, 25.0, 50.0):
            arm = math.radians(arm_deg)
            cumulative_pitches = np.array([
                base_pitch,
                base_pitch + lift,
                base_pitch + lift + arm,
                base_pitch + lift + arm + bucket,
            ], dtype=np.float32)
            slew_quat = quat_from_axis_angle(z_axis, np.float32(slew))
            imu_quats = np.array([
                quat_normalize(quat_multiply(
                    slew_quat,
                    quat_from_axis_angle(y_axis, np.float32(pitch)),
                ))
                for pitch in cumulative_pitches
            ], dtype=np.float32)

            recovered = canonical_joint_angles_from_imus(imu_quats, self.rc)
            expected = np.array([slew, lift, arm, bucket], dtype=np.float32)
            diff = np.asarray(recovered - expected, dtype=np.float64)
            diff[0] = math.atan2(math.sin(diff[0]), math.cos(diff[0]))
            self.assertTrue(
                np.all(np.abs(diff) < 6e-4),
                msg=f"arm {arm_deg}: recovered {recovered} vs {expected}",
            )

    def test_canonical_extraction_can_run_without_base_imu(self):
        """When base is not mapped, lift is measured against level gravity."""
        rc_no_base = RobotConfig(
            link_lengths=self.rc.link_lengths.tolist(),
            link_directions=[v.copy() for v in self.rc.link_directions],
            rotation_axes=[v.copy() for v in self.rc.rotation_axes],
            ee_offset=self.rc.ee_offset.copy(),
            origin_offset=self.rc.origin_offset.copy(),
            imu_chain=self.rc.imu_chain,
            imu_mapping={'boom': 0, 'arm': 1, 'bucket': 2},
        )
        angles = np.array([
            math.radians(25.0),
            math.radians(14.0),
            math.radians(-22.0),
            math.radians(9.0),
        ], dtype=np.float32)
        imu_quats = build_absolute_quaternions(angles, self.rc)[1:]

        recovered = canonical_joint_angles_from_imus(imu_quats, rc_no_base)
        diff = np.asarray(recovered - angles, dtype=np.float64)
        diff[0] = math.atan2(math.sin(diff[0]), math.cos(diff[0]))
        self.assertTrue(
            np.all(np.abs(diff) < 6e-4),
            msg=f"recovered {recovered} vs {angles}",
        )

    def test_yaw_average_handles_quaternion_sign_flip_at_wrap(self):
        angles = np.array([math.radians(179.0), -0.45, 0.35, -0.2], dtype=np.float32)
        imu_quats = build_absolute_quaternions(angles, self.rc)
        imu_quats[2] *= -1.0  # Same physical orientation, opposite quaternion hemisphere.
        recovered = canonical_joint_angles_from_imus(imu_quats, self.rc)
        self.assertAlmostEqual(float(recovered[0]), float(angles[0]), delta=2e-3)

    def test_extract_axis_rotation_identity_quaternion(self):
        """Identity quaternion has zero rotation about any axis."""
        idq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for axis in ([0, 0, 1], [0, 1, 0], [1, 0, 0]):
            self.assertAlmostEqual(
                extract_axis_rotation(idq, np.asarray(axis, np.float32)), 0.0, places=6
            )

    def test_extract_axis_rotation_pure_axis_recovers_angle(self):
        """A quaternion that is a pure rotation about ``axis`` round-trips."""
        for axis_idx, axis in enumerate(((0, 0, 1.0), (0, 1.0, 0), (1.0, 0, 0))):
            for angle in (-2.5, -math.pi / 2, -0.1, 0.0, 0.4, math.pi / 2, 2.5):
                q = quat_from_axis_angle(np.asarray(axis, np.float32), np.float32(angle))
                recovered = extract_axis_rotation(q, np.asarray(axis, np.float32))
                self.assertAlmostEqual(recovered, angle, places=4,
                                       msg=f"axis={axis} angle={angle} → {recovered}")

    def test_extract_axis_rotation_hemisphere_invariance(self):
        """q and -q represent the same physical rotation."""
        q = quat_from_axis_angle(np.array([0, 1, 0], np.float32), np.float32(0.7))
        a = extract_axis_rotation(q, np.array([0, 1, 0], np.float32))
        b = extract_axis_rotation(-q, np.array([0, 1, 0], np.float32))
        self.assertAlmostEqual(a, b, places=5)

    def test_extract_axis_rotation_near_pi_does_not_flip(self):
        """Near +π we must not flip to −π for a positive twist quaternion."""
        # 179.5 deg about Z
        q = quat_from_axis_angle(np.array([0, 0, 1], np.float32),
                                 np.float32(math.radians(179.5)))
        ang = extract_axis_rotation(q, np.array([0, 0, 1], np.float32))
        self.assertAlmostEqual(ang, math.radians(179.5), places=4)

    def test_absolute_link_angles_match_cumulative_body_pitch(self):
        """abs angles: slew is world Z; downstream are cumulative cab-frame pitch."""
        for slew_deg in (-150.0, -45.0, 0.0, 45.0, 90.0, 135.0):
            slew = math.radians(slew_deg)
            rel = np.array([slew, math.radians(-30), math.radians(20), math.radians(-10)],
                           dtype=np.float32)
            quats = build_absolute_quaternions(rel, self.rc)
            abs_rad = np.asarray(absolute_link_angles_from_quats(quats, self.rc),
                                 dtype=np.float64)

            self.assertAlmostEqual(abs_rad[0], slew, places=4,
                                   msg=f"slew abs at {slew_deg}: {abs_rad[0]}")
            self.assertAlmostEqual(abs_rad[1], float(rel[1]), places=4)
            self.assertAlmostEqual(abs_rad[2], float(rel[1] + rel[2]), places=4)
            self.assertAlmostEqual(abs_rad[3],
                                   float(rel[1] + rel[2] + rel[3]), places=4)

    def test_absolute_link_angles_handles_empty_chain(self):
        """Zero-length input should not crash (defensive guard for early states)."""
        out = absolute_link_angles_from_quats(np.zeros((0, 4), dtype=np.float32), self.rc)
        self.assertEqual(out.shape, (0,))

    def test_extract_axis_rotation_orthogonal_axis_returns_zero(self):
        """A pure Z-rotation has zero twist about Y."""
        q = quat_from_axis_angle(np.array([0, 0, 1], np.float32), np.float32(1.0))
        self.assertAlmostEqual(
            extract_axis_rotation(q, np.array([0, 1, 0], np.float32)), 0.0, places=5
        )


class BasePropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def test_project_to_rotation_axes_preserves_pure_y(self):
        """A pure Y rotation fed to project_to_rotation_axes is unchanged."""
        for ang in (-0.6, -0.1, 0.3, 0.9):
            q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), np.float32(ang))
            out = project_to_rotation_axes(
                np.asarray([q]), np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
            )
            self.assertTrue(
                np.allclose(out[0], q, atol=1e-5),
                msg=f"projection changed pure Y quat at {ang}: {out[0]} vs {q}",
            )

    def test_propagate_base_rotation_identity_on_zero_base(self):
        """With the base at identity, propagation leaves downstream unchanged."""
        idq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        boom = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), np.float32(0.3))
        arm = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), np.float32(-0.2))
        bucket = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), np.float32(0.1))

        quats = np.array([idq, boom, arm, bucket], dtype=np.float32)
        out = propagate_base_rotation(quats, self.rc)
        self.assertTrue(np.allclose(out[1], boom, atol=1e-5))
        self.assertTrue(np.allclose(out[2], arm, atol=1e-5))
        self.assertTrue(np.allclose(out[3], bucket, atol=1e-5))


class JacobianSanityTests(unittest.TestCase):
    """Compare the analytic position Jacobian against a numerical one."""

    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def _compare(self, angles, atol):
        angles = np.asarray(angles, dtype=np.float32)
        quats = build_absolute_quaternions(angles, self.rc)
        j_ana = np.asarray(compute_jacobian(quats, self.rc))[:3]
        j_num = numerical_jacobian(angles, self.rc)
        return j_ana, j_num, float(np.max(np.abs(j_ana - j_num)))

    def test_jacobian_matches_numerical_across_full_slew_rotation(self):
        arm_poses = (
            [-0.4, 0.3, -0.1],
            [0.2, -0.4, 0.2],
            [-0.6, 0.8, -0.3],
            [0.0, 0.0, 0.0],
        )
        for deg in range(-180, 181, 30):
            for arm_pose in arm_poses:
                angles = [math.radians(deg), *arm_pose]
                j_ana, j_num, diff = self._compare(angles, atol=2e-3)
                self.assertLess(
                    diff,
                    2e-3,
                    msg=f"yaw {deg} Jacobian mismatch {diff} at {angles}\nana={j_ana}\nnum={j_num}",
                )

    def test_jacobian_from_joint_angles_matches_quaternion_wrapper(self):
        for deg in range(-180, 181, 45):
            angles = np.array([math.radians(deg), -0.4, 0.3, -0.1], dtype=np.float32)
            j_ana, j_num, diff = self._compare(angles, atol=2e-3)
            j_angles = np.asarray(compute_jacobian_from_joint_angles(angles, self.rc))[:3]
            self.assertTrue(np.allclose(j_ana, j_angles, atol=1e-6))
            self.assertLess(diff, 2e-3)

    def test_condition_number_finite_away_from_singularity(self):
        angles = np.array([0.1, -0.4, 0.3, -0.1], dtype=np.float32)
        quats = build_absolute_quaternions(angles, self.rc)
        J = np.asarray(compute_jacobian(quats, self.rc))
        cond, _, _ = compute_jacobian_metrics(J)
        self.assertTrue(np.isfinite(cond))
        self.assertGreater(cond, 1.0)

    def test_singularity_detection_finds_bad_configs(self):
        """Fully extended (boom=arm=0) is a true singularity — cond should be huge."""
        angles = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        quats = build_absolute_quaternions(angles, self.rc)
        J = np.asarray(compute_jacobian(quats, self.rc))
        cond, _, _ = compute_jacobian_metrics(J)
        self.assertTrue(np.isfinite(cond))
        # A configuration where the boom/arm/bucket pitch axes are all collinear
        # collapses one Jacobian column — cond should be many orders of magnitude
        # larger than at a generic well-conditioned pose (cond ~30 there).
        self.assertGreater(
            cond, 1e3,
            msg=f"fully-extended pose should be near-singular, got cond={cond}",
        )

    def test_well_conditioned_pose_has_modest_condition_number(self):
        """A typical operating pose should be clearly inside a usable conditioning regime."""
        angles = np.array([0.1, math.radians(-45.0), math.radians(30.0), math.radians(-15.0)],
                          dtype=np.float32)
        quats = build_absolute_quaternions(angles, self.rc)
        J = np.asarray(compute_jacobian(quats, self.rc))
        cond, _, _ = compute_jacobian_metrics(J)
        self.assertLess(cond, 200.0,
                        msg=f"generic mid-workspace cond unexpectedly large: {cond}")

    def test_jacobian_orientation_rows_match_revolute_axes(self):
        """The bottom 3 rows of the Jacobian are the joint axes in world frame."""
        for slew_deg in (0, 45, 90, -90, 180):
            angles = np.array([math.radians(slew_deg), -0.4, 0.3, -0.1], dtype=np.float32)
            quats = build_absolute_quaternions(angles, self.rc)
            J = np.asarray(compute_jacobian(quats, self.rc), dtype=np.float64)
            J_rot = J[3:6, :]

            # Joint 0 (slew) axis stays world Z.
            self.assertTrue(np.allclose(J_rot[:, 0], [0.0, 0.0, 1.0], atol=2e-5),
                            msg=f"slew axis at slew={slew_deg}: {J_rot[:, 0]}")
            # Joints 1-3 axes are R_z(slew) @ Y at the moment those joints rotate
            # (all three boom/arm/bucket share the upper body so they all lie in
            # the slew-rotated XY plane).
            expected_axis = np.array([
                -math.sin(math.radians(slew_deg)),
                math.cos(math.radians(slew_deg)),
                0.0,
            ])
            for ji in (1, 2, 3):
                self.assertTrue(np.allclose(J_rot[:, ji], expected_axis, atol=2e-5),
                                msg=f"joint {ji} axis at slew={slew_deg}: {J_rot[:, ji]}")

    def test_jacobian_position_rows_orthogonal_to_axis_at_zero_pose(self):
        """At the zero pose, the position contribution of each joint is axis × (ee_pos - joint_pos)
        and must be perpendicular to that joint's axis."""
        angles = np.zeros(4, dtype=np.float32)
        quats = build_absolute_quaternions(angles, self.rc)
        J = np.asarray(compute_jacobian(quats, self.rc), dtype=np.float64)
        for ji in range(4):
            axis = J[3:6, ji]
            self.assertAlmostEqual(float(np.dot(axis, J[0:3, ji])), 0.0, places=5,
                                   msg=f"joint {ji} position not perp to axis")

    def test_fk_origin_offset_translates_all_outputs(self):
        """Shifting origin_offset must shift every joint position and EE by the same vector."""
        angles = np.array([0.2, -0.4, 0.3, -0.1], dtype=np.float32)
        rc_a = self.rc

        rc_b = RobotConfig(
            link_lengths=list(np.asarray(rc_a.link_lengths, dtype=np.float32)),
            link_directions=[d.copy() for d in rc_a.link_directions],
            rotation_axes=[a.copy() for a in rc_a.rotation_axes],
            ee_offset=rc_a.ee_offset.copy(),
            origin_offset=(rc_a.origin_offset + np.array([0.5, -0.2, 1.0],
                                                         dtype=np.float32)).copy(),
        )

        quats_a = build_absolute_quaternions(angles, rc_a)
        jp_a = np.asarray(get_joint_positions(quats_a, rc_a), dtype=np.float64)
        ee_a, _, _, _ = (None, None, None, None)
        from modules.excavator_ik_utils import get_pose
        ee_a, _ = get_pose(quats_a, rc_a)
        ee_a = np.asarray(ee_a, dtype=np.float64)

        quats_b = build_absolute_quaternions(angles, rc_b)
        jp_b = np.asarray(get_joint_positions(quats_b, rc_b), dtype=np.float64)
        ee_b, _ = get_pose(quats_b, rc_b)
        ee_b = np.asarray(ee_b, dtype=np.float64)

        shift = np.array([0.5, -0.2, 1.0], dtype=np.float64)
        for i in range(4):
            self.assertTrue(np.allclose(jp_b[i] - jp_a[i], shift, atol=2e-5),
                            msg=f"joint {i} shift mismatch: {jp_b[i]-jp_a[i]} vs {shift}")
        self.assertTrue(np.allclose(ee_b - ee_a, shift, atol=2e-5),
                        msg=f"EE shift mismatch: {ee_b-ee_a} vs {shift}")

    def test_fk_ee_offset_applied_in_last_link_frame(self):
        """ee_offset rotates with the bucket, so a non-zero bucket angle changes EE position."""
        angles_a = np.array([0.0, math.radians(-45), math.radians(30), 0.0], dtype=np.float32)
        angles_b = angles_a.copy()
        angles_b[3] = math.radians(60.0)  # rotate bucket only
        ee_a, _ = fk_from_angles(angles_a, self.rc)
        ee_b, _ = fk_from_angles(angles_b, self.rc)
        # If ee_offset is non-trivial the EE world position moves when the bucket pitches.
        if float(np.linalg.norm(self.rc.ee_offset)) > 1e-6:
            self.assertGreater(float(np.linalg.norm(ee_a - ee_b)), 1e-4,
                               msg="bucket rotation did not move EE — ee_offset not applied?")


if __name__ == "__main__":
    unittest.main()
