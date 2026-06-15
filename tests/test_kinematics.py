"""Forward kinematics and Jacobian correctness tests.

These tests verify that the excavator's FK pipeline (URDF-style chain in
``modules.ik.model.ExcavatorModel`` + ``get_state``) and Jacobian behave
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

from modules.ik import (  # noqa: E402
    extract_axis_rotation,
    get_state,
    joint_angles_from_imus,
    joint_angles_to_absolute_quaternions,
    load_imu_config,
    project_to_rotation_axes,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    quat_rotate_vector,
)
from modules.ik.kinematics import _jacobian_metrics  # noqa: E402
from modules.ik.model import build_excavator_model  # noqa: E402

from tests.common_kinematics import (  # noqa: E402
    build_absolute_quaternions,
    fk_from_angles,
    joint_positions_from_angles,
    load_robot_config,
    numerical_jacobian,
)


_CFG_PATH = "configuration_files/profiles/rpi/control_config.yaml"


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

    def test_zero_pose_matches_summed_offsets(self):
        """With all joints at zero the EE tip equals the summed URDF offsets + tip."""
        angles = np.zeros(self.rc.num_joints, dtype=np.float32)
        pos, quat = fk_from_angles(angles, self.rc)

        expected = np.zeros(3, dtype=np.float64)
        for offset in self.rc.offsets:
            expected += np.asarray(offset, dtype=np.float64)
        expected += np.asarray(self.rc.tip_offset, dtype=np.float64)

        self._assert_close(pos, expected, atol=2e-5, msg="zero-pose EE position")
        self._assert_close(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-6,
                           msg="zero-pose EE orientation")

    def test_pure_slew_rotates_ee_around_z(self):
        """A pure slew rotation moves the EE on a circle at constant radius and z."""
        angles0 = np.zeros(self.rc.num_joints, dtype=np.float32)
        pos0, _ = fk_from_angles(angles0, self.rc)
        r0 = math.hypot(float(pos0[0]), float(pos0[1]))
        z0 = float(pos0[2])

        for deg in (30, 60, 90, -45, -135, 170):
            angles = np.zeros(self.rc.num_joints, dtype=np.float32)
            angles[0] = math.radians(deg)
            pos, _ = fk_from_angles(angles, self.rc)
            r = math.hypot(float(pos[0]), float(pos[1]))
            self.assertAlmostEqual(r, r0, delta=3e-4,
                                   msg=f"slew {deg} changed radius from {r0:.4f} to {r:.4f}")
            self.assertAlmostEqual(float(pos[2]), z0, delta=3e-4,
                                   msg=f"slew {deg} changed z from {z0:.4f} to {pos[2]:.4f}")
            yaw = math.degrees(math.atan2(float(pos[1]), float(pos[0])))
            wrap = lambda a: (a + 180.0) % 360.0 - 180.0
            self.assertAlmostEqual(wrap(yaw), wrap(deg), delta=0.2,
                                   msg=f"slew {deg} yields yaw {yaw}")

    def test_boom_pitch_keeps_ee_in_xz_plane_at_zero_slew(self):
        """At zero slew, a positive boom relative angle keeps the EE in the x/z plane."""
        angles = np.array([0.0, math.radians(30.0), 0.0, 0.0], dtype=np.float32)
        pos, _ = fk_from_angles(angles, self.rc)
        self.assertAlmostEqual(float(pos[1]), 0.0, delta=1e-4)

    def test_joint_origins_walk_matches_chain(self):
        """``joint_origins_world`` should equal a direct walk along the chain."""
        angles = np.array([0.2, -0.3, 0.4, -0.1], dtype=np.float32)
        jp = joint_positions_from_angles(angles, self.rc)
        self.assertEqual(jp.shape, (self.rc.num_joints, 3))

        # Last joint origin + tool_offset rotated by EE orientation must equal EE position.
        ee_pos, ee_quat = fk_from_angles(angles, self.rc)
        last = np.asarray(jp[-1], dtype=np.float64)
        ee_from_chain = last + np.asarray(
            quat_rotate_vector(ee_quat, self.rc.tip_offset), dtype=np.float64
        )
        self.assertTrue(np.allclose(ee_from_chain, np.asarray(ee_pos, dtype=np.float64), atol=2e-5))

    def test_get_state_bundle_is_self_consistent(self):
        """The single rich state bundle should agree with the lighter wrappers."""
        angles = np.array([0.5, -0.4, 0.2, -0.3], dtype=np.float32)
        state = get_state(angles, self.rc)
        quats = build_absolute_quaternions(angles, self.rc)
        ee_pos, ee_quat = fk_from_angles(angles, self.rc)
        jp = joint_positions_from_angles(angles, self.rc)

        self.assertTrue(np.allclose(state.joint_angles_rad, angles, atol=1e-6))
        self.assertTrue(np.allclose(state.link_orientations, quats, atol=1e-5))
        self.assertTrue(np.allclose(state.joint_origins_world, jp, atol=1e-5))
        self.assertTrue(np.allclose(state.ee_position, ee_pos, atol=1e-5))
        self.assertTrue(np.allclose(state.ee_orientation, ee_quat, atol=1e-5))
        # EE orientation should be the last link's orientation.
        self.assertTrue(np.allclose(state.ee_orientation, state.link_orientations[-1], atol=1e-6))

    def test_joint_angles_to_absolute_quaternions_round_trips_through_state(self):
        angles = np.array([0.3, -0.2, 0.4, -0.5], dtype=np.float32)
        quats = joint_angles_to_absolute_quaternions(angles, self.rc)
        state = get_state(angles, self.rc, include_jacobian=False)
        self.assertTrue(np.allclose(quats, state.link_orientations, atol=1e-6))


class IMUExtractionTests(unittest.TestCase):
    """``joint_angles_from_imus`` should recover the joint vector from synthetic IMU quats."""

    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()
        cls.imu_cfg = load_imu_config(_CFG_PATH)

    def _synthesize_imu_quats(self, slew, base_pitch, lift, arm, bucket):
        """Build IMU quats for roles [base, boom, arm, bucket] from physical pitches."""
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        cumulative_pitches = np.array([
            base_pitch,
            base_pitch + lift,
            base_pitch + lift + arm,
            base_pitch + lift + arm + bucket,
        ], dtype=np.float32)
        slew_quat = quat_from_axis_angle(z_axis, np.float32(slew))
        return np.array([
            quat_normalize(quat_multiply(
                slew_quat, quat_from_axis_angle(y_axis, np.float32(p))
            ))
            for p in cumulative_pitches
        ], dtype=np.float32)

    def test_canonical_extraction_recovers_inputs_across_yaw(self):
        """Four absolute IMU quats reduce to one canonical joint vector for any slew yaw."""
        for deg in range(-180, 181, 30):
            slew = math.radians(deg)
            lift = -0.45
            arm = 0.35
            bucket = -0.2
            imu_quats = self._synthesize_imu_quats(slew, 0.0, lift, arm, bucket)
            recovered = joint_angles_from_imus(imu_quats, self.imu_cfg, self.rc)
            expected = np.array([slew, lift, arm, bucket], dtype=np.float32)
            diff = np.asarray(recovered - expected, dtype=np.float64)
            diff[0] = math.atan2(math.sin(diff[0]), math.cos(diff[0]))
            self.assertTrue(
                np.all(np.abs(diff) < 6e-4),
                msg=f"yaw {deg}: recovered {recovered} vs {expected}",
            )

    def test_pitch_chain_is_gravity_pitch_delta_against_parent(self):
        """Hinge angles are link gravity-pitch differences: boom/base, arm/boom, bucket/arm."""
        slew = math.radians(35.0)
        base_pitch = math.radians(7.0)
        lift = math.radians(12.0)
        bucket = math.radians(-18.0)
        for arm_deg in (-35.0, -10.0, 0.0, 25.0, 50.0):
            arm = math.radians(arm_deg)
            imu_quats = self._synthesize_imu_quats(slew, base_pitch, lift, arm, bucket)
            recovered = joint_angles_from_imus(imu_quats, self.imu_cfg, self.rc)
            expected = np.array([slew, lift, arm, bucket], dtype=np.float32)
            diff = np.asarray(recovered - expected, dtype=np.float64)
            diff[0] = math.atan2(math.sin(diff[0]), math.cos(diff[0]))
            self.assertTrue(
                np.all(np.abs(diff) < 6e-4),
                msg=f"arm {arm_deg}: recovered {recovered} vs {expected}",
            )

    def test_yaw_average_handles_quaternion_sign_flip_at_wrap(self):
        slew = math.radians(179.0)
        imu_quats = self._synthesize_imu_quats(slew, 0.0, -0.45, 0.35, -0.2)
        imu_quats[2] *= -1.0  # Same physical orientation, opposite quaternion hemisphere.
        recovered = joint_angles_from_imus(imu_quats, self.imu_cfg, self.rc)
        self.assertAlmostEqual(float(recovered[0]), float(slew), delta=2e-3)


class AxisRotationTests(unittest.TestCase):

    def test_identity_quaternion(self):
        idq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for axis in ([0, 0, 1], [0, 1, 0], [1, 0, 0]):
            self.assertAlmostEqual(
                extract_axis_rotation(idq, np.asarray(axis, np.float32)), 0.0, places=6
            )

    def test_pure_axis_recovers_angle(self):
        for axis in ((0, 0, 1.0), (0, 1.0, 0), (1.0, 0, 0)):
            for angle in (-2.5, -math.pi / 2, -0.1, 0.0, 0.4, math.pi / 2, 2.5):
                q = quat_from_axis_angle(np.asarray(axis, np.float32), np.float32(angle))
                recovered = extract_axis_rotation(q, np.asarray(axis, np.float32))
                self.assertAlmostEqual(recovered, angle, places=4,
                                       msg=f"axis={axis} angle={angle} → {recovered}")

    def test_hemisphere_invariance(self):
        q = quat_from_axis_angle(np.array([0, 1, 0], np.float32), np.float32(0.7))
        a = extract_axis_rotation(q, np.array([0, 1, 0], np.float32))
        b = extract_axis_rotation(-q, np.array([0, 1, 0], np.float32))
        self.assertAlmostEqual(a, b, places=5)

    def test_near_pi_does_not_flip(self):
        q = quat_from_axis_angle(np.array([0, 0, 1], np.float32),
                                 np.float32(math.radians(179.5)))
        ang = extract_axis_rotation(q, np.array([0, 0, 1], np.float32))
        self.assertAlmostEqual(ang, math.radians(179.5), places=4)

    def test_orthogonal_axis_returns_zero(self):
        q = quat_from_axis_angle(np.array([0, 0, 1], np.float32), np.float32(1.0))
        self.assertAlmostEqual(
            extract_axis_rotation(q, np.array([0, 1, 0], np.float32)), 0.0, places=5
        )

    def test_project_to_rotation_axes_preserves_pure_y(self):
        for ang in (-0.6, -0.1, 0.3, 0.9):
            q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32),
                                     np.float32(ang))
            out = project_to_rotation_axes(
                np.asarray([q]), np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
            )
            self.assertTrue(
                np.allclose(out[0], q, atol=1e-5),
                msg=f"projection changed pure Y quat at {ang}: {out[0]} vs {q}",
            )


class JacobianSanityTests(unittest.TestCase):
    """Compare the analytic position Jacobian against a numerical one."""

    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def _analytic_jacobian(self, angles):
        return np.asarray(get_state(angles, self.rc).jacobian, dtype=np.float64)

    def _compare(self, angles):
        angles = np.asarray(angles, dtype=np.float32)
        j_ana = self._analytic_jacobian(angles)[:3]
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
                _, _, diff = self._compare(angles)
                self.assertLess(
                    diff,
                    2e-3,
                    msg=f"yaw {deg} Jacobian mismatch {diff} at {angles}",
                )

    def test_state_jacobian_metrics_are_consistent(self):
        """``get_state.condition_number / yoshikawa`` agree with a fresh _jacobian_metrics call."""
        angles = np.array([0.4, -0.3, 0.2, -0.1], dtype=np.float32)
        state = get_state(angles, self.rc)
        cond, sv, yosh = _jacobian_metrics(state.jacobian)
        self.assertAlmostEqual(state.condition_number, cond, places=4)
        self.assertTrue(np.allclose(state.singular_values, sv, atol=1e-6))
        self.assertAlmostEqual(state.yoshikawa_index, yosh, places=4)

    def test_condition_number_finite_away_from_singularity(self):
        angles = np.array([0.1, -0.4, 0.3, -0.1], dtype=np.float32)
        cond = get_state(angles, self.rc).condition_number
        self.assertTrue(np.isfinite(cond))
        self.assertGreater(cond, 1.0)

    def test_fully_extended_pose_is_singular(self):
        """Fully extended (boom=arm=bucket=0) is a true singularity."""
        angles = np.zeros(self.rc.num_joints, dtype=np.float32)
        cond = get_state(angles, self.rc).condition_number
        self.assertTrue(np.isfinite(cond))
        self.assertGreater(
            cond, 1e3,
            msg=f"fully-extended pose should be near-singular, got cond={cond}",
        )

    def test_well_conditioned_pose_has_modest_condition_number(self):
        angles = np.array(
            [0.1, math.radians(-45.0), math.radians(30.0), math.radians(-15.0)],
            dtype=np.float32,
        )
        cond = get_state(angles, self.rc).condition_number
        self.assertLess(cond, 200.0,
                        msg=f"generic mid-workspace cond unexpectedly large: {cond}")

    def test_jacobian_orientation_rows_match_revolute_axes(self):
        """The bottom 3 rows of the Jacobian are the joint axes in world frame."""
        for slew_deg in (0, 45, 90, -90, 180):
            angles = np.array([math.radians(slew_deg), -0.4, 0.3, -0.1], dtype=np.float32)
            J = np.asarray(get_state(angles, self.rc).jacobian, dtype=np.float64)
            J_rot = J[3:6, :]

            # Joint 0 (slew) axis stays world Z.
            self.assertTrue(np.allclose(J_rot[:, 0], [0.0, 0.0, 1.0], atol=2e-5),
                            msg=f"slew axis at slew={slew_deg}: {J_rot[:, 0]}")
            # Joints 1-3 share the upper body, all axes are R_z(slew) @ Y.
            expected_axis = np.array([
                -math.sin(math.radians(slew_deg)),
                math.cos(math.radians(slew_deg)),
                0.0,
            ])
            for ji in (1, 2, 3):
                self.assertTrue(np.allclose(J_rot[:, ji], expected_axis, atol=2e-5),
                                msg=f"joint {ji} axis at slew={slew_deg}: {J_rot[:, ji]}")

    def test_jacobian_position_rows_orthogonal_to_axis_at_zero_pose(self):
        """At the zero pose, axis × (ee_pos - joint_pos) must be perpendicular to axis."""
        angles = np.zeros(self.rc.num_joints, dtype=np.float32)
        J = np.asarray(get_state(angles, self.rc).jacobian, dtype=np.float64)
        for ji in range(self.rc.num_joints):
            axis = J[3:6, ji]
            self.assertAlmostEqual(float(np.dot(axis, J[0:3, ji])), 0.0, places=5,
                                   msg=f"joint {ji} position not perp to axis")


class TranslationCovarianceTests(unittest.TestCase):
    """Shifting the first joint's ``parent_to_joint_xyz`` should shift every output."""

    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def test_shifted_base_translates_all_outputs(self):
        shift = np.array([0.5, -0.2, 1.0], dtype=np.float32)
        joint_specs = []
        for i, j in enumerate(self.rc.joints):
            offset = j.parent_to_joint_xyz.copy()
            if i == 0:
                offset = offset + shift
            joint_specs.append({
                "name": j.name,
                "axis": j.axis.tolist(),
                "parent_to_joint_xyz": offset.tolist(),
            })
        tool_spec = {"parent_to_tip_xyz": self.rc.tool.parent_to_tip_xyz.tolist()}
        rc_b = build_excavator_model(joint_specs, tool_spec)

        angles = np.array([0.2, -0.4, 0.3, -0.1], dtype=np.float32)
        sa = get_state(angles, self.rc, include_jacobian=False)
        sb = get_state(angles, rc_b, include_jacobian=False)

        shift64 = shift.astype(np.float64)
        for i in range(self.rc.num_joints):
            self.assertTrue(np.allclose(
                np.asarray(sb.joint_origins_world[i] - sa.joint_origins_world[i],
                           dtype=np.float64),
                shift64, atol=2e-5,
            ), msg=f"joint {i} shift mismatch")
        self.assertTrue(np.allclose(
            np.asarray(sb.ee_position - sa.ee_position, dtype=np.float64),
            shift64, atol=2e-5,
        ))


class ToolOffsetTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def test_bucket_rotation_moves_ee_when_tip_offset_nontrivial(self):
        """tool.parent_to_tip_xyz rotates with the bucket, so changing bucket moves EE."""
        angles_a = np.array([0.0, math.radians(-45), math.radians(30), 0.0], dtype=np.float32)
        angles_b = angles_a.copy()
        angles_b[3] = math.radians(60.0)
        ee_a, _ = fk_from_angles(angles_a, self.rc)
        ee_b, _ = fk_from_angles(angles_b, self.rc)
        if float(np.linalg.norm(self.rc.tip_offset)) > 1e-6:
            self.assertGreater(float(np.linalg.norm(ee_a - ee_b)), 1e-4,
                               msg="bucket rotation did not move EE — tip offset not applied?")


if __name__ == "__main__":
    unittest.main()
