"""
Excavator-specific kinematics utilities.

Contains FK wrappers, relative joint angle computation, and numba warmup.
These functions handle the IMU-quaternion-based state representation
specific to the real excavator hardware.

Corresponds to the sim's excavator_state.py role, but without
cylindrical/radial coordinate support (to be added later).
"""

import numpy as np
from typing import Tuple

from .differential_ik_cfg import RobotConfig
from .differential_ik import (
    extract_axis_rotation,
    project_to_rotation_axes,
    propagate_base_rotation,
    forward_kinematics_core,
    forward_kinematics_with_ee_offset_core,
    compute_jacobian_core,
    compute_condition_number,
    ik_method_pinv,
    ik_method_svd,
    ik_method_transpose,
    ik_method_damped_least_squares,
)
from .quaternion_math import (
    quat_normalize, quat_multiply, quat_conjugate, quat_from_axis_angle,
)


def average_axis_twist_quaternion(quats: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Average the twist component of multiple quaternions about one axis.

    This keeps yaw/pitch extraction in quaternion space and avoids Euler-angle
    wrap issues.  Quaternion signs are hemisphere-aligned before summing so
    +179/-179 degree samples average to 180 instead of cancelling.
    """
    quats = np.asarray(quats, dtype=np.float32)
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    if quats.ndim != 2 or quats.shape[1] != 4 or len(quats) == 0:
        raise ValueError("Expected quats with shape (n, 4)")

    accum = np.zeros(4, dtype=np.float32)
    reference = None
    for q in quats:
        angle = extract_axis_rotation(q, axis)
        twist = quat_from_axis_angle(axis, np.float32(angle))
        if reference is None:
            reference = twist.copy()
        elif float(np.dot(reference, twist)) < 0.0:
            twist = -twist
        accum += twist

    if float(np.linalg.norm(accum)) < 1e-9:
        return reference if reference is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat_normalize(accum)


def canonical_joint_angles_from_imus(imu_quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """Convert four corrected absolute IMU quats to canonical joint angles.

    Input order is [base, boom, arm, bucket].  Common slew yaw is estimated by
    averaging the Z-axis twist from all four IMUs, then removed before extracting
    relative Y-axis hinge rotations.
    """
    imu_quats = np.asarray(imu_quats, dtype=np.float32)
    if imu_quats.shape != (4, 4):
        raise ValueError(f"Expected four IMU quaternions with shape (4, 4), got {imu_quats.shape}")

    z_axis = np.asarray(robot_config.rotation_axes[0], dtype=np.float32)
    slew_quat = average_axis_twist_quaternion(imu_quats, z_axis)
    slew_inv = quat_conjugate(slew_quat)

    body_quats = np.zeros_like(imu_quats, dtype=np.float32)
    for i in range(4):
        body_quats[i] = quat_normalize(quat_multiply(slew_inv, imu_quats[i]))

    angles = np.zeros(4, dtype=np.float32)
    angles[0] = extract_axis_rotation(slew_quat, z_axis)

    for i in range(1, 4):
        parent_inv = quat_conjugate(body_quats[i - 1])
        rel_quat = quat_normalize(quat_multiply(parent_inv, body_quats[i]))
        angles[i] = extract_axis_rotation(rel_quat, robot_config.rotation_axes[i])

    return angles


def absolute_link_angles_from_quats(
    quats: np.ndarray, robot_config: RobotConfig
) -> np.ndarray:
    """Per-link absolute angles in radians.

    For the excavator chain (slew about Z, boom/arm/bucket about body Y) this
    returns:
      - angles[0]  = world-frame slew yaw (twist of quats[0] about Z)
      - angles[i>=1] = cab-frame twist of link i about its own rotation axis,
        i.e. cumulative pitch from horizontal as a link-mounted inclinometer
        would read it.

    The slew yaw is removed before extracting downstream twists so the result
    is independent of cab heading. Generalises naturally to a future rototilt
    tool (roll about X, yaw about Z) because each joint's own rotation axis
    drives the extraction.

    Mounting offsets are NOT re-applied here — the hardware layer has already
    mounting-corrected the IMU quaternions (see ``imu.mounting_offsets_quat``
    in ``control_config.yaml`` and ``hardware_interface._correct_imu_quaternion``)
    before they reach the controller, so the canonical absolute link quats are
    the right inputs as-is.
    """
    quats = np.asarray(quats, dtype=np.float32)
    n = len(quats)
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out

    z_axis = np.asarray(robot_config.rotation_axes[0], dtype=np.float32)
    out[0] = extract_axis_rotation(quats[0], z_axis)

    if n > 1:
        slew_quat = quat_from_axis_angle(z_axis, np.float32(out[0]))
        slew_inv = quat_conjugate(slew_quat)
        for i in range(1, n):
            body_q = quat_normalize(quat_multiply(slew_inv, quats[i]))
            axis_i = np.asarray(robot_config.rotation_axes[i], dtype=np.float32)
            out[i] = extract_axis_rotation(body_q, axis_i)
    return out


def compute_relative_joint_angles(quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """
    Compute relative joint angles from absolute IMU quaternions.

    For a robot with absolute IMU orientations, joint limits are defined
    relative to the parent link. This function extracts the relative rotation
    about each joint's axis.

    Example for excavator:
    - Joint 0 (slew): Absolute yaw angle around Z-axis
    - Joint 1 (boom): Relative pitch from world horizontal (parent is slew, which only rotates in Z)
    - Joint 2 (arm): Relative pitch from boom orientation
    - Joint 3 (bucket): Relative pitch from arm orientation

    Args:
        quats: Absolute joint quaternions already corrected into joint frames
        robot_config: Robot configuration with rotation axes

    Returns:
        np.ndarray: Relative joint angles in radians [n_joints]
    """
    quats = np.asarray(quats, dtype=np.float32)

    n_joints = len(quats)
    relative_angles = np.zeros(n_joints, dtype=np.float32)

    # Joint 0 (slew): Extract absolute rotation about Z-axis
    relative_angles[0] = extract_axis_rotation(
        quats[0],
        robot_config.rotation_axes[0]
    )

    # Joints 1+ : Extract relative rotation from parent link
    for i in range(1, n_joints):
        # Get parent orientation (world frame)
        parent_quat = quats[i-1]
        current_quat = quats[i]

        # Compute relative orientation: q_rel = q_parent^-1 * q_current
        parent_quat_inv = quat_conjugate(parent_quat)
        relative_quat = quat_normalize(quat_multiply(parent_quat_inv, current_quat))

        # Extract rotation about this joint's axis (in parent's local frame)
        relative_angles[i] = extract_axis_rotation(
            relative_quat,
            robot_config.rotation_axes[i]
        )

    return relative_angles


def get_joint_positions(quats: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """Get joint positions from absolute link quats with origin_offset applied.

    Returns:
        np.ndarray: Joint positions [n x 3] including origin_offset, without end-effector offset
    """
    quats = np.asarray(quats, dtype=np.float32)
    return forward_kinematics_core(
        quats,
        robot_config.link_lengths,
        robot_config.link_directions,
        robot_config.origin_offset
    )


def get_all_poses(quats: np.ndarray, robot_config: RobotConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get all joint poses (positions + orientations) and end-effector pose.

    This is the most comprehensive FK function - returns everything computed.
    Zero overhead compared to get_pose() since all values are computed anyway.

    Args:
        quats: Absolute link quaternions from canonical joint angles
        robot_config: Robot configuration

    Returns:
        Tuple of:
        - joint_positions [n x 3]: Position of each joint (with origin_offset)
        - joint_orientations [n x 4]: Orientation of each joint [w, x, y, z]
        - ee_position [3]: End-effector position (with origin_offset + ee_offset)
        - ee_orientation [4]: End-effector orientation [w, x, y, z]
    """
    quats = np.asarray(quats, dtype=np.float32)

    # Get joint positions and ee_position with offsets
    joint_positions, ee_position = forward_kinematics_with_ee_offset_core(
        quats,
        robot_config.link_lengths,
        robot_config.link_directions,
        robot_config.origin_offset,
        robot_config.ee_offset
    )

    # Joint orientations are the supplied absolute link quaternions
    joint_orientations = quats.copy()

    # End-effector orientation is the last joint's orientation
    ee_orientation = quats[-1].copy()

    return joint_positions, joint_orientations, ee_position, ee_orientation


def get_pose(quats: np.ndarray, robot_config: RobotConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Get end-effector pose (position and orientation) only.

    Convenience wrapper around get_all_poses() for when you only need EE pose.

    Args:
        quats: Absolute link quaternions from canonical joint angles
        robot_config: Robot configuration

    Returns:
        Tuple of (ee_position [x, y, z], ee_orientation [w, x, y, z])
    """
    _, _, ee_position, ee_orientation = get_all_poses(quats, robot_config)
    return ee_position, ee_orientation


def warmup_numba_functions():
    """
    Warmup Numba JIT compilation by calling functions with dummy data.
    This prevents compilation delays during actual operation.
    """
    # Create dummy data with correct types
    dummy_quats = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.0, 0.3, 0.0],
        [0.95, 0.0, 0.2, 0.0]
    ], dtype=np.float32)

    dummy_link_lengths = np.array([0.2, 0.15, 0.1], dtype=np.float32)
    dummy_link_directions = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float32)
    dummy_rotation_axes = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    dummy_origin_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dummy_ee_offset = np.array([0.0, 0.0, -0.1], dtype=np.float32)

    # Warmup all Numba functions
    try:
        dummy_jacobian = np.random.rand(6, 3).astype(np.float32)
        dummy_delta_pose = np.random.rand(6).astype(np.float32)

        for _ in range(3):  # Multiple calls to ensure compilation
            # Forward kinematics functions
            _ = forward_kinematics_core(dummy_quats, dummy_link_lengths, dummy_link_directions, dummy_origin_offset)
            _ = forward_kinematics_with_ee_offset_core(dummy_quats, dummy_link_lengths, dummy_link_directions, dummy_origin_offset, dummy_ee_offset)
            _ = compute_jacobian_core(dummy_quats, dummy_link_lengths, dummy_link_directions, dummy_rotation_axes, dummy_origin_offset, dummy_ee_offset)

            # IK method functions
            _ = ik_method_pinv(dummy_jacobian, dummy_delta_pose, np.float32(1.0))
            _ = ik_method_svd(dummy_jacobian, dummy_delta_pose, np.float32(1.0), np.float32(1e-5))
            _ = ik_method_transpose(dummy_jacobian, dummy_delta_pose, np.float32(1.0))
            _ = ik_method_damped_least_squares(dummy_jacobian, dummy_delta_pose, np.float32(0.1))
            _ = compute_condition_number(dummy_jacobian)
    except Exception as e:
        print(f"Numba warmup failed: {e}")
