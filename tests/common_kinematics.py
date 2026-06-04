"""Shared kinematics helpers for IK/FK/radial tests.

These helpers let tests operate on the real robot's differential IK stack
without needing physical IMU feedback — we generate absolute joint
quaternions directly from a joint-angle vector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.ik import (  # noqa: E402
    IKControllerConfig, RobotConfig, load_excavator_robot_config,
    IKController,
    get_joint_positions, get_pose, compute_relative_joint_angles,
    warmup_numba_functions,
    quat_from_axis_angle, quat_multiply, quat_normalize,
)


def build_absolute_quaternions(
    joint_angles: np.ndarray,
    robot_config: RobotConfig,
) -> np.ndarray:
    """Compose absolute (world-frame) quaternions from relative joint angles.

    The IRL stack consumes ALREADY corrected absolute IMU quaternions, so for
    tests we build them by composing rotations along the kinematic chain.

    Args:
        joint_angles: Relative joint angles in radians [slew, boom, arm, bucket].
        robot_config: Robot model (rotation axes).

    Returns:
        np.ndarray of shape (n_joints, 4) float32 [w, x, y, z].
    """
    joint_angles = np.asarray(joint_angles, dtype=np.float32)
    axes = robot_config.rotation_axes

    absolute = np.zeros((len(joint_angles), 4), dtype=np.float32)

    # Slew is absolute about Z (matches what excavator_controller feeds from encoder).
    absolute[0] = quat_from_axis_angle(axes[0], np.float32(joint_angles[0]))

    # Downstream joints are composed: q_abs[i] = q_abs[i-1] * q_local[i]
    for i in range(1, len(joint_angles)):
        local = quat_from_axis_angle(axes[i], np.float32(joint_angles[i]))
        absolute[i] = quat_normalize(quat_multiply(absolute[i - 1], local))

    return absolute


def fk_from_angles(
    joint_angles: np.ndarray,
    robot_config: RobotConfig,
):
    """Return (ee_pos, ee_quat) for a given relative joint-angle vector."""
    quats = build_absolute_quaternions(joint_angles, robot_config)
    ee_pos, ee_quat = get_pose(quats, robot_config)
    return np.asarray(ee_pos, dtype=np.float32), np.asarray(ee_quat, dtype=np.float32)


def joint_positions_from_angles(
    joint_angles: np.ndarray,
    robot_config: RobotConfig,
) -> np.ndarray:
    quats = build_absolute_quaternions(joint_angles, robot_config)
    return np.asarray(
        get_joint_positions(quats, robot_config), dtype=np.float32
    )


def numerical_jacobian(
    joint_angles: np.ndarray,
    robot_config: RobotConfig,
    eps: float = 1e-4,
) -> np.ndarray:
    """3xN numeric position Jacobian via central differencing of FK.

    Only the linear (position) rows are returned — that is what pairs with
    how the analytic Jacobian is used by the IK solver for position control.
    """
    joint_angles = np.asarray(joint_angles, dtype=np.float64)
    n = len(joint_angles)
    jac = np.zeros((3, n), dtype=np.float64)
    for i in range(n):
        plus = joint_angles.copy()
        minus = joint_angles.copy()
        plus[i] += eps
        minus[i] -= eps
        pp, _ = fk_from_angles(plus.astype(np.float32), robot_config)
        pm, _ = fk_from_angles(minus.astype(np.float32), robot_config)
        jac[:, i] = (pp.astype(np.float64) - pm.astype(np.float64)) / (2 * eps)
    return jac


def simulate_ik(
    robot_config,
    ik_controller,
    start_angles: np.ndarray,
    target_pos: np.ndarray,
    target_rot_y_deg: float = 0.0,
    max_iters: int = 400,
    pos_tol: float = 2e-3,
    dt: float = 0.005,
):
    """Iteratively drive IK toward a Cartesian target and return trace.

    Each iteration:
      1. Compute FK from current joint angles.
      2. Call ik_controller.compute(...) to get the next joint-angle vector.
      3. Write that straight back (no PID / no hardware delays).

    Returns dict with ``converged``, ``iters``, ``final_angles``, ``final_pos``,
    ``pos_error``, ``history`` (positions per iter).
    """
    from modules.ik import quat_from_axis_angle

    ik_controller.ee_pos_des = np.asarray(target_pos, dtype=np.float32)
    ik_controller.ee_quat_des = quat_from_axis_angle(
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.float32(np.radians(target_rot_y_deg)),
    )

    angles = np.asarray(start_angles, dtype=np.float32).copy()
    history = []

    for i in range(max_iters):
        quats = build_absolute_quaternions(angles, robot_config)
        ee_pos, ee_quat = get_pose(quats, robot_config)
        history.append(np.asarray(ee_pos, dtype=np.float32).copy())

        err = np.linalg.norm(np.asarray(target_pos, dtype=np.float32) - ee_pos)
        if err < pos_tol:
            return {
                "converged": True,
                "iters": i,
                "final_angles": angles.copy(),
                "final_pos": np.asarray(ee_pos, dtype=np.float32).copy(),
                "pos_error": float(err),
                "history": history,
            }

        rel_angles = compute_relative_joint_angles(quats, robot_config)
        new_angles = ik_controller.compute(
            ee_pos=np.asarray(ee_pos, dtype=np.float32),
            ee_quat=np.asarray(ee_quat, dtype=np.float32),
            joint_angles=np.asarray(rel_angles, dtype=np.float32),
            joint_quats=quats,
            dt=dt,
        )
        angles = np.asarray(new_angles, dtype=np.float32)

    # Final FK after last update
    quats = build_absolute_quaternions(angles, robot_config)
    ee_pos, _ = get_pose(quats, robot_config)
    err = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float32) - ee_pos))
    return {
        "converged": err < pos_tol,
        "iters": max_iters,
        "final_angles": angles.copy(),
        "final_pos": np.asarray(ee_pos, dtype=np.float32).copy(),
        "pos_error": err,
        "history": history,
    }


def make_ik_config(
    method: str = "dls",
    command_type: str = "position",
    velocity_mode: bool = False,
    enable_velocity_limiting: bool = False,
    max_joint_velocities=None,
    joint_limits=None,
    enable_joint_limit_avoidance: bool = False,
    ik_params=None,
    enable_adaptive_damping: bool = True,
) -> IKControllerConfig:
    """Build a reasonable IK config for tests."""
    if ik_params is None:
        ik_params = {
            "k_val": 1.0,
            "min_singular_value": 1e-4,
            "lambda_val": 0.02,
        }
    return IKControllerConfig(
        command_type=command_type,
        ik_method=method,
        use_rotational_velocity=False,
        use_relative_mode=False,
        velocity_mode=velocity_mode,
        ik_params=ik_params,
        enable_velocity_limiting=enable_velocity_limiting,
        max_joint_velocities=max_joint_velocities,
        joint_limits=joint_limits,
        enable_adaptive_damping=enable_adaptive_damping,
        adaptive_damping_max_multiplier=10.0,
        enable_joint_limit_avoidance=enable_joint_limit_avoidance,
    )


def load_robot_config():
    """Ensure numba is warm and return the excavator robot config."""
    warmup_numba_functions()
    return load_excavator_robot_config("configuration_files/profiles/rpi/control_config.yaml")
