"""Excavator IK package.

Submodules:
    math       — quaternion + vector primitives (used everywhere)
    solver     — IKController, @njit FK/Jacobian cores, solver dispatch
    config     — IKControllerConfig, RobotConfig, YAML loader
    excavator  — IMU→canonical-joint-angle glue, FK wrappers, numba warmup

Public API is re-exported here so callers can do::

    from modules.ik import IKController, RobotConfig, canonical_joint_angles_from_imus

Submodule imports are still supported and are preferable in tight hot-path
code where you want the dependency to be visually explicit.

See REFACTOR_NOTES.md for in-progress follow-ups (split solver into
kinematics+solver, lock the chain in as excavator-specific).
"""

from .config import (
    IKControllerConfig,
    RobotConfig,
    CheckDOF,
    load_excavator_robot_config,
    create_excavator_config,
)
from .math import (
    normalize_vector,
    quat_normalize,
    quat_multiply,
    quat_conjugate,
    quat_rotate_vector,
    quat_from_axis_angle,
    axis_angle_from_quat,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    quat_unique,
    quat_enforce_hemisphere,
    quat_slerp,
    quat_remove_yaw,
    compute_quat_error,
    compute_pose_error,
    apply_delta_pose,
)
from .solver import (
    IKController,
    extract_axis_rotation,
    project_to_rotation_axes,
    propagate_base_rotation,
    forward_kinematics_core,
    forward_kinematics_with_ee_offset_core,
    compute_jacobian_core,
    compute_jacobian_metrics,
    compute_jacobian,
    compute_jacobian_from_joint_angles,
    joint_angles_to_absolute_quaternions,
    get_all_poses_from_joint_angles,
    get_pose_from_joint_angles,
    ik_method_pinv,
    ik_method_svd,
    ik_method_transpose,
    ik_method_damped_least_squares,
)
from .excavator import (
    average_axis_twist_quaternion,
    gravity_pitch_from_quat,
    canonical_joint_angles_from_imus,
    absolute_link_angles_from_quats,
    compute_relative_joint_angles,
    get_joint_positions,
    get_all_poses,
    get_pose,
    warmup_numba_functions,
)


__all__ = [
    # config
    "IKControllerConfig",
    "RobotConfig",
    "CheckDOF",
    "load_excavator_robot_config",
    "create_excavator_config",
    # math
    "normalize_vector",
    "quat_normalize",
    "quat_multiply",
    "quat_conjugate",
    "quat_rotate_vector",
    "quat_from_axis_angle",
    "axis_angle_from_quat",
    "quat_from_euler_xyz",
    "euler_xyz_from_quat",
    "quat_unique",
    "quat_enforce_hemisphere",
    "quat_slerp",
    "quat_remove_yaw",
    "compute_quat_error",
    "compute_pose_error",
    "apply_delta_pose",
    # solver
    "IKController",
    "extract_axis_rotation",
    "project_to_rotation_axes",
    "propagate_base_rotation",
    "forward_kinematics_core",
    "forward_kinematics_with_ee_offset_core",
    "compute_jacobian_core",
    "compute_jacobian_metrics",
    "compute_jacobian",
    "compute_jacobian_from_joint_angles",
    "joint_angles_to_absolute_quaternions",
    "get_all_poses_from_joint_angles",
    "get_pose_from_joint_angles",
    "ik_method_pinv",
    "ik_method_svd",
    "ik_method_transpose",
    "ik_method_damped_least_squares",
    # excavator glue
    "average_axis_twist_quaternion",
    "gravity_pitch_from_quat",
    "canonical_joint_angles_from_imus",
    "absolute_link_angles_from_quats",
    "compute_relative_joint_angles",
    "get_joint_positions",
    "get_all_poses",
    "get_pose",
    "warmup_numba_functions",
]
