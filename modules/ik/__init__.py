"""Excavator IK package — URDF-style kinematics + pure-functional IK step.

Submodules:
    math        — quaternion + vector primitives, axis-rotation helpers
    model       — Joint, Tool, ExcavatorModel, YAML loader (URDF-style)
    kinematics  — RobotKinematicState + get_state(q_rad, model)
    solver      — IKConfig, IKResult, solve_ik_step(...)
    excavator   — IMUConfig, joint_angles_from_imus(...), numba warmup

Typical use::

    from modules.ik import (
        load_excavator_model, load_imu_config,
        get_state, IKConfig, solve_ik_step,
        joint_angles_from_imus, warmup_numba_functions,
    )

    model   = load_excavator_model("configuration_files/profiles/jetson/control_config.yaml")
    imu_cfg = load_imu_config("configuration_files/profiles/jetson/control_config.yaml")
    ik_cfg  = IKConfig(method="dls", lambda_val=1e-3, ...)

    q       = joint_angles_from_imus(corrected_imu_quats, imu_cfg, model)
    result  = solve_ik_step(q, target_pos, target_quat, model, ik_cfg, dt=0.01)
    q_next  = result.q_next_rad
"""

from .math import (
    apply_delta_pose,
    axis_angle_from_quat,
    compute_pose_error,
    compute_quat_error,
    euler_xyz_from_quat,
    extract_axis_rotation,
    normalize_vector,
    project_to_rotation_axes,
    quat_conjugate,
    quat_enforce_hemisphere,
    quat_from_axis_angle,
    quat_from_euler_xyz,
    quat_multiply,
    quat_normalize,
    quat_remove_yaw,
    quat_rotate_vector,
    quat_slerp,
    quat_unique,
)
from .model import (
    ExcavatorModel,
    Joint,
    Tool,
    build_excavator_model,
    load_excavator_model,
)
from .kinematics import (
    RobotKinematicState,
    get_state,
    joint_angles_to_absolute_quaternions,
)
from .solver import (
    IKConfig,
    IKResult,
    solve_ik_step,
)
from .excavator import (
    IMUChainStep,
    IMUConfig,
    average_axis_twist_quaternion,
    build_imu_config,
    gravity_pitch_from_quat,
    joint_angles_from_imus,
    load_imu_config,
    warmup_numba_functions,
)


__all__ = [
    # math
    "apply_delta_pose",
    "axis_angle_from_quat",
    "compute_pose_error",
    "compute_quat_error",
    "euler_xyz_from_quat",
    "extract_axis_rotation",
    "normalize_vector",
    "project_to_rotation_axes",
    "quat_conjugate",
    "quat_enforce_hemisphere",
    "quat_from_axis_angle",
    "quat_from_euler_xyz",
    "quat_multiply",
    "quat_normalize",
    "quat_remove_yaw",
    "quat_rotate_vector",
    "quat_slerp",
    "quat_unique",
    # model
    "ExcavatorModel",
    "Joint",
    "Tool",
    "build_excavator_model",
    "load_excavator_model",
    # kinematics
    "RobotKinematicState",
    "get_state",
    "joint_angles_to_absolute_quaternions",
    # solver
    "IKConfig",
    "IKResult",
    "solve_ik_step",
    # excavator / IMU
    "IMUChainStep",
    "IMUConfig",
    "average_axis_twist_quaternion",
    "build_imu_config",
    "gravity_pitch_from_quat",
    "joint_angles_from_imus",
    "load_imu_config",
    "warmup_numba_functions",
]
