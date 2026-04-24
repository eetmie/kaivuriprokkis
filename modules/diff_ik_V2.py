"""
Backward-compatibility shim.

All code has been split into:
- differential_ik_cfg.py  : Config dataclasses, robot model loading
- differential_ik.py      : IK solver, numba core math, axis helpers, Jacobian wrapper
- excavator_ik_utils.py   : FK wrappers, relative joint angles, warmup

This file re-exports every public name so existing consumers
(excavator_controller.py, robot_service.py, server_diagnostics.py)
continue to work without import changes.
"""

# Config & robot model
from .differential_ik_cfg import (
    IKControllerConfig,
    RobotConfig,
    CheckDOF,
    load_excavator_robot_config,
    create_excavator_config,
)

# IK solver, numba core, axis helpers, propagation
from .differential_ik import (
    IKController,
    extract_axis_rotation,
    project_to_rotation_axes,
    propagate_base_rotation,
    compute_jacobian,
    forward_kinematics_core,
    forward_kinematics_with_ee_offset_core,
    compute_jacobian_core,
    compute_condition_number,
    ik_method_pinv,
    ik_method_svd,
    ik_method_transpose,
    ik_method_damped_least_squares,
)

# Excavator-specific FK wrappers & utilities
from .excavator_ik_utils import (
    compute_relative_joint_angles,
    get_joint_positions,
    get_all_poses,
    get_pose,
    warmup_numba_functions,
)

# Re-export quaternion_math functions accessed via diff_ik module
from .quaternion_math import compute_pose_error, apply_delta_pose
