"""Shared kinematics helpers for IK/FK/radial tests.

These helpers let tests operate on the real robot's differential IK stack
without needing physical IMU feedback — we generate absolute link
quaternions and FK poses directly from a joint-angle vector via the new
URDF-style ``ExcavatorModel`` + ``get_state`` / ``solve_ik_step`` API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.ik import (  # noqa: E402
    ExcavatorModel,
    IKConfig,
    get_state,
    load_excavator_model,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    solve_ik_step,
    warmup_numba_functions,
)


# ----------------------------
# FK helpers (delegate to get_state)
# ----------------------------

def build_absolute_quaternions(
    joint_angles: np.ndarray,
    model: ExcavatorModel,
) -> np.ndarray:
    """Compose absolute (world-frame) quaternions from relative joint angles.

    Returns the link_orientations array of shape (n_joints, 4) float32
    [w, x, y, z]. Equivalent to ``get_state(q, model).link_orientations``.
    """
    state = get_state(joint_angles, model, include_jacobian=False)
    return np.asarray(state.link_orientations, dtype=np.float32).copy()


def fk_from_angles(
    joint_angles: np.ndarray,
    model: ExcavatorModel,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ee_pos, ee_quat) for a given joint-angle vector."""
    state = get_state(joint_angles, model, include_jacobian=False)
    return (
        np.asarray(state.ee_position, dtype=np.float32).copy(),
        np.asarray(state.ee_orientation, dtype=np.float32).copy(),
    )


def joint_positions_from_angles(
    joint_angles: np.ndarray,
    model: ExcavatorModel,
) -> np.ndarray:
    """Return per-joint world-frame origins as a (n_joints, 3) float32 array."""
    state = get_state(joint_angles, model, include_jacobian=False)
    return np.asarray(state.joint_origins_world, dtype=np.float32).copy()


def numerical_jacobian(
    joint_angles: np.ndarray,
    model: ExcavatorModel,
    eps: float = 1e-4,
) -> np.ndarray:
    """3xN numeric position Jacobian via central differencing of FK."""
    joint_angles = np.asarray(joint_angles, dtype=np.float64)
    n = len(joint_angles)
    jac = np.zeros((3, n), dtype=np.float64)
    for i in range(n):
        plus = joint_angles.copy()
        minus = joint_angles.copy()
        plus[i] += eps
        minus[i] -= eps
        pp, _ = fk_from_angles(plus.astype(np.float32), model)
        pm, _ = fk_from_angles(minus.astype(np.float32), model)
        jac[:, i] = (pp.astype(np.float64) - pm.astype(np.float64)) / (2 * eps)
    return jac


# ----------------------------
# IK rollout driver (replaces stateful IKController loop)
# ----------------------------

def simulate_ik(
    model: ExcavatorModel,
    ik_cfg: IKConfig,
    start_angles: np.ndarray,
    target_pos: np.ndarray,
    target_rot_y_deg: float = 0.0,  # kept for back-compat; only used when pose_mode=True
    max_iters: int = 400,
    pos_tol: float = 2e-3,
    dt: float = 0.005,
    pose_mode: bool = False,
) -> dict:
    """Iteratively drive ``solve_ik_step`` toward a Cartesian target.

    By default this is **position-only** mode (target_quat=None), matching
    the historical test expectation. Set ``pose_mode=True`` to compose a
    pitch-follows-slew target orientation each tick.
    """
    target_pos = np.asarray(target_pos, dtype=np.float32)
    angles = np.asarray(start_angles, dtype=np.float32).copy()

    pitch_quat = quat_from_axis_angle(
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.float32(np.radians(target_rot_y_deg)),
    )

    history = []

    for i in range(max_iters):
        state = get_state(angles, model)
        ee_pos = np.asarray(state.ee_position, dtype=np.float32)
        history.append(ee_pos.copy())

        err = float(np.linalg.norm(target_pos - ee_pos))
        if err < pos_tol:
            return {
                "converged": True,
                "iters": i,
                "final_angles": angles.copy(),
                "final_pos": ee_pos.copy(),
                "pos_error": err,
                "history": history,
            }

        if pose_mode:
            slew_quat = quat_from_axis_angle(
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                np.float32(angles[0]),
            )
            target_quat = quat_normalize(quat_multiply(slew_quat, pitch_quat))
        else:
            target_quat = None

        result = solve_ik_step(
            angles, target_pos, target_quat, model, ik_cfg, dt=dt, state=state,
        )
        angles = np.asarray(result.q_next_rad, dtype=np.float32)

    # Final FK after last update
    state = get_state(angles, model, include_jacobian=False)
    ee_pos = np.asarray(state.ee_position, dtype=np.float32)
    err = float(np.linalg.norm(target_pos - ee_pos))
    return {
        "converged": err < pos_tol,
        "iters": max_iters,
        "final_angles": angles.copy(),
        "final_pos": ee_pos.copy(),
        "pos_error": err,
        "history": history,
    }


# ----------------------------
# Config builders
# ----------------------------

def make_ik_config(
    method: str = "dls",
    command_type: str = "position",  # 'position' or 'pose'; controls target_quat in callers
    enable_velocity_limiting: bool = False,
    max_joint_velocities=None,
    joint_limits=None,
    enable_joint_limit_avoidance: bool = False,
    ik_params: Optional[dict] = None,
    enable_adaptive_damping: bool = True,
    condition_number_threshold: float = 0.0,
) -> IKConfig:
    """Build a reasonable IKConfig for tests.

    Note: the legacy ``command_type``, ``use_relative_mode``, ``velocity_mode``
    knobs are gone from the solver. ``command_type`` is accepted here only
    so the rest of the test suite can still pass the keyword; it has no
    effect on the produced IKConfig (callers choose position vs pose by
    passing ``target_quat=None`` or not).
    """
    if ik_params is None:
        ik_params = {
            "k_val": 1.0,
            "min_singular_value": 1e-4,
            "lambda_val": 0.02,
        }

    # Adapt legacy list-of-pairs joint_limits to the new tuple-of-pairs/None form.
    if joint_limits is not None:
        adapted_limits = []
        for p in joint_limits:
            if p is None:
                adapted_limits.append(None)
            elif isinstance(p, (list, tuple)) and len(p) == 2:
                a, b = float(p[0]), float(p[1])
                if not np.isfinite(a) or not np.isfinite(b):
                    adapted_limits.append(None)
                else:
                    adapted_limits.append((a, b))
            else:
                adapted_limits.append(None)
        joint_limits_tuple = tuple(adapted_limits)
    else:
        joint_limits_tuple = None

    if max_joint_velocities is not None:
        max_jv_tuple = tuple(float(v) for v in max_joint_velocities)
    else:
        max_jv_tuple = None

    return IKConfig(
        method=method,
        k_val=float(ik_params.get("k_val", 1.0)),
        lambda_val=float(ik_params.get("lambda_val", 0.02)),
        min_singular_value=float(ik_params.get("min_singular_value", 1e-4)),
        joint_weights=None,
        joint_limits_rad=joint_limits_tuple,
        max_joint_velocity_rad_per_step=max_jv_tuple,
        enable_velocity_limiting=enable_velocity_limiting,
        enable_joint_limit_avoidance=enable_joint_limit_avoidance,
        enable_adaptive_damping=enable_adaptive_damping,
        adaptive_damping_max_multiplier=10.0,
        condition_number_threshold=condition_number_threshold,
    )


def load_robot_config() -> ExcavatorModel:
    """Ensure numba is warm and return the excavator URDF model.

    Kept under the legacy name so existing tests need fewer touch-ups.
    Returns the new ``ExcavatorModel``.
    """
    warmup_numba_functions()
    return load_excavator_model(
        "configuration_files/profiles/rpi/control_config.yaml"
    )
