"""
Forward kinematics, Jacobian, and the single rich RobotKinematicState used
everywhere downstream of the IK.

URDF semantics::

    R = I, p = 0
    for i in range(n):
        p = p + R @ offsets[i]                  # joint origin (before its rotation)
        joint_origins_world[i] = p
        joint_axes_world[i]    = R @ axes[i]    # axis in world frame
        R = R @ rot_axis_angle(axes[i], q[i])
        link_orientations[i]   = R              # absolute orientation after joint i
    ee_pos  = p + R @ tip_offset
    ee_quat = R

Jacobian column i (revolute joint):
    J[0:3, i] = cross(joint_axes_world[i], ee_pos - joint_origins_world[i])
    J[3:6, i] = joint_axes_world[i]

A single ``get_state(q_rad, model)`` returns a frozen RobotKinematicState
with everything callers need (poses, Jacobian, metrics). This replaces the
old wrapper zoo (`get_pose_from_joint_angles`, `compute_jacobian_*`, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numba
import numpy as np

from .model import ExcavatorModel


@dataclass(frozen=True)
class RobotKinematicState:
    """One rich FK + Jacobian bundle from canonical joint angles in radians.

    All arrays are float32 and shape-stable for a given model. ``jacobian``,
    ``condition_number``, ``singular_values``, and ``yoshikawa_index`` are
    only populated when ``get_state(..., include_jacobian=True)``.
    """

    joint_names: tuple[str, ...]
    joint_angles_rad: np.ndarray         # (n,)
    joint_origins_world: np.ndarray      # (n, 3)
    joint_axes_world: np.ndarray         # (n, 3)
    link_orientations: np.ndarray        # (n, 4) absolute quats [w, x, y, z]
    ee_position: np.ndarray              # (3,)
    ee_orientation: np.ndarray           # (4,) equals link_orientations[-1]
    jacobian: Optional[np.ndarray]       # (6, n) or None
    condition_number: float
    singular_values: np.ndarray          # (k,) or empty
    yoshikawa_index: float


# ----------------------------
# Numba-optimized cores
# ----------------------------

@numba.njit(fastmath=True, nogil=True)
def _rot_axis_angle_quat(axis: np.ndarray, angle: np.float32) -> np.ndarray:
    """Quaternion from unit axis and angle. Inlinable in numba."""
    half = np.float32(0.5) * angle
    s = np.float32(np.sin(half))
    c = np.float32(np.cos(half))
    q = np.empty(4, dtype=np.float32)
    q[0] = c
    q[1] = s * axis[0]
    q[2] = s * axis[1]
    q[3] = s * axis[2]
    return q


@numba.njit(fastmath=True, nogil=True)
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b in [w, x, y, z] convention."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    out = np.empty(4, dtype=np.float32)
    out[0] = aw * bw - ax * bx - ay * by - az * bz
    out[1] = aw * bx + ax * bw + ay * bz - az * by
    out[2] = aw * by - ax * bz + ay * bw + az * bx
    out[3] = aw * bz + ax * by - ay * bx + az * bw
    return out


@numba.njit(fastmath=True, nogil=True)
def _quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply unit quaternion q to vector v: v' = q * v * q^-1."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # t = 2 * (qv × v)
    tx = np.float32(2.0) * (y * v[2] - z * v[1])
    ty = np.float32(2.0) * (z * v[0] - x * v[2])
    tz = np.float32(2.0) * (x * v[1] - y * v[0])
    out = np.empty(3, dtype=np.float32)
    out[0] = v[0] + w * tx + (y * tz - z * ty)
    out[1] = v[1] + w * ty + (z * tx - x * tz)
    out[2] = v[2] + w * tz + (x * ty - y * tx)
    return out


@numba.njit(fastmath=True, nogil=True)
def _fk_core(
    q: np.ndarray,
    axes: np.ndarray,
    offsets: np.ndarray,
    tip_offset: np.ndarray,
):
    """Single FK pass. Returns (joint_origins, joint_axes, link_quats, ee_pos, ee_quat)."""
    n = q.shape[0]
    joint_origins = np.zeros((n, 3), dtype=np.float32)
    joint_axes = np.zeros((n, 3), dtype=np.float32)
    link_quats = np.zeros((n, 4), dtype=np.float32)

    # Cumulative rotation as quaternion. Start at identity.
    R = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p = np.zeros(3, dtype=np.float32)

    for i in range(n):
        # Translate by parent-to-joint offset, expressed in current rotated frame.
        offset_world = _quat_rotate_vec(R, offsets[i])
        p = p + offset_world
        joint_origins[i] = p

        # Joint axis in world frame is parent's rotation applied to local axis.
        joint_axes[i] = _quat_rotate_vec(R, axes[i])

        # Apply this joint's rotation about the world-frame axis.
        # Equivalently R_new = R_parent * rot_local(axis_local, q).
        local_rot = _rot_axis_angle_quat(axes[i], np.float32(q[i]))
        R = _quat_mul(R, local_rot)
        # Renormalize to keep float32 drift bounded.
        nrm = np.sqrt(R[0] * R[0] + R[1] * R[1] + R[2] * R[2] + R[3] * R[3])
        if nrm > np.float32(1e-9):
            R = R / nrm
        link_quats[i] = R

    ee_pos = p + _quat_rotate_vec(R, tip_offset)
    ee_quat = R.copy()
    return joint_origins, joint_axes, link_quats, ee_pos, ee_quat


@numba.njit(fastmath=True, nogil=True)
def _jacobian_core(
    joint_origins_world: np.ndarray,
    joint_axes_world: np.ndarray,
    ee_pos: np.ndarray,
) -> np.ndarray:
    n = joint_origins_world.shape[0]
    J = np.zeros((6, n), dtype=np.float32)
    for i in range(n):
        z = joint_axes_world[i]
        r = ee_pos - joint_origins_world[i]
        # linear part = z × r
        J[0, i] = z[1] * r[2] - z[2] * r[1]
        J[1, i] = z[2] * r[0] - z[0] * r[2]
        J[2, i] = z[0] * r[1] - z[1] * r[0]
        # angular part = z
        J[3, i] = z[0]
        J[4, i] = z[1]
        J[5, i] = z[2]
    return J


@numba.njit(fastmath=True, nogil=True)
def _jacobian_metrics(jacobian: np.ndarray):
    """Return (condition_number, singular_values_f32, yoshikawa_index) from one SVD."""
    _, S, _ = np.linalg.svd(jacobian.astype(np.float64), full_matrices=False)
    cond = S[0] / (S[-1] + 1e-12)
    yoshikawa = np.float32(1.0)
    for s in S:
        yoshikawa *= np.float32(s)
    return float(cond), S.astype(np.float32), float(yoshikawa)


# ----------------------------
# Public API
# ----------------------------

def get_state(
    q_rad: np.ndarray,
    model: ExcavatorModel,
    include_jacobian: bool = True,
) -> RobotKinematicState:
    """Compute the rich kinematic state at joint angles ``q_rad``.

    Args:
        q_rad: canonical joint angles in radians, shape (model.num_joints,).
        model: ExcavatorModel describing the chain.
        include_jacobian: when False, skips the Jacobian + metrics
            computation. Useful for hot paths that only need poses.

    Returns:
        Frozen RobotKinematicState. Fields tied to the Jacobian
        (``jacobian``, ``condition_number``, ``singular_values``,
        ``yoshikawa_index``) are populated with None / 0.0 / empty when
        ``include_jacobian`` is False.
    """
    q = np.asarray(q_rad, dtype=np.float32)
    if q.shape != (model.num_joints,):
        raise ValueError(
            f"q_rad must have shape ({model.num_joints},), got {q.shape}"
        )

    joint_origins, joint_axes, link_quats, ee_pos, ee_quat = _fk_core(
        q, model.axes, model.offsets, model.tip_offset,
    )

    if include_jacobian:
        J = _jacobian_core(joint_origins, joint_axes, ee_pos)
        cond, S, yosh = _jacobian_metrics(J)
    else:
        J = None
        cond = 0.0
        S = np.zeros(0, dtype=np.float32)
        yosh = 0.0

    return RobotKinematicState(
        joint_names=model.joint_names,
        joint_angles_rad=q.copy(),
        joint_origins_world=joint_origins,
        joint_axes_world=joint_axes,
        link_orientations=link_quats,
        ee_position=ee_pos,
        ee_orientation=ee_quat,
        jacobian=J,
        condition_number=cond,
        singular_values=S,
        yoshikawa_index=yosh,
    )


def joint_angles_to_absolute_quaternions(
    q_rad: np.ndarray, model: ExcavatorModel,
) -> np.ndarray:
    """Compose absolute link quaternions from canonical joint angles.

    Thin convenience wrapper around the FK pass for the IMU/glue code that
    historically asked for the link-quat array directly.
    """
    q = np.asarray(q_rad, dtype=np.float32)
    if q.shape != (model.num_joints,):
        raise ValueError(
            f"q_rad must have shape ({model.num_joints},), got {q.shape}"
        )
    _, _, link_quats, _, _ = _fk_core(q, model.axes, model.offsets, model.tip_offset)
    return link_quats


# Re-export pure-Python helpers used by warmup / external callers that don't
# want to go through the dataclass.
__all__ = [
    "RobotKinematicState",
    "get_state",
    "joint_angles_to_absolute_quaternions",
]
