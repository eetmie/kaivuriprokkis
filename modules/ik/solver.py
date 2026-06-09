"""
Differential inverse kinematics step.

Public surface (pure-functional):

    solve_ik_step(q_rad, target_pos, target_quat, model, ik_cfg, dt,
                  state=None) -> IKResult

The caller holds its own target/desired pose and joint state; this module
takes a snapshot, returns the next joint angles plus rich diagnostics, and
keeps no internal state. Use ``state=...`` to reuse a RobotKinematicState
already computed for the same q (so reachability/IK/telemetry share one FK
pass).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numba
import numpy as np

from .kinematics import (
    RobotKinematicState,
    _jacobian_metrics,
    get_state,
)
from .math import (
    compute_pose_error,
    extract_axis_rotation,
)
from .model import ExcavatorModel


# ----------------------------
# Config + result dataclasses
# ----------------------------

@dataclass(frozen=True)
class IKConfig:
    """IK solver configuration. All numerical settings the solver needs."""

    method: str = "dls"                       # 'pinv' | 'svd' | 'trans' | 'dls'
    k_val: float = 1.0
    lambda_val: float = 1e-3
    min_singular_value: float = 1e-5
    joint_weights: Optional[Tuple[float, ...]] = None

    # Per-joint kinematic limits (radians). None per joint = unbounded.
    joint_limits_rad: Optional[Tuple[Optional[Tuple[float, float]], ...]] = None

    # Per-joint velocity cap (radians per IK step). None = no per-joint cap.
    max_joint_velocity_rad_per_step: Optional[Tuple[float, ...]] = None

    enable_velocity_limiting: bool = True
    enable_joint_limit_avoidance: bool = True

    # Damped least squares adaptive damping
    enable_adaptive_damping: bool = True
    adaptive_damping_max_multiplier: float = 2.0
    condition_number_threshold: float = 40.0

    def __post_init__(self):
        if self.method not in ("pinv", "svd", "trans", "dls"):
            raise ValueError(f"IKConfig.method must be one of pinv/svd/trans/dls, got {self.method!r}")


@dataclass(frozen=True)
class IKResult:
    """Rich result of one IK step."""

    q_next_rad: np.ndarray             # (n,) clipped, next joint angles
    dq_rad: np.ndarray                 # (n,) applied delta this step
    task_error: np.ndarray             # reduced-DOF error fed to the solver
    rejected: bool                     # True when condition number gated
    reason: Optional[str]              # human-readable reason when rejected
    state: RobotKinematicState         # FK/Jacobian at input q (pre-step)
    adaptive_lambda: float             # damping actually used (DLS only)


# ----------------------------
# Numba IK kernels
# ----------------------------

@numba.njit(fastmath=True, nogil=True)
def _ik_pinv(J: np.ndarray, e: np.ndarray, k_val: np.float32) -> np.ndarray:
    return k_val * np.dot(np.linalg.pinv(J.astype(np.float32)), e.astype(np.float32))


@numba.njit(fastmath=True, nogil=True)
def _ik_svd(J: np.ndarray, e: np.ndarray, k_val: np.float32, min_sv: np.float32) -> np.ndarray:
    J32 = J.astype(np.float32)
    e32 = e.astype(np.float32)
    U, S, Vh = np.linalg.svd(J32, full_matrices=False)
    ut_e = np.dot(U.T, e32)
    out = np.zeros(J32.shape[1], dtype=np.float32)
    for i in range(len(S)):
        if S[i] > min_sv:
            out += Vh[i] * (ut_e[i] / S[i])
    return k_val * out


@numba.njit(fastmath=True, nogil=True)
def _ik_trans(J: np.ndarray, e: np.ndarray, k_val: np.float32) -> np.ndarray:
    return k_val * np.dot(J.astype(np.float32).T, e.astype(np.float32))


@numba.njit(fastmath=True, nogil=True)
def _ik_dls(J: np.ndarray, e: np.ndarray, lam: np.float32) -> np.ndarray:
    J32 = J.astype(np.float32)
    e32 = e.astype(np.float32)
    JJt = np.dot(J32, J32.T)
    A = JJt + (lam * lam) * np.eye(JJt.shape[0], dtype=np.float32)
    sol = np.linalg.solve(A, e32)
    return np.dot(J32.T, sol)


# ----------------------------
# Helpers
# ----------------------------

def _controllable_dofs(model: ExcavatorModel) -> Tuple[int, ...]:
    """Which 6D DOFs the chain can actually excite.

    Position (rows 0..2) is always controllable for any non-trivial chain.
    Rotation (rows 3..5) is included when any joint's local axis dominates
    the corresponding global direction (cos(30°) threshold). For the
    standard excavator (slew=Z, boom/arm/bucket=Y) this yields
    [x, y, z, pitch, yaw] = (0, 1, 2, 4, 5).
    """
    dofs = [0, 1, 2]
    has_x = False
    has_y = False
    has_z = False
    for axis in model.axes:
        a = axis / (float(np.linalg.norm(axis)) + 1e-12)
        if abs(a[0]) > 0.866:
            has_x = True
        if abs(a[1]) > 0.866:
            has_y = True
        if abs(a[2]) > 0.866:
            has_z = True
    if has_x:
        dofs.append(3)
    if has_y:
        dofs.append(4)
    if has_z:
        dofs.append(5)
    return tuple(dofs)


def _adaptive_lambda(ik_cfg: IKConfig, condition_number: float) -> float:
    """DLS adaptive damping (matches old IKController behavior)."""
    base = float(ik_cfg.lambda_val)
    if not ik_cfg.enable_adaptive_damping:
        return base
    thr = float(ik_cfg.condition_number_threshold)
    if thr <= 0.0:
        return base
    scaling = (ik_cfg.adaptive_damping_max_multiplier - 1.0) / np.log(1.0 + thr)
    lam = base * (1.0 + scaling * np.log(1.0 + max(condition_number, 0.0)))
    return float(np.clip(lam, base, base * ik_cfg.adaptive_damping_max_multiplier))


def _solve_dq(
    J: np.ndarray, e: np.ndarray, ik_cfg: IKConfig, adaptive_lambda: float,
) -> np.ndarray:
    """Dispatch to the chosen IK kernel with optional joint weighting."""
    J = np.asarray(J, dtype=np.float32)
    e = np.asarray(e, dtype=np.float32)

    if ik_cfg.joint_weights is not None:
        w = np.clip(np.asarray(ik_cfg.joint_weights, dtype=np.float32), 1e-6, None)
        w_sqrt = np.diag(np.sqrt(w))
        J_weighted = J @ w_sqrt
    else:
        w_sqrt = None
        J_weighted = J

    method = ik_cfg.method
    if method == "pinv":
        dq = _ik_pinv(J_weighted, e, np.float32(ik_cfg.k_val))
    elif method == "svd":
        dq = _ik_svd(
            J_weighted, e,
            np.float32(ik_cfg.k_val),
            np.float32(ik_cfg.min_singular_value),
        )
    elif method == "trans":
        dq = _ik_trans(J_weighted, e, np.float32(ik_cfg.k_val))
    else:  # dls
        dq = _ik_dls(J_weighted, e, np.float32(adaptive_lambda))

    if w_sqrt is not None:
        dq = w_sqrt @ dq
    return dq


def _apply_velocity_limit(
    dq: np.ndarray, ik_cfg: IKConfig, default_cap: float = 0.035,
) -> np.ndarray:
    """Clip per-joint delta. default_cap ≈ 2 deg if no per-joint cap."""
    if not ik_cfg.enable_velocity_limiting:
        return dq
    if ik_cfg.max_joint_velocity_rad_per_step is None:
        cap = np.full_like(dq, default_cap)
    else:
        cap = np.asarray(ik_cfg.max_joint_velocity_rad_per_step, dtype=np.float32)
    return np.clip(dq, -cap, cap)


def _apply_limit_avoidance(
    dq: np.ndarray,
    q: np.ndarray,
    ik_cfg: IKConfig,
    repulsion_strength: float = 0.1,
    margin_fraction: float = 0.15,
) -> np.ndarray:
    """Push away from joint limits when within the margin band."""
    if not ik_cfg.enable_joint_limit_avoidance or ik_cfg.joint_limits_rad is None:
        return dq
    out = dq.copy()
    for i, lim in enumerate(ik_cfg.joint_limits_rad):
        if lim is None:
            continue
        q_min, q_max = lim
        q_range = q_max - q_min
        margin = margin_fraction * q_range
        if q[i] < q_min + margin:
            r = np.clip((q[i] - q_min) / margin, 0.0, 1.0)
            out[i] += repulsion_strength * (1.0 - r) ** 2
        elif q[i] > q_max - margin:
            r = np.clip((q_max - q[i]) / margin, 0.0, 1.0)
            out[i] -= repulsion_strength * (1.0 - r) ** 2
    return out


def _clip_to_limits(q_next: np.ndarray, ik_cfg: IKConfig) -> np.ndarray:
    if ik_cfg.joint_limits_rad is None:
        return q_next
    out = q_next.copy()
    for i, lim in enumerate(ik_cfg.joint_limits_rad):
        if lim is None:
            continue
        q_min, q_max = lim
        out[i] = np.clip(out[i], q_min, q_max)
    return out


# ----------------------------
# Public API
# ----------------------------

def solve_ik_step(
    q_rad: np.ndarray,
    target_pos: np.ndarray,
    target_quat: Optional[np.ndarray],
    model: ExcavatorModel,
    ik_cfg: IKConfig,
    dt: float = 0.01,
    state: Optional[RobotKinematicState] = None,
) -> IKResult:
    """One differential IK step toward an absolute Cartesian target.

    Args:
        q_rad: current canonical joint angles in radians.
        target_pos: desired end-effector position (world frame, meters).
        target_quat: desired end-effector orientation (world frame, [w,x,y,z])
            or None for position-only mode.
        model: ExcavatorModel.
        ik_cfg: IKConfig.
        dt: control timestep in seconds (currently unused except for
            future velocity integration; kept for API stability).
        state: optional precomputed RobotKinematicState for q_rad. When
            None, get_state(..., include_jacobian=True) is called.

    Returns:
        IKResult.
    """
    q = np.asarray(q_rad, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    if target_quat is not None:
        target_quat = np.asarray(target_quat, dtype=np.float32)

    if state is None:
        state = get_state(q, model, include_jacobian=True)
    if state.jacobian is None:
        # Caller passed a state with no Jacobian — recompute.
        state = get_state(q, model, include_jacobian=True)

    J_full = state.jacobian.copy()

    # Express Cartesian quantities in the slew-rotated body (cab) frame.
    # This decouples slew yaw from x/y position commands so a body-frame
    # pitch error stays in the pitch row of the reduced Jacobian at any
    # slew angle.
    slew_axis = model.axes[0]
    slew_angle = extract_axis_rotation(state.link_orientations[0], slew_axis)
    c = float(np.cos(-slew_angle))
    s = float(np.sin(-slew_angle))
    R_body = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    cur_pos_body = R_body @ state.ee_position
    tgt_pos_body = R_body @ target_pos

    J_full[:3, :] = R_body @ J_full[:3, :]
    J_full[3:, :] = R_body @ J_full[3:, :]

    dofs = _controllable_dofs(model)
    J_reduced = J_full[list(dofs), :]

    if target_quat is None:
        position_error = tgt_pos_body - cur_pos_body
        # Position-only: take the first 3 rows of the reduced Jacobian.
        # Position rows are guaranteed to be at indices 0..2 of `dofs`.
        J_task = J_reduced[:3, :]
        task_error = position_error
    else:
        pos_err, rot_err = compute_pose_error(
            state.ee_position, state.ee_orientation, target_pos, target_quat,
        )
        # Rotate orientation error into body frame to match J orientation rows.
        rot_err = (R_body @ np.asarray(rot_err, dtype=np.float32)).astype(np.float32)
        # Position error gets rotated too (we already work in body frame).
        pos_err_body = tgt_pos_body - cur_pos_body
        full_err = np.concatenate([pos_err_body, rot_err]).astype(np.float32)
        task_error = full_err[list(dofs)]
        J_task = J_reduced

    # Condition-number gating + adaptive damping.
    cond_J, _, _ = _jacobian_metrics(J_task)
    rejected = False
    reason: Optional[str] = None
    thr = float(ik_cfg.condition_number_threshold)
    if thr > 0.0 and cond_J > thr:
        rejected = True
        reason = f"condition_number={cond_J:.2f} exceeds threshold {thr:.2f}"

    adaptive_lambda = _adaptive_lambda(ik_cfg, cond_J)

    if rejected:
        dq = np.zeros_like(q)
    else:
        dq = _solve_dq(J_task, task_error, ik_cfg, adaptive_lambda).astype(np.float32)
        dq = _apply_velocity_limit(dq, ik_cfg)
        dq = _apply_limit_avoidance(dq, q, ik_cfg)

    q_next = _clip_to_limits(q + dq, ik_cfg)

    return IKResult(
        q_next_rad=q_next,
        dq_rad=dq,
        task_error=task_error,
        rejected=rejected,
        reason=reason,
        state=state,
        adaptive_lambda=float(adaptive_lambda),
    )


__all__ = [
    "IKConfig",
    "IKResult",
    "solve_ik_step",
]
