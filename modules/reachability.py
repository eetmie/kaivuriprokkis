"""Pre-flight reachability check for the excavator IK target.

Runs a simulated IK rollout from the current joint state, returning the
closest pose the arm can reach. Used by ``ExcavatorController.give_pose``
to reject targets the geometry cannot satisfy before any hydraulic motion
starts.

The rollout uses the same :func:`modules.ik.solve_ik_step` and
:func:`modules.ik.get_state` as live control so it does not drift from
runtime behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ik import (
    ExcavatorModel,
    IKConfig,
    get_state,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    solve_ik_step,
)


@dataclass(frozen=True)
class ReachabilityResult:
    reachable: bool
    closest_position: np.ndarray
    pos_error_m: float
    iters: int
    final_cond_number: float


def _stall_detected(history: list, window: int, rel_improvement: float) -> bool:
    if len(history) < window + 1:
        return False
    recent = history[-(window + 1):]
    best_old = min(recent[:-1])
    latest = recent[-1]
    if best_old <= 1e-9:
        return True
    return (best_old - latest) / best_old < rel_improvement


def check_reachability(
    model: ExcavatorModel,
    ik_cfg: IKConfig,
    current_joint_angles: np.ndarray,
    target_pos: np.ndarray,
    target_rot_y_deg: float = 0.0,
    *,
    pos_tol: float = 0.005,
    max_iters: int = 80,
    cond_threshold: float = 0.0,
    dt: float = 0.01,
    stall_window: int = 10,
    stall_rel_improvement: float = 0.01,
    position_only: bool = False,
) -> ReachabilityResult:
    """Simulate an IK rollout from canonical joint angles toward ``target_pos``.

    Args:
        model: ExcavatorModel describing the chain.
        ik_cfg: IKConfig used for the rollout. Pass the same config the live
            controller uses so reachability mirrors runtime behavior.
        current_joint_angles: Canonical joint angles (n_joints,) in radians.
        target_pos: Desired EE position [x, y, z] in world frame.
        target_rot_y_deg: Desired pitch about Y in degrees.
        pos_tol: Convergence tolerance in meters.
        max_iters: Hard iteration cap.
        cond_threshold: Reject as unreachable if Jacobian condition exceeds
            this value at the converged pose. ``0`` disables the check.
        dt: Integration timestep handed to ``solve_ik_step``.
        stall_window: Iterations to look back when measuring improvement.
        stall_rel_improvement: Minimum fractional reduction in pos_error
            over ``stall_window`` iterations; below this we early-exit.

    Returns:
        ReachabilityResult.
    """
    target_pos = np.asarray(target_pos, dtype=np.float32)
    angles = np.asarray(current_joint_angles, dtype=np.float32).copy()

    pitch_quat = quat_from_axis_angle(
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.float32(np.radians(target_rot_y_deg)),
    )

    err_history: list = []
    state = get_state(angles, model)
    ee_pos = np.asarray(state.ee_position, dtype=np.float32)
    final_cond = float(state.condition_number)

    iters_used = 0
    for i in range(max_iters):
        iters_used = i + 1

        # Compose target orientation each step: slew(q[0]) ∘ pitch — matches
        # the live controller's "pitch follows slew" convention so yaw error
        # stays zero by construction. When ``position_only`` is True we skip
        # the orientation constraint entirely (matches tests / callers that
        # only care about Cartesian reach).
        if position_only:
            target_quat = None
        else:
            slew_quat = quat_from_axis_angle(
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
                np.float32(angles[0]),
            )
            target_quat = quat_normalize(quat_multiply(slew_quat, pitch_quat))

        err = float(np.linalg.norm(target_pos - ee_pos))
        err_history.append(err)

        if err <= pos_tol:
            break

        if _stall_detected(err_history, stall_window, stall_rel_improvement):
            break

        result = solve_ik_step(
            angles, target_pos, target_quat, model, ik_cfg, dt=dt, state=state,
        )
        angles = np.asarray(result.q_next_rad, dtype=np.float32)
        state = get_state(angles, model)
        ee_pos = np.asarray(state.ee_position, dtype=np.float32)
        final_cond = float(state.condition_number)

    pos_error = float(np.linalg.norm(target_pos - ee_pos))
    cond_ok = (cond_threshold <= 0) or (final_cond <= cond_threshold)
    reachable = (pos_error <= pos_tol) and cond_ok

    return ReachabilityResult(
        reachable=reachable,
        closest_position=ee_pos.copy(),
        pos_error_m=pos_error,
        iters=iters_used,
        final_cond_number=final_cond,
    )
