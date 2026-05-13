"""Smooth reachability limiter for relative IK teleoperation.

Scales an EE velocity command by a unitless reachability margin in [0,1].
The margin is the min of three signals derived from live IK telemetry:

  - joint-limit distance (per joint, slowest one wins)
  - Yoshikawa manipulability vs a calibrated nominal
  - Jacobian condition number vs a warn threshold

The combined margin is shaped by a smoothstep so motion eases into the
boundary instead of hitting a binary reject. There is intentionally no
Cartesian workspace box here -- joint-limit distance is the workspace.

Pure numpy. No controller / hardware imports. This file is the lift target
for the next machine: drop it in next to the same IK stack and feed it the
same four signals (joint_angles_rad, joint_limits_rad, yoshikawa, cond).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


JointLimit = Optional[Tuple[float, float]]


@dataclass(frozen=True)
class LimiterConfig:
    """Tunables for the smooth reachability limiter."""

    # Joint-limit margin: smoothstep ramp width inside each joint stop (deg).
    joint_slow_zone_deg: float = 8.0

    # Yoshikawa manipulability margin: linear ramp from floor -> nominal.
    yoshikawa_nominal: float = 0.10
    yoshikawa_floor: float = 0.0

    # Condition-number margin: linear ramp from cond_ok -> cond_warn.
    cond_ok: float = 20.0
    cond_warn: float = 40.0

    # Velocity caps applied AFTER the smoothstep scale (operator-feel ceiling).
    max_ee_speed_mps: float = 0.15
    max_yaw_rate_dps: float = 30.0


# --------------------------------------------------------------------------
# Pure scalar helpers
# --------------------------------------------------------------------------

def smoothstep(s: float) -> float:
    """Hermite smoothstep s*s*(3 - 2*s), clamped to [0, 1]."""
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def joint_limit_margin(
    angles_rad: Sequence[float],
    joint_limits_rad: Sequence[JointLimit],
    slow_zone_rad: float,
) -> Tuple[float, int]:
    """Return (margin in [0, 1], index of the worst joint).

    Per-joint margin = min(angle - lo, hi - angle) / slow_zone.
    Overall margin = min across joints. Joints with None / non-finite bounds
    are treated as unbounded and skipped.
    """
    angles = np.asarray(angles_rad, dtype=np.float64)
    worst = 1.0
    worst_idx = -1
    for i, lim in enumerate(joint_limits_rad):
        if lim is None:
            continue
        lo, hi = float(lim[0]), float(lim[1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if hi <= lo:
            continue
        d = min(float(angles[i]) - lo, hi - float(angles[i]))
        if slow_zone_rad > 0:
            m = d / slow_zone_rad
        else:
            m = 1.0 if d > 0 else 0.0
        if m < worst:
            worst = m
            worst_idx = i
    return float(np.clip(worst, 0.0, 1.0)), worst_idx


def yoshikawa_margin(yoshikawa: float, nominal: float, floor: float = 0.0) -> float:
    """Linear ramp from `floor` -> `nominal`, clamped to [0, 1]."""
    span = float(nominal) - float(floor)
    if span <= 0.0:
        return 1.0
    s = (float(yoshikawa) - float(floor)) / span
    return float(np.clip(s, 0.0, 1.0))


def cond_margin(cond: float, cond_ok: float, cond_warn: float) -> float:
    """Linear ramp from `cond_ok` (margin 1) -> `cond_warn` (margin 0)."""
    span = float(cond_warn) - float(cond_ok)
    if span <= 0.0:
        return 1.0 if float(cond) <= float(cond_ok) else 0.0
    s = (float(cond_warn) - float(cond)) / span
    return float(np.clip(s, 0.0, 1.0))


def combine_margins(*margins: float) -> Tuple[float, int]:
    """Return (min margin, index of the dominant constraint)."""
    if not margins:
        return 1.0, -1
    arr = np.asarray([float(m) for m in margins], dtype=np.float64)
    idx = int(np.argmin(arr))
    return float(arr[idx]), idx


# --------------------------------------------------------------------------
# One-shot convenience used by the prototype server
# --------------------------------------------------------------------------

_DOMINANT_NAMES = ("joint", "yoshikawa", "cond")


def compute_scale(
    *,
    joint_angles_rad: Sequence[float],
    joint_limits_rad: Sequence[JointLimit],
    yoshikawa: float,
    cond: float,
    cfg: LimiterConfig,
) -> dict:
    """Compute the smoothstep scale and a debug dict from the four live signals.

    Returns:
        {
            "scale": float in [0,1],   # smoothstep of raw_margin
            "raw_margin": float,       # min of the three component margins
            "dominant": str,           # "joint" | "yoshikawa" | "cond"
            "worst_joint_idx": int,    # -1 if joint margin is not dominant
            "margins": {"joint": .., "yoshikawa": .., "cond": ..},
        }
    """
    slow_zone_rad = float(np.radians(cfg.joint_slow_zone_deg))
    # Fail-safe to zero on any non-finite live signal (sensor dropout, IK init
    # corner cases). Better to freeze the EE than to multiply v_cmd by NaN.
    if not _all_finite(joint_angles_rad) or not np.isfinite(yoshikawa) or not np.isfinite(cond):
        return {
            "scale": 0.0,
            "raw_margin": 0.0,
            "dominant": "nonfinite",
            "worst_joint_idx": -1,
            "margins": {"joint": 0.0, "yoshikawa": 0.0, "cond": 0.0},
        }
    m_joint, worst_joint = joint_limit_margin(joint_angles_rad, joint_limits_rad, slow_zone_rad)
    m_yosh = yoshikawa_margin(yoshikawa, cfg.yoshikawa_nominal, cfg.yoshikawa_floor)
    m_cond = cond_margin(cond, cfg.cond_ok, cfg.cond_warn)
    raw_margin, dominant_idx = combine_margins(m_joint, m_yosh, m_cond)
    return {
        "scale": smoothstep(raw_margin),
        "raw_margin": raw_margin,
        "dominant": _DOMINANT_NAMES[dominant_idx],
        "worst_joint_idx": worst_joint if dominant_idx == 0 else -1,
        "margins": {"joint": m_joint, "yoshikawa": m_yosh, "cond": m_cond},
    }


def _all_finite(values) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isfinite(arr)))


def apply_scale(
    v_xyz: Sequence[float],
    v_yaw_dps: float,
    scale: float,
    cfg: LimiterConfig,
) -> Tuple[np.ndarray, float]:
    """Apply the smoothstep scale and operator-feel velocity caps.

    Caps are applied AFTER scaling, so the smooth boundary always wins over
    the ceiling.
    """
    v = np.asarray(v_xyz, dtype=np.float32) * float(scale)
    speed = float(np.linalg.norm(v))
    if speed > cfg.max_ee_speed_mps and speed > 0.0:
        v = v * (cfg.max_ee_speed_mps / speed)
    yaw = float(v_yaw_dps) * float(scale)
    yaw = float(np.clip(yaw, -cfg.max_yaw_rate_dps, cfg.max_yaw_rate_dps))
    return v, yaw
