"""Relative-IK command integrator with smooth reachability limiting.

This is the "command layer" between operator velocity input and the
absolute pose that the IK controller consumes.

Pipeline per tick:
    operator v_cmd
      -> reachability smoothstep scale (joint-limit / yoshikawa / cond)
      -> velocity cap
      -> integrate into absolute target pose
      -> hand to controller.give_pose(target)

The processor owns:
  - the integrated absolute target (the pose handed to give_pose)
  - re-sync of that target to the measured pose on (re)start / pause-resume
  - a debug dict suitable for printing / telemetry

No hardware or controller imports. Tested in isolation in
test_relative_command_processor.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from reachability_limiter import (
    JointLimit,
    LimiterConfig,
    apply_scale,
    compute_scale,
)


@dataclass
class TargetPose:
    """Absolute end-effector target in robot base frame."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rot_y_deg: float = 0.0

    def as_xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def copy(self) -> "TargetPose":
        return TargetPose(self.x, self.y, self.z, self.rot_y_deg)


@dataclass(frozen=True)
class ProcessorConfig:
    """Tunables for the relative-command integrator."""

    limiter: LimiterConfig = field(default_factory=LimiterConfig)
    # Hard ceiling on dt so a long pause / dropped packet does not translate
    # into a huge integration step on resume.
    max_dt_s: float = 0.1
    # Yaw stays inside [-yaw_wrap_deg, +yaw_wrap_deg) via modular wrap.
    yaw_wrap_deg: float = 180.0


class RelativeCommandProcessor:
    """Integrate a velocity command into an absolute target pose with smooth limiting.

    Usage::

        proc = RelativeCommandProcessor()
        proc.sync_to_measured(pos_xyz, rot_y_deg)          # first packet / on resume
        ...
        target, debug = proc.step(
            v_cmd_xyz, v_yaw_dps, dt,
            joint_angles_rad=..., joint_limits_rad=...,
            yoshikawa=..., cond=...,
        )
        controller.give_pose(target.as_xyz(), target.rot_y_deg)

    Thread-safety: not threadsafe. Drive it from one place (e.g. the UDP
    receive loop on the server).
    """

    def __init__(self, cfg: Optional[ProcessorConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else ProcessorConfig()
        self._target: Optional[TargetPose] = None

    # ---------------------------------------------------------------- lifecycle

    @property
    def target(self) -> Optional[TargetPose]:
        """Current integrated target, or None until first ``sync_to_measured``."""
        return None if self._target is None else self._target.copy()

    def sync_to_measured(self, pos_xyz: Sequence[float], rot_y_deg: float) -> None:
        """Snap the integrated target to a measured pose.

        Call on first packet, after pause/resume, and on any mode switch
        that would otherwise leave the target sitting in stale territory.
        """
        p = np.asarray(pos_xyz, dtype=np.float32)
        self._target = TargetPose(float(p[0]), float(p[1]), float(p[2]), float(rot_y_deg))

    def reset(self) -> None:
        """Drop the integrated target. Next ``step()`` will raise until re-synced."""
        self._target = None

    # ----------------------------------------------------------------- main step

    def step(
        self,
        v_cmd_xyz: Sequence[float],
        v_yaw_dps: float,
        dt: float,
        *,
        joint_angles_rad: Sequence[float],
        joint_limits_rad: Sequence[JointLimit],
        yoshikawa: float,
        cond: float,
    ) -> Tuple[TargetPose, dict]:
        """Advance the integrated target by one tick.

        Returns a copy of the new target and a debug dict carrying the
        limiter output plus the actually-applied velocity (for telemetry).
        """
        if self._target is None:
            raise RuntimeError(
                "RelativeCommandProcessor.step() called before sync_to_measured(); "
                "the integrator has no anchor pose yet."
            )

        dt = float(np.clip(float(dt), 0.0, self.cfg.max_dt_s))

        limiter_out = compute_scale(
            joint_angles_rad=joint_angles_rad,
            joint_limits_rad=joint_limits_rad,
            yoshikawa=yoshikawa,
            cond=cond,
            cfg=self.cfg.limiter,
        )
        v_xyz, v_yaw = apply_scale(v_cmd_xyz, v_yaw_dps, limiter_out["scale"], self.cfg.limiter)

        self._target.x += float(v_xyz[0]) * dt
        self._target.y += float(v_xyz[1]) * dt
        self._target.z += float(v_xyz[2]) * dt
        self._target.rot_y_deg = _wrap_deg(
            self._target.rot_y_deg + float(v_yaw) * dt,
            self.cfg.yaw_wrap_deg,
        )

        debug = dict(limiter_out)
        debug["applied_v_xyz"] = (float(v_xyz[0]), float(v_xyz[1]), float(v_xyz[2]))
        debug["applied_v_yaw_dps"] = float(v_yaw)
        debug["dt"] = dt
        return self._target.copy(), debug


def _wrap_deg(deg: float, half_range: float) -> float:
    """Wrap an angle to [-half_range, +half_range)."""
    full = 2.0 * float(half_range)
    if full <= 0.0:
        return float(deg)
    return ((float(deg) + half_range) % full) - half_range


# ============================================================================
# Proposed future work -- keep this list in sync with smooth_reachability_limiter_readme.md
# ============================================================================
#
# TODO(directional-scaling): Replace the single scalar `scale` with per-axis
#   scales by probing reachability with small candidate steps. For each
#   command axis: predict the joint-angle change from a unit EE step via the
#   Jacobian, score the resulting margin, damp ONLY the axes that worsen it.
#   This preserves tangent motion along a margin level set instead of the
#   current global slowdown. Matches readme Phase 3.
#
# TODO(predictive-safety): Evaluate `margin(t + lookahead_s)` from measured
#   joint velocities (or a short rollout of the current integrated target),
#   not just `margin(now)`. Pre-empts PID overshoot and inertial drift that
#   the static limiter does not see. Readme Phase 4.
#
# TODO(input-shaping): Add a jerk/accel-limited shaper on v_cmd before it
#   enters step(), so a hard stick deflection from zero does not become a
#   step in commanded velocity. Today the smoothstep handles boundaries but
#   assumes operator input is already reasonably smooth.
#
# TODO(per-axis-velocity-caps): cfg.limiter.max_ee_speed_mps caps the
#   magnitude. Diagonal stick deflection gives each axis sqrt(2)/2 of the cap.
#   Decide whether per-axis caps make gamepad teleop feel more predictable.
#
# TODO(table-floor): If the rig sits over a table, add a one-sided smooth
#   z-floor here (NOT a full workspace box) so external obstacles outside the
#   kinematic envelope are still respected. Reuse the same smoothstep with
#   outward normal n=[0, 0, -1]. Joint limits cannot see external geometry.
#
# TODO(config-loader): Load ProcessorConfig + LimiterConfig from YAML next to
#   this file so tunables do not require an edit-restart cycle. Keep it
#   self-contained in control_prototype/ -- do NOT pull from configuration_files/.
#
# TODO(telemetry-channel): Expose the debug dict over the existing telemetry
#   protocol (extra floats) so the client side can render the active scale
#   and dominant constraint live during teleop.
#
# TODO(measured-vs-target-anchor): step() integrates a virtual target. If the
#   controller falls behind (slow hydraulics, stalled joint), the integrated
#   target can run ahead of the measured pose. Consider clamping the
#   integration so |target - measured| does not exceed a "lead window" --
#   prevents wind-up similar to what relative-mode was supposed to solve at
#   the IK level but applied at this layer too.
#   See excv_gui_relative.py for the current lead-window + zero-snap approach.
#
# TODO(rubber-band-model): Consider replacing the integrator entirely with a
#   pure rubber-band: target = meas_pos + normalize(v_cmd) * lookahead_m.
#   The target is re-derived from the measured EE every tick, so windup is
#   impossible by construction and zero velocity naturally holds the arm in
#   place — no snap, no clamp, no accumulated state.
#
#   Slew-rate limiting belongs on the SERVER, not the client. The client
#   sends raw fast-switching commands; the server ramps them to suit the
#   robot's dynamics. This way any client (gamepad, ROS node, web UI) gets
#   safe smooth behaviour for free without knowing anything about the machine.
#   A simple per-axis rate limiter (max delta per tick) applied to v_cmd
#   before the rubber-band offset is computed is all that is needed:
#
#       v_cmd_slewed = slew(v_cmd_raw, v_prev, max_delta_per_tick)
#       target = meas_pos + normalize(v_cmd_slewed) * lookahead_m
#
#   Smooth start/stop then falls out naturally from the ramp.
#   The reachability smoothstep scaling of v_cmd is unchanged.
#   Only open question: does a fixed lookahead_m (e.g. 30-50 mm) give the
#   PID enough authority to drive the hydraulics across the full speed range?
