#!/usr/bin/env python3
"""End-effector PID tuner — robot/host side.

This tool runs the full production control stack (``ExcavatorController`` +
``HardwareInterface``) and drives the EE along a straight line in the world X
axis, back and forth, at a configurable height (Z) and EE-rate
(``20..70 mm/s``). EE rotation is held at zero.

What it is good for
-------------------
* **Manual tuning:** pick gains in the client GUI, watch the EE follow the
  ideal X ramp in real time, see RMSE / max / mean tracking error per stroke.
* **Auto tuning:** let the host search PID gains across multiple runs while
  the client tracks progress. The default search is a coordinate-descent
  ("twiddle") loop with adaptive step size — it works well for asymmetric
  hydraulic actuators because we can score whole runs (not single steps) and
  it does not need a derivative of the cost surface.

The host does **not** modify the control stack on disk. It only mutates the
existing ``ExcavatorController.joint_pids`` list at runtime — installing a
:class:`DirectionalPID` wrapper on the tuned joint when direction-specific
tuning is requested. Restarting the host restores the controller's defaults
(from ``control_config.yaml``) on the next process.

Protocol
--------
See :mod:`tools.pid_tuner_ee_common` for packet layout. Default UDP port is
``8091`` (one above the per-joint tuner on ``8090``).

Safety
------
On this test bench the actuators are not connected, so the script is safe to
run blind. On real hardware the same rules as production apply: the controller
zeroes outputs when paused, the pump is gated by ``hardware.set_pump_enabled``,
and unreachable targets are rejected by the controller's reachability check.

CLI
---
Run with no args for the rpi profile; ``-h`` lists every option.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.board import resolve_profile as resolve_board_profile
from modules.excavator_controller import ExcavatorController
from modules.hardware_interface import HardwareInterface
from modules.pid import PIDController
from modules.udp_socket import UDPSocket

from tools.pid_tuner_ee_common import (
    CMD_COST_W_MAX,
    CMD_COST_W_RMSE,
    CMD_FLAGS,
    CMD_JOINT_MASK,
    CMD_KD_FWD,
    CMD_KD_REV,
    CMD_KI_FWD,
    CMD_KI_REV,
    CMD_KP_FWD,
    CMD_KP_REV,
    CMD_NUM_RUNS,
    CMD_SPEED,
    CMD_SPEED_MAX,
    CMD_SPEED_MIN,
    CMD_STROKES_PER_RUN,
    CMD_TUNED_JOINT,
    CMD_X_MAX,
    CMD_X_MIN,
    CMD_Y,
    CMD_Z,
    CMD_Z_MAX,
    CMD_Z_MIN,
    COMMAND_SIZE,
    CommandFlag,
    DEFAULT_COST_WEIGHTS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    TELEMETRY_SIZE,
    TLM_BEST_COST,
    TLM_BEST_COST_FWD,
    TLM_BEST_COST_REV,
    TLM_CMD_X,
    TLM_CMD_Y,
    TLM_CMD_Z,
    TLM_CUR_SPEED,
    TLM_CUR_X_MAX,
    TLM_CUR_X_MIN,
    TLM_CUR_Z,
    TLM_DIR_PID,
    TLM_EE_X,
    TLM_EE_Y,
    TLM_EE_Z,
    TLM_ERR_NOW,
    TLM_HW_READY,
    TLM_ITER,
    TLM_KD_FWD,
    TLM_KD_REV,
    TLM_KI_FWD,
    TLM_KI_REV,
    TLM_KP_FWD,
    TLM_KP_REV,
    TLM_LAST_COST,
    TLM_LAST_COST_FWD,
    TLM_LAST_COST_REV,
    TLM_MAXERR_LAST,
    TLM_MAXERR_LAST_FWD,
    TLM_MAXERR_LAST_REV,
    TLM_MEANERR_LAST,
    TLM_MEANERR_LAST_FWD,
    TLM_MEANERR_LAST_REV,
    TLM_NUM_RUNS,
    TLM_RAMP_DIR,
    TLM_RMSE_LAST,
    TLM_RMSE_LAST_FWD,
    TLM_RMSE_LAST_REV,
    TLM_RUN_INDEX,
    TLM_STATE,
    TLM_STROKE_INDEX,
    TLM_TOTAL_STROKES,
    TLM_TUNED_JOINT,
    TunerState,
)


JOINT_NAMES = ["slew", "boom", "arm", "bucket"]


# ---------------------------------------------------------------------------
# Direction-specific PID
# ---------------------------------------------------------------------------


class DirectionalPID:
    """Two :class:`PIDController` instances chosen by stroke direction.

    The production controller calls every joint PID as
    ``pid.compute(0.0, -angle_error(target, current))``. By default we route
    positive ``error`` to ``pid_pos`` and negative ``error`` to ``pid_neg``
    (sign-of-error switching) so the wrapper degrades gracefully when no
    external direction is known.

    For tuning runs we override that with an *explicit* stroke direction:
    the host calls :meth:`set_active_direction` at the start of every stroke
    so the +X (out) strokes always use ``pid_pos`` and the -X (in) strokes
    always use ``pid_neg``. This gives the auto-tuner a clean attribution
    between gain dimensions and cost subscores — the fwd cost is shaped only
    by ``pid_pos``, the rev cost only by ``pid_neg``.

    On every tick we feed the *current measurement* into the inactive PID so
    its derivative-on-measurement filter and ``last_value`` do not jump when
    control switches sides.
    """

    def __init__(
        self,
        gains_pos: Tuple[float, float, float],
        gains_neg: Tuple[float, float, float],
        min_output: float = -1.0,
        max_output: float = 1.0,
        deriv_filter_tau: float = 0.10,
    ) -> None:
        self.pid_pos = PIDController(
            kp=gains_pos[0], ki=gains_pos[1], kd=gains_pos[2],
            min_output=min_output, max_output=max_output,
            deriv_filter_tau=deriv_filter_tau,
        )
        self.pid_neg = PIDController(
            kp=gains_neg[0], ki=gains_neg[1], kd=gains_neg[2],
            min_output=min_output, max_output=max_output,
            deriv_filter_tau=deriv_filter_tau,
        )
        # Mirror the public attributes the production controller reads so the
        # rest of the stack can treat this as a drop-in PID.
        self.min_output = min_output
        self.max_output = max_output
        # +1 forces pid_pos, -1 forces pid_neg, 0 falls back to sign-of-error.
        self._active_dir: int = 0

    def set_active_direction(self, direction: int) -> None:
        """Pin which PID handles this tick. Use +1/-1 during a stroke, 0 idle."""
        self._active_dir = +1 if direction > 0 else (-1 if direction < 0 else 0)

    @property
    def kp(self) -> float:
        return self.pid_pos.kp

    @property
    def ki(self) -> float:
        return self.pid_pos.ki

    @property
    def kd(self) -> float:
        return self.pid_pos.kd

    def gains_pos(self) -> Tuple[float, float, float]:
        return (self.pid_pos.kp, self.pid_pos.ki, self.pid_pos.kd)

    def gains_neg(self) -> Tuple[float, float, float]:
        return (self.pid_neg.kp, self.pid_neg.ki, self.pid_neg.kd)

    def set_gains_pos(self, kp: float, ki: float, kd: float) -> None:
        self.pid_pos.kp, self.pid_pos.ki, self.pid_pos.kd = kp, ki, kd
        if ki != 0.0:
            self.pid_pos.set_integral_limits(self.min_output / ki, self.max_output / ki)

    def set_gains_neg(self, kp: float, ki: float, kd: float) -> None:
        self.pid_neg.kp, self.pid_neg.ki, self.pid_neg.kd = kp, ki, kd
        if ki != 0.0:
            self.pid_neg.set_integral_limits(self.min_output / ki, self.max_output / ki)

    def reset(self, keep_integral: bool = False) -> None:
        self.pid_pos.reset(keep_integral=keep_integral)
        self.pid_neg.reset(keep_integral=keep_integral)

    def compute(self, setpoint: float, current_value: float, dt: Optional[float] = None) -> float:
        if self._active_dir > 0:
            active, inactive = self.pid_pos, self.pid_neg
        elif self._active_dir < 0:
            active, inactive = self.pid_neg, self.pid_pos
        else:
            error = setpoint - current_value
            if error >= 0.0:
                active, inactive = self.pid_pos, self.pid_neg
            else:
                active, inactive = self.pid_neg, self.pid_pos
        # Keep inactive's measurement memory hot to avoid derivative kicks on
        # zero-crossing or direction-pin transitions.
        inactive.last_value = current_value
        return active.compute(setpoint, current_value, dt=dt)


# ---------------------------------------------------------------------------
# Auto-tuner — coordinate descent with adaptive step size
# ---------------------------------------------------------------------------


@dataclass
class GainsPair:
    """Six PID gains: (kp, ki, kd) for forward and reverse directions."""

    fwd: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rev: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def as_vector(self) -> List[float]:
        return [*self.fwd, *self.rev]

    @staticmethod
    def from_vector(v: List[float]) -> "GainsPair":
        return GainsPair(fwd=list(v[:3]), rev=list(v[3:6]))

    def copy(self) -> "GainsPair":
        return GainsPair(fwd=list(self.fwd), rev=list(self.rev))


_FWD_DIMS = (0, 1, 2)   # kp_fwd, ki_fwd, kd_fwd
_REV_DIMS = (3, 4, 5)   # kp_rev, ki_rev, kd_rev


class CoordinateDescentTuner:
    """Twiddle-style coordinate descent over PID gains.

    Algorithm sketch (per iteration over each of the 6 dimensions)::

        try   x[i] += step[i];          run; cost
        if better: keep,         step[i] *= 1.1
        else:
            try   x[i] -= 2*step[i];   run; cost
            if better: keep,     step[i] *= 1.1
            else:     revert,    step[i] *= 0.5

    Why this algorithm
    ------------------
    * Each "evaluation" is a full back-and-forth run on the real plant — slow
      and noisy. Coordinate descent only needs one evaluation per direction
      probed, with no gradient estimation, and tolerates noise well because
      each step is gated on improvement vs. the current best.
    * The step-size annealing converges around a local minimum without ever
      requiring an explicit stopping derivative.
    * We can freeze ``ki`` / ``kd`` at zero (see ``frozen_mask``) and tune
      only ``kp`` first, which is the classical hand-tune pattern.

    Limits
    ------
    Local minima are real — start the search near a reasonable hand-tuned
    point (the controller's current gains are loaded by default). The cost
    must be a *scalar* — see ``score_run`` below for the default weighting.
    """

    def __init__(
        self,
        initial: GainsPair,
        steps: GainsPair,
        frozen_mask: Optional[List[bool]] = None,
        grow_factor: float = 1.1,
        shrink_factor: float = 0.5,
        min_step: float = 1e-3,
        gain_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (0.0, 50.0),  # kp
            (0.0, 20.0),  # ki
            (0.0, 20.0),  # kd
        ),
    ) -> None:
        self.current = initial.copy()
        self.best = initial.copy()
        # Best-cost is tracked **per direction** so each gain triple is judged
        # only against the strokes it actually shapes.
        self.best_cost_fwd: float = math.inf
        self.best_cost_rev: float = math.inf
        self.step = steps.as_vector()
        # 6 dims: [kp_f, ki_f, kd_f, kp_r, ki_r, kd_r]
        self.frozen = list(frozen_mask) if frozen_mask is not None else [False] * 6
        self.grow = float(grow_factor)
        self.shrink = float(shrink_factor)
        self.min_step = float(min_step)
        self.bounds = gain_bounds
        # State machine for which dimension / which probe direction is active.
        self._dim_index = 0
        # Probe phase: 0 = need to probe +, 1 = need to probe -, 2 = dim done.
        self._phase = 0
        self._pending_revert: Optional[float] = None
        self._iter = 0

    @property
    def best_cost(self) -> float:
        """Combined-for-display best cost (sum of fwd + rev best). Backwards-compat."""
        f = self.best_cost_fwd if math.isfinite(self.best_cost_fwd) else 0.0
        r = self.best_cost_rev if math.isfinite(self.best_cost_rev) else 0.0
        return f + r

    def active_side(self) -> str:
        """Which gain triple the *next* propose() will modify: 'fwd' or 'rev'."""
        return "fwd" if self._dim_index in _FWD_DIMS else "rev"

    def converged(self) -> bool:
        return all(s <= self.min_step for s, f in zip(self.step, self.frozen) if not f)

    def iter_count(self) -> int:
        return self._iter

    def _clamp(self, dim: int, value: float) -> float:
        lo, hi = self.bounds[dim % 3]
        return max(lo, min(hi, float(value)))

    def _vector(self) -> List[float]:
        return self.current.as_vector()

    def _write_vector(self, v: List[float]) -> None:
        self.current = GainsPair.from_vector(v)

    def _advance_dim(self) -> None:
        # Move to next non-frozen dimension. Wrap around indefinitely.
        for _ in range(6):
            self._dim_index = (self._dim_index + 1) % 6
            if not self.frozen[self._dim_index]:
                break
        self._phase = 0
        self._pending_revert = None

    def propose(self) -> GainsPair:
        """Return the candidate gain set the host should test next."""
        # Skip frozen dims if we happen to land on one.
        for _ in range(6):
            if not self.frozen[self._dim_index]:
                break
            self._advance_dim()
        v = self._vector()
        d = self._dim_index
        if self._phase == 0:
            self._pending_revert = v[d]
            v[d] = self._clamp(d, v[d] + self.step[d])
        elif self._phase == 1:
            assert self._pending_revert is not None
            v[d] = self._clamp(d, self._pending_revert - self.step[d])
        # phase 2 means dim is finished and we should not be proposing on it;
        # caller is expected to call observe() before propose() again.
        return GainsPair.from_vector(v)

    def observe(self, cost_fwd: float, cost_rev: float) -> None:
        """Update internal state with the cost of the most recently proposed gains.

        ``cost_fwd`` and ``cost_rev`` are the per-direction subscores from the
        last completed run. Whichever side the active dimension belongs to is
        the one compared against ``best_cost_<side>`` — the *other* direction's
        cost is ignored for the gating decision because changing a fwd gain
        cannot have systematically improved the reverse stroke (and vice
        versa). We still record the unrelated side's best whenever it
        coincidentally improves, so the GUI tracks both.
        """
        self._iter += 1
        d = self._dim_index
        side = "fwd" if d in _FWD_DIMS else "rev"
        my_cost = cost_fwd if side == "fwd" else cost_rev
        my_best = self.best_cost_fwd if side == "fwd" else self.best_cost_rev

        # Opportunistic best tracking on the *other* side — costs there can
        # also drift around if both directions share dynamics, but we never
        # gate the search on this.
        if cost_fwd < self.best_cost_fwd:
            self.best_cost_fwd = cost_fwd
        if cost_rev < self.best_cost_rev:
            self.best_cost_rev = cost_rev

        if self._phase == 0:
            candidate = self._vector()
            candidate[d] = self._clamp(d, candidate[d] + self.step[d])
            if my_cost < my_best:
                self._write_vector(candidate)
                self.best = self.current.copy()
                self.step[d] *= self.grow
                self._advance_dim()
            else:
                self._phase = 1
        elif self._phase == 1:
            candidate = self._vector()
            candidate[d] = self._clamp(d, candidate[d] - self.step[d])
            if my_cost < my_best:
                self._write_vector(candidate)
                self.best = self.current.copy()
                self.step[d] *= self.grow
            else:
                # Neither direction improved this side — shrink and move on.
                self.step[d] *= self.shrink
            self._advance_dim()
        # else: misuse of the API; ignore silently.


# ---------------------------------------------------------------------------
# Run executor — drives a single back-and-forth program and scores it
# ---------------------------------------------------------------------------


@dataclass
class DirectionStats:
    """Per-direction subscores from a single back-and-forth run."""
    rmse: float = 0.0
    max_abs_err: float = 0.0
    mean_abs_err: float = 0.0
    cost: float = math.inf
    samples: int = 0


@dataclass
class RunResult:
    """Combined stats plus per-direction breakdown.

    ``cost`` is the combined cost for back-compat / display. ``fwd`` and
    ``rev`` carry the cost the auto-tuner actually uses when probing
    the forward (``kp_fwd / ki_fwd / kd_fwd``) or reverse gain dimensions.
    """
    rmse: float
    max_abs_err: float
    mean_abs_err: float
    cost: float
    samples: int
    fwd: DirectionStats = field(default_factory=DirectionStats)
    rev: DirectionStats = field(default_factory=DirectionStats)


@dataclass
class ProgramConfig:
    x_min: float
    x_max: float
    y: float
    z: float
    speed: float
    strokes_per_run: int
    cost_w_rmse: float = DEFAULT_COST_WEIGHTS[0]
    cost_w_max: float = DEFAULT_COST_WEIGHTS[1]
    cost_w_overshoot: float = DEFAULT_COST_WEIGHTS[2]
    # Auto-tune sweep brackets. Manual runs use z / speed above; the auto-tune
    # loop rotates through a small grid built from these (min, max) brackets.
    z_min: float = 0.18
    z_max: float = 0.22
    speed_min: float = 0.020
    speed_max: float = 0.070


def _autotune_grid(cfg: "ProgramConfig", n_z: int = 3, n_v: int = 3) -> List[Tuple[float, float]]:
    """Return a deterministic (z, speed) grid for auto-tune iteration.

    Three z values × three speeds = nine conditions cycled round-robin so the
    coordinate-descent tuner sees a fixed cost surface from run to run.
    """
    zs = np.linspace(cfg.z_min, cfg.z_max, n_z).tolist() if cfg.z_max > cfg.z_min else [cfg.z]
    vs = np.linspace(cfg.speed_min, cfg.speed_max, n_v).tolist() if cfg.speed_max > cfg.speed_min else [cfg.speed]
    return [(float(z), float(v)) for z in zs for v in vs]


def _score_subset(err_mm: np.ndarray, cfg: ProgramConfig) -> DirectionStats:
    """Cost for a single direction's slice of error samples (mm)."""
    if err_mm.size == 0:
        return DirectionStats()
    rmse = float(np.sqrt(np.mean(err_mm ** 2)))
    max_abs = float(np.max(np.abs(err_mm)))
    mean_abs = float(np.mean(np.abs(err_mm)))
    tail = err_mm[int(0.9 * err_mm.size):]
    overshoot = float(np.mean(np.clip(np.abs(tail) - rmse, 0.0, None))) if tail.size else 0.0
    cost = (
        cfg.cost_w_rmse * rmse
        + cfg.cost_w_max * max_abs
        + cfg.cost_w_overshoot * overshoot
    )
    return DirectionStats(
        rmse=rmse, max_abs_err=max_abs, mean_abs_err=mean_abs,
        cost=cost, samples=int(err_mm.size),
    )


def score_run(
    cmd_x_samples: List[float],
    ee_x_samples: List[float],
    stroke_dirs: List[int],
    cfg: ProgramConfig,
) -> RunResult:
    """Score a completed back-and-forth run with a per-direction breakdown.

    The cost is in **millimetre-error units** so that auto-tune step sizes
    work on intuitive magnitudes. For each sample we know which stroke it
    belonged to (``stroke_dirs[i] == +1`` for an out / +X stroke, ``-1`` for
    an in / -X stroke), so we compute three statistics::

        err_mm = (ee_x - cmd_x) * 1000
        cost(s) = w_rmse*RMSE(s) + w_max*max|err|(s) + w_overshoot*tail(s)

    against three sample sets: all samples (combined), only +X samples
    (``result.fwd``) and only -X samples (``result.rev``). The auto-tuner
    feeds the fwd / rev subscore back when it is probing the matching gain
    triple — that way each direction's three gains see the noise of their
    own strokes only.

    Speed enters via ``ee_x = cmd_x`` only — runs at different ``speed_mps``
    naturally accumulate larger errors so the auto-tuner is automatically
    rewarded for converging at all configured speeds.
    """
    if not cmd_x_samples or not ee_x_samples:
        return RunResult(0.0, 0.0, 0.0, math.inf, 0)
    n = min(len(cmd_x_samples), len(ee_x_samples), len(stroke_dirs))
    err_mm = np.array(
        [(ee_x_samples[i] - cmd_x_samples[i]) * 1000.0 for i in range(n)],
        dtype=np.float64,
    )
    dirs = np.array(stroke_dirs[:n], dtype=np.int8)
    fwd = _score_subset(err_mm[dirs > 0], cfg)
    rev = _score_subset(err_mm[dirs < 0], cfg)
    combined = _score_subset(err_mm, cfg)
    return RunResult(
        rmse=combined.rmse,
        max_abs_err=combined.max_abs_err,
        mean_abs_err=combined.mean_abs_err,
        cost=combined.cost,
        samples=combined.samples,
        fwd=fwd,
        rev=rev,
    )


# ---------------------------------------------------------------------------
# Host service
# ---------------------------------------------------------------------------


class _SharedState:
    """Mutable state shared between the UDP / control / tune threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # Snapshot consumed by the telemetry sender.
        self.telemetry: List[float] = [0.0] * TELEMETRY_SIZE
        # Latest fields the tuner thread publishes:
        self.state: TunerState = TunerState.IDLE
        self.run_index: int = 0
        self.stroke_index: int = 0
        self.ee_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.cmd_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.ramp_dir: int = 0
        self.last_result: Optional[RunResult] = None
        self.best_cost: float = math.inf
        self.best_cost_fwd: float = math.inf
        self.best_cost_rev: float = math.inf
        self.tuner_iter: int = 0
        self.dir_pid_enabled: bool = False
        self.tuned_joint: int = 1  # boom by default
        self.cfg: ProgramConfig = ProgramConfig(
            x_min=0.30, x_max=0.50, y=0.0, z=0.20, speed=0.040,
            strokes_per_run=4,
        )
        self.num_runs_remaining: int = 0
        self.err_now: float = 0.0


class PIDTunerEEHost:
    """Glue between UDP / control loop / auto-tuner."""

    def __init__(
        self,
        host: str,
        port: int,
        rate_hz: float,
        controller: ExcavatorController,
        hardware: HardwareInterface,
        log: logging.Logger,
    ) -> None:
        self.host = host
        self.port = port
        self.rate_hz = float(rate_hz)
        self.controller = controller
        self.hardware = hardware
        self.log = log
        self.shared = _SharedState()
        self._stop = threading.Event()
        self._tuner_thread: Optional[threading.Thread] = None
        self._cmd_lock = threading.Lock()
        self._pending_cmd: Optional[List[float]] = None
        # Track which PID slots we have replaced so we can restore them.
        self._original_pids: Dict[int, PIDController] = {}

        self.sock = UDPSocket(local_id=14, max_age_seconds=0.5, nominal_rate_hz=rate_hz)
        self.sock.setup(
            host, port,
            inputs=f"{COMMAND_SIZE}f",
            outputs=f"{TELEMETRY_SIZE}f",
            is_server=True,
        )

    # ----- joint PID install / restore ------------------------------------

    def _install_directional_pid(self, joint_idx: int, gains: GainsPair) -> None:
        if joint_idx not in range(4):
            return
        existing = self.controller.joint_pids[joint_idx]
        if joint_idx not in self._original_pids:
            self._original_pids[joint_idx] = existing
        # Inherit output limits from whatever the controller is currently using.
        min_out = float(getattr(existing, "min_output", -1.0))
        max_out = float(getattr(existing, "max_output", 1.0))
        wrapper = DirectionalPID(
            gains_pos=tuple(gains.fwd),
            gains_neg=tuple(gains.rev),
            min_output=min_out,
            max_output=max_out,
        )
        self.controller.joint_pids[joint_idx] = wrapper
        self.shared.dir_pid_enabled = True

    def _restore_pids(self) -> None:
        for idx, original in self._original_pids.items():
            self.controller.joint_pids[idx] = original
            original.reset()
        self._original_pids.clear()
        self.shared.dir_pid_enabled = False

    def _apply_scalar_gains(self, joint_idx: int, kp: float, ki: float, kd: float) -> None:
        pid = self.controller.joint_pids[joint_idx]
        if isinstance(pid, DirectionalPID):
            pid.set_gains_pos(kp, ki, kd)
            pid.set_gains_neg(kp, ki, kd)
            pid.reset()
        else:
            pid.kp, pid.ki, pid.kd = kp, ki, kd
            if ki != 0.0:
                pid.set_integral_limits(pid.min_output / ki, pid.max_output / ki)
            pid.reset()

    def _read_installed_gains(self, joint_idx: int) -> GainsPair:
        pid = self.controller.joint_pids[joint_idx]
        if isinstance(pid, DirectionalPID):
            return GainsPair(fwd=list(pid.gains_pos()), rev=list(pid.gains_neg()))
        g = [float(pid.kp), float(pid.ki), float(pid.kd)]
        return GainsPair(fwd=g, rev=list(g))

    # ----- UDP receive ----------------------------------------------------

    def _consume_command(self) -> Optional[List[float]]:
        pkt = self.sock.get_latest()
        if pkt is None or len(pkt) < COMMAND_SIZE:
            return None
        return [float(v) for v in pkt[:COMMAND_SIZE]]

    def _handle_command(self, cmd: List[float]) -> None:
        flags = int(round(cmd[CMD_FLAGS])) & 0xFFFF
        tuned_joint = int(round(cmd[CMD_TUNED_JOINT])) & 0x3
        self.shared.tuned_joint = tuned_joint

        # Always refresh the program config so the next run uses the latest.
        with self.shared.lock:
            self.shared.cfg = ProgramConfig(
                x_min=float(cmd[CMD_X_MIN]),
                x_max=float(cmd[CMD_X_MAX]),
                y=float(cmd[CMD_Y]),
                z=float(cmd[CMD_Z]),
                speed=float(np.clip(cmd[CMD_SPEED], 0.005, 0.200)),
                strokes_per_run=max(1, int(round(cmd[CMD_STROKES_PER_RUN]))),
                cost_w_rmse=float(cmd[CMD_COST_W_RMSE]),
                cost_w_max=float(cmd[CMD_COST_W_MAX]),
                cost_w_overshoot=DEFAULT_COST_WEIGHTS[2],
                z_min=float(cmd[CMD_Z_MIN]),
                z_max=float(cmd[CMD_Z_MAX]),
                speed_min=float(np.clip(cmd[CMD_SPEED_MIN], 0.005, 0.200)),
                speed_max=float(np.clip(cmd[CMD_SPEED_MAX], 0.005, 0.200)),
            )
            self.shared.num_runs_remaining = max(0, int(round(cmd[CMD_NUM_RUNS])))

        if flags & int(CommandFlag.RELOAD_CONFIG):
            try:
                self.hardware.reload_config()
            except Exception as exc:
                self.log.warning("reload_config failed: %s", exc)

        if flags & int(CommandFlag.RESET_PIDS):
            for pid in self.controller.joint_pids:
                try:
                    pid.reset()
                except Exception:
                    pass

        if flags & int(CommandFlag.DIR_PID):
            current = self._read_installed_gains(tuned_joint)
            self._install_directional_pid(tuned_joint, current)
        # Note: turning DIR_PID off requires a STOP+RESET — we don't quietly
        # restore in the middle of a run.

        if flags & int(CommandFlag.APPLY_GAINS):
            kp_f = float(cmd[CMD_KP_FWD])
            ki_f = float(cmd[CMD_KI_FWD])
            kd_f = float(cmd[CMD_KD_FWD])
            kp_r = float(cmd[CMD_KP_REV])
            ki_r = float(cmd[CMD_KI_REV])
            kd_r = float(cmd[CMD_KD_REV])
            pid = self.controller.joint_pids[tuned_joint]
            if isinstance(pid, DirectionalPID):
                pid.set_gains_pos(kp_f, ki_f, kd_f)
                pid.set_gains_neg(kp_r, ki_r, kd_r)
                pid.reset()
            else:
                self._apply_scalar_gains(tuned_joint, kp_f, ki_f, kd_f)

        if flags & int(CommandFlag.STOP):
            self._stop_run()

        if flags & int(CommandFlag.START):
            auto = bool(flags & int(CommandFlag.AUTO_TUNE))
            self._start_run(auto=auto)

    # ----- tuner thread lifecycle ----------------------------------------

    def _start_run(self, auto: bool) -> None:
        if self._tuner_thread is not None and self._tuner_thread.is_alive():
            self.log.info("Tuner already running — ignoring START")
            return
        self._stop.clear()
        target = self._auto_tune_loop if auto else self._manual_run_loop
        self._tuner_thread = threading.Thread(target=target, daemon=True)
        self._tuner_thread.start()

    def _stop_run(self) -> None:
        self._stop.set()
        if self._tuner_thread is not None:
            self._tuner_thread.join(timeout=2.0)
            self._tuner_thread = None
        try:
            self.controller.pause()
        except Exception:
            pass
        with self.shared.lock:
            self.shared.state = TunerState.IDLE
            self.shared.ramp_dir = 0

    # ----- back-and-forth runner -----------------------------------------

    def _execute_run(self) -> Optional[RunResult]:
        """Drive one back-and-forth program. Returns the run cost."""
        with self.shared.lock:
            cfg = self.shared.cfg
            self.shared.state = TunerState.RUNNING

        # Resume controller into a known-clean state.
        try:
            self.controller.resume()
        except Exception as exc:
            self.log.warning("controller.resume failed: %s", exc)

        # Seed motion smoother with a high velocity ceiling so our ramp drives
        # the actual reference without being slowed by jerk-limited smoothing.
        try:
            self.controller.motion_processor.max_velocity = max(
                self.controller.motion_processor.max_velocity, cfg.speed * 4.0
            )
        except Exception:
            pass

        dt = 1.0 / self.rate_hz
        cmd_x_samples: List[float] = []
        ee_x_samples: List[float] = []
        stroke_dirs: List[int] = []

        # Stroke counter: even index = travelling from x_min -> x_max, odd = back.
        for stroke in range(cfg.strokes_per_run):
            if self._stop.is_set():
                return None
            forward = (stroke % 2 == 0)
            start_x = cfg.x_min if forward else cfg.x_max
            end_x = cfg.x_max if forward else cfg.x_min
            direction = +1 if forward else -1
            with self.shared.lock:
                self.shared.stroke_index = stroke
                self.shared.ramp_dir = direction
            # Pin every DirectionalPID we own to this stroke's direction so
            # the per-direction subscores cleanly attribute to the matching
            # gain triple.
            for pid in self.controller.joint_pids:
                if isinstance(pid, DirectionalPID):
                    pid.set_active_direction(direction)
            x = start_x
            distance = abs(end_x - start_x)
            duration = distance / max(cfg.speed, 1e-3)
            t0 = time.perf_counter()
            next_t = t0
            while True:
                if self._stop.is_set():
                    return None
                t = time.perf_counter() - t0
                if t >= duration:
                    x = end_x
                else:
                    x = start_x + direction * cfg.speed * t
                pos = np.array([x, cfg.y, cfg.z], dtype=np.float32)
                try:
                    self.controller.give_pose(pos, 0.0)
                except Exception as exc:
                    self.log.warning("give_pose failed: %s", exc)
                # Sample measured EE pose.
                try:
                    measured, _ = self.controller.get_pose()
                except Exception:
                    measured = np.array([np.nan, np.nan, np.nan])
                cmd_x_samples.append(float(x))
                ee_x_samples.append(float(measured[0]))
                stroke_dirs.append(direction)
                with self.shared.lock:
                    self.shared.cmd_pos = (float(x), cfg.y, cfg.z)
                    self.shared.ee_pos = (
                        float(measured[0]), float(measured[1]), float(measured[2])
                    )
                    self.shared.err_now = abs(float(measured[0]) - float(x)) * 1000.0
                if t >= duration:
                    break
                next_t += dt
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        with self.shared.lock:
            self.shared.state = TunerState.SCORING
            self.shared.ramp_dir = 0
        # Release the directional pin so the wrapper falls back to sign-of-error
        # if anything calls compute() between runs.
        for pid in self.controller.joint_pids:
            if isinstance(pid, DirectionalPID):
                pid.set_active_direction(0)
        result = score_run(cmd_x_samples, ee_x_samples, stroke_dirs, cfg)
        with self.shared.lock:
            self.shared.last_result = result
        return result

    def _manual_run_loop(self) -> None:
        try:
            result = self._execute_run()
            if result is not None:
                self.log.info(
                    "manual run: rmse=%.2fmm max=%.2fmm mean=%.2fmm",
                    result.rmse, result.max_abs_err, result.mean_abs_err,
                )
        except Exception as exc:
            self.log.exception("manual run failed: %s", exc)
        finally:
            try:
                self.controller.pause()
            except Exception:
                pass
            with self.shared.lock:
                self.shared.state = TunerState.IDLE

    # ----- auto-tune loop -------------------------------------------------

    def _auto_tune_loop(self) -> None:
        try:
            self._auto_tune_inner()
        except Exception as exc:
            self.log.exception("auto-tune failed: %s", exc)
        finally:
            try:
                self.controller.pause()
            except Exception:
                pass
            with self.shared.lock:
                self.shared.state = TunerState.IDLE

    def _auto_tune_inner(self) -> None:
        with self.shared.lock:
            joint_idx = self.shared.tuned_joint
            budget = max(1, self.shared.num_runs_remaining)
            cfg_base = self.shared.cfg

        # Ensure directional PID is installed for asymmetric tuning.
        current = self._read_installed_gains(joint_idx)
        self._install_directional_pid(joint_idx, current)

        # Initial step sizes — heuristic: kp 1.0, ki 0.1, kd 0.05.
        steps = GainsPair(fwd=[1.0, 0.1, 0.05], rev=[1.0, 0.1, 0.05])
        tuner = CoordinateDescentTuner(initial=current, steps=steps)

        # Build a (z, speed) grid so each candidate is scored across the
        # configured envelope rather than a single condition.
        grid = _autotune_grid(cfg_base)
        if not grid:
            grid = [(cfg_base.z, cfg_base.speed)]
        self.log.info("auto-tune grid (%d points): %s",
                      len(grid), [f"z={z:.3f} v={v*1000:.0f}mm/s" for z, v in grid])

        # Seed best_cost with a baseline run at the first grid point.
        with self.shared.lock:
            self.shared.state = TunerState.AUTO_TUNING
            self.shared.cfg.z = grid[0][0]
            self.shared.cfg.speed = grid[0][1]

        baseline = self._execute_run()
        if baseline is None:
            return
        tuner.best_cost_fwd = baseline.fwd.cost
        tuner.best_cost_rev = baseline.rev.cost
        tuner.best = current.copy()
        with self.shared.lock:
            self.shared.best_cost = tuner.best_cost
            self.shared.best_cost_fwd = baseline.fwd.cost
            self.shared.best_cost_rev = baseline.rev.cost

        for run_idx in range(budget):
            if self._stop.is_set():
                break
            if tuner.converged():
                self.log.info("auto-tune converged after %d runs", run_idx)
                break
            candidate = tuner.propose()

            # Cycle through the (z, speed) grid round-robin so the tuner sees
            # a fixed reference cost surface (deterministic, not randomised).
            z_now, v_now = grid[run_idx % len(grid)]
            with self.shared.lock:
                self.shared.cfg.z = z_now
                self.shared.cfg.speed = v_now
                self.shared.run_index = run_idx + 1
                self.shared.tuner_iter = tuner.iter_count()

            # Push candidate into the joint's directional PID.
            pid = self.controller.joint_pids[joint_idx]
            if isinstance(pid, DirectionalPID):
                pid.set_gains_pos(*candidate.fwd)
                pid.set_gains_neg(*candidate.rev)
                pid.reset()
            else:
                self._apply_scalar_gains(joint_idx, *candidate.fwd)

            side = tuner.active_side()
            result = self._execute_run()
            if result is None:
                break
            tuner.observe(result.fwd.cost, result.rev.cost)
            with self.shared.lock:
                self.shared.best_cost = tuner.best_cost
                self.shared.best_cost_fwd = tuner.best_cost_fwd
                self.shared.best_cost_rev = tuner.best_cost_rev
                self.shared.num_runs_remaining = max(0, budget - (run_idx + 1))

            self.log.info(
                "iter %d side=%s (z=%.3f v=%.3f): cost_fwd=%.2f cost_rev=%.2f "
                "best_fwd=%.2f best_rev=%.2f gains_fwd=%s gains_rev=%s",
                run_idx + 1, side, z_now, v_now,
                result.fwd.cost, result.rev.cost,
                tuner.best_cost_fwd, tuner.best_cost_rev,
                ["%.3f" % g for g in candidate.fwd],
                ["%.3f" % g for g in candidate.rev],
            )

        # On exit, install the best known gains.
        pid = self.controller.joint_pids[joint_idx]
        if isinstance(pid, DirectionalPID):
            pid.set_gains_pos(*tuner.best.fwd)
            pid.set_gains_neg(*tuner.best.rev)
            pid.reset()

    # ----- telemetry pack -------------------------------------------------

    def _build_telemetry(self) -> List[float]:
        out = [0.0] * TELEMETRY_SIZE
        with self.shared.lock:
            cfg = self.shared.cfg
            state = self.shared.state
            ee = self.shared.ee_pos
            cmd = self.shared.cmd_pos
            last = self.shared.last_result
            ramp = self.shared.ramp_dir
            dir_pid = self.shared.dir_pid_enabled
            joint = self.shared.tuned_joint
            run_idx = self.shared.run_index
            stroke_idx = self.shared.stroke_index
            iter_count = self.shared.tuner_iter
            best_cost = self.shared.best_cost
            best_cost_fwd = self.shared.best_cost_fwd
            best_cost_rev = self.shared.best_cost_rev
            num_runs = self.shared.num_runs_remaining
            err_now = self.shared.err_now

        try:
            hw_ready = bool(self.hardware.is_hardware_ready())
        except Exception:
            hw_ready = False

        installed = self._read_installed_gains(joint)

        out[TLM_STATE] = float(int(state))
        out[TLM_HW_READY] = 1.0 if hw_ready else 0.0
        out[TLM_DIR_PID] = 1.0 if dir_pid else 0.0
        out[TLM_TUNED_JOINT] = float(joint)
        out[TLM_RUN_INDEX] = float(run_idx)
        out[TLM_STROKE_INDEX] = float(stroke_idx)
        out[TLM_TOTAL_STROKES] = float(cfg.strokes_per_run)
        out[TLM_EE_X], out[TLM_EE_Y], out[TLM_EE_Z] = ee
        out[TLM_CMD_X], out[TLM_CMD_Y], out[TLM_CMD_Z] = cmd
        out[TLM_RAMP_DIR] = float(ramp)
        out[TLM_CUR_SPEED] = float(cfg.speed)
        out[TLM_CUR_Z] = float(cfg.z)
        out[TLM_CUR_X_MIN] = float(cfg.x_min)
        out[TLM_CUR_X_MAX] = float(cfg.x_max)
        if last is not None:
            out[TLM_RMSE_LAST] = float(last.rmse)
            out[TLM_MAXERR_LAST] = float(last.max_abs_err)
            out[TLM_MEANERR_LAST] = float(last.mean_abs_err)
            out[TLM_LAST_COST] = float(last.cost)
            out[TLM_RMSE_LAST_FWD] = float(last.fwd.rmse)
            out[TLM_RMSE_LAST_REV] = float(last.rev.rmse)
            out[TLM_MAXERR_LAST_FWD] = float(last.fwd.max_abs_err)
            out[TLM_MAXERR_LAST_REV] = float(last.rev.max_abs_err)
            out[TLM_MEANERR_LAST_FWD] = float(last.fwd.mean_abs_err)
            out[TLM_MEANERR_LAST_REV] = float(last.rev.mean_abs_err)
            out[TLM_LAST_COST_FWD] = float(last.fwd.cost if math.isfinite(last.fwd.cost) else 0.0)
            out[TLM_LAST_COST_REV] = float(last.rev.cost if math.isfinite(last.rev.cost) else 0.0)
        out[TLM_BEST_COST] = float(best_cost if math.isfinite(best_cost) else 0.0)
        out[TLM_BEST_COST_FWD] = float(best_cost_fwd if math.isfinite(best_cost_fwd) else 0.0)
        out[TLM_BEST_COST_REV] = float(best_cost_rev if math.isfinite(best_cost_rev) else 0.0)
        out[TLM_ITER] = float(iter_count)
        out[TLM_KP_FWD] = float(installed.fwd[0])
        out[TLM_KI_FWD] = float(installed.fwd[1])
        out[TLM_KD_FWD] = float(installed.fwd[2])
        out[TLM_KP_REV] = float(installed.rev[0])
        out[TLM_KI_REV] = float(installed.rev[1])
        out[TLM_KD_REV] = float(installed.rev[2])
        out[TLM_NUM_RUNS] = float(num_runs)
        out[TLM_ERR_NOW] = float(err_now)
        return out

    # ----- main loop ------------------------------------------------------

    def run_forever(self) -> None:
        self.log.info("Waiting for client on %s:%d", self.host, self.port)
        if not self.sock.handshake(timeout=120.0):
            self.log.error("Handshake failed")
            return
        self.sock.start_receiving()
        period = 1.0 / self.rate_hz
        next_t = time.perf_counter()
        try:
            while True:
                cmd = self._consume_command()
                if cmd is not None:
                    try:
                        self._handle_command(cmd)
                    except Exception as exc:
                        self.log.exception("command handler raised: %s", exc)
                try:
                    self.sock.send(self._build_telemetry())
                except Exception as exc:
                    self.log.warning("send failed: %s", exc)
                next_t += period
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.perf_counter()
        except KeyboardInterrupt:
            self.log.info("Interrupted, shutting down")
        finally:
            self._stop_run()
            self._restore_pids()
            try:
                self.sock.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EE-plane PID tuner (robot/host side)")
    parser.add_argument("--robot", default="auto", help="board profile name (auto/rpi/jetson)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Local IP to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP port")
    parser.add_argument("--rate", type=float, default=50.0, help="UDP / inner loop rate Hz")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--warmup", type=float, default=5.0,
                        help="seconds to wait after controller.start() for numba warmup")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("pid_tuner_ee")

    profile = resolve_board_profile(args.robot)
    log.info(
        "Profile: %s board=%s servo=%s control=%s",
        profile["profile_name"], profile["board"],
        profile["servo_config_file"], profile["control_config_file"],
    )

    log.info("Initializing HardwareInterface...")
    hardware = HardwareInterface(
        config_file=profile["servo_config_file"],
        control_config_file=profile["control_config_file"],
        cleanup_disable_osc=False,
        enable_adc=False,
        enable_imu=bool(profile.get("enable_imu", True)),
        start_adc_reader=False,
        pwm_i2c_bus=profile["pwm_i2c_bus"],
        pwm_i2c_addr=profile["pwm_i2c_addr"],
    )

    log.info("Initializing ExcavatorController...")
    controller = ExcavatorController(
        hardware_interface=hardware,
        enable_perf_tracking=False,
        log_level=args.log_level,
        control_config_file=profile["control_config_file"],
    )
    controller.start()

    if args.warmup > 0:
        log.info("Warmup sleep %.1fs", args.warmup)
        time.sleep(args.warmup)

    log.info("Waiting for hardware readiness...")
    t_start = time.time()
    while not hardware.is_hardware_ready():
        if time.time() - t_start > 90.0:
            log.error("Timed out waiting for hardware readiness")
            break
        time.sleep(0.5)

    # Start paused — only run when client sends START.
    try:
        controller.pause()
    except Exception:
        pass

    tuner_host = PIDTunerEEHost(
        host=args.host,
        port=args.port,
        rate_hz=args.rate,
        controller=controller,
        hardware=hardware,
        log=log,
    )
    try:
        tuner_host.run_forever()
    finally:
        try:
            controller.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
