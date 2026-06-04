#!/usr/bin/env python3
"""PID tuner — robot / host side.

Runs both tuning modes under a single process:

* **Per-joint mode** (port 8090): direct IMU → PID → single-valve loop via
  ``HardwareInterface``. Good for rough per-joint gain setting.
* **EE-plane mode** (port 8091): full IK stack via ``ExcavatorController``.
  Auto-tunes boom + arm + bucket simultaneously against cartesian X RMSE.

Only one mode may write hardware outputs at a time. The per-joint handler takes
priority: when it enables output the EE host refuses to start (and aborts any
running tune). Switch back to EE mode by disabling the per-joint output first.

TODO(refactor — pid_tuner cleanup):
    Two things to merge in one pass:

    1. Collapse to a SINGLE UDP port. The dual-port design (8090 joint,
       8091 EE) only exists because the schemas differ. Add a 1-byte
       ``mode`` tag in the packet header (see ``common.py``) and route on
       it server-side. That deletes ``CombinedServer``, the
       joint-takes-priority mutex, and one handshake.

    2. Move the EE *optimizer* (cost function, ``AllJointGains`` packing,
       ``CoordinateDescentTuner``, gain ramping, best-cost tracking) to
       ``tools/pid_tuner/client.py``. The robot's job here should be
       "execute a stroke with gains X, return (rmse, max_err, time)" — no
       more. That shrinks ``robot.py`` from ~1.5 kLOC to ~400 LOC and
       lets the optimizer keep state across robot restarts.

    See the marked sections below and ``common.py`` for the port + mode
    byte change.

Run on the robot
----------------
::

    python -m tools.pid_tuner.robot

or::

    python tools/pid_tuner/robot.py

Then open ``tools/pid_tuner/client.py`` on the PC. Both tabs connect
independently; you can connect/disconnect each without restarting the robot.

Safety
------
The hydraulic pump is enabled when an EE run starts and stays on between
auto-tune iterations for stability. It is disabled on explicit STOP, on error,
and at exit. The controller zeroes valve outputs while between runs. Unreachable
targets are rejected by the controller's reachability check. Always confirm a
safe A/B sweep range before starting on real hardware.
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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.board import resolve_profile as resolve_board_profile
from modules.excavator_controller import ExcavatorController
from modules.ik import load_excavator_robot_config, canonical_joint_angles_from_imus
from modules.hardware_interface import HardwareInterface
from modules.pid import PIDController
from modules.udp_socket import UDPSocket

from tools.pid_tuner.common import (
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
    DEFAULT_COST_WEIGHTS,
    DEFAULT_HOST,
    EE_COMMAND_SIZE,
    EE_PORT,
    EE_TELEMETRY_SIZE,
    EE_TUNABLE_JOINTS,
    EECommandFlag,
    EETunerState,
    JOINT_COMMAND_SIZE,
    JOINT_NAMES,
    JOINT_PORT,
    JOINT_TELEMETRY_SIZE,
    PWM_NAMES,
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
)

# Backward-compat aliases used by tests.
COMMAND_SIZE = JOINT_COMMAND_SIZE
TELEMETRY_SIZE = JOINT_TELEMETRY_SIZE


# ---------------------------------------------------------------------------
# Per-joint helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _angle_error(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


def _safe_joint_id(value: float) -> int:
    return max(0, min(3, int(round(float(value)))))


def _read_joint_angles(hardware: HardwareInterface, robot_config) -> np.ndarray:
    quats = hardware.read_all_imu_quaternions()
    if quats is None:
        raise RuntimeError("IMU quaternions unavailable")
    return canonical_joint_angles_from_imus(
        np.asarray(quats, dtype=np.float32), robot_config
    )


# ---------------------------------------------------------------------------
# Direction-specific PID
# ---------------------------------------------------------------------------


class DirectionalPID:
    """Two :class:`PIDController` instances chosen by stroke direction.

    Positive ``error`` → ``pid_pos``; negative → ``pid_neg``.
    Call :meth:`set_active_direction` at the start of each stroke to pin the
    active side so per-direction subscores are cleanly attributed.
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
        self.min_output = min_output
        self.max_output = max_output
        self._active_dir: int = 0

    def set_active_direction(self, direction: int) -> None:
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

    def compute(self, setpoint: float, current_value: float,
                dt: Optional[float] = None) -> float:
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
        inactive.last_value = current_value
        return active.compute(setpoint, current_value, dt=dt)


# ---------------------------------------------------------------------------
# Auto-tuner data structures
# ---------------------------------------------------------------------------
# TODO(refactor #1): Move EVERYTHING from here through the end of
# ``CoordinateDescentTuner`` (~line 440) into ``tools/pid_tuner/client.py``.
# The robot never needs to know how gains are chosen — only how to apply
# them and report the resulting per-stroke metrics.
# ---------------------------------------------------------------------------


@dataclass
class GainsPair:
    fwd: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rev: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def as_vector(self) -> List[float]:
        return [*self.fwd, *self.rev]

    @staticmethod
    def from_vector(v: List[float]) -> "GainsPair":
        return GainsPair(fwd=list(v[:3]), rev=list(v[3:6]))

    def copy(self) -> "GainsPair":
        return GainsPair(fwd=list(self.fwd), rev=list(self.rev))


@dataclass
class AllJointGains:
    """Gains for all EE-tunable joints.

    Flat vector layout::

        [kp_fwd_0, ki_fwd_0, kd_fwd_0,  # boom fwd
         kp_fwd_1, ...                   # arm, bucket fwd
         kp_rev_0, ...                   # boom, arm, bucket rev]
    """

    per_joint: List[GainsPair]

    @property
    def n_joints(self) -> int:
        return len(self.per_joint)

    def as_vector(self) -> List[float]:
        return ([g for gp in self.per_joint for g in gp.fwd] +
                [g for gp in self.per_joint for g in gp.rev])

    @staticmethod
    def from_vector(v: List[float]) -> "AllJointGains":
        n = len(v) // 6
        return AllJointGains(per_joint=[
            GainsPair(fwd=list(v[3 * i: 3 * i + 3]),
                      rev=list(v[3 * n + 3 * i: 3 * n + 3 * i + 3]))
            for i in range(n)
        ])

    def copy(self) -> "AllJointGains":
        return AllJointGains(per_joint=[gp.copy() for gp in self.per_joint])


class CoordinateDescentTuner:
    """Twiddle-style coordinate descent over all EE-joint PID gains simultaneously.

    The search covers boom, arm, and bucket at once so the cost signal
    (cartesian X RMSE) receives contributions from every joint. Fwd stroke costs
    gate only fwd-gain dimensions; rev costs gate only rev-gain dimensions.
    """

    def __init__(
        self,
        initial: AllJointGains,
        steps: AllJointGains,
        frozen_mask: Optional[List[bool]] = None,
        grow_factor: float = 1.1,
        shrink_factor: float = 0.5,
        min_step: float = 1e-3,
        gain_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (0.0, 50.0),
            (0.0, 20.0),
            (0.0, 20.0),
        ),
    ) -> None:
        n = initial.n_joints
        self._n_joints = n
        self._n_fwd = n * 3
        self._n_dims = n * 6
        self._fwd_dims: frozenset = frozenset(range(self._n_fwd))
        self._rev_dims: frozenset = frozenset(range(self._n_fwd, self._n_dims))

        self.current = initial.copy()
        self.best = initial.copy()
        self.best_cost_fwd: float = math.inf
        self.best_cost_rev: float = math.inf
        self.step = steps.as_vector()
        self.frozen = list(frozen_mask) if frozen_mask is not None else [False] * self._n_dims
        self.grow = float(grow_factor)
        self.shrink = float(shrink_factor)
        self.min_step = float(min_step)
        self.bounds = gain_bounds
        self._dim_index = 0
        self._phase = 0
        self._pending_revert: Optional[float] = None
        self._iter = 0

    @property
    def best_cost(self) -> float:
        f = self.best_cost_fwd if math.isfinite(self.best_cost_fwd) else 0.0
        r = self.best_cost_rev if math.isfinite(self.best_cost_rev) else 0.0
        return f + r

    def active_side(self) -> str:
        return "fwd" if self._dim_index in self._fwd_dims else "rev"

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
        self.current = AllJointGains.from_vector(v)

    def _advance_dim(self) -> None:
        for _ in range(self._n_dims):
            self._dim_index = (self._dim_index + 1) % self._n_dims
            if not self.frozen[self._dim_index]:
                break
        self._phase = 0
        self._pending_revert = None

    def propose(self) -> AllJointGains:
        for _ in range(self._n_dims):
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
        return AllJointGains.from_vector(v)

    def observe(self, cost_fwd: float, cost_rev: float) -> None:
        self._iter += 1
        d = self._dim_index
        side = "fwd" if d in self._fwd_dims else "rev"
        my_cost = cost_fwd if side == "fwd" else cost_rev
        my_best = self.best_cost_fwd if side == "fwd" else self.best_cost_rev

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
                self.step[d] *= self.shrink
            self._advance_dim()


# ---------------------------------------------------------------------------
# Run scoring
# ---------------------------------------------------------------------------


@dataclass
class DirectionStats:
    rmse: float = 0.0
    max_abs_err: float = 0.0
    mean_abs_err: float = 0.0
    cost: float = math.inf
    samples: int = 0


@dataclass
class RunResult:
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
    z_min: float = 0.18
    z_max: float = 0.22
    speed_min: float = 0.020
    speed_max: float = 0.070


def _autotune_grid(cfg: ProgramConfig, n_z: int = 3, n_v: int = 3) -> List[Tuple[float, float]]:
    zs = np.linspace(cfg.z_min, cfg.z_max, n_z).tolist() if cfg.z_max > cfg.z_min else [cfg.z]
    vs = np.linspace(cfg.speed_min, cfg.speed_max, n_v).tolist() if cfg.speed_max > cfg.speed_min else [cfg.speed]
    return [(float(z), float(v)) for z in zs for v in vs]


def _score_subset(err_mm: np.ndarray, cfg: ProgramConfig) -> DirectionStats:
    if err_mm.size == 0:
        return DirectionStats()
    rmse = float(np.sqrt(np.mean(err_mm ** 2)))
    max_abs = float(np.max(np.abs(err_mm)))
    mean_abs = float(np.mean(np.abs(err_mm)))
    tail = err_mm[int(0.9 * err_mm.size):]
    overshoot = float(np.mean(np.clip(np.abs(tail) - rmse, 0.0, None))) if tail.size else 0.0
    cost = cfg.cost_w_rmse * rmse + cfg.cost_w_max * max_abs + cfg.cost_w_overshoot * overshoot
    return DirectionStats(rmse=rmse, max_abs_err=max_abs, mean_abs_err=mean_abs,
                          cost=cost, samples=int(err_mm.size))


def score_run(
    cmd_x_samples: List[float],
    ee_x_samples: List[float],
    stroke_dirs: List[int],
    cfg: ProgramConfig,
) -> RunResult:
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
# EE host shared state
# ---------------------------------------------------------------------------


class _SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.telemetry: List[float] = [0.0] * EE_TELEMETRY_SIZE
        self.state: EETunerState = EETunerState.IDLE
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
        self.tuned_joint: int = 1
        self.cfg: ProgramConfig = ProgramConfig(
            x_min=0.30, x_max=0.50, y=0.0, z=0.20, speed=0.040, strokes_per_run=4,
        )
        self.num_runs_remaining: int = 0
        self.err_now: float = 0.0


# ---------------------------------------------------------------------------
# EE host
# ---------------------------------------------------------------------------


# TODO(refactor #1): EEHost is half "stroke executor" and half "optimizer
# driver". Split:
#   - Keep here (rename to EEStrokeExecutor): PID install/restore, command
#     handling, the run executor that actually drives the controller.
#   - Move to client: the auto-tune loop (~line 866), best-cost tracking,
#     telemetry encoding of optimizer-only fields (best/last cost, RMSE
#     subscores per direction). Robot sends raw per-stroke metrics; client
#     does the scoring and picks the next gain vector.
class EEHost:
    """EE-plane PID tuner host (port 8091)."""

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
        self._original_pids: Dict[int, PIDController] = {}
        self._joint_active: Optional[threading.Event] = None

        self.sock = UDPSocket(local_id=14, max_age_seconds=0.5, nominal_rate_hz=rate_hz)
        self.sock.setup(
            host, port,
            inputs=f"{EE_COMMAND_SIZE}f",
            outputs=f"{EE_TELEMETRY_SIZE}f",
            is_server=True,
        )

    def set_joint_priority_flag(self, flag: threading.Event) -> None:
        self._joint_active = flag

    # ----- PID install / restore ------------------------------------------

    def _install_directional_pid(self, joint_idx: int, gains: GainsPair) -> None:
        if joint_idx not in range(4):
            return
        existing = self.controller.joint_pids[joint_idx]
        if joint_idx not in self._original_pids:
            self._original_pids[joint_idx] = existing
        min_out = float(getattr(existing, "min_output", -1.0))
        max_out = float(getattr(existing, "max_output", 1.0))
        self.controller.joint_pids[joint_idx] = DirectionalPID(
            gains_pos=tuple(gains.fwd),
            gains_neg=tuple(gains.rev),
            min_output=min_out,
            max_output=max_out,
        )
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

    # ----- command handling -----------------------------------------------

    def _consume_command(self) -> Optional[List[float]]:
        pkt = self.sock.get_latest()
        if pkt is None or len(pkt) < EE_COMMAND_SIZE:
            return None
        return [float(v) for v in pkt[:EE_COMMAND_SIZE]]

    def _handle_command(self, cmd: List[float]) -> None:
        flags = int(round(cmd[CMD_FLAGS])) & 0xFFFF
        tuned_joint = int(round(cmd[CMD_TUNED_JOINT])) & 0x3
        self.shared.tuned_joint = tuned_joint

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

        if flags & int(EECommandFlag.RELOAD_CONFIG):
            try:
                self.hardware.reload_config()
            except Exception as exc:
                self.log.warning("reload_config failed: %s", exc)

        if flags & int(EECommandFlag.RESET_PIDS):
            for pid in self.controller.joint_pids:
                try:
                    pid.reset()
                except Exception:
                    pass

        if flags & int(EECommandFlag.DIR_PID):
            self._install_directional_pid(tuned_joint, self._read_installed_gains(tuned_joint))

        if flags & int(EECommandFlag.APPLY_GAINS):
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

        if flags & int(EECommandFlag.STOP):
            self._stop_run()

        if flags & int(EECommandFlag.START):
            if self._joint_active is not None and self._joint_active.is_set():
                self.log.warning("START rejected: per-joint tuner is active")
                return
            with self.shared.lock:
                cfg = self.shared.cfg
            if cfg.x_min >= cfg.x_max:
                self.log.warning(
                    "START rejected: x_min (%.3f) >= x_max (%.3f)", cfg.x_min, cfg.x_max)
            elif cfg.z <= 0.0:
                self.log.warning("START rejected: z=%.3f is not positive", cfg.z)
            else:
                self._start_run(auto=bool(flags & int(EECommandFlag.AUTO_TUNE)))

    # ----- tuner lifecycle ------------------------------------------------

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
        try:
            self.hardware.set_pump_enabled(False)
        except Exception:
            pass
        with self.shared.lock:
            self.shared.state = EETunerState.IDLE
            self.shared.ramp_dir = 0

    # ----- run executor ---------------------------------------------------

    def _execute_run(self) -> Optional[RunResult]:
        with self.shared.lock:
            cfg = self.shared.cfg
            self.shared.state = EETunerState.RUNNING

        try:
            self.controller.resume()
        except Exception as exc:
            self.log.warning("controller.resume failed: %s", exc)

        try:
            self.hardware.set_pump_enabled(True)
        except Exception as exc:
            self.log.warning("pump enable failed: %s", exc)

        orig_max_vel: Optional[float] = None
        try:
            orig_max_vel = self.controller.motion_processor.max_velocity
            self.controller.motion_processor.max_velocity = max(orig_max_vel, cfg.speed * 4.0)
        except Exception:
            pass

        dt = 1.0 / self.rate_hz
        cmd_x_samples: List[float] = []
        ee_x_samples: List[float] = []
        stroke_dirs: List[int] = []

        try:
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
                for pid in self.controller.joint_pids:
                    if isinstance(pid, DirectionalPID):
                        pid.set_active_direction(direction)
                distance = abs(end_x - start_x)
                duration = distance / max(cfg.speed, 1e-3)
                t0 = time.perf_counter()
                next_t = t0
                while True:
                    if self._stop.is_set():
                        return None
                    if self._joint_active is not None and self._joint_active.is_set():
                        self.log.warning("run aborted: per-joint tuner became active")
                        return None
                    t = time.perf_counter() - t0
                    x = end_x if t >= duration else start_x + direction * cfg.speed * t
                    pos = np.array([x, cfg.y, cfg.z], dtype=np.float32)
                    try:
                        result_reach = self.controller.give_pose(pos, 0.0)
                        if result_reach is not None and not getattr(result_reach, "reachable", True):
                            self.log.warning("give_pose: unreachable at x=%.3f z=%.3f", x, cfg.z)
                    except Exception as exc:
                        self.log.warning("give_pose failed: %s", exc)
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
                            float(measured[0]), float(measured[1]), float(measured[2]))
                        self.shared.err_now = abs(float(measured[0]) - float(x)) * 1000.0
                    if t >= duration:
                        break
                    next_t += dt
                    sleep_s = next_t - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

            with self.shared.lock:
                self.shared.state = EETunerState.SCORING
                self.shared.ramp_dir = 0
            for pid in self.controller.joint_pids:
                if isinstance(pid, DirectionalPID):
                    pid.set_active_direction(0)
            result = score_run(cmd_x_samples, ee_x_samples, stroke_dirs, cfg)
            with self.shared.lock:
                self.shared.last_result = result
            return result
        finally:
            if orig_max_vel is not None:
                try:
                    self.controller.motion_processor.max_velocity = orig_max_vel
                except Exception:
                    pass

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
                self.controller.pause(reset_pump=False)
            except Exception:
                pass
            with self.shared.lock:
                self.shared.state = EETunerState.IDLE

    # ----- auto-tune loop -------------------------------------------------

    def _auto_tune_loop(self) -> None:
        try:
            self._auto_tune_inner()
        except Exception as exc:
            self.log.exception("auto-tune failed: %s", exc)
        finally:
            try:
                self.controller.pause(reset_pump=False)
            except Exception:
                pass
            with self.shared.lock:
                self.shared.state = EETunerState.IDLE

    def _auto_tune_inner(self) -> None:
        with self.shared.lock:
            budget = max(1, self.shared.num_runs_remaining)
            cfg_base = self.shared.cfg

        current_gains: List[GainsPair] = []
        for jidx in EE_TUNABLE_JOINTS:
            g = self._read_installed_gains(jidx)
            self._install_directional_pid(jidx, g)
            current_gains.append(g)
        initial = AllJointGains(per_joint=current_gains)

        steps = AllJointGains(per_joint=[
            GainsPair(fwd=[1.0, 0.1, 0.05], rev=[1.0, 0.1, 0.05])
            for _ in EE_TUNABLE_JOINTS
        ])
        tuner = CoordinateDescentTuner(initial=initial, steps=steps)

        grid = _autotune_grid(cfg_base)
        if not grid:
            grid = [(cfg_base.z, cfg_base.speed)]
        self.log.info(
            "auto-tune: %d joints (%s), %d-dim search, grid %d pts",
            len(EE_TUNABLE_JOINTS),
            [JOINT_NAMES[j] for j in EE_TUNABLE_JOINTS],
            tuner._n_dims,
            len(grid),
        )

        with self.shared.lock:
            self.shared.state = EETunerState.AUTO_TUNING
            self.shared.cfg.z = grid[0][0]
            self.shared.cfg.speed = grid[0][1]

        baseline = self._execute_run()
        if baseline is None:
            return
        tuner.best_cost_fwd = baseline.fwd.cost
        tuner.best_cost_rev = baseline.rev.cost
        tuner.best = initial.copy()
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
            z_now, v_now = grid[run_idx % len(grid)]
            with self.shared.lock:
                self.shared.cfg.z = z_now
                self.shared.cfg.speed = v_now
                self.shared.run_index = run_idx + 1
                self.shared.tuner_iter = tuner.iter_count()

            for i, jidx in enumerate(EE_TUNABLE_JOINTS):
                pid = self.controller.joint_pids[jidx]
                if isinstance(pid, DirectionalPID):
                    pid.set_gains_pos(*candidate.per_joint[i].fwd)
                    pid.set_gains_neg(*candidate.per_joint[i].rev)
                    pid.reset()
                else:
                    self._apply_scalar_gains(jidx, *candidate.per_joint[i].fwd)

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

            with self.shared.lock:
                display_jidx = self.shared.tuned_joint
            try:
                di = list(EE_TUNABLE_JOINTS).index(display_jidx)
            except ValueError:
                di = 0
            dg = candidate.per_joint[di]
            self.log.info(
                "iter %d side=%s (z=%.3f v=%.3f): cost_fwd=%.2f cost_rev=%.2f "
                "best_fwd=%.2f best_rev=%.2f [%s fwd=%s rev=%s]",
                run_idx + 1, side, z_now, v_now,
                result.fwd.cost, result.rev.cost,
                tuner.best_cost_fwd, tuner.best_cost_rev,
                JOINT_NAMES[display_jidx],
                ["%.3f" % g for g in dg.fwd],
                ["%.3f" % g for g in dg.rev],
            )

        for i, jidx in enumerate(EE_TUNABLE_JOINTS):
            pid = self.controller.joint_pids[jidx]
            if isinstance(pid, DirectionalPID):
                pid.set_gains_pos(*tuner.best.per_joint[i].fwd)
                pid.set_gains_neg(*tuner.best.per_joint[i].rev)
                pid.reset()

    # ----- telemetry ------------------------------------------------------

    def _build_telemetry(self) -> List[float]:
        out = [0.0] * EE_TELEMETRY_SIZE
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
        self.log.info("EE: waiting for client on %s:%d", self.host, self.port)
        while not self.sock.handshake(timeout=30.0):
            self.log.info("EE: no client yet, retrying on %s:%d ...", self.host, self.port)
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
            self.log.info("EE: interrupted")
        finally:
            self._stop_run()
            self._restore_pids()
            try:
                self.sock.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Per-joint handler (thread)
# ---------------------------------------------------------------------------


class JointHandler:
    """Per-joint IMU → PID → valve loop running in its own thread (port 8090)."""

    def __init__(
        self,
        host: str,
        port: int,
        rate_hz: float,
        hardware: HardwareInterface,
        robot_config,
        log: logging.Logger,
        joint_active: threading.Event,
    ) -> None:
        self.host = host
        self.port = port
        self.rate_hz = float(rate_hz)
        self.hardware = hardware
        self.robot_config = robot_config
        self.log = log
        self._joint_active = joint_active

        self.sock = UDPSocket(local_id=12, max_age_seconds=0.5, nominal_rate_hz=rate_hz)
        self.sock.setup(
            host, port,
            inputs=f"{JOINT_COMMAND_SIZE}f",
            outputs=f"{JOINT_TELEMETRY_SIZE}f",
            is_server=True,
        )

    def run(self) -> None:
        self.log.info("Joint: waiting for client on %s:%d", self.host, self.port)
        while not self.sock.handshake(timeout=30.0):
            self.log.info("Joint: no client yet, retrying on %s:%d ...", self.host, self.port)
        self.sock.start_receiving()

        pid = PIDController(kp=0.0, ki=0.0, kd=0.0, min_output=-1.0, max_output=1.0)
        joint_id = 0
        target_deg = 0.0
        enabled = False
        pump_enabled = False
        last_gains = (0.0, 0.0, 0.0)
        last_joint_id = joint_id
        last_reset_flag = 0.0
        last_reload_flag = 0.0
        output = 0.0
        last_joint_angles: Optional[np.ndarray] = None

        period = 1.0 / max(1.0, self.rate_hz)
        next_t = time.perf_counter()
        last_status = time.time()

        try:
            while True:
                command = self.sock.get_latest()
                if command is not None and len(command) >= JOINT_COMMAND_SIZE:
                    requested_joint_id = _safe_joint_id(command[0])
                    requested_target_deg = _clamp(command[1], -180.0, 180.0)
                    kp = _clamp(command[2], 0.0, 50.0)
                    ki = _clamp(command[3], 0.0, 20.0)
                    kd = _clamp(command[4], 0.0, 20.0)
                    new_enabled = command[5] > 0.5
                    requested_pump = command[6] > 0.5
                    reload_flag = float(command[7])
                    reset_flag = float(command[8])
                    max_output = _clamp(command[9], 0.05, 1.0)

                    joint_changed = requested_joint_id != last_joint_id
                    joint_id = requested_joint_id
                    target_deg = requested_target_deg

                    gains = (kp, ki, kd)
                    if gains != last_gains or joint_changed:
                        pid = PIDController(
                            kp=kp, ki=ki, kd=kd,
                            min_output=-max_output, max_output=max_output,
                        )
                        last_gains = gains
                        last_joint_id = joint_id
                    else:
                        pid.min_output = -max_output
                        pid.max_output = max_output

                    if joint_changed and last_joint_angles is not None:
                        target_deg = float(math.degrees(last_joint_angles[joint_id]))
                        pid.reset()

                    if reset_flag > 0.5 and last_reset_flag <= 0.5:
                        pid.reset()
                    last_reset_flag = reset_flag

                    if reload_flag > 0.5 and last_reload_flag <= 0.5:
                        self.hardware.reload_config()
                    last_reload_flag = reload_flag

                    if requested_pump != pump_enabled:
                        pump_enabled = requested_pump
                        self.hardware.set_pump_enabled(pump_enabled)

                    if new_enabled and not enabled:
                        self._joint_active.set()
                        self.log.info("Joint output enabled — EE locked out")
                    elif not new_enabled and enabled:
                        self._joint_active.clear()
                        self.log.info("Joint output disabled — EE unlocked")
                    enabled = new_enabled
                else:
                    if enabled:
                        self._joint_active.clear()
                    enabled = False
                    if pump_enabled:
                        pump_enabled = False
                        self.hardware.set_pump_enabled(False)

                try:
                    joint_angles = _read_joint_angles(self.hardware, self.robot_config)
                    last_joint_angles = joint_angles.copy()
                    current_rad = float(joint_angles[joint_id])
                    target_rad = math.radians(target_deg)
                    error = _angle_error(target_rad, current_rad)

                    if enabled:
                        output = pid.compute(0.0, -error, dt=period)
                        self.hardware.send_named_pwm_commands(
                            {PWM_NAMES[joint_id]: output}, unset_to_zero=True
                        )
                    else:
                        output = 0.0
                        self.hardware.reset(reset_pump=False)

                    telemetry = [
                        float(joint_id),
                        float(math.degrees(joint_angles[0])),
                        float(math.degrees(joint_angles[1])),
                        float(math.degrees(joint_angles[2])),
                        float(math.degrees(joint_angles[3])),
                        float(target_deg),
                        float(output),
                        float(pid.kp),
                        float(pid.ki),
                        float(pid.kd),
                        1.0 if enabled else 0.0,
                        1.0 if pump_enabled else 0.0,
                    ]
                    self.sock.send(telemetry)
                except Exception as exc:
                    self.log.warning("Joint loop error: %s", exc)
                    self.hardware.reset(reset_pump=False)

                now = time.time()
                if now - last_status >= 1.0:
                    self.log.info(
                        "joint: %s target=%+.1f deg output=%+.3f enabled=%s pump=%s",
                        JOINT_NAMES[joint_id], target_deg, output, enabled, pump_enabled,
                    )
                    last_status = now

                next_t += period
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.perf_counter()

        except KeyboardInterrupt:
            self.log.info("Joint: interrupted")
        finally:
            self._joint_active.clear()
            try:
                self.hardware.reset(reset_pump=True)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Combined server
# ---------------------------------------------------------------------------


# TODO(refactor #1): Delete CombinedServer once the single-port migration
# lands. Replace with one UDPSocket bound to the new unified port (see
# ``common.py``); demux ``JointHandler`` vs the new ``EEStrokeExecutor``
# from the mode byte in the packet header. The "joint mode locks out EE
# mode" coupling disappears with the second socket.
class CombinedServer:
    def __init__(
        self,
        host: str,
        joint_port: int,
        ee_port: int,
        rate_hz: float,
        hardware: HardwareInterface,
        robot_config,
        controller: ExcavatorController,
        log: logging.Logger,
    ) -> None:
        self._joint_active = threading.Event()

        self.joint_handler = JointHandler(
            host=host, port=joint_port, rate_hz=rate_hz,
            hardware=hardware, robot_config=robot_config,
            log=log.getChild("joint"), joint_active=self._joint_active,
        )
        self.ee_host = EEHost(
            host=host, port=ee_port, rate_hz=rate_hz,
            controller=controller, hardware=hardware,
            log=log.getChild("ee"),
        )
        self.ee_host.set_joint_priority_flag(self._joint_active)
        self._log = log

    def run(self) -> None:
        joint_thread = threading.Thread(
            target=self.joint_handler.run, daemon=True, name="joint-tuner"
        )
        ee_thread = threading.Thread(
            target=self.ee_host.run_forever, daemon=True, name="ee-tuner"
        )
        joint_thread.start()
        ee_thread.start()
        self._log.info(
            "Both handlers started — joint port %d, EE port %d",
            self.joint_handler.port, self.ee_host.port,
        )
        try:
            while joint_thread.is_alive() or ee_thread.is_alive():
                joint_thread.join(timeout=1.0)
                ee_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            self._log.info("Interrupted")
        finally:
            self.ee_host._stop.set()
            joint_thread.join(timeout=3.0)
            ee_thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """Combined server: both per-joint (port 8090) and EE-plane (port 8091)."""
    parser = argparse.ArgumentParser(
        description="PID tuner — combined robot server (per-joint + EE-plane)"
    )
    parser.add_argument("--robot", default="auto")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--joint-port", type=int, default=JOINT_PORT)
    parser.add_argument("--ee-port", type=int, default=EE_PORT)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--warmup", type=float, default=5.0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("pid_tuner")

    profile = resolve_board_profile(args.robot)
    log.info("Profile: %s board=%s", profile["profile_name"], profile["board"])

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
    robot_config = load_excavator_robot_config(profile["control_config_file"])

    controller = ExcavatorController(
        hardware_interface=hardware,
        enable_perf_tracking=False,
        log_level=args.log_level,
        control_config_file=profile["control_config_file"],
    )
    controller.start()

    if args.warmup > 0:
        log.info("Warmup %.1fs", args.warmup)
        time.sleep(args.warmup)

    log.info("Waiting for hardware readiness...")
    t_start = time.time()
    while not hardware.is_hardware_ready():
        if time.time() - t_start > 90.0:
            log.error("Timed out waiting for hardware")
            break
        time.sleep(0.5)

    try:
        controller.pause()
    except Exception:
        pass

    server = CombinedServer(
        host=args.host,
        joint_port=args.joint_port,
        ee_port=args.ee_port,
        rate_hz=args.rate,
        hardware=hardware,
        robot_config=robot_config,
        controller=controller,
        log=log,
    )
    try:
        server.run()
    finally:
        try:
            controller.stop()
        except Exception:
            pass
        try:
            hardware.reset(reset_pump=True)
            hardware.shutdown()
        except Exception:
            pass
    return 0


def main_joint(argv=None) -> int:
    """Standalone per-joint server only (no EE, no ExcavatorController)."""
    parser = argparse.ArgumentParser(
        description="PID tuner — per-joint server only"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=JOINT_PORT)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--config-file",
                        default="configuration_files/profiles/rpi/servo_config.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("pid_tuner.joint")

    hardware = HardwareInterface(
        config_file=args.config_file,
        control_config_file="configuration_files/profiles/rpi/control_config.yaml",
        pump_auto_mode=False,
        cleanup_disable_osc=False,
        enable_adc=False,
        start_adc_reader=False,
        log_level=args.log_level,
    )
    robot_config = load_excavator_robot_config(
        "configuration_files/profiles/rpi/control_config.yaml"
    )

    log.info("Waiting for hardware")
    while not hardware.is_hardware_ready():
        time.sleep(0.1)

    handler = JointHandler(
        host=args.host, port=args.port, rate_hz=args.rate,
        hardware=hardware, robot_config=robot_config,
        log=log, joint_active=threading.Event(),
    )
    try:
        handler.run()
    finally:
        try:
            hardware.reset(reset_pump=True)
            hardware.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
