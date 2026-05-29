"""Shared constants and helpers for the EE-plane PID tuner.

Used by both the robot-side host (``tools/pid_tuner_ee.py``) and the client GUI
(``tools/pid_tuner_ee_client.py``) so they stay in sync without dragging the
production stack into the package.

Wire format
-----------
* All packets are flat ``<float32`` arrays exchanged via
  :class:`modules.udp_socket.UDPSocket`.
* Command   (client -> host) : ``COMMAND_SIZE``    floats — see ``CMD_*`` indices.
* Telemetry (host  -> client): ``TELEMETRY_SIZE``  floats — see ``TLM_*`` indices.

Keep the layout append-only: add new fields at the end and bump the size
constants in lock-step on both sides.

State codes
-----------
Both ends use the integer values in :class:`TunerState` for the ``TLM_STATE``
slot, transmitted as ``float(state)``.

Auto-tune cost
--------------
The cost reported in telemetry is::

    cost = w_rmse * RMSE + w_max * max|err| + w_overshoot * overshoot_metric

where ``err = ee_x - cmd_x`` over a single back-and-forth run, sampled at the
control rate. Weight defaults live in ``DEFAULT_COST_WEIGHTS``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Tuple


# ---------------------------------------------------------------------------
# Wire-format sizes
# ---------------------------------------------------------------------------

COMMAND_SIZE = 22
TELEMETRY_SIZE = 42


# ---------------------------------------------------------------------------
# Command packet layout — indices into the float32 array sent client -> host
# ---------------------------------------------------------------------------

CMD_FLAGS = 0          # bit field, see CommandFlag
CMD_X_MIN = 1          # metres, EE x sweep low end
CMD_X_MAX = 2          # metres, EE x sweep high end
CMD_Z = 3              # metres, fixed EE z height
CMD_Y = 4              # metres, fixed EE y (lateral); usually 0
CMD_SPEED = 5          # m/s, target linear EE speed (0.020..0.070)
CMD_STROKES_PER_RUN = 6  # int, sweeps per scored run (e.g. 4 = there/back/there/back)
CMD_NUM_RUNS = 7       # int, auto-tune iteration budget
CMD_TUNED_JOINT = 8    # int 0..3, slew/boom/arm/bucket
CMD_JOINT_MASK = 9     # bit field 0..3, which joint PIDs are owned by the tuner
CMD_KP_FWD = 10        # gain to push into tuned joint, forward direction
CMD_KI_FWD = 11
CMD_KD_FWD = 12
CMD_KP_REV = 13        # gain to push into tuned joint, reverse direction
CMD_KI_REV = 14
CMD_KD_REV = 15
CMD_COST_W_RMSE = 16   # auto-tune cost weights (informational; host applies)
CMD_COST_W_MAX = 17
# Auto-tune sweep ranges. CMD_Z / CMD_SPEED above are used for manual runs;
# during auto-tune each candidate run cycles through a small grid built from
# these (min, max) brackets so the tuned gains are robust across conditions.
CMD_Z_MIN = 18
CMD_Z_MAX = 19
CMD_SPEED_MIN = 20     # m/s, lower edge of EE-speed sweep (e.g. 0.020)
CMD_SPEED_MAX = 21     # m/s, upper edge of EE-speed sweep (e.g. 0.070)


class CommandFlag(IntEnum):
    START = 1 << 0        # begin executing the configured program
    STOP = 1 << 1         # abort current run, return to idle
    APPLY_GAINS = 1 << 2  # write CMD_K{P,I,D}_{FWD,REV} into tuned joint's PID(s)
    RELOAD_CONFIG = 1 << 3  # hardware.reload_config()
    RESET_PIDS = 1 << 4   # reset integral/derivative state on all joint PIDs
    AUTO_TUNE = 1 << 5    # if set during START, run auto-tune loop
    DIR_PID = 1 << 6      # when set, install DirectionalPID on tuned joint
    SYNC_FROM_HW = 1 << 7  # sync gains back from host to the client (host echoes)


# ---------------------------------------------------------------------------
# Telemetry packet layout — indices into the float32 array sent host -> client
# ---------------------------------------------------------------------------

TLM_STATE = 0               # TunerState as float
TLM_HW_READY = 1            # 0/1
TLM_DIR_PID = 2             # 0/1 — directional PID currently installed?
TLM_TUNED_JOINT = 3         # 0..3
TLM_RUN_INDEX = 4           # current auto-tune iteration index
TLM_STROKE_INDEX = 5        # current stroke within run
TLM_TOTAL_STROKES = 6       # strokes_per_run echo
TLM_EE_X = 7                # metres, measured EE position
TLM_EE_Y = 8
TLM_EE_Z = 9
TLM_CMD_X = 10              # metres, instantaneous ramp reference
TLM_CMD_Y = 11
TLM_CMD_Z = 12
TLM_RAMP_DIR = 13           # +1 / -1 / 0 (direction of x ramp)
TLM_CUR_SPEED = 14          # m/s
TLM_CUR_Z = 15              # m
TLM_CUR_X_MIN = 16
TLM_CUR_X_MAX = 17
TLM_RMSE_LAST = 18          # last run scoring stats — over completed run
TLM_MAXERR_LAST = 19
TLM_MEANERR_LAST = 20
TLM_BEST_COST = 21          # best auto-tune cost so far
TLM_LAST_COST = 22          # last completed run cost
TLM_ITER = 23               # auto-tune internal step counter
TLM_KP_FWD = 24             # gains currently installed on tuned joint
TLM_KI_FWD = 25
TLM_KD_FWD = 26
TLM_KP_REV = 27
TLM_KI_REV = 28
TLM_KD_REV = 29
TLM_NUM_RUNS = 30           # remaining auto-tune budget
TLM_ERR_NOW = 31            # instantaneous |ee_x - cmd_x|, for live plot annotations
# Per-direction breakdown of last-run stats. "fwd" = +X strokes (out),
# "rev" = -X strokes (in). The auto-tuner uses the matching subscore to
# score the gain dimension it is probing.
TLM_RMSE_LAST_FWD = 32
TLM_RMSE_LAST_REV = 33
TLM_MAXERR_LAST_FWD = 34
TLM_MAXERR_LAST_REV = 35
TLM_MEANERR_LAST_FWD = 36
TLM_MEANERR_LAST_REV = 37
TLM_LAST_COST_FWD = 38
TLM_LAST_COST_REV = 39
TLM_BEST_COST_FWD = 40
TLM_BEST_COST_REV = 41


class TunerState(IntEnum):
    IDLE = 0           # nothing running, accepting commands
    ARMED = 1          # parameters validated, waiting for first stroke
    RUNNING = 2        # in the middle of a manual or scored run
    SCORING = 3        # between strokes, computing stats
    AUTO_TUNING = 4    # auto-tune loop active (running + updating gains)
    HALTED = 5         # safety stop (hw fault / unreachable target)
    ERROR = 6          # transient error reported; client should display


# ---------------------------------------------------------------------------
# Sensible defaults that both sides reference for their initial UI / config
# ---------------------------------------------------------------------------

DEFAULT_HOST = "192.168.0.132"
DEFAULT_PORT = 8091   # one above existing per-joint pid_tuner_robot port

# Conservative placeholder ranges — user will overwrite from the GUI. Picked to
# be physically reasonable for a tabletop excavator EE in metres.
DEFAULT_X_MIN = 0.30
DEFAULT_X_MAX = 0.50
DEFAULT_Z = 0.20
DEFAULT_Y = 0.0
DEFAULT_SPEED = 0.040       # 40 mm/s — middle of 20..70 mm/s band
DEFAULT_STROKES_PER_RUN = 4
DEFAULT_NUM_RUNS = 20
# Auto-tune sweep defaults — values you'll likely overwrite from the GUI
# once a safe envelope is confirmed for the rig.
DEFAULT_Z_MIN = 0.18
DEFAULT_Z_MAX = 0.22
DEFAULT_SPEED_MIN = 0.020
DEFAULT_SPEED_MAX = 0.070

# Cost weights: RMSE dominates by default. Tuned for metres → multiply by 1000
# inside the host so the numeric range matches mm-scale errors and the auto-tune
# step heuristics work on familiar magnitudes.
DEFAULT_COST_WEIGHTS: Tuple[float, float, float] = (1.0, 0.5, 0.25)


# ---------------------------------------------------------------------------
# State-code labels for the client status line (host echoes via TLM_STATE).
# ---------------------------------------------------------------------------

STATE_LABELS = {
    int(TunerState.IDLE):        "idle",
    int(TunerState.ARMED):       "armed",
    int(TunerState.RUNNING):     "running",
    int(TunerState.SCORING):     "scoring",
    int(TunerState.AUTO_TUNING): "auto-tuning",
    int(TunerState.HALTED):      "halted",
    int(TunerState.ERROR):       "error",
}


def state_label(value: float) -> str:
    """Map a TLM_STATE float to a human-readable label."""
    return STATE_LABELS.get(int(round(float(value))), f"?{value:.0f}")
