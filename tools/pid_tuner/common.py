"""Shared constants and helpers for the PID tuner package.

Used by both ``robot.py`` (robot/host side) and ``client.py`` (PC GUI) so the
two ends stay in sync without dragging the production stack into the package.

Wire format
-----------
All packets are flat ``<float32`` arrays via :class:`modules.udp_socket.UDPSocket`.

Per-joint protocol (port 8090)
    Command   (client → host): ``JOINT_COMMAND_SIZE``    floats — see ``JCMD_*``
    Telemetry (host  → client): ``JOINT_TELEMETRY_SIZE`` floats — see ``JTLM_*``

EE-plane protocol (port 8091)
    Command   (client → host): ``EE_COMMAND_SIZE``    floats — see ``CMD_*``
    Telemetry (host  → client): ``EE_TELEMETRY_SIZE`` floats — see ``TLM_*``

Keep layouts append-only: add new fields at the end and bump size constants in
lock-step on both sides.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Tuple


# ---------------------------------------------------------------------------
# Per-joint protocol
# ---------------------------------------------------------------------------

# TODO(refactor #1 — single-port pid_tuner):
#   Drop JOINT_PORT / EE_PORT. Introduce one UNIFIED_PORT (e.g. 8090) plus
#   a 1-byte MODE tag at index 0 of every command and telemetry packet:
#       MODE_JOINT = 0    # rest of packet is the JCMD_* / JTLM_* layout
#       MODE_EE    = 1    # rest of packet is the CMD_* / TLM_* layout
#   Both packet sizes already differ, so the server demuxes by mode and
#   parses with the matching schema. Shift all existing JCMD_* / JTLM_* /
#   CMD_* / TLM_* indices by +1 to make room for the mode byte.
JOINT_COMMAND_SIZE = 10
JOINT_TELEMETRY_SIZE = 12
JOINT_PORT = 8090

JOINT_NAMES = ["slew", "boom", "arm", "bucket"]
PWM_NAMES = ["slew", "boom", "arm", "bucket"]

# Command indices (client → host)
JCMD_JOINT_ID = 0
JCMD_TARGET_DEG = 1
JCMD_KP = 2
JCMD_KI = 3
JCMD_KD = 4
JCMD_ENABLED = 5
JCMD_PUMP = 6
JCMD_RELOAD = 7
JCMD_RESET_PID = 8
JCMD_MAX_OUTPUT = 9

# Telemetry indices (host → client)
JTLM_JOINT_ID = 0
JTLM_SLEW_DEG = 1
JTLM_BOOM_DEG = 2
JTLM_ARM_DEG = 3
JTLM_BUCKET_DEG = 4
JTLM_TARGET_DEG = 5
JTLM_OUTPUT = 6
JTLM_KP = 7
JTLM_KI = 8
JTLM_KD = 9
JTLM_ENABLED = 10
JTLM_PUMP = 11


# ---------------------------------------------------------------------------
# EE-plane protocol
# ---------------------------------------------------------------------------

EE_COMMAND_SIZE = 22
EE_TELEMETRY_SIZE = 42
EE_PORT = 8091

# Command packet layout — indices into the float32 array sent client → host
CMD_FLAGS = 0          # bit field, see EECommandFlag
CMD_X_MIN = 1          # metres, EE x sweep low end
CMD_X_MAX = 2          # metres, EE x sweep high end
CMD_Z = 3              # metres, fixed EE z height
CMD_Y = 4              # metres, fixed EE y (lateral); usually 0
CMD_SPEED = 5          # m/s, target linear EE speed (0.020..0.070)
CMD_STROKES_PER_RUN = 6
CMD_NUM_RUNS = 7
CMD_TUNED_JOINT = 8    # int 0..3, display/manual-push joint
CMD_JOINT_MASK = 9     # bit field 0..3, which joints are owned by the tuner
CMD_KP_FWD = 10
CMD_KI_FWD = 11
CMD_KD_FWD = 12
CMD_KP_REV = 13
CMD_KI_REV = 14
CMD_KD_REV = 15
CMD_COST_W_RMSE = 16
CMD_COST_W_MAX = 17
CMD_Z_MIN = 18
CMD_Z_MAX = 19
CMD_SPEED_MIN = 20
CMD_SPEED_MAX = 21


class EECommandFlag(IntEnum):
    START = 1 << 0
    STOP = 1 << 1
    APPLY_GAINS = 1 << 2
    RELOAD_CONFIG = 1 << 3
    RESET_PIDS = 1 << 4
    AUTO_TUNE = 1 << 5
    DIR_PID = 1 << 6
    SYNC_FROM_HW = 1 << 7


# Telemetry packet layout — indices into the float32 array sent host → client
TLM_STATE = 0
TLM_HW_READY = 1
TLM_DIR_PID = 2
TLM_TUNED_JOINT = 3
TLM_RUN_INDEX = 4
TLM_STROKE_INDEX = 5
TLM_TOTAL_STROKES = 6
TLM_EE_X = 7
TLM_EE_Y = 8
TLM_EE_Z = 9
TLM_CMD_X = 10
TLM_CMD_Y = 11
TLM_CMD_Z = 12
TLM_RAMP_DIR = 13
TLM_CUR_SPEED = 14
TLM_CUR_Z = 15
TLM_CUR_X_MIN = 16
TLM_CUR_X_MAX = 17
TLM_RMSE_LAST = 18
TLM_MAXERR_LAST = 19
TLM_MEANERR_LAST = 20
TLM_BEST_COST = 21
TLM_LAST_COST = 22
TLM_ITER = 23
TLM_KP_FWD = 24
TLM_KI_FWD = 25
TLM_KD_FWD = 26
TLM_KP_REV = 27
TLM_KI_REV = 28
TLM_KD_REV = 29
TLM_NUM_RUNS = 30
TLM_ERR_NOW = 31
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


class EETunerState(IntEnum):
    IDLE = 0
    ARMED = 1
    RUNNING = 2
    SCORING = 3
    AUTO_TUNING = 4
    HALTED = 5
    ERROR = 6


_STATE_LABELS = {
    int(EETunerState.IDLE):        "idle",
    int(EETunerState.ARMED):       "armed",
    int(EETunerState.RUNNING):     "running",
    int(EETunerState.SCORING):     "scoring",
    int(EETunerState.AUTO_TUNING): "auto-tuning",
    int(EETunerState.HALTED):      "halted",
    int(EETunerState.ERROR):       "error",
}


def ee_state_label(value: float) -> str:
    return _STATE_LABELS.get(int(round(float(value))), f"?{value:.0f}")


# ---------------------------------------------------------------------------
# Defaults shared by both sides
# ---------------------------------------------------------------------------

DEFAULT_HOST = "192.168.0.132"

DEFAULT_X_MIN = 0.30
DEFAULT_X_MAX = 0.50
DEFAULT_Z = 0.20
DEFAULT_Y = 0.0
DEFAULT_SPEED = 0.040
DEFAULT_STROKES_PER_RUN = 4
DEFAULT_NUM_RUNS = 20
DEFAULT_Z_MIN = 0.18
DEFAULT_Z_MAX = 0.22
DEFAULT_SPEED_MIN = 0.020
DEFAULT_SPEED_MAX = 0.070

DEFAULT_COST_WEIGHTS: Tuple[float, float, float] = (1.0, 0.5, 0.25)

# Joints tuned together for EE cartesian tracking.
# Slew (0) excluded — no effect on +X/-X when facing forward.
EE_TUNABLE_JOINTS: Tuple[int, ...] = (1, 2, 3)  # boom, arm, bucket
EE_TUNABLE_NAMES: Tuple[str, ...] = tuple(JOINT_NAMES[i] for i in EE_TUNABLE_JOINTS)


# ---------------------------------------------------------------------------
# Backward-compat aliases (old module-level names used by existing code/tests)
# ---------------------------------------------------------------------------

# EE command/telemetry sizes under their old names
COMMAND_SIZE = EE_COMMAND_SIZE
TELEMETRY_SIZE = EE_TELEMETRY_SIZE

# Old CommandFlag / TunerState names
CommandFlag = EECommandFlag
TunerState = EETunerState
state_label = ee_state_label

DEFAULT_PORT = EE_PORT
