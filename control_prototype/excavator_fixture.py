"""Real-robot test fixture for the smooth reachability prototype.

Hardcoded to match configuration_files/control_config.yaml as of 2026-05-13.
Kept hardcoded (instead of YAML-loading) so the prototype stays self-contained
and the tests are not sensitive to future config edits.

If you change control_config.yaml's joint_limits_relative or robot.links,
update REAL_JOINT_LIMITS_RAD / REAL_LINK_LENGTHS / REAL_EE_OFFSET below.

Source values:
    control_config.yaml lines 74-87  -> robot geometry
    control_config.yaml lines 202-210 -> joint_limits_relative (deg)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Joint limits  (None = unbounded; matches the YAML's null entry for slew)
# --------------------------------------------------------------------------
# control_config.yaml:
#   - null              # slew: unbounded -- continuous cab rotation
#   - [-50.0, 30.0]     # boom
#   - [ 28.0, 144.0]    # arm
#   - [-120.0, 20.0]    # bucket

REAL_JOINT_LIMITS_DEG: List[Optional[Tuple[float, float]]] = [
    None,
    (-50.0, 30.0),
    (28.0, 144.0),
    (-120.0, 20.0),
]

REAL_JOINT_LIMITS_RAD: List[Optional[Tuple[float, float]]] = [
    None if lim is None else (math.radians(lim[0]), math.radians(lim[1]))
    for lim in REAL_JOINT_LIMITS_DEG
]


def _midpoint_or_zero(lim: Optional[Tuple[float, float]]) -> float:
    if lim is None:
        return 0.0
    return 0.5 * (lim[0] + lim[1])


# Comfortable mid-workspace pose, well outside every slow zone.
SAFE_ANGLES_RAD = np.array(
    [math.radians(_midpoint_or_zero(lim)) for lim in REAL_JOINT_LIMITS_DEG],
    dtype=np.float32,
)


# --------------------------------------------------------------------------
# Link geometry  (robot.frame / robot.links / robot.tools.presets.<active>)
# --------------------------------------------------------------------------
# Slew "length" is the planar distance from the rotation axis to the boom
# mount; matches load_excavator_robot_config() in modules/differential_ik_cfg.py.

_BOOM_MOUNT_OFFSET = np.array([0.0165, 0.0, 0.0645], dtype=np.float32)
_SLEW_LENGTH = float(np.linalg.norm(_BOOM_MOUNT_OFFSET))

REAL_ORIGIN_HEIGHT_M = 0.07
REAL_LINK_LENGTHS: Tuple[float, float, float, float] = (
    _SLEW_LENGTH,  # ~0.06658 m
    0.468,         # boom
    0.250,         # arm
    0.031,         # coupler (bucket link)
)
# Active tool tip offset from the coupler (bucket points down by default).
REAL_EE_OFFSET = np.array([0.0, 0.0, -0.142], dtype=np.float32)
REAL_ORIGIN_OFFSET = np.array([0.0, 0.0, REAL_ORIGIN_HEIGHT_M], dtype=np.float32)


def load_real_robot_config():
    """Return a live RobotConfig loaded from the real YAML.

    Imported lazily so unit tests that don't need the full IK stack
    (and its numba compilation cost) avoid the import.
    """
    from modules.differential_ik_cfg import load_excavator_robot_config

    return load_excavator_robot_config()
