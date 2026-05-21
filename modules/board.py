"""Board detection and robot profile resolution.

Centralised so every top-level script gets the same defaults without copy-paste.

Usage::

    from modules.board import resolve_profile

    profile = resolve_profile()          # auto-detect
    profile = resolve_profile("jetson")  # explicit
    profile = resolve_profile("rpi")

Profile keys
------------
config_file   : path to servo YAML (relative to repo root)
pwm_i2c_bus   : Linux I2C bus number for PCA9685
pwm_i2c_addr  : PCA9685 I2C address (default 0x40 on all boards)
enable_imu    : whether IMU startup is on by default
"""

from __future__ import annotations

from pathlib import Path

PROFILES: dict[str, dict] = {
    "rpi": {
        "config_file": "configuration_files/servo_config_200.yaml",
        "pwm_i2c_bus": 1,
        "pwm_i2c_addr": 0x40,
        "enable_imu": True,
    },
    "jetson": {
        "config_file": "configuration_files/servo_config_jetson.yaml",
        "pwm_i2c_bus": 7,
        "pwm_i2c_addr": 0x40,
        "enable_imu": True,
    },
}


def detect() -> str:
    """Return 'jetson' or 'rpi' based on hostname / device-tree model string."""
    import platform

    hostname = platform.uname()[1].lower()
    if "jetson" in hostname:
        return "jetson"
    if "raspberry" in hostname or hostname == "raspberrypi":
        return "rpi"
    try:
        model = Path("/proc/device-tree/model").read_text(errors="replace").lower()
        if "jetson" in model:
            return "jetson"
        if "raspberry" in model:
            return "rpi"
    except OSError:
        pass
    return "rpi"


def resolve_profile(name: str = "auto") -> dict:
    """Return a copy of the named profile, auto-detecting when name is 'auto'.

    The returned dict is a shallow copy — callers may mutate it freely.
    A ``_resolved_board`` key records which profile was actually used.
    """
    board = detect() if name == "auto" else name
    if board not in PROFILES:
        raise ValueError(f"Unknown board {board!r}. Choose from: {sorted(PROFILES)}")
    profile = dict(PROFILES[board])
    profile["_resolved_board"] = board
    return profile
