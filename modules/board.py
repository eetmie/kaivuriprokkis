"""Board detection and robot profile resolution.

Profiles live under ``configuration_files/profiles/<name>/profile.yaml``.
The profile name is the deployable robot/config set. The ``board`` field inside
the profile selects compute-platform defaults such as I2C bus numbers.

Usage::

    from modules.board import resolve_profile

    profile = resolve_profile()          # auto-detect board profile only
    profile = resolve_profile("jetson")  # explicit profile
    profile = resolve_profile("rpi")

Profile keys
------------
profile_name       : selected profile folder name
board              : compute platform, e.g. rpi or jetson
profile_dir        : profile folder path relative to repo root
servo_config_file  : resolved path to servo/PWM YAML, relative to repo root
control_config_file: resolved path to controller/IK/IMU YAML, relative to repo root
pwm_i2c_bus        : Linux I2C bus number for PCA9685
pwm_i2c_addr       : PCA9685 I2C address
enable_imu         : whether IMU startup is on by default
enable_adc         : whether pressure ADC startup is on by default
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
PROFILES_DIR = ROOT_DIR / "configuration_files" / "profiles"

BOARD_DEFAULTS: dict[str, dict[str, Any]] = {
    "rpi": {
        "pwm_i2c_bus": 1,
        "pwm_i2c_addr": 0x40,
        "enable_imu": True,
        "enable_adc": False,
    },
    "jetson": {
        "pwm_i2c_bus": 7,
        "pwm_i2c_addr": 0x40,
        "enable_imu": True,
        "enable_adc": False,
    },
}


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(resolved)


def _discover_profiles() -> dict[str, dict[str, str]]:
    profiles: dict[str, dict[str, str]] = {}
    if not PROFILES_DIR.exists():
        return profiles
    for profile_dir in sorted(PROFILES_DIR.iterdir()):
        profile_file = profile_dir / "profile.yaml"
        if profile_dir.is_dir() and profile_file.exists():
            profiles[profile_dir.name] = {"profile_file": _repo_relative(profile_file)}
    return profiles


# Backward-compatible profile-name registry used by CLI choices.
PROFILES: dict[str, dict[str, str]] = _discover_profiles()


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


def _load_profile_yaml(profile_name: str) -> tuple[dict[str, Any], Path]:
    profile_dir = PROFILES_DIR / profile_name
    profile_file = profile_dir / "profile.yaml"
    if not profile_file.exists():
        available = sorted(_discover_profiles())
        raise ValueError(f"Unknown profile {profile_name!r}. Choose from: {available}")
    try:
        with profile_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse profile '{profile_file}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Profile '{profile_file}' must contain a top-level mapping")
    return data, profile_dir


def _resolve_profile_path(profile_dir: Path, value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string in profile.yaml")
    path = Path(value)
    if not path.is_absolute():
        path = profile_dir / path
    if not path.exists():
        raise ValueError(f"{field_name} path from profile.yaml does not exist: {path}")
    return _repo_relative(path)


def resolve_profile(name: str = "auto") -> dict[str, Any]:
    """Return a copy of the named profile, auto-detecting board profile for 'auto'.

    ``auto`` intentionally detects only the board and resolves the matching
    board-named profile (``rpi`` or ``jetson``). Robot-specific profiles such as
    ``robot13`` must be selected explicitly.
    """
    profile_name = detect() if name == "auto" else name
    raw_profile, profile_dir = _load_profile_yaml(profile_name)

    board = str(raw_profile.get("board", "")).strip().lower()
    if board not in BOARD_DEFAULTS:
        raise ValueError(
            f"Profile {profile_name!r} has unknown board {board!r}. "
            f"Choose from: {sorted(BOARD_DEFAULTS)}"
        )

    profile = dict(BOARD_DEFAULTS[board])
    for key in ("pwm_i2c_bus", "pwm_i2c_addr", "enable_imu", "enable_adc"):
        if key in raw_profile:
            profile[key] = raw_profile[key]

    profile["profile_name"] = profile_name
    profile["name"] = str(raw_profile.get("name", profile_name))
    profile["board"] = board
    profile["profile_dir"] = _repo_relative(profile_dir)
    profile["servo_config_file"] = _resolve_profile_path(
        profile_dir,
        raw_profile.get("servo_config_file"),
        "servo_config_file",
    )
    profile["control_config_file"] = _resolve_profile_path(
        profile_dir,
        raw_profile.get("control_config_file"),
        "control_config_file",
    )
    # Backward-compatible alias while scripts migrate to the clearer name.
    profile["config_file"] = profile["servo_config_file"]
    profile["_resolved_board"] = board
    return profile
