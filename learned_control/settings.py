"""Controller timing and command bounds shared with training, plus small helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

JOINT_NAMES = ["revolute_lift", "revolute_tilt", "revolute_tool", "revolute_carriage_pitch"]
HOME = [-0.5498, 1.2549, -0.7540, 0.0]


@dataclass
class ControllerSettings:
    """Shared timing and command bounds, in seconds, meters and radians."""

    dt: float = 0.01
    decimation: int = 5
    speed_max: float = 0.12
    pitch_rate_max: float = 0.3
    joint_margin: float = 0.06
    lookahead: float = 0.8
    velocity_limit: float = 2.0

    def __post_init__(self):
        if self.dt != 0.01:
            raise ValueError("This controller contract requires 100 Hz hydraulics")
        if isinstance(self.decimation, bool) or not isinstance(self.decimation, int) or self.decimation < 1:
            raise ValueError("Policy decimation must be a positive integer")
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Controller settings must be finite and positive")

    @property
    def policy_dt(self) -> float:
        """Policy period [s]."""
        return self.dt * self.decimation

    @property
    def policy_hz(self) -> float:
        """Policy update rate [Hz]."""
        return 1.0 / self.policy_dt


def sha256(path: Path) -> str:
    """Fingerprint a model or geometry file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap_angle(value: torch.Tensor) -> torch.Tensor:
    """Wrap angles [rad] to [-pi, pi]."""
    return torch.atan2(torch.sin(value), torch.cos(value))
