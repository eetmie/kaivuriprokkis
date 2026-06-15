"""PWM configuration dataclasses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChannelConfig:
    """Configuration for a single PWM channel."""

    output_channel: int
    pulse_min: int
    pulse_max: int
    direction: int
    center: Optional[float] = None
    deadzone: float = 0.0
    affects_pump: bool = False
    toggleable: bool = False
    deadband_us_pos: float = 0.0
    deadband_us_neg: float = 0.0
    dither_enable: bool = False
    dither_amp_us: float = 0.0
    dither_hz: float = 0.0
    dither_taper: bool = False
    dither_taper_us: float = 0.0
    ramp_enable: bool = False
    ramp_limit: float = 0.0
    ramp_skip_deadband: bool = False
    gamma: float = 1.0

    def __post_init__(self):
        self.pulse_range = self.pulse_max - self.pulse_min
        if self.center is None:
            self.center = self.pulse_min + (self.pulse_range / 2)
        self.deadzone_threshold = self.deadzone / 100.0
        self._dither_omega = 2.0 * math.pi * float(self.dither_hz)
        self._dither_phase_offset = self.output_channel * 1.0471975512


@dataclass
class PumpConfig:
    """Configuration specific to pump control."""

    output_channel: int
    pulse_min: int
    pulse_max: int
    static_pulse_us: float
    base_pulse_us: float
    activity_gain_us: float
