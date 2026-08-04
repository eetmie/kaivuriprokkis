"""Compatibility facade for the PWM controller package.

New code should import from ``modules.pwm``. This module remains so existing
scripts and tests that import ``modules.PCA9685_controller`` keep working.
"""

from __future__ import annotations

from .pwm import (
    CANONICAL_CHANNEL_NAMES,
    PCA9685_I2C_BUS,
    PCA9685_I2C_ADDRESS,
    PWMControllerError,
    PWMConfigError,
    PWMConfigLoadError,
    PWMConfigValidationError,
    PWMHardwareIOError,
    DirectPWMWriter,
    ChannelConfig,
    PumpConfig,
    PWMConstants,
)
from .pwm import controller as _controller
from .pwm import writer as _writer


def _open_smbus(bus_number: int | None = None):
    """Legacy monkeypatch target -- delegates to the one real implementation.

    Kept as a module-level name because existing tests patch
    ``modules.PCA9685_controller._open_smbus``; see PWMController below.
    ``bus_number=None`` resolves from the active board profile.
    """
    return _writer._open_smbus(bus_number)


class PWMController(_controller.PWMController):
    """Compatibility subclass preserving the old monkeypatch target.

    Older tests patch ``modules.PCA9685_controller._open_smbus``. The real
    implementation now lives in ``modules.pwm.controller``, so bridge that patch
    target during construction.
    """

    def __init__(self, *args, **kwargs):
        original_open_smbus = _controller._open_smbus
        _controller._open_smbus = _open_smbus
        try:
            super().__init__(*args, **kwargs)
        finally:
            _controller._open_smbus = original_open_smbus


__all__ = [
    "CANONICAL_CHANNEL_NAMES",
    "PCA9685_I2C_BUS",
    "PCA9685_I2C_ADDRESS",
    "PWMControllerError",
    "PWMConfigError",
    "PWMConfigLoadError",
    "PWMConfigValidationError",
    "PWMHardwareIOError",
    "DirectPWMWriter",
    "ChannelConfig",
    "PumpConfig",
    "PWMConstants",
    "PWMController",
    "_open_smbus",
]
