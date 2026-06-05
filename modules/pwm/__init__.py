"""PWM control package."""

from .config import ChannelConfig, PumpConfig
from .constants import CANONICAL_CHANNEL_NAMES, PCA9685_I2C_BUS, PCA9685_I2C_ADDRESS, PWMConstants
from .controller import PWMController
from .errors import (
    PWMControllerError,
    PWMConfigError,
    PWMConfigLoadError,
    PWMConfigValidationError,
    PWMHardwareIOError,
)
from .writer import DirectPWMWriter, _open_smbus

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
