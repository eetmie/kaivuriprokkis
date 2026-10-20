"""PWM constants and canonical channel names."""

CANONICAL_CHANNEL_NAMES = ("slew", "boom", "arm", "bucket", "trackL", "trackR")

# Legacy Raspberry Pi bus number. Kept only so existing imports of
# PCA9685_I2C_BUS keep resolving; it is NOT a safe cross-board default -- the
# Jetson Orin Nano has the PCA9685 on bus 7, and bus 1 there is the kernel-owned
# ina3221 power monitor. Pass bus=None (the default) and let resolve_i2c_bus()
# read the active board profile instead of relying on this value.
PCA9685_I2C_BUS = 1
PCA9685_I2C_ADDRESS = 0x40


def resolve_i2c_bus(bus: int | None = None) -> int:
    """Return the PCA9685 I2C bus number for this board.

    ``bus=None`` means "ask the active board profile", which is what every
    caller should do unless it has an explicit override (CLI flag, ROS param).
    An explicit value is passed through untouched.

    The ``modules.board`` import is deliberately local: it keeps ``modules.pwm``
    free of an import-time dependency on the profile/YAML layer.
    """
    if bus is not None:
        return int(bus)
    from ..board import resolve_profile

    return int(resolve_profile()["pwm_i2c_bus"])


class PWMConstants:
    """Hardware and timing constants."""

    PWM_FREQUENCY_DEFAULT = 50
    MAX_CHANNELS = 16
    PCA9685_COUNT_MAX = 4095
    RAMP_DT_MAX = 0.05

    PULSE_MIN = 0
    PULSE_MAX = 4095
    PWM_FREQ_MIN = 30
    PWM_FREQ_MAX = 1000
    NORMALIZED_COMMAND_MIN = -1.0
    NORMALIZED_COMMAND_MAX = 1.0

    DEFAULT_TIME_WINDOW = 0.5
