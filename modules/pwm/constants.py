"""PWM constants and canonical channel names."""

CANONICAL_CHANNEL_NAMES = ("slew", "boom", "arm", "bucket", "trackL", "trackR")

PCA9685_I2C_BUS = 1
PCA9685_I2C_ADDRESS = 0x40


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
