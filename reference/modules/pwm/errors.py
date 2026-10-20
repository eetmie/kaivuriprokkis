"""PWM controller exception hierarchy."""


class PWMControllerError(Exception):
    """Base exception for PWM controller failures."""


class PWMConfigError(PWMControllerError):
    """Base exception for PWM configuration failures."""


class PWMConfigLoadError(PWMConfigError):
    """Raised when a PWM config file cannot be loaded or parsed."""


class PWMConfigValidationError(PWMConfigError):
    """Raised when a PWM config is syntactically valid but invalid semantically."""


class PWMHardwareIOError(PWMControllerError):
    """Raised when a low-level PWM hardware operation fails."""
