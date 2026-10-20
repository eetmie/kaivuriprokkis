"""PWM configuration validation helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional


def collect_config_validation_errors(
    channel_configs: Dict[str, Any],
    pump_config: Optional[Any],
    pwm_frequency: Any,
    constants: Any,
) -> list[str]:
    """Return semantic validation errors for parsed PWM config objects."""
    errors: list[str] = []
    used_outputs: dict[int, str] = {}

    if pwm_frequency is None:
        errors.append("pwm_frequency: missing from servo config (required)")
    elif isinstance(pwm_frequency, bool) or not isinstance(pwm_frequency, (int, float)):
        errors.append(f"pwm_frequency: must be a number (got {type(pwm_frequency).__name__})")
    else:
        freq = int(pwm_frequency)
        if not constants.PWM_FREQ_MIN <= freq <= constants.PWM_FREQ_MAX:
            errors.append(
                f"pwm_frequency: {freq} Hz out of range "
                f"({constants.PWM_FREQ_MIN}-{constants.PWM_FREQ_MAX} Hz)"
            )

    if not channel_configs and pump_config is None:
        errors.append("CHANNEL_CONFIGS: must define at least one channel or a pump")

    for name, config in channel_configs.items():
        if config.direction not in [-1, 1]:
            errors.append(f"Channel '{name}': direction must be -1 or 1")

        if config.output_channel in used_outputs:
            errors.append(f"Channel '{name}': output {config.output_channel} already used")
        elif not 0 <= config.output_channel < constants.MAX_CHANNELS:
            errors.append(f"Channel '{name}': output must be 0-{constants.MAX_CHANNELS - 1}")
        else:
            used_outputs[config.output_channel] = name

        if not constants.PULSE_MIN <= config.pulse_min <= constants.PULSE_MAX:
            errors.append(f"Channel '{name}': pulse_min out of range")
        if not constants.PULSE_MIN <= config.pulse_max <= constants.PULSE_MAX:
            errors.append(f"Channel '{name}': pulse_max out of range")
        if config.pulse_min >= config.pulse_max:
            errors.append(f"Channel '{name}': pulse_min must be less than pulse_max")

        if config.center is not None and not (config.pulse_min <= float(config.center) <= config.pulse_max):
            errors.append(f"Channel '{name}': center must be within [pulse_min, pulse_max]")

        rng = config.pulse_max - config.pulse_min
        if float(config.deadband_us_pos) < 0.0 or float(config.deadband_us_pos) > (rng * 0.5):
            errors.append(f"Channel '{name}': deadband_us_pos is unrealistic (0 .. {rng*0.5:.1f}us)")
        if float(config.deadband_us_neg) < 0.0 or float(config.deadband_us_neg) > (rng * 0.5):
            errors.append(f"Channel '{name}': deadband_us_neg is unrealistic (0 .. {rng*0.5:.1f}us)")
        if float(config.dither_amp_us) < 0.0 or float(config.dither_amp_us) > (rng * 0.25):
            errors.append(f"Channel '{name}': dither_amp_us is unrealistic (0 .. {rng*0.25:.1f}us)")
        if float(config.dither_hz) < 0.0 or float(config.dither_hz) > 200.0:
            errors.append(f"Channel '{name}': dither_hz must be within [0, 200]")
        if config.dither_enable:
            if float(config.dither_amp_us) <= 0.0:
                errors.append(f"Channel '{name}': dither_amp_us must be > 0 when dither_enable is true")
            if float(config.dither_hz) <= 0.0:
                errors.append(f"Channel '{name}': dither_hz must be > 0 when dither_enable is true")
            if config.dither_taper and float(config.dither_taper_us) <= 0.0:
                errors.append(f"Channel '{name}': dither_taper_us must be > 0 when dither_taper is true")
        if config.ramp_enable and float(config.ramp_limit) <= 0.0:
            errors.append(f"Channel '{name}': ramp_limit must be > 0 when ramp_enable is true")
        if float(config.gamma) <= 0.0 or float(config.gamma) > 5.0:
            errors.append(f"Channel '{name}': gamma must be within (0, 5]")

    if pump_config:
        if pump_config.output_channel in used_outputs:
            errors.append(f"Pump: output {pump_config.output_channel} already used")
        elif not 0 <= pump_config.output_channel < constants.MAX_CHANNELS:
            errors.append(f"Pump: output must be 0-{constants.MAX_CHANNELS - 1}")

        if not constants.PULSE_MIN <= pump_config.pulse_min <= constants.PULSE_MAX:
            errors.append("Pump: pulse_min out of range")
        if not constants.PULSE_MIN <= pump_config.pulse_max <= constants.PULSE_MAX:
            errors.append("Pump: pulse_max out of range")
        if pump_config.pulse_min >= pump_config.pulse_max:
            errors.append("Pump: pulse_min must be less than pulse_max")

        if not (pump_config.pulse_min <= pump_config.static_pulse_us <= pump_config.pulse_max):
            errors.append("Pump: static_pulse_us must be within [pulse_min, pulse_max]")
        if not (pump_config.pulse_min <= pump_config.base_pulse_us <= pump_config.pulse_max):
            errors.append("Pump: base_pulse_us must be within [pulse_min, pulse_max]")
        if not 0.0 <= pump_config.activity_gain_us <= (pump_config.pulse_max - pump_config.pulse_min):
            errors.append("Pump: activity_gain_us must be within [0, pulse_max - pulse_min]")

    return errors
