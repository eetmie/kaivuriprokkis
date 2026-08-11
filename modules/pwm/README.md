# PWM Module

This package owns PCA9685 PWM output, valve pulse mapping, pump control, and PWM-side safety checks.

## Layer Map

```text
HardwareInterface
    -> PWMController
        -> DirectPWMWriter
            -> PCA9685 over I2C
```

## Files

- `config.py`: dataclasses for parsed channel and pump config.
- `constants.py`: canonical channel names, PCA9685 bus/address defaults, validation limits.
- `controller.py`: high-level PWM control, safety watchdogs, normalized mapping, direct-us control, pump logic.
- `writer.py`: low-level PCA9685 I2C writer using native 12-bit counts.
- `validation.py`: semantic config validation after YAML parsing.
- `errors.py`: PWM-specific exception types.

## Command Modes

- Normalized valve control: `PWMController.update_named({"boom": 0.25})`
- Direct pulse-width control: `PWMController.update_named_us({"boom": 1800.0})`
- Explicit direct pulse alias: `PWMController.set_named_pulse_us({"boom": 1800.0})`
- Pump direct speed: `PWMController.set_pump_speed_us(1550.0)`
- Pulse readback: `PWMController.get_current_pulses_us()` returns `{name: pulse_us}`

Normalized commands apply deadzone, gamma, deadband mapping, ramp, and dither. Direct-us commands bypass those modifiers but still use controller safety checks and config pulse clamps.

## Call Rate Is Part of the Contract

`PWMController` has no clock of its own. Dither phase, ramp integration, the stale-command watchdog and the input-rate gate all advance **only when `update_named()` is called**, so the output is a function of the caller's rate, not just the command values:

- Dither is sampled at the call rate. The Jetson config runs `dither_hz: 33`; calling at 100 Hz gives ~3 samples per cycle, while calling at 30 Hz aliases it to a ~3 Hz wobble worth roughly 13% of boom's command authority.
- Falling below `input_rate_threshold` sets `is_safe_state = False`, after which `update_named()` returns early and **silently discards every command** until the measured rate recovers.
- `stale_timeout_s` centers all channels if calls stop for that long.

So a slow or bursty producer (a VLA policy, a recording loop that also encodes video) must not drive this class directly. Hand setpoints to `ExcavatorController.enter_direct_command_mode()` instead: its control thread resamples them onto a fixed `control_hz` and calls `update_named()` every tick. See `modules/setpoint_schedule.py`.

## Safety Ownership

Use `PWMController` or `HardwareInterface` for normal control. These paths keep stale-command handling, rate checks, pulse clamping, pump state, and reset behavior active.

Do not call `DirectPWMWriter` directly from control loops unless intentionally bypassing safety. It only writes PCA9685 counts and does not know channel names, pulse limits, pump logic, or watchdog state.

## Reset Behavior

`PWMController.reset(reset_pump=True)` sends valve channels to configured `center`, clears direct pump override, resets ramp state, and sends pump to `pulse_min` when `reset_pump` is true.

## Units

- Public valve command values: normalized `[-1.0, 1.0]`
- Public direct pulse values: microseconds
- Low-level writer values: PCA9685 native 12-bit counts `0..4095`
