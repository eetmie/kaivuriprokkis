# Velocity-Assisted Raw Control Note

Idea for later: do not make closed-loop velocity control a separate replacement for raw valve control. Instead, treat velocity feedback as a small trim layer on top of raw/driver-intent control.

Current test split:

```text
RAW: joystick -> valve command
VEL: joystick -> target deg/s -> PI/PID -> valve command
```

Possible later direction:

```text
joystick -> direction-based valve feed-forward -> velocity PID trim -> valve command
```

Example shape:

```python
target_dps = joystick_cmd * max_degps
base_cmd = ff_gain[joint][direction] * joystick_cmd
trim_cmd = pid.compute(target_dps, actual_dps, dt)
trim_cmd = clip(trim_cmd, -max_trim, max_trim)
valve_cmd = clip(base_cmd + trim_cmd, -1.0, 1.0)
```

Rationale:

- The miniature hydraulic valve has real stiction/no-motion behavior that strict velocity PID fights poorly.
- Raw valve control already captures useful driver intent and feel.
- Feed-forward can handle most valve opening, rod/cap-side asymmetry, gravity bias, and joint-specific behavior.
- Velocity feedback should ideally only correct residual error, not be responsible for opening the valve from zero.
- A trim clamp such as `max_trim = 0.15..0.25` would keep feedback from fully overriding the driver/feed-forward command.

Layering preference:

- Keep `PCA9685_controller.py` as physical output calibration only: center, pulse limits, deadband, gamma, dither, ramp.
- Keep velocity assistance in the higher-level control script/controller: target velocity, gyro feedback, gravity/side bias, feed-forward, PID trim.

Bench result from `test_vel_ctrl.py`:

- `RAW`, `VEL`, and `ASSIST` modes are useful as a demo/tuning split.
- The important stabilizing trick was not integral tuning, but preventing active velocity control from crossing to the opposite valve direction while the joystick is held one way.
- The miniature valve has enough deadband/stiction that PI briefly backing through zero/opposite sign can drop the valve back into the no-motion region, then the delayed breakout causes oscillation.
- A temporary `ACTIVE_COMMAND_FLOOR = 0.05` worked well, but it is effectively static feed-forward on every nonzero command.

Cleaner control rule to test:

```text
zero joystick -> zero command
nonzero joystick -> remember/use joystick direction
PID/final command may only live on that valve side
opposite-side PID/final command is gated to zero, not flipped and not floored
```

This keeps the hydraulic-specific behavior in the velocity/raw controller, not in `PCA9685_controller.py`.
