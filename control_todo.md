# Control TODO

## Current Status

- Legacy service/control tests pass.
- `tests/test_vel_ctrl_stack.py` covers the current velocity-control helpers.
- `simple_drive.py` has integrated manual modes:
  - `OFF`
  - raw linkage-rate valve scaling
  - universal pump-gain shape
  - velocity PID
- `test_vel_ctrl.py` is an older prototype with useful ideas:
  - RAW / VEL / ASSIST modes
  - direction-specific gains
  - target velocity slew limiting
  - assist mode where PID trims raw joystick command

## Current Gaps

- `LinkageRateCompensator` and `UniversalShapeCompensator` do not have direct unit tests.
- The measured CSV tables load and the CLI preview works, but spline interpolation,
  clamp behavior, direction handling, and pump correction weighting are not tested.
- `simple_drive.py` does not yet combine linkage feed-forward with velocity PID trim.
- Velocity PID currently uses simple global gains in `simple_drive.py`; the older
  prototype has direction-specific gains that may be worth preserving.

## Preferred Control Direction

Use both deterministic feed-forward and feedback:

```text
desired manual/trajectory joint command
        -> measured linkage-rate feed-forward
        -> deadband / valve calibration
        -> small velocity PID trim
        -> PWM command
```

The measured spline/feed-forward layer should handle predictable geometry-driven
rate variation. Velocity PID should correct residual effects such as load,
temperature, pump pressure, model error, and sensor drift.

Avoid making velocity PID the only correction layer unless the spline data proves
too noisy or hard to maintain.

## Next Steps

1. Add direct tests for `tools/linkage_rate_compensation.py`.
   - table loading
   - interpolation
   - min/max clamp
   - low-command passthrough
   - missing joint/direction fallback
   - universal pump correction weighting

2. Add a combined control mode to `simple_drive.py`.
   - apply linkage feed-forward first
   - apply velocity PID as trim on top
   - reset/clamp integrator aggressively
   - zero controlled joints when velocity feedback is stale

3. Preserve useful ideas from `test_vel_ctrl.py`.
   - direction-specific PID gains
   - target velocity slew limiting
   - ASSIST-style trim mode

4. Log comparison runs.
   - raw command
   - spline-only
   - velocity-PID-only
   - spline + PID trim

5. Compare each mode by:
   - desired vs actual joint velocity
   - overshoot
   - lag
   - command smoothness
   - operator feel
   - safety/watchdog behavior

## Notes

- The Python control stack has already been tested up to 200 Hz with total jitter
  under 1 ms, so do not rewrite working control code for performance without a
  measured bottleneck.
- Keep the local controller authoritative; ROS should wrap and observe it before
  replacing any low-level behavior.
