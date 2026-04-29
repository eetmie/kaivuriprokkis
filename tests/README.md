# Excavator Test Suite

Unit and integration tests for the excavator IK/FK/radial control stack, plus
hardware stress scripts for the 200 Hz real-time path.

## Running the unit suite

From the repository root:

```bash
python -m unittest discover tests -v
```

The full suite completes in under 20 seconds (Numba JIT compilation is the
main cost on a cold run).

### Focused runs

```bash
# FK / Jacobian math
python -m unittest tests.test_kinematics -v

# IK solver methods (pinv / svd / trans / dls) + features
python -m unittest tests.test_ik_methods -v

# ExcavatorTargetState (radial ↔ cartesian)
python -m unittest tests.test_radial -v

# Radial delta → IK convergence end-to-end
python -m unittest tests.test_radial_ik_integration -v

# RobotService + control protocol round-trips
python -m unittest tests.test_robot_service_flow -v

# Real PCA9685 smoke test (auto-skipped if no I2C)
python -m unittest tests.test_hardware_smoke -v
```

## What is covered

| Area | File | Highlights |
| --- | --- | --- |
| Forward kinematics | `test_kinematics.py` | zero-pose geometry, pure-slew circular motion, joint-position walk, relative-angle decomposition, `propagate_base_rotation`, Jacobian vs numerical diff at slew=0, condition number sanity |
| IK solvers | `test_ik_methods.py` | convergence for `pinv`, `svd`, `trans`, `dls`; per-cycle velocity clamp; joint-limit repulsion; `enable_frame_transform` toggle; `ignore_axes=[yaw]`; adaptive damping |
| Radial target state | `test_radial.py` | cartesian↔radial round-trips, clamp re-sync, slew wrapping, negative-radius clamp, incremental accumulation, keyboard + gamepad mapping |
| Radial + IK | `test_radial_ik_integration.py` | small slew sweeps keep radius constant, pure radius extension/retraction, pure-z radial changes, interleaved cartesian + radial deltas, large accumulated slew |
| Service flow | `test_robot_service_flow.py` | mode switches, paused gating, pump toggle, reload flag, command-packet round-trip with radial-composed poses |
| Hardware smoke | `test_hardware_smoke.py` | real PCA9685 init, zero-PWM command path, pump enable/disable, reset with pump stop. Auto-skipped if no PCA9685 on I2C bus 1 |

`common_kinematics.py` holds shared helpers (quaternion chain, numerical
Jacobian, end-to-end IK simulator) so tests don't duplicate setup.

## Stress tests (hardware integration)

These scripts exercise the real control loop at 200 Hz and are kept separate
from the unit suite.

### Full Stack Zero-Command Stress

Runs real hardware + controller + service + UDP protocol + telemetry. The
controller is forced into direct mode and still sends explicit zero-valued
named PWM commands every cycle.

```bash
sudo python tests/stress_full_stack.py --rate-hz 200 --duration-s 60
```

Useful RT options:

```bash
sudo python tests/stress_full_stack.py --rate-hz 200 --duration-s 60 \
    --fifo-priority 75 --lock-memory --control-core 2 --io-core 3
```

### USB Reader Stress

Runs the Pico USB IMU reader by itself to measure the practical host-side
ceiling.

```bash
sudo python tests/stress_usb_reader.py --target-hz 200 --duration-s 30
```

```bash
sudo python tests/stress_usb_reader.py --target-hz 200 --duration-s 30 \
    --fifo-priority 75 --lock-memory --cpu-core 3
```
