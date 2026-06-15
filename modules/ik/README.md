# IK Module

This package owns the excavator's URDF-style kinematic model, FK/Jacobian
computation, and a pure-functional differential IK step. It also hosts the
IMU-to-canonical-joint-angle glue used by the controller.

## Layer Map

```text
ExcavatorController / robot_service / reachability
    -> modules.ik.{model, kinematics, solver, excavator}
        -> joint targets, FK state, Jacobian, IK result
```

PWM output is not owned here. IK produces joint-space results that the
controller turns into PID setpoints and PWM commands through the hardware
stack.

## Files

- `model.py`: URDF-style `Joint`, `Tool`, `ExcavatorModel`, and
  `load_excavator_model()` YAML loader.
- `math.py`: quaternion and vector primitives plus axis-rotation helpers.
  Quaternion convention is `[w, x, y, z]`.
- `kinematics.py`: numba FK/Jacobian cores and the single rich
  `RobotKinematicState` returned by `get_state(q_rad, model)`.
- `solver.py`: `IKConfig`, `IKResult`, `solve_ik_step(...)` — pure-functional
  differential IK plus the numba `ik_method_*` kernels.
- `excavator.py`: `IMUConfig`, `joint_angles_from_imus(...)`, numba warmup.
- `__init__.py`: public re-exports.

## Data Flow

```text
Corrected IMU quaternions
    -> joint_angles_from_imus(...)              # q_rad
    -> get_state(q_rad, model)                  # one rich RobotKinematicState
    -> solve_ik_step(q_rad, target_pos, target_quat, model, ik_cfg, dt, state=...)
    -> IKResult.q_next_rad                      # next joint targets
```

Canonical joint angles in radians are the source of truth across the
stack. Absolute link quaternions live inside `RobotKinematicState` for the
code that needs them (slew-yaw extraction, debug visualisation) and are
not passed around independently.

## Main Entry Points

- `load_excavator_model(path)`: build an `ExcavatorModel` from
  `control_config.yaml` (URDF-style `robot.joints[]` + `robot.tool`).
- `load_imu_config(path)`: build an `IMUConfig` from the `imu:` section.
- `get_state(q_rad, model, include_jacobian=True)`: one FK + Jacobian pass
  returning joint origins/axes in world frame, link orientations, EE pose,
  Jacobian, condition number, singular values, and Yoshikawa index.
- `solve_ik_step(q_rad, target_pos, target_quat, model, ik_cfg, dt,
  state=None)`: one differential IK step toward an absolute Cartesian
  target. Pass `target_quat=None` for position-only mode. Reuses
  `state` if you already computed it for the same `q`.
- `joint_angles_from_imus(imu_quats, imu_cfg, model)`: convert corrected
  absolute IMU quats to canonical joint angles. Extraction rules
  (`average_z_yaw`, `gravity_pitch_delta`, `relative_axis_twist`) live in
  the YAML `imu.chain`.
- `warmup_numba_functions()`: compile the numba hot path before the
  control loop starts so the first tick doesn't pay the JIT cost.

## Notes

- `IKConfig` is the only config the solver needs; method-specific knobs
  (`k_val`, `lambda_val`, `min_singular_value`) and behavior switches
  (joint limits, velocity caps, adaptive damping) live on it.
- Adding a wrist or rototilt joint is one extra entry in
  `robot.joints[]`; the FK and Jacobian cores are chain-length agnostic.
- Caller-side state lives outside the solver (target pose, last-tick
  metrics). `IKResult` carries `state`, `adaptive_lambda`, and a
  `rejected` flag with `reason` when the Jacobian is near-singular.
