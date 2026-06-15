# IK package — current state

The package is now a clean URDF-style kinematics layer with a single rich
state object and a pure-functional IK step.

## Layout

| File          | Role                                                                  |
|---------------|-----------------------------------------------------------------------|
| `math.py`     | Quaternion / vector primitives + axis-rotation helpers                |
| `model.py`    | `Joint`, `Tool`, `ExcavatorModel`, `load_excavator_model()` (URDF YAML) |
| `kinematics.py` | `RobotKinematicState`, `get_state(q, model)` — one rich FK + Jacobian pass |
| `solver.py`   | `IKConfig`, `IKResult`, `solve_ik_step(q, target_pos, target_quat, model, ik_cfg, dt)` |
| `excavator.py` | `IMUConfig`, `joint_angles_from_imus(imu_quats, imu_cfg, model)`, numba warmup |
| `__init__.py` | Public re-exports                                                     |

## YAML — new shape

The `robot:` section is now URDF-style:

```yaml
robot:
  joints:
    - { name: slew,   axis: [0,0,1], parent_to_joint_xyz: [0.0,    0.0, 0.07  ] }
    - { name: boom,   axis: [0,1,0], parent_to_joint_xyz: [0.0165, 0.0, 0.0645] }
    - { name: arm,    axis: [0,1,0], parent_to_joint_xyz: [0.468,  0.0, 0.0   ] }
    - { name: bucket, axis: [0,1,0], parent_to_joint_xyz: [0.250,  0.0, 0.0   ] }
  tool:
    parent_to_tip_xyz: [0.031, 0.0, -0.142]
```

Adding a wrist/tool joint (rototilt etc.) is one extra entry in `joints`.

## Done

- File moves into `modules/ik/` (previous pass).
- Split solver out of FK; axis-rotation helpers moved into `math.py`.
- **URDF-style `ExcavatorModel`** replaces the old `RobotConfig` (no more
  `link_lengths` / `link_directions` / `origin_offset` / `ee_offset`
  separation). One joint = one fixed translation + one rotation axis.
- **One rich state**: `get_state(q, model)` returns `RobotKinematicState`
  with joint origins, joint axes in world frame, link orientations, EE
  pose, Jacobian, condition number, singular values, and Yoshikawa index
  in a single FK+Jacobian pass. The wrapper zoo
  (`get_pose_from_joint_angles`, `get_all_poses_from_joint_angles`,
  `compute_jacobian_state`, `compute_jacobian_from_joint_angles`,
  `compute_jacobian`, `joint_angles_to_absolute_quaternions` as a public
  helper, etc.) is gone — only `joint_angles_to_absolute_quaternions` is
  kept as a thin compat shim used by the IMU code.
- **`solve_ik_step(...) → IKResult`** replaces the stateful
  `IKController`. The caller holds its own target and joint state; the
  solver is pure-functional. `IKResult` carries `q_next_rad`, `dq_rad`,
  `task_error`, `rejected`, `reason`, the `state` snapshot used, and the
  adaptive damping that was applied.
- **IMU layer** uses `IMUConfig` + `joint_angles_from_imus(imu_quats,
  imu_cfg, model)`; no more `RobotConfig`-embedded IMU fields.
- `command_type: "pose"` vs `"position"` is now expressed at call time:
  pass `target_quat=None` for position-only mode.

## Modules-level migration — DONE

The rest of `modules/` is now on the new API:

- `modules/reachability.py` drives `solve_ik_step` + `get_state` directly
  (signature: `check_reachability(model, ik_cfg, current_joint_angles,
  target_pos, target_rot_y_deg, ...)`).
- `modules/excavator_controller.py` holds `self.model`, `self.imu_cfg`,
  `self.ik_cfg`, plus its own target pose buffers and last-tick metrics.
  Every tick: `joint_angles_from_imus` → `get_state(..., include_jacobian=False)`
  for the state-update half, then `solve_ik_step` in the compute half.
- `modules/joint_compensator.py` takes the `ExcavatorModel` as `model`
  (reads only `num_joints`).
- `modules/robot_service.py` exposes `self.model` and calls
  `get_state(q_rad, model, include_jacobian=False)` to populate
  joint-position telemetry from joint angles.
- `modules/bringup.py` never touched the IK package — unchanged.

## Top-level files still on the old API

These are intentionally left for the user to rewrite — top-level scripts
are easy to update:

- `excv_gui.py` — used `compute_jacobian`, `extract_axis_rotation`,
  `project_to_rotation_axes`. `extract_axis_rotation` and
  `project_to_rotation_axes` are still in `modules.ik` (math).
  `compute_jacobian` is now `get_state(q, model).jacobian`.
- `control_prototype/excv_gui_relative.py` — same pattern.
- Any standalone tools/demos referencing the old IK surface.

## Follow-ups still on the table (todo.txt)

- Decide whether to drop `pinv` / `svd` / `trans` methods if real
  hardware only ever uses `dls`. Right now all four are still wired.
- Decide whether the IK step should know about velocity-mode integration
  or whether the controller stays responsible for converting joystick
  twists into absolute targets.
- Move PID gains / `controller.*` out of `control_config.yaml` if the
  IK package no longer owns them (it doesn't).
