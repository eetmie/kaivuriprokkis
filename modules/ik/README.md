# IK Module

This package owns excavator kinematics, Jacobian-based IK, and IMU-quaternion-to-joint-state helpers.

## Layer Map

```text
ExcavatorController / ROS IK nodes
    -> modules.ik config + excavator helpers
        -> IKController / solver functions
            -> joint targets / pose estimates
```

PWM output is not owned here. IK produces joint-space or task-space results that higher-level controllers convert into valve commands through the hardware/PWM stack.

## Files

- `config.py`: `IKControllerConfig`, `RobotConfig`, and YAML robot model loading.
- `math.py`: quaternion and vector primitives. Quaternion convention is `[w, x, y, z]`.
- `solver.py`: FK/Jacobian numba cores, IK methods, and `IKController`.
- `excavator.py`: excavator-specific IMU role handling, canonical joint-angle extraction, FK wrappers, numba warmup.
- `__init__.py`: public re-exports for common callers.

## Data Flow

```text
Corrected IMU quaternions / joint angles
    -> canonical excavator joint state
    -> forward kinematics / Jacobian
    -> IK update toward target pose or position
    -> desired joint motion
```

The active controller treats canonical joint angles as the source of truth. Quaternion helpers assume corrected joint-frame quaternions unless a function explicitly states otherwise.

## Main Entry Points

- `load_excavator_robot_config(...)`: load kinematic model from YAML.
- `IKController(...)`: differential IK solver.
- `canonical_joint_angles_from_imus(...)`: convert IMU quaternions to canonical excavator joints.
- `get_kinematic_state(...)`: from radians joint angles, return all link poses and EE pose in one named bundle.
- `compute_jacobian_state(...)`: from radians joint angles, return Jacobian, condition number, singular values, and Yoshikawa index in one named bundle.
- `get_pose_from_joint_angles(...)`: FK pose lookup from joint angles.
- `warmup_numba_functions(...)`: compile hot numba functions before real-time use.

Prefer radians joint angles as the public/internal IK boundary. IMU quaternions should be reduced to canonical joint angles as early as possible; most callers should not need absolute link quaternions directly.

## Notes

- IK config intentionally requires method-specific `ik_params`; no fallback solver gains are supplied.
- Joint velocity limiting, joint limits, adaptive damping, and Jacobian metrics are configured in `IKControllerConfig`.
- `solver.py` still bundles helpers, numba kinematics cores, and the controller. Split candidates are documented in comments and `REFACTOR_NOTES.md` if present.
