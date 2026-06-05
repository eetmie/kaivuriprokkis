# IK package — what was moved and what's next

## Done

The four top-level IK files were moved into this package:

| Before                              | After                  |
|-------------------------------------|------------------------|
| `modules/quaternion_math.py`        | `modules/ik/math.py`    |
| `modules/differential_ik.py`        | `modules/ik/solver.py`  |
| `modules/differential_ik_cfg.py`    | `modules/ik/config.py`  |
| `modules/excavator_ik_utils.py`     | `modules/ik/excavator.py` |

`modules/ik/__init__.py` re-exports the full public API, so callers can do
`from modules.ik import IKController, RobotConfig, canonical_joint_angles_from_imus, quat_normalize, ...`. Submodule imports
(`from modules.ik.solver import IKController`) also work and are preferable in
hot-path code where the dependency should be visually explicit.

All call sites were updated in one pass; the old paths are gone.

## Follow-ups (next IK pass)

### 1. Split `solver.py` into `kinematics.py` + `solver.py`

`solver.py` still bundles three concerns:
- Axis-rotation helpers (`extract_axis_rotation`, `project_to_rotation_axes`) — pure quaternion math, belongs in `math.py`.
- `@njit` FK/Jacobian cores (`forward_kinematics_core`, `forward_kinematics_with_ee_offset_core`, `compute_jacobian_core`, `_compute_ee_position_core`) — the numba hot path; pull into `kinematics.py`.
- `IKController` + `ik_method_*` solver dispatch — what `solver.py` actually ends up being.

This split is a pure code move within the package; the public re-exports in `__init__.py` insulate callers from it.

### 2. Lock it in as excavator-specific

The fleet is excavators only. Stop pretending the IK is robot-agnostic.

- **Joint order is fixed**: `slew → boom → arm → bucket` (4 DOF). The `RobotConfig` YAML loader in `config.py` can stop accepting arbitrary chain lengths and hard-code the four named links. Catch shape mismatches at load time with a clear error instead of silently propagating wrong-sized arrays through FK.
- **All joint axes follow the same pattern** (slew around world-Y, the rest around the arm plane). The generic axis-rotation helpers can be replaced with a small `EXCAVATOR_JOINT_AXES` constant and inline rotation that drops branches in the numba hot path.
- **End-effector variation**: standard bucket vs. an optional rototilt addon (adds two extra DOF at the wrist). Model this as a small `ToolKinematics` enum / dataclass appended after `scoop`, not as a fully generic chain.
- **Drop unused flexibility**: if `command_type: "pose"` is never actually used on real hardware, delete it. Same goes for any `ik_method_*` that's never selected by a real config.

Net effect: smaller hot path, fewer branches in numba, no more "what if someone configures a 7-DOF arm" defensive code.

## Don't pull into this package

- Hall homing (still robot-specific, lives in `run_hw_v2.py`).
- `joint_compensator.py` / linkage-rate compensation — those are servo/hydraulic, not kinematics.
