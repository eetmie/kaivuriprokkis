# control_prototype

Sandbox for the **smooth reachability limiter** idea. Lives outside
production code so the existing `excv_gui.py` / client GUI stack is
untouched until the design is validated.

## The idea

Today the production stack uses **binary reachability gating**: when a
commanded pose fails a pre-flight IK rollout (`modules/reachability.py`,
called from `ExcavatorController.give_pose`), the command is dropped
and the robot freezes at the last accepted target. This is the
"sticky-boundary, dead-zone, wind-up" failure mode described in
`smooth_reachability_limiter_readme.md`.

The prototype replaces that with a **smooth, soft-wall limiter**:

```
operator (vx, vy, vz, vyaw)   relative EE velocity command
        |
        v
  reachability_limiter        scalar margin in [0,1] from
                              joint-limit distance, Yoshikawa,
                              and Jacobian condition number,
                              shaped by a smoothstep curve
        |
        v
  scaled v_cmd                tangent motion preserved, motion
                              into the boundary smoothly damped
        |
        v
  RelativeCommandProcessor    integrates v_cmd * dt into an
                              absolute target pose
        |
        v
  controller.give_pose(...)   binary reject DISABLED for the
                              prototype; the smooth limiter is
                              the sole authority on safety
```

Joint-limit distance handles the workspace envelope implicitly --
there is no separate Cartesian workspace box. Yoshikawa and condition
number catch coupling singularities that joint limits alone miss.

Phases (mirrors `smooth_reachability_limiter_readme.md`):

1. **Workspace + smooth scaling** -- *here, but workspace is the joint envelope*
2. **Global reachability scaling** -- *here*
3. **Directional reachability** -- *TODO, see relative_command_processor.py*
4. **Predictive safety** -- *TODO*

## Files

| File | Role |
|---|---|
| `smooth_reachability_limiter_readme.md` | Design doc (the spec we are implementing) |
| `reachability_limiter.py` | Pure-numpy smoothstep + margin math. **Lift target.** |
| `relative_command_processor.py` | Velocity-to-pose integrator. **Lift target.** |
| `excavator_fixture.py` | Real joint limits and link lengths mirrored from `configuration_files/control_config.yaml` |
| `test_reachability_limiter.py` | 24 unit tests (smoothstep, margins, NaN-safety) |
| `test_relative_command_processor.py` | 14 unit tests (lifecycle, integration, wrap, caps) |
| `bench_reachability_limiter.py` | Performance baseline at 3 cost levels |
| `excv_gui_relative.py` | **Server demo** -- forks `excv_gui.py`, no binary gating |
| `gamepad_velocity_client.py` | **Client demo** -- streams `(vx, vy, vz, vyaw)` over UDP |

## How to run

Unit tests (no hardware):

```
python control_prototype/test_reachability_limiter.py
python control_prototype/test_relative_command_processor.py
```

Performance baseline (uses the real numba IK kernel):

```
python control_prototype/bench_reachability_limiter.py
```

End-to-end demo (server on the robot, client anywhere with a gamepad):

```
# on the robot
python control_prototype/excv_gui_relative.py

# on the operator machine (Xbox controller plugged in)
python control_prototype/gamepad_velocity_client.py --host 192.168.0.132
```

The demo uses **port 8081** -- the production server stays on 8080.
This is deliberate: the demo's UDP packets reinterpret
`ControlCommand.pose` as velocity `(vx, vy, vz, vyaw_dps)`. If the
existing client connected, the integrated absolute target would be
read as a velocity and the robot would fly. Different port = no foot-gun.

## Migration target

When the limiter feels right here, the lift to the other machine is:

1. `reachability_limiter.py` -- numpy only, drop in as-is.
2. `relative_command_processor.py` -- numpy only, drop in as-is.
3. `excavator_fixture.py` -- update joint limits and link lengths to
   match the target robot.
4. Replace `controller._reach_enabled = False` (monkey-patch in this
   prototype) with a proper `config_override` kwarg on the target's
   `ExcavatorController.__init__`.

## Not yet here

See the TODO block at the bottom of `relative_command_processor.py`:
directional scaling, predictive safety, input shaping, per-axis caps,
table-floor, config loader, telemetry channel, anti-windup vs measured.

## Likely next architecture: rubber-band model

The current integrator approach may be replaced with a simpler rubber-band
model. The full intended stack would be:

```
client (raw, fast-switching commands)
        |
        v
  server slew limiter       ramps v_cmd per-axis to suit robot dynamics.
                            lives on the server so any client is safe —
                            gamepad, ROS node, web UI, all get the same
                            smooth behaviour for free.
        |
        v
  directional reachability  per-axis smoothstep scale derived from
  scaling                   joint-limit distance, Yoshikawa, condition
                            number. Only damps axes pointing INTO a
                            constraint; tangential axes stay at full scale.
        |
        v
  rubber-band offset        target = meas_pos + v_scaled * lookahead_m
                            target is re-derived from measured EE every
                            tick. zero velocity → target = EE → arm holds
                            naturally. windup impossible by construction.
                            no lead-window clamp or zero-snap needed.
        |
        v
  controller.give_pose(...)
```

Key properties:
- Smooth start/stop falls out of the slew limiter. No special stop logic.
- Zero velocity holds the arm where it physically is, not where the
  integrator drifted to.
- Boundary sliding falls out of directional scaling. Motion tangent to a
  limit keeps full lookahead; motion into it gets a shortened band.
- `reachability_limiter.py` is reused unchanged — it already computes
  the per-margin scalars needed for directional scaling.
- The only new pieces are the server slew limiter and the rubber-band
  offset (both trivial); `RelativeCommandProcessor` is replaced or
  simplified to a one-liner.

The `lookahead_m` tunable (e.g. 30-50 mm) controls how hard the
controller pulls. The open question for hardware testing is whether a
fixed lookahead gives the PID enough authority to drive the hydraulics
smoothly across the full commanded speed range.

## Boundary with production code

This folder reads `modules/*` (IK kernel, controller, hardware, UDP)
but writes nothing back to it. The only production file that needs to
change for migration is `ExcavatorController.__init__` (add config
override). Until then the production server (`excv_gui.py`) and the
production client (`clients/client_gui.py`) keep their original
binary-gating behavior on port 8080.

This folder must NOT import from `configuration_files/pathing_config.py`.
