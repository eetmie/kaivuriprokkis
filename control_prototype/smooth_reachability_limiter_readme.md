# Smooth Reachability and Workspace Limiting for Relative IK Teleoperation

## Overview

This document describes a control architecture for safe and intuitive robot teleoperation using relative end-effector (EE) commands from a gamepad together with inverse kinematics (IK).

The main goal is to prevent users from driving the robot into:

- unreachable IK configurations
- workspace boundaries
- singularity-adjacent configurations
- unsafe operating regions, such as tables

while still preserving smooth and natural motion.

The proposed system focuses on **direction-aware smooth limiting** rather than binary stop behavior.

The intended user experience is:

- the robot behaves normally in free space
- motion gradually slows near limits
- motion along safe directions remains possible
- the robot slides along boundaries instead of freezing
- unreachable targets do not accumulate or wind up
- the operator feels soft constraints rather than hard controller discontinuities

The first implementation phase focuses on workspace and reachability limiting only. Predictive collision compensation and dynamic safety prediction can be added later once the base system behaves correctly.

---

## Problem Description

### Relative EE Teleoperation

The robot is controlled using relative end-effector commands:

```text
dx, dy, dz
droll, dpitch, dyaw
```

These commands are generated continuously from a gamepad.

A relative command system already solves one major issue:

- unreachable absolute targets do not accumulate indefinitely
- target poses do not wind up outside the workspace

However, simple reachability gating is still insufficient.

---

## The Problem with Binary Reachability Gating

A naive implementation looks like this:

```text
if next_pose_reachable:
    allow_motion()
else:
    stop_motion()
```

This creates undesirable behavior near workspace boundaries.

Example:

- robot is already near the +Z workspace limit
- user commands +X
- IK predicts slight increase in +Z or the reachability check fails
- system rejects the entire motion
- robot freezes

This produces:

- sticky boundaries
- discontinuous behavior
- poor teleoperation feel
- unstable operator interaction

The core issue is that the system treats all motion as unsafe once any constraint becomes active.

---

## Desired Behavior

The correct behavior is directional.

At a boundary:

- motion farther into the boundary should be reduced
- motion tangent to the boundary should remain allowed
- motion away from the boundary should remain fully allowed

Example at +Z limit:

| Command | Behavior |
|---|---|
| +Z | slowed or stopped |
| -Z | fully allowed |
| +X | allowed |
| +Y | allowed |
| wrist rotation that worsens the limit | reduced |
| wrist rotation that does not worsen the limit | allowed |

The user should feel a smooth soft wall instead of a hard stop.

---

## High-Level System Architecture

```text
gamepad input
→ relative EE command
→ workspace smooth limiter
→ reachability limiter
→ IK solve
→ joint-space safety limits
→ low-level controller
→ robot
```

The first prototype phase focuses on:

1. workspace smooth limiting
2. reachability limiting
3. directional filtering

Dynamic prediction and collision prediction are intentionally postponed until the base architecture is stable.

---

## Workspace Smooth Limiter

### Core Idea

Instead of enforcing a hard limit only at the workspace boundary, define:

```text
hard_limit
slow_zone
```

The robot gradually slows as it approaches the hard limit.

---

## Example: Table Protection

Never use the physical table surface as the software limit.

Instead:

```text
physical_table_z
safe_z_min = physical_table_z + safety_clearance
slow_zone_start = safe_z_min + slow_zone
```

This compensates for:

- PID overshoot
- controller lag
- IK transient motion
- communication delay
- lack of feedforward

---

## Margin Definition

For a lower Z boundary:

```text
margin = ee_z - safe_z_min
```

Interpretation:

| Margin | Meaning |
|---|---|
| large positive | safe |
| near zero | near limit |
| negative | unsafe |

---

## Smooth Scaling Function

Use a continuous scale function:

```text
s = clamp(margin / slow_zone, 0, 1)
scale = s*s*(3 - 2*s)
```

This gives:

| Position | Scale |
|---|---|
| far from limit | 1.0 |
| near limit | gradually decreases |
| at limit | 0.0 |

The smoothstep form avoids abrupt discontinuities.

---

## Directional Limiting

Only reduce motion that pushes farther into the limit.

Example for table protection:

```text
if v_cmd.z < 0:
    v_cmd.z *= scale
```

Behavior:

| Motion | Result |
|---|---|
| toward table | slowed |
| away from table | allowed |
| lateral motion | allowed |

This preserves free sliding along the boundary.

---

## General Directional Limiter Formulation

Each boundary defines an unsafe outward normal:

```text
n
```

Examples:

| Boundary | Normal |
|---|---|
| lower Z limit | [0, 0, -1] |
| upper Z limit | [0, 0, 1] |
| +X limit | [1, 0, 0] |
| -X limit | [-1, 0, 0] |

For a commanded velocity:

```text
v_cmd
```

Compute the outward component:

```text
outward = dot(v_cmd, n)
```

If:

```text
outward > 0
```

then the motion worsens the violation.

Reduce only this component:

```text
v_cmd -= (1 - scale) * outward * n
```

This preserves tangent motion.

---

## Reachability Limiting

Workspace limits alone are insufficient.

The robot may still approach:

- joint limits
- poor IK conditioning
- unreachable orientations
- singularities
- solver instability

Therefore a second limiter layer is needed.

---

## Reachability Margin

Define a scalar safety metric:

```text
reachability_margin
```

Interpretation:

| Margin | Meaning |
|---|---|
| positive | safe |
| zero | boundary |
| negative | unsafe |

Possible initial metrics:

- IK solve success/failure
- joint-limit distance
- manipulability measure
- IK residual
- solver convergence quality

---

## Initial Simple Reachability Limiter

The first implementation can be conservative.

If reachability becomes poor:

```text
v_cmd *= scale
```

where:

```text
scale = smoothstep(reachability_margin / slow_zone)
```

This globally slows the robot near bad IK regions.

This is intentionally simple for prototyping.

---

## Future Directional Reachability Limiter

Later, only the motion directions that worsen reachability should be reduced.

The concept:

```text
margin_now = reachability(current_pose)
margin_after = reachability(current_pose + small_test_step)
```

If:

```text
margin_after < margin_now
```

then the tested command direction worsens reachability.

Only those directions should be attenuated.

This enables:

- sliding along reachability boundaries
- natural constrained motion
- smoother teleoperation feel

A practical implementation can test individual command axes:

```text
for each command axis:
    apply small step
    evaluate reachability margin
    if margin decreases:
        scale that axis near the boundary
```

---

## Why Relative IK Helps

Relative commands already provide an important advantage.

With absolute IK targets:

- unreachable targets accumulate
- the controller continuously tries to chase impossible poses

With relative commands:

- commands only represent incremental motion
- the target naturally remains near reachable space
- unreachable accumulation is avoided

The proposed limiter architecture builds on this advantage.

---

## PID Overshoot and Dynamic Issues

A static workspace limiter alone does not solve dynamic effects.

Example:

- user rotates the wrist
- upstream joints do not compensate fast enough
- the end effector or tool dips downward
- the tool contacts the table

This can be caused by:

- PID lag
- insufficient feedforward
- joint acceleration mismatch
- coupled kinematics
- delayed response in the IK or joint controller

The long-term solution is predictive safety.

However, the first implementation intentionally skips prediction to keep the system manageable. The base limiter should be implemented and tuned first.

---

## Planned Future Improvement: Predictive Safety

Future architecture:

```text
current state
+ current velocity
+ candidate command
→ short-horizon prediction
→ predicted margin
→ safety scaling
```

Instead of evaluating only:

```text
margin(current_pose)
```

evaluate:

```text
margin(predicted_pose)
```

This will eventually handle:

- wrist-induced table collisions
- inertial overshoot
- delayed compensation
- transient IK motion
- controller lag

Angular commands should also be included in this future safety layer, because wrist rotation can reduce clearance even when the EE position command has no downward Z component.

---

## Practical Safety Recommendations

Even in the initial implementation:

### Use safety clearance

Never place software limits directly on physical objects.

For the table:

```text
safe_z_min = physical_table_z + safety_clearance
```

The clearance should be larger than expected PID overshoot and transient motion.

### Use slowdown zones

Always slow before the hard limit.

```text
slow_zone_start = safe_limit + slow_zone
```

### Limit velocities near boundaries

Especially:

- EE linear speed
- EE angular speed
- joint velocity

### Add command rate limiting

Avoid discontinuous operator inputs.

### Add hard emergency constraints

If:

- IK fails
- margin becomes negative
- joint limits are exceeded
- solver residual becomes too large

then enforce hard stop behavior.

---

## Recommended Prototype Order

### Phase 1 — Workspace Limits

Implement:

- X limits
- Y limits
- Z limits
- smooth directional scaling

Goal:

- robot slides along boundaries instead of freezing

### Phase 2 — Reachability Scaling

Add:

- reachability margin
- global command scaling near poor IK

Goal:

- avoid unstable IK regions

### Phase 3 — Directional Reachability

Determine whether command directions worsen reachability.

Goal:

- preserve free motion near reachability limits

### Phase 4 — Predictive Safety

Add short-horizon prediction.

Goal:

- handle PID lag and transient collisions

---

## Desired Final Behavior

The final system should feel physically intuitive.

The operator should experience:

- soft workspace boundaries
- smooth slowdown near limits
- stable IK behavior
- preserved free motion
- no target windup
- no sudden freezes
- no violent corrections

The robot should naturally slide along constraints while remaining responsive and predictable.
