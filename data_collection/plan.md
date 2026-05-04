# Hydraulic Model Plan

## Goal

Train and compare simple prediction models for the excavator hydraulic arm using logged valve commands, IMU joint feedback, and pressure signals.

Primary target:

- predict joint rates for `boom`, `arm`, and `bucket`
- improve multi-channel control beyond plain PID
- keep the first usable model simple enough to debug on real hardware

## Existing Logger

`drive_logger.py` already logs the key signals needed for offline identification:

- valve commands actually sent: `combined_cmd_*`
- manual commands: `manual_cmd_*`
- sine excitation overlay: `sine_cmd_*`
- joint positions: `joint_pos_*`
- joint velocities: `joint_vel_*`
- cylinder pressures: `pext_*`, `pret_*`
- pump pressure: `pump_ps`
- quality flags: `cmd_stale`, `pressure_stale`

Collection runs in direct mode, so IK/PID is bypassed during recording.

## Pressure Handling

Log raw pressure channels and derive extra features offline later.

Prefer keeping:

- `pext_*`
- `pret_*`
- `pump_ps`

Then derive as needed during analysis:

- chamber delta pressure: `dp = pext - pret`
- absolute chamber pressures
- filtered pressure signals
- load heuristics / normalized variants

This keeps the dataset flexible and avoids locking the project into one pressure interpretation too early.

Practical bandwidth note:

- 20 Hz pressure is still useful as a slow context/load feature for offline modeling
- 20 Hz pressure is not a good primary signal for fast inner-loop pressure control

## Data Collection Strategy

Record multiple sessions instead of one long run.

Each dataset should include:

- single-joint excitation
- two-joint simultaneous excitation
- three-joint simultaneous excitation
- both directions for each hydraulic axis
- low, medium, and high command amplitudes
- a wide range of arm poses
- natural variation in hydraulic load and pump pressure

Prefer excitation that keeps channels separable offline:

- phased multisine
- small random steps / PRBS
- mixed manual driving plus sine overlay

Avoid overfocusing on end stops. Most model fitting should use mid-stroke data.

## Data Quality Rules

Before fitting models:

- drop rows with `cmd_stale != 0`
- drop rows with invalid or stale pressure if the model uses pressure
- exclude near-zero commands when identifying valve gain
- exclude near-stall and end-stop regions for the first models
- verify the logged velocity source is consistent across sessions

If velocity estimates are noisy, prefer gyro-only or fused velocity during logging if that proves cleaner than finite-difference.

## Modeling Roadmap

### Stage 1: Linear MIMO Baseline

Fit a simple static multi-input model:

`qdot = B u`

where:

- `qdot` = measured joint rates
- `u` = commanded valve vector

Purpose:

- establish a baseline
- measure crosstalk directly from off-diagonal terms

### Stage 2: Angle-Scheduled Gain Model

Fit a pose-dependent rate model:

`qdot = K(theta) u`

Start with low-order terms only. Example:

- per-joint quadratic self-gain vs angle
- optional linear/quadratic off-diagonal crosstalk terms

This is the preferred first practical upgrade.

### Stage 3: Pressure-Augmented Model

Extend the scheduled model with hydraulic pressure features:

`qdot = K(theta, p) u`

Useful if prediction quality changes strongly with load.

Start by deriving pressure features offline from the raw logged channels instead of changing the logger to emit only precomputed deltas.

### Stage 4: Lightweight Hammerstein Model

Add a static valve nonlinearity before the MIMO map:

`z = f(u)`

`qdot = G(theta, p) z`

Typical `f(u)` should start simple:

- deadband compensation
- separate extend/retract gains
- saturation-aware scaling

This is likely the best next step if multi-channel motion shows strong nonlinearity.

### Stage 5: Simple Dynamic Extension

Only add dynamics if static models are clearly insufficient:

`qdot[k] = A qdot[k-1] + G(theta[k], p[k]) z[k-1]`

Add this only after proving the static/Hammerstein models leave meaningful lag-related error.

## Offline Evaluation Plan

Every candidate model should be tested on held-out log files, not just the training session.

Evaluate:

1. one-step rate prediction
2. short rollout prediction by integrating predicted rates into joint angles
3. per-joint and overall error
4. regime-specific error

Metrics:

- RMSE
- MAE
- R^2
- max error
- bias per joint

Split reporting by:

- extend vs retract
- low vs high pressure
- single-channel vs simultaneous-channel motion
- low-speed vs high-speed motion

## Recommended First Comparison Set

Compare these models in order:

1. `qdot = B u`
2. `qdot = K(theta) u`
3. `qdot = K(theta, p) u`
4. `qdot = G(theta) f(u)`
5. optional dynamic version with one-step memory

Current benchmark status:

- the baseline comparisons do not require pressure
- pressure-aware comparison should be treated as an additional optional model, not the only benchmark path
- if pressure validity or coverage is weak, skip pressure-based models rather than forcing them into every comparison

This should reveal whether the dominant missing effect is:

- geometry
- pressure/load
- valve nonlinearity
- dynamics/lag

## Runtime Control Direction

If the offline model is good enough, use it first as feedforward, not as a full controller replacement.

Recommended runtime structure:

`u = u_ff + u_pid`

where:

- `u_ff` comes from the identified model or its inverse/pseudoinverse
- `u_pid` handles residual error and unmodeled disturbances

For multi-channel motion:

`u_ff = K(theta)^+ qdot_des`

or with valve nonlinearity:

`u_ff = f^{-1}(z_des)` after solving for desired `z`

## Immediate Next Steps

1. Record several direct-mode datasets with multi-channel excitation.
2. Keep notes for each session: surface/load/task conditions, pump behavior, and anything unusual.
3. Build an offline benchmark script that reads the current CSV format.
4. Fit and compare the baseline linear, angle-scheduled, and Hammerstein-style models.
5. Promote the best model into a feedforward block while keeping PID trim active.
