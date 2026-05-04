# Hydraulic Actuator Neural Network Model

Reference: Egli & Hutter, "A General Approach for the Automation of Hydraulic Excavator Arms Using Reinforcement Learning" (IEEE RA-L, 2022)

---

## Overview

A single feed-forward neural network models all four arm actuators. The network is trained in a supervised fashion using data collected during machine operation. Given valve commands and sensor measurements (current + history), the model predicts joint velocities at the next timestep.

The model implicitly captures:
- Hydraulic actuator dynamics (delays, dead zones, nonlinearities)
- Cylinder-to-joint-space conversion (no need to derive linkage geometry)
- Coupling effects from shared pump supply
- Load sensing behavior

---

## Model Architecture

**Network:** 3 × 128 fully connected layers with ReLU activation

**Prediction target:** Joint velocities at next timestep (t + 0.01s)

**Inference rate:** 100 Hz

### Inputs

| Input | Description | History |
|-------|-------------|---------|
| Joint Positions | `q_t^j` | Current only |
| Joint Velocities | `q̇_t^j` | Current + 10 past samples at 0.01s intervals (spans 0.1s) |
| Valve Setpoints | `u_t^j` | 34 samples at 0.03s intervals (spans ~1.0s) |
| Diesel Engine RPM | `R_t` | Current only |
| Hydraulic Oil Temperature | `T_t` | Current only |

Superscript `j` denotes the four actuated joints (boom, dipper, telescope, shovel).

### Outputs

| Output | Description |
|--------|-------------|
| Joint Velocities | `q̇_{t+0.01s}^j` for all four joints |

---

## Data Collection

### Control Signal Design

Randomized control signals are applied to the four arm actuators. Each signal consists of:

1. **Base ramp profiles** - Ensure coverage of the full motion range of each cylinder
2. **Overlaid chirp signal** - Sinusoidal signal with frequency changing over time (adds dynamic excitation)

### Dataset Composition

| Parameter | Value |
|-----------|-------|
| Total duration | 100 minutes |
| Sample rate | 100 Hz |
| Data split | ~50% with chirp overlay, ~50% without chirp overlay |

The mix of perturbed (chirp) and unperturbed (ramp only) data is important. Using exclusively either type led to worse performance on the real machine. The data needs to cover all relevant modes of operation.

### Valve Setpoint Encoding

- Range: `u ∈ [-1, 1]`
- `-1` = fully open in negative direction
- `+1` = fully open in positive direction  
- `0` = closed

The valves are configured with a constant offset applied to commanded setpoints to partially compensate the dead zone.

---

## Training

**Method:** Supervised learning

**Task:** Given inputs at time `t`, predict joint velocities at time `t + 0.01s`

**Loss:** Not explicitly stated in paper (likely MSE on velocity predictions)

Joint positions during simulation/inference are obtained by integrating the predicted velocities.

---

## What the Model Captures Implicitly

### Cylinder-to-Joint-Space Conversion

Hydraulic excavators use linkage mechanisms to transform linear cylinder motion into angular joint motion. These configurations differ between machines and are laborious to derive analytically.

By training the model end-to-end (valve commands → joint velocities), the network learns this conversion implicitly. You only need to know the distances between joints (link lengths), not the full linkage geometry.

### Actuator Nonlinearities

The two-stage hydraulic circuit (pilot stage → main valves) introduces:
- Control delays
- Dead zones
- Coupling effects (multiple cylinders share one pump)
- Saturation

These are captured in the learned model through the history buffers, particularly the ~1s valve command history.

### Machine-Specific Characteristics

Even machines of the same type behave differently due to:
- Wear
- Manufacturing tolerances
- Hydraulic system implementation differences
- Operating/loading conditions

The data-driven approach captures these particularities without requiring an analytical model.

---

## Integration with Simulation

Once trained, the actuator model serves as the dynamics simulator:

1. Initialize joint states
2. At each timestep:
   - Feed current observations + history to network
   - Network outputs predicted joint velocities
   - Integrate velocities to get new joint positions
   - Update history buffers
3. Compute end effector pose using forward kinematics (requires link lengths)

The simulation runs at 100 Hz (matching the actuator model's training rate).

---

## Hardware Requirements

Minimum requirements for this approach on an off-the-shelf excavator:

1. **Joint state measurements** - Positions and velocities for each actuated joint
2. **Link lengths** - Distances between joints (for forward kinematics)
3. **Electric pilot stage valves** - For automatic control command injection
4. **Engine RPM sensor** - Already standard on most machines
5. **Hydraulic oil temperature sensor** - Already standard on most machines

No high-precision valves or detailed hydraulic system modeling required.

---

## Notes

- The history lengths (0.1s for velocities, ~1s for valve commands) were determined empirically and capture the relevant temporal dynamics including delays
- Training the actuator model and subsequent controller uses the same 100 min dataset
- The approach was validated on a Menzi Muck M545 (12 ton walking excavator) with four arm joints: boom (θ_B), dipper (θ_D), telescope (x_T), shovel (θ_S)
