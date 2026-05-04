# Hydraulic actuator model creation notes from the uploaded excavator papers

## Scope
This note only extracts material that is directly useful for **creating, parameterizing, training, and validating a hydraulic actuator model** for excavator-arm control.

I intentionally leave out most digging-policy, reward-shaping, mapping, and planning details unless they clearly affect the actuator-model choice.

---

## Executive summary

### Papers that actually describe a learned hydraulic actuator model
1. **Egli & Hutter (2020)** - *Towards RL-Based Hydraulic Excavator Automation*  
   This is the **main source** for the actuator-network recipe: data collection, network structure, inputs/outputs, optimizer/loss, and the strongest empirical findings.

2. **Egli & Hutter (2022)** - *A General Approach for the Automation of Hydraulic Excavator Arms Using Reinforcement Learning*  
   This paper **reuses the same actuator-model idea** and adds an important data-collection lesson: a **mix of perturbed and unperturbed data** improved real-machine performance.

### Papers that do **not** build a learned hydraulic actuator model
3. **Egli et al. (2022)** - *Soil-Adaptive Excavation Using Reinforcement Learning*  
   Uses **joint-velocity control + low-level PID** instead of a learned actuator network.

4. **Egli, Terenzi & Hutter (2024)** - *Reinforcement Learning-Based Bucket-Filling for Autonomous Excavation*  
   Again avoids a learned actuator network; explicitly notes that for digging in soil, training such a network would require **much broader data collection**.

5. **Terenzi & Hutter (2023)** - *Towards Autonomous Excavation Planning*  
   Planning paper only; no new actuator-model details.

---

## Comparison table

| Paper | Learned hydraulic actuator model? | Low-level abstraction used | What is useful for actuator-model creation? |
|---|---:|---|---|
| *Towards RL-Based Hydraulic Excavator Automation* (2020) | Yes | Direct learned model from valve commands to next-step joint velocities | Full recipe: excitation signals, model IO, architecture, training settings, validation approach |
| *A General Approach for the Automation of Hydraulic Excavator Arms Using Reinforcement Learning* (2022) | Yes (same concept) | Same learned actuator model, reused in better controller setup | Important empirical lesson: mixed chirp / no-chirp data improved real-machine performance |
| *Soil-Adaptive Excavation Using Reinforcement Learning* (2022) | No | Joint-velocity policy + PID torque control | Good contrast: when the authors decide a learned actuator model is not necessary |
| *Reinforcement Learning-Based Bucket-Filling for Autonomous Excavation* (2024) | No | Joint-velocity references + manually tuned inverse dynamics controller | Explains why a digging-specific actuator network would be expensive to build |
| *Towards Autonomous Excavation Planning* (2023) | No | Higher-level planning over interchangeable digging controllers | No direct actuator-model details |

---

# 1) Egli & Hutter (2020): *Towards RL-Based Hydraulic Excavator Automation*
**Most important paper for actuator-model creation**  
**Relevant pages:** mainly pp. 3-5 in the uploaded PDF (`content.pdf`)

## 1.1 What problem the actuator model is solving
The paper proposes a **data-driven actuator model** so that excavator-arm control does **not** require:
- a full analytical hydraulic model,
- machine-specific gain tuning,
- full manual identification of the cylinder-to-joint geometry beyond what is needed for task-space kinematics.

The learned model is meant to capture:
- hydraulic coupling between actuators,
- friction / stiction effects,
- dead-zones,
- delays,
- temperature dependence,
- engine-rpm dependence,
- nonlinear cylinder-to-joint effects.

## 1.2 Practical prerequisites
The paper lists the following hardware/state requirements for applying the method:
- accurate joint angle / displacement measurements,
- link lengths between actuated joints,
- electric pilot-stage valves.

Implementation details used in the paper:
- Cylinder displacement measured with **Sick draw-wire sensors**.
- Reported resolution: **0.01 mm**.
- Cylinder velocities obtained by differentiating position measurements.
- The Menzi Muck M545 was retrofitted with **Hawe PMZ proportional pressure reducing valves** in the pilot stage.

## 1.3 Data-collection procedure
The data for the actuator model was collected during **semi-autonomous operation** through the electric pilot-stage valves.

### Command design
The valve current setpoint signal was built from two overlaid parts:
1. **Base signal** to ensure coverage of the full motion range.
2. **Frequency- and amplitude-modulated sine**:

```text
s(t) = 0.3 sin(2 pi 0.02 t + phi1) * sin(2 pi (t + 0.99 sin(2 pi 0.1 t) + phi2))
```

### Reported settings
- Maximum excitation frequency: **about 2 Hz**.
- Reason for 2 Hz choice: chosen to be higher than the M545 velocity-control cutoff frequency (**0.5 Hz**).
- For safety during data collection:
  - **boom** and **dipper** base signals were commanded manually with a gamepad,
  - **telescopic** and **shovel** joints used randomized ramp profiles.
- Engine speed kept **constant** during collection.
- Cabin kept in a **constant horizontal orientation**.
- Dataset size: **2 hours**.
- Sampling rate: **100 Hz**.

## 1.4 Network architecture
The hydraulic actuation of the 4 arm joints is modeled with:
- **one single MLP** for all 4 actuators,
- **3 hidden layers**,
- **128 hidden units per layer**,
- **ReLU** activations,
- **linear output layer**.

### Why one shared network matters
The paper explicitly says one shared network is used to account for **hydraulic coupling**, because all 4 joints are supplied by the same accumulator and pump.

## 1.5 Actuator-model inputs and outputs
### Inputs
- Joint positions `q_t^j`
- Joint velocities `qdot_t^j, qdot_(t-0.01)^j, ..., qdot_(t-0.1)^j`
- Valve setpoints `u_t^j, u_(t-0.03)^j, ..., u_(t-0.99)^j`
- Diesel engine rpm `R_t`
- Hydraulic oil temperature `T_t`

### Output
- Next-step joint velocities `qdot_(t+0.01)^j`

### State update convention
- The network predicts **next-step joint velocities**.
- Joint positions are then obtained by **forward integration**.

## 1.6 Design choices that clearly improved the model
This is the most useful part of the paper for reproduction.

### A. Short velocity history
They provide a short history of past joint velocities so the network can average over:
- **stiction friction** effects,
- noise / discretization caused by differentiating position measurements.

### B. Long input-command history
The model receives almost **1 second of command history** because actuator response delays can be **as high as 0.7 s**.

### C. Sparse history sampling
The command history is sampled with a **0.03 s stride** rather than every 0.01 s.
Reason: the commands do not change that fast, so sparse sampling reduces dimensionality without throwing away much information.

### D. Engine rpm as an input
Even though the engine speed was nominally kept constant during collection, the authors observed **momentary rpm drops under heavy load** when several actuators moved quickly at once. Including rpm therefore improved realism.

### E. Hydraulic oil temperature as an input
The paper explicitly states that actuator behavior varies with **oil temperature**, so temperature was included as a model input.

### F. Input normalization
All inputs were normalized to:
- **mean 0**,
- **standard deviation 1**.

This is described as a training-speed improvement.

### G. Predicting velocity difference instead of absolute next-step velocity
This is one of the strongest explicit empirical findings:
- The model quality improved **a lot** when training for the **difference in joint velocity between two time steps** instead of predicting the absolute next-step velocity directly.

This is probably the single most actionable modeling tip in the corpus.

## 1.7 Training settings
The actuator model was trained:
- in **supervised learning**,
- with **Adam** optimizer,
- with **MSE loss**,
- for **20,000 epochs**,
- with a **single batch** (effectively full-batch training).

Reported training time:
- about **10 hours**
- on a machine with **AMD Ryzen 9 3950X**, **32 GB RAM**, and **NVIDIA RTX 2080 Super**.

## 1.8 Validation strategy
### Quantitative check
The paper compares the learned model against a trivial **identity mapping** baseline.

Reported RMSE values (`10^-3 [rad, m] / s`):

| Model | Boom | Dipper | Telescope | Shovel |
|---|---:|---:|---:|---:|
| Identity mapping | 3.58 | 4.00 | 4.64 | 11.36 |
| Training (90%) | 2.21 | 2.06 | 2.36 | 7.06 |
| Validation (10%) | 2.28 | 2.10 | 2.44 | 7.96 |

### More important qualitative check
The paper says one-step RMSE is **not enough**, because in simulation the actuator model is rolled forward using its **own previous predictions**.

So the stronger validation is:
- apply the same input sequence to the machine and the model,
- start from the same state,
- roll the model forward on its own predictions,
- compare measured and simulated velocity / position traces.

The telescopic-joint rollout shown in the paper is reported to match the measured behavior closely, with small accumulated position error.

## 1.9 What I would copy directly from this paper
If I were implementing the actuator model from scratch, I would reproduce these choices first:
1. **one shared network** for all 4 actuators,
2. **3 x 128 ReLU MLP**,
3. **0.1 s velocity history**,
4. **0.99 s valve-command history**,
5. **0.03 s history stride** for commands,
6. include **engine rpm** and **oil temperature**,
7. predict **Delta qdot** rather than absolute next-step qdot,
8. validate with **rollouts**, not only one-step loss.

## 1.10 Limitations / future extensions mentioned by the authors
The paper itself points toward several extensions:
- More data would likely improve the model.
- A Dyna-style loop could iteratively improve the actuator model and controller.
- Cabin orientation / gravity direction should be added if the machine operates with non-horizontal cabin poses.
- The actuator model could be extended to include the **cabin turn actuator**.
- This paper only considers **free-space motions**, not soil interaction.

---

# 2) Egli & Hutter (2022): *A General Approach for the Automation of Hydraulic Excavator Arms Using Reinforcement Learning*
**Same actuator-model idea, with a better data-mix lesson**  
**Relevant pages:** mainly pp. 3-5 in `RA_L_ICRA_2021.pdf`

## 2.1 What is new vs the 2020 paper
This paper explicitly says it is **following the previous actuator-modeling work**.
So the actuator model is best read as a **continuation / reuse** of the 2020 recipe, not as a completely new hydraulic-actuation model.

The figure in the paper again shows the actuator network as **3 x 128, ReLU**.

## 2.2 What is restated clearly
The paper restates the key structure:
- one **single feed-forward neural network** models the four main arm actuators,
- the network uses valve commands and current measurements plus their history,
- it predicts **next-step joint velocities**,
- joint positions are obtained by integration,
- the nonlinear cylinder-to-joint conversion is captured **inside** the actuator network.

### Restated inputs / outputs
The same IO table appears again:
- `q_t^j`
- `qdot_t^j, qdot_(t-0.01)^j, ..., qdot_(t-0.1)^j`
- `u_t^j, u_(t-0.03)^j, ..., u_(t-0.99)^j`
- engine rpm
- hydraulic oil temperature
- output: `qdot_(t+0.01)^j`

## 2.3 Most important actuator-model lesson in this paper
### Mixed perturbed and unperturbed data improved performance
This is the clearest actuator-model creation lesson added by the 2022 paper.

The authors report that:
- controller performance on the real machine improved when the dataset also contained data **without** overlaid chirp,
- using **only** perturbed signals or **only** unperturbed signals produced worse performance,
- therefore, the dataset must cover **all relevant operating modes**, not just one excitation style.

This is a very practical and believable recommendation for system identification on hydraulic machinery.

## 2.4 Updated data collection notes
### Signal design
The paper describes the collection signal as:
- **ramp profiles** ensuring full cylinder-range coverage,
- **overlaid chirp signal**.

### Reported settings
- Final results use **100 minutes** of data at **100 Hz**.
- Roughly **half** of the data was collected **without** overlaid chirp.
- Valve setpoints are in **[-1, 1]**:
  - `-1` = fully open in negative direction,
  - `+1` = fully open in positive direction,
  - `0` = closed.
- A **constant offset** is applied to partially compensate the **dead zone**.

## 2.5 What is inherited from the 2020 paper rather than fully rederived here
The 2022 paper does **not** restate all low-level model-training details.
So, for these items, the 2020 paper remains the best source in this paper set:
- exact optimizer / loss / epoch count,
- the explicit statement that predicting **Delta qdot** improved model quality,
- the strongest discussion of why rpm and oil temperature help,
- the strongest discussion of why long history is needed.

## 2.6 Best practical reading of this paper
Treat this paper as saying:
- **keep the same actuator-network recipe**, but
- collect a **richer mix of data**, not only heavily perturbed excitation.

In other words, the paper strengthens the **dataset-design** story more than the **model-architecture** story.

---

# 3) Egli et al. (2022): *Soil-Adaptive Excavation Using Reinforcement Learning*
**Useful mainly as a contrast case**  
**Relevant pages:** mainly pp. 3-6 in `Soil-Adaptive_Excavation_Using_Reinforcement_Learning 1.pdf`

## 3.1 Does this paper create a learned hydraulic actuator model?
**No.**

Instead, it deliberately moves to a different abstraction:
- excavator dynamics simulated in **RaiSim**,
- policy outputs **joint velocity commands**,
- an explicit **PID controller per joint** converts velocity references to torques,
- torque and velocity limits are clipped.

## 3.2 Hydraulic-actuation details it still models approximately
Although there is no learned actuator network, the paper does include one actuator-specific approximation:
- hydraulic actuators can absorb much more force in the direction opposite to the command,
- so the authors change the **torque limits depending on desired joint velocity**.

This is useful if you want a lightweight actuator approximation without full system identification.

## 3.3 Why they did not use a learned actuator network here
Two reasons are especially relevant:
1. **Joint velocities are easier to transfer** than torques because the dynamic model is not perfectly accurate.
2. Standard hydraulic valves have a **large deadband**, so direct force control would require a more advanced hydraulic setup.

## 3.4 Transfer lesson that still matters for actuator-model thinking
The paper notes that smooth velocity commands with minimal direction changes are critical for successful transfer.

That matters because it suggests a practical tradeoff:
- either build a highly realistic actuator model,
- or constrain the policy / controller interface so the commanded motion stays in a transfer-friendly regime.

## 3.5 Usefulness for your actuator-model notes
This paper is not a source for training a learned actuator network, but it is a very useful explanation of **when the authors decided not to use one**.

---

# 4) Egli, Terenzi & Hutter (2024): *Reinforcement Learning-Based Bucket-Filling for Autonomous Excavation*
**Important for understanding why a digging-specific actuator model was not built**  
**Relevant pages:** mainly pp. 5-6 and discussion sections in `Reinforcement_Learning-Based_Bucket-Filling_for_Autonomous_Excavation 1.pdf`

## 4.1 Does this paper create a learned hydraulic actuator model?
**No.**

The simulation uses:
- a serial manipulator with floating base in **Isaac Gym**,
- policy outputs **joint-velocity references**,
- a manually tuned **inverse-dynamics controller** tracks them.

## 4.2 Most important actuator-model takeaway from this paper
The paper explicitly explains why a learned actuator network was not used for this full excavation task:
- in the previous grading work, an actuator network trained on real-world data was critical for good sim-to-real transfer,
- but for **full-fledged excavation**, such a network would require **much more extensive data collection in different soils**,
- using **joint velocities as outputs** made a more accurate machine model less necessary.

This is a very strong practical point.

### Interpretation
For free-space arm control, the learned actuator model is worth the effort.
For full excavation in soil, the data-collection burden may become so high that the authors preferred to redesign the control interface rather than identify a bigger actuator-plus-soil model.

## 4.3 Extra notes relevant for future actuator modeling
The paper also highlights simulation gaps that a richer machine / actuator model could improve:
- imperfect joint-velocity tracking,
- sensor noise,
- tracking delays,
- inaccurate constant joint-torque limits due to nonlinear piston-joint linkage.

So even though this paper does not build the actuator model, it points to what a future digging-specific actuator model should include.

---

# 5) Terenzi & Hutter (2023): *Towards Autonomous Excavation Planning*
**Planning-only context**  
**Relevant pages:** mainly method / system-overview pages in `Towards Autonomous Excavation Planning.pdf`

## 5.1 Does this paper add actuator-model details?
**No.**

It is a planning paper. The digging controller is treated as a lower-level module that can be swapped out.
The paper mentions that different digging controllers can be used, ranging from kinematic controllers to reinforcement-learning strategies.

## 5.2 Why it still matters to note
It tells you that, in this research line, the actuator model is considered a **lower-level component**, separate from global and local excavation planning.
But this paper does not add new information on how to build that model.

---

# Cross-paper distilled recipe for building the actuator model
If your goal is to reproduce the hydraulic actuator model from this paper set, the most defensible recipe is:

## Data collection
1. Sample on the real machine at **100 Hz**.
2. Use electrically commanded pilot-stage valves.
3. Excite the system with a mix of:
   - ramps / base motions,
   - chirp or modulated-sine perturbations,
   - some segments **without** the perturbation.
4. Ensure full range coverage of all actuated joints.
5. Keep or record machine operating variables that affect dynamics:
   - engine rpm,
   - oil temperature,
   - cabin orientation if it changes.

## Model inputs
Use at minimum:
- current joint positions,
- short joint-velocity history (~0.1 s),
- long valve-command history (~0.99 s),
- engine rpm,
- hydraulic oil temperature.

## Model structure
Start with:
- **one shared MLP** across all 4 actuators,
- **3 hidden layers x 128 units**,
- **ReLU** activations,
- predict next-step velocity and integrate position.

## Training
Start with:
- supervised training,
- **Adam**,
- **MSE**,
- full-batch / single-batch training,
- around **20k epochs**.

## Best-target choice
Prefer learning **Delta qdot** over absolute next-step `qdot`.

## Validation
Do both:
1. one-step error against a trivial baseline,
2. **rollout validation** under recorded command sequences.

---

# Most actionable empirical findings
These are the points I would consider the highest-value engineering takeaways.

1. **Predicting Delta qdot instead of absolute qdot improved model quality a lot.**
2. **A long command history is necessary** because hydraulic delays can be very large (up to ~0.7 s in the paper).
3. **Engine rpm and oil temperature matter** and should not be ignored.
4. **One shared network across joints is justified by hydraulic coupling**.
5. **Rollout validation matters more than one-step loss alone**.
6. **Mixed perturbed + unperturbed identification data improved real-machine results**.
7. For soil excavation, a learned actuator network may become data-hungry enough that **joint-velocity control** is the more practical abstraction.

---

# What I would trust most from this corpus
## Highest-confidence sources for actuator-model creation
1. **2020 paper** - complete recipe.
2. **2022 RA-L paper** - same recipe plus improved data-mixture guidance.

## Papers to use only as supporting context
- *Soil-Adaptive Excavation Using Reinforcement Learning* - contrast case without learned actuator model.
- *Reinforcement Learning-Based Bucket-Filling for Autonomous Excavation* - explains why the learned actuator model was not extended directly to soil excavation.
- *Towards Autonomous Excavation Planning* - system-level planning context only.

---

# Source map
- `content.pdf` - **Towards RL-Based Hydraulic Excavator Automation**
- `RA_L_ICRA_2021.pdf` - **A General Approach for the Automation of Hydraulic Excavator Arms Using Reinforcement Learning**
- `Soil-Adaptive_Excavation_Using_Reinforcement_Learning 1.pdf` - **Soil-Adaptive Excavation Using Reinforcement Learning**
- `Reinforcement_Learning-Based_Bucket-Filling_for_Autonomous_Excavation 1.pdf` - **Reinforcement Learning-Based Bucket-Filling for Autonomous Excavation**
- `Towards Autonomous Excavation Planning.pdf` - **Towards Autonomous Excavation Planning**
