# IK-assisted actuator collection

20 October 2026. The native randomized IK/PID prototype now also has a separate
initial collector, `kaivuri-pid-collect`, with joint-command guards, compatible
recording and its own controller sidecars. See [PID_COLLECTION.md](PID_COLLECTION.md).
IK-assisted driving is not added to `kaivuri-mlp-collect`. Powered acceptance
and checked full-reach routing remain pending. The frequency-default update is
a separate, working change.
Example settings are in
[ik_collection.example.yaml](../config/ik_collection.example.yaml); the current
collector does not load that file. The new PID executable loads
[pid_collection.yaml](../config/pid_collection.yaml).
It permits inward PID recovery from measured poses outside the working range,
and prints rough measured joint-pose coverage every 30 seconds and at exit.

## Current randomized prototype

`kaivuri-ik-preview` exercises the native controller against an explicitly
synthetic lag/deadband plant. It never opens robot devices. Run from the repository:

This older synthetic plant clamps simulated joint positions; it is not evidence
of physical limit containment. The separate collector adds a post-combination
command guard and records its intervention as described in `PID_COLLECTION.md`.

```sh
native_stack/build/mlp/kaivuri-ik-preview \
  --out native_stack/results/ik-preview-new \
  --profile configuration_files/profiles/jetson_bucket \
  --seconds 600 --seed 42
python3 native_stack/tools/ik_preview_report.py native_stack/results/ik-preview-new
```

The report creates `preview.png` and `report.json` with measured configuration
coverage, command spectra, duration/gain distributions and PID rate dwell.

Each command specifies cutting-lip pitch plus tip velocity direction and speed
in the carriage X/Z plane. Command durations span **0.1–3 seconds**, sampled on
a logarithmic scale and quantized to 10 ms. Coverage travel uses the upper third
of the duration range, so short commands do not prevent longer travel. The
initial mix is 40% coverage-directed travel, 35% free directions, 10% reversals,
10% pitch-only and 5% holds. Percentages describe segment draws, not time shares.

The user confirmed randomizing **both PID frequency and gains**. At each segment,
each driven joint independently draws 25, 50 or 100 Hz and P/I/D gain multipliers
from 0.5–1.5 times its profile gains. Current driven-joint D gains are zero and
remain zero. PWM/recording stays fixed at 100 Hz; the preview emits one row at
that cadence. PID output is held between updates, with actual elapsed dt,
requested rate, update flag and actually applied gains recorded. A gain change
preserves the integral contribution and derivative history; feedback gaps reset
state without catching up an old reference. These are prototype ranges, without
powered tuning acceptance.

Initial travel speeds are 5–20 mm/s, pitch goals are within 45 degrees of measured
pitch, pitch slew is at most 0.4 rad/s, positional reference lead is at most 30 mm
and pitch lead at most 0.25 rad. Base PID output is bounded to ±0.5 before the
overlay and final ±1 clamp. This bounded reference follows the measured machine
rather than allowing an unreachable virtual target to run ahead.

Excitation has an **independent clock**. The preview alternates 20-second clean
and excited blocks, with sine and chirp alternating in excited blocks. The
preview chirp sweeps 0.05–12 Hz in five seconds each direction, then rests for
ten seconds. Its amplitude varies between 0.2 and 0.4, with single-joint, pair
and all-three targets cycled. These short preview sweeps are distinct from the
live collector's 60-second default sweeps. Short command segments never restart
the excitation phase; otherwise they would repeatedly skip the fast end. The
overlay is evaluated every 100 Hz tick even while the PID output is held.

`preview.csv` records measured joints/task pose, references, assist, requested
and effective excitation, final command, requested frequencies, update times,
applied gains, IK errors and failure reasons. `protocol.yaml` freezes the seed,
settings, segments and excitation parameters; the selected profile files are
copied beside it. `coverage.csv` credits measured full joint configurations,
including separate moving dwell. Unreachable references zero the output and
are explicitly labelled; a requested segment may consequently finish early.

Coverage selection currently maintains a **fresh in-memory** 10-degree joint
grid. It draws 64 goal candidates, prefers low moving dwell with a travel cost,
and retains a goal across short commands for up to 30 seconds. It does not yet
import the historical map, validate collisions or route around obstacles.
Synthetic coverage is diagnostic, never credit toward a training dataset or a
claim of physical full-reach coverage.

Native tests cover Jacobians, position **and** pitch acceptance, PID saturation
and overlay anti-windup, held outputs, gain changes, actual update dt, clock
gaps, reproducibility and measured-versus-scheduled coverage. The parity check
compares 800 FK samples across both profiles and 6,000 PID samples against the
existing Python implementations. The preview integration test checks a repeatable
150-second run, all rate choices, exact measured dwell, command decomposition,
input validation and preserved fast chirp power. The ten-minute seed-42 example
visits 318 joint cells across 544 segments; this is synthetic controller exercise,
not powered tracking validation.

## Intended behavior

Use measured pose coverage to choose useful missing arm configurations, move
between them under slow IK/joint PID control, and add the existing valve
excitation after the PID output. Collect clean motion and motion with fast
excitation. The operator can take over with the sticks at any point.

The first scope is boom, arm and bucket in the carriage-relative arm plane.
Slew and tracks remain manual and are not part of automatic reach coverage.
"Full reach" means the connected, checked motion envelope, with excluded cells
and reasons reported. It does not mean every point inside the geometric outline
can be reached by a straight Cartesian move.

Keep one 100 Hz hardware owner, the existing raw 200 Hz IMU capture, shared valve
calibration/PWM writer and committed recording format. Do not bring Python,
Torch, camera services or a second controller into the collection loop.

## What exists and what needs bridging

| Component | Reuse and gap |
| --- | --- |
| [Pose coverage](../../data_collection/plot_coverage.py) | Reuse FK, bins, log selection and interval accounting. Current output is PNGs plus a summary, with no reusable 3D occupancy payload or collision-checked routes. |
| [Native FK](../src/core.cpp) | Reuse profile-driven joints and carriage-relative link poses. Add a collection task-pose adapter for the cutting-lip pitch offset. |
| [PID engine](../../modules/pid.py) | Port explicit-dt scalar math and test parity, including derivative handling, integral limits and anti-windup. Load boom/arm/bucket gains from the selected profile. |
| [Python IK/reachability](../../modules/reachability.py) | Reference DLS, warm starts and bounded rollouts. Its success check currently tests position and conditioning, not final pitch or collision: it is not the complete automatic-motion validator. |
| [Relative-command prototype](../../control_prototype/relative_command_processor.py) | Reuse measured-state synchronization and bounded reference lead. Avoid allowing an integrated virtual target to run ahead of the machine. |
| [Bundle kinematics](../../learned_control/kinematics.py) | Reference weighted planar IK, polygon collision pairs and edge backtracking. Port the needed geometry math, not the Torch runtime or a learned policy. |
| [Native excitation](../src/mlp_excitation.cpp) | Reuse generator version 4: sine carriers 0.2–10 Hz, instantaneous limit 12 Hz; chirp 0.05–12 Hz; existing stroke mode remains separate. |
| [Native recorder](../src/mlp_record.cpp) | Extend fixed records with controller contributions, targets and plan labels. Preserve requested and final command meanings. |

The existing September 11 reports in `/home/masi-pgx/robot-data-coverage/` cover
3496.78 seconds across seven recordings, including 2816.56 seconds classified
as moving. Their frozen bucket profile exactly matches the current
`jetson_bucket/control_config.yaml` (SHA-256
`5b7042d6eafac5cfb05625ddf2533aaa0889843e4da27cf2d1f92b038daf6039`).
They explicitly say `workspace_collision_checked: false`. Their summaries and
images cannot reconstruct a full joint-space occupancy table; rebuild that
table from the source logs. Those seven CSVs are not in this checkout's
`data_collection/hydraulic_data/` directory.

## Coordinate and calibration contract first

Use the selected profile for both map generation and control. Start with
`jetson_bucket`, which describes the cutting tip, slew-bearing origin and
cutting-lip orientation used by the existing coverage reports. Do not silently
switch the current collector's profile or mix its default `jetson` geometry
with a bucket-profile map.

| Quantity | Order / convention |
| --- | --- |
| Native measured angles | `[slew, boom, arm, bucket]`, radians |
| Planner joint coordinates | `[boom, arm, bucket]`, radians |
| Task pose | `[x, z, cutting_lip_pitch]`, metres/metres/radians, in the carriage frame |
| Native valve commands | `[slew, boom, arm, bucket, trackL, trackR]`, normalized |
| Training/learned bundle state | `[boom, arm, bucket, passive_carriage_pitch]`; its fourth value is **not slew** |

Native `bucket_pitch` currently describes the joint/tool axis and does not read
`robot.tool.pitch_offset_rad`. The bucket profile's cutting-lip offset is about
160.8 degrees. Define a separate task-pose adapter:

```text
task_position = native_fk.upper.position
task_pitch = wrap(native_fk.bucket_pitch + profile.robot.tool.pitch_offset_rad)
```

Preserve the existing shared FK/status convention. Test the adapter against
`plot_coverage.forward_kinematics`, including pitch wrap and nonzero link Z
offsets, before accepting an IK trajectory. For the bucket profile the slew
joint translation is zero; other origin conventions require explicit handling.

Record independent hashes for geometry/limits/tool frame, runtime IMU mapping
and mounting, and the checked motion envelope. Preserve the full profile hash
too. Reject incompatible coverage for automatic selection. Historical training
coordinates use a fixed sensor contract that can differ from runtime mounting:
do not feed training NPZ positions straight back as controller targets or
subtract today's mounting correction twice. Reconstruct from raw data or use an
explicit, verified conversion when migrating calibration groups.

## Coverage representation

Add an offline export beside the plotter: a sparse 3D joint-cell table, explicit
axis edges, profile/calibration identities and source hashes, stored as NPZ plus
JSON metadata. Keep the X/Z/pitch and pairwise plots as views of this table.
Different bucket orientations at the same tip position must remain distinct.

Start with the existing 10-degree joint bins, 20 mm spatial bins and 15-degree
pitch slices. These are reporting resolutions, not motion step sizes.

Maintain separate quantities:

1. Measured, eligible pose duration per complete joint configuration.
2. Active-motion duration and coverage of clean/excited motion, command signs,
   single-axis/pair/all targets, and frequency bands 0–1, 1–3, 3–6, 6–9, 9–12 Hz.
3. Training-usable windows confirmed by the importer. Pose occupancy alone does
   not establish complete histories, valid labels or adequate excitation.

Do not require every Cartesian product of pose/frequency/sign/target cells to
be filled in the first session. Prioritize pose holes and underrepresented
dynamic bands within a stated time budget, then inspect regional diagnostics.

For native data, require completed manifests and data integrity, live pump,
successful writes, valid input/state, causal packet availability and fresh
timestamps. Explicitly exclude dry/bench runs. Count actual elapsed intervals
only when both endpoints are eligible and consecutive; never bridge a gap or
credit a scheduled target as visited.

The existing plotter's motion-only path reads `joint_vel_*` and `vel_age_s`.
Those are intentionally NaN in native MLP logs. The new export must use
same-packet `joint_gyro_vel_*` or reconstruct aligned gyro rates from raw IMUs.
The old plotter also does not check pump/dry flags, so its historical duration
cannot be treated as native training coverage without revalidation.

## Selecting targets and connecting them

Build candidate joint configurations inside the selected limits, initially
using the stroke mode's 10-degree margin. Convert candidates through the same
FK to task poses. Associate each candidate with its full joint cell and
coverage deficit.

Create a graph whose edges are short candidate transitions. Validate the
intermediate configurations, not only endpoints: joint margins, tool position
and pitch, calibrated geometry, self-collision, defined floor/exclusion regions,
IK branch continuity and conditioning. The exported bundle contains six planar
collision silhouettes and nine collision pairs; these are a useful starting
contract, but do not describe external obstacles or prove 3D clearance.
Do not treat a nearest-rounded coarse collision grid as a conservative
clearance certificate. Refine or check the actual polygons along each edge.

Use weighted DLS for `[x, z, pitch]`, seeded from measured/current joint state.
The existing bundle uses a 0.2 m orientation weight; retain that as the initial
reference convention and measure conditioning in that same scaled task. Old
20/40 condition thresholds are not portable across differently scaled/full
Jacobians without calibration. Require both position and wrapped pitch error
to converge.

Prefer short Cartesian strokes and short-arc pitch interpolation. When a
straight Cartesian segment has no continuous valid IK branch, route through
checked neighboring configurations or use a labelled, checked joint transition.
Do not force a straight move through a hole in the workspace. Search from the
measured pose's connected component and report disconnected/excluded targets.

Choose the next target by coverage deficit, missing dynamic-band value and
estimated graph travel cost. Update priorities from measured data, with a
bounded retry/skip record for unreachable or stalled segments. A preview must
show the path, selected targets, excluded cells and reasons before execution
support is added.

## Tracking and excitation

```text
coverage selector -> checked segment -> bounded task reference -> DLS IK
                                                               |
measured joint angles -------------------------------> joint PID
                                                               |
existing excitation ------------------------------------------> +
                                                               |
                                      clamp -> valve shaping -> PWM
```

The base trajectory is slow. Start with a proposed 20 mm/s tip-speed limit and
30 mm maximum reference lead, then establish useful acceleration, angular-rate
and tracking tolerances in simulation and the operator's short powered tests.
These are design starting points, not measured hydraulic limits. Slow progress
when tracking falls behind; freeze/reset references to measured state on mode
changes, input loss, feedback discontinuity and resume. Never catch up to a
wall-clock trajectory after a pause.

Run the native FK, bounded warm-start IK and explicit-dt PID at 100 Hz. Do graph
search/map updates offline or in a bounded lower-rate planner, not inside the
10 ms I/O loop. Candidate plans pass through a fixed-size handoff and are
checked against the state/envelope identity before use. Keep allocations,
CSV formatting, hashing and file reads outside the tick.

Compose the commands explicitly:

```text
u_requested = u_assist + u_excitation
u_final = clip(u_requested, -1, 1)
```

Manual takeover cancels assistance/excitation before direct stick output,
rather than adding an unexpected stick correction to an automatic path.
Reuse the profile PID gains in radian units. Initially cap base-controller
authority at 0.5 and use an explicit 0.35 excitation amplitude for assisted
chirps, leaving useful headroom; the broader frequency defaults remain.
Sine's larger amplitudes can be selected explicitly and their clipping logged.
Anti-windup must account for final combined-command saturation, not only the
PID's own clamp. Clear/reset integral state at takeover and reference changes
that would otherwise cause a jump.

Adding excitation after PID is necessary but does not by itself prevent feedback
from cancelling it. Keep base tracking slow, log the PID and excitation
contributions separately, and measure the opposing PID energy in the 6–12 Hz
band. Establish the tracking bandwidth/gains on clean motion first; only add a
control-branch filter if measurements justify it. Any such filter must leave
raw IMUs, recorded pose/gyro and training labels untouched. Do not smooth the
final valve command or reduce the excitation frequency to make tracking look
better.

Alternate clean and excited traversals through useful regions. Include both
directions and later pair/all-joint traversals for flow sharing. The selector
must not spend an entire session holding still in sparse cells: full strokes,
reversals, different loads and short neutral holds remain useful. The old
learned actuator model is a simulation aid, not proof of the fast real response
it previously underestimated.

## Operator behavior

Preserve A record/save, B excitation permission, X pump, Y reload between
strips, D-pad excitation target, and direct manual controls. Propose the unused
Start button as the assistance on/off toggle. Starting assistance synchronizes
to measured state; an active recording is required so automatic motion is
captured. Sticks outside the existing deadzone immediately return to direct
manual mode. Resume is explicit, not automatic when sticks are released.

First support a single operator-selected, validated waypoint. Then enable
coverage-selected continuation within the same motion envelope. B-off stops
excitation immediately; the planner must not silently turn it back on.
Pump-off, strip save/timeout, lost input, stale feedback or an invalid planned
segment cancels assistance and its excitation contribution. Keep the existing
manual-mode pump semantics and neutral/pump-minimum exit path.

This remains the supervised miniature-robot operating model: no new arming,
hold-to-run deadman, command lease or separate producer permission layer.
Motion-envelope checks determine whether an assisted reference is valid; the
physical power switch remains the cutoff. They are not a replacement for it.

## Recording changes

Preserve `combined_cmd_*` as the final normalized command fed into the existing
shaping path. Preserve `manual_cmd_*` as actual manual input. Add fixed-size
fields for assist commands, joint/task references and measured task pose, plan
and segment IDs, controller mode, tracking errors, progress scale, IK residuals,
constraint reason, envelope identity and effective excitation.

In assisted mode:

```text
effective_excitation = u_final - u_manual - u_assist - u_constraint
```

The shared writer now subtracts assist and post-combination constraint correction
for PID takes, while manual takes retain zero for both. The importer also counts
PID/manual/idle sources explicitly for these takes, so assisted clean motion
is not called manual. The base training input remains the final command.

Freeze the plan, PID/IK settings, geometry/envelope/calibration identities,
seed/protocol and any skips in the session provenance. Keep IMU epoch splitting,
committed manifests and final-command training inputs. Control-mode changes
need labels, not fabricated continuity across real gaps. Retain complete-session
train/dev/test separation and the native importer's gyro targets.

## Implementation sequence and acceptance

1. **Offline coverage export and preview.** Rebuild a full joint-space table
   from explicit compatible logs; export map/route identities, deficits and
   exclusions. Verify native quality flags, gyro motion filtering, discontinuity
   accounting and full configuration identity. No device access.
2. **Native task-pose, Jacobian, IK and PID modules.** Compare with Python over
   random and limit poses, both tool-pitch conventions and pitch wrapping.
   Test Jacobians by finite differences, wrong-pitch/right-position rejection,
   singular/no-solution cases, continuous branches and PID parity/saturation.
3. **Single-waypoint assistance with logging.** Test measured-state start/resume,
   takeover, strip lifecycle, stale/reset feedback, final command decomposition
   and bounded tick work using byte replay/intercepted I2C. Round-trip through
   the real actuator importer. Powered acceptance is an operator session.
4. **Coverage-guided execution.** Add graph transitions and bounded replanning.
   Test valid endpoints with an invalid midpoint, disconnected components,
   frame/calibration mismatch, measured-vs-planned coverage and unavailable
   exclusion geometry. Stop at a configurable collection budget/strip timeout.
5. **Fast-excitation acceptance.** Inspect actual logged command spectra and
   feedback cancellation; compare clean/excited regional coverage and independent
   train/dev/test sessions. Recheck 100/200 Hz timing and writer loss on Jetson.

First implementation target: step 1 and the native frame/PID parity pieces.
Do not enable automatic full-envelope driving as part of the frequency-default
deployment. Remaining physical inputs are the actual floor/obstacle envelope,
validated clearances and acceptable tracking/error limits; the plots alone
cannot supply them.
