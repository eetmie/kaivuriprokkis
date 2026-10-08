# Bucket circle on the real robot

`learned_control/run_circle.py` runs the same 100 mm diameter, 20 mm/s X/Z circle with
the learned controller or the simulation-tuned DLS/joint PID. The PID is this repository's
`modules/pid.py` behind the same DLS step as the IK loop (the calculation
Isaac-hydraulic-actuator `tune_pid.py` tunes), bypassing the pose smoother. Slew,
tracks and auxiliary outputs stay neutral. No shadow run is required.

Bundles live in `learned_control/bundles/<controller>_<profile>`, exported by
Isaac-hydraulic-actuator `export_robot_bundle.py` for a profile it generated
(`jetson_bucket`, or the zero-deadzone experiment `jetson_bucket_dz0`). A bundle
pins its profile files; `--robot` must name the profile it was exported for.
On load it replays reference cases computed by the training code and refuses to
run if this runtime does not reproduce them. `proto_*` is controller-proto, the
actor of the 2026-10-07 sessions; `v6_*` is controller V6.

The profile's origin is the slew bearing, boom-pivot height is 78.5 mm
(production CAD), and link vectors and the bucket cutting tip come from the
training USD. The frozen actor sees its training joint zeros through a fixed
calibration offset stored in the bundle; FK and PID use the robot's own zeros.

From the repository root on the robot (`--bundle` is required):

```bash
.venv-lerobot/bin/python learned_control/run_circle.py check --bundle learned_control/bundles/v6_jetson_bucket_dz0 --robot jetson_bucket_dz0
.venv-lerobot/bin/python learned_control/run_circle.py run --controller pid_tuned --pid_gains learned_control/gains/pid_sim_tuned.yaml --log data_collection/circle_logs/circle_pid_ccw_01.csv
.venv-lerobot/bin/python learned_control/run_circle.py run --controller mlp --log data_collection/circle_logs/circle_mlp_ccw_01.csv
```

`check` is an offline geometry/actor check and opens no devices. `run` preflights
the complete circle from the measured pose with the pump off. Release then hold
**Left Bumper** to start; **B**, release or disconnect latches a stop. A one-second
neutral history warmup is retained for inference. Each default run contains
one lap, a one-second lead hold and two seconds of settling. Use `--direction cw`
for the opposite direction. Run each controller/direction three times, alternate
controller order and return to the same starting pose between runs.

Logs are exclusive CSV plus JSON files, with reference/measured tip positions,
joint angles/rates, requested/emitted valves, timing, tracking and radial errors,
and a completion/fault result. PID gains are loaded per run; `pid_robot` uses
the profile gains and `pid_tuned` requires an explicit gains file. Software stops
include stale sensors/controller, loop overruns, position/angle error, and
joint/collision margins. Use the physical emergency stop and free-space motion.
IMU-based FK metrics describe internal tracking; externally measured tip motion
is needed to establish physical accuracy. Hardware acceptance remains pending.

After recording runs, compare matching profiles/speeds without opening hardware:

```bash
.venv-lerobot/bin/python learned_control/run_circle.py compare --logs data_collection/circle_logs/circle_pid_ccw_01.csv data_collection/circle_logs/circle_mlp_ccw_01.csv --out data_collection/circle_logs/comparison --plot
```

This writes a comparison table and overlays the circles and timed errors.
Stopped trials remain in the report. Plotting optionally needs matplotlib.

For repeated warm-up/testing passes, use `--continuous`. Release then press LB
once to start; it may then be released. **A stops motion and the pump**;
**B starts logging** for the rest of the session. Gamepad disconnect and the
existing fault limits still stop the robot. Passes reuse the original center,
including the lead/settling holds. Warm-up samples are discarded until B;
recorded samples stay in a compact preallocated RAM buffer. B starts a
five-minute recording (`--record_seconds 300`); the pump shuts off at the end,
then all passes are saved together with `pass_index`. A can stop and save early.
The JSON `passes` list marks partially recorded passes. Continuous sessions
end on A; their session result is separate from individual pass scores.

```bash
.venv-lerobot/bin/python learned_control/run_circle.py check --bundle learned_control/bundles/v6_jetson_bucket_dz0 --robot jetson_bucket_dz0
.venv-lerobot/bin/python learned_control/run_circle.py run --continuous --log data_collection/circle_logs/circle_mlp_20hz_session_01.csv
```

The actor runs at its trained 20 Hz by default. `--policy_hz 100` is an
experiment with the same frozen actor: it sees fresh history at 100 Hz, while
the twist projection and governor stay at 20 Hz. On 2026-10-07 it chattered at
6–7 Hz on the machine (valve travel ~20/s against the PID's 1.7/s; see
Isaac-hydraulic-actuator `drive_logs/20261007_circle_mlp_vs_pid`). The history holds the controller's own
command for each interval, as in training, not the delayed emitted value.
The run loop writes the valves in the tick it computes them, as `simple_drive.py`
did when the actuator data was recorded (`--valve_writes loop`, default). The
2026-10-07 sessions went through the robot controller's direct-command thread,
about 20 ms later; `--valve_writes thread` keeps that path for comparison. If
the loop stalls, the 150 ms PWM watchdog and the gate's policy timeout still
stop the valves. For continuous PID use
`--continuous --controller pid_tuned --pid_gains learned_control/gains/pid_sim_tuned.yaml`.
Passive carriage rocking allows +/-3 degrees (`--max_carriage_pitch_deg`),
matching the model geometry. The configurable driven-joint velocity guard
(`--max_joint_velocity_rad_s`, default 2) applies only to boom/arm/bucket.
Passive carriage rate is logged but does not trip that driven-joint limit.
To approach the same start as an earlier successful trial, add
`--start_from_log learned_control/circle_start_pose.csv` and a `--pid_gains`
file. LB authorizes a quintic joint approach at <=0.05 rad/s reference speed,
with valves capped to +/-0.25 by default (`--approach_output_limit 0.5` allows
the stronger tested approach), followed by the selected circle controller.
`--joint_margin_deg 0` removes the extra software margin within the pinned
joint bounds; it does not expand those bounds. The approach and circle are
checked independently before the pump starts.
Recorded CSVs include sensor-frame acceleration XYZ [g] and gyro XYZ [deg/s]
for every physical IMU (`imu0_...` through `imu3_...`), plus the role-corrected
quaternions and gyro XYZ. These share the exact packet used by the controller;
firmware startup gyro-bias removal precedes this capture. The JSON records the
sensor-to-role mapping. CSV encoding, disk writes and per-pass scoring occur
only after pump/output shutdown. No saving occurs during circle transitions.
For timing diagnostics, `--allow_timing_overruns` records and reschedules missed
loop deadlines instead of aborting on 30 ms lateness / 40 ms computation.
The independent sensor/controller freshness gate and operator stops remain
active. The requested actor rate and measured timing are stored in the log;
use actual timestamps when evaluating the high-rate experiment.
