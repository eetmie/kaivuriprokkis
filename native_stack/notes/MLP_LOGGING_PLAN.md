# Native MLP data collection plan

19 October 2026. Scope: implementation plan for the next hydraulic actuator
dataset. Implementation and launch instructions are in
[MLP_RECORDING.md](MLP_RECORDING.md); measured acceptance is tracked in
[MLP_VALIDATION.md](MLP_VALIDATION.md).

Build one native collection tool, `kaivuri-mlp-collect`, that works like
`simple_drive.py`: read the gamepad and IMU stream, add excitation, write the
PCA9685 directly and log what it wrote. It shares calibration math and the PWM
writer with the native control service, but it does not run through that
service. The first milestone is a complete, reproducible actuator recording
that trains the next model with the existing loader.

## Operating model: no software safety gates

The robot is a small, supervised miniature excavator. Servo and pump power is
normally switched off, so the robot cannot move. The physical power switch is
the cut-off. The MLP tool therefore omits the native stack's safety layer:

- no `--live`/`--live-pwm` flag: PWM writes are live by default (`--dry` skips
  I2C for desk testing without a PCA9685)
- no arming, hold-to-run, deadman, command lease, decay or safety state
- no pump permission gating: X toggles the pump, as in `simple_drive.py`
- no process supervisor or heartbeat between separate producer and recorder
  processes, because there is only one process

Bench tests and final runs use the same binary and the same command. Only the
servo/pump power differs. Two behaviors remain because they are not gates:
writing neutral pulses and minimum pump on exit or error, and taking the native
`control.lock`, so `kaivuri-control` and this tool never write the same I2C bus
at once.

Software cannot see whether servo power was on. Strips recorded with power off
look like valid data with no motion. Keep them out of training by output
directory (`--out`) or `--suffix`. Training preparation only reads files that
are passed to it explicitly.

## Rates

| Activity | Rate |
| --- | --- |
| PCA9685 carrier | 100 Hz (`pwm_frequency` in every Jetson profile) |
| Collection loop: gamepad, excitation, valve shaping, one I2C write | 100 Hz |
| `drive_log_*.csv` rows | 100 Hz, one row per loop tick, recording exactly what was written |
| `imu_raw_*.csv` | Every accepted packet, nominally 200 Hz, drained each tick |
| Training representation | 100 Hz (`training/dataset.py` `dt = 0.01`) |

The loop runs at the PWM carrier rate because faster writes do not reach the
valves faster. The PCA9685 emits one pulse per 10 ms period, so at 200 Hz half
of the register writes are overwritten before they produce a pulse. Host writes
are not phase-locked to the PCA9685 oscillator. Each write therefore takes
effect at the next carrier period, 0 to 10 ms later, whatever the host rate.
Log post-write time and keep this bound in the schema notes.

A 200 Hz loop would only give fresher state: up to 5 ms old instead of 10 ms.
The model does not need that. If a later controller does, add a serial reader
thread that timestamps packets on arrival rather than speeding up the loop.

A 200 Hz MLP is out of scope. It would change the model contract (`dt`, the V5
history of 41 velocity taps and 21 command taps at 30 ms), the Isaac actuator's
`sim_dt` and the trained weights. Its outputs would still reach the valves at
only 100 Hz. Revisit it only together with a higher PWM carrier and an explicit
model/sample-rate change.

The VLA control service keeps its own configuration and its 200 Hz launcher
default. This milestone does not change it.

## Loop

Each 10 ms tick, scheduled on an absolute `CLOCK_MONOTONIC` grid:

1. Drain the IMU serial port without blocking and feed `ImuParser`. Copy every
   accepted packet into a preallocated raw-IMU queue. `ImuParser::sample()`
   keeps only the latest packet, so the parser must emit each packet as it
   accepts it.
2. Read pending evdev events. Use the axis signs and deadzone of `kaivuri-record`
   and `simple_drive.py`.
3. Compute joint angles from the latest complete packet, the excitation value
   and the clipped combined command.
4. Shape pulses with the existing `valve_pulse()`/`pulse_count()` code and the
   profile calibration (deadband, gamma, ramp, dither). Write them in one
   PCA9685 burst.
5. Push one row (scheduled, start and post-write times plus the values used)
   into a preallocated queue for the writer thread.

The loop does no disk I/O or CSV formatting. A writer thread formats and writes
the rows and raw packets. If a queue overflows or a write fails, the current
strip is marked incomplete and capture continues. A missed period is counted
and logged. It does not stop the robot.

Move the `Pwm` class from `control_main.cpp` into a shared module so that both
binaries use the same register sequence, and add a small shared gamepad reader.
Keep waveform generation and the CSV schema in MLP-specific modules.

## Output

Write the formats the training side already reads, under `*.partial` names,
renamed on clean completion:

```text
<out>/
  drive_log_<stamp>[_<suffix>].csv
  imu_raw_<stamp>[_<suffix>].csv
  excitation_<stamp>[_<suffix>].json
  session_<stamp>.json          # provenance, effective configuration, counters
```

The drive log keeps the `modules/drive_log.py` column names and meanings.
`training.drive_logs` requires `timestamp`, `sample_idx`, `state_imu_ts_us`,
`cmd_stale`, `cmd_age_s` and `combined_cmd_lift|tilt|scoop`. Because there is
no controller between the tool and the PCA9685, `combined_cmd_*` is both the
clipped request and the written command. There is no separate applied-command
column to reconcile. Add `pwm_us_*` pulse columns, `pump_on`, `write_ok` and
the loop timing columns. `timestamp` is the post-write monotonic time.
`state_imu_ts_us` is the device timestamp of the packet behind that row's
joint angles. `cmd_age_s` is post-write time minus command computation time.
`cmd_stale` is 1 only for rows whose write failed or whose input/excitation
state is unknown. Write other historical columns where the native tool has the
same quantity. Otherwise write NaN and document them.

`imu_raw` keeps every packet in firmware units (gyro in degrees/second,
acceleration in g) and keeps raw, uncorrected quaternions. The loader
subtracts the contract's fixed `MOUNT_PITCH`, so profile mounting offsets must
not be applied a second time. The drive-log IMU columns keep their historical
SI units (rad/s, m/s²). Record the host read time of every packet. Packets read
in one tick share that time; it is the conservative availability time, not the
USB arrival time. Type 1 packets carry quaternion + gyro, with no acceleration, and stay distinguishable
from type 4.

`drive_logs.py` rejects a raw device clock that runs backwards, so a device
reset closes the current file pair and continues in a new one. Parser error,
reset and gap counters, queue high-water marks and missed periods go in
`session_<stamp>.json`. So do source/executable hashes, profile and calibration
bytes, physical sensor mapping, firmware range metadata, waveform
version/parameters and the operator suffix.

## Operator workflow

Same button map as `simple_drive.py`:

| Input | Behavior |
| --- | --- |
| Sticks | Drive valves directly, at all times |
| A | Start a strip; press again to stop and save |
| B | Toggle excitation (enters through the one-second taper; off is immediate) |
| X | Toggle pump |
| Y | Reload servo config from disk; only between strips, so a strip has one calibration |
| D-pad up/down | Cycle excitation target |
| Ctrl+C or strip timeout | Save the strip, write neutral, exit (Ctrl+C) or wait for A (timeout) |

A disconnected pad reads as centered sticks and disables excitation. Stopping a
strip disables excitation, as in `simple_drive.py`.

## Excitation

Port sine, logarithmic chirp and stroke-plus-chirp from `simple_drive.py`.
Preserve target/channel names, tapering, clipping diagnostics, neutral
intervals and seed/block metadata. Stroke uses fresh joint feedback and the
profile's joint limits with the existing stroke margin. While feedback is
stale, the stroke contribution is held at zero; this is needed for correct
waveforms, not a safety gate. Slew excitation stays explicit, and tracks are
excluded from automatic excitation.

Generate noise on the fixed 100 Hz grid so that loop jitter does not change the
waveform. NumPy's random stream must not be presumed identical to a C++ RNG.
Freeze cross-language reference fixtures for analytic chirp/stroke behavior.
Give a changed random implementation a new generator version, with its
algorithm, drawn parameters and seed recorded. Compare waveforms with numerical
tolerances rather than claiming identical samples from the seed alone.

## Training compatibility

Rows are already on the 100 Hz grid and `combined_cmd_*` is the written
command, so `training.drive_logs prepare` reads native strips as it reads
`simple_drive.py` strips. The one loader change: when a `pump_on` column is
present, rows with the pump off are not live. That excludes intervals where
the valves were commanded but no oil flowed.

At each training sample the loader selects the last row at or before the tick
and the exact packet named by `state_imu_ts_us`. State and gyro come from that
same packet. Never align by row number, select a future packet or interpolate
across a reset/gap. Default velocity labels remain causal gyro values. Keep
`smooth21` only as an explicit comparison option, because it attenuates the
fast dynamics this collection is intended to capture. Validate the four sensor
roles and axis assumptions against `SENSOR_CONTRACT`, and reject unsupported
mounting calibration.

Use complete sessions for train/development/test separation. Preserve source
and excitation labels in split manifests so that manual and excited motion can
be scored separately.

## Implementation order

| Step | Deliverable and completion check |
| --- | --- |
| 1 | Per-packet emission from `ImuParser`; move `Pwm` and the gamepad reader into shared modules (`kaivuri-control` behavior unchanged). Add an IMU byte-replay input, such as a pty or file source with configurable read chunking, through the same `read()`/`feed()` path. Test bursts, 32-bit wrap, resets and invalid packets. |
| 2 | `kaivuri-mlp-collect`: 100 Hz loop, manual driving, pump toggle, writer thread, CSV trio, provenance and reset splitting. A replayed IMU stream must appear in `imu_raw` packet for packet, with one drive-log row per tick. |
| 3 | Port sine, chirp and stroke with reference fixtures. Verify waveforms against the Python implementation without robot motion. |
| 4 | `pump_on` handling in `Isaac-hydraulic-actuator/training/drive_logs.py` with a test. Run `prepare` on a replayed strip and check q/v/u against the known motion. |
| 5 | Bench acceptance below on `pikkumasi2tb` with servo power off, then a short powered run with the same command. Write the MLP runbook. |

`--simulate` in the control service skips the serial read and the parser and
reports identity quaternions. It exercises neither raw capture nor stroke
feedback, so use the byte replay or `pico_imu_simulator` instead.

## Acceptance

**Data:** a replayed 200 Hz stream with multi-packet reads, timestamp wrap, a
device reset, invalid packets and deliberate gaps. Every accepted packet
appears in `imu_raw`. Resets split the file pair. Drive-log rows are
consecutive, and `training.drive_logs` recovers the expected q/v/u.

**Timing and resources:** a 30-minute bench run with real IMU, live PCA writes
and servo power off. Report tick lateness and work time (p50/p99/p99.9/max),
missed periods, serial gaps and queue high-water marks. Target: no missed
packets, no missed periods in steady operation, and flat memory over the run.

**Lifecycle:** repeated start/stop, pad disconnect, Ctrl+C mid-strip and a full
disk all leave either a completed strip or a clearly incomplete `.partial`, and
exit with neutral written.

## Deferred

IK/PID driving, the old PPO/MLP actor, circle trajectories, controller
comparisons and model training are not part of this milestone. A later trained
controller can run as another command source inside this same loop, so it
drives through the same calibration, PWM writes and logging.

Relevant references: [control protocol](PROTOCOL.md),
[Python excitation reference](../../reference/data_collection/EXCITATION.md),
[shared legacy log schema](../../reference/modules/drive_log.py),
[legacy collector](../../reference/simple_drive.py), and
[current actuator importer](../../../Isaac-hydraulic-actuator/training/drive_logs.py).
