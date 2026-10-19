# Native dual-IR episode recording

`kaivuri-record` is a C++17 teleoperation and episode capture process. The existing
C++ controller runs at 200 Hz by default through the launcher. The recorder polls
the local Xbox evdev device and full controller status at 100 Hz; the independent
camera service captures packed Y8 IR at 30 FPS. A bounded writer queue streams
pixels and JSONL snapshots to disk. Disk writes, video encoding, LeRobot, Torch and
Python callbacks do not run on the controller thread.

The Python launcher only starts/stops processes and saves provenance. It neither
polls hardware nor publishes control commands. Python/LeRobot conversion runs
**after the recording session and controller have stopped**, on the board or a
workstation with LeRobot 0.5.1. This is deliberately a two-step workflow. Episodes
are immediately saved to the native spool; LeRobot video encoding is offline.

## Camera contract

| Dataset view | Physical serial | IR imager |
| --- | --- | --- |
| `observation.images.cam1` | `233522073919` | IR2, closer to the USB connector; mounted upside down, captured rotated 180° (`--rotate-camera1`, default on) |
| `observation.images.cam2` | `141722076647` | IR1, away from the USB connector |

This order is explicit and saved with the session. The emitter is disabled; no
RGB or depth stream is opened. Export repeats gray into three identical channels
for the existing LeRobot image convention, then encodes H.264. Raw Y8 is retained
unchanged. Physical cameras are USB-only/free-running: timestamp domain, validity
bits, hardware sensor/readout timestamps, host arrivals, exposure/gain, counters,
and signed pair skew are logged. Host arrival pairing is **not** synchronized
exposure. Default maximum arrival skew is 40 ms and maximum frame age is 200 ms.

## Build and dry validation on the Jetson

```bash
cd ~/kaivuriprokkis-recorder/native_stack
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/home/joel/kaivuriprokkis/native_stack/.deps/realsense-2.58.4
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure

# Synthetic cameras/IMUs; cannot open physical PWM.
python3 tools/record_native.py \
  --session results/new-synthetic-session --task "recorder verification" \
  --synthetic --bench-seconds 3

# Real IR1 + IR2 and real IMUs; zero commands, no I2C.
python3 tools/record_native.py \
  --session results/new-real-session --task "dual IR verification" \
  --bench-seconds 30 --dry
python3 tools/export_episodes.py results/new-real-session --validate-only
```

The launcher owns all three processes, uses a fresh private runtime directory,
and shuts the recorder down before the controller/cameras. Bench sessions also
save `control-timing.csv`: every 200 Hz tick's scheduling, work and FK work,
including startup and shutdown. It is not needed in the timed recording loop.

## Operator recording

Connect the Xbox gamepad and identify its evdev event path (`/dev/input/by-id/...
-event-joystick`, or the corresponding `/dev/input/eventN`). No gamepad was plugged
in during the initial hardware capture validation; check the joystick mapping
with servo power off first. Signs, 25% deadzone, trigger range
probing and bumper track reversal match the existing Linux Python recorder.

```bash
python3 tools/record_native.py \
  --session /home/joel/data_collection/native_sessions/session-001 \
  --task "scoop sand and dump it to the left" \
  --gamepad /dev/input/by-id/YOUR_CONTROLLER-event-joystick \
  --enable-slew
```

PWM output is live by default; the servo/pump power switch is the cut-off. Add
`--dry` to run without opening I2C (`--synthetic` is always dry). The old
`--live-pwm` flag is still accepted and does nothing. Tracks require `--enable-tracks`
on the launcher; they are disabled and exported actions stay four-dimensional by
default. The launcher forwards the same slew/track flags to both controller and
recorder; direct IPC use refuses mismatched flags.

The recorder arms the controller by itself as soon as the IMUs are healthy. It
sends the controller's 250 ms centered hold for you, whatever the sticks are doing,
and then the sticks drive at all times, inside or outside a take. After a
controller fault (IMU glitch, deadline miss, E-stop) it clears the fault and
re-arms automatically, at most once per second. A take that was running during the
fault is stopped and kept as `.partial`, which export skips; the session carries on.

| Input | Behavior |
| --- | --- |
| A | Start a take (once armed); while recording, save |
| B | Discard the current take |
| X | Toggle the pump |
| Y | Re-arm now (normally not needed) |
| Ctrl+C | Save a valid active take, disarm, shut down all services |

Saving or discarding a take leaves the controller armed and the pump as it was.
Configuration reload is not implemented by the native controller: stop and restart
after edits. Default maximum take length is 180 s, with automatic save.

## Test sessions without an operator

`--bench-seconds S` records automatically with zero commands. `--bench-episodes N`
records N takes in one session and `--bench-sine A` replaces the zero commands with a per-joint
sine of amplitude A, so a test dataset has non-constant actions. With real IMUs this
writes live PWM unless `--dry` is given. In automatic mode a controller fault still
ends the run with an error, so failures are visible in tests.

For the full teleop path, `tools/virtual_gamepad.py` plays an operator through a FIFO given to
`--gamepad`. This L4T kernel has no uinput, so `kaivuri-record` reads raw `input_event` structs
from a FIFO with xpad Xbox ranges; deadzone, signs, buttons, arming and pump all run
unchanged. `session.json` records `"virtual_gamepad": true`.

```bash
F=/tmp/kvr-pad.fifo
python3 tools/record_native.py --session results/vpad-test --task "scoop sand" \
  --gamepad $F --enable-slew [--dry] &
python3 tools/virtual_gamepad.py $F --episodes 5 --seconds 20 --discard-every 4 --stop-pid $!
```

## Native spool and additional data

Every new session contains `session.json` and calibration/source snapshots with
SHA-256 checksums under `provenance/`. There can be multiple saved/discarded takes
within one session. A new invocation requires a new session directory.

Each take starts as `episode-<monotonic_ns>.partial`. Its bounded writer streams:

- `cam1.y8` and `cam2.y8`: one packed grayscale frame per committed frame pair;
- `samples.jsonl`: full 100 Hz controller snapshots, requested action and command
  sequence, sampler lateness, and camera metadata/frame indices on image rows;
- `episode.json`: number of pairs, real start/end times and stop reason.

On save, files and directories are synced and the take is atomically renamed to
`.ready`. Discard deletes only that take. Queue exhaustion, write failure, camera
stall/stale/duplicate frames, excess pair skew, controller fault, IMU failure or
pad disconnect stops publication and disarms. An interrupted/failed take remains
`.partial` for diagnosis; export never treats it as a complete episode. Disk or
power failure can still interrupt a session; preserve raw files for recovery.

Default 640x480 dual Y8 storage is approximately **18.4 MB/s**, or **1.1 GB/min**, plus
100 Hz JSONL. Space does not grow in RAM with episode length: default queue depth
is 64 samples. A slow/full disk fails the take instead of silently dropping rows.
The shared-frame service remains latest-value; counters and timestamps reveal
camera skips before the recorder samples them.

Full snapshots include full-base and upper-carriage tool XYZ/quaternions, XZ,
bucket pitch/slew yaw, all joint pivots and local/link quaternions, firmware and
mounting-corrected IMU quaternions, gyro/acceleration, IMU device/arrival times,
health, actual PWM pulses, pump state, and controller timing/counters. The native
controller now reports `applied_action`, its command sequence and post-write host
time separately from newly received intent. LeRobot `action` uses this actually
applied normalized vector; `diagnostics.requested_action` preserves the joystick
request. Dither/deadband/ramp effects remain visible in `diagnostics.pulses_us`.

## Export

Use an environment with `lerobot[dataset]==0.6.1` (0.5.1 also works; both write
dataset format v3.0). On the Jetson it is `~/kaivuriprokkis/.venv-lerobot061`. The
recording runtime does not need it. On a workstation, copy the native sessions
first, keeping their layout intact.

```bash
.venv-lerobot061/bin/python native_stack/tools/export_episodes.py \
  /path/to/session-a /path/to/session-b \
  --root /path/to/new-lerobot-dataset --repo-id masi/excavator_dual_ir_v1
```

Several sessions go into one dataset in the order given, each episode keeping its
session's task. Sessions must share FPS, track layout and camera resolution.

Exporter validates every committed take **before creating** a dataset: counts,
packed pixel sizes, consistent resolution, healthy armed FK, applied action
range, strictly new camera frames, achieved capture FPS (within 3%), and absence
of master-camera intervals over 1.5 periods. Inaccurate timebases are rejected
instead of silently compressing motion into `frame_index/fps`. The real timing
columns are retained alongside LeRobot's nominal timestamps. Failed export leaves
raw data intact; choose a new destination when retrying an incomplete export.

State defaults to `[lift,tilt,scoop]` in degrees, preserving the legacy convention
and avoiding unanchored slew yaw. Use `--state-joints slew,lift,tilt,scoop` to include
it. FK stays in meters/radians and `[w,x,y,z]`; IMU gyro stays degrees/s and
acceleration g. Matrix fields flatten joint-major in `[slew,boom,arm,bucket]` order.
All FK/IMU features are exported under `feedback.fk.*` and `feedback.imu.*` (like the Python
recorder's `feedback.imu_quaternion`), not
`observation.*`: LeRobot types every non-image `observation.*` key as a STATE input
feature, so policies would pick them up silently. Derive a policy state from them
deliberately.
Camera/IMU host/device times remain separate `clock.*` columns. Do not mix them as
a common exposure clock without calibration.

The output is LeRobot v3 Parquet + two H.264 videos, plus the complete 100 Hz JSONL
and provenance under `native/` (`session-NNN/` per source session, `episode-NNNNNN/` per
episode). `native/EXPORT_COMPLETE` is written only after successful dataset
finalization. Export always creates a fresh dataset; it does not append to an
existing one. Keep the original IR+RGB
Python recorder for old datasets and workflows that need its resume behavior.
