# Native dual-IR episode recording

`kaivuri-record` is a C++17 teleoperation and episode capture process. The existing
C++ controller runs at 200 Hz by default through the launcher. The recorder polls
the local Xbox evdev device and full controller status at 100 Hz; the independent
camera service captures packed Y8 IR at 30 FPS, plus same-sensor Z16 depth unless
`--depth none`. A bounded writer queue streams pixels, zstd-compressed depth and JSONL
snapshots to disk. Disk writes, video encoding, LeRobot, Torch and
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
RGB stream is opened. Export repeats gray into three identical channels
for the existing LeRobot image convention, then encodes H.264. Raw Y8 is retained
unchanged. Physical cameras are USB-only/free-running: timestamp domain, validity
bits, hardware sensor/readout timestamps, host arrivals, exposure/gain, counters,
and signed pair skew are logged. Host arrival pairing is **not** synchronized
exposure. Default maximum arrival skew is 40 ms and maximum frame age is 200 ms.

### Depth

`--depth both` (default; or `cam1`, `cam2`, `none`) adds Z16 depth from each
camera's stereo ASIC. It costs the host no depth computation and no GPU. IR and depth
come from one sensor, so they share resolution and FPS; the D435i silently stops
IR when they differ. Depth uses the same auto exposure with the emitter off, so
the recorded IR is unchanged. Passive stereo filled about 52–58% of all pixels on
the bench scene, and the textured sand nearly completely.
With the emitter at its default 150 mW at these ranges, IR saturated and fill dropped.

Every depth frame is paired with the IR frame of **the same exposure** by
`sensor_timestamp_us`: the two streams share it exactly. Frame counters do not pair
them on firmware 5.17.3.10: depth reads +1 against IR1 and −1 against IR2 for the
same exposure. Depth arrives within about 1 ms of its IR frame. The recorder holds
a sample for a later 100 Hz tick until its depth is present. An unpaired depth stream
fails the take like a stalled camera.

Depth is always in the **IR1 (left imager) frame**. `cam2` records IR1, so they
overlay. `cam1` records IR2, so its depth is offset by the 50 mm stereo baseline.
`camN.depth.json` holds depth and IR intrinsics, `depth_to_ir_extrinsics`
(translation `[-0.0499, 0, 0]` m for cam1) and `depth_scale_m` (0.001). These are SDK values for
unrotated pixels, and `rotation_deg` says when stored pixels (IR and depth alike)
are rotated 180°.

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
- `cam1.depth.zst` / `cam2.depth.zst` (depth cameras only): one independent zstd
  level-1 frame of packed little-endian uint16 per committed frame. They are
  self-delimiting, so no index is needed. `camN_depth` metadata sits on the same
  `samples.jsonl` row (`ir_index` 0);
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
100 Hz JSONL. Depth adds about **4 MB/s** for both cameras on the bench scene: per frame
cam2 compresses 6.2x to 100 kB and cam1 16.9x to 36 kB, which is about 119 MB per
30 s take. Moving sand will likely compress less. Raw Z16 would add 36.9 MB/s.
Compression runs on the writer thread: the recorder used 25.8% of one core instead of
14.3%, and the camera 7.4% instead of 3.5%. RSS grew by about 5 MB and 19 MB. Space does not grow in RAM with episode length: default queue depth
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

**Open issue (2026-10-20): startup frames can fail export timing validation.**
In `sand-depth-001`, `episode-332085182766231.ready` skipped one cam1 frame
between its first two saved frame pairs, creating a 66.7 ms interval at 30 FPS
(the export limit is 50 ms). Its saved IR/depth files are intact, but the current
validator rejects the take. `episode-332144087887833.ready` passes validation.
Check frame pairing at the start of each take and whether a short pairing warmup
is needed, using more recordings. This remains unresolved; no warmup fix has
been validated. Run `tools/export_episodes.py SESSION --validate-only` from
`native_stack` before export, and preserve the raw recordings while investigating.

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

Depth is validated (geometry, same-exposure pairing, every zstd frame decodes to
the IR size, one per sample) but **is not a LeRobot feature**: H.264 would quantize
it. `camN.depth.zst` files are copied unchanged to `native/episode-NNNNNN/`, with
`camN.depth.json` in `native/session-NNN/`. Read them with
`export_episodes.depth_frames(path)`. It uses libzstd via ctypes, so Python needs no
extra package:

```python
import numpy as np
from export_episodes import depth_frames
for z in depth_frames('native/episode-000000/cam2.depth.zst'):
    depth_m = np.frombuffer(z, '<u2').reshape(480, 640) * 0.001  # 0 = no depth
```

To view a raw session with depth, use Rerun (`.venv-rerun`, rerun-sdk 0.26.2). It shows each
camera's IR, colormapped depth and a 3D point cloud coloured by that camera's IR
image; cam1 uses the depth-to-IR2 extrinsics. In the 3D view, double-click the
cloud to frame it.

```bash
.venv-rerun/bin/python native_stack/tools/view_native.py SESSION --save take.rrd [--max-fps 30]
.venv-rerun/bin/rerun take.rrd
```

The output is LeRobot v3 Parquet + two H.264 videos, plus the complete 100 Hz JSONL
and provenance under `native/` (`session-NNN/` per source session, `episode-NNNNNN/` per
episode). `native/EXPORT_COMPLETE` is written only after successful dataset
finalization. Export always creates a fresh dataset; it does not append to an
existing one. Keep the original IR+RGB
Python recorder for old datasets and workflows that need its resume behavior.
