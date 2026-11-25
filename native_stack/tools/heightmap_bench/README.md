# Offline excavator heightmaps

This exporter reads committed native episodes, including resumed recording runs, and writes two LeRobot 0.6.1 datasets. It uses the robot checkout's `sand_height.py`, `native_sessions.py`, `export_episodes.py` and arm mesh distance grid. Run it with those tools on `PYTHONPATH`. Calibration needs NumPy and SciPy. Replay needs LeRobot with its dataset dependencies, Pillow and PyAV. The attached C++ adapter uses the same FastDEM core as the ROS workspace.

Build the adapter after running the repo's ROS workspace setup:

```bash
f=ros2/ws/src/FastDEM/fastdem
b=ros2/ws/build/fastdem
g++ -O3 -std=c++17 -I/usr/include/eigen3 -I"$f/include" \
  -I"$f/lib/nanoPCL/include" -I"$b/_deps/nanogrid-src/include" \
  offline_heightmaps/fastdem_replay.cpp "$b/libfastdem.a" \
  "$b/_deps/nanogrid-build/libnanoGrid.a" -lyaml-cpp -lspdlog -lfmt \
  -o /tmp/fastdem_replay
```

Fit the mounts using a flat arm-motion take and a separate holdout. The prior supplies the camera parameters the flat take cannot identify. The depth-fit file is the output of the native `sand_height.py calibrate` command for the same cameras and depth settings.

```bash
export PYTHONPATH="$NATIVE_STACK/tools"
OPENBLAS_NUM_THREADS=1 python offline_heightmaps/calibrate.py "$SESSION" \
  --prior "$MOUNT_PRIOR" --depth-fit "$DEPTH_FIT" --out /tmp/heightmap-calibration.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python offline_heightmaps/replay.py "$SESSION" \
  --calibration /tmp/heightmap-calibration.json --arm-model "$NATIVE_STACK/config/arm_model.npz" \
  --fastdem /tmp/fastdem_replay --config ros2/fastdem.yaml --out "$OUTPUT"
python offline_heightmaps/verify.py "$OUTPUT" --counts 341 1194 3061
```

Both variants use a fixed 64×64 forward observation and a fixed 160×160 surround observation. The forward grid follows the cabin yaw and uses 12.5 mm cells. The surround grid stays aligned to the undercarriage and uses 10 mm cells. Rows run from positive to negative X, columns from positive to negative Y. Heights are measured relative to flat ground. The images map ±120 mm to grayscale and reserve black for unknown. The surround map covers the annulus from 0.14 m through 0.80 m with no inpainting, and persists across the takes of one recorder start, as live. Offline replay only sees the saved takes, so driving between takes is missing from replayed maps.

The forward map uses current observations. The surround map uses FastDEM's Kalman estimator and raycasting with camera origins from timestamp-matched FK. The fused variant gives the cabin first priority and fills unobserved cells with the arm camera. Both variants store identical state and actions. The separate side-by-side episode videos include both diagnostic depth views. Red indicates FK-predicted boom, arm or bucket returns excluded from mapping.

For quantitative review, the `heightmaps/` sidecars contain signed integer millimetres with `-32768` for unknown. Surround age is serialized as frames since the last observation and uses `65535` for unknown. The robot's depth timestamps and source frame indices are in the LeRobot parquet data.

## Add the views to a full native export

First run the native stack's `tools/export_episodes.py` on the same session with
`--state-joints slew,lift,tilt,scoop`. Use the resulting LeRobot root as the first
argument, and the matching map-only variant as the second. The dataset keeps the
native images, state, actions, FK, IMU, timing, and the raw depth and controller
records under `native/`. The recorded camera streams are monochrome IR in the
three-channel LeRobot image fields.

```bash
python offline_heightmaps/add_to_native.py "$NATIVE_EXPORT" "$OUTPUT/cabin-only" "$FULL_OUTPUT/cabin-only"
python offline_heightmaps/add_to_native.py "$NATIVE_EXPORT" "$OUTPUT/cabin-arm-fused" "$FULL_OUTPUT/cabin-arm-fused"
```

The script checks frame, episode and source alignment before adding the two map
views, their diagnostics and numeric sidecars. Unchanged native files are
hardlinked on the same filesystem. Keep the map-only review videos at the
parent of the two full dataset roots to run the verifier with `--full`.

## Live throughput benchmark on pikkumasi2tb

The sidecar in `~/kp-depth-close/native_stack/tools/heightmap_bench` runs the
existing native camera publisher and a private, dry 200 Hz controller with real
IMU/FK. It reads the latest native shared-memory depth frames and matches each
camera independently to timestamped gravity-aligned FK. It skips duplicate or
stale input and keeps no processing backlog. The mapper returns a 64×64 float
front grid, a 160×160 float FastDEM surround grid and a surround age grid in
map updates. `mapper.py` is shared with offline replay. The diagnostic files
are outside the camera publisher and control loop.

Run this while the camera and IMU are free. Start with `--mode off` to measure
inference under the same dual IR + depth capture load, then use `--mode fused`
with a new output directory. Use `--mode cabin` to measure cabin-only mapping.
Omit the four model arguments for a mapping-only run. Each mode keeps both IR
streams configured for 640×480 at 30 FPS, with the recording’s 100 µm depth
units and disparity shifts of 25 on the arm and 0 on the cabin. The default
live map schedule is capped at 5 Hz by default. Use `--map-hz 6` to repeat the throughput test.

```bash
cd ~/kp-depth-close/native_stack
export PYTHONPATH=tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
h=tools/heightmap_bench
b="$HOME/bundles/pi05-dig10k-rowtime"
python3 "$h/bench_live.py" --native . --out "$HOME/heightmap-bench-$(date +%s)" \
  --mode fused --seconds 120 --map-hz 5 --policy-hz 2 \
  --calibration "$h/calibration.json" --fastdem "$h/fastdem_replay" --config "$h/fastdem.yaml" \
  --inference "$HOME/kp-pi05-rtc/native_stack/build/kaivuri-inference" \
  --bundle "$b" --cache "$b.cache" --fixtures "$b-fixtures"
```

The private controller stays disarmed with dry PWM and pump off. The model
consumes live IR and joint state and computes actions with the native inference
bench. The cached checkpoint is an IR/state model. The maps run concurrently
and are available as arrays for a future observation adapter. The benchmark
does not include training or running a policy conditioned on maps.

`summary.json` evaluates both camera rates (at least 29.5 FPS against the 30 FPS
target) and completed maps (at least 5 Hz). Inspect camera counter gaps,
skipped input, longest map gap, depth age, control lateness, model latency and
the resource logs before choosing a production cadence. `updates.jsonl`
contains per-update logs, with the final maps in `latest.npz` and PNG previews.
With inference enabled, the measured window starts after native parity and
warm-up. Camera, control, inference and map statistics use that common window.


### Compare live and recorded depth

The original `tools/record_native.py` sessions, including resumed takes, are
inputs to `heightmap_bench/replay.py`. Record with that tool as usual. To
reproduce the flat calibration setup, use `--depth-units-um 100
--cam1-disparity-shift 25 --cam2-disparity-shift 0` if those options are not
already part of your recording command. Replay reads the actual depth metadata
from each run.

Both tools accept `--stride` (default 4), `--arm-model`, and
`--arm-margins-m BOOM ARM BUCKET` (defaults 0.045, 0.045, 0.100 m).
`--arm-model none --arm-radius-m 0.11` uses FK capsules for comparison.
Pass the same calibration and FastDEM YAML to both. Offline replay follows the
live cadence by default: it maps the newest frame every 1/map_hz (5 Hz from the
pipeline config), skips an update when FK is older than max_fk_age_ms, and holds
the last maps in between. `--map-hz 0` maps every recorded frame. FastDEM age counts accepted updates in each case. The review video shows
the arm mask over recorded depth. Live `latest.npz` contains the same strided
depth and arm masks (0 missing, 1 depth return, 2 rejected by FK) alongside the
final maps, and `updates.jsonl` reports each camera’s rejection rate.

```bash
cd ~/kp-depth-close/native_stack
export PYTHONPATH=tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
h=tools/heightmap_bench
"$HOME/kaivuriprokkis/.venv-lerobot061/bin/python" "$h/replay.py" \
  "$HOME/data_collection/native_sessions/flat-calib-texture-001" --episode 2 \
  --calibration "$h/calibration.json" --fastdem "$h/fastdem_replay" --config "$h/fastdem.yaml" \
  --arm-model config/arm_model.npz --arm-margins-m .045 .045 .100 \
  --out "$HOME/heightmap-replay-$(date +%s)"
```

### Configuration and live log

`native_stack/config/heightmap_pipeline.json` owns the camera capture settings,
map and policy cadence, FK matching limit and projection defaults. Both tools
read it automatically on the robot. Use `--pipeline-config PATH` for a separate
test profile. Explicit CLI values override the profile. Keep hardware geometry
in the Jetson control profile and the elevation estimator settings in
`fastdem.yaml`.

While the benchmark runs, open a second terminal and follow the output
directory you passed to `--out`:

```bash
tail -n +1 -F "$RUN/updates.jsonl" "$RUN/skips.jsonl"
```

Each accepted update logs camera depth and FK source timestamps, slew and
joint angles, cabin tilt, per-camera arm rejection, map coverage and timing.
Skipped inputs log the reason. The final NPZ contains both depth masks and
both maps. For a movement check, use native recordings for the full sequence
of depth, FK and masks. Compare arm sweeps and slew across the same patch of
flat ground to see residual arm hits, over-filtering and map alignment.

### Record heightmaps in place of raw depth

On pikkumasi2tb, use the native recorder and add `--depth heightmaps` to your
usual gamepad, session, task and motion options. The mapper reads both Z16
streams from shared memory, saves a fresh fused front and surround at up to
5 Hz. FastDEM starts with the recorder and keeps mapping between takes, so
drive around once (e.g. a few slews) before the first take and every take
starts on that map; a new recorder start begins an empty map. Both 30 Hz IR
streams, FK, IMU, timing and action records remain in the take. The mode uses
the calibrated capture and mask settings in `config/heightmap_pipeline.json`. Use the usual raw-depth mode when collecting
data expressly for recalibration.

Each saved episode has `heightmaps.npz` with signed int16 millimetres and
`-32768` for unknown. Per update it stores `front_mm` (64×64),
`surround_mm` (160×160), `publish_host_ns`, both `depth_host_ns` values and
both `depth_sequence` values, in cabin/arm order. Use the latest
`publish_host_ns` at or before the observation to align it causally with IR.
The native LeRobot exporter preserves these 5 Hz arrays at
`native/episode-NNNNNN/heightmaps.npz` alongside the regular IR videos and
telemetry.

For a stationary, dry preflight with real cameras and IMUs:

```bash
cd ~/kp-depth-close/native_stack
python3 tools/record_native.py --session "$HOME/data_collection/hm-preflight-$(date +%s)" \
  --task "heightmap preflight" --depth heightmaps --dry --enable-slew --bench-seconds 15
```
