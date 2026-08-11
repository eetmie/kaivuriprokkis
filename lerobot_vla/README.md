# lerobot_vla — LeRobot integration for the MASI excavator

The VLA layer on top of the kaivuriprokkis control stack: LeRobot-format
dataset collection (for SmolVLA fine-tuning on the DGX Spark) and on-device
SmolVLA inference (split TensorRT engines on this Orin Nano).

```
gamepad ─┐                                             ┌─> LeRobot v3 dataset ──> DGX Spark finetune
         ├─> setpoint ─┐                               │      (lerobot 0.5.1, h264 video)
IR cam1 ─┼─────────────┼───────────────────────────────┤
IMUs ────┴─> ExcavatorController (joint angles) ───────┘
                       │
                       └─> control thread @100 Hz ──> valves   (direct-command mode)

split ONNX engines (vision/prefill/decode + projectors) ──> action chunk ──┘
```

Producers (gamepad at 30 Hz, policy at ~1 Hz chunks) only *store* a setpoint;
the 100 Hz control thread resamples it and writes the valves every tick. They
must not write the PWM layer directly — its dither, watchdog and rate gate all
advance per call. See `modules/setpoint_schedule.py` and `modules/pwm/README.md`.

## Environment

One venv for everything here, created with `--system-site-packages` so it sees
the system `pyrealsense2` and JetPack `tensorrt`:

```bash
cd ~/GitHub/kaivuriprokkis
.venv-lerobot/bin/python   # lerobot 0.5.1 (same pin as the Spark), torch-cpu,
                           # onnxruntime-gpu 1.24 (TRT+CUDA EPs), transformers
```

`lerobot==0.5.1` matches `spark-projects/smolvla-spark-finetune`, so datasets
recorded here load directly in `lerobot-train` on the Spark. Videos are
encoded **h264** (not the AV1 default) — the Spark side needed an H.264
transcode to read AV1 reliably.

## Robot definition

`excavator_robot.MasiExcavator` — the LeRobot-style robot:

| key | shape | meaning |
|---|---|---|
| `observation.state` | float32[4] | joint angles [slew, lift, tilt, scoop], degrees |
| `observation.images.cam1` | uint8 480×640×3 | D435i **infrared left imager**, laser emitter DISABLED, gray→3ch |
| `action` | float32[4] | normalized valve commands [-1, 1], [slew, lift, tilt, scoop] |

Actions drive the valves open-loop (no IK/PID in the loop) via
`ExcavatorController.enter_direct_command_mode()`. `send_action()` stores a
setpoint and returns immediately; `send_action_chunk()` hands over a whole
policy chunk to be played as a trajectory. `--legacy-direct-write` restores the
old `DirectController` path for bench comparison only.

## 1. Dataset collection (gamepad teleop)

```bash
.venv-lerobot/bin/python -m lerobot_vla.record_episodes \
    --task "scoop sand and dump it to the left" \
    --repo-id masi/excavator_sand_v0 --exposure-us 16000
```

**Lock the exposure.** Auto-exposure drifts with the scene, which the policy
then has to learn around. Find a value with the sweep tool (headless: writes
PNGs + clip stats to inspect from another machine), then pass the same
`--exposure-us` to BOTH record_episodes and run_inference:

```bash
.venv-lerobot/bin/python -m lerobot_vla.tune_exposure --out /tmp/ir_sweep
```

Target mean ~80–130 with clip-hi under ~1%; at 30 fps the ceiling is
~33 000 µs. With the work lights on, **16 000 µs @ gain 16** measured right
(2026-08-10); re-sweep if the lighting setup changes.

Sticks = same mapping as simple_drive.py (left = slew/tilt, right = lift/scoop).
Buttons: **A** start / stop+save episode · **B** discard episode · **X** pump ·
**Y** reload servo config. Episodes auto-save at `--max-episode-s` (120 s).
Default output root: `data_collection/lerobot_datasets/<repo_id>/`.
Use `--resume` to append episodes to an existing dataset.

Record one task phrasing per dataset run (`--task` is stored per frame); for
the left/right sand task, either record separate sessions per direction with
matching instructions, or keep one canonical phrasing throughout.

Check a dataset quickly:

```bash
.venv-lerobot/bin/python - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("masi/kaivuri_juusto",
                    root="data_collection/lerobot_datasets/masi_kaivuri_juusto")
print(ds.meta.total_episodes, "episodes,", ds.meta.total_frames, "frames")
print(ds[0]["observation.state"], ds[0]["task"])
EOF
```

Inspect episodes visually from another machine (the Jetson is headless). Run on
the Jetson:

```bash
.venv-lerobot/bin/lerobot-dataset-viz \
    --repo-id masi/kaivuri_juusto \
    --root data_collection/lerobot_datasets/masi_kaivuri_juusto \
    --episode-index 0 --mode distant
```

It serves two ports, both on 0.0.0.0: the web viewer on 9090 and the gRPC data
stream on 9876. **Both must be reachable** from the viewing machine — the page
is only the app, the data arrives over 9876.

Opening bare `http://<jetson-ip>:9090` gives an empty welcome screen. The viewer
takes its source from a `?url=` query parameter, and lerobot calls rerun's
`serve_web_viewer(open_browser=False, ...)`, whose `connect_to` argument is
applied *only* when it opens the browser itself. So pass the stream URL yourself:

```
http://<jetson-ip>:9090/?url=rerun%2Bhttp://<jetson-ip>:9876/proxy
```

The `%2B` matters: a literal `+` in a query string decodes to a space.

With the Rerun desktop app installed instead, skip the web viewer entirely:

```bash
rerun rerun+http://<jetson-ip>:9876/proxy
```

Over SSH without LAN access, forward both ports and use `localhost`:

```bash
ssh -L 9090:localhost:9090 -L 9876:localhost:9876 joel@<jetson-ip>
# then http://localhost:9090/?url=rerun%2Bhttp://localhost:9876/proxy
```

(Alternatively the mp4s under `videos/` play in VLC directly, or `--save 1`
writes an .rrd for the Rerun desktop app.)

Then rsync the dataset directory to the Spark and point `lerobot-train` at it
(`--dataset.repo_id=masi/kaivuri_juusto --dataset.root=...`).

## 2. Inference (split TensorRT engines)

The monolithic SmolVLA ONNX cannot TRT-build on 8 GB; the deploy path is the
9-graph split with the flow-matching denoise loop in Python
(`smolvla_split.py`; see `spark-projects/orin-nano/smolvla-runtime/notes/findings.md`).

```bash
# model-only proof (no robot, no camera; first run builds engines — minutes):
TRT_DROP_CUDA_EP=1 .venv-lerobot/bin/python -m lerobot_vla.run_inference --synthetic --loops 3

# real observations, actions PRINTED only (safe default):
.venv-lerobot/bin/python -m lerobot_vla.run_inference \
    --instruction "scoop sand and dump it to the left" --exposure-us 16000

# actually drive the valves (only with a finetuned checkpoint!):
.venv-lerobot/bin/python -m lerobot_vla.run_inference --live \
    --dataset-stats <dataset>/meta/stats.json ...
```

Defaults point at the base-weight split export
(`spark-projects/.../exports/ainekko_base_split`) + its tokenizer bundle.
When the finetuned split export lands from the Spark, pass `--split-dir` at it
and `--dataset-stats` at the training dataset's stats for correct
state/action normalization. Base weights produce meaningless actions — they
only prove the loop and its latency.

Engines are pre-built automatically, one subprocess per graph (two TRT builds
in one process OOM 8 GB), into `/tmp/smolvla_split_cache` — note /tmp clears
on reboot, so the first run after boot rebuilds (~5 min). RAM is tight:
run ONE thing at a time (never viz or recording alongside an engine build).

**fps vs inference rate:** the policy is chunked — one ~220 ms inference emits
50 actions authored at the dataset fps (30 Hz). The chunk is handed to the
control thread, which interpolates it by elapsed time and drives the valves at
100 Hz; inference being far slower than the valve rate is exactly what that
split is for. The camera is only *read* at each re-plan; `--n-action-steps`
(default 25 ≈ 0.8 s) sets how often the robot re-looks, so ~4.5 Hz inference
capability is plenty. Re-inference starts early by the measured inference time
so the schedule never runs dry; if it does, the held command decays to zero
(`--setpoint-hold-s`, `--setpoint-decay-s`) rather than latching.

## Files

```
excavator_robot.py   MasiExcavator: control stack + IR camera as one robot
ir_camera.py         D435i IR-left reader (Y8@30, emitter off, gray→3ch)
record_episodes.py   gamepad teleop -> LeRobot v3 episodes
smolvla_split.py     9-graph split policy: ORT/TRT sessions + denoise loop
run_inference.py     obs -> action-chunk -> valves loop (synthetic/dry/live)
```
