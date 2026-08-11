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
    --repo-id masi/<new_dataset_name> --exposure-us 16000
```

`--repo-id` names a **new** dataset; recording refuses to start if its folder
already exists (use `--resume` to append). The folder is the repo-id with `/`
replaced by `_`, under `data_collection/lerobot_datasets/`. Existing datasets:
`masi/kaivuri_juusto` (31 episodes, blocks task).

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

The viz server loads **one episode per invocation** (`--episode-index`);
restart it to look at another. It holds the decoded episode in RAM, so don't
run it alongside a TRT engine build on this 8 GB board.

### Getting the dataset to the training machine

The viz URL is **not** a data source — it streams rendered frames to a Rerun
viewer, and nothing on the other end can train from it. `lerobot-train` needs
the actual dataset tree (parquet + mp4 + meta). Copy it:

```bash
# 67 MB for 31 episodes, ~66 MB of that video
rsync -avP data_collection/lerobot_datasets/masi_kaivuri_juusto/ \
    <spark-host>:~/datasets/masi_kaivuri_juusto/

# on the Spark
lerobot-train --dataset.repo_id=masi/kaivuri_juusto \
              --dataset.root=~/datasets/masi_kaivuri_juusto ...
```

`--dataset.root` is what makes the repo-id a local lookup; without it lerobot
searches `$HF_LEROBOT_HOME` and then the Hub.

Via the Hub instead, if you'd rather not copy by hand:

```bash
.venv-lerobot/bin/python - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("masi/kaivuri_juusto",
                    root="data_collection/lerobot_datasets/masi_kaivuri_juusto")
ds.push_to_hub(private=True)     # needs `huggingface-cli login`
EOF
```

Then the Spark can train straight from `--dataset.repo_id=masi/kaivuri_juusto`
with no `--dataset.root`.

### Sanity-check a dataset before training

Beyond frame counts, the check worth running is whether the recorded actions
actually explain the recorded motion — it validates the whole capture path in
one number:

```bash
.venv-lerobot/bin/python - <<'EOF'
import pandas as pd, numpy as np
D="data_collection/lerobot_datasets/masi_kaivuri_juusto"
df=pd.read_parquet(f"{D}/data/chunk-000/file-000.parquet")
A=np.stack(df["action"].values); S=np.stack(df["observation.state"].values)
ep=df.episode_index.values
for j,n in enumerate(["slew","lift","tilt","scoop"]):
    best=max(((np.corrcoef(
        np.concatenate([A[ep==e][:-l or None,j] for e in np.unique(ep)]),
        np.concatenate([(np.gradient(S[ep==e][:,j])*30)[l:] for e in np.unique(ep)])
    )[0,1], l) for l in range(1,9)), key=lambda t: abs(t[0]))
    print(f"{n:6} r={best[0]:+.3f} at {best[1]} frames ({best[1]/30*1000:.0f} ms)")
EOF
```

Healthy looks like `r = 0.75..0.96` at a 67–200 ms lag (the hydraulic response
delay). Much lower means commands are not reaching the valves — which is what
the pre-`b246c69` 30 Hz direct-write path did, silently dropping ~67% of them.

Worth also checking action saturation per joint: on `masi/kaivuri_juusto`, lift
sits at −1.0 for 17.5% of frames while its positive side never exceeds +0.77.
That is real operator asymmetry (boom slammed down, raised gently), not a
broken axis, but it skews the normalization stats the policy is trained with.

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

Setpoint-timing flags (all optional; defaults are sane):

| Flag | Default | Effect |
|---|---|---|
| `--setpoint-hold-s` | 0.25 | How long the chunk's last action keeps full authority after the chunk runs out |
| `--setpoint-decay-s` | 0.25 | Ramp-to-zero window once the setpoint goes stale |
| `--blend-s` | 0.0 | Cross-fade into each new chunk, to soften the step at a chunk boundary |
| `--legacy-direct-write` | off | Drive valves from this thread instead of the control thread (bench A/B only) |

Each cycle logs `age=` and `decay=` for the held setpoint. `decay < 1.00` in
steady state means inference is not keeping ahead of chunk playback — raise
`--n-action-steps` (longer chunks, more time to think) before touching the
hold/decay windows, which exist to make a stall safe, not to hide one.

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
