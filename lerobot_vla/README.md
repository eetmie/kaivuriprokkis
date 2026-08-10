# lerobot_vla — LeRobot integration for the MASI excavator

The VLA layer on top of the kaivuriprokkis control stack: LeRobot-format
dataset collection (for SmolVLA fine-tuning on the DGX Spark) and on-device
SmolVLA inference (split TensorRT engines on this Orin Nano).

```
gamepad ─┐                                             ┌─> LeRobot v3 dataset ──> DGX Spark finetune
         ├─> DirectController (valves, open loop)      │      (lerobot 0.5.1, h264 video)
IR cam1 ─┼─────────────────────────────────────────────┤
IMUs ────┴─> ExcavatorController (joint angles) ───────┘

split ONNX engines (vision/prefill/decode + projectors) ──> action chunk ──> DirectController
```

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

Actions drive the valves open-loop through `DirectController` — exactly the
simple_drive.py surface, no IK/PID in the loop.

## 1. Dataset collection (gamepad teleop)

```bash
.venv-lerobot/bin/python -m lerobot_vla.record_episodes \
    --task "scoop sand and dump it to the left" \
    --repo-id masi/excavator_sand_v0
```

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
ds = LeRobotDataset("masi/excavator_sand_v0",
                    root="data_collection/lerobot_datasets/masi_excavator_sand_v0")
print(ds.meta.total_episodes, "episodes,", ds.meta.total_frames, "frames")
print(ds[0]["observation.state"], ds[0]["task"])
EOF
```

Then rsync the dataset directory to the Spark and point `lerobot-train` at it
(`--dataset.repo_id=masi/excavator_sand_v0 --dataset.root=...`).

## 2. Inference (split TensorRT engines)

The monolithic SmolVLA ONNX cannot TRT-build on 8 GB; the deploy path is the
9-graph split with the flow-matching denoise loop in Python
(`smolvla_split.py`; see `spark-projects/orin-nano/smolvla-runtime/notes/findings.md`).

```bash
# model-only proof (no robot, no camera; first run builds engines — minutes):
TRT_DROP_CUDA_EP=1 .venv-lerobot/bin/python -m lerobot_vla.run_inference --synthetic --loops 3

# real observations, actions PRINTED only (safe default):
.venv-lerobot/bin/python -m lerobot_vla.run_inference \
    --instruction "scoop sand and dump it to the left"

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

`TRT_DROP_CUDA_EP=1` keeps the one-time engine build inside 8 GB; once
`/tmp/smolvla_split_cache` holds the engines, run without it.

## Files

```
excavator_robot.py   MasiExcavator: control stack + IR camera as one robot
ir_camera.py         D435i IR-left reader (Y8@30, emitter off, gray→3ch)
record_episodes.py   gamepad teleop -> LeRobot v3 episodes
smolvla_split.py     9-graph split policy: ORT/TRT sessions + denoise loop
run_inference.py     obs -> action-chunk -> valves loop (synthetic/dry/live)
```
