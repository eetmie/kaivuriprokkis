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

Producers (gamepad at 30 Hz, policy at ~8 Hz chunks) only *store* a setpoint;
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
| `observation.state` | float32[3] | joint angles [lift, tilt, scoop], degrees. **Slew IMU feedback is dropped at the moment due to drift** — see below |
| `observation.images.cam1` | uint8 480×640×3 | D435i **infrared left imager**, laser emitter DISABLED, gray→3ch |
| `observation.images.cam2` | uint8 480×640×3 | D435i **color imager**, rgb8 |
| `action` | float32[4] | normalized valve commands [-1, 1], [slew, lift, tilt, scoop] |
| `clock.loop` | float64 | seconds since episode start, `perf_counter` |
| `clock.cam1_age` | float32 | seconds this cam1 frame had been sitting in the camera cache |
| `clock.cam2_age` | float32 | same for cam2; equal to cam1 when both come off one `wait_for_frames()` |
| `clock.state_age` | float32 | seconds since the IMU frame the joint angles were derived from |
| `clock.imu_us` | int64 | Pico device clock in microseconds (counts from board power-on) |

The `clock.*` columns are diagnostics, not observations — policies pick their
features by name and never see these. They exist because nothing else in the
dataset records when anything happened: lerobot's `timestamp` is
`frame_index / fps`, so it reads as a flawless 33.33 ms no matter what the loop
did, and both the camera and the 100 Hz control thread sit behind latest-value
caches that the 30 Hz record loop samples at its own rate. A repeated frame, a
late tick and a clean recording are indistinguishable without them.

Ages are NaN when a source has not reported yet; `clock.imu_us` uses -1 for the
same case, because int64 has no NaN and 0 is a real Pico timestamp. Measured on
the bench at 30 Hz: camera age ~5 ms, state age ~6 ms (max 13, so the control
thread does occasionally slip past its 10 ms period), device dt 26–41 ms.

**Slew IMU feedback is dropped at the moment due to drift.** Slew comes from
`average_z_yaw` over the IMUs, an absolute world yaw with no magnetometer to
anchor it and no zeroing anywhere in the stack — so the same physical pose can
read any value in ±180° after a power cycle, and a policy trained on one
session's origin gets fed angles it never saw. Within a session it is actually
stable (measured 2026-08-19: dig/dump centroids move only −1.1°/−1.5° over 31
episodes); the problem is the origin, not the noise. Episode-start zeroing does
not fix it either — slew at episode start has std 4.05° across those episodes,
a sixth of the ~24° working range.

The cameras observe slew directly, so little is lost. **Actions stay 4-dim: slew
is still commanded**, it just is not fed back. `--state-joints` on both
`record_episodes.py` and `run_inference.py` overrides this; `run_inference`
refuses to start when the joint count disagrees with `--dataset-stats`. Bring
slew back only with a real yaw correction (magnetometer, visual heading, or a
mechanical index).

Actions drive the valves open-loop (no IK/PID in the loop) via
`ExcavatorController.enter_direct_command_mode()`. `send_action()` stores a
setpoint and returns immediately; `send_action_chunk()` hands over a whole
policy chunk to be played as a trajectory. `run_inference` keeps a
`--legacy-direct-write` escape hatch onto the old `DirectController` path for
bench comparison; recording has no such flag — a dataset recorded through it
would carry the wrong action→motion mapping.

## 1. Dataset collection (gamepad teleop)

```bash
.venv-lerobot/bin/python -m lerobot_vla.record_episodes \
    --task "scoop sand and dump it to the left" \
    --repo-id masi/<new_dataset_name> \
    --exposure-ir 16000 --gain-ir 16 --exposure-rgb 16000 --gain-rgb 128
```

`--repo-id` is **required** — there is no default, so a session can never
silently land in a leftover dataset. It names a **new** dataset; recording
refuses to start if its folder already exists (use `--resume` to append). The
folder is the repo-id with `/` replaced by `_`, under
`data_collection/lerobot_datasets/`. Existing datasets: `masi/kaivuri_juusto`
(31 episodes, blocks task).

The repo-id is a *name*, not a destination: nothing is uploaded anywhere, and it
is not even written into the dataset (`meta/info.json` has no `repo_id` key). It
only becomes a real lookup on the training side when `--dataset.root` is omitted,
or if you explicitly call `push_to_hub` — see below.

A run that saves **no** episodes deletes its own dataset folder on exit, so using
this script just to drive around no longer litters `lerobot_datasets/` with
metadata-only stubs that block reusing the same `--repo-id`. `--resume` runs never
delete anything.

**Lock the exposure.** Auto-exposure drifts with the scene, which the policy
then has to learn around. `tune_exposure` sweeps exposure × gain for **both**
imagers on one pipeline — so every frame of the sweep sees the same scene under
the same light — and writes a PNG per setting plus a stats table, inspectable
from another machine:

```bash
.venv-lerobot/bin/python -m lerobot_vla.tune_exposure --out /tmp/cam_sweep

# narrow it down: one camera, finer ladder, single gain
.venv-lerobot/bin/python -m lerobot_vla.tune_exposure --camera ir \
    --exposures 8000,12000,16000,20000 --gains-ir 16
```

It stars the rows in band and prints the flags to copy. Target mean ~80–130 with
clip-hi under ~1%; at 30 fps the exposure ceiling is ~33 000 µs for either
imager. Pass the chosen values to BOTH `record_episodes` and `run_inference`.

Re-sweep whenever the lighting setup changes. Measured over the sandbox with the
work lights on (2026-08-19):

| camera | setting | mean | clip-hi |
|---|---|---|---|
| IR cam1 | `--exposure-ir 16000 --gain-ir 16` | 89.5 | 0.07% |
| RGB cam2 | `--exposure-rgb 16000 --gain-rgb 128` | 107.2 | 0.00% |

The colour imager needs far more gain than the IR one for the same brightness —
it is behind a Bayer filter, the IR imager is not. Its dark end suffers for it:
at that setting ~2.8% of pixels are crushed to black, against 0% for IR.

Gains are swept per camera because the scales differ (IR 16..248, RGB 0..128),
but `--exposures` is microseconds for both: librealsense reports the colour
sensor's exposure in 100 µs ticks (range 1..10000) and the stereo module's in
microseconds (1..165000), and the conversion is derived from the range each
sensor reports rather than hardcoded.

### Two cameras, always

**Both imagers are always recorded** — cam1 IR, cam2 colour. They ride one
RealSense pipeline, so a cam1/cam2 pair comes out of a single
`wait_for_frames()` and shares a capture instant. Recording both is cheap and
which one a policy should use is a training-time question: drop the unwanted
camera when deriving the training dataset (`lerobot.datasets.dataset_tools.
remove_feature`), not at record time. That also means an IR-vs-RGB comparison
runs on literally the same episodes.

If a frame from either camera is missing the whole frame is skipped, rather than
written with one camera — otherwise the two video streams desynchronize against
the parquet rows.

They are **separate sensors with separate exposure controls**, and the two do not
even use the same unit — the stereo module reports exposure in microseconds
(range 1..165000), the color sensor in 100 µs ticks (range 1..10000). Both
`--exposure-ir` and `--exposure-rgb` are given in microseconds and converted from
the range the sensor reports, so `--exposure-ir 16000` and `--exposure-rgb 16000`
mean the same 16 ms. Passing microseconds straight to the color sensor would have
overexposed it by 100×. Gain ranges differ too: IR 16..248, RGB 0..128.

Cost: measured clean at 640×480×30 for both streams (0 dropped frames, USB ~6% —
see `realsense_logging_bandwidth.md`). The second video encode is host CPU, not
bandwidth.

Sticks = same mapping as simple_drive.py (left = slew/tilt, right = lift/scoop).
Buttons: **A** start / stop+save episode · **B** discard episode · **X** pump ·
**Y** reload servo config. Episodes auto-save at `--max-episode-s` (120 s).
Default output root: `data_collection/lerobot_datasets/<repo_id>/`.
Use `--resume` to append episodes to an existing dataset.

**The pump is cut across every save, and comes back when the save finishes.**
`save_episode()` blocks the record loop for seconds while it encodes video —
measured ~4 s for 60 frames × 2 cameras — during which the setpoint goes stale
and the control thread ramps the valves to zero. Leaving the pump running through
that just dead-heads it against closed valves and heats oil.

```
[pump] OFF (saving)
[EP] saving 36 frames (1.2s) (button A)... done.
[pump] ON (save complete — ready to record)
```

The pump returning is also the operator's cue that the board has finished writing
and the next take can start. Discarding with **B** does not cycle it — nothing is
encoded, so there is nothing to wait out.

If `--resume` refuses with "holds no saved episodes", the folder is a
metadata-only stub from a run that created the dataset and exited before saving
anything — remove it and record without `--resume`. (Older builds instead fell
through to *downloading* the repo-id from the Hugging Face Hub and died on a
confusing `401 RepositoryNotFoundError`; these datasets are local-only, so the
check now stays local.)

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

On datasets that carry `clock.*`, check the loop and the caches too:

```bash
.venv-lerobot/bin/python - <<'EOF'
import pandas as pd, numpy as np
D="data_collection/lerobot_datasets/masi_kaivuri_juusto"
df=pd.read_parquet(f"{D}/data/chunk-000/file-000.parquet")
for e, g in df.groupby("episode_index"):
    jit = (np.diff(g["clock.loop"].values) - 1/30) * 1000
    dup = int((np.diff(g["clock.imu_us"].values) == 0).sum())
    print(f"ep{e:3d} jitter mean {jit.mean():+6.2f} ms max {jit.max():+7.2f} | "
          f"cam age {g['clock.cam1_age'].max()*1000:5.1f} ms max | "
          f"state age {g['clock.state_age'].max()*1000:5.1f} ms max | "
          f"repeated IMU frames {dup}")
EOF
```

A jitter max far above a few ms means the loop stalled — usually the video
encode. **This is not hypothetical:** with lerobot's default
`streaming_encoding=False`, `add_frame()` writes a PNG per camera per frame on
the calling thread — measured 46 ms each on this board, so two cameras cost
~69 ms against a 33.3 ms budget. The loop settles at ~14.5 Hz, every tick late,
while `timestamp` still claims `frame_index / fps`. The result is a dataset
asserting 30 fps for motion that happened at half that: it plays back ~2x fast,
trains on a 2x-wrong timebase, and `run_inference.py` — which resolves its rate
from the export bundle — then commands the machine about twice too fast. None of
lerobot's fps-mismatch guards catch it, because the recorded fps itself is the
lie. `streaming_encoding=True` (set since 2026-08-19) hands frames to a
background encoder instead: `add_frame` drops to 0.5 ms, the loop holds a
measured 30.00 Hz with 0/150 late ticks, and `save_episode` falls from 8.7 s to
0.4 s. Datasets recorded before that date carry the 2x error.

The loop also paces on **camera frame arrival**, not on a sleep (since
2026-08-19). A 30.000 Hz sleep loop is a second clock beating against a D435i
that actually delivers at ~29.976 Hz: over one 35 s episode the recorded frame
age walked 22.35 ms -> 4.60 ms, and continuing that slide means alternately
reusing and skipping frames. Blocking on the frame removes the second clock
rather than correcting for it. Measured over 600 frames: age 3.25 ms mean
(sd 0.42, max 7.44), no drift (3.31 -> 3.22 ms), 0 reused, 0 skipped.

Two consequences worth knowing. The achieved rate is the camera's true
29.985 Hz, so a dataset declaring 30 carries a 0.05% timebase error -- 17 ms
over a 35 s episode, against the 2x this replaced. And a tick with no new frame
records nothing at all: a duplicate image is a worse lie than a short gap, and
the gap surfaces in the achieved rate printed at save. Repeated IMU frames mean the record loop outran the 200 Hz stream and
wrote the same pose twice. A camera age near a full frame period means the
policy is being trained on images that were already stale when recorded.

Note that a naive "are consecutive video frames identical?" check does **not**
find dropped camera frames. lerobot encodes with a two-frame GOP for random
access, so the stream is `IPIPIP…`; on a static scene each P-frame quantizes the
sensor noise away and reconstructs as its I-frame. That looks exactly like a
15 Hz camera duplicating into a 30 Hz loop, and it is purely the codec. Use
`clock.cam1_age` instead — it is measured before encoding.

## 2. Inference (split TensorRT engines)

The monolithic SmolVLA ONNX cannot TRT-build on 8 GB; the deploy path is the
9-graph split with the flow-matching denoise loop in Python
(`smolvla_split.py`; see `spark-projects/orin-nano/smolvla-runtime/notes/findings.md`).

A finetuned bundle ships its own `export_info.json`, `stats.json` and
`tokenizer/`, so `--split-dir` is (almost) the whole command — see
*What the bundle resolves* below.

```bash
BUNDLE=~/Desktop/smolvla-digging-ir-20k

# model-only proof (no robot, no camera; first run builds engines — minutes):
TRT_DROP_CUDA_EP=1 .venv-lerobot/bin/python -m lerobot_vla.run_inference \
    --split-dir $BUNDLE --synthetic --loops 3

# real observations, actions PRINTED only (safe default):
.venv-lerobot/bin/python -m lerobot_vla.run_inference \
    --split-dir $BUNDLE --exposure-ir 16000 --gain-ir 16

# actually drive the valves (only with a finetuned checkpoint!):
.venv-lerobot/bin/python -m lerobot_vla.run_inference --live \
    --split-dir $BUNDLE --exposure-ir 16000 --gain-ir 16
```

### What the bundle resolves

`run_inference` reads these from `<split-dir>` rather than making the operator
carry them, and an explicit flag always wins:

| Flag | Read from | If it is missing |
|---|---|---|
| `--task` | `export_info.json` → `task` | **hard error** — there is no default |
| `--fps` | `export_info.json` → `fps` | 30, with a warning |
| `--state-blind` | `export_info.json` → `state_blind` | off |
| `--tokenizer` | `<split-dir>/tokenizer/` | the base-export tokenizer |
| `--dataset-stats` | `<split-dir>/stats.json` | identity normalization, with a warning |

`--task` used to default to `"scoop sand and dump it to the left"`. That is the
kaivuri task, and it silently mislabelled every run against any other
checkpoint — the policy conditions on the language embedding, so a phrasing it
was never finetuned on is out of distribution and the prefix is still perfectly
well-formed. The default is gone: the string comes from the bundle, or from
`--task`, or the run refuses to start. It is spelled `--task` because that is
what `record_episodes.py` calls it, and the two must carry the *same string*.

`--camera` is cross-checked against the image keys in the stats file, so a
cam1-trained checkpoint refuses to start on `--camera rgb`.

Setpoint-timing flags (all optional; defaults are sane):

| Flag | Default | Effect |
|---|---|---|
| `--setpoint-hold-s` | 0.25 | How long the chunk's last action keeps full authority after the chunk runs out |
| `--setpoint-decay-s` | 0.25 | Ramp-to-zero window once the setpoint goes stale |
| `--blend-s` | 0.0 | Cross-fade into each new chunk, to soften the step at a chunk boundary |
| `--legacy-direct-write` | off | Drive valves from this thread instead of the control thread (bench A/B only) |
| `--camera` | `ir` | Which imager the policy sees: `ir` = cam1, `rgb` = cam2. **Must match what the checkpoint was trained on** — refused when the stats file disagrees |

Each cycle logs `played=` — how many steps of the *previous* chunk actually ran
before this one replaced it, measured rather than assumed. That is the real
execution horizon. With the default `--min-replan-s 0` it is set purely by
inference latency (`played ~= infer_s * fps`), so at the measured 0.12 s and 30
fps expect ~4. `played` climbing toward the full chunk length means inference is
falling behind playback and the machine is running off the plan's tail; that is
the signal to look at, before touching the hold/decay windows, which exist to
make a stall safe rather than to hide one.

`--split-dir` is **required** — there is no default bundle. A checkpoint decides
what the machine does, so it is named on the command line, never inherited from an
argparse default. Point it at a finetuned bundle
(`~/bundles/smolvla-digging-clean-ir12-35k`) and `--task`, `--fps`,
`--dataset-stats`, the tokenizer and the chunk length all resolve from what the
bundle ships.

It used to default to the public `ainekko/smolvla_base_onnx` base-weight split.
That export did not work for me: scored against a LeRobot 0.5.1 reference it
disagrees with the checkpoint it claims to be, by ~13% of commanded range. Their
export notebook pins `lerobot==0.3.3`, so a version gap between their trace and my
reference is the obvious explanation — but I have not tested it against its own
0.3.3-era module, so that is a guess, not a finding. It may well be perfectly
correct on the software it was built for. YMMV.

### Runtime flags: `--projectors` and `--no-iobinding`

Both default to the settings validated on this board in `jetson-orin-nano-vla`
(`docs/06-optimization-backlog.md`) and re-measured here against the deployed
`smolvla-digging-clean-ir12-35k` bundle, 30 cycles, pinned clocks, MAXN_SUPER:

| projectors | iobinding | p50 ms | p95 ms | Hz | CPU cores busy | peak RSS |
|---|---|---:|---:|---:|---:|---:|
| cpu | off | 147.8 | 176.9 | 6.76 | 2.46 | 1873 MB |
| cpu | **on** | 147.9 | 163.5 | 6.76 | 2.35 | 1879 MB |
| **gpu** | off | 131.8 | 132.4 | 7.59 | 0.45 | 2065 MB |
| **gpu** | **on** | **121.6** | **122.0** | **8.22** | **0.41** | 2060 MB |

`--projectors gpu` moves the four projectors the denoise loop calls *once per
step* (`action_in`, `time_in`, `time_out`, `action_out`) off the CPU EP. TensorRT
declines graphs that small and they run on the CUDA EP — expected, not a failure,
and `get_providers()[0]` will still claim TensorRT because that is the session's
provider *preference*, not what executed. `--iobinding` (on by default) binds the
prefill's KV cache straight to the device and leaves it there for all N steps
instead of re-feeding 7.2 MB of numpy per step.

Together: **1.22x faster, and 2.05 of the six CPU cores handed back** to the
control stack. Parity against the old `cpu`/off default, same observation and
same noise:

- IOBinding is **exactly bit-identical** (max abs diff 0.000e+00), in both
  projector configurations. It is free speed, not a precision trade.
- The GPU projectors account for all of the difference: max abs 3.26e-4, i.e.
  **0.042% of the action range**, cosine 0.999999940. That is CUDA-EP vs CPU-EP
  arithmetic on tiny matmuls, ~24x inside this project's own 1%-of-range parity
  gate.

`--projectors cpu --no-iobinding` reproduces the pre-benchmark loop exactly, for
regression comparison. There is no other reason to use them.

The p95 column is the quiet win for a chunk-12 bundle: 176.9 -> 122.0 ms. That
bundle buffers 0.4 s of plan, so the margin over inference goes from ~2.3x to
~3.3x, and the worst case stops being the thing that eats it.

**NOT taken from that benchmark:** the NaN-guard-stripped vision graph (faster,
but still outside the parity gate) and a reduced `--num-steps` (faster, but it
changes the policy rather than its runtime — validate against the robot first).

Engines are pre-built automatically, one subprocess per graph (two TRT builds
in one process OOM 8 GB), into `/tmp/smolvla_split_cache` — note /tmp clears
on reboot, so the first run after boot rebuilds (~5 min). RAM is tight:
run ONE thing at a time (never viz or recording alongside an engine build).

**fps vs inference rate:** the policy is chunked — one ~122 ms inference emits
the bundle's chunk (12 actions for the deployed digging bundle) authored at the
dataset fps (30 Hz). The chunk is handed to the
control thread, which interpolates it by elapsed time and drives the valves at
100 Hz; inference being far slower than the valve rate is exactly what that
split is for. The camera is only *read* at each re-plan.

The FULL 50-step chunk is always handed to the scheduler, and **nothing
truncates it** — `--min-replan-s` gates only when the next inference may start.
So the executed horizon is not a setting: it is `infer_s * fps`, whatever that
happens to be (~4 steps at 0.12 s and 30 fps). This is why the loop runs
smoothly at any replan cadence including none — the controller always holds a
full chunk as a trajectory (1.67 s at chunk 50, 0.4 s at the deployed chunk 12)
and simply gets a fresher one whenever inference lands.
The unplayed tail is the late-inference fallback: a slow replan keeps following
the predicted plan instead of freezing at the boundary. Hold/decay to zero
(`--setpoint-hold-s`, `--setpoint-decay-s`) engages only if the whole chunk
runs out, i.e. inference stalled outright.

`--min-replan-s` therefore buys idle time, not reactivity: raise it to stop the
Orin re-inferring flat out when the task does not need it (power and thermals on
an 8 GB board), and to reduce how often a freshly sampled chunk replaces the
current plan — the denoise loop draws new noise per call, so consecutive chunks
from near-identical observations differ slightly. Lower it (or leave it at 0)
when you want maximum reactivity.

## Files

```
excavator_robot.py   MasiExcavator: control stack + IR camera as one robot
camera.py            D435i reader: IR-left (Y8@30, emitter off, gray→3ch)
                     + optional color (rgb8@30) on the same pipeline
record_episodes.py   gamepad teleop -> LeRobot v3 episodes
smolvla_split.py     9-graph split policy: ORT/TRT sessions + denoise loop
run_inference.py     obs -> action-chunk -> valves loop (synthetic/dry/live)
```
