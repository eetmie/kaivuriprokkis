# Python X-VLA stack on the robot

Host: `joel@pikkumasi2tb` (Joel's Jetson/robot). Checkout:
`~/kaivuriprokkis-python`, branch `feat/python-dual-ir`.

This is the Python recording and X-VLA deployment project. The native C++
project and its datasets remain separate in `~/kaivuriprokkis`.

## Record

```bash
cd ~/kaivuriprokkis-python
.venv/bin/python -m lerobot_vla.record_episodes \
    --repo-id masi/python_dual_ir \
    --task "scoop the sand and dump it into the container"
```

Datasets land in `data_collection/lerobot_datasets/masi_python_dual_ir`.
Use a new repo ID for tomorrow's new camera mounts. To append later, repeat
that command with **`--resume`**; it is never enabled by default. You can change
`--task` when resuming. Short episodes are kept.

A starts/saves a take, B discards it, X toggles the pump, Y reloads the servo
configuration. Ctrl+C saves a nonempty active take and closes the dataset.
Episode numbers displayed to the operator start at 1.

Both cameras are **IR**, 640×480, 30 FPS, gray replicated into three channels:

| Dataset feature | Physical view | RealSense serial | IR imager | Rotate |
| --- | --- | --- | --- | --- |
| `observation.images.cam1` | wrist | `233522073919` | 2 | 180° |
| `observation.images.cam2` | cabin | `141722076647` | 1 | none |

Depth, RGB and IR emitters stay off. Identities and orientation match native
capture. `--cam1-serial`, `--cam2-serial`, `--cam1-ir-index`, `--cam2-ir-index`,
`--[no-]cam1-rotate180` and `--[no-]cam2-rotate180` override them explicitly.
Exposure is automatic unless `--exposure-ir` / `--exposure-ir2` are specified
in microseconds. `--gain-ir` / `--gain-ir2` accompany manual exposure.

Each camera has its own acquisition thread; the two devices are not hardware
synchronized. Missing/stale images, excessive arrival-time skew and repeated
pairs are rejected. `clock.cam1_age`, `clock.cam2_age`, `clock.state_age`,
`clock.loop` and `clock.imu_us` retain timing diagnostics. They measure host
receipt/state availability, not synchronized sensor exposure. The recorder
warns if its achieved rate falls below the advertised dataset FPS.

`meta/camera_config.json` records device identity and orientation. `--resume`
refuses incompatible capture settings or a legacy RGB dataset with no dual-IR
metadata. Exposure may change between sessions.

LeRobot **0.6.1** writes Parquet plus H.264 MP4 directly, using background
encoding (`veryfast`, CRF 18, GOP 2). This video is lossy; native Y8 recording
retains the original camera bytes. Default state is `[lift, tilt, scoop]` in
degrees, and action is `[slew, lift, tilt, scoop]` in valve units. Slew feedback
stays excluded because its absolute yaw has no stable origin across boots.
`--enable-tracks` adds two action channels; match that setting in training.

## Run a new X-VLA fine-tune

The old checkpoint used different mounts; train a new one with both views.
Export its processor statistics, tokenizer, training FPS and reference fixture.
Export tooling is in `~/jetson-orin-nano-vla` and
`~/spark-projects/vla-onnx/xvla`. Use mixed-FP16 ONNX graphs and include the
physical processor contract, fast tokenizer and stock reference fixture.

```bash
cd ~/kaivuriprokkis-python
# Real observations, printed actions:
.venv/bin/python -m lerobot_vla.run_inference \
    --split-dir ~/bundles/xvla-dual-ir
# Add --live to drive, after checking the new model.
```

`--camera auto` reads the checkpoint's ordered VISUAL features. A two-view
checkpoint gets both images; a one-view checkpoint gets its declared view.
Inference briefly waits for both acquisition threads after GPU completion,
then rejects stale images. Unknown image features, incorrect order and incorrect view count fail before
model loading. Recording and inference share camera identity/orientation.
The same Python 100 Hz controller plays teleoperation setpoints and policy
chunks. D-pad changes tasks; A switches manual takeover during live inference.

The default backend is **direct TensorRT with CUDA graphs**. It keeps the
vision/text/conditioning/denoising chain on the GPU, shares scratch memory,
reuses intermediate buffers and downloads only the final chunk. Image
normalization uses a lookup table with parallel view preprocessing; the
exported fast tokenizer runs without transformers/PyTorch. Saved processor
statistics normalize state and trim/unnormalize actions. Generic `auto` action
mode does not apply a gripper sigmoid.

Engine builds run one subprocess at a time. Caches live in
`<bundle>/trt_native_cache`; graph hashes, TensorRT/CUDA and builder options
invalidate stale engines. `--rebuild` clears that generated cache.
`--trt-chain host|device|graph` is available for diagnostics; `graph` is default.
No Triton kernels are required. New exports must contain mixed-FP16 graphs and
an exporter reference fixture. Loading checks preprocessing, token IDs, raw
image-to-action parity and the device chain against that fixture.

A base model remains model-only: `--allow-base-bundle --synthetic --fps 30
--task "..."` permits diagnostics, and refuses `--live`. Benchmarking a base
model is not evidence that a new two-view excavator checkpoint is validated.

## Setup and checks

The dedicated `.venv` is already installed on the Jetson. To recreate its
Python dependencies on the same JetPack system:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r lerobot_vla/requirements.txt
```

JetPack supplies the TensorRT Python bindings and CUDA libraries. No SmolVLA,
LeRobot 0.5.1, transformers, ONNX Runtime or Triton is needed by this workflow.
LeRobot's dataset writer itself depends on PyTorch; inference does not import it.

```bash
.venv/bin/python -m pytest -q tests/test_dual_ir.py tests/test_xvla_native.py \
    tests/test_xvla_bundle.py tests/test_record_action_layout.py \
    tests/test_record_clocks.py tests/test_task_switching.py
.venv/bin/python -m lerobot_vla.tools.benchmark_xvla \
    --split-dir ~/bundles/xvla-dual-ir --seconds 60 \
    --output results/xvla-benchmark.json
```

Keep Python and C++ datasets separate initially. The language alone does not
prove a large delay difference; compare camera age, control timing, encoding
load and action scheduling. Each model should first be trained and deployed
through its own measured pipeline.

## Retained checks (2026-10-27)

Measured on this Jetson with the **three-view base** X-VLA bundle, 10 denoising
steps and 30 actions; these are model/runtime checks, not excavator task success.
Triton/AOT kernels are excluded.

| Check | Median inference | Resident process memory |
| --- | --- | --- |
| Direct TensorRT host chain, 60 s | 404.4 ms | 2478 MiB |
| Direct TensorRT CUDA graph chain, 60 s | 381.9 ms | 2469 MiB |
| Graph chain beside real cameras, IMUs and controller, 30 s | 387.6 ms | 2674 MiB |

The graph chain is 5.6% faster than the paired TensorRT host-chain check. The
upstream retained ONNX Runtime run used 4843 MiB; its old timing is historical,
not a freshly paired measurement. The current model-only graph process uses
49% less memory than that retained run. Neither PyTorch nor ONNX Runtime is
imported in the new inference checks.

The integrated check kept the controller at 100.002 Hz (mean interval 10.000 ms,
max 11.606 ms over the final 1000 samples). All valve commands were zero and the
pump was off. Because the base bundle requires three views, that diagnostic
used the two real IR images plus one fixed fixture image; it never drove policy
actions. A real two-view excavator model still needs its own validation.

Additional checks passed: analytic two-view `auto` action graphs at horizons
30/50, stock/raw-image fixture parity, 89 focused tests, live dual-IR H.264
capture at ~30 FPS, and the actual recorder's save/discard/resume flow with
real camera/IMU/controller threads. Diagnostic takes stay under `results`,
separate from training data. Reports are in `results/xvla-trt-20261027` and
`results/dual-ir-check-20261027`.
