# pi0.5 (OpenPI) fine-tune pipeline — working notes

Dev notes for building the excavator pi0.5 fine-tune on the DGX Spark and running it
through `kaivuri-inference`. Written 2026-10-16 at the end of the native recorder session.
`jetson-orin-nano-vla` stays a clean public example repo: read it for reference, never
change it for this project. Training and export code go in `spark-projects`, robot and
runtime code here.

## Where things are

| What | Where |
|---|---|
| Recorder + exporter (v3) | here, branch `feat/native-recorder`: `native_stack/tools/record_native.py`, `export_episodes.py`, `notes/RECORDING.md` |
| v3 → v2.1 converter | spark-projects, branch `feat/pi05-openpi`: `vla-onnx/common/vla_common/dataset/to_v21.py` |
| Test dataset (v3 + v2.1) | Jetson `~/kaivuriprokkis/data_collection/lerobot_datasets/masi_native_test_{v3,v21}`: 8 episodes, 4,606 frame pairs, 2 tasks, virtual pad, live PWM into unpowered servos |
| Jetson envs | `~/kaivuriprokkis/.venv-lerobot061` (lerobot 0.6.1 export); `~/venvs/openpi-lerobot` (lerobot 0cf8648 + datasets 3.6.0 + torch 2.7.1 + av 13.1.0, OpenPI loader test) |
| OpenPI pi0.5 export reference | jetson-orin-nano-vla `export/pi05/` (OpenPI commit c23745b, NGC container) — reference only |
| C++ runtime | `native_stack/src/inference_pi05.cpp` + `pi05_policy.cpp` (pi05_libero and pi05_openpi_state) |

## Dataset format: decided

Record and keep **LeRobot v3.0** (lerobot 0.6.1). Derive **v2.1** for OpenPI (and GR00T)
with `to_v21`. OpenPI main still pins lerobot `0cf8648` (v2.1 only, openpi #818 open).

Two traps found, both handled or documented:

1. **Parquet metadata.** lerobot 0.6.1 writes through datasets 4 (`"_type": "List"`);
   OpenPI's `datasets==3.6.0` rejects every file. `to_v21` rewrites it to `Sequence`.
   NVIDIA's `convert_v3_to_v2.py` does not, so its output does not open in OpenPI.
2. **Video decode on aarch64 (the Spark).** OpenPI's `uv.lock` has torchcodec only for
   x86/macOS, so lerobot falls back to torchvision's pyav reader, which calls
   `av.open(metadata_errors=...)`, removed in PyAV ≥ 14. The lock has `av==17.0.0`.
   **Override `av<14` (13.1.0 tested)** in the OpenPI env on the Spark.

Verified on the Jetson: the v2.1 copy loads under the pinned lerobot and 72 sampled
tensors (both cameras, state, 50-step action chunks with padding, tasks) equal lerobot
0.6.1's reads of the v3 source.

## Spark steps

1. `git switch feat/pi05-openpi` in spark-projects; rsync the test dataset (or real data).
2. `python -m vla_common.dataset.to_v21 --src <v3> --dst <v21>` (pyarrow + av; any lerobot 0.6.1 venv).
3. OpenPI env: `uv sync` from its lock, plus the `av<14` override. Keep `datasets==3.6.0`.
4. Excavator `TrainConfig` (keep it in spark-projects; register it at runtime rather than
   forking OpenPI):
   - repack: `observation.images.cam1` → `base_0_rgb`, `observation.images.cam2` →
     `left_wrist_0_rgb` (third slot masked), `observation.state`, `action`, prompt from task
   - `Pi0Config(pi05=True, action_horizon=50, discrete_state_input=True)`. **Decided
     2026-10-16: dual IR + joint angles.** pi0.5 has no continuous state input, so `False`
     would drop the angles entirely; `True` (OpenPI's pi0.5 default) writes them into the
     prompt, `Task: T, State: <256 bins>;\nAction: `, rebuilt every step.
     Possible later A/B: append FK tool pose (`feedback.fk.fk_upper_xz_m`,
     `feedback.fk.bucket_pitch_rad`, already in every exported frame) to the state.
   - output transform: first 4 dims (slew, lift, tilt, scoop), 6 with tracks
   - quantile norm (pi0.5); `scripts/compute_norm_stats.py` on the v2.1 copy
5. Smoke run, then a real run. `fps` stays the dataset's 30.

## Exporter for the fine-tuned checkpoint (to build in spark-projects)

Generalize jetson-orin-nano-vla `export/pi05` into `spark-projects/vla-onnx/pi05/`:

| Hardcoded for LIBERO | Change |
|---|---|
| `export.py` rejects any weights but LIBERO's (tensor-hash check) | record the checkpoint's own hashes instead |
| `cfg.get_config('pi05_libero')` in `convert.py` + all four `prepare_*` stages | `--openpi-config file.py:NAME`, registered at runtime |
| LIBERO fixtures: `observation/image` + wrist, 8-dim state, 5 LIBERO prompts, `LiberoOutputs` assert | build fixtures from the config's inputs and `--task` strings (or real dataset frames) |
| constants: 10 rows, 7 action dims, noise (10, 32), prefix 968/521, prompt capacity 200 | derive from the config and bundle layout |
| `discrete_state_input=True` | bundle carries task heads + state-digit token pieces; same prompt format as LeRobot pi05, so port `pi05_lerobot/export_split.py`'s logic |

Each stage must stay byte-identical for LIBERO, and the fine-tune gets its own fixture gate.

## Bundle contract (same idea as the Python runtime's export_info.json)

The Python stack's exporters write `export_info.json` / `bundle.json` from the checkpoint
(`fps`, `tasks`, `state_blind`, shapes, normalizer stats), and `run_inference.py`
configures itself from them and refuses mismatches (`resolve_policy_fps`,
`resolve_state_blind`, `check_camera_against_stats`, `resolve_tasks`). Do the same for
pi0.5, in a `kaivuri` section of `bundle.json`, filled by the exporter from the checkpoint
and the training dataset:

| Field | Source | Runtime use |
|---|---|---|
| `fps` | dataset `meta/info.json` | chunk playback rate; refuse a contradicting override |
| `tasks` (task_index order) | dataset `meta/tasks.jsonl` | prompt table; D-pad order; warn on an unknown task |
| `state.names`, units, `discrete_state_input` | dataset features + train config | build the state vector; `False` = state ignored (the pi0.5 `state_blind`) |
| `action.names`, range, tracks | dataset features | map rows to `[slew, lift, tilt, scoop, trackL, trackR]`; pad tracks with 0 |
| `norm` (state/action q01/q99) | checkpoint `assets/<asset_id>/norm_stats.json` | quantile normalize/unnormalize; never from another run |
| `action_horizon`, `num_steps` | train config | plan length; lease/renewal timing |
| `cameras`: slot → dataset key → serial, IR index, rotation, resolution | dataset `native/session-*/provenance/manifest.json` (`camera_order`, `camera_rotations_deg`) | **refuse to start if live cameras differ** |
| `profile_sha256` | the same manifest's `files` (profile YAMLs) | warn/refuse if the live controller profile (geometry, IMU mapping) differs, because the state depends on it |
| `preprocess` | exporter | OpenPI antialiased resize, gray → 3 channels, centered padding |
| provenance | exporter | checkpoint hashes, OpenPI commit, config name, dataset path + info hash, train step, exporter git SHA |

The camera/profile checks are new compared to the Python stack: the native recorder's
`native/` provenance makes them possible.

## Runtime (branch `feat/pi05-finetune-runtime`)

`kaivuri-inference` accepts `pi05_openpi_state` bundles (`vla-onnx/pi05/BUNDLE.md`) next to
`pi05_libero`. CUDA-free parts are in `src/pi05_policy.cpp` (`pi05-tests` in CTest).

- **Cache**: `candidate.json` = AOT attention cache (LIBERO); otherwise the native template
  cache, `<kind>.engine` + `<kind>.onnx.sha256` (ONNX hash checked). The Euler `op_axpy_*`
  engine is optional: without one it is built in-process; its constant is checked numerically.
- **Layout**: `language_residual: float32` (FP32 prefix, prompt rows rounded to FP16 then
  widened) and a K/V-only last `language_kv` layer.
- **Per step**: `angles_rad` from a status-only control connection (IMU healthy, FK valid,
  IMU sample <= 50 ms old) -> float32 degrees -> OpenPI float64 quantile norm -> bins ->
  bundle pieces -> `set_prompt`. Rows upload every step, masks/RoPE only when the token count
  changes, CUDA graph captured once. Host cost on GB10: ~1.1 ms with a length change, a
  few microseconds when the prompt is unchanged. Observation time = oldest of both camera
  arrivals and the IMU sample.
- **Preprocessing**: OpenPI's `ResizeImages` is Pillow BILINEAR (`openpi_client`), not the JAX
  resize the LIBERO path emulates; `pil_bilinear` reproduces Pillow's fixed-point taps exactly.
- **Actions**: first `robot_action_dim` columns, `(x + 1) / 2` in float32 then float64 stats
  (numpy promotion), `[slew, lift, tilt, scoop]` = controller order.
- **Parity gate** (`prepare_pi05.py` maps the fixture npz): prompt from `raw_state` == the
  fixture tokens, normalized state and unnormalized FP16 reference bit-exact, Pillow
  preprocessing bit-exact (and gray == replicated RGB), full chunk vs `fp16_actions_normalized`
  cosine >= 0.9998 and max |diff| <= 0.01 (override `--parity-cosine/--parity-max-abs`).
  smoke9 on GB10 (TRT 10.14, native): worst 0.99990 / 0.0070; graph == eager.
- **Cameras**: frame files must be the bundle's `kaivuri.cameras` in slot order; rotation
  comes from `kaivuri-camera --rotate-180` (`bench_pi05.py --rotate-180`), not checkable here.

### Driving (`--live`)

`pi05::Producer` owns its own control connection on its own thread. Without `--live` nothing
is sent (`bench_pi05.py` never passes it).

- Refuses to start unless control is disarmed. `--live` sends plans, accepts a live-PWM
  controller and grants pump permission (it replaced `--drive --allow-live --pump`);
  `--enable-slew` / `--enable-tracks` need the same daemon flag, otherwise those columns
  are sent as 0.
- Arms after one warm-up inference (centered hold, renewed) once its observation-to-plan
  time plus the start period fits inside both the chunk length and the observation limit.
- Each chunk: `t0` = observation time (+ `--action-offset-ms`), so the controller starts at
  the row matching its clock and the inference delay is skipped; replaces the previous plan;
  only newer observations are accepted. Out-of-range values are clamped to [-1, 1] with a
  `[drive] warning` (the controller clamps too); non-finite values stop the run. A finite
  `--action-tolerance X` makes |v| > 1 + X a stop instead.
- Renewal every 100 ms with the same rows, `t0` and observation time. Stops (disarms) at
  `--max-observation-age-ms` (1400, daemon faults at 1500), on chunk exhaustion, on any
  controller state other than armed, on any inference error. Never re-arms.
- Timing: at 2 Hz pacing a chunk must last until the next one lands, about one period plus
  one latency (~1.0 s at 500 ms). 50 rows at 30 fps last 1.63 s from the observation; a shorter
  export must keep `(rows - 1) / fps` above that, which the arming check enforces.

**Jetson end-to-end, 2026-10-17** (pikkumasi2tb, TRT 10.16, native cache from
`tools/build_pi05_cache.py`; smoke checkpoint, servos/pump unpowered): real IMUs, 200 Hz control
with live PWM and slew, both IR cameras, `--drive --rate-hz 2`, 120 s. Inference p50/p99
434.0/437.5 ms (vision 61, language 264, denoise 109, prompt 3.3), camera arrival to plan
458/479 ms, 240/240 plans, 956 renewals, start interval 500.03-500.07 ms, 0 overruns, control
lateness max 1.7 ms, 6.4 GB used / 1.1 GB available. Board parity gate: `--parity-cosine 0.9998
--parity-max-abs 0.02` (fixture 3 max 0.0119). ~60 ms per 2 Hz cycle is spare.

Dry end-to-end on the Spark (simulated-IMU control, replayed fixture frames, GB10): 60 plans at
2 Hz (start interval 500.05-500.30 ms), 239 renewals, max send gap 107 ms, armed 30 s, clean
disarm. Simulated IMU angles are out of distribution (all bins -1); smoke9 then outputs up to
1.06 and stops at the default tolerance.

## Before real data collection (open from `TODO.md`)

Real Xbox pad dry check (only the virtual pad has driven the C++ recorder); hydraulics-on
pulse/stop checks; lock IR exposure/gain (`--exposure-us`, `--gain`). IMU readout and FK
are expected to be correct as of 2026-10-16. The jetson profile's valve deadzone is now 0%.
