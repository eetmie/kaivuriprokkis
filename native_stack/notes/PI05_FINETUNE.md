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
| C++ runtime | `native_stack/src/inference_pi05.cpp` (OpenPI pi05_libero contract today) |

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

## Runtime work here afterwards

- `kaivuri-inference`: accept the fine-tune bundle (drop the `pi05_libero` check at
  `inference_pi05.cpp:324`), read the contract above, feed live state from `kaivuri-control`;
  with `discrete_state_input=True` (the decided contract), assemble the prompt per step.
- A policy → controller producer: send chunks through the `PROTOCOL.md` socket with the
  oldest observation time, delay-aware chunk start, and lease renewal that never refreshes
  observation age.
- Benchmark recording + inference + control together.

## Before real data collection (open from `TODO.md`)

Real Xbox pad dry check (only the virtual pad has driven the C++ recorder); hydraulics-on
pulse/stop checks; lock IR exposure/gain (`--exposure-us`, `--gain`). IMU readout and FK
are expected to be correct as of 2026-10-16. The jetson profile's valve deadzone is now 0%.
