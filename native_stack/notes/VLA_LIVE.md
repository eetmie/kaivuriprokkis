# VLA live runner

`tools/vla_live.py` drives the excavator with a fine-tuned GR00T N1.7 or pi0.5 bundle. The bundle
picks the model; the observation, operator, supervisor, logging and CLI are shared.

| File | Role |
| --- | --- |
| `tools/vla_live.py` | CLI (`parity`, `run`), the inference loop, RTC prefix, summary |
| `tools/vla_operator.py` | Xbox pad, task list, supervisor thread (arming, plan renewal, stop rules) |
| `tools/vla_observe.py` | `--start-services`, coherent snapshot of both IR cameras + controller status |
| `tools/vla_groot.py` | GR00T adapter: jetson-orin-nano-vla engines, min-max boundary, runtime prompts, RTC, frame history |
| `tools/vla_pi05.py` | pi0.5 adapter: ctypes over `build/libkaivuri_pi05_runtime.so` (`src/pi05_capi.cpp`), Mode C RTC |
| `src/pi05_runtime.cpp` | pi0.5 TensorRT runtime, shared by the library and `kaivuri-inference` |

## Run

```bash
cd native_stack/tools      # of the checkout you built (the defaults follow it)
PY=~/vla-clean-install-20261012/source/.venv-trt/bin/python

# GR00T: cache defaults to ~/.cache/kaivuri-groot/<bundle name>
$PY vla_live.py run --bundle ~/bundles/groot17-digging640_rtc12_fullft \
  --start-services --live --enable-slew --seconds 0 \
  --task "scoop the sand and dump it into the container" --out ~/groot-live/F0_$(date +%H%M%S)

# pi0.5: cache defaults to <bundle>.cache; --fixtures is required (checked at every load)
$PY vla_live.py run --bundle ~/bundles/pi05-excavator-smoke9 \
  --fixtures ~/bundles/smoke9-export-test-fixtures --max-abs 0.02 \
  --start-services --live --enable-slew --seconds 0 --out ~/groot-live/P0_$(date +%H%M%S)

$PY vla_live.py parity --bundle BUNDLE [--fixtures PREPARED]
```

Without `--live` nothing is sent to the controller (dry PWM with `--start-services`). With the pad
the run starts in manual mode: A = model/manual, X = pump, B = success marker, D-pad = task.
Defaults (`--build`, `--profile`, `--safety`) are this checkout's, so a worktree never runs
another checkout's binaries.

## Model interface

An adapter provides `name, fps, noise_shape, action_cols, tasks, task, task_tokens, rtc, rtc_max,
history, lag_s`, `describe(rtc_on, a)`, `check_prompt(text)`, `same_prompt(a, b)`,
`prompt_note()`, `set_prompt(text)`, `step(frames_y8, status, noise, timings, rtc_delay,
rtc_prefix, capture_ns) -> (commands [H, A] in [-1, 1], degrees)`, `keep_warm(frames_y8,
capture_ns)` and, for RTC, `model_actions(commands)`.

## pi0.5 RTC (Mode C, training-time)

A `pi05_excavator_rtc` checkpoint exported with `--row-time --rtc-max-rows 20` (spark-projects
`vla-onnx/kaivuriprokkis/export.sh`) declares `kaivuri.rtc.row_time`, and `vla_live.py run` turns RTC
on by itself (`--rtc off` runs the plain sampler on the same engines). The d committed rows go in
clean: normalized with the bundle's quantiles (`model_actions`), padded with zeros to the model
width, fed the t = 0 style and reset after every Euler step, so they come back bit-exact. Two small
TensorRT ops do this inside the CUDA graph (`op_rtc_rows`, `op_rtc_prefix`, built and self-checked at
load); d is only buffer contents, so one graph serves every d.

```bash
~/venvs/openpi-lerobot/bin/python -B tools/prepare_pi05.py --bundle B --out B-fixtures   # + Mode C references
$PY tools/build_pi05_cache.py --bundle B --out B.cache       # copy a same-model cache first: only action* rebuild
$PY vla_live.py parity --bundle B --cache B.cache --fixtures B-fixtures --max-abs 0.02
```

Parity also runs the bundle's Mode C references (RTC off and d = 15, prefix exact) and refuses
references from exporters before the zero-padding fix (spark-projects ef9bdd6). Measured
2026-10-30 on smoke9 (`~/bundles/pi05-smoke9-rowtime`): all 10 fixtures pass, worst Mode C max
0.0185. Step p50 437.1 ms plain, 439.9 / 440.2 / 440.7 ms at d = 0 / 15 / 20 (denoise
109.1 -> 111.2 ms), plus ~11 MB of per-row modulation.

## Not yet

- pi0.5: typed prompts (no tokenizer on the board: only the exported tasks run).
- `kaivuri-inference` keeps its own C++ drive loop (`--live`); the operator features are only in
  `vla_live.py`.
- `tools/policy_probe.py` is the older observation-only GR00T scaffold.
