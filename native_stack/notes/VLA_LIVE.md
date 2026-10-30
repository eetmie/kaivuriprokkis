# VLA live runner

`tools/vla_live.py` drives the excavator with a fine-tuned GR00T N1.7 or pi0.5 bundle. The bundle
picks the model; the observation, operator, supervisor, logging and CLI are shared.

| File | Role |
| --- | --- |
| `tools/vla_live.py` | CLI (`parity`, `run`), the inference loop, RTC prefix, summary |
| `tools/vla_operator.py` | Xbox pad, task list, supervisor thread (arming, plan renewal, stop rules) |
| `tools/vla_observe.py` | `--start-services`, coherent snapshot of both IR cameras + controller status |
| `tools/vla_groot.py` | GR00T adapter: jetson-orin-nano-vla engines, min-max boundary, runtime prompts, RTC, frame history |
| `tools/vla_pi05.py` | pi0.5 adapter: ctypes over `build/libkaivuri_pi05_runtime.so` (`src/pi05_capi.cpp`) |
| `src/pi05_runtime.cpp` | pi0.5 TensorRT runtime, shared by the library and `kaivuri-inference` |

## Run

```bash
cd ~/kaivuriprokkis/native_stack/tools
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

## Not yet

- pi0.5: RTC, typed prompts (no tokenizer on the board: only the exported tasks run).
- `kaivuri-inference` keeps its own C++ drive loop (`--live`); the operator features are only in
  `vla_live.py`.
- `tools/policy_probe.py` is the older observation-only GR00T scaffold.
