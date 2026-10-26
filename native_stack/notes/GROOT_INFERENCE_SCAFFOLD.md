# GR00T inference scaffold

`tools/policy_probe.py` is a runnable observation-only adapter to the native stack.
It prepares the inference boundary while the excavator dataset and fine-tune are
being collected. It has three independent commands:

| Command | What works now | Dependencies |
| --- | --- | --- |
| `inspect` | Read actual GR00T dimensions, temporal indices, baked prompt and optional bundle checksums | Python standard library |
| `capture` | Native shared-memory IR plus status-only IMU/FK, real timestamped history, arbitrary task text | Standard library, built native frame library; running services or `--synthetic` |
| `fixture` | Execute existing GR00T N1.7 AOT engines with CUDA graphs and compare raw-image preprocessing, full chunk and active action dimensions to the reference | Existing torch-free TRT environment, demo source, verified bundle and optimized cache |

This is a Python scaffold beside native services, not the final C++ model backend.
It has no drive/arm/PWM operation. Engine results stay in model space. Capture
does not normalize state for a policy or interpret actions as excavator commands.
Live-image-to-model execution and the excavator processor contract remain to be
implemented after export of the fine-tune. Existing OpenPI paths are unchanged.

## Try temporal capture without hardware

From the repository root, with the native binaries built:

```bash
python3 -B native_stack/tools/policy_probe.py capture \
  --synthetic --task "scoop up the sand and dump it into the container" \
  --history-ms 1000 --history-ms 500 \
  --duration-s 10 --out native_stack/results/groot-history-first
```

Use a new output directory per run. `--synthetic` launches only simulated-IMU
control and synthetic cameras, disarmed, in a private runtime directory. Its
children stop on exit. Run this test while another controller is not holding the
native controller lock. To reuse an existing build, pass `--build PATH` and
`--library PATH/libkaivuri_frames.so`.

To attach to already running native services:

```bash
python3 -B native_stack/tools/policy_probe.py capture \
  --camera-frame /path/to/cameras/233522073919-ir2.frame \
  --camera-frame /path/to/cameras/141722076647-ir1.frame \
  --control-socket /path/to/control.sock \
  --task-file /path/to/current-task.txt \
  --history-ms 1000 --duration-s 60 \
  --out native_stack/results/groot-history-live-input
```

Use the socket/frame paths belonging to the running session. This command does
not start services, claim control ownership, arm or change an existing control
session. `--task-file` is reread for each snapshot: replace its UTF-8 text to test
another instruction. There is no fixed prompt list. A missing/empty task file
stops capture explicitly. `--task TEXT` supplies an arbitrary constant string.
Camera order is the order of `--camera-frame`; physical orientation/calibration
remain properties of the running camera service and recorded provenance.

## Temporal observation contract

The sampler polls the latest native frames independently of its snapshot rate
(2 Hz by default) and retains a bounded raw Y8 history. Offsets are explicit
milliseconds on the host monotonic arrival clock; for example, a training index
of -30 at 30 training FPS corresponds to `--history-ms 1000`. Do not substitute
live camera FPS for the training sampling contract. Repeat the flag for multiple
past slots; current input is always included. `--history-ms 0` selects current
frames only.

Slots are ordered oldest first, current last. For each camera, a past slot uses
the newest retained frame no later than `current_arrival - requested_offset`.
The default extra lag tolerance is 60 ms. Every slot also checks cross-camera
arrival skew (40 ms default). Current frames must be at most 200 ms old and the
IMU at most 50 ms old. Source gaps and camera sequence resets are detected.

Startup waits for real history. There is no current-frame duplication to fill an
unavailable past slot. A stalled source times out (8 s default); a session with
no complete observation fails. History is capped at 128 MiB by default; memory
evictions are reported and an insufficient buffer causes unavailable history,
not silent reuse. Increasing history duration needs a suitable byte budget.

Each committed `000000.ready/` contains PGM images and `observation.json` with:

- Exact task text, full controller status and four raw joint-angle inputs in
  dataset order `[slew,lift,tilt,scoop]`, float32 degrees.
- Per-slot requested and actual offsets, full camera metadata and file SHA256.
- Current observation time as the oldest current-camera/IMU arrival. Historical
  visual context retains its own timestamp and does not reset current freshness.
- Memory use/eviction counts and `actuator_output=false`.

`observations.jsonl` indexes committed snapshots. Interrupted writes remain
`.partial`; the session summary identifies completion or failure. This format is
for replay/inspection of inference inputs. Continue using `kaivuri-record` for
training demonstrations and their action/time attribution. Keep its raw spools
so different history offsets can be derived later without re-recording.

## Verify the current GR00T engine path on pikkumasi2tb

The board already has the portable bundle and a matching optimized cache. Run
one model worker at a time; do not run model builds beside a loaded worker.

```bash
VLA_ROOT=/home/joel/vla-clean-install-20261012

python3 -B native_stack/tools/policy_probe.py inspect \
  --bundle "$VLA_ROOT/bundles/groot-n17-base"

"$VLA_ROOT/source/.venv-trt/bin/python" -B native_stack/tools/policy_probe.py fixture \
  --bundle "$VLA_ROOT/bundles/groot-n17-base" \
  --cache "$VLA_ROOT/runs/groot-n17-base/optimized-cache" \
  --vla-root /home/joel/jetson-orin-nano-vla \
  --iters 5 --out native_stack/results/groot-fixture-first
```

The adapter verifies portable artifacts, candidate engine hashes, timestep
constants and runtime identity before loading. It never builds engines, rewrites
the cache, or monkeypatches the demo CLI. It compares the raw-image transform
(max absolute error <= 1e-6), full model output and active dimensions (cosine >=
0.99999 and max error <= 0.13% of reference range). Fixture history uses its real
earlier/current images with an explicitly synthetic one-second separation, not
an assumption about live camera sampling.

Results contain `summary.json`, `timings.json` and `model_actions.npy`. One
unmeasured warm-up pair precedes measured pairs; latency reports the current
inference with historical features already cached. Full pair work takes longer.
The worker exits to release its GPU allocations. These are fixture diagnostics,
not task-quality or live-robot latency measurements.

## Prompt support and next implementation

Observation capture accepts arbitrary text now. The current demo bundle bakes one
instruction into exported model artifacts; `inspect` exposes that limitation and
`fixture` uses the reference instruction. It does not pretend that relabeling an
output changes model conditioning. The final model interface must prepare proper
tokens, masks/positions and KV selections for arbitrary user instructions.

Next: export the excavator fine-tune's exact processor boundary and desired
temporal indices; support its actual camera/view count and prompt conditioning;
replay captured observations through it; then port the verified chain to the
native backend and connect the existing plan supervisor. Retain real temporal
fixtures as well as current-only tests. Drive integration remains a separate step.

## Checks

```bash
python3 -B native_stack/tests/policy_probe_tests.py
python3 -B native_stack/tests/policy_probe_integration.py native_stack/build
```

Both are registered with CTest (integration requires `BUILD_CAMERA=ON`). The
integration uses only simulated services, checks selected historical timestamps,
changes prompt text between snapshots, validates artifact hashes and confirms
that control stayed disarmed.

Board validation on 2026-10-26: the CPU suite and native simulated integration
passed. The existing optimized N1.7 cache passed all fixture gates with real
earlier/current fixture images. Five measured current-frame inferences after
warm-up gave p50 270.32 ms, p99 271.20 ms. Full-chunk cosine was 0.999999982;
active-dimension cosine 0.999999843 and maximum error 0.0685% of reference range.
The raw-image transform maximum difference was 5.91e-8. This is a short model
smoke check, not a sustained live-input benchmark. TensorRT emitted device-plan
compatibility and AOT hidden-argument warnings while deserializing the retained
cache; execution and numerical gates passed. Results are retained under
`results/groot-scaffold-fixture-20261026/` on the board.
