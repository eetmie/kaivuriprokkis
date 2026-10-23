# pi0.5 chunk consistency: implementation plan (A now, C-ready)

Companion to [RTC_SCOPE.md](RTC_SCOPE.md) (options, references, reference-code mechanism).
This file is the work plan. Written 2026-10-23, **nothing below is implemented yet**; the only
prototyping done was read-only inspection (findings in section 1).

Goal: replace "each chunk is a fresh plan" with chunk-to-chunk consistency that works at our
asynchronous ~430 ms latency, in two modes that share one code path:

- **Mode A, inference-time inpainting.** Works on any model, including one trained with plain
  OpenPI flow matching. Anchor rows are held on the flow path (noised to the step time) and a
  covariance correction is pushed to the free rows, gated by `t > t_ip`.
- **Mode C, training-time conditioning.** The model is trained with a clean action prefix
  (per-row time 0). At inference the prefix rows are fed clean with time 0 and are never
  integrated. No extra forward pass, so latency matches the baseline.

Design rule: modes differ only in data (per-step blend coefficients, whether rows get a clean
time conditioning), not in structure. Adding C later must not need a second refactor.

## 1. Findings that shape the plan

Facts verified this session (references are to the repo at `cleanup/native-stack` @ dc6d497):

1. **Time conditioning is precomputed and row-broadcast.** `conditioning_10step_fp16.npy` is
   `[steps, 2*action_layers + 1, 3072]` fp16 (`inference_pi05.cpp:530-536`). The ONNX templates
   of the smoke9 bundle take `mod_in`/`mod_post`/`mod_final` as `[1, 1, 3072]` (3 x 1024 =
   shift/scale/gate) and `hidden` as `[1, 50, 1024]`. So today one style vector is applied to all
   50 rows.
2. **Mode C therefore needs per-row modulation in the engines**: `[1, 50, 3072]` instead of
   `[1, 1, 3072]`. The graph slices `mod_*` (ONNX `Slice` on the modulation tensor), so the
   change may be only a different traced input shape and an engine rebuild. **Checked
   2026-10-23 on the smoke9 `action.onnx` template:** all six modulation slices (3 on `mod_in`,
   3 on `mod_post`) use axes `[-1]`, i.e. the last axis, so a `[1, 50, 3072]` input slices the
   same way. Still to verify: that the downstream multiplies broadcast against `[1, 50, 1024]`
   without a reshape that assumes the singleton (re-trace and compare outputs; see P5).
3. **Style depends only on the time value.** For binary prefix rows only two time values exist:
   `t_s` (per step, already tabulated) and `t = 0` (one extra `[2L+1, 3072]` table to export).
   Do **not** store a fully per-row table: `10 x 37 x 50 x 3072 x 2 B` is about 114 MB resident on
   a board with ~5 GiB free. Assemble per step on the device instead (about 11 MB working set,
   one broadcast select; negligible against ~11 ms/step of denoise).
4. **Ops are built as TensorRT networks, not CUDA kernels.** `CMakeLists.txt` declares
   `LANGUAGES CXX`; `kaivuri-inference` links `CUDA::cudart` only. The Euler update is a TRT
   network built in-process (`build_axpy`, `inference_pi05.cpp:331`), run via `run(euler, …)`
   (line ~322). The blend, correction and mod-assembly ops should follow the same pattern
   (ElementWise + MatrixMultiply layers), so no new toolchain is needed.
5. **Noise is host-supplied** (`infer(…, noise)`, `inference_pi05.cpp:688`): correlated noise
   needs only a host-side Cholesky multiply.
6. **Reference algorithm** (BEHAVIOR-1K / LeHome code) is in RTC_SCOPE.md: hard-set observed
   positions to `(1-t)·x0 + t·z`, push the correction to free positions via
   `C = Sigma_UO · Sigma_OO^-1`, only while `t > t_ip`, skip the step's correction if
   `max|corr| > 1.0`.
7. **Board limits** (pikkumasi2tb): Orin Nano, ~7 GiB total / ~5 GiB free, CPU-only torch. The
   Spark is the dev/training host; the board only builds and runs the native stack.

## 2. Architecture

```
            +---------------- host (CPU, CUDA-free, unit-testable) -----------------+
 chunk N-1  | ChunkMemory: normalized chunk, observation time, fps                  |
 ---------> | AnchorPlan(now, latency_budget) -> d rows, offset, x0[d][D]           |
            | RtcSchedule(mode, steps, t_ip) -> per step: alpha_s, beta_s, gate_s   |
            | Correction(Sigma, d) -> C[d*D][(H-d)*D]  (cached for every d <= d_max)|
            +-----------------------------------+-----------------------------------+
                                                | per inference: x0, z, mask, C
            +-----------------------------------v-----------------------------------+
 device     | initial: x = z, then rows<d := alpha_0*x0 + beta_0*z                  |
 (TRT ops)  | per step s: [mode C only] mod rows<d := style0, rows>=d := style_s    |
            |             action_input -> 18 x action layer -> action_output -> v   |
            |             x' = x + dt*v                         (Euler, unchanged)  |
            |             blend: rows<d := alpha_{s+1}*x0 + beta_{s+1}*z  (if gate) |
            |             [mode A only] free rows += delta_O @ C  (if gate, stable) |
            +------------------------------------------------------------------------+
```

Mode parameters (the only difference between A and C):

| | alpha_s | beta_s | gate_s | correction | mod rows<d | training needed |
|---|---|---|---|---|---|---|
| off | n/a | n/a | never | no | n/a | no |
| A | `1 - t_s` | `t_s` | `t_s > t_ip` | yes | `style_s` (same as free rows) | no |
| C | 1 | 0 | always | no | `style0` (t = 0) | per-row-time fine-tune |

Notes: for C the prefix rows are also forced to the clean anchor in the *initial* state, and the
velocity output on those rows is discarded (the blend overwrites them each step). In both modes
`d` may be 0 for the very first chunk, which must reproduce today's outputs **bit-exactly**.

## 3. Phases

Each phase ends with a commit that builds and passes `ctest`. Branch: `feat/pi05-rtc`.

### P0. Decisions that must precede data collection (no code)

- Fix **action horizon H, action rate, and dims** (4 or 6). The covariance `Sigma` and every
  cached `C` are tied to them. Lower rates (15–20 Hz) cut the anchor length
  `d ~ latency x rate` and the matrix sizes.
- Decide **correlated vs i.i.d. noise** for the first fine-tune (default: plan an A/B, train
  correlated first; see RTC_SCOPE "Greenfield").
- Decide whether the first real fine-tune is **plain (A only)** or **per-row-time (C)**. Plan:
  collect data once; train plain first, then C as a second run on the same data.
- Dataset must keep per-frame observation and command timestamps (already in the recorder
  format) so join discontinuity can be measured offline.

### P1. Host library `rtc` (CPU only) — `include/kaivuri/pi05_rtc.hpp`, `src/pi05_rtc.cpp`

Add to the `kaivuri_pi05` library (no CUDA, `-ffp-contract=off` like the rest).

- `enum class RtcMode { Off, Inpaint, Clean }`.
- `struct RtcOptions { RtcMode mode; int max_rows; double t_ip, stability_limit; double latency_budget_ms; }`
  with validation via `require()`; the same helpers as `Args::finite/natural` for flags.
- `class ChunkMemory`: stores the previous chunk **in normalized space** (the inverse of
  `robot_chunk`; add `normalize_chunk()` next to `unnormalize`), its `observation_ns`, `fps`, and
  the dims layout. `anchor(now_ns, d)` returns rows `[elapsed .. elapsed + d)` where
  `elapsed = ceil((new_observation_ns - old_observation_ns) / frame_ns)`, throws/returns empty if
  the old chunk has fewer than `elapsed + d` rows left (then fall back to Off for that chunk).
  Decide how `d` is chosen: `d = ceil(latency_budget_ms / frame_ms)` from the p99 of measured
  inference latency, clamped to `max_rows`; keep it constant per run (required by the cache).
- `struct Schedule { std::vector<float> alpha, beta; std::vector<uint8_t> gate; }` built from
  `steps`, mode and `t_ip`. Step times: `t_s = 1 - s/steps`; **check against the exporter's time
  grid** (`dt = -1/steps`, `inference_pi05.cpp:467`) and assert they match.
- `class Correction`: input flattened `H*D` covariance (or its Cholesky), shrinkage `beta`;
  `matrix(d)` returns `C = Sigma_UO Sigma_OO^-1` via Cholesky solve with the reference jitter
  `1e-6 * max(mean(diag(Sigma_OO)), 1)`. Cache for all `d <= max_rows`; `require()` on a larger
  request (the reference repo silently clipped an out-of-range index, see RTC_SCOPE item 4).
  Sizes: `|O| <= max_rows*D`, `|U| <= H*D`; for 50 x 4 and d = 20 that is 80 x 120, trivial.
- CPU reference `apply_blend(x, x0, z, mask, alpha, beta, gate, C, limit)` in FP32. This is the
  **oracle** for the device op and for tests.
- For the i.i.d.-noise case (no retrain), add `Correction::matrix_iid(d, t)`, the time-dependent
  propagation `(1-t)^2 Sigma_UO (t^2 I + (1-t)^2 Sigma_OO)^-1` (my derivation in RTC_SCOPE
  item 5; **unvalidated**). Keep it behind a flag; delete if the offline test shows no benefit.

Tests (`tests/pi05_tests.cpp`, CPU, run locally and on the Spark):
- `d = 0` returns the input unchanged, bit-exact.
- Anchor alignment: synthetic chunks with known timestamps give exactly the expected rows.
- `Correction` vs a brute-force `solve` on a small random SPD matrix; reproduces the reference's
  index convention (`C[j, i]` maps delta at observed `j` to free `i`).
- Stability guard skips the whole step for an out-of-range correction.
- Mode C: prefix rows equal the anchor exactly after every step; free rows untouched by blend.
- Schedule: alpha/beta/gate for 10 steps and `t_ip = 0.4, 0.5`.

### P2. Device op for Mode A — `src/inference_pi05.cpp`

- New in-process TRT op (like `build_axpy`, kept out of the engine cache or optionally cached the
  same way `op_axpy_*` is): inputs `x [H,D]`, `x0 [H,D]` (zero beyond `d`), `z [H,D]`,
  `mask [H,D]`, `delta`-correction matrix `C [H*D, H*D]` (zero rows/cols outside the active
  block), scalars `alpha`, `beta`, `gate`, `limit`; output `x_out`. Compose from ElementWise,
  MatrixMultiply (`delta [1,H*D] @ C`), Reduce(max abs), and a Select for the stability guard.
  Padding C to the full `H*D x H*D` (200x200 or 300x300 fp32, ~0.35 MB) keeps shapes static for the
  CUDA graph.
- Insert after `run(euler, …)` inside the step loop (`:322`), and one extra application before
  step 0. Per-step scalars come from `Schedule` and are fixed for the run, so they can be device
  constants indexed by `s`, or small buffers updated per inference, whichever keeps the graph
  capturable once. Per inference, upload `x0`, `z`, `mask`, `C` (like `noise`).
- Keep the **same `z`** as the initial noise: `infer()` already receives it; add the copy to a
  device buffer.
- Parity gate: device op vs `apply_blend` on random inputs (cosine >= 0.99999, max abs <= 1e-5
  in FP32), then end to end vs the CPU-reference-blended chunk using the existing FP16 split
  reference. With `d = 0` the full-chunk fixture parity must equal today's numbers.
- Timing gate: report denoise ms with the op on and `d = 20` against the 109 ms baseline; accept
  <= +3 ms. If it misses, fuse or reduce (for example apply the matmul only on gated steps).

### P3. Producer / main-loop integration — `src/inference_pi05.cpp`, `pi05_policy.cpp`

- `ChunkMemory` lives next to `Producer`. After each accepted plan, store the normalized chunk.
  The main loop (`:1259-1340`) obtains the anchor for the next inference **before** launching it
  (needs only the planned observation time), passes it to `infer()`.
- First chunk, stale chunk (older than `elapsed + d`), or a rejected plan: mode falls back to Off
  for that inference and a counter increments (`rtc_fallbacks` in the summary JSON).
- Revisit `t0 = observation time (+ --action-offset-ms)` (`PI05_FINETUNE.md`, Driving). With a
  consistent head the first `d` rows of the new chunk match the old plan, so starting at the
  observation time and skipping the delay should still hold; confirm with the replay metric below.
- Flags (all off by default): `--rtc off|inpaint|clean`, `--rtc-rows N` (or `auto` from the
  latency budget), `--rtc-time-threshold T`, `--rtc-stability LIMIT`. Reject `clean` unless the
  bundle declares per-row-time support. Add them to `--help`, `README.md` and the summary JSON.
- Metrics (summary JSON and `--timing-csv`): `d`, anchor offset, per-chunk join error
  `max |new[0..d) - old[elapsed..elapsed+d)|` (normalized and robot units), and the discontinuity
  at the first free row `|new[d] - new[d-1]|` against the median step delta.
- Offline replay test: feed a recorded sequence of chunks through `ChunkMemory` to measure the
  boundary jump with and without Mode A, without hardware.

### P4. Mode C runtime support (engines and bundle contract)

Prepared early, **enabled only for bundles that declare it**.

- Bundle contract additions in `bundle.json` under `kaivuri`:
  `rtc: { row_time: true, style_clean: "conditioning_clean_fp16.npy", max_prefix: N,
  noise: "iid" | "correlated", covariance: "action_covariance_fp32.npy" }`. Absent means "plain
  OpenPI model": Mode C refused, Mode A allowed.
- `conditioning_clean_fp16.npy`: `[2L + 1, 3072]` style at time 0, exported next to
  `conditioning_10step_fp16.npy`. Contract checks mirror `inference_pi05.cpp:530`.
- Engines: when the templates' `mod_*` inputs are `[1, H, 3072]` the loader builds the row-wise
  mod buffers instead of single rows. Detect from the engine's input dims, not a flag, and
  `require()` that the bundle and the engines agree.
- Per-step mod assembly op (TRT, one launch per step over all 37 sites):
  `mod = style_s + m_col * (style0 - style_s)` with `style_s, style0 : [37, 1, 3072]`,
  `m_col : [1, H, 1]` (1 for prefix rows). Output `[37, H, 3072]` fp16 (~11 MB), shared by all
  layers of that step. Row mask `m_col` is set per inference from `d`.
- `build_pi05_cache.py` / cache checksums: engines with the new input shape are a **different
  cache**; keep the old cache valid for plain bundles and key the new one by the template
  `structure_sha256`.
- Mode C velocity handling: the Euler op already runs on all rows; the blend overwrites prefix
  rows afterwards, so no change to the Euler op is needed.
- Parity gate: requires the exporter's reference with a prefix (see P5). Until then, Mode C is
  covered by CPU-side tests and by a "self-consistency" run: with `d = 0` it must equal Off.

### P5. Exporter and training side (Spark; outside this repo)

The exporter and OpenPI configs live in `spark-projects` (`vla-onnx/pi05`, branch
`feat/pi05-openpi` per `PI05_FINETUNE.md`). The Spark checkout I could see from the board had no
`vla-onnx/pi05` directory, so confirm the location first.

- **Per-row AdaRMS conditioning.** OpenPI's pi0.5 builds one `adarms_cond` of shape `[B, dim]`
  from the flow time; per-row time needs `[B, H, dim]` through `embed_suffix` and the Gemma
  expert's AdaRMS layers. Read the pinned OpenPI source on the Spark and list the exact patches
  (I have not read it). Keep the patch minimal and behind a config flag so the plain model is
  unchanged.
- **Training loss for C.** Per sample draw `d ~ U{0..d_max}` (with probability `p0` set `d = 0`),
  set rows `< d` to the clean ground-truth actions with time 0, rows `>= d` as usual
  (`x_t = t·eps + (1-t)·a`), and **mask the loss on the prefix rows**. `d_max` must be at least
  the largest inference `d` and should cover the latency spread. Verify the exact recipe against
  the published training-time RTC description before fixing the loss (not read yet).
- **Correlated noise** (both modes, optional): compute the `H*D x H*D` covariance in
  `compute_norm_stats`, store it (and the Cholesky) in `norm_stats.json`/`.npy`, sample
  `eps = L z` in the noise sampler, apply shrinkage `beta = 0.5`. The runtime reads the same
  file for `Correction`. One seed across train, export and runtime avoids mismatches.
- **Exporter:** trace `mod_*` with shape `[1, H, 3072]` when `row_time` is on; also emit
  `conditioning_clean_fp16.npy` and the new bundle fields; extend `reference_comparisons` with a
  fixture that includes a prefix (clean rows, `d` rows) so the native parity gate can test Mode C.
- **Fixtures** (`prepare_pi05.py`): add the anchor, mask, `d` and noise-seeded reference output.

### P6. Validation and A/B

1. CPU unit tests pass on the Spark and on the board (`ctest -R pi05`).
2. Device parity, then timing on the board (denoise ms, total ms, `d = 0 / 10 / 20`).
3. `--rate-hz 2` paced run, 300 s: zero overruns, same RSS/PSS as the baseline within noise.
4. Offline join-metric comparison on the first real dataset: Off vs A (several `d`, `t_ip`) vs
   C. Compare boundary jump, plan smoothness, and (when a sim or rig exists) task outcome.
5. Decide defaults, then write the numbers into `README.md` and `PI05_FINETUNE.md`.

## 4. Acceptance criteria

- Off mode is **bit-identical** to today's chunks and timings are unchanged.
- Mode A device blend matches the CPU oracle to the tolerances above; denoise overhead <= 3 ms at
  `d = 20`; no new allocations after warm-up; CUDA graph still captured once.
- Mode C refuses to start on a plain bundle with a clear error; on a row-time bundle with
  `d = 0` it matches Off, with `d > 0` prefix rows equal the anchor after every step.
- Join error in Mode A/C is at the numerical floor over the `d` anchored rows; the first free row
  does not exceed the typical step delta by more than a measured margin (fix after P6.4).
- All new behavior is off by default; `ctest` is green; docs and `--help` describe every flag.

## 5. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| i.i.d.-trained model + propagation matrix mismatched to noise | Bad joins in A | Correlated noise from day one, or per-step matrices; A/B in P6 |
| Per-row `mod_*` does not trace/broadcast as assumed | C blocked | Slice axes confirmed last-axis; still re-trace and compare outputs (P5) before P4 |
| OpenPI per-token AdaRMS is invasive | C delayed | Behind a config flag; ship A first; measure the diff size first |
| 114 MB full per-row tables | Memory on 8 GB board | Use `style0` + assembly op (section 1, item 3) |
| Anchor stale when inference stalls | Wrong overlap | Fallback to Off, counter, hold rules unchanged |
| `d` too small vs real latency jitter | Visible jerk at join | Choose `d` from p99 latency; log the join metric every chunk |
| Hard anchor conflicts with a changed observation | Policy cannot self-correct | `t_ip` gate (A), small `d`; C is learned, not forced |
| Correlation `C` built for a different `H`/rate/dims | Silent corruption | Store `H, D, rate` with the covariance; `require()` on load |

Open questions to settle on the Spark: where the exporter lives and how it takes shape arguments;
whether OpenPI's Gemma AdaRMS code takes per-token conditioning with a small patch; what training
recipe the published training-time RTC uses (prefix sampling distribution, loss masking).

## 6. Quick checks to run first on the Spark

```bash
# 1. Do the action templates slice modulation on the last axis (so [1,50,3072] would slice the same)?
#    Result on 2026-10-23: every slice used axes [-1]. Needs `pip install onnx` (a venv is fine).
python3 -I - <<'E'
import onnx, onnx.numpy_helper
m = onnx.load("bundles/pi05-excavator-smoke9/templates/action.onnx", load_external_data=False)
cs = {}
for n in m.graph.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                cs[n.output[0]] = onnx.numpy_helper.to_array(a.t).tolist()
for n in m.graph.node:
    if n.op_type == "Slice" and n.input[0] in ("mod_in", "mod_post"):
        print(n.input[0], [cs.get(i, i) for i in n.input[1:]])
E
# 2. Baseline CPU tests (nothing to build for CUDA):
cmake -S native_stack -B build && cmake --build build --target pi05-tests && ctest --test-dir build -R pi05
# 3. Prepare fixtures for the smoke9 bundle (needed for any device parity run):
#    prepare_pi05.py, see README.md "Fixture preparation"
```

If a future template slices a different axis, per-row modulation needs a graph change.

## 7. Suggested order of work

P1 (host library + tests) is independent of everything else and can start immediately.
Run the section 6 checks in parallel. P2 follows P1. P3 can be developed against the replay test
before the device op lands. P4 and P5 are one workstream on the Spark and gate on the section 6
result. P6 closes both.
