# pi0.5 chunk-to-chunk consistency (RTC) — scope and patch list

Status: **scoping only, nothing implemented.** Phased work plan: [RTC_PLAN.md](RTC_PLAN.md). `kaivuri-inference` does not implement RTC
(`README.md`: "does not implement RTC"). Each chunk is a fresh plan; `Producer` replaces the
previous plan and starts it at the observation time (`PI05_FINETUNE.md`, Driving).

## References

- pi0.5: Physical Intelligence, *π0.5: a Vision-Language-Action Model with Open-World
  Generalization*, [arXiv 2504.16054](https://arxiv.org/abs/2504.16054) (base model; flow-matching
  action expert; the thing our engines implement).
- RTC: Black, Galliker, Levine, *Real-Time Execution of Action Chunking Flow Policies*,
  [arXiv 2506.07339](https://arxiv.org/abs/2506.07339) (guided inpainting through a VJP).
- BEHAVIOR-1K 1st place: Larchenko et al., *Task Adaptation of Vision-Language-Action Model*,
  [arXiv 2512.06951](https://arxiv.org/abs/2512.06951), section 5 (correlated noise) and 7.1
  (correlation-aware soft inpainting). The mechanism is described here, not in the LeHome report.
- LeHome 2026 report: Larchenko, *Learning to Fold*, [arXiv 2606.27163](https://arxiv.org/abs/2606.27163),
  section 7 (inference knobs, Thompson-sampling tuning). Code:
  [IliaLarchenko/lehome_solution](https://github.com/IliaLarchenko/lehome_solution),
  `src/lehome_solution/models/pi_modified.py` (`_precompute_all_correction_matrices`,
  `sample_actions` step loop), `shared/eval_wrapper.py` (anchor extraction).

## Options

**A. Inference-time inpainting (training-free) — recommended first.** Runs on the existing
inference-only TensorRT engines. Needs one small device op in the denoise loop and host-side
chunk bookkeeping. **B. Guided RTC (VJP).** Needs a backward pass through the 18-layer action
expert. Our engines are FP16, per layer, with an AOT attention plugin; the board's torch is
`2.7.1+cpu`. A VJP is ~2x a forward pass, so denoise ~109 ms becomes ~300+ ms and the 500 ms
period is lost. Not practical natively. **C. Training-time RTC** (per-row timesteps) needs a
re-fine-tune and re-export; later, only if A is not enough.

## What the reference inpainting actually does (BEHAVIOR-1K / LeHome code)

Per Euler step, after `x_t_new = x_t + dt * v`, with `time_new = time + dt`:

1. `x_desired = (1 - time_new) * x0 + time_new * z` on observed positions O, where `x0` is the
   saved anchor and `z` is the **same initial noise** at those positions (stays on the flow path).
2. `delta = (x_desired - x_t_new) * mask`.
3. Hard-set the observed positions to `x_desired`.
4. Propagate: `x_U += delta_O @ C`, with `C = Sigma_UO @ Sigma_OO^-1` (Sigma_OO gets
   `1e-6 * mean(diag)` jitter). Only at free positions.
5. Skip the correction for the whole step if `max|correction| > 1.0` (stability guard).
6. All of this runs **only while `time_new > t_ip`** (0.3 default; tuned 0.4–0.5 in LeHome).
   Below that the model is free. This gate is what the papers call "soft".

Note: the mechanism is *hard replace + covariance propagation + time gating*, not a weighted
pull. Their tuning found light anchoring best; strong forced inpainting hurt.

Anchor extraction (`eval_wrapper.py`): `actions[n_e : n_e + n_a]` of the previous chunk becomes
rows `0..n_a-1` of the next target. This is a **synchronous** setup: no inference-delay
compensation. LeHome sim used `n_a` 3–6; BEHAVIOR-1K 4; the real robot used 15 (1 s of overlap).

## Patch list (what to change here)

1. **Anchor alignment for latency.** We replan asynchronously: at ~430 ms latency and 30 Hz the
   first ~13–15 rows of the old chunk will already have played when the new chunk lands. The
   anchor must be old-chunk rows `[elapsed .. elapsed + d)` mapped to new-chunk rows `0..d-1`,
   with `d` fixed from the p99 latency budget. Keep the previous *normalized* chunk and its
   observation time in `Producer`; convert via the inverse of `robot_chunk`
   (`inference_pi05.cpp:851`).
2. **Blend op in the denoise loop** (`inference_pi05.cpp` enqueue loop, after `run(euler, …)`
   near line 322): a device op doing steps 1–5 above (a row/dim mask, `x0`, `z`, and a
   `|U| x |O|` matrix multiply). Build it in-process like `build_axpy`. Step times are fixed
   (conditioning is precomputed in `styles`), so per-step `time_new` and the `t_ip` gate can be
   baked as per-step constants and the CUDA graph stays captured once; only `x0`, `z`, mask and
   `C` selection are uploaded per inference (as `noise` is today).
3. **Keep the initial noise.** `infer(…, noise)` takes host noise; the same `z` must feed both
   the initial `actions[0]` and the blend op.
4. **Correction matrices.** Size is small for us (flat dim 50x4 = 200, or 300 with tracks). Cache
   them for every anchor length we may use. Pitfall from the reference repo: its cache was built
   for `max_initial=10` while real used 15, and JAX's out-of-bounds clipping silently used the
   wrong matrix for 5 rows. Build to the largest `d` we allow and `require()` on a larger request.
5. **Noise distribution vs covariance (open question).** The reference model was *trained* with
   correlated noise `eps ~ N(0, Sigma_reg)` (Cholesky of the action covariance, shrinkage
   `beta = 0.5`), and `C` is built from that Sigma. Our openpi fine-tune uses i.i.d. noise.
   My derivation (not validated): for `x_t = t*eps + (1-t)*a` with i.i.d. eps, the right
   propagation is time-dependent, `Cov(x_U,x_O) = (1-t)^2 Sigma_UO` and
   `Cov(x_O,x_O) = t^2 I + (1-t)^2 Sigma_OO`, so use per-step matrices (10 of them, precomputed).
   Alternative: correlated noise at training time, which needs a fine-tune change and a
   statistics step in `compute_norm_stats`. Since no fine-tune exists yet, this is cheap; see
   "Greenfield" below.
6. **Producer / controller.** With a consistent head, the new chunk's first `d` rows equal the
   old chunk's, so replacement is seamless. Revisit the `t0 = observation time` skip and
   `--action-offset-ms`. Chunk coverage rules (`inference_pi05.cpp:1261–1270`) stay.
7. **New flags** (all off by default): `--inpaint-rows`, `--inpaint-time-threshold` (t_ip),
   `--inpaint-stability-limit`.
8. **Tests.** Parity of the blend op vs a NumPy/Torch reference (fixed noise, fixed anchor);
   with the mask empty the output must equal today's chunks bit-exactly; join-continuity metric
   (max |delta| across the old/new boundary) on replayed fixtures; latency regression vs the
   434 ms / 109 ms denoise baseline.

## Greenfield: no dataset or fine-tune exists yet (decided context, 2026-10-23)

The robot is custom, so the data format, fine-tune config and exporter can be shaped around
chunk consistency from the start. Consequences for the patch list:

- **Item 5 (noise) becomes a day-one choice, not a retrofit.** Train with correlated noise
  `eps ~ N(0, Sigma_reg)` and the reference propagation matrix applies as published, with one
  fixed `C` per anchor length instead of 10 per-step matrices. Needs: Sigma from our own
  dataset (extend `compute_norm_stats` to store the Cholesky of the flattened `H x D` action
  covariance, `beta = 0.5`), and the noise sampler in the OpenPI train config. Inference side is
  already host-supplied noise (`infer(…, noise)`), so only a Cholesky multiply in `Producer`.
  Caveat: the LeHome author says his architecture "was not properly ablated"; the claimed gain
  is training efficiency, not shown by a controlled test. Plan a cheap A/B (i.i.d. vs correlated)
  on the first real dataset before relying on it.
- **Sigma is tied to the action space.** Fix `H`, the action rate and the 4-vs-6 dim layout
  (tracks) *before* computing it. Changing any of them invalidates `C` and the Cholesky.
- **Pick rate and horizon with latency in mind.** The anchor length `d` is roughly
  `latency x rate`. A lower action rate (e.g. 15–20 Hz instead of 30) halves `d` and the matrix
  sizes, and 50 rows then cover a longer time. Record the dataset at the rate we intend to run.
- **Record what RTC evaluation needs.** Per-frame observation and command timestamps (already in
  the recorder format) so offline replay can measure the join discontinuity on real data.
- **Option C (training-time RTC) is no longer a retrofit.** It conditions the model on a clean
  action prefix during training (per-row timesteps), which fixes inpainting's weak point that
  the model never saw partly fixed chunks. Cost: pi0.5's AdaRMS takes one time vector per step,
  so per-row time means per-row style inputs in the action engines (bigger `styles` tables,
  exporter change) plus a custom train loss. I have not read the training-time RTC literature
  for this; verify the method and its published results before committing. Reasonable plan:
  ship A first (no export change), keep the exporter flexible enough to add C later.
- **Parity gate.** The fixture/parity flow (`prepare_pi05.py`) will need the blend op's
  reference output added when the first excavator bundle is exported.

## Not worth porting now (LeHome report)

- Advantage CFG (section 7.3): doubles the action-expert pass, ~109 -> ~218 ms denoise, ~540 ms
  total, over the 500 ms period. Also needs an advantage-conditioned model.
- Best-of-N: needs batch>1 action-expert engines and a Q-head; the board has ~5 GiB free.
- Thompson-sampling tuning of inference knobs (`n_a`, `t_ip`, …): cheap and portable once a
  success signal exists. Candidate later.
- Cubic-spline action compression (BEHAVIOR-1K 7.2): trivial host-side speed-up, independent.

## Dry-run findings (pikkumasi2tb, 2026-10-23)

- Orin Nano, ~7 GiB RAM, ~5 GiB available; torch in `~/venvs/openpi-lerobot` is CPU-only.
- `kaivuri-inference --validate-only` on `pi05-excavator-smoke9` loaded all engines and built
  the Euler op in-process, then stopped at `bad file: …/fixtures/manifest.json`: the bundle's
  fixtures need `prepare_pi05.py` first. No new timing numbers were taken.
