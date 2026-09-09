# RTC (real-time chunking / inpainting) — implementation plan

Not started yet. This is the design to pick up from; nothing referenced here has
been written. Decisions already made with the user are marked (decided);
everything else is open for the next session.

2026-09-09: the upstream implementation has now been read (lerobot 0.5.1, installed
in `.venv-lerobot`, `lerobot/policies/rtc/`). The sections below the backend survey
were rewritten against it -- the earlier hand-rolled "hard clamp with taper" is
replaced by the real guidance term, which turns out to be *cheaper* to port than
the clamp was. See "What LeRobot officially does".

## Why

`--blend-s` and `--lag-compensation` (run_inference.py, modules/setpoint_schedule.py)
smooth the chunk handoff **after** the chunk is already sampled — `--blend-s` cross-fades
the *executed valve signal* in command space, `--lag-compensation` only re-times the
new chunk's index 0 to "now". Neither has any visibility into how the new chunk was
denoised, so the model is free to produce a first action that is nothing like where the
previous chunk actually left the machine, and blend-s just papers over the seam.

RTC ("real-time chunking", aka diffusion/flow inpainting of the handoff — Physical
Intelligence's real-time-chunking technique) instead constrains the **denoising process
itself**: the start of the new chunk is guided to be consistent with the tail of the
previous chunk during sampling, not blended afterward. This is upstream of and
complementary to blend-s/lag-comp — expected to reduce or remove the need for
`--blend-s`, but lag-compensation (timestamp alignment) is still needed regardless.

(decided) Scope: must work on **all three model paths** — SmolVLA, and both X-VLA
bundles (the deployed digging bundle and the base ee6d diagnostic export, so the
mechanism is testable in the model-only path even though `--allow-base-bundle` never
goes `--live`).

(decided) Ship behind an opt-in flag (`--rtc`), not on by default — this changes the
denoising loop of a live excavator controller and needs bench validation first.

(decided) Overlap length: auto-computed from measured inference time by default
(mirrors `--lag-compensation auto`), with a flag to force a fixed value for A/B testing.

## How the two backends denoise (already researched, see below)

### SmolVLA — `lerobot_vla/runtime/smolvla.py`

Euler ODE integration over a mutable `x_t` buffer, shape `(1, chunk_size, MAX_ACTION_DIM=32)`,
in **normalized** action space (unnormalized only at the very end):

```
dt = -1.0 / num_steps; t = 1.0
while t >= -dt/2:
    x_t += dt * velocity(x_t, t, ...)
    t += dt
return norm.unnormalize_action(x_t[0, :, :action_dim])
```

Two copies of this loop exist: `_prefill_and_denoise_feeds` (smolvla.py:541-560, numpy
KV feed) and `_prefill_and_denoise_iobind` (smolvla.py:562-615, IOBinding/CUDA KV).
`sample_actions` (smolvla.py:480) already accepts a `noise` override, drawn once in
`_initial_noise` (smolvla.py:523-527) and used as `x_t`'s initial value (t=1, pure noise).
`x_t = t*noise + (1-t)*clean_action` is the flow-matching interpolant this loop is
integrating, so a known clean target `a` at time `t` should look like `t*noise + (1-t)*a`
if we want to inject it consistently with the noise draw already burned into `x_t`.

NormStats (smolvla.py:272-299) currently only has `normalize_state` and
`unnormalize_action` — **no `normalize_action`**. Need to add it (mirror
`normalize_state`'s formula) to turn a previous chunk's real-world (unnormalized) action
values back into the space `x_t` lives in.

### X-VLA — `lerobot_vla/runtime/vendor/xvla_split_ort.py` (vendored, `XVLASplitPolicy`)

NOT Euler. Rectified-flow interpolation with a **fixed** noise draw `x1`, and the model
predicts the clean action **directly** every step (`sample_actions`, split_ort.py:601-660):

```
x1 = randn(...); action = zeros_like(x1)          # (or x1 override, already supported)
for i in range(steps, 0, -1):
    t = i / steps
    x_t = x1 * t + action * (1 - t)                # reconstructed each step, not integrated
    action = denoise_step(x_t, t, ...)             # model outputs the CLEAN estimate
```

This is actually easier to inpaint than SmolVLA: `action` *is* the clean estimate at every
step, so a known target can be blended straight into `action` — no forward-noising needed.
Normalization/unnormalization goes through `xvla_bundle_contract.normalize_vector` /
`unnormalize_vector` using `processor_contract["action"]` (split_ort.py:673-674) — need
the equivalent normalize call for a prior action before feeding it in as a known target.

`lerobot_vla/runtime/xvla.py`'s `XVLAExcavatorPolicy.sample_actions` (xvla.py:350-369) is
currently a thin pass-through with **no** noise/prior seam exposed at that layer — it will
need a `prior_actions`/`prior_offset` parameter added and threaded to the vendored call.

Vendoring note: split_ort.py carries a header documenting deltas from the upstream
`spark-projects/vla-onnx/xvla/split_ort.py` copy. Any RTC change here needs to be added to
that "MODIFICATIONS vs the original" block (split_ort.py:9-13) since it will newly diverge
from upstream — this file is presently a clean, unmodified copy of the denoise loop itself.

## What LeRobot officially does (verified: lerobot 0.5.1, `.venv-lerobot`)

Source: `lerobot/policies/rtc/{modeling_rtc,configuration_rtc,action_queue}.py`, and the
call site in `lerobot/policies/smolvla/modeling_smolvla.py:849-865`. Docs:
https://huggingface.co/docs/lerobot/rtc. Upstream of that: Physical Intelligence's
real-time-chunking-kinetix.

**Which policies.** `grep -rl RTCProcessor lerobot/policies` -> pi0, pi05, pi0_fast,
smolvla. **NOT xvla** -- lerobot ships an X-VLA policy and it has no RTC support, so the
X-VLA half of this plan has no reference implementation to copy. (Relevant to the
(decided) "all three model paths" scope; see Open questions.)

**Two horizons, not one.** The earlier draft of this plan had a single overlap length `K`.
Upstream has two, and they mean different things:

- `inference_delay` (per call, not config -- "it may vary at runtime"): how many chunk
  steps elapse *during* inference. Positions `[0, inference_delay)` are weighted **1.0** --
  hard-frozen to the previous chunk, because the machine is executing them right now and
  the new chunk cannot change them retroactively.
- `execution_horizon` (config, default 10, docs suggest 8-12): where the soft weight
  decays to 0. Positions `[inference_delay, execution_horizon)` carry the decaying weight;
  everything past `execution_horizon` is free.

`RTCProcessor.get_prefix_weights(start=inference_delay, end=execution_horizon, total=chunk_size)`
builds that: leading ones, then `torch.linspace(1, 0, n+2)[1:-1]`, then trailing zeros.
Schedules: `LINEAR`, `EXP` (`w * expm1(w) / (e - 1)`, docs recommend it), `ONES` (no decay,
full weight to `execution_horizon`), `ZEROS` (binary, hard prefix only).

**The guidance term, per denoise step** (`RTCProcessor.denoise_step`, flow-matching
convention `time` 1 -> 0, same as ours):

```
v_t  = model(x_t, t)                      # the unguided velocity
x1_t = x_t - t * v_t                      # implied clean action at this step
err  = (prev_chunk - x1_t) * weights      # prev_chunk zero-padded to chunk width
correction = d(x1_t)/d(x_t) . err         # via torch.autograd.grad
v_guided = v_t - guidance_weight * correction
```

with

```
tau = 1 - t
guidance_weight = min(max_guidance_weight, ((1-tau)/tau) * ((1-tau)^2 + tau^2)/(1-tau)^2)
                = min(max_guidance_weight, (t^2 + (1-t)^2) / (t * (1-t)))
```

`max_guidance_weight` defaults to 10.0; the docs call 10.0 optimal for 10-step flow
matching (which is our `--num-steps` default). The weight is U-shaped in `t`: it hits the
clamp at both ends (t -> 1 and t -> 0) and bottoms out at 2.0 at t = 0.5.

**The autograd call is a no-op, and that is the whole feasibility story.** In
`modeling_rtc.py` the order is:

```
v_t = original_denoise_step_partial(x_t)   # x_t does NOT require grad yet
x_t.requires_grad_(True)                   # set AFTER the forward
x1_t = x_t - time * v_t                    # v_t carries no graph back to x_t
correction = torch.autograd.grad(x1_t, x_t, err)[0]
```

so `d(x1_t)/d(x_t)` is the identity and **`correction == err` exactly**. Whether or not
that was intended (the paper's version backprops through the denoiser), it means the
shipped algorithm needs no Jacobian, no graph, no torch:

```
v_guided = v_t - guidance_weight * (prev_chunk - x1_t) * weights
```

That is a handful of numpy lines applied to the velocity our TRT `action_out` graph
already returns -- which is exactly what makes this portable to a split-engine runtime
that cannot differentiate through the model at all.

**Action space of `prev_chunk`.** `ActionQueue` keeps two queues: `original_queue` (raw
policy output, i.e. **normalized** model space, padded width) and `queue` (post-processed,
real units, what the robot executes). `get_left_over()` returns the *original* one, so
guidance happens in normalized space. Note the consequence for us: the target should be
the chunk as the model emitted it, not the clipped/slew-masked chunk
`send_action_chunk` returns.

**RTC subsumes lag compensation.** `ActionQueue._replace_actions_queue` throws away the
first `real_delay` steps of the new chunk (`original_actions[clamped_delay:]`) and restarts
the index at 0. The frozen prefix is guidance material, never executed. Our
`--lag-compensation auto` is the same thing by a different mechanism (it timestamps the
chunk from the observation so the scheduler starts at the step that is due now), so
**`--rtc` requires lag compensation to be on** -- with it off we would replay the frozen
prefix and actively re-execute the past.

## Algorithm to implement

Port the above verbatim, in numpy, into both SmolVLA denoise loops:

```
w  = prefix_weights(delay, horizon, chunk_size)[None, :, None]   # (1, chunk, 1)
gw = min(max_gw, (t**2 + (1-t)**2) / (t * (1-t)))                # inf -> max_gw
x1 = x_t - t * v_t
v_t = v_t - gw * (prev_norm - x1) * w
x_t += dt * v_t
```

Two deviations from upstream to make deliberately:

1. **Do not guide the padding columns.** `x_t` is `(1, chunk, MAX_ACTION_DIM=32)` and only
   the first 4 columns are valve commands. Upstream zero-pads `prev_chunk` to full width,
   which drags columns 4..31 toward zero. Zero the weights on those columns instead, so
   the guidance touches only real action dims.
2. **`prev_norm` is cached, not re-normalized.** Have `SmolVLASplitPolicy` keep
   `last_normalized_chunk = x_t[0, :, :action_dim].copy()` (X-VLA's vendored runtime
   already does exactly this, `last_normalized_action` / `last_model_action`,
   xvla_split_ort.py:660-672). That is upstream's `original_queue` and it removes the need
   for the `NormStats.normalize_action` the earlier draft called for -- one less place for
   an inverse to disagree with its forward.

**Chunk-size reality check.** The deployed digging bundle is chunk_size 12 at 30 fps with
~0.12 s inference -> `inference_delay` ~= 4. An `execution_horizon` of 10 would leave the
model 2 free steps out of 12. Upstream's ratio (pi0: delay ~10 of 50) is ~1/5 frozen, and
ours is already 1/3, so the sane analogue here is `execution_horizon` ~6-8, not 10. The
50-step base X-VLA export is the only bundle where the upstream defaults are literally
appropriate, and that one never goes `--live`.

## Alignment: what `prev_chunk` actually is

The previous chunk must be sliced so that **its index 0 lines up with the new chunk's
index 0**, i.e. with the observation instant (`t_obs` in the loop, run_inference.py:914).
Upstream gets this for free by calling `get_left_over()` immediately before the
observation.

Prefer asking the scheduler over doing clock arithmetic: `robot.get_setpoint_status()`
returns `chunk_pos`, the fractional index the control thread is currently playing
(modules/setpoint_schedule.py:52, `SetpointSample.chunk_pos`). Read it next to
`get_observation()`:

```
offset   = int(round(status["chunk_pos"]))          # None/exhausted -> skip RTC
prev     = prev_norm_chunk[offset:]                 # (<= chunk_size, action_dim)
delay    = round(infer_lead_s * fps)                # steps consumed during inference
```

This avoids the epoch question entirely -- run_inference times chunks with
`time.monotonic()` while `chunk_t0` is `time.perf_counter()` (run_inference.py:914, :945),
so any clock-difference approach has to reconcile those two first.

Skip RTC for a cycle when: `cycle == 0`; `chunk_pos is None` or the schedule is
`exhausted`; a teleop hand-over or task switch cut the chunk (`chunk_cut is not None`) --
in all of those there is no plan the new chunk should stay consistent with, and a task
switch is precisely the case where consistency with the old plan is *wrong*. Dry runs have
no scheduler, so fall back to the clock there (or just log that RTC is inert).

## Plumbing

Interface (documented in `lerobot_vla/policy.py:1-9`):

```
policy.sample_actions(img, task, state, prev_chunk=None, inference_delay=0)
policy.last_normalized_chunk        # (chunk_size, action_dim), None before the first call
```

- `runtime/smolvla.py`: `NormStats` unchanged; `sample_actions` (smolvla.py:480) gains the
  two kwargs and threads them into `_prefill_and_denoise`; a `_rtc_guide(v_t, x_t, t)`
  helper is called from both `_prefill_and_denoise_feeds` (smolvla.py:541-560) and
  `_prefill_and_denoise_iobind` (smolvla.py:562-615). The iobind loop inlines the
  `action_out` call rather than going through `_denoise_step`, so it needs its own call
  site -- worth factoring the guidance into one place first so the two loops cannot drift.
- `runtime/xvla.py` / `vendor/xvla_split_ort.py`: see Open questions -- no upstream
  reference exists, and the vendored loop is not an ODE integration.
- `run_inference.py:916`: compute `prev`/`delay` as above, pass them in; add the flags
  below beside the existing lag/blend ones (run_inference.py:626-647); add `rtc=` to the
  cycle log line next to `skip=` (run_inference.py:948-958) and an `rtc` field to
  `ActionLogger.log_chunk`.

Flags:

- `--rtc` (store_true) -- enable. Refuse (or force) it when `--lag-compensation` is off:
  the frozen prefix is guidance material and must not be executed.
- `--rtc-horizon N` -- `execution_horizon`. Default: auto from the bundle, ~`chunk_size`
  minus a free tail; forcing a value is what makes the A/B testable.
- `--rtc-delay N` -- force `inference_delay` instead of `round(infer_lead_s * fps)`.
- `--rtc-schedule {linear,exp,ones,zeros}` -- default `exp` (upstream's recommendation;
  their config default is still `linear` with a `# Todo change to exp` beside it).
- `--rtc-guidance-weight F` -- `max_guidance_weight`, default 10.0.

## Testing

The strongest offline check available: **lerobot 0.5.1 is installed in `.venv-lerobot`**,
so a unit test can import `RTCProcessor` and assert our numpy `prefix_weights` and guided
velocity match `get_prefix_weights` / `denoise_step` to float tolerance, for every
schedule and a grid of (delay, horizon, chunk_size, t). No Jetson, no engines, no robot --
and it pins the port to upstream rather than to our reading of it.

Then: `--synthetic --log-actions` for a chunk-continuity metric (|chunk[k] - prev[k]| at
the seam), dry run on the real robot, and only then `--live` with `--blend-s 0`.

## Open questions for next session

1. **X-VLA.** No upstream reference (lerobot's xvla policy has no RTC), and its loop is
   rectified-flow interpolation where the model returns the **clean** estimate each step,
   not a velocity -- so there is no `v_t` to correct. The algebraic equivalent of the
   guidance term for a direct-x1 predictor is `action += t * gw * w * (prev - action)`,
   but since the loop re-derives `x_t` from scratch each step rather than integrating,
   the per-step scaling is not 1:1 and would need tuning by eye. Proposal: **ship SmolVLA
   first** (it is the deployed path anyway) and treat X-VLA RTC as a separate experiment,
   rather than holding the deployed path behind a mechanism only testable on a bundle that
   never goes `--live`. This revises the (decided) all-three-paths scope -- needs the
   user's call.
2. `--rtc` and `--blend-s`: still leaning independent (don't silently change another
   flag's default), but `--rtc` should probably *warn* when `--blend-s > 0`, since RTC is
   meant to remove the reason for it.
3. Whether the frozen prefix should target the raw policy chunk or the chunk the machine
   actually got (`send_action_chunk` clips to [-1,1] and zeroes slew when disabled).
   Upstream targets the raw one. If slew is disabled, guiding toward a slew command the
   valves never received is arguably wrong -- check on the bench.
4. Multi-camera SmolVLA path (`_prefill_and_denoise` note, smolvla.py:513-517, about a
   caller building its own prefix): make sure it goes through the guided loops.
