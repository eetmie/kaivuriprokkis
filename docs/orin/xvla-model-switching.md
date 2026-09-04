# Switching between SmolVLA and X-VLA at the robot

Goal: one inference entry point where the model is a flag, not a code path.

## The caveat that has to come first

**`lerobot/xvla-base` cannot drive the excavator.** It is a base checkpoint in the `ee6d`
action space — 20 dimensions of end-effector xyz + 6D rotation + gripper, for arms — while
the MASI excavator is 4 channels of normalized valve command (slew, boom, arm, bucket). Its
soft prompts encode the seven pretraining platforms, none of which is an excavator.

So this runtime proves *feasibility*, not deployability, exactly as `smolvla-runtime` did
with the `ainekko` base weights before the fine-tune landed.

## Checklist to get X-VLA onto the excavator

Everything below the line is done and measured; everything above it is not.

**On the Spark (the long pole — nothing else can start until this exists):**

1. Fine-tune from `lerobot/xvla-base` on the existing
   `kaivuriprokkis/lerobot_vla/record_episodes.py` dataset (same lerobot version family,
   1 camera, 4-dim state/action) with `--policy.action_mode=auto` — it reads
   `action_feature.shape[-1] = 4`, pads to the model's 20, and computes loss only on the
   real dims — plus `--policy.dtype=bfloat16` and a new domain id.
2. Record `fps` and the ACTION `MEAN_STD` normalisation stats with the checkpoint. Both are
   needed on this side and neither is in `xvla-base`.

**On the Orin (each step already works; ~30 min total, mostly the engine build):**

3. `tools/export_split_onnx.py --checkpoint <finetuned> --domain-id <new> --valid-views 1`
4. `tools/fp16_weights.py --split-dir exports/split --out-dir exports/split_fp16`
5. `prebuild_engines('exports/split_fp16', ...)` — ~5 min cold, 69 s warm
6. `parity.py --split-dir exports/split_fp16` — must pass before anything drives a valve
7. `run_pipeline.py --split-dir exports/split_fp16 --duration-s 300` for latency + memory

**Then the glue that does not exist yet:**

8. An X-VLA adapter beside `smolvla_split.py`: action unnormalise (MEAN_STD), trim 20 → 4
   dims, and the `fps` handling `resolve_policy_fps` expects.
9. Measure the *combined* footprint with the D435i reader and the 100 Hz controller in the
   same process. The policy alone is ~6.1 GB of 7.4 GB, so this is the one number that
   could still say no.

The dataset from `kaivuriprokkis/lerobot_vla/record_episodes.py` is already the right
shape for this — same lerobot version family, one camera, 4-dim state and action.

**The exporter must start recording `fps` in `bundle.json`.** `run_inference.py`
(`resolve_policy_fps`) treats the playback rate as a property of the checkpoint and refuses
an `--fps` that contradicts the bundle, precisely because a chunk is *rate* commands: a
model trained on 10 fps data replayed at 30 Hz moves the machine at a third speed and
nothing about that failure is loud. X-VLA's `bundle.json` currently has no `fps` field —
`xvla-base` has no training rate for our robot — so the fine-tune export has to copy it
across, the way `export_info.json` does on the SmolVLA side.

That rate also decides whether ~490 ms is comfortable. A 30-action chunk is 3 s of motion
at 10 fps (16% duty — plenty) but only 1 s at 30 fps (49% duty — workable, though with
less slack than SmolVLA's 50-action chunk gives).

## The switch itself

Both policies already expose the same shape of call, so the adapter is thin:

| | SmolVLA | X-VLA |
|---|---|---|
| class | `SmolVLASplitPolicy` (`lerobot_vla/smolvla_split.py`) | `XVLASplitPolicy` (`xvla_runtime/split_ort.py`) |
| images | one `HxWx3` uint8 | list of `HxWx3` uint8, one per real camera |
| chunk | 50 actions | 30 actions |
| model action dim | 32 (padded, 4 used) | 20 (`ee6d`, 4 used after fine-tune) |
| engines | 9 | 12 |

A common wrapper in `lerobot_vla` — `make_policy(name, **kw)` returning something with
`sample_actions(images, instruction, state) -> (chunk, action_dim)` — is enough, with
`run_inference.py --model {smolvla,xvla}` selecting it. Normalization stats and the
action-dim trim stay inside each adapter, since they differ per model.

**Do not load both at once.** Measured: X-VLA's twelve sessions alone sit at ~5.7 GB RSS
with ~0.47 GB free on this 7.4 GB board. SmolVLA's nine engines do not fit alongside that,
so the switch is a **restart-level choice, not a runtime toggle**, and `make_policy` should
say so rather than leave it to be discovered by an OOM mid-run.

Engine caches stay separate. X-VLA's now defaults to `exports/split/trt_cache` (persistent,
so the ~5 min cold build is paid once per export); SmolVLA's is still `/tmp/smolvla_split_cache`
and is therefore rebuilt every boot — worth moving for the same reason.

## Cost of the switch, once both are deployable

| | SmolVLA | X-VLA |
|---|---:|---:|
| chunk latency | 210–240 ms | ~490 ms |
| actions per chunk | 50 | 30 |
| replan rate | ~4.5 Hz | ~2.0 Hz |
| resident | 9 engines | 12 engines, ~5.7 GB |

X-VLA buys whatever its larger backbone and cross-embodiment pretraining are worth, and
costs roughly 2x the latency for 0.6x the actions per chunk. At a 10 Hz control rate a
30-action chunk is 3 s of motion against a ~0.5 s replan, so it is comfortable open-loop —
but it re-plans less often than SmolVLA, which matters more for reactive work than for
smooth trajectories. `--steps` (`num_denoising_steps`) is the dial: it is 84% of the
runtime, so halving it nearly halves the gap.
