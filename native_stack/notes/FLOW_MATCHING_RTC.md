# Flow matching and real time chunking for native inference

The native OpenPI pi0.5 runtime already samples a flow-matching action chunk.
GR00T N1.7 also uses flow matching. Real-time chunking (RTC) adds continuity
constraints between asynchronously generated chunks. The first fine-tunes can
use standard training recipes: the original RTC method requires no training
changes. Its benefit on this excavator needs comparison against ordinary
asynchronous chunk execution.

This branch supplies tested C++ sampler math, timestamp alignment, an optional
CUDA adapter for GR00T's fused Euler engine, and analytic examples. These pieces
are not connected to a trained model or an actuator command path.

## How the models sample actions

| Property | OpenPI pi0.5 | NVIDIA GR00T N1.7 |
| --- | --- | --- |
| Initial trajectory | Gaussian noise | Gaussian noise |
| Flow time | Noise at 1, actions at 0 | Noise at 0, actions at 1 |
| Euler update | `x_next = x - v / N` | `x_next = x + v / N` |
| Reference step count | `sample_actions` defaults to 10 | Configuration defaults to 4 |
| Observation work | Image/text prefix and KV cache once per chunk | Vision/language features, state features and reusable cross-attention KV |
| Repeated work | Action expert at each noise level | Action encoder, DiT, decoder at each noise level |

Denoising steps are internal sampling iterations, not physical control ticks.
A model can generate chunks below 30 Hz while the executor consumes actions at
30 FPS. Camera history offsets are separate again; RTC does not replace GR00T's
temporal image history. Actual step counts and time embeddings belong to the
bundle contract, including any precomputed constants.

OpenPI's current public sampler and `ActionChunkBroker` do not implement the RTC
paper's guidance. The broker consumes a configured number of rows before another
policy call. Sources: [OpenPI sampler](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py),
[chunk broker](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py).

## OpenPI route from the RTC paper

The previous chunk's unexecuted tail constrains the next chunk. Weights emphasize
actions committed during inference and taper across the rest of the overlap.
Each flow step uses a vector-Jacobian product (VJP) of the estimated clean
trajectory with respect to the noisy trajectory. This operates at inference
time without changing training.
[Paper](https://www.physicalintelligence.company/download/real_time_chunking.pdf),
[authors' implementation](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py).

For OpenPI's reverse-time convention, the clean estimate is `D = x - s*v`.
The C++ interface requires a backend for `J_D(x,s)^T * error`, then subtracts
the capped guidance correction from velocity. Reversing the time convention
also reverses the guidance sign. The unit test checks this with a coupled
Jacobian and an independent finite-difference gradient.

**Engine gap:** `inference_pi05.cpp` has forward expert blocks and a negative
Euler op. Those engines provide no VJP. The scaffold rejects a missing VJP
callback; it does not substitute an identity Jacobian or label a hard-prefix
mask as the paper's method. Full native support needs a differentiated action
expert path, including attention, or a separately validated alternative.
Check the quantized forward path and derivatives together. Measure added latency
and memory on Orin before selecting this route for deployment.

Guidance encourages the prefix but does not make it exactly equal to committed
actions. Continue the accepted old plan until the reserved handover tick.

## GR00T experimental overlap primitive

NVIDIA copies the previous chunk's tail into the new sample, freezes committed
rows with zero update strength, and applies a normalized exponential ramp to
the rest of the overlap. Fresh rows have strength one. Euler integration
multiplies velocity by these strengths. No VJP is needed. This primitive exists
in the exporter's pinned N1.7 release and current upstream.
[Pinned implementation](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/gr00t/model/gr00t_n1d7/gr00t_n1d7.py).

NVIDIA describes RTC as experimental and not wired into standard `Gr00tPolicy`
or the server/client path. Its guide recommends asynchronous execution and
longer chunks (at least 32 rows). That is deployment advice, not a universal
mathematical minimum.
[Deployment guide](https://github.com/NVIDIA/Isaac-GR00T/blob/944fe70c4fd54e3fcb80a7d7258a328f9c061176/getting_started/real_world_deployment.md).

The demo's last DiT engine fuses `actions_next = actions + dt*v`. The optional
CUDA helper adapts its output after **each denoising step**:

```text
seed = previous normalized tail in overlap; Gaussian noise elsewhere
raw_next = existing_last_dit_engine(actions, observation_features, step_constants)
next = actions + strength * (raw_next - actions)
```

The kernel explicitly selects `actions` at strength zero and `raw_next` at
strength one, preserving frozen rows and disabled-RTC output exactly.
Intermediate strengths can round differently from masking velocity before
Euler integration; trained-checkpoint parity remains required. Integration needs
scratch output, a device strength buffer, a correction launch and new CUDA graph
capture. Existing transformer engines and weights can stay intact. Keep the
new runtime graph separate from verified caches. A future exporter could
instead expose velocity or accept strength as an input.
[Demo split exporter](https://github.com/eetmie/jetson-orin-nano-vla/blob/6e2d857ca5e3342fb2e29afd0cb60e295c10cdf5/export/groot/groot_split.py).

## Shared timing and action contract

`flow_rtc.hpp` describes one normalized, padded action chunk, its action grid,
checkpoint/statistics/action identity, task epoch and observation timestamp.

1. Keep one monotonic action grid across chunks. Anchor model row zero to the
   tick containing its current observation, then slice the chunk actually
   accepted for execution at that tick. Do not move the action origin forward
   to inference launch or completion. The adapter must define the mapping from
   capture time to this action grid consistently with its training timeline.
   Do not interpolate valve commands or take an arbitrary number of final rows.
2. Convert measured end-to-end inference budget to a commit tick using ceiling.
   Include observation age and the partial tick elapsed at launch. Freeze old rows before that
   tick and require the previous tail to cover the reserved interval.
3. Continue executing the accepted old plan while inference runs. Require the
   result **before** handover; reject an overrun. Existing exhausted-plan and
   stale-observation rules still apply.
4. At handover, publish the new chunk with its original action-grid start and
   current-observation timestamp. Skip elapsed rows. Preserve existing lease
   renewal behavior: renewal cannot freshen an observation.
5. Once execution accepts the new chunk, retain its model-space actions for the
   next RTC request. Unnormalization, channel gates and clamping remain in the
   command path. Material controller interventions require an RTC reset.

Use end-to-end tail latency plus margin, including capture age, preprocessing,
transfers and sampling, rather than a median GPU time. The example's 300 ms
budget is not a robot default: at 30 FPS it consumes nine rows for a fresh
observation exactly on a tick, potentially ten with a partial tick, and more
for an observation that predates launch. Dataset/model
FPS determines this, not the faster PWM or IMU update frequency.

A prompt change, manual override, model/statistics change, fault or re-arm
starts a new task epoch and clears pending results and cached RTC source.
`align_prefix` rejects cross-epoch and cross-contract requests. Bootstrap under
the normal initial-plan handling, then enable RTC after accepting a chunk.
Never keep old-task actions through an operator stop.

Preserve full normalized padded actions. Do not feed clipped PWM commands back
into a normalized sampler. For observation-relative pose actions, rebase old
targets to the new observation first. Initial absolute excavator valve commands
do not need that geometric rebase.

Freshness uses the current two IR views and IMU-derived state; old image-history
slots must not make the current observation stale. Depth and FK remain outside
initial model inputs. Arbitrary text belongs in each model's normal processor.

## First training and integration plan

1. Record dense timestamped demonstrations, arbitrary task text, two IR views
   and IMU state. Whole episodes allow offline selection of chunks and image
   history. Preserve depth/raw IMU/FK attachments for future experiments.
2. Fine-tune ordinary flow-matching policies. GR00T's base 40-row horizon and
   OpenPI's usual 50 are useful starting points if retained by the checkpoint
   contract: about 1.33 s and 1.67 s at 30 FPS. The existing LIBERO pi05 bench
   export has only 10 rows, so its timings do not establish performance for a
   50-row excavator export. Changes to horizon, FPS, padding, normalization or
   sampler steps require corresponding export/contract validation.
3. Establish native excavator-checkpoint parity, asynchronous execution,
   stale-result rejection and immediate operator takeover.
4. Wire GR00T's seed and CUDA correction into its native sampler. First prove
   zero-RTC behavior unchanged, then test frozen rows at every denoising step
   and compare full chunks to NVIDIA's primitive with identical noise.
5. Evaluate original-paper OpenPI guidance in a derivative-capable reference,
   then profile native VJP support. Keep ordinary forward-only execution as the
   comparison baseline while resolving that gap.
6. Compare baseline and RTC at matched observations, noise, sampling steps and
   execution timing. Inspect chunk discontinuities, valve reversals, deadline
   overruns and task success before increasing physical authority.

Prefix-conditioned **training** is a separate experiment, not an original-RTC
prerequisite. It needs a specified masking/noise-time recipe, matching sampling
and export support, and validation of its effect on pretrained behavior. Do not
silently introduce it into the initial fine-tune.

## Build and exercise the scaffold

Board worktree: `/home/joel/kaivuriprokkis-rtc-scaffold`, branch
`feat/flow-rtc-scaffold`. Existing pi05 and GR00T worktrees are untouched.

```bash
cmake -S native_stack -B native_stack/build-rtc \
  -DBUILD_CAMERA=OFF -DBUILD_INFERENCE=OFF -DBUILD_FLOW_RTC_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build native_stack/build-rtc \
  --target flow-rtc-tests flow-rtc-probe flow-rtc-cuda-tests --parallel 2
ctest --test-dir native_stack/build-rtc -R '^flow-rtc' --output-on-failure
native_stack/build-rtc/flow-rtc-probe groot17
native_stack/build-rtc/flow-rtc-probe openpi
```

Leave `BUILD_FLOW_RTC_CUDA` off for CPU-only builds. These targets never open a
control socket. Examples use an analytic velocity field, not model weights.
Production adapters should keep step tensors on the GPU; the host functions
are numerical references, not a suggested per-step download path.

`tools/check_flow_rtc_upstream.py` checks these examples against NVIDIA's actual
pinned seed block and PI's prefix-weight function with an independent PyTorch
autograd forward-time sampler. It requires CPU torch/numpy and explicitly
supplied trusted source files. Source revisions inspected:

| Repository | Revision |
| --- | --- |
| Physical-Intelligence/openpi | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Physical-Intelligence/real-time-chunking-kinetix | `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b` |
| NVIDIA/Isaac-GR00T main | `944fe70c4fd54e3fcb80a7d7258a328f9c061176` |
| NVIDIA/Isaac-GR00T export pin | `23ace64f17aa5015259b8609d371eb61a357c776` |
| eetmie/jetson-orin-nano-vla | `6e2d857ca5e3342fb2e29afd0cb60e295c10cdf5` |

```bash
python3 -B native_stack/tools/check_flow_rtc_upstream.py \
  --probe native_stack/build-rtc/flow-rtc-probe \
  --groot-source /path/to/pinned/gr00t/model/gr00t_n1d7/gr00t_n1d7.py \
  --rtc-source /path/to/real-time-chunking-kinetix/src/model.py
```

Use a CPU torch/numpy environment for this optional check. The board's existing
`/home/joel/venvs/openpi-lerobot/bin/python` was used here. Analytic end-to-end
sampler differences were `5.0664e-07` for GR00T and `4.3213e-07` for OpenPI
(160 elements each, including the probe's printed-value rounding).

Verification on 2026-10-26: 2,269 CPU checks pass, including local ASan/UBSan;
all four board CTests pass. The CUDA test checks 5,280 elements, graph capture,
in-place output and exact disabled-RTC behavior. Maximum difference between
post-step reconstruction and direct masked FP32 Euler in that analytic test is
`1.19209e-07`. This validates helpers, not trained-checkpoint parity or physical
task improvement.
