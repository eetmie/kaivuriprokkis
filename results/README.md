# results/

Evidence from on-device runs of **fine-tuned** bundles. Base-model benchmarks live in
[jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla); that repo is
deliberately base-only, so anything measured on a digging checkpoint belongs here.

- `xvla-digging-contract/` — per-graph TensorRT placement profiles for the X-VLA digging
  bundle, `placement-profile/` against `placement-profile-fused/`. One gzipped ORT profile
  per graph (vision 0-3, text_encoder 0-2, denoise 0-3, cond). Moved here 2026-09-04 when
  the benchmark repo was scoped to base models.
