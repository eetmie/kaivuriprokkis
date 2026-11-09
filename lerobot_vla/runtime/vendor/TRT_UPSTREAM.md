# X-VLA TensorRT runtime source

Extracted from https://github.com/eetmie/jetson-orin-nano-vla at commit
`7474f6c` (2026-10-27). The source repo is left unchanged.

- `trt_engines.py`: generic CUDA, engine cache/build and shared scratch helpers
  extracted from `bench/vendor/groot_trt.py`; unrelated model code omitted.
- `xvla_device.py`: `DevBuf`, `Device`, `XVLADevice` from `trt_device.py`;
  unrelated model chains omitted.
- `trt_ops.py`: only X-VLA's interpolation operation retained.
- `imaging.py`, `xvla_trt.py`: host preprocessing and X-VLA graph orchestration.

Local changes: imports and builder subprocess module paths; explicit device,
pinned-host, event, graph, stream and scratch cleanup; initialized warmup
buffers; runtime-specific cache keys for the interpolation engine; generic
`auto` action mode; tokenizer.json via Rust tokenizers rather than transformers;
mandatory preprocessing and raw-action fixture gates. The OpenCV/PyTorch
bilinear numerical floor is gated at 5e-4 normalized image units (the retained
640x480 base fixture's max is 2.76e-4), and raw-image action output must also
pass the source runtime's full-chunk tolerance.

`runtime/xvla_native.py` adds physical state/action processor boundaries,
shape checks, checkpoint-derived view/horizon sizes and serialized inference.
No three-camera/horizon-specific Triton AOT kernel is included.
