"""X-VLA physical boundary around a direct TensorRT / CUDA graph chain.

No ONNX Runtime or PyTorch imports in inference. The saved processor contract
still owns state normalization and action unnormalization.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from .vendor.xvla_bundle_contract import normalize_vector, unnormalize_vector

LOG = logging.getLogger(__name__)


def physical_state(state, contract, model_dim):
    flat = np.asarray(state, np.float32).reshape(-1)
    expected = int(contract["state"]["dim"]) if contract else model_dim
    if flat.size != expected or not np.isfinite(flat).all():
        raise ValueError(f"State must contain {expected} finite physical axes")
    if contract:
        flat = normalize_vector(flat, contract["state"])
    proprio = np.zeros((1, model_dim), np.float32)
    proprio[0, :expected] = flat
    return proprio


def physical_actions(chunk, contract):
    action = np.asarray(chunk, np.float32)
    if contract:
        action = unnormalize_vector(action[..., :int(contract["action"]["dim"])],
                                    contract["action"])
    if not np.isfinite(action).all():
        raise ValueError("X-VLA produced nonfinite actions")
    return action


class XVLANativePolicy:
    def __init__(self, split_dir, cache_dir=None, precision="fp16", num_denoising_steps=None,
                 seed=None, tokenizer_dir=None, chain="graph"):
        from .vendor.trt_engines import Engines, prebuild_engines, _cmp
        from .vendor.xvla_trt import Bundle, validate_fixture, infer
        from .vendor.xvla_device import XVLADevice
        from .vendor.trt_ops import Ops

        if precision != "fp16":
            raise ValueError("The direct TensorRT path expects a mixed-FP16 export")
        if chain not in ("host", "device", "graph"):
            raise ValueError(f"Unsupported TensorRT chain {chain}")
        self.engines = self.device = None
        self._lock = threading.Lock()
        self.rng = np.random.default_rng(seed)
        self.bd = bd = Bundle(split_dir)
        if tokenizer_dir:
            bd.tokenizer_path = Path(tokenizer_dir)
        if num_denoising_steps is not None:
            if num_denoising_steps <= 0:
                raise ValueError("Denoising steps must be positive")
            bd.steps = num_denoising_steps
        self.steps, self.chunk_size = bd.steps, bd.chunk_size
        self.action_dim, self.model_state_dim = bd.action_dim, bd.state_dim
        self.valid_views, self.num_views = bd.valid_views, bd.num_views
        raw_contract = bd.b.get("processor_contract") or {}
        self.processor_contract = (raw_contract if raw_contract.get("physical_boundary_complete")
                                   else None)
        c = self.processor_contract
        self.state_dim = int(c["state"]["dim"]) if c else bd.state_dim
        self.real_action_dim = int(c["action"]["dim"]) if c else None
        if c and (int(c["state"]["model_dim"]) != bd.state_dim or
                  int(c["action"]["model_dim"]) != bd.action_dim):
            raise ValueError("Processor dimensions disagree with the model export")
        self.last_timings = {}
        self.chain = chain
        cache = Path(cache_dir or Path(split_dir) / "trt_native_cache").expanduser().resolve()
        self.cache_dir = cache
        try:
            # Build and validate at the export's step count. A runtime override
            # changes sampling, so the saved fixture cannot validate that override.
            configured_steps, bd.steps = bd.steps, int(bd.b["num_denoising_steps"])
            self.built = prebuild_engines(bd, cache)
            self.engines = Engines(bd, cache)
            self.fixture_parity = validate_fixture(bd, self.engines)
            if self.fixture_parity["status"] != "PASS":
                raise ValueError(f"X-VLA fixture/preprocessing mismatch: {self.fixture_parity}")
            if chain != "host":
                self.device = XVLADevice(bd, self.engines, Ops(self.engines, cache),
                                         graph=chain == "graph")
                with np.load(bd.root / bd.b["fixture"]["file"]) as r:
                    inputs = (r["pixel_values"].astype(np.float32), r["input_ids"].astype(np.int64),
                              r["proprio"].astype(np.float32), r["x1"])
                    expected = infer(bd, self.engines.run, *inputs)
                    actual = self.device.infer(*inputs)
                    report = _cmp(actual, expected)
                    stock = _cmp(actual, r["action_pred"])
                self.fixture_parity["device_vs_host"] = report
                self.fixture_parity["device_vs_stock"] = stock
                if stock["cosine"] < .999 or stock["max_pct_range"] > 1:
                    raise ValueError(f"Device chain failed stock fixture: {stock}")
                self.engines.release_buffers()
            if configured_steps != bd.steps:
                if self.device:
                    self.device.close()
                bd.steps = configured_steps
                self.device = (XVLADevice(bd, self.engines, Ops(self.engines, cache),
                                          graph=chain == "graph") if chain != "host" else None)
            LOG.info("Direct TensorRT chain=%s, shared scratch=%.1f MiB; fixture PASS",
                     chain, self.engines.scratch_bytes / 2**20)
        except BaseException:
            self.close()
            raise

    def sample_actions(self, images_hwc, instruction, state, x1=None):
        from .vendor.imaging import map_views
        from .vendor.xvla_trt import infer

        if len(images_hwc) != self.valid_views:
            raise ValueError(f"Expected {self.valid_views} views, got {len(images_hwc)}")
        for img in images_hwc:
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("X-VLA images must be HxWx3 uint8")
        with self._lock:
            if self.engines is None:
                raise RuntimeError("X-VLA policy is closed")
            start = time.perf_counter()
            pixels = np.stack(map_views(self.bd.preprocess, images_hwc))
            ids = self.bd.input_ids(instruction)
            proprio = physical_state(state, self.processor_contract, self.model_state_dim)
            shape = (1, self.chunk_size, self.action_dim)
            noise = (self.rng.standard_normal(shape).astype(np.float32) if x1 is None
                     else np.ascontiguousarray(x1, np.float32))
            if noise.shape != shape or not np.isfinite(noise).all():
                raise ValueError(f"Noise must be finite with shape {shape}")
            pre_ms = (time.perf_counter() - start) * 1000
            timings = {}
            action = (self.device.infer(pixels, ids, proprio, noise, timings)
                      if self.device is not None else
                      infer(self.bd, self.engines.run, pixels, ids, proprio, noise, timings))
            action = physical_actions(action, self.processor_contract)
            self.last_timings = {f"{key}_ms": value for key, value in timings.items()}
            self.last_timings.update(preprocess_ms=pre_ms,
                                     total_ms=(time.perf_counter() - start) * 1000)
            return action[0]

    def close(self):
        if self.device is not None:
            self.device.close()
            self.device = None
        if self.engines is not None:
            self.engines.close()
            self.engines = None
