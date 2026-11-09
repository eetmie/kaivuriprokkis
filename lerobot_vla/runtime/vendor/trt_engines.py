# Extracted from eetmie/jetson-orin-nano-vla @ 7474f6c.
# X-VLA-only integration; see TRT_UPSTREAM.md for local changes.
from __future__ import annotations
import ctypes
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

FIXTURE_MAX_PCT_RANGE = 1.0
FIXTURE_MIN_COSINE = 0.999
_cudart = None
_H2D, _D2H = 1, 2
ACCUMULATE = "fp32"

def _cuda():
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so.13")
    return _cudart


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: cudaError {rc}")


def _malloc(n: int) -> int:
    p = ctypes.c_void_p()
    _ck(_cuda().cudaMalloc(ctypes.byref(p), ctypes.c_size_t(max(n, 1))), f"cudaMalloc({n})")
    return p.value


def _trt_version() -> str:
    # Package metadata, not `import tensorrt`: the builder runs in child processes and
    # this one should not hold the library while they do.
    from importlib import metadata
    for dist in ("tensorrt", "tensorrt_cu13", "tensorrt-cu13"):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    import tensorrt
    return tensorrt.__version__


def _cudart_version() -> int:
    v = ctypes.c_int()
    _ck(_cuda().cudaRuntimeGetVersion(ctypes.byref(v)), "cudaRuntimeGetVersion")
    return v.value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def accumulate_fp32(net) -> int:
    """Rewrite each FP16 x FP16 MatrixMultiply as Cast(FP32) -> MatMul -> Cast(FP16).

    The operands are FP16 values, so every product is exact in FP32; TensorRT builds the
    pattern as FP16-input, FP32-accumulate tensor-core kernels (pi0.5's exporter writes
    its graphs this way). On SmolVLA: +1.1 % latency, action error -21..39 %.
    Returns how many MatMuls were rewritten.
    """
    import tensorrt as trt

    layers = [net.get_layer(i) for i in range(net.num_layers)]
    users: dict[str, list] = {}
    for layer in layers:
        for j in range(layer.num_inputs):
            t = layer.get_input(j)
            if t is not None:
                users.setdefault(t.name, []).append((layer, j))
    outputs = {net.get_output(i).name for i in range(net.num_outputs)}
    count = 0
    for layer in layers:
        if layer.type != trt.LayerType.MATRIX_MULTIPLY:
            continue
        a, b = layer.get_input(0), layer.get_input(1)
        if a.dtype != trt.float16 or b.dtype != trt.float16:
            continue
        layer.__class__ = trt.IMatrixMultiplyLayer
        mm = net.add_matrix_multiply(net.add_cast(a, trt.float32).get_output(0), layer.op0,
                                     net.add_cast(b, trt.float32).get_output(0), layer.op1)
        mm.name = f"{layer.name}_fp32acc"
        y = net.add_cast(mm.get_output(0), trt.float16).get_output(0)
        old = layer.get_output(0)
        for consumer, j in users.get(old.name, []):
            consumer.set_input(j, y)
        if old.name in outputs:
            name = old.name
            net.unmark_output(old)
            old.name = f"{name}_fp16acc"
            y.name = name
            net.mark_output(y)
        count += 1
    return count


def configure_accumulate(net, cfg, mode: str) -> int:
    """Apply an --accumulate mode to a parsed network and its builder config."""
    import tensorrt as trt

    if mode == "auto":
        return 0
    if mode != "fp32":
        raise ValueError(f"accumulate must be fp32 or auto, not {mode!r}")
    cfg.clear_flag(trt.BuilderFlag.TF32)
    return accumulate_fp32(net)


def build_one(onnx_path: str, engine_path: str, timing_cache: str, opt_level: int,
              workspace_mb: int, accumulate: str = "auto") -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Strongly typed: the mixed-FP16 ONNX already says which ops stay FP32 (norms,
    # softmax, the RMSNorms, the time sinusoids). A weakly typed FP16 build may re-pick
    # them in FP16, and a Qwen3 massive activation squared in FP16 overflows.
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(onnx_path):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    cfg.builder_optimization_level = opt_level
    configure_accumulate(net, cfg, accumulate)
    tc = Path(timing_cache)
    cache = cfg.create_timing_cache(tc.read_bytes() if tc.exists() else b"")
    cfg.set_timing_cache(cache, ignore_mismatch=False)
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise SystemExit(f"build failed: {onnx_path}")
    Path(engine_path).write_bytes(blob)
    tc.write_bytes(cfg.get_timing_cache().serialize())


def prebuild_engines(bundle: Bundle, cache_dir: str | Path, opt_level: int = 2,
                     workspace_mb: int = 512, accumulate: str | None = None) -> dict:
    """Build every missing or stale engine, one subprocess each, serially.

    An engine is keyed by the sha256 of the ONNX it came from plus the TensorRT and CUDA
    versions and the builder options (`<name>.onnx.sha256` beside it), so neither a
    re-exported bundle nor an upgraded stack can run against an old engine. Serial,
    one process per graph: two resident TensorRT builders OOM this board.
    opt_level 2 / 512 MB workspace is what every GR00T build was measured with: each
    peaked at 3.3 GB RSS with >= 3.9 GB still available.
    """
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    accumulate = accumulate or ACCUMULATE
    stack = (f"tensorrt={_trt_version()} cudart={_cudart_version()} "
             f"opt_level={opt_level} workspace_mb={workspace_mb} strongly_typed")
    if accumulate != "auto":            # auto keeps the key engines were built under before
        stack += f" accumulate={accumulate}"
    onnx_path_of = getattr(bundle, "onnx_path", lambda n: bundle.root / f"{n}.onnx")
    built = {}
    for name in bundle.names:
        onnx_path = onnx_path_of(name)
        sha = f"{_sha256(onnx_path)} {stack}"
        eng, key = cache / f"{name}.engine", cache / f"{name}.onnx.sha256"
        digest = cache / f"{name}.engine.sha256"
        if eng.exists() and key.exists() and key.read_text().strip() == sha:
            actual = _sha256(eng)
            if digest.exists() and digest.read_text().strip() != actual:
                raise ValueError(f"Engine bytes changed: {eng}. Use --rebuild.")
            if not digest.exists():
                digest.write_text(actual + "\n")
            continue
        t0 = time.time()
        subprocess.run(
            [sys.executable, "-c",
             "import sys; from lerobot_vla.runtime.vendor.trt_engines import build_one; "
             "build_one(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), "
             "sys.argv[6])",
             str(onnx_path), str(eng), str(cache / "timing.cache"), str(opt_level),
             str(workspace_mb), accumulate],
            check=True, cwd=str(Path(__file__).resolve().parents[3]))
        key.write_text(sha + "\n")
        digest.write_text(_sha256(eng) + "\n")
        built[name] = round(time.time() - t0, 1)
        print(f"[trt] built {name} in {built[name]} s", flush=True)
    return built


class Engines:
    def __init__(self, bundle: Bundle, cache_dir: str | Path, skip=()):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        cache = Path(cache_dir).expanduser()
        self.engines, self.contexts = {}, {}
        for name in bundle.names:
            if name in skip:
                continue
            blob = (cache / f"{name}.engine").read_bytes()
            self.engines[name] = self.runtime.deserialize_cuda_engine(blob)
            del blob
            if self.engines[name] is None:
                raise RuntimeError(f"could not deserialize {name}.engine")
        self.scratch_bytes = max(e.device_memory_size_v2 for e in self.engines.values())
        self.scratch = _malloc(self.scratch_bytes)
        for name, e in self.engines.items():
            ctx = e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            ctx.set_device_memory(self.scratch, self.scratch_bytes)
            self.contexts[name] = ctx
        self.stream = ctypes.c_void_p()
        _ck(_cuda().cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        # Device buffers shared by tensor name+shape; an input whose host array is the
        # same object as the last upload (vl, biases) is not re-copied. The object is
        # held, not its id(): a freed array's id can be reused.
        self.buffers: dict[tuple, int] = {}
        self.last_upload: dict[tuple, object] = {}
        self.io = {n: self._io(e) for n, e in self.engines.items()}

    def add(self, name: str, path) -> None:
        """Load one more engine onto the shared scratch (the chains' small op engines)."""
        trt = self.trt
        e = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if e is None:
            raise RuntimeError(f"could not deserialize {path}")
        if e.device_memory_size_v2 > self.scratch_bytes:
            raise RuntimeError(f"{name} needs more scratch than the shared buffer")
        ctx = e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        ctx.set_device_memory(self.scratch, self.scratch_bytes)
        self.engines[name], self.contexts[name] = e, ctx
        self.io[name] = self._io(e)

    def _io(self, e):
        trt = self.trt
        out = []
        for i in range(e.num_io_tensors):
            n = e.get_tensor_name(i)
            shape = tuple(e.get_tensor_shape(n))
            dt = np.dtype(trt.nptype(e.get_tensor_dtype(n)))
            mode = e.get_tensor_mode(n)
            key = (n if mode == trt.TensorIOMode.INPUT else "out:" + n, shape, dt.str)
            out.append((n, mode, shape, dt, key))
        return out

    def release_buffers(self) -> None:
        """Free run()'s staging buffers. A device-resident chain allocates its own, and
        these (allocated on first use, e.g. by the load-time host-chain check) would
        otherwise stay resident beside them."""
        for ptr in self.buffers.values():
            _ck(_cuda().cudaFree(ctypes.c_void_p(ptr)), "cudaFree")
        self.buffers.clear()
        self.last_upload.clear()

    def close(self):
        if not getattr(self, "stream", None):
            return
        _ck(_cuda().cudaStreamSynchronize(self.stream), "sync on close")
        self.release_buffers()
        self.contexts.clear()
        self.engines.clear()
        if self.scratch:
            _ck(_cuda().cudaFree(ctypes.c_void_p(self.scratch)), "free scratch")
            self.scratch = None
        _ck(_cuda().cudaStreamDestroy(self.stream), "destroy stream")
        self.stream = None
        self.runtime = None

    def run(self, name, feeds):
        trt, cu = self.trt, _cuda()
        ctx, outs = self.contexts[name], {}
        for n, mode, shape, dt, key in self.io[name]:
            if key not in self.buffers:          # allocated on first use
                self.buffers[key] = _malloc(int(np.prod(shape)) * dt.itemsize)
            ptr = self.buffers[key]
            ctx.set_tensor_address(n, ptr)
            if mode == trt.TensorIOMode.INPUT:
                src = feeds[n]
                if self.last_upload.get(key) is src:
                    continue
                x = np.ascontiguousarray(src, dtype=dt)
                assert x.shape == shape, (name, n, x.shape, shape)
                _ck(cu.cudaMemcpyAsync(ctypes.c_void_p(ptr), x.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_size_t(x.nbytes), _H2D, self.stream), "H2D")
                self.last_upload[key] = src
            else:
                outs[n] = (ptr, np.empty(shape, dt))
        if not ctx.execute_async_v3(self.stream.value):
            raise RuntimeError(f"{name}: execute failed")
        for n, (ptr, arr) in outs.items():
            _ck(cu.cudaMemcpyAsync(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(ptr),
                                   ctypes.c_size_t(arr.nbytes), _D2H, self.stream), "D2H")
        _ck(cu.cudaStreamSynchronize(self.stream), "sync")
        return {n: arr for n, (ptr, arr) in outs.items()}


def _cmp(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = np.abs(a - b)
    rng = float(b.max() - b.min()) or 1.0
    return {"cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "max_pct_range": float(d.max() / rng * 100)}

