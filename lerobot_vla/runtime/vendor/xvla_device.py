# Extracted from eetmie/jetson-orin-nano-vla @ 7474f6c.
# X-VLA-only integration; see TRT_UPSTREAM.md for local changes.
from __future__ import annotations
import ctypes
import time
import numpy as np
from .trt_engines import _ck, _cuda, _malloc
_H2D, _D2H, _D2D = 1, 2, 3
_CAPTURE_THREAD_LOCAL = 1

class DevBuf:
    """A device tensor: pointer, shape and dtype, plus a pinned host mirror on demand."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(int(x) for x in shape), np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.ptr = _malloc(self.nbytes)
        self._owns_ptr = True
        self._base = None
        self._host = None

    @classmethod
    def view(cls, base: "DevBuf", offset_bytes: int, shape) -> "DevBuf":
        """A non-owning window into `base` (e.g. one camera's tokens)."""
        v = cls.__new__(cls)
        v.shape, v.dtype = tuple(int(x) for x in shape), base.dtype
        v.nbytes = int(np.prod(v.shape)) * v.dtype.itemsize
        if offset_bytes + v.nbytes > base.nbytes:
            raise ValueError("view runs past its buffer")
        v.ptr, v._host = base.ptr + offset_bytes, None
        v._owns_ptr, v._base = False, base
        return v

    def host(self) -> np.ndarray:
        """Pinned host array of the same shape (allocated once): a fixed address the
        copy engine can DMA from and that a captured graph may reference."""
        if self._host is None:
            p = ctypes.c_void_p()
            _ck(_cuda().cudaHostAlloc(ctypes.byref(p), ctypes.c_size_t(max(self.nbytes, 1)),
                                      ctypes.c_uint(0)), "cudaHostAlloc")
            raw = (ctypes.c_byte * self.nbytes).from_address(p.value)
            self._host = np.frombuffer(raw, self.dtype).reshape(self.shape)
        return self._host

    def close(self):
        if self._host is not None:
            _ck(_cuda().cudaFreeHost(ctypes.c_void_p(self._host.ctypes.data)), "free host mirror")
            self._host = None
        if self.ptr and self._owns_ptr:
            _ck(_cuda().cudaFree(ctypes.c_void_p(self.ptr)), "free device buffer")
        self.ptr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Device:
    """Enqueue-only helpers over an Engines instance (its contexts, scratch and stream)."""

    def __init__(self, engines):
        self.e = engines
        self.trt = engines.trt
        self.stream = engines.stream
        self.cu = _cuda()
        self.capturing = False

    # -- buffers --------------------------------------------------------------
    def spec(self, name: str) -> dict:
        """{tensor: (shape, dtype, is_input)} of one engine."""
        trt = self.trt
        return {n: (shape, dt, mode == trt.TensorIOMode.INPUT)
                for n, mode, shape, dt, _key in self.e.io[name]}

    def outputs(self, name: str) -> dict[str, DevBuf]:
        return {n: DevBuf(shape, dt) for n, (shape, dt, is_in) in self.spec(name).items()
                if not is_in}

    def buf_for(self, name: str, tensor: str) -> DevBuf:
        shape, dt, _ = self.spec(name)[tensor]
        return DevBuf(shape, dt)

    def chain_outputs(self, links: list[tuple[str, str]]) -> list[dict[str, DevBuf]]:
        """Output buffers for a chain where each (engine, tensor) output is read only by
        the next engine: two buffers per shape, used alternately, instead of one per
        engine. Other outputs get their own buffers."""
        pools: dict[tuple, list[DevBuf]] = {}
        turn: dict[tuple, int] = {}
        out = []
        for name, chained in links:
            bufs = {}
            for n, (shape, dt, is_in) in self.spec(name).items():
                if is_in:
                    continue
                if n != chained:
                    bufs[n] = DevBuf(shape, dt)
                    continue
                key = (tuple(shape), np.dtype(dt).str)
                if key not in pools:
                    pools[key] = [DevBuf(shape, dt), DevBuf(shape, dt)]
                pool = pools[key]
                i = turn.get(key, 0)
                bufs[n], turn[key] = pool[i % 2], i + 1
            out.append(bufs)
        return out

    # -- one-time writes (load time; no pinned mirror kept) ------------------------
    def set_once(self, dst: DevBuf, arr: np.ndarray) -> None:
        """Synchronous copy from pageable memory, for constants written once at load:
        unlike upload(), it leaves no pinned host buffer behind."""
        x = np.ascontiguousarray(np.asarray(arr, dtype=dst.dtype).reshape(dst.shape))
        _ck(self.cu.cudaStreamSynchronize(self.stream), "sync")
        _ck(self.cu.cudaMemcpy(ctypes.c_void_p(dst.ptr), ctypes.c_void_p(x.ctypes.data),
                               ctypes.c_size_t(dst.nbytes), _H2D), "H2D once")

    def zero_once(self, dst: DevBuf) -> None:
        _ck(self.cu.cudaStreamSynchronize(self.stream), "sync")
        _ck(self.cu.cudaMemset(ctypes.c_void_p(dst.ptr), 0, ctypes.c_size_t(dst.nbytes)),
            "cudaMemset")

    # -- stream work ------------------------------------------------------------
    def upload(self, dst: DevBuf, arr: np.ndarray) -> None:
        h = dst.host()
        h[...] = np.asarray(arr, dtype=dst.dtype).reshape(dst.shape)
        self.upload_staged(dst)

    def upload_staged(self, dst: DevBuf) -> None:
        _ck(self.cu.cudaMemcpyAsync(ctypes.c_void_p(dst.ptr),
                                    ctypes.c_void_p(dst.host().ctypes.data),
                                    ctypes.c_size_t(dst.nbytes), _H2D, self.stream), "H2D")

    def download(self, src: DevBuf) -> None:
        _ck(self.cu.cudaMemcpyAsync(ctypes.c_void_p(src.host().ctypes.data),
                                    ctypes.c_void_p(src.ptr),
                                    ctypes.c_size_t(src.nbytes), _D2H, self.stream), "D2H")

    def d2d(self, dst: DevBuf, src: DevBuf, nbytes: int, dst_off: int = 0,
            src_off: int = 0) -> None:
        _ck(self.cu.cudaMemcpyAsync(ctypes.c_void_p(dst.ptr + dst_off),
                                    ctypes.c_void_p(src.ptr + src_off),
                                    ctypes.c_size_t(nbytes), _D2D, self.stream), "D2D")

    def copy2d(self, dst: DevBuf, dpitch: int, src: DevBuf, spitch: int, width: int,
               height: int, dst_off: int = 0, src_off: int = 0) -> None:
        """Strided device copy: `height` rows of `width` bytes (e.g. one half of a concat)."""
        _ck(self.cu.cudaMemcpy2DAsync(ctypes.c_void_p(dst.ptr + dst_off), ctypes.c_size_t(dpitch),
                                      ctypes.c_void_p(src.ptr + src_off), ctypes.c_size_t(spitch),
                                      ctypes.c_size_t(width), ctypes.c_size_t(height), _D2D,
                                      self.stream), "D2D 2D")

    def enqueue(self, name: str, bind: dict[str, DevBuf]) -> None:
        """Run one engine on the stream with these buffers; no synchronization."""
        ctx = self.e.contexts[name]
        for n in self.spec(name):
            ctx.set_tensor_address(n, bind[n].ptr)
        if not ctx.execute_async_v3(self.stream.value):
            raise RuntimeError(f"{name}: enqueue failed")

    def sync(self) -> None:
        _ck(self.cu.cudaStreamSynchronize(self.stream), "sync")

    # -- events -----------------------------------------------------------------
    def event(self) -> ctypes.c_void_p:
        ev = ctypes.c_void_p()
        _ck(self.cu.cudaEventCreate(ctypes.byref(ev)), "cudaEventCreate")
        return ev

    def record(self, ev) -> None:
        # External: under stream capture this becomes an event node the graph records on
        # every launch; a plain record would only be a capture-time dependency.
        if self.capturing:
            _ck(self.cu.cudaEventRecordWithFlags(ev, self.stream, ctypes.c_uint(1)),
                "cudaEventRecordWithFlags")
        else:
            _ck(self.cu.cudaEventRecord(ev, self.stream), "cudaEventRecord")

    def elapsed_ms(self, a, b) -> float:
        ms = ctypes.c_float()
        _ck(self.cu.cudaEventElapsedTime(ctypes.byref(ms), a, b), "cudaEventElapsedTime")
        return float(ms.value)

    # -- CUDA graphs ------------------------------------------------------------
    def capture(self, enqueue_fn) -> ctypes.c_void_p:
        """Capture enqueue_fn's stream work into an instantiated graph.

        Every engine must have run once uncaptured first (TensorRT may allocate on its
        first enqueue), and the bound addresses must not change afterwards.
        """
        cu = self.cu
        _ck(cu.cudaStreamBeginCapture(self.stream, ctypes.c_int(_CAPTURE_THREAD_LOCAL)),
            "cudaStreamBeginCapture")
        self.capturing = True
        graph = ctypes.c_void_p()
        try:
            try:
                enqueue_fn()
            finally:
                self.capturing = False
                rc = cu.cudaStreamEndCapture(self.stream, ctypes.byref(graph))
            _ck(rc, "cudaStreamEndCapture")
            gexec = ctypes.c_void_p()
            _ck(cu.cudaGraphInstantiate(ctypes.byref(gexec), graph, ctypes.c_ulonglong(0)),
                "cudaGraphInstantiate")
            return gexec
        finally:
            # The executable owns its graph state after instantiation. Release the
            # source graph on success and on failed enqueue/instantiation alike.
            if graph:
                _ck(cu.cudaGraphDestroy(graph), "cudaGraphDestroy")

    def destroy_graph(self, gexec) -> None:
        """Release an executable that will no longer be launched."""
        if gexec is not None:
            _ck(self.cu.cudaGraphExecDestroy(gexec), "cudaGraphExecDestroy")

    def launch(self, gexec) -> None:
        _ck(self.cu.cudaGraphLaunch(gexec, self.stream), "cudaGraphLaunch")


class XVLADevice:
    """X-VLA as one device-resident chain (see xvla_trt.infer for the host one).

    The interpolation x_t = x1 * t + action * (1 - t) runs as a small FP32 op engine
    with one weight buffer per step; the clean-action estimate stays in one buffer that
    denoise_3 rewrites each step. Prompt IDs are uploaded only when they change. The
    gripper sigmoid stays on the host, after the one download.
    """

    def __init__(self, bundle, engines, ops, graph: bool = False):
        self.b, self.d = bundle, Device(engines)
        d, bd = self.d, bundle
        self.vision_in = d.buf_for(bd.vision[0], "pixel_values")
        self.vision_out = d.chain_outputs([(n, "hidden_out") for n in bd.vision])
        feats = next(iter(self.vision_out[-1].values()))
        self.ids = d.buf_for(bd.text[0], "input_ids")
        img0 = d.spec(bd.text[0])["image_tokens"][0]
        aux = d.spec("cond")["aux_visual"][0]
        if bd.valid_views == bd.num_views:
            self.full = feats
        else:
            # Padded views read as exactly zero: a zeroed buffer the real views are
            # copied into, the rest never written.
            self.full = DevBuf((bd.num_views,) + feats.shape[1:], feats.dtype)
            d.zero_once(self.full)
        per_view = int(np.prod(feats.shape[1:])) * feats.dtype.itemsize
        self.image_tokens = DevBuf.view(self.full, 0, img0)
        self.aux = DevBuf.view(self.full, per_view, aux)
        self.text_out = d.chain_outputs([(n, "hidden_out") for n in bd.text])
        self.cond_out = d.outputs("cond")
        x_t_spec = d.spec(bd.denoise[0])["x_t"][0]
        self.x1 = DevBuf(x_t_spec, np.float32)
        self.x_t = DevBuf(x_t_spec, np.float32)
        self.action = DevBuf(x_t_spec, np.float32)
        self.proprio = d.buf_for(bd.denoise[0], "proprio")
        self.lerp = ops.get("lerp", x_t_spec)
        self.steps = []
        for i in range(bd.steps, 0, -1):
            s = i / bd.steps
            w = DevBuf((1,) * len(x_t_spec), np.float32)
            t = d.buf_for(bd.denoise[0], "t")
            d.set_once(w, np.array(s, np.float32))
            d.set_once(t, np.array([s], np.float32))
            self.steps.append((w, t))
        self.den_out = d.chain_outputs([(n, "hidden_out") for n in bd.denoise[:-1]])
        self.zeros = DevBuf(x_t_spec, np.float32)
        d.zero_once(self.zeros)
        d.sync()
        self.feats = feats
        self.task_ids = None
        self.ev = [d.event() for _ in range(4)]
        self.graph = None
        for buf in (self.vision_in, self.ids, self.proprio, self.x1):
            d.zero_once(buf)
        self._enqueue()
        d.sync()
        if graph:
            self.graph = d.capture(self._enqueue)

    def _enqueue(self) -> None:
        d, bd = self.d, self.b
        d.record(self.ev[0])
        x = self.vision_in
        for name, out in zip(bd.vision, self.vision_out):
            d.enqueue(name, {("pixel_values" if name == bd.vision[0] else "hidden_in"): x,
                             **out})
            x = next(iter(out.values()))
        if self.full is not self.feats:
            d.d2d(self.full, self.feats, self.feats.nbytes)
        d.record(self.ev[1])
        h = None
        for name, out in zip(bd.text, self.text_out):
            feeds = ({"input_ids": self.ids, "image_tokens": self.image_tokens}
                     if name == bd.text[0] else {"hidden_in": h})
            d.enqueue(name, {**feeds, **out})
            h = next(iter(out.values()))
        d.enqueue("cond", {"vlm_features": h, "aux_visual": self.aux, **self.cond_out})
        cond = self.cond_out["cond_tokens"]
        d.record(self.ev[2])
        d.d2d(self.action, self.zeros, self.action.nbytes)
        for w, t in self.steps:
            d.enqueue(self.lerp, {"x1": self.x1, "a": self.action, "w": w, "out": self.x_t})
            d.enqueue(bd.denoise[0], {"x_t": self.x_t, "t": t, "proprio": self.proprio,
                                      "cond_tokens": cond, **self.den_out[0]})
            h = next(iter(self.den_out[0].values()))
            for name, out in zip(bd.denoise[1:-1], self.den_out[1:]):
                d.enqueue(name, {"hidden_in": h, **out})
                h = next(iter(out.values()))
            d.enqueue(bd.denoise[-1], {"hidden_in": h, "action": self.action})
        d.record(self.ev[3])

    def infer(self, pixel_values, input_ids, proprio, x1, timings: dict | None = None):
        d = self.d
        t0 = time.perf_counter()
        d.upload(self.vision_in, pixel_values)
        if self.task_ids is None or not np.array_equal(input_ids, self.task_ids):
            d.upload(self.ids, input_ids)
            self.task_ids = input_ids.copy()
        d.upload(self.proprio, proprio)
        d.upload(self.x1, x1)
        if self.graph is not None:
            d.launch(self.graph)
        else:
            self._enqueue()
        d.download(self.action)
        d.sync()
        action = self.action.host().copy()
        g = self.b.gripper
        action[..., g] = 1.0 / (1.0 + np.exp(-action[..., g]))
        if timings is not None:
            ev = self.ev
            timings.update(vision=d.elapsed_ms(ev[0], ev[1]), text=d.elapsed_ms(ev[1], ev[2]),
                           cond=0.0, denoise=d.elapsed_ms(ev[2], ev[3]),
                           total=(time.perf_counter() - t0) * 1e3)
        return action


    def close(self):
        if getattr(self, "d", None) is None:
            return
        self.d.sync()
        if self.graph is not None:
            self.d.destroy_graph(self.graph)
            self.graph = None
        for ev in self.ev:
            _ck(self.d.cu.cudaEventDestroy(ev), "destroy event")
        self.ev = []
        seen = set()
        def release(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, DevBuf):
                value.close()
            elif isinstance(value, dict):
                for item in value.values(): release(item)
            elif isinstance(value, (tuple, list)):
                for item in value: release(item)
        for value in vars(self).values(): release(value)
        self.d = None
