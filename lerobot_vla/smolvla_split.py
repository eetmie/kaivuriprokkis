"""SmolVLA split-engine inference — ONNX Runtime + TensorRT EP, numpy only.

Runs the 9-graph split export produced by spark-projects/smolvla-spark-finetune/
export_split_onnx.py, with the flow-matching denoise loop in Python:

    vision (x per camera) + text + state_proj  ->  prefix embeddings [1,177,960]
    expert_prefill (once)                      ->  KV cache
    for each of num_steps:  action_in + time MLP -> expert_decode -> action_out
                            x_t += dt * v_t

This is why it fits the Orin Nano's 8 GB: each engine carries only its slice
of the 450M weights, so the TRT builds peak at ~2-3 GB instead of the ~7 GB
that killed the monolithic export (notes/findings.md, 2026-06-16/17).

Orchestration mirrors github.com/aifoundry-org/ETARS modeling_smolvla_ort.py,
checked against lerobot 0.5.1 modeling_smolvla.py. The prefix length is read
from the prefill graph at load time — it depends on how many camera slots the
bundle was exported with (ainekko base: 2 slots -> 177 = 2x64 + 48 lang + 1
state; our single-camera excavator export: 1 slot -> 113). The chunk length is
likewise read from the decode graph (50 for the base export, 12 for the deployed
digging bundle); padded action dim 32.

Heavy graphs (vision / prefill / decode) run on the TensorRT EP with an
on-disk engine cache. The text embedding and the state projector stay on the
CPU EP -- they run once per inference. The four PER-STEP projectors and the
denoise loop's KV feeds do not, and both are handled on the GPU by default:

  * `projectors="gpu"` re-creates action_in / time_in / time_out / action_out
    on the TRT->CUDA stack. TensorRT declines graphs this small (they fall
    under trt_min_subgraph_size) and they run on the CUDA EP -- that is the
    expected placement, not a failure.
  * `iobinding=True` binds the prefill's KV cache straight to device and keeps
    it there for the whole denoise loop, instead of round-tripping 7.2 MB of
    numpy through the host on every one of the N steps.

Both were validated on the board in github.com/eetmie/jetson-orin-nano-vla
(docs/06-optimization-backlog.md), and re-measured here against the deployed
smolvla-digging-clean-ir12-35k bundle: 147.8 -> 121.6 ms p50 (and 176.9 ->
122.0 p95), with 2.05 of the six CPU cores handed back to the control stack.

Parity against the previous cpu/off default, same observation and same noise:
IOBinding is EXACTLY bit-identical (max abs 0.000e+00) in both projector
configurations. The GPU projectors account for all of the difference -- max
abs 3.26e-4, 0.042% of the action range, cosine 0.999999940 -- which is
CUDA-EP vs CPU-EP arithmetic on tiny matmuls, well inside this project's
1%-of-range gate. `projectors="cpu"` / `iobinding=False` reproduce the old
loop exactly, for regression comparisons.

NOT taken from that benchmark: the NaN-guard-stripped vision graph (faster,
but still outside the repo's own parity gate) and a reduced `num_steps`
(faster, but it changes the policy rather than its runtime).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

LOG = logging.getLogger("smolvla_split")

# Static dims of the split export (verified from the ONNX graphs).
VLM_DIM = 960
EXPERT_DIM = 720
IMG_TOKENS = 64
# Camera slots (and with them the prefix length) are baked into the prefill
# graph at export time and DIFFER between bundles: the ainekko base export has
# 2 slots (prefix 64+64+48+1 = 177), our fine-tuned excavator export has 1
# (64+48+1 = 113). The real value is read from the graph in __init__;
# this is only the fallback when the graph axis is dynamic.
DEFAULT_NUM_CAM_SLOTS = 2
LANG_LEN = 48
# The action chunk length is baked into the expert-decode graph at export time and
# DIFFERS between bundles: lerobot/smolvla_base and every excavator export so far use
# 50, but a bundle trained with --policy.chunk_size=12 or 30 carries that instead. The
# real value is read from the graph in __init__; this is only the fallback when the
# graph axis is dynamic. Feeding a 50-step suffix to a 12-step graph is a shape error,
# and feeding 12 to a 50-step graph would silently under-drive the action head.
DEFAULT_CHUNK_SIZE = 50
MAX_ACTION_DIM = 32
MAX_STATE_DIM = 32
IMG_SIZE = 512
MIN_PERIOD = 4e-3
MAX_PERIOD = 4.0

_DEFAULT_TRT_WORKSPACE = 1 << 30   # 1 GiB
_DEFAULT_CUDA_MEM_LIMIT = 3 << 30  # 3 GiB

# The four projectors the denoise loop calls ONCE PER STEP -- ten host round
# trips each per inference at the default budget, which is what makes them
# worth moving to the GPU. `smolvlm_text` and `state_projector` run once per
# inference and stay on the CPU EP.
_PER_STEP_PROJECTORS = {
    "action_in": "action_in_projector.onnx",
    "time_in": "time_in_projector.onnx",
    "time_out": "time_out_projector.onnx",
    "action_out": "action_out_projector.onnx",
}


def build_providers(cache_dir: str, precision: str = "fp16"):
    """TensorRT EP -> CUDA EP -> CPU EP, engine cache on disk.

    Same stack smolvla-runtime validated on this board (backends/ort.py).
    """
    os.makedirs(cache_dir, exist_ok=True)
    trt_opts = {
        "device_id": 0,
        "trt_fp16_enable": precision == "fp16",
        "trt_bf16_enable": precision == "bf16",
        "trt_layer_norm_fp32_fallback": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache_dir,
        "trt_max_workspace_size": int(os.environ.get("TRT_WORKSPACE_MB", "1024")) * (1 << 20),
        "trt_min_subgraph_size": 5,
    }
    # Lower optimization level explores fewer tactics -> smaller build peak.
    # Only affects the one-time build; a cached engine reloads identically.
    if os.environ.get("TRT_OPT_LEVEL"):
        trt_opts["trt_builder_optimization_level"] = int(os.environ["TRT_OPT_LEVEL"])
    cuda_opts = {
        "device_id": 0,
        "gpu_mem_limit": _DEFAULT_CUDA_MEM_LIMIT,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "do_copy_in_default_stream": True,
    }
    providers: list = [("TensorrtExecutionProvider", trt_opts)]
    if not os.environ.get("TRT_DROP_CUDA_EP"):
        providers.append(("CUDAExecutionProvider", cuda_opts))
    providers.append("CPUExecutionProvider")
    return providers


def make_att_2d_masks(pad_masks: np.ndarray, att_masks: np.ndarray) -> np.ndarray:
    """SmolVLA's blockwise-causal 2D attention mask from 1D pad + att masks."""
    cumsum = np.cumsum(att_masks, axis=1)
    att_2d = cumsum[:, np.newaxis, :] <= cumsum[:, :, np.newaxis]
    pad_2d = pad_masks[:, np.newaxis, :] & pad_masks[:, :, np.newaxis]
    return att_2d & pad_2d


def sinusoidal_time_embedding(t: float, dim: int = EXPERT_DIM) -> np.ndarray:
    fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** fraction
    sin_input = (2 * math.pi / period) * t
    return np.concatenate([np.sin(sin_input), np.cos(sin_input)]).astype(np.float32)


def resize_with_pad_uint8(img_hwc: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """uint8 HxWx3 -> float32 [1,3,size,size] in [-1,1].

    Matches lerobot's resize_with_pad: keep aspect, bilinear, pad LEFT and TOP
    with 0 (in [0,1] space, i.e. -1 after the SigLIP [-1,1] normalization).
    """
    import cv2

    h, w = img_hwc.shape[:2]
    ratio = max(w / size, h / size)
    rw, rh = int(w / ratio), int(h / ratio)
    resized = cv2.resize(img_hwc, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[size - rh:, size - rw:] = resized          # pad top + left
    chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return (chw * 2.0 - 1.0)[None]


HEAVY_GRAPHS = ("smolvlm_vision.onnx", "smolvlm_expert_prefill.onnx",
                "smolvlm_expert_decode.onnx")


def _bundle_fingerprint(split_dir: Path) -> str:
    """Identify the WEIGHTS of a bundle, not just its path.

    A bundle re-exported in place keeps its directory name, so the path alone
    would happily serve the previous checkpoint's engines. Size + mtime of each
    heavy graph and its external-weights sidecar is enough to notice, and costs
    six stats rather than 1.4 GB of hashing.
    """
    parts = []
    for name in HEAVY_GRAPHS:
        for f in (split_dir / name, split_dir / f"{name}.data"):
            st = f.stat() if f.exists() else None
            parts.append(f"{f.name}:{st.st_size if st else 0}:"
                         f"{int(st.st_mtime) if st else 0}")
    return "\n".join(parts) + "\n"


def prepare_cache(split_dir: str | Path, cache_root: str | Path,
                  rebuild: bool = False) -> Path:
    """Give this bundle its OWN engine-cache directory, and invalidate it.

    The engine cache used to be one flat directory shared by every bundle, which
    is why a second model could not be run after a first: `prebuild_engines`
    treats "three engines are already here" as "this bundle is built", so the
    new bundle skipped its per-graph subprocess builds and tried to build all
    three graphs inside the inference process — the 8 GB OOM the subprocess
    split exists to avoid.

    Keyed on the resolved path, so switching back to a model reuses its engines
    instead of paying the multi-minute build again. The directory is wiped when
    the bundle's weights change underneath it (re-export in place) or when the
    caller asks for it.
    """
    split_dir = Path(split_dir).resolve()
    tag = hashlib.sha1(str(split_dir).encode()).hexdigest()[:8]
    cache = Path(cache_root) / f"{split_dir.name}-{tag}"
    stamp = cache / "bundle.fingerprint"
    fingerprint = _bundle_fingerprint(split_dir)

    if cache.is_dir():
        stale = not stamp.exists() or stamp.read_text() != fingerprint
        if rebuild or stale:
            LOG.info("clearing engine cache %s (%s)", cache,
                     "asked for a rebuild" if rebuild else "bundle changed on disk")
            shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    stamp.write_text(fingerprint)
    LOG.info("engine cache: %s", cache)
    return cache


def prebuild_engines(split_dir: str | Path, cache_dir: str,
                     precision: str = "fp16") -> None:
    """Build + cache the heavy TRT engines, ONE SUBPROCESS PER GRAPH.

    The builds cannot share a process on 8 GB: a build peaks at ~2-3 GB of
    scratch, and every already-created session keeps its engine + weights
    resident — building vision then prefill in one process OOMs exactly the
    way the monolith did. A subprocess per graph returns the memory between
    builds; afterwards the main process only ever loads from cache.
    """
    import subprocess
    import sys as _sys

    cache = Path(cache_dir)
    # Written only after every graph has built. Counting *.engine files instead
    # would call a half-finished cache complete, and TRT is free to emit more
    # than one engine per graph.
    done = cache / "prebuilt"
    if done.exists():
        return
    for name in HEAVY_GRAPHS:
        LOG.info("prebuilding TRT engine for %s (subprocess, ~1 min)...", name)
        t0 = time.perf_counter()
        # Build-peak settings proven to fit 8 GB on this board: no CUDA EP,
        # 512 MB workspace, optimization level 2. Only the one-time build is
        # affected — cached engines reload identically (and load with the
        # full provider stack later).
        env = dict(os.environ, TRT_DROP_CUDA_EP="1")
        env.setdefault("TRT_WORKSPACE_MB", "512")
        env.setdefault("TRT_OPT_LEVEL", "2")
        r = subprocess.run(
            [_sys.executable, "-m", "lerobot_vla.smolvla_split",
             "--build-one", str(Path(split_dir) / name),
             "--cache-dir", str(cache_dir), "--precision", precision],
            env=env, cwd=str(Path(__file__).resolve().parents[1]),
        )
        if r.returncode != 0:
            raise RuntimeError(f"engine build failed for {name}")
        LOG.info("  built %s in %.0fs", name, time.perf_counter() - t0)
    done.touch()


class NormStats:
    """MEAN_STD normalization from a LeRobot stats dict (or identity)."""

    def __init__(self, state_mean=None, state_std=None,
                 action_mean=None, action_std=None):
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std

    @classmethod
    def from_lerobot_stats(cls, stats: dict, state_key="observation.state",
                           action_key="action"):
        def get(key, field):
            v = stats.get(key, {}).get(field)
            return None if v is None else np.asarray(v, dtype=np.float32).reshape(-1)
        return cls(get(state_key, "mean"), get(state_key, "std"),
                   get(action_key, "mean"), get(action_key, "std"))

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_mean is None:
            return state
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mean is None:
            return action
        return action * (self.action_std + 1e-8) + self.action_mean


class SmolVLASplitPolicy:
    """The 9-graph split SmolVLA policy with the denoise loop in Python."""

    def __init__(self,
                 split_dir: str | Path,
                 tokenizer_dir: str | Path,
                 cache_dir: str = "/tmp/smolvla_split_cache",
                 rebuild: bool = False,
                 precision: str = "fp16",
                 num_steps: int = 10,
                 action_dim: int = 4,
                 norm: NormStats | None = None,
                 seed: int | None = None,
                 projectors: str = "gpu",
                 iobinding: bool = True):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        split_dir = Path(split_dir)
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if projectors not in ("gpu", "cpu"):
            raise ValueError(f"projectors must be 'gpu' or 'cpu', got {projectors!r}")
        self.num_steps = num_steps
        self.action_dim = action_dim
        self.norm = norm or NormStats()
        self._rng = np.random.default_rng(seed)
        self.projectors = projectors
        self.iobinding = iobinding

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        # cache_dir is a ROOT holding one subdirectory per bundle — see
        # prepare_cache. Two bundles cannot share one engine cache.
        self.cache_dir = prepare_cache(split_dir, cache_dir, rebuild=rebuild)
        prebuild_engines(split_dir, self.cache_dir, precision)

        heavy = build_providers(str(self.cache_dir), precision=precision)
        cpu = ["CPUExecutionProvider"]

        def sess(name, providers):
            t0 = time.perf_counter()
            s = ort.InferenceSession(str(split_dir / name), providers=providers)
            LOG.info("session %-28s %-26s %.1fs", name, s.get_providers()[0],
                     time.perf_counter() - t0)
            return s

        LOG.info("Loading split sessions (first run builds TRT engines — minutes)...")
        self.vision = sess("smolvlm_vision.onnx", heavy)
        self.prefill = sess("smolvlm_expert_prefill.onnx", heavy)
        self.decode = sess("smolvlm_expert_decode.onnx", heavy)
        # Once per inference -> the host round trip is a rounding error.
        self.text = sess("smolvlm_text.onnx", cpu)
        self.state_proj = sess("state_projector.onnx", cpu)
        # Once per DENOISE STEP. On the CPU EP these four cost ~26 ms and 2.0
        # CPU cores per inference on this board; on the GPU stack they cost
        # neither, at 0.042% of the action range in arithmetic difference (see
        # the module docstring). Falling back to the CPU EP
        # per graph is deliberate: a projector that will not load on CUDA must
        # not take the excavator down with it, it must only be slower.
        self.projectors_on_gpu: list[str] = []
        for attr, fname in _PER_STEP_PROJECTORS.items():
            providers = cpu
            if projectors == "gpu":
                try:
                    setattr(self, attr, sess(fname, heavy))
                    self.projectors_on_gpu.append(attr)
                    continue
                except Exception as e:
                    LOG.warning("projector %s would not load on the GPU stack "
                                "(%s: %s) -- falling back to the CPU EP",
                                fname, type(e).__name__, e)
            setattr(self, attr, sess(fname, providers))
        if projectors == "gpu":
            LOG.info("per-step projectors on GPU: %s",
                     ", ".join(self.projectors_on_gpu) or "none")

        self.n_layers = sum(1 for i in self.decode.get_inputs()
                            if i.name.startswith("past_key_"))
        LOG.info("decode expects %d KV layers", self.n_layers)

        # Ask the prefill for its KV outputs BY NAME. Output order differs
        # between bundles (ainekko base emits a leading vlm_output_embeds, our
        # excavator export emits only the 32 KV tensors); names are stable.
        self._prefill_kv_names = [
            name for i in range(self.n_layers)
            for name in (f"present_key_{i}", f"present_value_{i}")]
        have = {o.name for o in self.prefill.get_outputs()}
        missing = [n for n in self._prefill_kv_names if n not in have]
        if missing:
            raise ValueError(f"prefill graph lacks expected KV outputs: {missing[:4]} ...")

        # Read the prefix length the prefill graph was exported with (see the
        # note at DEFAULT_NUM_CAM_SLOTS) and derive the camera-slot count.
        dim = next((i.shape[1] for i in self.prefill.get_inputs()
                    if i.name == "position_ids"), None)
        if isinstance(dim, int):
            n_cams, rem = divmod(dim - LANG_LEN - 1, IMG_TOKENS)
            if rem or n_cams < 1:
                raise ValueError(
                    f"prefill prefix length {dim} does not decompose as "
                    f"n*{IMG_TOKENS} + {LANG_LEN} + 1 — unknown export layout")
            self.n_cam_slots, self.prefix_len = n_cams, dim
        else:  # dynamic axis: keep the historical layout
            self.n_cam_slots = DEFAULT_NUM_CAM_SLOTS
            self.prefix_len = self.n_cam_slots * IMG_TOKENS + LANG_LEN + 1
        LOG.info("prefill expects prefix %d (%d camera slot(s))",
                 self.prefix_len, self.n_cam_slots)

        # Same treatment for the action chunk length: the decode graph was exported with
        # position_ids [1, chunk] and attention_mask [1, chunk, prefix + chunk], so the
        # value is recoverable and cross-checkable rather than assumed.
        dim = next((i.shape[1] for i in self.decode.get_inputs()
                    if i.name == "position_ids"), None)
        if isinstance(dim, int):
            self.chunk_size = dim
            total = next((i.shape[2] for i in self.decode.get_inputs()
                          if i.name == "attention_mask"), None)
            if isinstance(total, int) and total != self.prefix_len + self.chunk_size:
                raise ValueError(
                    f"decode graph is inconsistent: attention_mask spans {total} but "
                    f"prefix {self.prefix_len} + chunk {self.chunk_size} = "
                    f"{self.prefix_len + self.chunk_size} — mismatched export")
        else:  # dynamic axis: keep the historical layout
            self.chunk_size = DEFAULT_CHUNK_SIZE
        LOG.info("decode expects a %d-step action chunk", self.chunk_size)

        # The empty-camera slot embedding is a constant (all -1 image, the
        # lerobot padding convention) — compute once, reuse every step.
        self._pad_cam_emb = None
        if self.n_cam_slots > 1:
            pad_img = -np.ones((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            self._pad_cam_emb = self._run_vision(pad_img)

        # Language is fixed per instruction in practice — cache by string.
        self._lang_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Every bundle's decode emits exactly one tensor; IOBinding wants its name.
        self._decode_out_name = self.decode.get_outputs()[0].name
        self._ort = ort
        self._pio = self.prefill.io_binding() if iobinding else None
        self._io = self.decode.io_binding() if iobinding else None
        LOG.info("denoise loop: %s KV feeds",
                 "device-resident (IOBinding)" if iobinding else "numpy (stock)")

    # ── component wrappers ───────────────────────────────────────────────────

    @staticmethod
    def _run_single(sess, value: np.ndarray) -> np.ndarray:
        """Run a single-input graph, feeding by the graph's DECLARED input name.

        Tensor names differ between export generations (ainekko base: 'time' /
        'action'; our exporter: 'action_time' / 'expert_out' / 'hidden'), but
        every one of these graphs takes exactly one input of identical shape,
        so binding by declared name works for all bundles.
        """
        return sess.run(None, {sess.get_inputs()[0].name: value})[0]

    def _run_vision(self, img_bchw: np.ndarray) -> np.ndarray:
        emb = self._run_single(self.vision, img_bchw)            # [1,64,960]
        return emb * math.sqrt(emb.shape[-1])

    def _embed_language(self, instruction: str) -> tuple[np.ndarray, np.ndarray]:
        if instruction in self._lang_cache:
            return self._lang_cache[instruction]
        task = instruction if instruction.endswith("\n") else instruction + "\n"
        tok = self.tokenizer(task, padding="max_length", padding_side="right",
                             max_length=LANG_LEN, truncation=True,
                             return_tensors="np")
        tokens = tok["input_ids"].astype(np.int64)               # [1,48]
        mask = tok["attention_mask"].astype(bool)                # [1,48]
        emb = self._run_single(self.text, tokens)               # [1,48,960]
        emb = emb * math.sqrt(emb.shape[-1])
        self._lang_cache[instruction] = (emb, mask)
        return emb, mask

    # ── main inference ───────────────────────────────────────────────────────

    def sample_actions(self, image_hwc_uint8: np.ndarray, instruction: str,
                       state: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        """One observation -> (chunk_size, action_dim) unnormalized action chunk."""
        # prefix: [cam1(64), pad-cam(64) x (slots-1), lang(48), state(1)]
        img = resize_with_pad_uint8(image_hwc_uint8)
        img_emb = self._run_vision(img)                          # [1,64,960]
        lang_emb, lang_mask = self._embed_language(instruction)

        s = self.norm.normalize_state(np.asarray(state, dtype=np.float32).reshape(-1))
        s_pad = np.zeros((1, MAX_STATE_DIM), dtype=np.float32)
        s_pad[0, :s.shape[0]] = s
        state_emb = self._run_single(self.state_proj, s_pad).reshape(1, 1, VLM_DIM)

        n_pad_cams = self.n_cam_slots - 1
        embs = np.concatenate(
            [img_emb] + [self._pad_cam_emb] * n_pad_cams + [lang_emb, state_emb],
            axis=1,
        ).astype(np.float32)                                     # [1,prefix,960]
        pad_masks = np.concatenate(
            [np.ones((1, IMG_TOKENS), dtype=bool)]               # real camera
            + [np.zeros((1, IMG_TOKENS), dtype=bool)] * n_pad_cams  # empty slots
            + [lang_mask,
               np.ones((1, 1), dtype=bool)],                     # state token
            axis=1)                                              # [1,prefix]
        att_masks = np.zeros((1, self.prefix_len), dtype=bool)
        att_masks[0, -1] = True                                  # state starts a new block

        return self._prefill_and_denoise(embs, pad_masks, att_masks, noise)

    # ── prefill + denoise ────────────────────────────────────────────────────

    def _prefill_and_denoise(self, embs, pad_masks, att_masks,
                             noise: np.ndarray | None = None) -> np.ndarray:
        """Prefix embeddings -> unnormalized action chunk.

        Split out from `sample_actions` because the two KV strategies differ
        only from here on, and because a multi-camera caller builds its own
        prefix but wants the identical loop.
        """
        if self.iobinding:
            return self._prefill_and_denoise_iobind(embs, pad_masks, att_masks, noise)
        return self._prefill_and_denoise_feeds(embs, pad_masks, att_masks, noise)

    def _initial_noise(self, noise: np.ndarray | None) -> np.ndarray:
        if noise is None:
            noise = self._rng.standard_normal(
                (1, self.chunk_size, MAX_ACTION_DIM)).astype(np.float32)
        return np.asarray(noise, dtype=np.float32).copy()   # x_t is updated in place

    def _denoise_constants(self, pad_masks) -> tuple[np.ndarray, np.ndarray]:
        """The decode inputs that never change across the N steps."""
        prefix_pad_2d = np.broadcast_to(
            pad_masks[:, None, :], (1, self.chunk_size, self.prefix_len))
        suffix = np.ones((1, self.chunk_size), dtype=bool)   # action block: causal-in-block
        full_att_2d = np.ascontiguousarray(np.concatenate(
            [prefix_pad_2d, make_att_2d_masks(suffix, suffix)], axis=2))  # [1,chunk,prefix+chunk]
        pos_ids = np.ascontiguousarray(
            (pad_masks.sum(axis=-1, keepdims=True) + np.cumsum(suffix, axis=1) - 1
             ).astype(np.int64))
        return full_att_2d, pos_ids

    def _prefill_and_denoise_feeds(self, embs, pad_masks, att_masks, noise) -> np.ndarray:
        """Stock path: the KV cache is re-fed as numpy on every step."""
        kv = self.prefill.run(self._prefill_kv_names, {
            "attention_mask": make_att_2d_masks(pad_masks, att_masks),
            "position_ids": (np.cumsum(pad_masks, axis=1) - 1).astype(np.int64),
            "vlm_embeds": embs,
        })
        x_t = self._initial_noise(noise)
        full_att_2d, pos_ids = self._denoise_constants(pad_masks)
        kv_feed = {}
        for i in range(self.n_layers):
            kv_feed[f"past_key_{i}"] = kv[2 * i]
            kv_feed[f"past_value_{i}"] = kv[2 * i + 1]

        dt = -1.0 / self.num_steps
        t = 1.0
        while t >= -dt / 2:
            x_t += dt * self._denoise_step(x_t, t, full_att_2d, pos_ids, kv_feed)
            t += dt
        return self.norm.unnormalize_action(x_t[0, :, :self.action_dim])

    def _prefill_and_denoise_iobind(self, embs, pad_masks, att_masks, noise) -> np.ndarray:
        """Device-resident path: prefill writes its KV to the GPU and it stays there.

        The KV cache is ~7.2 MB and is identical for all N steps, so the stock
        path copies it host->device N times (72 MB per inference at N=10) after
        having copied it device->host once. Here prefill's outputs are bound
        straight to CUDA and handed to decode as device pointers; only
        `expert_embeds` is rebound per step. Measured bit-identical.
        """
        ort = self._ort

        pio = self._pio
        pio.clear_binding_inputs()
        pio.clear_binding_outputs()
        pio.bind_cpu_input("attention_mask",
                           np.ascontiguousarray(make_att_2d_masks(pad_masks, att_masks)))
        pio.bind_cpu_input("position_ids", np.ascontiguousarray(
            (np.cumsum(pad_masks, axis=1) - 1).astype(np.int64)))
        pio.bind_cpu_input("vlm_embeds", np.ascontiguousarray(embs))
        # Bound BY NAME, in _prefill_kv_names order. IOBinding returns exactly the
        # outputs that were bound, in bind order — so a bundle whose prefill also
        # emits vlm_output_embeds (ainekko's does, ours does not) cannot shift the
        # KV indices out from under the loop below.
        for name in self._prefill_kv_names:
            pio.bind_output(name, "cuda", 0)
        self.prefill.run_with_iobinding(pio)
        kv = pio.get_outputs()                        # OrtValues, already on device

        x_t = self._initial_noise(noise)
        full_att_2d, pos_ids = self._denoise_constants(pad_masks)

        io = self._io
        io.clear_binding_inputs()
        io.clear_binding_outputs()
        for i in range(self.n_layers):
            io.bind_ortvalue_input(f"past_key_{i}", kv[2 * i])
            io.bind_ortvalue_input(f"past_value_{i}", kv[2 * i + 1])
        io.bind_ortvalue_input("attention_mask",
                               ort.OrtValue.ortvalue_from_numpy(full_att_2d, "cuda", 0))
        io.bind_ortvalue_input("position_ids",
                               ort.OrtValue.ortvalue_from_numpy(pos_ids, "cuda", 0))
        io.bind_output(self._decode_out_name, "cuda", 0)

        dt = -1.0 / self.num_steps
        t = 1.0
        while t >= -dt / 2:
            suffix_embs = np.ascontiguousarray(self._suffix_embeds(x_t, t))
            io.bind_ortvalue_input(
                "expert_embeds", ort.OrtValue.ortvalue_from_numpy(suffix_embs, "cuda", 0))
            self.decode.run_with_iobinding(io)
            expert_out = io.get_outputs()[0].numpy()
            x_t += dt * self._run_single(self.action_out, expert_out.astype(np.float32))
            t += dt
        return self.norm.unnormalize_action(x_t[0, :, :self.action_dim])

    def _suffix_embeds(self, x_t, t) -> np.ndarray:
        """Action chunk + timestep -> the expert's suffix embeddings for one step."""
        action_emb = self._run_single(self.action_in, x_t)               # [1,chunk,720]
        time_emb = np.broadcast_to(
            sinusoidal_time_embedding(t)[None, None, :], action_emb.shape)
        ate = np.concatenate([action_emb, time_emb], axis=2).astype(np.float32)
        ate = self._run_single(self.time_in, ate)
        ate = ate * (1.0 / (1.0 + np.exp(-ate)))                         # SiLU
        return self._run_single(self.time_out, ate)                      # [1,chunk,720]

    def _denoise_step(self, x_t, t, full_att_2d, pos_ids, kv_feed) -> np.ndarray:
        feeds = {
            "attention_mask": full_att_2d,
            "position_ids": pos_ids,
            "expert_embeds": self._suffix_embeds(x_t, t),
            **kv_feed,
        }
        expert_out = self.decode.run(None, feeds)[0]  # single output in every bundle
        return self._run_single(self.action_out, expert_out.astype(np.float32))


def _build_one(onnx_path: str, cache_dir: str, precision: str) -> None:
    """Subprocess entry: create one TRT-EP session so its engine gets cached."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path,
                                providers=build_providers(cache_dir, precision))
    if sess.get_providers()[0] != "TensorrtExecutionProvider":
        raise SystemExit(f"TRT EP did not register for {onnx_path}: "
                         f"{sess.get_providers()}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--build-one", required=True)
    ap.add_argument("--cache-dir", default="/tmp/smolvla_split_cache")
    ap.add_argument("--precision", default="fp16")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    _build_one(a.build_one, a.cache_dir, a.precision)
