"""pi0.5 (OpenPI pi05_openpi_state) adapter for vla_live.py over the C++ runtime.

Everything model-side runs in libkaivuri_pi05_runtime.so (src/pi05_capi.cpp, the same code as
kaivuri-inference): engines, the per-step joint-angle prompt (float32 degrees -> OpenPI quantile
bins -> tokens), Pillow-exact preprocessing of the Y8 IR frames and OpenPI unnormalization. Loading
runs every prepared fixture (prepare_pi05.py) through the engines; any failure stops the load.

    state    angles_rad -> float32 degrees of the bundle's state joints -> bins in the prompt
    actions  normalized chunk -> OpenPI unnormalize -> [slew, lift, tilt, scoop], clamped to [-1, 1]

Tasks are the bundle's exported prompt heads; there is no tokenizer on the board, so typed
prompts outside that list are refused. No RTC and no frame history yet.
"""
from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path

import numpy as np

from vla_operator import JOINTS

_c = ctypes
_ERR = 4096


class Model:
    name = "pi0.5"
    rtc, rtc_max = None, 0
    history, lag_s = None, 0.0

    def __init__(self, bundle: Path, cache: Path, fixtures: Path | None, build: Path,
                 min_cosine: float | None = None, max_abs: float | None = None):
        if fixtures is None:
            raise ValueError("pi0.5 needs --fixtures (prepare_pi05.py output for this bundle)")
        lib_path = Path(build) / "libkaivuri_pi05_runtime.so"
        if not lib_path.exists():
            raise FileNotFoundError(f"{lib_path}: build native_stack with -DBUILD_INFERENCE=ON")
        self.lib = lib = _c.CDLL(str(lib_path))
        lib.kvr_pi05_open.restype = _c.c_void_p
        lib.kvr_pi05_open.argtypes = [_c.c_char_p, _c.c_char_p, _c.c_char_p, _c.c_double, _c.c_double,
                                      _c.c_char_p, _c.c_size_t, _c.c_char_p, _c.c_size_t]
        lib.kvr_pi05_close.argtypes = [_c.c_void_p]
        lib.kvr_pi05_info.argtypes = [_c.c_void_p, _c.c_char_p, _c.c_size_t, _c.c_char_p, _c.c_size_t]
        lib.kvr_pi05_infer.argtypes = [_c.c_void_p, _c.c_char_p, _c.c_void_p, _c.c_void_p, _c.c_void_p,
                                       _c.c_int, _c.c_int, _c.c_void_p, _c.c_void_p, _c.c_void_p,
                                       _c.c_void_p, _c.POINTER(_c.c_int), _c.c_char_p, _c.c_size_t]
        self.err = _c.create_string_buffer(_ERR)
        report = _c.create_string_buffer(1 << 20)
        t0 = time.time()
        self.h = lib.kvr_pi05_open(str(bundle).encode(), str(cache).encode(), str(fixtures).encode(),
                                   min_cosine or 0.0, max_abs or 0.0, report, len(report), self.err, _ERR)
        if not self.h:
            raise RuntimeError(f"pi0.5 load: {self.err.value.decode()}")
        self.load_s = round(time.time() - t0, 1)
        self.parity = json.loads(report.value.decode())
        info = _c.create_string_buffer(1 << 16)
        self._check(lib.kvr_pi05_info(self.h, info, len(info), self.err, _ERR))
        self.info = json.loads(info.value.decode())
        b = json.loads((Path(bundle) / "bundle.json").read_text())
        k = b["kaivuri"]
        self.horizon, self.width = self.info["horizon"], self.info["action_dim"]
        self.noise_shape = (self.horizon, self.width)
        self.fps = float(self.info["fps"] or k["fps"])
        heads = set(self.info["tasks"])
        self._tasks = [t for t in k.get("tasks", []) if t in heads] + sorted(heads - set(k.get("tasks", [])))
        names = k["action"]["names"]
        if len(names) != self.info["robot_dim"]:
            raise ValueError(f"kaivuri.action.names {names} vs robot_dim {self.info['robot_dim']}")
        self.action_cols = [JOINTS[n] for n in names]
        self.task = self._tasks[0]
        self.task_tokens = 0

    def _check(self, rc: int):
        if rc:
            raise RuntimeError(self.err.value.decode())

    def close(self):
        if getattr(self, "h", None):
            self.lib.kvr_pi05_close(self.h)
            self.h = None

    # ── the model interface vla_live.py drives ──

    @property
    def tasks(self) -> list[str]:
        return self._tasks

    def check_prompt(self, text: str) -> int:
        if text not in self._tasks:
            raise ValueError("pi0.5 bundles run only their exported tasks: " + "; ".join(self._tasks))
        return 0

    def prompt_note(self) -> str:
        return "this bundle runs only its exported tasks"

    def same_prompt(self, a: str, b: str) -> bool:
        return a == b

    def set_prompt(self, text: str) -> int:
        self.check_prompt(text)
        self.task = text          # the tokens change every step with the state bins
        return self.task_tokens

    def describe(self, rtc_on: bool, a) -> str:
        worst = min(self.parity, key=lambda r: r["fp16"]["cosine"])
        return (f"[pi05] engines ready in {self.load_s} s, {len(self.parity)} fixtures PASS (worst cos "
                f"{worst['fp16']['cosine']:.6f}, max {max(r['fp16']['max_abs'] for r in self.parity):.4f}), "
                f"RTC off (not wired for pi0.5 yet)")

    def model_actions(self, valves):
        raise NotImplementedError("pi0.5 RTC is not wired yet")

    def step(self, frames_y8: list[np.ndarray], status: dict, noise: np.ndarray, timings: dict,
             rtc_delay: int = 0, rtc_prefix=None, capture_ns=None) -> tuple[np.ndarray, np.ndarray]:
        """IR frames + controller status -> ([H, robot_dim] commands in [-1, 1], the degrees used).
        Out-of-range commands are clamped (the controller clamps too) and counted."""
        if rtc_delay:
            raise ValueError("pi0.5 RTC is not wired yet")
        f0, f1 = (np.ascontiguousarray(f, np.uint8) for f in frames_y8)
        if f0.shape != f1.shape or f0.ndim != 2:
            raise ValueError("two Y8 frames of one size required")
        angles = np.ascontiguousarray(status["angles_rad"], np.float64)
        if angles.shape != (4,):
            raise ValueError("angles_rad must have four joints")
        noise = np.ascontiguousarray(noise, np.float32).reshape(-1)
        if noise.size != self.horizon * self.width:
            raise ValueError(f"noise has {noise.size} values, the model takes {self.horizon * self.width}")
        robot = np.empty((self.horizon, self.info["robot_dim"]), np.float64)
        deg = np.empty(self.info["state_dim"], np.float32)
        t = np.empty(6, np.float64)
        tokens = _c.c_int(0)
        self._check(self.lib.kvr_pi05_infer(
            self.h, self.task.encode(), angles.ctypes.data, f0.ctypes.data, f1.ctypes.data,
            f0.shape[1], f0.shape[0], noise.ctypes.data, robot.ctypes.data, deg.ctypes.data,
            t.ctypes.data, _c.byref(tokens), self.err, _ERR))
        self.task_tokens = tokens.value
        if not np.isfinite(robot).all():
            raise RuntimeError("non-finite pi0.5 commands")
        timings.update(prompt=t[0], preprocess=t[1], vision=t[2], language=t[3], denoise=t[4], total=t[5])
        clamped = int((np.abs(robot) > 1.0).sum())
        if clamped:
            timings["clamped"] = clamped
        return np.clip(robot, -1.0, 1.0).astype(np.float32), deg

    def keep_warm(self, frames_y8, capture_ns) -> None:
        pass

    def check_cameras(self, files) -> None:
        """Frame files must be the bundle's cameras in slot order (<serial>-ir<index>.frame)."""
        want = [c.replace(":", "-ir") + ".frame" for c in self.info["cameras"]]
        got = [Path(f).name for f in files]
        if got != want:
            raise ValueError(f"camera frames {got}, the bundle was trained on {want} in this order")


def parity(a) -> int:
    """Load = every prepared fixture through the engines; print the per-fixture report."""
    m = Model(Path(a.bundle), Path(a.cache), Path(a.fixtures) if a.fixtures else None, Path(a.build),
              a.min_cosine, a.max_abs)
    for r in m.parity:
        print(f"PASS {r['fixture']} {r.get('task', '')!r}: cos {r['fp16']['cosine']:.6f} "
              f"max {r['fp16']['max_abs']:.4f}  prompt {r.get('prompt_tokens')} tokens", flush=True)
    summary = {"status": "PASS", "model": "pi05", "load_s": m.load_s, "fixtures": m.parity}
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"status": "PASS", "load_s": m.load_s, "fixtures": len(m.parity)}), flush=True)
    m.close()
    return 0
