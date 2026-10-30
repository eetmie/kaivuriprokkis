"""pi0.5 (OpenPI pi05_openpi_state) adapter for vla_live.py over the C++ runtime.

Everything model-side runs in libkaivuri_pi05_runtime.so (src/pi05_capi.cpp, the same code as
kaivuri-inference): engines, the per-step joint-angle prompt (float32 degrees -> OpenPI quantile
bins -> tokens), Pillow-exact preprocessing of the Y8 IR frames and OpenPI unnormalization. Loading
runs every prepared fixture (prepare_pi05.py) through the engines; any failure stops the load.

    state    angles_rad -> float32 degrees of the bundle's state joints -> bins in the prompt
    actions  normalized chunk -> OpenPI unnormalize -> [slew, lift, tilt, scoop], clamped to [-1, 1]

Tasks are the bundle's exported prompt heads; there is no tokenizer on the board, so typed
prompts outside that list are refused. No frame history yet.

RTC is training-time Mode C, on bundles exported with --row-time (kaivuri.rtc.row_time): the d
committed rows go in as the clean prefix, in model units (`model_actions`: OpenPI's quantile
normalization with the bundle's own norm_stats.json, unclipped as in training), and come back
unchanged. Loading also runs the bundle's Mode C references, with RTC off and on.
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
                                       _c.c_int, _c.c_int, _c.c_void_p, _c.c_int, _c.c_void_p,
                                       _c.c_void_p, _c.c_void_p, _c.c_void_p, _c.POINTER(_c.c_int),
                                       _c.c_char_p, _c.c_size_t]
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
        if self.info.get("row_time"):
            self.rtc, self.rtc_max = k["rtc"], int(self.info["max_prefix"])
        # The runtime unnormalizes with these (bundle_norm_stats, MANIFEST-checked); the prefix
        # goes back through the inverse.
        q = json.loads((Path(bundle) / "norm_stats.json").read_text())["norm_stats"]["actions"]
        self.set_action_quantiles(q["q01"], q["q99"])

    def set_action_quantiles(self, q01, q99):
        r = self.info["robot_dim"]
        self._q01, self._q99 = (np.asarray(v, np.float64)[:r] for v in (q01, q99))
        if self._q01.shape != (r,) or self._q99.shape != (r,) or not (self._q99 > self._q01).all():
            raise ValueError("norm_stats.json action quantiles do not cover the robot dims")

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
        modes = sum(r.get("mode") == "C" for r in self.parity)
        rtc = (f"RTC ON (Mode C, max {self.rtc_max} rows, margin {a.rtc_margin_ms:g} ms)" if rtc_on
               else "RTC off" + (" (--rtc off)" if self.rtc else " (bundle has no row-time templates)"))
        return (f"[pi05] engines ready in {self.load_s} s, {len(self.parity)} fixtures PASS"
                + (f" ({modes} Mode C)" if modes else "")
                + f" (worst cos {worst['fp16']['cosine']:.6f}, "
                f"max {max(r['fp16']['max_abs'] for r in self.parity):.4f}), {rtc}")

    def model_actions(self, valves) -> np.ndarray:
        """[N, robot_dim] commands -> normalized model units: OpenPI's quantile Normalize,
        (x - q01) / (q99 - q01 + 1e-6) * 2 - 1, the inverse of the runtime's unnormalize.
        Not clipped: training saw the dataset's actions unclipped too."""
        x = np.asarray(valves, np.float64)
        if x.ndim != 2 or x.shape[1] != len(self._q01):
            raise ValueError(f"valves {x.shape}, expected [N, {len(self._q01)}]")
        return ((x - self._q01) / (self._q99 - self._q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)

    def step(self, frames_y8: list[np.ndarray], status: dict, noise: np.ndarray, timings: dict,
             rtc_delay: int = 0, rtc_prefix=None, capture_ns=None) -> tuple[np.ndarray, np.ndarray]:
        """IR frames + controller status -> ([H, robot_dim] commands in [-1, 1], the degrees used).
        Out-of-range commands are clamped (the controller clamps too) and counted.
        rtc_delay d > 0: rtc_prefix [>= d, robot_dim] model units (`model_actions`) for the d
        committed rows (Mode C)."""
        rows, prefix = int(rtc_delay), None
        if rows:
            if not self.rtc:
                raise ValueError("RTC delay given, but this pi0.5 bundle has no row-time templates")
            if not 0 < rows <= self.rtc_max:
                raise ValueError(f"RTC delay {rows} outside [0, {self.rtc_max}]")
            p = np.asarray(rtc_prefix, np.float32) if rtc_prefix is not None else None
            if p is None or p.ndim != 2 or p.shape[0] < rows or p.shape[1] != self.info["robot_dim"] \
                    or not np.isfinite(p[:rows]).all():
                raise ValueError(f"bad RTC prefix {None if p is None else p.shape} for d = {rows}")
            prefix = np.ascontiguousarray(p[:rows])
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
            f0.shape[1], f0.shape[0], noise.ctypes.data, rows,
            prefix.ctypes.data if prefix is not None else None, robot.ctypes.data, deg.ctypes.data,
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
        what = (f"Mode C d={r['rows']} on {r['source']} (RTC off: cos {r['off']['cosine']:.6f} "
                f"max {r['off']['max_abs']:.4f}), prefix exact" if r.get("mode") == "C"
                else f"{r.get('task', '')!r}, prompt {r.get('prompt_tokens')} tokens")
        print(f"PASS {r['fixture']}: cos {r['fp16']['cosine']:.6f} max {r['fp16']['max_abs']:.4f}  {what}",
              flush=True)
    summary = {"status": "PASS", "model": "pi05", "load_s": m.load_s, "fixtures": m.parity}
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"status": "PASS", "load_s": m.load_s, "fixtures": len(m.parity)}), flush=True)
    m.close()
    return 0
