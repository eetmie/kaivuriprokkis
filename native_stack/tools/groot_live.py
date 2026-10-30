"""GR00T N1.7 fine-tune on the native stack: physical parity and a live drive loop.

Engines and the host chain come from jetson-orin-nano-vla (`--vla-root`, imported read-only).
This file owns the excavator boundary that Gr00tPolicy applies on the Spark:

    cameras  native IR Y8 -> gray repeated to RGB -> the bundle's eval transform
    state    angles_rad[boom, arm, bucket] -> float32 degrees -> min-max [lo, hi] -> [-1, 1], clipped
    actions  model [-1, 1] -> clipped -> [lo, hi] (the contract's stats) -> [slew, lift, tilt, scoop]

`parity` runs a physical fixture (spark-projects groot/excavator/physical_fixture.py) from raw
frames and degrees to valve commands. `run` reads live cameras and the controller, infers
continuously and, with --live, sends each chunk as a timed plan (t0 = observation time, so
the rows that went stale during inference are skipped by the controller). A supervisor thread
renews the plan every 100 ms and disarms on chunk exhaustion, on observation age, on any
controller state but armed and on any inference error. It never re-arms. With an Xbox pad
(--gamepad auto): A toggles manual (sticks drive, inference paused) and model; a stale or
exhausted chunk then falls back to manual instead of disarming. X toggles pump permission.
Ctrl+C disarms, stops the services and still writes the summary (with --out; without it nothing
is written and the outcome is printed only). --live drives with pump permission on and, with
--start-services, starts the controller with live PWM.

Prompts: bundles from spark-projects groot/excavator/export_bundle.py tokenize at runtime. The
run starts on --task or the first trained task; D-pad right/left cycles the list, and a line
typed in the terminal becomes a prompt, appended to the list. A change drops the current plan;
the machine holds until the first chunk observed under the new prompt, at most
--first-plan-timeout-s.

RTC (bundles with an `rtc` section, a training-time RTC checkpoint): each inference commits the
first d rows of its chunk to what the running plan executes at those times (read back from the
plan the controller holds, in normalized model units), and the model denoises only the rest. d
covers the observation -> new-plan-applied latency (recent maximum + --rtc-margin-ms), capped at
the trained maximum, so the new plan agrees with the old one where the controller switches.
`--rtc off` runs the stock sampler on the same engines.

Frame history (bundles trained with GROOT_VIDEO_DELTAS="-k,0"): the model also sees each camera k
dataset frames earlier. The runtime keeps the vision encodes of its own earlier observations and
feeds the one closest to k/fps old, within --history-tolerance-ms; until it has one (start-up)
no plan is made, only the encode is stored. Manual mode keeps encoding, so A returns to a plan
at once. `parity` takes the past frames from the fixture (`raw_past`).

Exit status: 0 run finished, 130 Ctrl+C, 3 gamepad lost, 4 controller fault, 5 plan stopped
(observation age, chunk exhausted, no first plan), 1 any other error.

    python groot_live.py parity --bundle B                 # every prompt (and RTC) fixture
    python groot_live.py run --bundle B --start-services --live --enable-slew [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAM1, CAM2 = "233522073919", "141722076647"        # digging_native_640: cam1 = ir2, rotated 180
JOINTS = {"slew": 0, "lift": 1, "tilt": 2, "scoop": 3}  # controller angles_rad / plan columns


# ── boundary ────────────────────────────────────────────────────────────────


class Boundary:
    def __init__(self, contract: dict, state_names=("lift", "tilt", "scoop"),
                 action_names=("slew", "lift", "tilt", "scoop")):
        (sk, s), = contract["state"].items()
        (ak, a), = contract["action"].items()
        self.s_lo, self.s_hi = np.asarray(s["min"]), np.asarray(s["max"])
        self.a_lo, self.a_hi = np.asarray(a["min"]), np.asarray(a["max"])
        self.clip_state = contract["clip_state"]
        self.state_idx = [JOINTS[n] for n in state_names]
        self.action_cols = [JOINTS[n] for n in action_names]
        assert len(self.s_lo) == len(state_names) and len(self.a_lo) == len(action_names)

    def state(self, degrees: np.ndarray, width: int) -> np.ndarray:
        """float32 degrees -> [1,1,width] model state (gr00t.data.utils.normalize_values_minmax)."""
        x = np.asarray(degrees, np.float32)
        n = np.zeros_like(x)
        m = ~np.isclose(self.s_hi, self.s_lo)
        n[m] = 2 * ((x[m] - self.s_lo[m]) / (self.s_hi[m] - self.s_lo[m])) - 1
        if self.clip_state:
            n = np.clip(n, -1.0, 1.0)
        out = np.zeros((1, 1, width), np.float32)
        out[0, 0, :n.size] = n
        return out

    def actions(self, model: np.ndarray) -> np.ndarray:
        """[1,H,W] model actions -> [H,A] valve commands (unnormalize_values_minmax).
        Callers check the raw model output first: clipping turns +-inf into valid bounds."""
        x = np.clip(model[0, :, :len(self.a_lo)].astype(np.float64), -1.0, 1.0)
        return ((x + 1.0) / 2.0 * (self.a_hi - self.a_lo) + self.a_lo).astype(np.float32)

    def model_actions(self, valves: np.ndarray) -> np.ndarray:
        """[N,A] valve commands -> normalized model units, clipped (the processor's training targets)."""
        x = np.asarray(valves, np.float64)
        return np.clip(2 * (x - self.a_lo) / (self.a_hi - self.a_lo) - 1, -1.0, 1.0).astype(np.float32)

    def degrees(self, status: dict) -> np.ndarray:
        return np.rad2deg(np.asarray([status["angles_rad"][i] for i in self.state_idx],
                                     np.float64)).astype(np.float32)


def formalize(text: str) -> str:
    """Gr00tN1d7Processor's formalize_language: lower case, drop punctuation."""
    return re.sub(r"[^\w\s]", "", text.lower())


class Prompts:
    """Runtime prompt for a bundle with a `prompt` section (spark-projects export_bundle.py):
    ids = prefix (chat start + image slots) + tokenize(formalize(text)) + suffix, exactly
    Gr00tN1d7Processor's. A bundle without one runs only its baked prompt."""

    def __init__(self, root: Path, b: dict):
        pr = b.get("prompt")
        self.dynamic = pr is not None
        self.baked = b["task"]
        self.tasks = list(pr["tasks"]) if pr else [b["task"]]
        if pr:
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_file(str(root / pr["tokenizer"]))
            self.prefix, self.suffix = pr["prefix_ids"], pr["suffix_ids"]
            self.capacity = int(pr["capacity_tokens"])

    def ids(self, text: str) -> list[int]:
        """The real (unpadded) prompt ids; ValueError when the text cannot run."""
        if not self.dynamic:
            if text != self.baked:
                raise ValueError(f"this bundle bakes {self.baked!r}; re-export with export_bundle.py "
                                 "for runtime prompts")
            return []
        body = self.tok.encode(formalize(text), add_special_tokens=False).ids
        if len(body) > self.capacity:
            raise ValueError(f"{len(body)} tokens, the bundle holds {self.capacity}")
        return self.prefix + body + self.suffix

    def text_tokens(self, text: str) -> int:
        """Token count of the instruction itself; ValueError as ids()."""
        return len(self.ids(text)) - len(self.prefix) - len(self.suffix) if self.dynamic else len(self.ids(text))


def frame_slots(bb: dict) -> tuple[int, float]:
    """(time slots, lag s) of a bundle: video_delta_indices [0] -> (1, 0); [-k, 0] -> (2, k/fps)
    with the dataset rate the export recorded (kaivuri.fps)."""
    deltas = list(bb["video_delta_indices"])
    if bb["frames"] != len(deltas) or not (deltas == [0] or (len(deltas) == 2 and deltas[0] < 0 and deltas[1] == 0)):
        raise ValueError(f"video_delta_indices {deltas}: this runtime handles [0] and [-k, 0]")
    if len(deltas) == 1:
        return 1, 0.0
    fps = (bb.get("kaivuri") or {}).get("fps")
    if not fps:
        raise ValueError("a frame-history bundle needs kaivuri.fps (the dataset rate)")
    return 2, -deltas[0] / float(fps)


class HistoryFilling(RuntimeError):
    """A two-frame bundle has no earlier encode near the lag yet; this frame was stored."""


class History:
    """Vision encodes of earlier observations by capture time. The past slot is the entry closest
    to `lag_s` old; none within `tolerance_s` -> None (still filling, or a gap)."""

    def __init__(self, lag_s: float, tolerance_s: float):
        self.lag, self.tol = lag_s, tolerance_s
        self.entries = collections.deque()      # (capture ns, encode), oldest first

    def push(self, t_ns: int, enc) -> None:
        if self.entries and t_ns <= self.entries[-1][0]:
            return                              # same frame again: keep the first encode
        self.entries.append((t_ns, enc))
        # Ages only grow: an entry past lag + tolerance can never be picked again.
        while (t_ns - self.entries[0][0]) / 1e9 > self.lag + self.tol:
            self.entries.popleft()

    def pick(self, t_ns: int):
        """(encode, age s) of the entry closest to the lag, or None."""
        best = min(self.entries, key=lambda e: abs((t_ns - e[0]) / 1e9 - self.lag), default=None)
        if best is None:
            return None
        age = (t_ns - best[0]) / 1e9
        return (best[1], age) if abs(age - self.lag) <= self.tol else None


class Policy:
    """Bundle + engines + boundary. Each camera's current frame (video_delta_indices [0]), or
    the current frame and one k dataset frames earlier ([-k, 0])."""

    def __init__(self, bundle: Path, contract: dict | None, cache: Path, vla_root: Path,
                 history_tolerance_s: float = 0.15):
        sys.path.insert(0, str(vla_root))
        os.chdir(vla_root)      # the engine builders run `-c "from bench.vendor..."` subprocesses
        from bench.vendor import groot17_trt as g17
        from bench.vendor import groot_trt as g16

        self.g17, self.g16 = g17, g16
        self.b = g17.Bundle(bundle)
        bb = self.b.b
        self.frames, self.lag_s = frame_slots(bb)
        self.history = History(self.lag_s, history_tolerance_s) if self.frames == 2 else None
        contract = contract or (bb.get("kaivuri") or {}).get("contract")
        if contract is None:
            raise ValueError("the bundle carries no normalization contract: pass --contract")
        if bb["embodiment"] != contract["embodiment"]:
            raise ValueError("contract and bundle embodiments differ")
        self.contract = contract
        self.boundary = Boundary(contract)
        self.prompts = Prompts(Path(bundle), bb)
        self.horizon, self.width = bb["action_horizon"], bb["action_dim"]
        self.rtc = bb.get("rtc")
        self.rtc_max = int(self.rtc["max_delay"]) if self.rtc else 0
        if self.rtc:
            # RTC graphs take rtc_mask/prefix: added to the DiT pool at groot17_trt's one denoise
            # call (both its infer and its fixture gate go through it). All zero = stock sampler.
            self._rtc_feeds = self.rtc_feeds(0, None)
            real = g17.denoise
            g17.denoise = lambda bundle, run, pool, noise: real(bundle, run, {**pool, **self._rtc_feeds}, noise)
        t0 = time.time()
        built = g16.prebuild_engines(self.b, cache)
        if self.b.mod:
            g16.load_step_constants(self.b, cache)
        self.engines = g16.Engines(self.b, cache, skip=self.b.mod)
        self.load_s = round(time.time() - t0, 1)
        self.built = built
        self.fixture = g17.validate_fixture(self.b, self.engines)   # on the baked prompt
        if self.fixture["status"] != "PASS":
            raise RuntimeError(f"bundle fixture gate failed: {self.fixture}")
        self.state_width = self.engine_shape("cond_0", "state")[-1]
        self.noise_shape = (1, bb["action_horizon"], bb["action_dim"])
        self.task = bb["task"]
        self.task_tokens = bb["prompt_tokens"]

    def set_prompt(self, text: str) -> int:
        """Condition the next inference on `text`; returns its token count. As Bundle.__init__
        does for the baked prompt: embedded rows, image slots and the three key masks."""
        ids = self.prompts.ids(text)
        if not ids:                                     # baked-only bundle, same text
            return self.task_tokens
        bb, S = self.b.b, self.b.b["seq_len"]
        full = np.asarray(ids + [bb["pad_token"]] * (S - len(ids)), np.int64)
        valid = np.arange(S) < len(ids)
        img = full == bb["image_token"]
        if img.sum() != len(self.b.image_positions.nonzero()[0]):
            raise RuntimeError("image slots moved: prefix does not match the bundle")
        neg = np.float32(bb["mask_neg"])
        self.b.image_positions = img
        self.b.text_bias = np.where(~img & valid, 0.0, neg).astype(np.float32)[None, None]
        self.b.image_bias = np.where(img & valid, 0.0, neg).astype(np.float32)[None, None]
        self.b.pad_bias = np.where(valid, 0.0, neg).astype(np.float32)[None, None]
        self.b.prompt_embeds = np.asarray(self.b.embed[full], dtype=np.float32)[None]
        self.task, self.task_tokens = text, len(ids)
        return len(ids)

    def engine_shape(self, graph, tensor):
        for n, _, shape, _, _ in self.engines.io[graph]:
            if n == tensor:
                return shape
        raise KeyError(tensor)

    def rtc_feeds(self, delay: int, prefix: np.ndarray | None) -> dict:
        """rtc_mask [1,H,1] and prefix [1,H,D] for `delay` committed rows. Fresh arrays every call:
        Engines.run skips the upload of an array object it uploaded last time."""
        if not 0 <= delay <= self.rtc_max:
            raise ValueError(f"RTC delay {delay} outside [0, {self.rtc_max}]")
        mask = np.zeros((1, self.horizon, 1), np.float32)
        mask[0, :delay] = 1.0
        full = np.zeros((1, self.horizon, self.width), np.float32)
        if delay:
            p = np.asarray(prefix, np.float32)
            if p.ndim != 2 or p.shape[0] < delay or p.shape[1] > self.width or not np.isfinite(p[:delay]).all():
                raise ValueError(f"bad RTC prefix {p.shape}")
            full[0, :delay, :p.shape[1]] = p[:delay]
        return {"rtc_mask": mask, "prefix": full}

    def encode(self, frames_rgb: list[np.ndarray], t: dict | None = None):
        """One frame per camera -> the vision engines' outputs for one time slot. Adds the
        preprocess and vision milliseconds to `t`."""
        t0 = time.perf_counter()
        pv = np.stack([self.b.preprocess(f) for f in frames_rgb])
        t1 = time.perf_counter()
        enc = self.g17.encode_frames(self.b, self.engines.run, pv)
        if t is not None:
            t["preprocess"] = t.get("preprocess", 0.0) + (t1 - t0) * 1e3
            t["vision"] = t.get("vision", 0.0) + (time.perf_counter() - t1) * 1e3
        return enc

    def remember(self, frames_rgb: list[np.ndarray], capture_ns: int) -> None:
        """Encode and store a frame for later history slots (no plan: manual mode)."""
        if self.history is not None:
            self.history.push(capture_ns, self.encode(frames_rgb))

    def infer(self, frames_rgb: list[np.ndarray], degrees: np.ndarray, noise: np.ndarray,
              timings: dict | None = None, rtc_delay: int = 0,
              rtc_prefix: np.ndarray | None = None, capture_ns: int | None = None,
              past_rgb: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Valve chunk [H,A] and the raw model chunk. rtc_prefix: [>=d, A or D] model units.

        Two-frame bundles: `past_rgb` gives the earlier frames (parity); otherwise the history at
        `capture_ns` does, and this frame joins it. HistoryFilling: nothing near the lag yet."""
        if rtc_delay and not self.rtc:
            raise ValueError("this bundle has no RTC inputs")
        t = {} if timings is None else timings
        t0 = time.perf_counter()
        state = self.boundary.state(degrees, self.state_width)
        enc = self.encode(frames_rgb, t)
        slots = [enc]
        if self.frames == 2:
            if past_rgb is not None:
                slots = [self.encode(past_rgb, t), enc]
            else:
                if capture_ns is None:
                    raise ValueError("a frame-history bundle needs the capture time")
                got = self.history.pick(capture_ns)
                self.history.push(capture_ns, enc)
                if got is None:
                    raise HistoryFilling(f"no encode {self.lag_s:.2f} s old yet")
                slots = [got[0], enc]
                t["history_age"] = got[1] * 1e3
        if self.rtc:
            self._rtc_feeds = self.rtc_feeds(rtc_delay, rtc_prefix)
        model = self.g17.infer(self.b, self.engines.run, slots, state, noise, timings=t)
        t3 = time.perf_counter()
        t["total"] = (t3 - t0) * 1e3
        # Raw output first: the boundary clips, which would turn +-inf into valid commands.
        if model.shape != (1, self.horizon, self.width):
            raise RuntimeError(f"model chunk shape {model.shape}")
        if not np.isfinite(model).all():
            raise RuntimeError("non-finite model output")
        act = self.boundary.actions(model)
        if not (np.isfinite(act).all() and np.abs(act).max() <= 1.0):
            raise RuntimeError("valve commands outside [-1, 1]")
        return act, model


def gray_to_rgb(y8: np.ndarray) -> np.ndarray:
    return np.repeat(y8[..., None], 3, axis=2)


# ── parity ──────────────────────────────────────────────────────────────────


def check_fixture(pol: Policy, fx, env, min_cosine: float, max_abs: float) -> tuple[bool, list]:
    """Raw frames + degrees -> valves through `pol` vs the fixture's FP32 Gr00tPolicy chunks.
    Gate per sample: cosine, and no further from FP32 than bfloat16 Gr00tPolicy (`env`)."""
    if env is not None and not np.array_equal(env["noise"], fx["noise"]):
        raise SystemExit("envelope fixture uses different inputs")
    if pol.frames == 2 and "raw_past" not in fx.files:
        raise SystemExit("frame-history bundle, fixture without raw_past: re-run physical_fixture.py")

    def as_live(raw):
        # The dataset decodes gray IR as identical RGB channels; the live path does too.
        gray = all(np.array_equal(raw[v][..., 0], raw[v][..., c]) for v in range(len(raw)) for c in (1, 2))
        return [gray_to_rgb(raw[v][..., 0]) if gray else raw[v] for v in range(len(raw))], gray

    rows, ok = [], True
    for i in range(len(fx["raw"])):
        frames, gray_ok = as_live(fx["raw"][i])
        past = as_live(fx["raw_past"][i])[0] if pol.frames == 2 else None
        t = {}
        delay = int(fx["rtc_delay"][i]) if "rtc_delay" in fx.files else 0
        act, model = pol.infer(frames, fx["state_deg"][i], fx["noise"][i][None], t, rtc_delay=delay,
                               rtc_prefix=fx["rtc_prefix"][i] if delay else None, past_rgb=past)
        ref = fx["actions"][i]
        d = np.abs(act - ref)
        cos = float((act.ravel() @ ref.ravel()) / (np.linalg.norm(act) * np.linalg.norm(ref)))
        if env is not None:
            e = np.abs(env["actions"][i] - ref)     # floored: bf16 can land closer than FP16 noise
            lim_max, lim_mean = max(float(e.max()), max_abs), max(float(e.mean()), max_abs / 10)
        else:
            lim_max, lim_mean = max_abs, np.inf
        passed = bool(d.max() <= lim_max and d.mean() <= lim_mean and cos >= min_cosine)
        held = None
        if delay:   # committed rows come back bit-exact in model units
            want = fx["rtc_prefix"][i][:delay]
            held = bool(np.array_equal(model[0, :delay, :want.shape[1]], want))
            passed &= held
        ok &= passed
        rows.append({"pass": passed, "rtc_delay": delay, "prefix_exact": held,
                     "episode": int(fx["episode"][i]), "frame": int(fx["frame"][i]),
                     "cosine": round(cos, 6), "max_abs": round(float(d.max()), 4),
                     "limit_max_abs": round(lim_max, 4), "mean_abs": round(float(d.mean()), 5),
                     "limit_mean_abs": round(lim_mean, 5) if np.isfinite(lim_mean) else None,
                     "mean_abs_vs_recorded": round(float(np.abs(act - fx["gt"][i]).mean()), 4),
                     "gray_rgb": gray_ok, "ms": round(t["total"], 1)})
    return ok, rows


def parity(a) -> int:
    contract = json.loads(Path(a.contract).read_text()) if a.contract else None
    pol = Policy(Path(a.bundle), contract, Path(a.cache).expanduser(), Path(a.vla_root))
    root = Path(a.bundle)
    if a.fixture:
        jobs = [{"task": None, "float32": a.fixture, "bfloat16": a.envelope}]
    else:
        jobs = (pol.b.b.get("prompt") or {}).get("fixtures") or []
        if not jobs:
            raise SystemExit("the bundle has no prompt fixtures: pass --fixture")
        jobs = jobs + [{**j, "rtc": True} for j in (pol.rtc or {}).get("fixtures", [])]
        jobs = [{**j, "float32": root / j["float32"], "bfloat16": root / j["bfloat16"]} for j in jobs]
    ok, results = True, []
    for j in jobs:
        fx = np.load(j["float32"])
        meta = json.loads(str(fx["meta"]))
        if meta["contract"] != pol.contract:
            raise SystemExit(f"{j['float32']}: fixture contract differs from the bundle's")
        if meta.get("video_delta_indices", [0]) != pol.b.b["video_delta_indices"]:
            raise SystemExit(f"{j['float32']}: fixture frames {meta.get('video_delta_indices', [0])}, "
                             f"bundle {pol.b.b['video_delta_indices']}")
        task = j["task"] or meta["tasks"][0]
        if set(meta["tasks"]) != {task}:
            raise SystemExit(f"{j['float32']}: mixed tasks {set(meta['tasks'])}")
        n = pol.set_prompt(task)         # runtime tokenization, compared with Gr00tPolicy's own
        env = np.load(j["bfloat16"]) if j.get("bfloat16") else None
        passed, rows = check_fixture(pol, fx, env, a.min_cosine, a.max_abs)
        ok &= passed
        results.append({"task": task, "tokens": n, "trained": j.get("trained"), "rtc": bool(j.get("rtc")),
                        "pass": passed, "samples": rows})
        print(f"{'PASS' if passed else 'FAIL'} {'RTC ' if j.get('rtc') else ''}{task!r} ({n} tokens): "
              + "  ".join((f"d{r['rtc_delay']}{'=' if r['prefix_exact'] else '!'} " if r["rtc_delay"] else "")
                          + f"cos {r['cosine']:.5f} max {r['max_abs']:.4f}/{r['limit_max_abs']:.4f}"
                          for r in rows), flush=True)
    summary = {"status": "PASS" if ok else "FAIL", "cosine_gate": a.min_cosine,
               "gate": "bfloat16 Gr00tPolicy envelope", "bundle_fixture": pol.fixture,
               "engines_built_s": pol.built, "load_s": pol.load_s,
               "ms_p50": float(np.median([r["ms"] for x in results for r in x["samples"]])),
               "prompts": results}
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "load_s", "ms_p50")}))
    return 0 if ok else 1


# ── live ────────────────────────────────────────────────────────────────────

EV_SYN, EV_KEY, EV_ABS, SYN_DROPPED = 0, 1, 3, 3
ABS_X, ABS_Y, ABS_RX, ABS_RY, ABS_HAT0X = 0, 1, 3, 4, 0x10
BTN_A, BTN_B, BTN_X = 0x130, 0x131, 0x133   # BTN_SOUTH, BTN_EAST, BTN_NORTH (the recorder's X)
BTN_DPAD_LEFT, BTN_DPAD_RIGHT = 0x222, 0x223
_EVENT = struct.Struct("llHHi")              # struct input_event, 64-bit


class Pad:
    """Xbox evdev pad without python-evdev: same axes, signs and 25% deadzone as
    native_stack/src/gamepad.cpp, so manual driving feels like recording."""

    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        self.axes, self.keys = {}, set()
        self.fifo = stat.S_ISFIFO(os.fstat(self.fd).st_mode)
        self._read_axes()

    def _read_axes(self):
        if self.fifo:                         # tools/virtual_gamepad.py: xpad ranges
            self.axes = {code: [0, -32768, 32767] for code in (ABS_X, ABS_Y, ABS_RX, ABS_RY)}
            return
        for code in (ABS_X, ABS_Y, ABS_RX, ABS_RY):
            buf = bytearray(24)               # struct input_absinfo, six int32
            fcntl.ioctl(self.fd, (2 << 30) | (24 << 16) | (ord("E") << 8) | (0x40 + code), buf)
            v, lo, hi = struct.unpack_from("iii", buf)
            if hi <= lo:
                raise RuntimeError("gamepad must expose Xbox X/Y/RX/RY axes")
            self.axes[code] = [v, lo, hi]

    def drain(self):
        """Drop everything queued so far (engine load can overflow the kernel queue) and
        re-read the sticks. Called just before arming; an overrun after that still stops."""
        while True:
            try:
                if not os.read(self.fd, _EVENT.size * 64):
                    raise RuntimeError("gamepad disconnected")
            except BlockingIOError:
                break
            except OSError as e:
                raise RuntimeError(f"gamepad disconnected ({e})") from e
        self.keys.clear()
        self._read_axes()

    def poll(self) -> list[int]:
        """Read all pending events; return buttons newly pressed since the last poll."""
        pressed = []
        while True:
            try:
                data = os.read(self.fd, _EVENT.size * 64)
            except BlockingIOError:
                return pressed
            except OSError as e:
                raise RuntimeError(f"gamepad disconnected ({e})") from e
            if not data:
                raise RuntimeError("gamepad disconnected")
            for off in range(0, len(data) - len(data) % _EVENT.size, _EVENT.size):
                _, _, typ, code, val = _EVENT.unpack_from(data, off)
                if typ == EV_SYN and code == SYN_DROPPED:
                    raise RuntimeError("gamepad event queue overrun")
                if typ == EV_KEY:
                    if val and code not in self.keys:
                        pressed.append(code)
                    (self.keys.add if val else self.keys.discard)(code)
                elif typ == EV_ABS and code in self.axes:
                    self.axes[code][0] = val
                elif typ == EV_ABS and code == ABS_HAT0X:   # xpad reports the D-pad as a hat
                    for b, on in ((BTN_DPAD_LEFT, val < 0), (BTN_DPAD_RIGHT, val > 0)):
                        if on and b not in self.keys:
                            pressed.append(b)
                        (self.keys.add if on else self.keys.discard)(b)

    def axis(self, code: int) -> float:
        v, lo, hi = self.axes[code]
        x = min(max(v / max(abs(lo), abs(hi) + 1), -1.0), 1.0)
        return 0.0 if abs(x) < 0.25 else float(np.copysign((abs(x) - 0.25) / 0.75, x))

    def action(self, slew: bool) -> tuple:
        """[slew, boom, arm, bucket, trackL, trackR] as Pad::action, tracks off."""
        return (-self.axis(ABS_X) if slew else 0.0, -self.axis(ABS_RY), self.axis(ABS_Y),
                -self.axis(ABS_RX), 0.0, 0.0)

    def close(self):
        os.close(self.fd)


class TaskList:
    """The prompts a run can switch between: the bundle's trained tasks, then --task and every
    prompt typed in the terminal, appended so switching away never loses them."""

    def __init__(self, tasks: list[str], start: str | None):
        self.lock = threading.Lock()
        self.items, self.i, self.version = list(tasks), 0, 0
        if start:
            self.add(start)

    def add(self, text: str) -> int:
        with self.lock:
            same = [k for k, t in enumerate(self.items) if formalize(t) == formalize(text)]
            if same:
                self.i = same[0]
            else:
                self.items.append(text)
                self.i = len(self.items) - 1
            self.version += 1
            return self.i

    def step(self, d: int):
        with self.lock:
            self.i = (self.i + d) % len(self.items)
            self.version += 1

    def current(self) -> tuple[int, str, int]:
        with self.lock:
            return self.i, self.items[self.i], self.version

    def show(self) -> str:
        with self.lock:
            return "\n".join(f"  {'>' if k == self.i else ' '} {k + 1}. {t}" for k, t in enumerate(self.items))


def find_pad(arg: str | None) -> str | None:
    if arg not in (None, "auto"):
        return arg if arg != "none" else None
    found = sorted(Path("/dev/input/by-id").glob("*-event-joystick"))
    return str(found[0]) if found else None


def interp_rows(rows: np.ndarray, pos) -> np.ndarray:
    """The controller's linear interpolation of plan rows at fractional row positions (clamped)."""
    pos = np.clip(np.asarray(pos, np.float64), 0.0, len(rows) - 1)
    i = np.floor(pos).astype(int)
    j = np.minimum(i + 1, len(rows) - 1)
    f = (pos - i)[..., None]
    return rows[i] + (rows[j] - rows[i]) * f


class Supervisor(threading.Thread):
    """Owns the drive connection: arms, sends the newest plan (or the sticks), stops on doubt.

    Model mode renews the newest plan every `renew_s`. Manual mode (gamepad) sends the sticks
    every 25 ms. A toggles model/manual, X toggles pump permission. Without a gamepad a stale or
    exhausted plan, or no first plan within `first_plan_s` of entering model mode or changing
    the prompt, disarms; with one it falls back to manual (sticks centred = hold). Any
    controller state but armed ends the run. It never re-arms.
    """

    LOG = 100_000      # bounded record buffers for unlimited runs

    def __init__(self, client_mod, socket_path, fps, max_obs_age_s=1.4, renew_s=0.1, slew=True,
                 pump=False, pad: Pad | None = None, start_model=False, tasks: TaskList | None = None,
                 first_plan_s=2.0):
        super().__init__(daemon=True)
        self.c = client_mod
        self.control = client_mod.Client(socket_path)
        self.fps, self.max_age, self.renew, self.first_plan_s = fps, max_obs_age_s, renew_s, first_plan_s
        self.slew, self.pump, self.pad, self.tasks = slew, pump, pad, tasks
        self.mode = "model" if (pad is None or start_model) else "manual"
        self.model_since = time.monotonic_ns()
        self.lock = threading.Lock()
        self.plan = None          # (rows [N,6], t0_ns, obs_ns, plan id)
        self.fresh = False
        self.next_id = 0
        self.stop_reason = self.stop_kind = None
        self.sent = 0
        self.applied = collections.deque(maxlen=self.LOG)    # (row pos, planned [4], applied_action [4])
        self.by_sequence = collections.OrderedDict()         # drive packet sequence -> (t0 ns, rows)
        self.dispatch = collections.deque(maxlen=self.LOG)   # first send of each plan: handover record
        self.events = collections.deque(maxlen=self.LOG)     # (monotonic s, what)
        self.ticks = collections.deque(maxlen=self.LOG)      # supervisor loop periods, ms
        self.halt = threading.Event()
        self.last_send = 0

    def flags(self):
        return self.c.DEADMAN | (self.c.PUMP if self.pump else 0)

    def note(self, what):
        self.events.append((round(time.monotonic(), 3), what))
        print(f"[pad] {what}", flush=True)

    def arm(self):
        st = self.control.status()
        if st["state"] != "disarmed":
            raise RuntimeError(f"controller is {st['state']} ({st.get('reason')}), not disarmed")
        self.control.arm(pump=self.pump)

    def set_mode(self, mode, why=""):
        with self.lock:
            self.mode = mode
            self.plan = None      # never resume a plan from before the switch
            self._sent_plan = None  # the next plan follows the centred hold, not the old plan
            self.model_since = time.monotonic_ns()
        self.note(f"{mode.upper()}{' (' + why + ')' if why else ''}")

    def drop_plan(self):
        """A new instruction: stop the old instruction's plan now (centred hold until the first
        chunk observed after this moment)."""
        with self.lock:
            self.plan = None
            self._sent_plan = None
            self.model_since = time.monotonic_ns()

    def running_plan(self):
        """(rows, t0_ns, plan id) the controller is executing, or None."""
        with self.lock:
            return None if self.plan is None else (self.plan[0], self.plan[1], self.plan[3])

    def offer(self, rows: np.ndarray, t0_ns: int, obs_ns: int):
        """-> (accepted, reason, plan id)."""
        with self.lock:
            if self.mode != "model":
                return False, "manual mode", None
            if obs_ns <= self.model_since:
                return False, "observed before the mode/prompt change", None
            if self.plan is not None and obs_ns <= self.plan[2]:
                return False, "not newer than the running plan", None
            self.next_id += 1
            self.plan = (rows, t0_ns, obs_ns, self.next_id)
            self.fresh = True
            return True, None, self.next_id

    def stop(self, reason, kind="finished"):
        if self.stop_reason is None:
            self.stop_reason, self.stop_kind = reason, kind
        self.halt.set()

    def _hold(self, problem):
        """A plan problem: manual fallback with a pad, otherwise end the run (disarm)."""
        if self.pad is None:
            self.stop(problem, "plan")
            return
        self.set_mode("manual", problem)
        self.control.send(self.c.DRIVE, [(0.0,) * 6], flags=self.flags())

    def _model_tick(self, now):
        with self.lock:
            plan, fresh, since = self.plan, self.fresh, self.model_since
            self.fresh = False
        if plan is None:          # centred hold until the first chunk after a switch, bounded
            if (now - since) / 1e9 > self.first_plan_s:
                self._hold(f"no plan {self.first_plan_s:.1f} s after entering model mode / the prompt change")
                return
            self.control.send(self.c.DRIVE, [(0.0,) * 6], flags=self.flags())
            return
        rows, t0, obs, pid = plan
        problem = (f"observation age {(now - obs) / 1e9:.2f} s" if (now - obs) / 1e9 > self.max_age
                   else "chunk exhausted" if now >= t0 + int((len(rows) - 1) / self.fps * 1e9) else None)
        if problem:
            self._hold(problem)
            return
        if fresh or now - self.last_send >= self.renew * 1e9:
            prev = self._sent_plan if fresh else None
            self.control.send(self.c.DRIVE, [tuple(map(float, r)) for r in rows],
                              flags=self.flags(), fps=self.fps, t0_ns=t0, observation_ns=obs)
            self.by_sequence[self.control.sequence] = (t0, rows)
            while len(self.by_sequence) > 256:
                self.by_sequence.popitem(last=False)
            if fresh:     # handover: where each plan stood at this send, and how far they disagree
                pos = (now - t0) * self.fps / 1e9
                rec = {"plan": pid, "send_ns": now, "row": round(pos, 2)}
                if prev is not None:
                    old = interp_rows(prev[0][:, :4], (now - prev[1]) * self.fps / 1e9)
                    rec["jump"] = np.round(np.abs(interp_rows(rows[:, :4], pos) - old), 4).tolist()
                    rec["prev"] = prev[2]
                self.dispatch.append(rec)
            self._sent_plan = (rows, t0, pid)
            self.sent += 1
            self.last_send = now

    def run(self):
        self._sent_plan = None
        period = 0.025 if self.pad else min(self.renew, 0.05)
        last = None
        try:
            while not self.halt.is_set():
                now = time.monotonic_ns()
                if last is not None:
                    self.ticks.append(round((now - last) / 1e6, 2))
                last = now
                if self.pad:
                    try:
                        pressed = self.pad.poll()
                    except RuntimeError as e:
                        self.stop(str(e), "gamepad")
                        break
                    for b in pressed:
                        if b == BTN_A:
                            self.set_mode("manual" if self.mode == "model" else "model")
                        elif b == BTN_X:
                            self.pump = not self.pump
                            self.note(f"pump {'ON' if self.pump else 'OFF'}")
                        elif b == BTN_B:   # operator marker only: one successful operation
                            self.note("success")
                        elif b in (BTN_DPAD_LEFT, BTN_DPAD_RIGHT) and self.tasks:
                            self.drop_plan()
                            self.tasks.step(1 if b == BTN_DPAD_RIGHT else -1)
                            k, text, _ = self.tasks.current()
                            self.note(f"task {k + 1}: {text}")
                if self.mode == "model":
                    self._model_tick(now)
                else:
                    self.control.send(self.c.DRIVE, [self.pad.action(self.slew)], flags=self.flags())
                if self.halt.is_set():
                    break
                st = self.control.status()
                held = self.by_sequence.get(st["applied_command_sequence"])
                if held:   # the controller's command at that tick: the packet it held, interpolated
                    t0, rows = held
                    pos = min(max((st["applied_host_monotonic_ns"] - t0) * self.fps / 1e9, 0.0), len(rows) - 1)
                    due = interp_rows(rows[:, :4], pos)
                    self.applied.append((round(pos, 2), due.tolist(), st["applied_action"][:4]))
                if st["state"] != "armed":
                    self.stop(f"controller {st['state']}: {st.get('reason')}", "controller")
                    break
                self.halt.wait(period)
        except Exception as e:  # noqa: BLE001
            self.stop(f"supervisor error: {e!r}", "error")
        finally:
            self.shutdown()

    def shutdown(self):
        """Disarm and close the connection; safe before start() and when called twice."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.control.send(self.c.DISARM)
        finally:
            self.control.close()


def start_services(a, runtime: Path, procs: list):
    """Start the controller and cameras. Each child joins `procs` as soon as it exists, so the
    caller can stop whatever started when a later step fails."""
    build = Path(a.build)
    control = [build / "kaivuri-control", "--profile", a.profile, "--safety", a.safety,
               "--socket", runtime / "control.sock", "--control-hz", str(a.control_hz),
               "--serial", a.serial]
    if a.live:
        control.append("--live-pwm")
    if a.enable_slew:
        control.append("--enable-slew")
    cams = [build / "kaivuri-camera", "--camera", f"{CAM1}:2", "--camera", f"{CAM2}:1",
            "--output-dir", runtime / "cameras", "--fps", "30", "--width", "640", "--height", "480",
            "--rotate-180", CAM1]
    log = open(runtime / "services.log", "w")
    for cmd, ready in ((control, [runtime / "control.sock"]),
                       (cams, [runtime / "cameras" / f"{CAM1}-ir2.frame", runtime / "cameras" / f"{CAM2}-ir1.frame"])):
        # Own session: a terminal Ctrl+C reaches only this script, which stops them in order.
        p = subprocess.Popen([str(x) for x in cmd], start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        end = time.time() + 30
        while not all(r.exists() for r in ready):
            if p.poll() is not None or time.time() > end:
                raise RuntimeError(f"{cmd[0].name} did not start (see {runtime / 'services.log'})")
            time.sleep(0.1)
    time.sleep(1.5)       # USB streams start one after the other
    return runtime / "control.sock", [runtime / "cameras" / f"{CAM1}-ir2.frame",
                                      runtime / "cameras" / f"{CAM2}-ir1.frame"]


def stop_services(procs: list) -> None:
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)


# ── observation ─────────────────────────────────────────────────────────────


class Incoherent(RuntimeError):
    """This snapshot cannot condition a plan; the loop observes again. `kind` groups the log."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}{' (' + detail + ')' if detail else ''}")
        self.kind = kind


class Observer:
    """One coherent snapshot of both cameras and the controller status.

    Each source is read, then the whole set is validated at ONE final monotonic time: every frame
    fresh and not from the future, the expected geometry and IR stream, no sequence going
    backwards (a restarted camera), camera-to-camera skew and state-to-image separation bounded,
    the IMU healthy, fresh and not from the future. The observation time is the oldest source.
    """

    def __init__(self, readers, files, status_conn, a):
        self.readers, self.status_conn, self.a = readers, status_conn, a
        self.ir = [int(m.group(1)) if (m := re.search(r"-ir(\d)\.frame$", str(f))) else None for f in files]
        self.last_seq = [None] * len(readers)

    def __call__(self):
        a = self.a
        frames, metas = [], []
        for r in self.readers:
            snap, end = None, time.monotonic() + 0.5
            while snap is None:
                snap = r.read()
                if snap is None:
                    if time.monotonic() > end:
                        raise Incoherent("no fresh camera frame")
                    time.sleep(0.002)
            m, px = snap
            frames.append(np.frombuffer(px, np.uint8).reshape(m.height, m.width).copy())
            metas.append((m.sequence, m.host_monotonic_ns, m.width, m.height, m.ir_index))
        st = self.status_conn.status()
        now = time.monotonic_ns()
        cams = []
        for k, (seq, t, w, h, ir) in enumerate(metas):
            age = (now - t) / 1e6
            if age < 0:
                raise Incoherent("frame from the future", f"cam{k + 1} {age:.1f} ms")
            if age > a.max_frame_age_ms:
                raise Incoherent("stale frame", f"cam{k + 1} {age:.0f} ms at the snapshot")
            if (w, h) != (640, 480) or (self.ir[k] is not None and ir != self.ir[k]):
                raise Incoherent("wrong camera stream", f"cam{k + 1} is {w}x{h} ir{ir}, expected 640x480 ir{self.ir[k]}")
            if self.last_seq[k] is not None and seq < self.last_seq[k]:
                was, self.last_seq = self.last_seq[k], [None] * len(self.readers)
                raise Incoherent("camera restarted", f"cam{k + 1} sequence {was} -> {seq}")
            cams.append(t)
        skew = (max(cams) - min(cams)) / 1e6
        if skew > a.max_camera_skew_ms:
            raise Incoherent("camera skew", f"{skew:.1f} ms")
        if not (st["imu_healthy"] and st["fk_valid"]):
            raise Incoherent("IMU unhealthy / FK invalid")
        imu = st["imu_host_monotonic_ns"]
        imu_age = (now - imu) / 1e6
        if imu_age < -1.0:
            raise Incoherent("IMU from the future", f"{imu_age:.1f} ms")
        if imu_age > a.max_imu_age_ms:
            raise Incoherent("stale IMU", f"{imu_age:.0f} ms")
        sep = max(abs(imu - t) for t in cams) / 1e6
        if sep > a.max_state_skew_ms:
            raise Incoherent("state-to-image separation", f"{sep:.1f} ms")
        self.last_seq = [m[0] for m in metas]
        sources = {"snapshot_ns": now, "imu_ns": imu, "cameras": [
            {"seq": int(m[0]), "ns": int(m[1])} for m in metas]}
        return frames, st, min([*cams, imu]), skew, sources


def frame_ns(sources: dict) -> int:
    """An observation's image time for the frame history: its oldest camera frame."""
    return min(c["ns"] for c in sources["cameras"])


# ── RTC ─────────────────────────────────────────────────────────────────────


class RtcDelay:
    """Committed rows d: from the observation (row 0) until the controller applies the new plan.
    The recent maximum observation -> result latency plus a dispatch margin, capped at the trained
    maximum; the prefix must also lie inside the running plan."""

    def __init__(self, max_delay: int, fps: float, margin_ms: float, window: int = 20):
        self.max, self.fps, self.margin = max_delay, fps, margin_ms
        self.lat = collections.deque(maxlen=window)

    def add(self, ms: float):
        self.lat.append(ms)

    def want(self) -> int:
        if not self.lat or self.max == 0:
            return self.max
        return min(self.max, math.ceil((max(self.lat) + self.margin) * self.fps / 1e3))


def committed_rows(pol: Policy, running, t0_ns: int, want: int, fps: float):
    """(d, prefix [d,A] model units, plan id): what `running` (rows, t0, id) executes at rows
    0..d-1 of a chunk anchored at t0_ns, as the controller interpolates it."""
    if running is None or want == 0:
        return 0, None, None
    rows, t0_old, pid = running
    pos = (t0_ns - t0_old) * fps / 1e9 + np.arange(want)
    d = int(np.sum((pos >= 0) & (pos <= len(rows) - 1)))
    if d == 0 or pos[0] < 0:
        return 0, None, pid
    valves = interp_rows(rows[:, pol.boundary.action_cols], pos[:d])
    return d, pol.boundary.model_actions(valves), pid


# ── live ────────────────────────────────────────────────────────────────────

EXIT = {"finished": 0, "interrupted": 130, "gamepad": 3, "controller": 4, "plan": 5, "error": 1}


def sha256(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def stats(xs) -> dict | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "p50": round(float(np.percentile(xs, 50)), 3),
            "p99": round(float(np.percentile(xs, 99)), 3), "max": round(float(max(xs)), 3)}


def write_summary(out: Path, a, chunks, sup, outcome: dict, tasks=None, extra=None) -> dict:
    np.save(out / "chunks.npy", np.stack(chunks) if chunks else np.zeros((0,)))
    summary = {"outcome": outcome, "chunks": len(chunks), "drive": a.live,
               "plans_sent": sup.sent if sup else 0, "pad_events": list(sup.events) if sup else [],
               "tasks": tasks.items if tasks else None, **(extra or {})}
    if sup and sup.applied:
        planned = np.array([p for _, p, _ in sup.applied])
        applied = np.array([q for _, _, q in sup.applied])
        summary["applied_vs_planned"] = {
            "samples": len(applied), "mean_abs": float(np.abs(applied - planned).mean()),
            "max_abs": float(np.abs(applied - planned).max()),
            "row_due": {"min": min(k for k, _, _ in sup.applied), "max": max(k for k, _, _ in sup.applied)}}
        json.dump(list(sup.applied), open(out / "applied.json", "w"))
    if sup:
        json.dump(list(sup.dispatch), open(out / "dispatch.json", "w"))
        jumps = [max(r["jump"]) for r in sup.dispatch if "jump" in r]
        summary["handover"] = {"max_jump": stats(jumps), "row": stats([r["row"] for r in sup.dispatch]),
                               "over_0.5": sum(j > 0.5 for j in jumps)}
        summary["supervisor_tick_ms"] = stats(list(sup.ticks))
    recs = [json.loads(l) for l in open(out / "chunks.jsonl")] if (out / "chunks.jsonl").exists() else []
    inferred = [r for r in recs if "obs_to_plan_ms" in r]
    if recs:
        summary["rejected_observations"] = collections.Counter(r["rejected"] for r in recs if "rejected" in r)
        summary["offers"] = collections.Counter(r.get("reason") or ("accepted" if r.get("accepted") else "not offered")
                                                for r in inferred)
    if inferred:
        summary["chunks_per_task"] = {t: sum(r.get("task") == t for r in inferred)
                                      for t in dict.fromkeys(r.get("task") for r in inferred)}
        summary.update(obs_to_plan_ms=stats([r["obs_to_plan_ms"] for r in inferred]),
                       infer_ms=stats([r["timings_ms"]["total"] for r in inferred]),
                       camera_skew_ms=stats([r["camera_skew_ms"] for r in inferred]),
                       rtc_delay=stats([r["rtc_delay"] for r in inferred]),
                       rtc_short=sum(r["rtc_delay"] < r["rtc_want"] for r in inferred),
                       history_age_ms=stats([r["timings_ms"].get("history_age") for r in inferred]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def run_live(a) -> int:
    # Started in the background by a non-interactive shell, SIGINT arrives ignored; restore it
    # so Ctrl+C / kill -INT always takes the graceful path below.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.path.insert(0, str(Path(a.tools)))
    import client as client_mod
    from frame_reader import FrameReader

    contract = json.loads(Path(a.contract).read_text()) if a.contract else None
    out = Path(a.out) if a.out else None       # None: nothing is written
    if out:
        out.mkdir(parents=True, exist_ok=False)
    identity = {"argv": sys.argv, "groot_live_sha256": sha256(__file__), "bundle": a.bundle,
                "bundle_json_sha256": sha256(Path(a.bundle) / "bundle.json"), "cache": a.cache,
                "vla_root": a.vla_root, "profile": a.profile, "safety": a.safety,
                "safety_sha256": sha256(a.safety), "control_hz": a.control_hz, "fps": a.fps,
                "start_monotonic_ns": time.monotonic_ns(), "start_unix": time.time()}
    if out:
        (out / "run.json").write_text(json.dumps(identity, indent=1) + "\n")
    procs, tmp, sup, pad, tasks, status_conn, readers, pol = [], None, None, None, None, None, [], None
    chunks = collections.deque(maxlen=20_000)
    outcome = {"kind": "finished", "reason": "run finished"}
    rtc_on = False
    try:
        pad_path = find_pad(a.gamepad) if a.live else None
        if pad_path:
            pad = Pad(pad_path)
            print(f"[pad] {pad_path}: A = model/manual, X = pump on/off, Ctrl+C = stop", flush=True)
        elif a.live:
            print("[pad] no gamepad: model drives until the run ends; Ctrl+C = stop", flush=True)
        if a.start_services:
            tmp = tempfile.TemporaryDirectory(prefix="kvr-groot-")
            sock, cam_files = start_services(a, Path(tmp.name), procs)
        else:
            sock, cam_files = a.control_socket, a.camera_frame
        pol = Policy(Path(a.bundle), contract, Path(a.cache).expanduser(), Path(a.vla_root),
                     history_tolerance_s=a.history_tolerance_ms / 1e3)
        rtc_on = bool(pol.rtc) and a.rtc != "off"
        if a.rtc == "on" and not pol.rtc:
            raise ValueError("--rtc on: this bundle has no RTC inputs")
        print(f"[groot] engines ready in {pol.load_s} s, fixture {pol.fixture['status']} "
              f"(chunk cos {pol.fixture['reports']['chunk']['cosine']:.6f}), RTC "
              + (f"ON (max {pol.rtc_max} rows, margin {a.rtc_margin_ms:g} ms)" if rtc_on
                 else "off" + (" (--rtc off)" if pol.rtc else " (bundle has none)")), flush=True)
        if pol.history:
            print(f"[groot] frame history: each camera now and {pol.lag_s:.2f} s ago "
                  f"(video_delta_indices {pol.b.b['video_delta_indices']}, "
                  f"tolerance {a.history_tolerance_ms:g} ms)", flush=True)
        if a.task:
            pol.prompts.ids(a.task)                     # refuse an unusable --task before arming
        tasks = TaskList(pol.prompts.tasks, a.task)
        prompt_note = (f"type a prompt + Enter to switch to it (kept in the list, up to "
                       f"{pol.prompts.capacity} tokens); '?' lists the tasks"
                       if pol.prompts.dynamic else "this bundle bakes one prompt")
        print(f"[task] {prompt_note}; D-pad right/left cycles\n{tasks.show()}", flush=True)
        for p in cam_files:
            readers.append(FrameReader(p, Path(a.build) / "libkaivuri_frames.so",
                                       max_age_s=a.max_frame_age_ms / 1e3))
        status_conn = client_mod.Client(sock)
        observe = Observer(readers, cam_files, status_conn, a)
        rng = np.random.default_rng(a.seed)
        fps = float(a.fps)
        delay = RtcDelay(pol.rtc_max if rtc_on else 0, fps, a.rtc_margin_ms)
        log = open(out / "chunks.jsonl" if out else os.devnull, "w")

        def observe_retry(deadline_s=2.0):
            end = time.monotonic() + deadline_s
            while True:
                try:
                    return observe()
                except Incoherent:
                    if time.monotonic() > end:
                        raise
                    time.sleep(0.005)

        # Warm-up: one inference before arming, so the first plan's latency (RTC delay) is known; a
        # frame-history bundle first fills its history (no plan is made until then).
        fill_end = time.monotonic() + pol.lag_s + 10
        while True:
            frames, st, obs_ns, _, src = observe_retry()
            try:
                pol.infer([gray_to_rgb(f) for f in frames], pol.boundary.degrees(st),
                          rng.standard_normal(pol.noise_shape).astype(np.float32), capture_ns=frame_ns(src))
            except HistoryFilling:
                if time.monotonic() > fill_end:
                    raise RuntimeError("the frame history did not fill during the warm-up")
                continue
            delay.add((time.monotonic_ns() - obs_ns) / 1e6)
            break
        if a.live:
            sup = Supervisor(client_mod, sock, fps, a.max_observation_age_ms / 1e3, slew=a.enable_slew,
                             pump=True, pad=pad, start_model=a.start_model, tasks=tasks,
                             first_plan_s=a.first_plan_timeout_s)
            if pad:
                pad.drain()
            sup.arm()
            sup.start()
            print(f"[groot] armed, {sup.mode.upper()} mode, pump ON", flush=True)

        def read_prompts():
            """Terminal lines become prompts: appended to the task list and selected."""
            for line in sys.stdin:
                text = line.strip()
                if not text:
                    continue
                if text == "?":
                    print(tasks.show(), flush=True)
                    continue
                try:
                    n = pol.prompts.text_tokens(text)
                except ValueError as e:
                    print(f"[task] not used: {e}", flush=True)
                    continue
                if sup:
                    sup.drop_plan()
                k = tasks.add(text)
                print(f"[task] {k + 1}: {text} ({n} text tokens)", flush=True)

        if sys.stdin and not sys.stdin.closed:
            threading.Thread(target=read_prompts, daemon=True).start()
        applied, virtual, first_written = None, None, False
        t_end = time.monotonic() + a.seconds if a.seconds > 0 else float("inf")
        next_start = time.monotonic()
        n = 0
        while time.monotonic() < t_end and not (sup and sup.halt.is_set()):
            if sup and sup.mode != "model":       # manual: no inference, sticks drive
                if pol.history:                   # keep the history warm for the switch back
                    try:
                        frames, _, _, _, src = observe()
                        pol.remember([gray_to_rgb(f) for f in frames], frame_ns(src))
                    except Incoherent:
                        pass
                time.sleep(0.02)
                next_start = time.monotonic()
                continue
            if a.rate_hz > 0:
                time.sleep(max(0.0, next_start - time.monotonic()))
                next_start += 1.0 / a.rate_hz
            k, text, version = tasks.current()
            if version != applied:
                pol.set_prompt(text)
                applied, virtual = version, None
                print(f"[task {k + 1}/{len(tasks.items)}] {text!r} ({pol.task_tokens} prompt tokens)", flush=True)
            try:
                frames, st, obs_ns, skew, sources = observe()
            except Incoherent as e:
                log.write(json.dumps({"rejected": e.kind, "detail": str(e), "ns": time.monotonic_ns()}) + "\n")
                time.sleep(0.005)
                continue
            if out and not first_written:        # what the model saw first, for an orientation check
                for c, f in enumerate(frames, 1):
                    (out / f"cam{c}_first.pgm").write_bytes(
                        f"P5\n{f.shape[1]} {f.shape[0]}\n255\n".encode() + f.tobytes())
                first_written = True
            deg = pol.boundary.degrees(st)
            t = {}
            noise = rng.standard_normal(pol.noise_shape).astype(np.float32)
            t0_ns = obs_ns + int(a.action_offset_ms * 1e6)
            want = delay.want() if rtc_on else 0
            running = sup.running_plan() if sup else virtual
            d, prefix, prefix_plan = committed_rows(pol, running, t0_ns, want, fps)
            try:
                act, model = pol.infer([gray_to_rgb(f) for f in frames], deg, noise, t, rtc_delay=d,
                                       rtc_prefix=prefix, capture_ns=frame_ns(sources))
            except HistoryFilling as e:           # the encode is stored; no plan from this frame
                log.write(json.dumps({"rejected": "history filling", "detail": str(e),
                                      "ns": time.monotonic_ns()}) + "\n")
                continue
            done_ns = time.monotonic_ns()
            delay.add((done_ns - obs_ns) / 1e6)
            rows = np.zeros((len(act), 6), np.float32)
            rows[:, pol.boundary.action_cols] = act
            if not a.enable_slew:
                rows[:, 0] = 0.0
            # A chunk of an instruction that changed meanwhile is never offered.
            if tasks.current()[2] != applied:
                accepted, reason, pid = False, "prompt changed during inference", None
            elif sup:
                accepted, reason, pid = sup.offer(rows, t0_ns, obs_ns)
            else:                         # no drive: the chunk becomes the virtual running plan
                accepted, reason, pid = None, None, n + 1
                virtual = (rows, t0_ns, pid)
            lag_rows = (done_ns - obs_ns) / 1e9 * fps
            rec = {"i": n, "obs_ns": obs_ns, "t0_ns": t0_ns, "done_ns": done_ns,
                   "obs_to_plan_ms": round((done_ns - obs_ns) / 1e6, 1), "stale_rows": round(lag_rows, 1),
                   "camera_skew_ms": round(skew, 1), "sources": sources, "state_deg": deg.round(2).tolist(),
                   "rtc_delay": d, "rtc_want": want, "prefix_plan": prefix_plan,
                   "row_now": act[min(int(lag_rows), len(act) - 1)].round(3).tolist(),
                   "row0": act[0].round(3).tolist(), "abs_max": np.abs(act).max(0).round(3).tolist(),
                   "timings_ms": {kk: round(v, 1) for kk, v in t.items()}, "accepted": accepted,
                   "reason": reason, "plan_id": pid, "pump": sup.pump if sup else None, "task": pol.task}
            chunks.append(act)
            n += 1
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{rec['i']:4d}] obs->plan {rec['obs_to_plan_ms']:6.1f} ms  infer {t['total']:6.1f}  "
                  + (f"past {t['history_age']:5.0f} ms  " if "history_age" in t else "")
                  + f"rtc {d:2d}/{want:2d}  deg {deg.round(1)}  now {np.round(rec['row_now'], 2)}", flush=True)
        if sup and sup.stop_kind:
            outcome = {"kind": sup.stop_kind, "reason": sup.stop_reason}
    except KeyboardInterrupt:
        outcome = {"kind": "interrupted", "reason": "Ctrl+C"}
        print("\n[groot] Ctrl+C: disarming and stopping services", flush=True)
    except Exception as e:  # noqa: BLE001
        outcome = {"kind": "error", "reason": repr(e)}
        import traceback
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # let the cleanup finish
        cleanup = []

        def step(name, fn):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                cleanup.append(f"{name}: {e!r}")

        if sup:
            step("supervisor stop", lambda: sup.stop(outcome["reason"], outcome["kind"]))
            if sup.ident is not None:
                step("supervisor join", lambda: sup.join(timeout=2))
            if sup.is_alive():
                cleanup.append("supervisor did not finish in 2 s")
            step("disarm", sup.shutdown)        # no-op after run(); disarms when arming failed
            if sup.stop_kind and outcome["kind"] == "finished":
                outcome = {"kind": sup.stop_kind, "reason": sup.stop_reason}
        final = {}
        if status_conn:
            def final_status():
                """The controller's own state after the disarm; armed past 1 s is a cleanup error."""
                end = time.monotonic() + 1.0
                while True:
                    st = status_conn.status()
                    if st["state"] != "armed" or not sup or time.monotonic() > end:
                        break
                    time.sleep(0.02)
                final.update({k: st.get(k) for k in ("state", "reason", "pump_on")})
                if sup and st["state"] == "armed":
                    raise RuntimeError("controller still armed 1 s after the disarm")
            step("final status", final_status)
            step("status close", status_conn.close)
        step("services", lambda: stop_services(procs))
        for r in readers:
            step("frame reader", r.close)
        if tmp:
            step("runtime dir", tmp.cleanup)
        if pad:
            step("gamepad", pad.close)
        outcome["final_controller"] = final or None
        outcome["cleanup_errors"] = cleanup
        outcome["exit"] = EXIT.get(outcome["kind"], 1)
        if not out:
            print(json.dumps({"outcome": outcome, "chunks": len(chunks),
                              "plans_sent": sup.sent if sup else 0}), flush=True)
        else:
            try:
                summary = write_summary(out, a, chunks, sup, outcome, tasks,
                                        {"rtc": {"on": rtc_on, "max_delay": pol.rtc_max if rtc_on else 0,
                                                 "margin_ms": a.rtc_margin_ms} if pol else None})
                print(json.dumps({k: v for k, v in summary.items() if k not in ("pad_events", "tasks")}),
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[groot] summary not written: {e!r}", flush=True)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    return outcome["exit"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("parity", "run"):
        s = sub.add_parser(name)
        s.add_argument("--bundle", required=True)
        s.add_argument("--contract", help="normalization contract JSON (default: the bundle's kaivuri.contract)")
        s.add_argument("--cache", default="~/.cache/kaivuri-groot/default")
        s.add_argument("--vla-root", default=str(Path.home() / "jetson-orin-nano-vla"))
        s.add_argument("--out", help="report directory (default: nothing is written)")
    s = sub.choices["parity"]
    s.add_argument("--fixture", help="one physical fixture (default: every prompt fixture in the bundle)")
    s.add_argument("--envelope", help="the same fixture from bfloat16 Gr00tPolicy: per-sample error limits")
    s.add_argument("--max-abs", type=float, default=0.02,
                   help="valve units: the max-error gate without an envelope, the floor of the envelope gate")
    s.add_argument("--min-cosine", type=float, default=0.999)
    s = sub.choices["run"]
    s.add_argument("--tools", default=str(Path.home() / "kaivuriprokkis/native_stack/tools"))
    s.add_argument("--build", default=str(Path.home() / "kaivuriprokkis/native_stack/build"))
    s.add_argument("--profile", default=str(Path.home() / "kaivuriprokkis/configuration_files/profiles/jetson"))
    s.add_argument("--safety", default=str(Path.home() / "kaivuriprokkis/native_stack/config/safety.yaml"))
    s.add_argument("--serial", default="/dev/ttyACM0")
    s.add_argument("--control-hz", type=int, default=200)
    s.add_argument("--start-services", action="store_true")
    s.add_argument("--enable-slew", action="store_true")
    s.add_argument("--control-socket")
    s.add_argument("--camera-frame", action="append", help="cam1 then cam2 frame files")
    s.add_argument("--live", action="store_true",
                   help="drive the machine: plans, pump permission on (X toggles it on the pad), "
                        "live PWM with --start-services")
    s.add_argument("--gamepad", default="auto", help="evdev path, 'auto' (first by-id joystick) or 'none'")
    s.add_argument("--start-model", action="store_true", help="with a gamepad, start in model mode, not manual")
    s.add_argument("--task", help="starting prompt (default: the first trained task); added to the list")
    s.add_argument("--seconds", type=float, default=30, help="0 = until Ctrl+C")
    s.add_argument("--rate-hz", type=float, default=0, help="0 = back to back")
    s.add_argument("--fps", type=float, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--action-offset-ms", type=float, default=0)
    s.add_argument("--max-frame-age-ms", type=float, default=200)
    s.add_argument("--max-imu-age-ms", type=float, default=50)
    s.add_argument("--max-observation-age-ms", type=float, default=1400)
    s.add_argument("--max-camera-skew-ms", type=float, default=50,
                   help="camera-to-camera arrival skew (recorded live max 22.8 ms; 30 fps = 33 ms/frame)")
    s.add_argument("--max-state-skew-ms", type=float, default=100, help="IMU sample vs each camera frame")
    s.add_argument("--first-plan-timeout-s", type=float, default=2.0,
                   help="model mode without a plan (start, A, prompt change) holds centred at most this long")
    s.add_argument("--rtc", choices=("auto", "on", "off"), default="auto",
                   help="auto: training-time RTC when the bundle has RTC inputs")
    s.add_argument("--rtc-margin-ms", type=float, default=40,
                   help="added to the recent max observation->result latency: supervisor tick + dispatch")
    s.add_argument("--history-tolerance-ms", type=float, default=150,
                   help="frame-history bundles: how far the past frame may sit from the trained lag "
                        "(back-to-back inference stores one every ~250 ms)")
    a = p.parse_args()
    if getattr(a, "task", None) is not None and not a.task.strip():
        p.error("--task is empty (unset shell variable?); omit it to start on the first trained task")
    # Policy() changes into --vla-root (its engine builders import from there).
    for k in ("bundle", "contract", "cache", "vla_root", "out", "fixture", "envelope", "tools", "build",
              "profile", "safety", "control_socket"):
        if getattr(a, k, None):
            setattr(a, k, str(Path(getattr(a, k)).expanduser().resolve()))
    if getattr(a, "camera_frame", None):
        a.camera_frame = [str(Path(x).resolve()) for x in a.camera_frame]
    return parity(a) if a.cmd == "parity" else run_live(a)


if __name__ == "__main__":
    sys.exit(main())
