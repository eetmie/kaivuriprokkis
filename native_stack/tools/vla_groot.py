"""GR00T N1.7 fine-tune adapter for vla_live.py: bundle, engines and the excavator boundary.

Engines and the host chain come from jetson-orin-nano-vla (`--vla-root`, imported read-only).
This file owns the excavator boundary that Gr00tPolicy applies on the Spark:

    cameras  native IR Y8 -> gray repeated to RGB -> the bundle's eval transform
    state    angles_rad[boom, arm, bucket] -> float32 degrees -> min-max [lo, hi] -> [-1, 1], clipped
    actions  model [-1, 1] -> clipped -> [lo, hi] (the contract's stats) -> [slew, lift, tilt, scoop]

`parity` runs a physical fixture (spark-projects groot/excavator/physical_fixture.py) from raw
frames and degrees to valve commands.

RTC (bundles with an `rtc` section, a training-time RTC checkpoint): each inference commits the
first d rows of its chunk to what the running plan executes at those times, and the model
denoises only the rest (vla_live.committed_rows).

Frame history (bundles trained with GROOT_VIDEO_DELTAS="-k,0"): the model also sees each camera k
dataset frames earlier. The runtime keeps the vision encodes of its own earlier observations and
feeds the one closest to k/fps old, within --history-tolerance-ms; until it has one (start-up)
no plan is made, only the encode is stored. `parity` takes the past frames from the fixture
(`raw_past`).
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from vla_observe import HistoryFilling
from vla_operator import JOINTS


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
        self.fps = float((bb.get("kaivuri") or {}).get("fps") or 30)   # the dataset rate it was trained at

    # ── the model interface vla_live.py drives ──

    name = "GR00T N1.7"

    @property
    def action_cols(self) -> list[int]:
        """Plan columns of the chunk's columns."""
        return self.boundary.action_cols

    def model_actions(self, valves: np.ndarray) -> np.ndarray:
        return self.boundary.model_actions(valves)

    @property
    def tasks(self) -> list[str]:
        return self.prompts.tasks

    def check_prompt(self, text: str) -> int:
        """Instruction token count; ValueError when the text cannot run."""
        return self.prompts.text_tokens(text)

    def prompt_note(self) -> str:
        return (f"type a prompt + Enter to switch to it (kept in the list, up to {self.prompts.capacity} "
                f"tokens); '?' lists the tasks" if self.prompts.dynamic else "this bundle bakes one prompt")

    def same_prompt(self, a: str, b: str) -> bool:
        return formalize(a) == formalize(b)

    def describe(self, rtc_on: bool, a) -> str:
        line = (f"[groot] engines ready in {self.load_s} s, fixture {self.fixture['status']} "
                f"(chunk cos {self.fixture['reports']['chunk']['cosine']:.6f}), RTC "
                + (f"ON (max {self.rtc_max} rows, margin {a.rtc_margin_ms:g} ms)" if rtc_on
                   else "off" + (" (--rtc off)" if self.rtc else " (bundle has none)")))
        if self.history:
            line += (f"\n[groot] frame history: each camera now and {self.lag_s:.2f} s ago "
                     f"(video_delta_indices {self.b.b['video_delta_indices']}, "
                     f"tolerance {a.history_tolerance_ms:g} ms)")
        return line

    def step(self, frames_y8: list[np.ndarray], status: dict, noise: np.ndarray, timings: dict,
             rtc_delay: int = 0, rtc_prefix: np.ndarray | None = None,
             capture_ns: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Live step: IR frames + controller status -> (valve chunk [H, A], the degrees used)."""
        deg = self.boundary.degrees(status)
        act, _ = self.infer([gray_to_rgb(f) for f in frames_y8], deg, noise, timings, rtc_delay=rtc_delay,
                            rtc_prefix=rtc_prefix, capture_ns=capture_ns)
        return act, deg

    def keep_warm(self, frames_y8: list[np.ndarray], capture_ns: int) -> None:
        """Manual mode: no plan, but a frame-history bundle keeps encoding for the switch back."""
        if self.history is not None:
            self.remember([gray_to_rgb(f) for f in frames_y8], capture_ns)

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
