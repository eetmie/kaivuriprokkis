"""GR00T N1.7 fine-tune on the native stack: physical parity and a live drive loop.

Engines and the host chain come from jetson-orin-nano-vla (`--vla-root`, imported read-only).
This file owns the excavator boundary that Gr00tPolicy applies on the Spark:

    cameras  native IR Y8 -> gray repeated to RGB -> the bundle's eval transform
    state    angles_rad[boom, arm, bucket] -> float32 degrees -> min-max [lo, hi] -> [-1, 1], clipped
    actions  model [-1, 1] -> clipped -> [lo, hi] (the contract's stats) -> [slew, lift, tilt, scoop]

`parity` runs a physical fixture (spark-projects groot/excavator/physical_fixture.py) from raw
frames and degrees to valve commands. `run` reads live cameras and the controller, infers
continuously and, with --drive, sends each chunk as a timed plan (t0 = observation time, so
the rows that went stale during inference are skipped by the controller). A supervisor thread
renews the plan every 100 ms and disarms on chunk exhaustion, on observation age, on any
controller state but armed and on any inference error. It never re-arms. With an Xbox pad
(--gamepad auto): A toggles manual (sticks drive, inference paused) and model; a stale or
exhausted chunk then falls back to manual instead of disarming. X toggles pump permission.
Ctrl+C disarms, stops the services and still writes the summary.

Prompts: bundles from spark-projects groot/excavator/export_bundle.py tokenize at runtime. The
run starts on --task or the first trained task; D-pad right/left cycles the list, and a line
typed in the terminal becomes a prompt, appended to the list. A change drops the current plan;
the machine holds until the first chunk observed under the new prompt.

    python groot_live.py parity --bundle B                 # every prompt fixture in the bundle
    python groot_live.py run --bundle B --start-services --live-pwm --enable-slew --drive --out DIR
"""
from __future__ import annotations

import argparse
import fcntl
import json
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
        """[1,H,W] model actions -> [H,A] valve commands (unnormalize_values_minmax)."""
        x = np.clip(model[0, :, :len(self.a_lo)].astype(np.float64), -1.0, 1.0)
        return ((x + 1.0) / 2.0 * (self.a_hi - self.a_lo) + self.a_lo).astype(np.float32)

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


class Policy:
    """Bundle + engines + boundary; one current frame per camera (video_delta_indices [0])."""

    def __init__(self, bundle: Path, contract: dict | None, cache: Path, vla_root: Path):
        sys.path.insert(0, str(vla_root))
        os.chdir(vla_root)      # the engine builders run `-c "from bench.vendor..."` subprocesses
        from bench.vendor import groot17_trt as g17
        from bench.vendor import groot_trt as g16

        self.g17, self.g16 = g17, g16
        self.b = g17.Bundle(bundle)
        bb = self.b.b
        if bb["frames"] != 1 or bb["video_delta_indices"] != [0]:
            raise ValueError("this runtime handles current-frame-only bundles")
        contract = contract or (bb.get("kaivuri") or {}).get("contract")
        if contract is None:
            raise ValueError("the bundle carries no normalization contract: pass --contract")
        if bb["embodiment"] != contract["embodiment"]:
            raise ValueError("contract and bundle embodiments differ")
        self.contract = contract
        self.boundary = Boundary(contract)
        self.prompts = Prompts(Path(bundle), bb)
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

    def infer(self, frames_rgb: list[np.ndarray], degrees: np.ndarray, noise: np.ndarray,
              timings: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        t = {} if timings is None else timings
        t0 = time.perf_counter()
        pv = np.stack([self.b.preprocess(f) for f in frames_rgb])
        state = self.boundary.state(degrees, self.state_width)
        t1 = time.perf_counter()
        enc = self.g17.encode_frames(self.b, self.engines.run, pv)
        t2 = time.perf_counter()
        model = self.g17.infer(self.b, self.engines.run, [enc], state, noise, timings=t)
        t3 = time.perf_counter()
        t.update(preprocess=(t1 - t0) * 1e3, vision=(t2 - t1) * 1e3, total=(t3 - t0) * 1e3)
        return self.boundary.actions(model), model


def gray_to_rgb(y8: np.ndarray) -> np.ndarray:
    return np.repeat(y8[..., None], 3, axis=2)


# ── parity ──────────────────────────────────────────────────────────────────


def check_fixture(pol: Policy, fx, env, min_cosine: float, max_abs: float) -> tuple[bool, list]:
    """Raw frames + degrees -> valves through `pol` vs the fixture's FP32 Gr00tPolicy chunks.
    Gate per sample: cosine, and no further from FP32 than bfloat16 Gr00tPolicy (`env`)."""
    if env is not None and not np.array_equal(env["noise"], fx["noise"]):
        raise SystemExit("envelope fixture uses different inputs")
    rows, ok = [], True
    for i in range(len(fx["raw"])):
        raw = fx["raw"][i]
        # The dataset decodes gray IR as identical RGB channels; the live path does too.
        gray_ok = all(np.array_equal(raw[v][..., 0], raw[v][..., c]) for v in range(len(raw)) for c in (1, 2))
        frames = [gray_to_rgb(raw[v][..., 0]) if gray_ok else raw[v] for v in range(len(raw))]
        t = {}
        act, _ = pol.infer(frames, fx["state_deg"][i], fx["noise"][i][None], t)
        ref = fx["actions"][i]
        d = np.abs(act - ref)
        cos = float((act.ravel() @ ref.ravel()) / (np.linalg.norm(act) * np.linalg.norm(ref)))
        if env is not None:
            e = np.abs(env["actions"][i] - ref)     # floored: bf16 can land closer than FP16 noise
            lim_max, lim_mean = max(float(e.max()), max_abs), max(float(e.mean()), max_abs / 10)
        else:
            lim_max, lim_mean = max_abs, np.inf
        passed = bool(d.max() <= lim_max and d.mean() <= lim_mean and cos >= min_cosine)
        ok &= passed
        rows.append({"pass": passed, "episode": int(fx["episode"][i]), "frame": int(fx["frame"][i]),
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
        jobs = [{**j, "float32": root / j["float32"], "bfloat16": root / j["bfloat16"]} for j in jobs]
    ok, results = True, []
    for j in jobs:
        fx = np.load(j["float32"])
        meta = json.loads(str(fx["meta"]))
        if meta["contract"] != pol.contract:
            raise SystemExit(f"{j['float32']}: fixture contract differs from the bundle's")
        task = j["task"] or meta["tasks"][0]
        if set(meta["tasks"]) != {task}:
            raise SystemExit(f"{j['float32']}: mixed tasks {set(meta['tasks'])}")
        n = pol.set_prompt(task)         # runtime tokenization, compared with Gr00tPolicy's own
        env = np.load(j["bfloat16"]) if j.get("bfloat16") else None
        passed, rows = check_fixture(pol, fx, env, a.min_cosine, a.max_abs)
        ok &= passed
        results.append({"task": task, "tokens": n, "trained": j.get("trained"),
                        "pass": passed, "samples": rows})
        print(f"{'PASS' if passed else 'FAIL'} {task!r} ({n} tokens): "
              + "  ".join(f"cos {r['cosine']:.5f} max {r['max_abs']:.4f}/{r['limit_max_abs']:.4f}"
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
        if stat.S_ISFIFO(os.fstat(self.fd).st_mode):   # tools/virtual_gamepad.py: xpad ranges
            self.axes = {code: [0, -32768, 32767] for code in (ABS_X, ABS_Y, ABS_RX, ABS_RY)}
            return
        for code in (ABS_X, ABS_Y, ABS_RX, ABS_RY):
            buf = bytearray(24)               # struct input_absinfo, six int32
            fcntl.ioctl(self.fd, (2 << 30) | (24 << 16) | (ord("E") << 8) | (0x40 + code), buf)
            v, lo, hi = struct.unpack_from("iii", buf)
            if hi <= lo:
                raise RuntimeError("gamepad must expose Xbox X/Y/RX/RY axes")
            self.axes[code] = [v, lo, hi]

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


class Supervisor(threading.Thread):
    """Owns the drive connection: arms, sends the newest plan (or the sticks), stops on doubt.

    Model mode renews the newest plan every `renew_s`. Manual mode (gamepad) sends the sticks
    every 25 ms. A toggles model/manual, X toggles pump permission. Without a gamepad a stale or
    exhausted plan disarms; with one it falls back to manual (sticks centred = hold). Any
    controller state but armed ends the run. It never re-arms.
    """

    def __init__(self, client_mod, socket_path, fps, max_obs_age_s=1.4, renew_s=0.1, slew=True,
                 pump=False, pad: Pad | None = None, start_model=False, tasks: TaskList | None = None):
        super().__init__(daemon=True)
        self.c = client_mod
        self.control = client_mod.Client(socket_path)
        self.fps, self.max_age, self.renew = fps, max_obs_age_s, renew_s
        self.slew, self.pump, self.pad, self.tasks = slew, pump, pad, tasks
        self.mode = "model" if (pad is None or start_model) else "manual"
        self.model_since = time.monotonic_ns()
        self.lock = threading.Lock()
        self.plan = None          # (rows [N,6], t0_ns, obs_ns)
        self.fresh = False
        self.stop_reason = None
        self.sent = 0
        self.applied = []         # (row position, planned [4] interpolated, controller applied_action [4])
        self.by_sequence = {}     # drive packet sequence -> (t0 ns, rows)
        self.events = []          # (monotonic s, what)
        self.halt = threading.Event()

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
            self.model_since = time.monotonic_ns()
        self.note(f"{mode.upper()}{' (' + why + ')' if why else ''}")

    def drop_plan(self):
        """A new instruction: stop the old instruction's plan now (centred hold until the first
        chunk observed after this moment)."""
        with self.lock:
            self.plan = None
            self.model_since = time.monotonic_ns()

    def offer(self, rows: np.ndarray, t0_ns: int, obs_ns: int):
        with self.lock:
            if self.mode != "model" or obs_ns <= self.model_since:
                return False
            if self.plan is not None and obs_ns <= self.plan[2]:
                return False
            self.plan = (rows, t0_ns, obs_ns)
            self.fresh = True
            return True

    def stop(self, reason):
        if self.stop_reason is None:
            self.stop_reason = reason
        self.halt.set()

    def _model_tick(self, now):
        with self.lock:
            plan, fresh = self.plan, self.fresh
            self.fresh = False
        if plan is None:          # centred hold until the first chunk after a switch
            self.control.send(self.c.DRIVE, [(0.0,) * 6], flags=self.flags())
            return
        rows, t0, obs = plan
        problem = (f"observation age {(now - obs) / 1e9:.2f} s" if (now - obs) / 1e9 > self.max_age
                   else "chunk exhausted" if now >= t0 + int((len(rows) - 1) / self.fps * 1e9) else None)
        if problem:
            if self.pad is None:
                self.stop(problem)
                return
            self.set_mode("manual", problem)
            self.control.send(self.c.DRIVE, [(0.0,) * 6], flags=self.flags())
            return
        if fresh or now - self.last_send >= self.renew * 1e9:
            self.control.send(self.c.DRIVE, [tuple(map(float, r)) for r in rows],
                              flags=self.flags(), fps=self.fps, t0_ns=t0, observation_ns=obs)
            self.by_sequence[self.control.sequence] = (t0, rows)
            self.sent += 1
            self.last_send = now

    def run(self):
        self.last_send = 0
        period = 0.025 if self.pad else min(self.renew, 0.05)
        try:
            while not self.halt.is_set():
                now = time.monotonic_ns()
                if self.pad:
                    for b in self.pad.poll():
                        if b == BTN_A:
                            self.set_mode("manual" if self.mode == "model" else "model")
                        elif b == BTN_X:
                            self.pump = not self.pump
                            self.note(f"pump {'ON' if self.pump else 'OFF'}")
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
                    i = int(pos)
                    j = min(i + 1, len(rows) - 1)
                    due = rows[i][:4] + (rows[j][:4] - rows[i][:4]) * (pos - i)
                    self.applied.append((round(pos, 2), due.tolist(), st["applied_action"][:4]))
                if st["state"] != "armed":
                    self.stop(f"controller {st['state']}: {st.get('reason')}")
                    break
                self.halt.wait(period)
        except Exception as e:  # noqa: BLE001
            self.stop(f"supervisor error: {e!r}")
        finally:
            try:
                self.control.send(self.c.DISARM)
            finally:
                self.control.close()


def start_services(a, runtime: Path):
    build = Path(a.build)
    control = [build / "kaivuri-control", "--profile", a.profile, "--safety", a.safety,
               "--socket", runtime / "control.sock", "--control-hz", str(a.control_hz),
               "--serial", a.serial]
    if a.live_pwm:
        control.append("--live-pwm")
    if a.enable_slew:
        control.append("--enable-slew")
    cams = [build / "kaivuri-camera", "--camera", f"{CAM1}:2", "--camera", f"{CAM2}:1",
            "--output-dir", runtime / "cameras", "--fps", "30", "--width", "640", "--height", "480",
            "--rotate-180", CAM1]
    procs = []
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
    return procs, runtime / "control.sock", [runtime / "cameras" / f"{CAM1}-ir2.frame",
                                             runtime / "cameras" / f"{CAM2}-ir1.frame"]


def write_summary(out: Path, a, chunks, sup, interrupted, tasks=None) -> dict:
    np.save(out / "chunks.npy", np.stack(chunks) if chunks else np.zeros((0,)))
    summary = {"chunks": len(chunks), "drive": a.drive, "interrupted": interrupted,
               "stop_reason": (sup.stop_reason if sup else None) or ("Ctrl+C" if interrupted else None),
               "plans_sent": sup.sent if sup else 0, "pad_events": sup.events if sup else [],
               "tasks": tasks.items if tasks else None}
    if sup and sup.applied:
        planned = np.array([p for _, p, _ in sup.applied])
        applied = np.array([q for _, _, q in sup.applied])
        summary["applied_vs_planned"] = {
            "samples": len(applied), "mean_abs": float(np.abs(applied - planned).mean()),
            "max_abs": float(np.abs(applied - planned).max()),
            "row_due": {"min": min(k for k, _, _ in sup.applied), "max": max(k for k, _, _ in sup.applied)}}
        json.dump(sup.applied, open(out / "applied.json", "w"))
    recs = [json.loads(l) for l in open(out / "chunks.jsonl")] if (out / "chunks.jsonl").exists() else []
    if recs:
        summary["chunks_per_task"] = {t: sum(r.get("task") == t for r in recs)
                                      for t in dict.fromkeys(r.get("task") for r in recs)}
        lat = [r["obs_to_plan_ms"] for r in recs]
        inf = [r["timings_ms"]["total"] for r in recs]
        summary.update(obs_to_plan_ms={"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99)),
                                       "max": max(lat)},
                       infer_ms={"p50": float(np.percentile(inf, 50)), "p99": float(np.percentile(inf, 99))})
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
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    procs, tmp, sup, pad, tasks = [], None, None, None, None
    chunks, interrupted = [], False
    try:
        pad_path = find_pad(a.gamepad) if a.drive else None
        if pad_path:
            pad = Pad(pad_path)
            print(f"[pad] {pad_path}: A = model/manual, X = pump on/off, Ctrl+C = stop", flush=True)
        elif a.drive:
            print("[pad] no gamepad: model drives until the run ends; Ctrl+C = stop", flush=True)
        if a.start_services:
            tmp = tempfile.TemporaryDirectory(prefix="kvr-groot-")
            procs, sock, cam_files = start_services(a, Path(tmp.name))
        else:
            sock, cam_files = a.control_socket, a.camera_frame
        pol = Policy(Path(a.bundle), contract, Path(a.cache).expanduser(), Path(a.vla_root))
        print(f"[groot] engines ready in {pol.load_s} s, fixture {pol.fixture['status']} "
              f"(chunk cos {pol.fixture['reports']['chunk']['cosine']:.6f})", flush=True)
        if a.task:
            pol.prompts.ids(a.task)                     # refuse an unusable --task before arming
        tasks = TaskList(pol.prompts.tasks, a.task)
        prompt_note = (f"type a prompt + Enter to switch to it (kept in the list, up to "
                       f"{pol.prompts.capacity} tokens); '?' lists the tasks"
                       if pol.prompts.dynamic else "this bundle bakes one prompt")
        print(f"[task] {prompt_note}; D-pad right/left cycles\n{tasks.show()}", flush=True)
        readers = [FrameReader(p, Path(a.build) / "libkaivuri_frames.so", max_age_s=a.max_frame_age_ms / 1e3)
                   for p in cam_files]
        status_conn = client_mod.Client(sock)
        rng = np.random.default_rng(a.seed)
        fps = float(a.fps)
        log = open(out / "chunks.jsonl", "w")

        def observe():
            frames, arrivals = [], []
            for r in readers:
                snap = None
                end = time.monotonic() + 1.0
                while snap is None:
                    snap = r.read()
                    if snap is None:
                        if time.monotonic() > end:
                            raise RuntimeError("no fresh camera frame")
                        time.sleep(0.002)
                m, px = snap
                frames.append(np.frombuffer(px, np.uint8).reshape(m.height, m.width).copy())
                arrivals.append(m.host_monotonic_ns)
            st = status_conn.status()
            now = time.monotonic_ns()
            if not (st["imu_healthy"] and st["fk_valid"]):
                raise RuntimeError("IMU unhealthy / FK invalid")
            if (now - st["imu_host_monotonic_ns"]) / 1e6 > a.max_imu_age_ms:
                raise RuntimeError("IMU sample too old")
            skew = (max(arrivals) - min(arrivals)) / 1e6
            return frames, st, min([*arrivals, st["imu_host_monotonic_ns"]]), skew

        # Warm-up: a few inferences before arming, so the first plan's latency is known.
        for _ in range(a.warmup):
            frames, st, obs_ns, _ = observe()
            pol.infer([gray_to_rgb(f) for f in frames], pol.boundary.degrees(st),
                      rng.standard_normal(pol.noise_shape).astype(np.float32))
        if a.drive:
            sup = Supervisor(client_mod, sock, fps, a.max_observation_age_ms / 1e3, slew=a.enable_slew,
                             pump=a.pump, pad=pad, start_model=a.start_model, tasks=tasks)
            sup.arm()
            sup.start()
            print(f"[groot] armed, {sup.mode.upper()} mode, pump {'ON' if a.pump else 'OFF'}", flush=True)
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
        applied = None
        t_end = time.monotonic() + a.seconds if a.seconds > 0 else float("inf")
        next_start = time.monotonic()
        while time.monotonic() < t_end and not (sup and sup.halt.is_set()):
            if sup and sup.mode != "model":       # manual: no inference, sticks drive
                time.sleep(0.02)
                next_start = time.monotonic()
                continue
            if a.rate_hz > 0:
                time.sleep(max(0.0, next_start - time.monotonic()))
                next_start += 1.0 / a.rate_hz
            k, text, version = tasks.current()
            if version != applied:
                n = pol.set_prompt(text)
                applied = version
                print(f"[task {k + 1}/{len(tasks.items)}] {text!r} ({n} prompt tokens)", flush=True)
            frames, st, obs_ns, skew = observe()
            if not chunks:        # what the model saw first, for an orientation check
                for k, f in enumerate(frames, 1):
                    (out / f"cam{k}_first.pgm").write_bytes(
                        f"P5\n{f.shape[1]} {f.shape[0]}\n255\n".encode() + f.tobytes())
            deg = pol.boundary.degrees(st)
            t = {}
            noise = rng.standard_normal(pol.noise_shape).astype(np.float32)
            act, model = pol.infer([gray_to_rgb(f) for f in frames], deg, noise, t)
            if not np.isfinite(act).all():
                raise RuntimeError("non-finite actions")
            done_ns = time.monotonic_ns()
            rows = np.zeros((len(act), 6), np.float32)
            rows[:, pol.boundary.action_cols] = np.clip(act, -1.0, 1.0)
            if not a.enable_slew:
                rows[:, 0] = 0.0
            # A chunk of an instruction that changed meanwhile is never offered.
            current = tasks.current()[2] == applied
            accepted = (sup.offer(rows, obs_ns + int(a.action_offset_ms * 1e6), obs_ns)
                        if sup and current else (False if sup else None))
            lag_rows = (done_ns - obs_ns) / 1e9 * fps
            rec = {"i": len(chunks), "obs_ns": obs_ns, "done_ns": done_ns,
                   "obs_to_plan_ms": round((done_ns - obs_ns) / 1e6, 1), "stale_rows": round(lag_rows, 1),
                   "camera_skew_ms": round(skew, 1), "state_deg": deg.round(2).tolist(),
                   "row_now": act[min(int(lag_rows), len(act) - 1)].round(3).tolist(),
                   "row0": act[0].round(3).tolist(), "abs_max": np.abs(act).max(0).round(3).tolist(),
                   "timings_ms": {k: round(v, 1) for k, v in t.items()}, "accepted": accepted,
                   "pump": sup.pump if sup else None, "task": pol.task}
            chunks.append(act)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{rec['i']:4d}] obs->plan {rec['obs_to_plan_ms']:6.1f} ms  infer {t['total']:6.1f}  "
                  f"deg {deg.round(1)}  now {np.round(rec['row_now'], 2)}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[groot] Ctrl+C: disarming and stopping services", flush=True)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # let the cleanup finish
        if sup:
            sup.stop(sup.stop_reason or ("Ctrl+C" if interrupted else "run finished"))
            sup.join(timeout=2)
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        if tmp:
            tmp.cleanup()
        if pad:
            pad.close()
        try:
            summary = write_summary(out, a, chunks, sup, interrupted, tasks)
            print(json.dumps({k: v for k, v in summary.items() if k not in ("pad_events", "tasks")}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[groot] summary not written: {e!r}", flush=True)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("parity", "run"):
        s = sub.add_parser(name)
        s.add_argument("--bundle", required=True)
        s.add_argument("--contract", help="normalization contract JSON (default: the bundle's kaivuri.contract)")
        s.add_argument("--cache", default="~/.cache/kaivuri-groot/default")
        s.add_argument("--vla-root", default=str(Path.home() / "jetson-orin-nano-vla"))
        s.add_argument("--out")
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
    s.add_argument("--live-pwm", action="store_true")
    s.add_argument("--enable-slew", action="store_true")
    s.add_argument("--control-socket")
    s.add_argument("--camera-frame", action="append", help="cam1 then cam2 frame files")
    s.add_argument("--drive", action="store_true")
    s.add_argument("--pump", action="store_true", help="start with pump permission (X toggles it on the pad)")
    s.add_argument("--gamepad", default="auto", help="evdev path, 'auto' (first by-id joystick) or 'none'")
    s.add_argument("--start-model", action="store_true", help="with a gamepad, start in model mode, not manual")
    s.add_argument("--task", help="starting prompt (default: the first trained task); added to the list")
    s.add_argument("--seconds", type=float, default=30, help="0 = until Ctrl+C")
    s.add_argument("--rate-hz", type=float, default=0, help="0 = back to back")
    s.add_argument("--fps", type=float, default=30)
    s.add_argument("--warmup", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--action-offset-ms", type=float, default=0)
    s.add_argument("--max-frame-age-ms", type=float, default=200)
    s.add_argument("--max-imu-age-ms", type=float, default=50)
    s.add_argument("--max-observation-age-ms", type=float, default=1400)
    a = p.parse_args()
    if a.cmd == "run" and not a.out:
        p.error("run needs --out")
    if a.cmd == "run" and a.pump and not a.drive:
        p.error("--pump needs --drive")
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
