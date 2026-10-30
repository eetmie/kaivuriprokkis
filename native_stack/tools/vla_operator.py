"""Operator side of vla_live.py, shared by every model: Xbox pad, task list and the supervisor
thread that owns the drive connection."""
from __future__ import annotations

import collections
import fcntl
import os
import stat
import struct
import threading
import time
from pathlib import Path

import numpy as np

JOINTS = {"slew": 0, "lift": 1, "tilt": 2, "scoop": 3}  # controller angles_rad / plan columns
EV_SYN, EV_KEY, EV_ABS, SYN_DROPPED = 0, 1, 3, 3
ABS_X, ABS_Y, ABS_RX, ABS_RY, ABS_HAT0X = 0, 1, 3, 4, 0x10
BTN_A, BTN_X = 0x130, 0x133   # BTN_SOUTH, BTN_NORTH (the recorder's X)
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
    prompt typed in the terminal, appended so switching away never loses them. `same` says when
    two texts are one instruction to the model (GR00T: its processor's formalize)."""

    def __init__(self, tasks: list[str], start: str | None, same=lambda a, b: a == b):
        self.lock = threading.Lock()
        self.same = same
        self.items, self.i, self.version = list(tasks), 0, 0
        if start:
            self.add(start)

    def add(self, text: str) -> int:
        with self.lock:
            same = [k for k, t in enumerate(self.items) if self.same(t, text)]
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
