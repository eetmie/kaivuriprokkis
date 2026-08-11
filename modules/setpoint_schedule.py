"""Time-indexed valve setpoint holder.

Bridges a slow, irregular command producer (a VLA policy emitting action
chunks every few hundred ms, a gamepad at 30 Hz, a UDP client at whatever the
network gives) to the fixed-rate control thread that writes the valves.

Why this exists: the PWM layer has no clock of its own. Dither phase, ramp
integration, the stale-command watchdog and the input-rate gate in
``modules/pwm/controller.py`` all advance *only* when ``update_named()`` is
called, so the valve output is a pure function of the caller's call rate. The
writer therefore has to tick at a fixed rate whether or not a new setpoint
arrived -- and this object decides what it writes in between.

Two producer shapes:

    set_point(cmds)          zero-order hold on a single setpoint
    set_chunk(chunk, fps)    a trajectory sampled at its own nominal fps,
                             interpolated up to the writer's rate

Staleness lives here rather than in the PWM layer. Age is measured from the
moment the schedule ran out of *fresh* data -- the end of a chunk, or the last
``set_point`` call -- not from the last hardware write, which is the
distinction the old code got wrong. Past ``hold_timeout_s`` the command ramps
to zero over ``decay_s`` instead of stepping there, so a producer that dies
mid-stroke gets a controlled ramp-down rather than a bang into the valve
deadband.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SetpointSample:
    """One evaluation of the schedule at a point in time."""

    commands: Dict[str, float]
    #: Seconds since the schedule last had fresh data (0.0 while inside a chunk).
    age_s: float
    #: 1.0 = fresh, 0.0 = fully decayed to zero. Fades over ``decay_s``.
    decay: float
    #: True when there is no data at all, or a chunk has played past its end.
    exhausted: bool
    #: Fractional index into the active chunk; None in point mode.
    chunk_pos: Optional[float] = None


@dataclass
class _Chunk:
    values: np.ndarray          # (N, J) float32
    names: Sequence[str]
    fps: float
    t0: float

    @property
    def duration_s(self) -> float:
        return max(0.0, (len(self.values) - 1) / self.fps)


class SetpointSchedule:
    """Holds the latest setpoint (or trajectory) and samples it at any instant.

    Thread-safe: the producer calls ``set_point``/``set_chunk``, the control
    thread calls ``sample``.

    Args:
        joint_names: Channel names this schedule emits, in the column order
            used by chunk arrays that do not carry their own names.
        hold_timeout_s: How long a setpoint stays at full authority after it
            goes stale. Set this above the producer's nominal period with
            margin -- a 30 Hz producer needs > 0.033 s.
        decay_s: Ramp-to-zero window once ``hold_timeout_s`` is exceeded.
            0.0 makes staleness an immediate cut to zero.
        blend_s: Optional cross-fade from the last emitted command into a
            newly-arrived setpoint/chunk, which softens the step at a chunk
            boundary. 0.0 (default) applies the new command immediately.
    """

    def __init__(self, joint_names: Sequence[str], *,
                 hold_timeout_s: float = 0.25,
                 decay_s: float = 0.25,
                 blend_s: float = 0.0) -> None:
        self.joint_names = list(joint_names)
        self.hold_timeout_s = float(hold_timeout_s)
        self.decay_s = max(0.0, float(decay_s))
        self.blend_s = max(0.0, float(blend_s))

        self._lock = threading.Lock()
        self._point: Optional[Dict[str, float]] = None
        self._point_t: float = 0.0
        self._chunk: Optional[_Chunk] = None
        # Last command actually emitted, for blending across a new setpoint.
        self._last_emitted: Dict[str, float] = {n: 0.0 for n in self.joint_names}
        self._blend_from: Optional[Dict[str, float]] = None
        self._blend_t0: float = 0.0

    # ── producer side ────────────────────────────────────────────────────────

    def set_point(self, commands: Dict[str, float],
                  t_now: Optional[float] = None) -> None:
        """Zero-order-hold this setpoint from ``t_now`` onward."""
        t_now = time.monotonic() if t_now is None else float(t_now)
        clean = {str(k): float(np.clip(float(v), -1.0, 1.0))
                 for k, v in commands.items()}
        with self._lock:
            self._point = clean
            self._point_t = t_now
            self._chunk = None
            self._arm_blend(t_now)

    def set_chunk(self, chunk, fps: float,
                  t_now: Optional[float] = None,
                  joint_names: Optional[Sequence[str]] = None) -> None:
        """Play ``chunk`` as a trajectory at its own nominal ``fps``.

        Args:
            chunk: (N, J) array of normalized valve commands.
            fps: Rate the chunk's steps were authored at -- the policy's
                action rate, not the control thread's rate.
            t_now: Playback start time (monotonic clock).
            joint_names: Column names; defaults to this schedule's names.
        """
        values = np.asarray(chunk, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"chunk must be 2-D (N, J), got shape {values.shape}")
        if len(values) == 0:
            raise ValueError("chunk is empty")
        names = list(joint_names) if joint_names is not None else list(self.joint_names)
        if values.shape[1] != len(names):
            raise ValueError(
                f"chunk has {values.shape[1]} columns but {len(names)} joint "
                f"names were given ({names})")
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")

        values = np.clip(values, -1.0, 1.0)
        t_now = time.monotonic() if t_now is None else float(t_now)
        with self._lock:
            self._chunk = _Chunk(values=values, names=names, fps=float(fps), t0=t_now)
            self._point = None
            self._arm_blend(t_now)

    def clear(self) -> None:
        """Drop all data. ``sample`` then returns zeros, marked exhausted."""
        with self._lock:
            self._point = None
            self._chunk = None
            self._blend_from = None

    def has_data(self) -> bool:
        with self._lock:
            return self._point is not None or self._chunk is not None

    def _arm_blend(self, t_now: float) -> None:
        """Start a cross-fade out of the last emitted command. Caller holds lock."""
        if self.blend_s > 0.0:
            self._blend_from = dict(self._last_emitted)
            self._blend_t0 = t_now

    # ── consumer side ────────────────────────────────────────────────────────

    def sample(self, t_now: Optional[float] = None) -> SetpointSample:
        """Evaluate the schedule. Safe to call at any rate, from one thread."""
        t_now = time.monotonic() if t_now is None else float(t_now)

        with self._lock:
            point, point_t = self._point, self._point_t
            chunk = self._chunk
            blend_from, blend_t0 = self._blend_from, self._blend_t0

        if point is None and chunk is None:
            zeros = {n: 0.0 for n in self.joint_names}
            self._remember(zeros)
            return SetpointSample(commands=zeros, age_s=float('inf'),
                                  decay=0.0, exhausted=True)

        if chunk is not None:
            raw, fresh_until, exhausted, chunk_pos = self._sample_chunk(chunk, t_now)
        else:
            raw = dict(point)
            fresh_until = point_t
            exhausted = False
            chunk_pos = None

        # Always emit every channel this schedule owns. A partial dict would
        # leave the omitted channels to the PWM layer's unset_to_zero default,
        # which is true today but is not something the setpoint layer should
        # depend on -- with it off, omitted channels would latch their last value.
        full = {n: 0.0 for n in self.joint_names}
        full.update(raw)
        raw = full

        age_s = max(0.0, t_now - fresh_until)
        decay = self._decay_factor(age_s)

        if blend_from is not None and self.blend_s > 0.0:
            w = min(1.0, max(0.0, (t_now - blend_t0) / self.blend_s))
            if w < 1.0:
                raw = {k: blend_from.get(k, 0.0) * (1.0 - w) + v * w
                       for k, v in raw.items()}
            else:
                with self._lock:
                    if self._blend_t0 == blend_t0:
                        self._blend_from = None

        commands = {k: float(v * decay) for k, v in raw.items()}
        self._remember(commands)
        return SetpointSample(commands=commands, age_s=age_s, decay=decay,
                              exhausted=exhausted, chunk_pos=chunk_pos)

    def _sample_chunk(self, chunk: _Chunk, t_now: float):
        """Linear interpolation between adjacent chunk steps.

        Valve commands are rates, so interpolating between steps is the honest
        reading of what a policy authored at ``fps`` meant in between them --
        and it removes the staircase a naive replay loop produces.
        """
        n = len(chunk.values)
        u = (t_now - chunk.t0) * chunk.fps

        if u <= 0.0:
            values = chunk.values[0]
            return (dict(zip(chunk.names, (float(v) for v in values))),
                    t_now, False, 0.0)

        idx = int(u)
        if idx >= n - 1:
            values = chunk.values[-1]
            fresh_until = chunk.t0 + chunk.duration_s
            return (dict(zip(chunk.names, (float(v) for v in values))),
                    fresh_until, True, float(n - 1))

        frac = u - idx
        values = chunk.values[idx] * (1.0 - frac) + chunk.values[idx + 1] * frac
        return (dict(zip(chunk.names, (float(v) for v in values))),
                t_now, False, float(u))

    def _decay_factor(self, age_s: float) -> float:
        if age_s <= self.hold_timeout_s:
            return 1.0
        if self.decay_s <= 0.0:
            return 0.0
        over = age_s - self.hold_timeout_s
        return float(max(0.0, 1.0 - over / self.decay_s))

    def _remember(self, commands: Dict[str, float]) -> None:
        self._last_emitted = dict(commands)
