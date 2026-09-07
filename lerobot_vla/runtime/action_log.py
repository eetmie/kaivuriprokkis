"""JSONL recorder for run_inference: what the policy asked for, what the valves got.

Exists to answer one question with data instead of a guess: how small are the
"small" valve commands, and how often do they happen? A command of magnitude eps
is not free on this machine -- the PWM layer jumps straight to the deadband edge
for any non-zero value (see ``_compute_base_pulse`` in modules/pwm/controller.py),
so +0.001 on ``arm`` is a 160 us step, larger than the 115 us of pulse width that
covers the whole rest of the command range. A command that oscillates around zero
therefore slams the spool across the deadband at the chunk rate while the boom
does not visibly move.

Two streams are recorded, and the difference between them matters:

    "chunk"     one record per inference: the raw policy output, the whole
                action chunk. This is the distribution a threshold would be
                applied to if the cut is made at the policy.
    "emitted"   the setpoint the control thread actually wrote, polled at a
                fixed rate. NOT a subsample of the chunk: the schedule
                interpolates between chunk steps, cross-fades at chunk
                boundaries and decays to zero when a producer stalls, so it
                manufactures small values of its own that no policy-side filter
                would ever see.

Both are cheap enough to leave on: ~500 B per inference plus ~120 B per emitted
sample, so an hour at 8 Hz inference and 50 Hz polling is about 25 MB.

Recording works in dry-run mode too (real observations, no valve output), which
is the safe way to collect the distribution for a new position or checkpoint.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def resolve_log_path(path: str | Path) -> Path:
    """A directory gets a timestamped file inside it; a file path is used as given."""
    p = Path(path).expanduser()
    if p.is_dir():
        return p / f"infer_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class ActionLogger:
    """Append-only JSONL writer. Thread-safe; one line per record.

    Records carry ``t``, seconds since the logger opened, on the same
    ``time.monotonic`` clock the control thread uses, so chunk and emitted
    records line up on one timeline.
    """

    #: Values are rounded before writing. 4 decimals is well under the
    #: resolution any deadband threshold will ever care about and roughly
    #: halves the file.
    NDIGITS = 4

    def __init__(self, path: str | Path, channels: Sequence[str],
                 meta: dict | None = None) -> None:
        self.path = resolve_log_path(path)
        self.channels = list(channels)
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._fh = self.path.open("w", buffering=1 << 16)
        self._sampler: threading.Thread | None = None
        self._stop = threading.Event()

        self._write({
            "k": "meta",
            "wall_start": datetime.now().isoformat(timespec="seconds"),
            "channels": self.channels,
            **(meta or {}),
        })

    # ── producer side ────────────────────────────────────────────────────────

    def log_chunk(self, cycle: int, infer_s: float, played: int | None,
                  state: np.ndarray, chunk: np.ndarray) -> None:
        """One inference: the observation that went in and the whole chunk out."""
        self._write({
            "k": "chunk",
            "t": round(time.monotonic() - self._t0, 4),
            "cycle": int(cycle),
            "infer_ms": round(infer_s * 1000.0, 1),
            "played": None if played is None else int(played),
            "state": [round(float(v), 3) for v in np.asarray(state).ravel()],
            "a": [[round(float(v), self.NDIGITS) for v in step]
                  for step in np.asarray(chunk)],
        })

    def log_event(self, name: str, **fields) -> None:
        """Mark something that breaks the timeline: teleop takeover, mode change."""
        self._write({"k": "event", "t": round(time.monotonic() - self._t0, 4),
                     "name": name, **fields})

    # ── emitted-setpoint sampler ─────────────────────────────────────────────

    def start_emitted_sampler(self, status_fn: Callable[[], dict], hz: float) -> None:
        """Poll ``status_fn`` (robot.get_setpoint_status) at ``hz`` in the background.

        Polling rather than hooking the control thread: this must not be able to
        slow the 100 Hz valve writer down, and for a magnitude histogram an
        occasional duplicated or missed sample changes nothing.
        """
        if hz <= 0 or self._sampler is not None:
            return
        period = 1.0 / float(hz)

        def run() -> None:
            while not self._stop.wait(period):
                try:
                    st = status_fn()
                except Exception:
                    continue
                cmds = st.get("commands") or {}
                if not cmds:
                    continue
                self._write({
                    "k": "emitted",
                    "t": round(time.monotonic() - self._t0, 4),
                    "c": [round(float(cmds.get(ch, 0.0)), self.NDIGITS)
                          for ch in self.channels],
                    "decay": round(float(st.get("decay", 1.0)), 3),
                    "pos": (None if st.get("chunk_pos") is None
                            else round(float(st["chunk_pos"]), 2)),
                })

        self._sampler = threading.Thread(target=run, name="action-log-sampler",
                                         daemon=True)
        self._sampler.start()

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _write(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            if not self._fh.closed:
                self._fh.write(line + "\n")

    def close(self) -> None:
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    def __enter__(self) -> "ActionLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
