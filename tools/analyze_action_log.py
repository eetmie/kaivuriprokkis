#!/usr/bin/env python3
"""Pick a valve deadzone threshold from run_inference --log-actions data.

The problem this sizes up: modules/pwm/controller.py maps a normalized command
to a pulse by jumping straight to the deadband edge for ANY non-zero value, then
scaling the remainder over what is left of the pulse range. On the jetson profile
that first step is most of the channel:

    channel   center   +edge jump   working range above the edge
    boom      1655 us     130 us          195 us
    arm       1575 us     160 us          115 us   <- the jump is 1.4x the range
    bucket    1605 us     160 us          115 us
    slew    1637.5 us      44 us         33.5 us

So a command of +-0.01 is not a small command. It is a full deadband step, and a
command that changes sign around zero drags the spool ~280 us back and forth
while the boom does not move. This tool measures how much of that is happening
and what a threshold would remove.

Two numbers do the work:

    spool travel (us/s)   total |delta pulse| per second, summed over the
                          emitted stream. The direct stress proxy.
    crossings/s           how often the command's sign changes through zero,
                          i.e. how often the spool is dragged across the whole
                          deadband. Each one is a full traverse.

Both are reported for the raw log and for a sweep of candidate gates, so the
threshold is chosen from the curve rather than guessed. A gate is scored two
ways: a plain symmetric deadzone (what modules/pwm ChannelConfig.deadzone
already implements, with rescaling so the top of the range is not lost), and a
Schmitt-trigger version that needs |v| > on_ratio*t to re-open once closed --
which is what actually kills chatter from a command hovering at the threshold.

Usage:
    python3 tools/analyze_action_log.py ~/logs/infer_20260827_101500.jsonl
    python3 tools/analyze_action_log.py LOG --thresholds 0.01,0.02,0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)

#: Fraction of the working range the deadband step is worth, per channel, is
#: read from here when it exists; otherwise the report skips the us columns.
DEFAULT_SERVO_CONFIG = _ROOT / "configuration_files/profiles/jetson/servo_config.yaml"


# ── log loading ──────────────────────────────────────────────────────────────

def load(path: Path):
    meta, chunks, emitted, events = {}, [], [], []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            k = rec.get("k")
            if k == "meta":
                meta = rec
            elif k == "chunk":
                chunks.append(rec)
            elif k == "emitted":
                emitted.append(rec)
            elif k == "event":
                events.append(rec)
    return meta, chunks, emitted, events


def chunk_matrix(chunks) -> np.ndarray:
    """All chunk steps ever emitted by the policy, stacked (M, J).

    Every step of every chunk, not just the ones that played: this is the
    distribution the policy produces, which is what a policy-side filter would
    act on. The emitted stream below is what the valves actually saw.
    """
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate([np.asarray(c["a"], dtype=np.float32) for c in chunks])


def chunk_head(chunks) -> np.ndarray:
    """Only step 0 of each chunk -- the action the policy is most committed to."""
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([np.asarray(c["a"], dtype=np.float32)[0] for c in chunks])


def emitted_series(emitted):
    """(t, values) for the setpoint the control thread wrote."""
    if not emitted:
        return np.zeros(0), np.zeros((0, 0), dtype=np.float32)
    t = np.asarray([r["t"] for r in emitted], dtype=np.float64)
    v = np.asarray([r["c"] for r in emitted], dtype=np.float32)
    return t, v


# ── valve model (mirrors modules/pwm/controller.py _compute_base_pulse) ──────

def load_channel_geometry(path: Path) -> dict:
    """center / edges / pulse limits per channel, or {} if the config is unreadable.

    Dither and ramp are deliberately left out: dither is a fixed-amplitude
    sinusoid on top of the base pulse and ramp is off on every channel in the
    shipped profiles, so neither changes the comparison between gates.
    """
    try:
        import yaml
    except ImportError:
        return {}
    try:
        cfg = yaml.safe_load(path.read_text()).get("CHANNEL_CONFIGS", {})
    except (OSError, yaml.YAMLError):
        return {}

    geo = {}
    for name, c in cfg.items():
        try:
            pmin, pmax = float(c["pulse_min"]), float(c["pulse_max"])
        except (KeyError, TypeError, ValueError):
            continue
        center = c.get("center")
        # `center: None` in the YAML is the literal string "None" -> midpoint.
        center = (pmin + (pmax - pmin) / 2.0
                  if center in (None, "None", "none") else float(center))
        geo[name] = {
            "center": center, "pulse_min": pmin, "pulse_max": pmax,
            "direction": float(c.get("direction", 1)),
            "db_pos": float(c.get("deadband_us_pos", 0.0)),
            "db_neg": float(c.get("deadband_us_neg", 0.0)),
            "dither": bool(c.get("dither_enable", False)),
            "dither_amp_us": float(c.get("dither_amp_us", 0.0)),
            "dither_hz": float(c.get("dither_hz", 0.0)),
        }
    return geo


def pulse_us(values: np.ndarray, g: dict) -> np.ndarray:
    """Normalized command -> base pulse width, vectorized."""
    v = np.asarray(values, dtype=np.float64)
    s = v * g["direction"]
    pos_base = g["center"] + g["db_pos"]
    neg_base = g["center"] - g["db_neg"]
    out = np.full(v.shape, g["center"], dtype=np.float64)
    up = s > 0.0
    dn = s < 0.0
    out[up] = pos_base + np.abs(v[up]) * (g["pulse_max"] - pos_base)
    out[dn] = neg_base - np.abs(v[dn]) * (neg_base - g["pulse_min"])
    return out


# ── gates ────────────────────────────────────────────────────────────────────

def gate_plain(v: np.ndarray, t: float) -> np.ndarray:
    """Symmetric deadzone with rescaling -- ChannelConfig.deadzone's behaviour."""
    if t <= 0.0:
        return v
    out = np.zeros_like(v)
    live = np.abs(v) >= t
    out[live] = np.sign(v[live]) * (np.abs(v[live]) - t) / (1.0 - t)
    return out


def gate_hysteresis(v: np.ndarray, t_off: float, on_ratio: float) -> np.ndarray:
    """Schmitt trigger: open above ``on_ratio*t_off``, close below ``t_off``.

    Sequential by construction -- the whole point is that the decision depends
    on whether the channel was already moving. Applied per sample in the order
    they were emitted.
    """
    if t_off <= 0.0:
        return v
    t_on = t_off * on_ratio
    out = np.zeros_like(v)
    open_ = False
    for i, x in enumerate(v):
        a = abs(x)
        if open_:
            open_ = a >= t_off
        else:
            open_ = a >= t_on
        if open_:
            out[i] = np.sign(x) * max(0.0, (a - t_off)) / (1.0 - t_off)
    return out


# ── metrics ──────────────────────────────────────────────────────────────────

def crossings_per_s(v: np.ndarray, duration_s: float) -> float:
    """Sign changes through zero: each one is a full deadband traverse."""
    if duration_s <= 0:
        return float("nan")
    sign = np.sign(v)
    nz = sign[sign != 0]
    if nz.size < 2:
        return 0.0
    return float(np.count_nonzero(np.diff(nz) != 0) / duration_s)


def spool_travel_us_per_s(v: np.ndarray, g: dict, duration_s: float) -> float:
    """Travel of the base pulse alone -- command changes, no dither."""
    if not g or duration_s <= 0 or v.size < 2:
        return float("nan")
    p = pulse_us(v, g)
    return float(np.abs(np.diff(p)).sum() / duration_s)


def dither_travel_us_per_s(v: np.ndarray, g: dict) -> float:
    """Travel the dither adds, which on this machine is the larger term.

    ``dither_active`` in the controller is just ``base_pulse != center``, so
    dither runs whenever the command is non-zero -- however small. One cycle of
    a +-amp sinusoid is 4*amp of travel, at dither_hz. Estimated from config
    rather than measured: the log records commands, not pulses. Taper is off on
    every channel in the shipped profiles, so amplitude is constant.
    """
    if not g or not g.get("dither") or v.size == 0:
        return float("nan") if not g else 0.0
    amp, hz = g["dither_amp_us"], g["dither_hz"]
    if amp == 0.0 or hz == 0.0:
        return 0.0
    active = float(np.count_nonzero(v != 0.0)) / v.size
    return 4.0 * amp * hz * active


def fmt(x: float, nd: int = 1) -> str:
    return "  n/a" if x != x else f"{x:.{nd}f}"


# ── report ───────────────────────────────────────────────────────────────────

def report_distribution(name: str, v: np.ndarray, g: dict) -> None:
    a = np.abs(v)
    nz = a[a > 0]
    print(f"\n  {name}")
    print(f"    samples={a.size}  zero={100.0 * (a.size - nz.size) / max(a.size, 1):.1f}%"
          f"  max={a.max() if a.size else 0:.3f}")
    if nz.size:
        qs = np.percentile(nz, [5, 25, 50, 75, 95])
        print("    non-zero |v| percentiles  p5=%.4f p25=%.4f p50=%.4f p75=%.4f p95=%.4f"
              % tuple(qs))
    edges = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4, 1.01]
    hist, _ = np.histogram(a, bins=edges)
    for lo, hi, n in zip(edges[:-1], edges[1:], hist):
        if n == 0:
            continue
        share = 100.0 * n / a.size
        bar = "#" * int(round(share / 2))
        print(f"    [{lo:5.3f},{hi:5.3f})  {share:5.1f}%  {bar}")
    if g:
        # What the smallest non-zero command already costs in spool travel.
        step = g["db_pos"]
        span = g["pulse_max"] - (g["center"] + g["db_pos"])
        print(f"    deadband step {step:.0f} us vs {span:.0f} us of working range "
              f"({step / max(span, 1e-9):.2f}x)")


def report_sweep(name: str, t: np.ndarray, v: np.ndarray, g: dict,
                 thresholds, on_ratio: float) -> None:
    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
    print(f"\n  {name}   ({duration:.1f}s of emitted setpoint)")
    total = np.abs(v).sum()

    def row(label, gated):
        zeroed = 100.0 * np.count_nonzero(gated == 0) / max(v.size, 1)
        cmd = spool_travel_us_per_s(gated, g, duration)
        dit = dither_travel_us_per_s(gated, g)
        tot = cmd + dit if cmd == cmd and dit == dit else float("nan")
        lost = (100.0 * (total - np.abs(gated).sum()) / total) if total else 0.0
        print(f"    {label:>6}  {zeroed:8.1f}  {lost:7.2f}  "
              f"{fmt(crossings_per_s(gated, duration), 2):>8}  "
              f"{fmt(cmd, 0):>8}  {fmt(dit, 0):>8}  {fmt(tot, 0):>8}")

    print(f"    {'gate':>6}  {'zeroed%':>8}  {'lost%':>7}  {'cross/s':>8}  "
          f"{'cmd us/s':>8}  {'dith us/s':>8}  {'tot us/s':>8}")
    row(f"{0.0:.3f}", v)
    for th in thresholds:
        row(f"{th:.3f}", gate_plain(v, th))
    print(f"    -- same, with hysteresis (re-open at {on_ratio:g}x) --")
    for th in thresholds:
        row(f"{th:.3f}", gate_hysteresis(v, th, on_ratio))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", type=Path, help="JSONL from run_inference --log-actions")
    p.add_argument("--thresholds", default=None,
                   help="Comma-separated gate thresholds to sweep "
                        f"(default {','.join(str(t) for t in DEFAULT_THRESHOLDS)})")
    p.add_argument("--on-ratio", type=float, default=1.5,
                   help="Hysteresis: re-open at on_ratio * threshold (default 1.5)")
    p.add_argument("--servo-config", type=Path, default=DEFAULT_SERVO_CONFIG,
                   help="Profile the us columns are computed from")
    args = p.parse_args()

    thresholds = (DEFAULT_THRESHOLDS if args.thresholds is None
                  else tuple(float(x) for x in args.thresholds.split(",")))

    meta, chunks, emitted, events = load(args.log)
    channels = meta.get("channels") or ["ch%d" % i for i in range(4)]
    geo = load_channel_geometry(args.servo_config)
    if not geo:
        print(f"note: {args.servo_config} unreadable (pyyaml missing?) — "
              f"us columns will be n/a", file=sys.stderr)

    print(f"log      {args.log}")
    print(f"bundle   {meta.get('split_dir')}")
    print(f"task     {meta.get('task')!r}")
    print(f"mode     {'live' if meta.get('live') else 'dry-run'}"
          f"{' synthetic' if meta.get('synthetic') else ''}"
          f"   fps={meta.get('fps')}  blend_s={meta.get('blend_s')}")
    print(f"records  {len(chunks)} inferences, {len(emitted)} emitted samples, "
          f"{len(events)} events")

    head = chunk_head(chunks)
    allsteps = chunk_matrix(chunks)
    if head.size:
        print("\n=== policy output, step 0 of each chunk ===")
        for j, ch in enumerate(channels[:head.shape[1]]):
            report_distribution(ch, head[:, j], geo.get(ch, {}))
        print("\n=== policy output, all chunk steps ===")
        for j, ch in enumerate(channels[:allsteps.shape[1]]):
            report_distribution(ch, allsteps[:, j], geo.get(ch, {}))

    t, v = emitted_series(emitted)
    if v.size:
        print("\n=== emitted setpoint (what the valves saw) ===")
        for j, ch in enumerate(channels[:v.shape[1]]):
            report_distribution(ch, v[:, j], geo.get(ch, {}))
        print("\n=== gate sweep on the emitted setpoint ===")
        print("    zeroed%   share of samples the gate cuts to zero")
        print("    lost%     sum|v| given up: the zeroed samples PLUS the ~t")
        print("              shrink rescaling puts on every surviving command")
        print("    cross/s   sign changes through zero = full deadband traverses")
        print("    dith us/s dither runs whenever the command is non-zero, so")
        print("              gating at rest is what silences it (config estimate)")
        print("    Lower cross/s and tot us/s is less valve stress. Pick the knee.")
        for j, ch in enumerate(channels[:v.shape[1]]):
            report_sweep(ch, t, v[:, j], geo.get(ch, {}), thresholds, args.on_ratio)
    else:
        print("\nNo emitted samples in this log (synthetic run, or "
              "--log-emitted-hz 0). The gate sweep needs them: the schedule's "
              "interpolation and decay are where most small values come from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
