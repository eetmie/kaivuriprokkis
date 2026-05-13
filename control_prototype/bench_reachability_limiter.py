"""Performance baseline for the smooth reachability prototype.

Measures three levels of cost so you can budget against the real-time loop:

  A) Limiter math only -- compute_scale + apply_scale on synthetic signals.
     This is what runs per UDP packet in the prototype server.

  B) Limiter + ONE Jacobian recompute via the real numba IK kernel.
     Closer to the realistic per-tick cost when yoshikawa/cond come from a
     fresh Jacobian evaluation rather than the last cached IK solve.

  C) Limiter + FIVE Jacobian recomputes (1 nominal + 4 axis probes).
     Projected cost of the directional-scaling TODO from
     relative_command_processor.py.

Run:
    python control_prototype/bench_reachability_limiter.py

Prints kops/sec and median microseconds per iter for each bench.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for p in (str(_HERE), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from reachability_limiter import LimiterConfig, apply_scale, compute_scale
from relative_command_processor import ProcessorConfig, RelativeCommandProcessor
from excavator_fixture import REAL_JOINT_LIMITS_RAD, SAFE_ANGLES_RAD


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

def _run_bench(name: str, fn: Callable[[], None], warmup: int = 200, target_seconds: float = 1.0) -> dict:
    """Run `fn` repeatedly. Reports kops/sec and per-iter median microseconds."""
    for _ in range(warmup):
        fn()

    iters = 0
    samples = []
    t_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
        iters += 1
        if (t1 - t_start) >= target_seconds:
            break

    elapsed = time.perf_counter() - t_start
    ops_per_s = iters / elapsed if elapsed > 0 else float("inf")
    arr = np.asarray(samples) * 1e6  # us
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(f"  {name:42s}  {ops_per_s/1000:9.1f} kops/s   "
          f"p50={p50:7.3f}us  p95={p95:7.3f}us  p99={p99:7.3f}us  (n={iters})")
    return {"kops_per_s": ops_per_s / 1000.0, "p50_us": float(p50),
            "p95_us": float(p95), "p99_us": float(p99), "iters": iters}


# --------------------------------------------------------------------------
# Bench A: limiter math only (no IK)
# --------------------------------------------------------------------------

def _bench_compute_scale_only() -> Callable[[], None]:
    cfg = LimiterConfig()
    angles = SAFE_ANGLES_RAD.copy()
    limits = REAL_JOINT_LIMITS_RAD
    state = {"y": 0.5, "c": 10.0}

    def step():
        compute_scale(
            joint_angles_rad=angles,
            joint_limits_rad=limits,
            yoshikawa=state["y"],
            cond=state["c"],
            cfg=cfg,
        )
    return step


def _bench_processor_step() -> Callable[[], None]:
    proc = RelativeCommandProcessor(ProcessorConfig(limiter=LimiterConfig()))
    proc.sync_to_measured([0.5, 0.0, 0.0], 0.0)
    angles = SAFE_ANGLES_RAD.copy()
    limits = REAL_JOINT_LIMITS_RAD

    def step():
        proc.step(
            [0.1, 0.0, 0.0], 0.0, 0.01,
            joint_angles_rad=angles,
            joint_limits_rad=limits,
            yoshikawa=0.5,
            cond=10.0,
        )
    return step


# --------------------------------------------------------------------------
# Bench B and C need the live Jacobian kernel
# --------------------------------------------------------------------------

def _build_ik_signals_fn() -> Tuple[Callable[[np.ndarray], Tuple[float, float]], "RobotConfig"]:
    """Return a (signals_fn, robot_config) using the real IK kernel.

    signals_fn(angles_rad) -> (yoshikawa, cond) for a 5x4 reduced Jacobian
    (x, y, z, pitch, yaw) -- matches what the live IK exposes.
    """
    from modules.differential_ik import (
        compute_jacobian_from_joint_angles,
        compute_jacobian_metrics,
    )
    from excavator_fixture import load_real_robot_config

    robot_config = load_real_robot_config()

    def signals(angles_rad: np.ndarray) -> Tuple[float, float]:
        J = compute_jacobian_from_joint_angles(angles_rad, robot_config)
        # 5x4 reduced: [x, y, z, pitch_y, yaw_z]. Drop roll (row 3) and we keep
        # the rest -- matches the live IK reduced Jacobian for this excavator
        # (see modules/differential_ik.py:_get_reduced_jacobian).
        J_reduced = np.vstack((J[0:3, :], J[4:6, :])).astype(np.float32)
        cond, _sv, yosh = compute_jacobian_metrics(J_reduced)
        return float(yosh), float(cond)

    return signals, robot_config


def _bench_with_one_jacobian(signals) -> Callable[[], None]:
    cfg = LimiterConfig()
    limits = REAL_JOINT_LIMITS_RAD
    angles = SAFE_ANGLES_RAD.copy()

    def step():
        y, c = signals(angles)
        compute_scale(
            joint_angles_rad=angles,
            joint_limits_rad=limits,
            yoshikawa=y,
            cond=c,
            cfg=cfg,
        )
    return step


def _bench_with_directional_probes(signals) -> Callable[[], None]:
    """Project the cost of the directional-scaling TODO.

    Per tick: 1 nominal Jacobian + 4 perturbed Jacobians (one per joint),
    each feeding the limiter. The limiter math is negligible compared to
    the 5 numba calls, so this is dominated by IK kernel cost.
    """
    cfg = LimiterConfig()
    limits = REAL_JOINT_LIMITS_RAD
    angles = SAFE_ANGLES_RAD.copy()
    eps = math.radians(0.5)

    def step():
        y0, c0 = signals(angles)
        compute_scale(
            joint_angles_rad=angles,
            joint_limits_rad=limits,
            yoshikawa=y0, cond=c0, cfg=cfg,
        )
        for j in range(4):
            probe = angles.copy()
            probe[j] += eps
            signals(probe)
    return step


# --------------------------------------------------------------------------
# Loop-rate translation
# --------------------------------------------------------------------------

def _print_loop_budget(kops_per_s: float, label: str) -> None:
    print(f"     -> {label}: "
          f"{kops_per_s*1000/50.0:>9.0f} iters per 50 Hz tick,  "
          f"{kops_per_s*1000/200.0:>9.0f} per 200 Hz tick,  "
          f"{kops_per_s*1000/1000.0:>9.0f} per 1 kHz tick")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> int:
    print("Smooth reachability limiter -- performance baseline")
    print(f"  python : {sys.version.split()[0]}")
    print(f"  numpy  : {np.__version__}")
    print()
    print("Bench A  Limiter math only (no IK kernel call)")
    a1 = _run_bench("compute_scale()", _bench_compute_scale_only())
    a2 = _run_bench("RelativeCommandProcessor.step()", _bench_processor_step())
    _print_loop_budget(a1["kops_per_s"], "compute_scale headroom")

    try:
        signals, _rc = _build_ik_signals_fn()
    except Exception as exc:
        print()
        print(f"  Skipping Bench B/C (real IK kernel unavailable): {exc!r}")
        return 0

    # Warm numba JIT before measuring.
    print()
    print("  Warming numba JIT (first-call compile cost is one-time)...")
    t0 = time.perf_counter()
    for _ in range(3):
        signals(SAFE_ANGLES_RAD)
    print(f"  JIT warmup: {(time.perf_counter() - t0) * 1000:.1f} ms wall")

    print()
    print("Bench B  Limiter + 1 live Jacobian per tick (realistic per-tick cost)")
    b = _run_bench("compute_scale() + 1x Jacobian+metrics", _bench_with_one_jacobian(signals))
    _print_loop_budget(b["kops_per_s"], "headroom for 1-Jacobian path")

    print()
    print("Bench C  Limiter + 5 Jacobians per tick (projected directional-scaling cost)")
    c = _run_bench("compute_scale() + 5x Jacobian+metrics", _bench_with_directional_probes(signals))
    _print_loop_budget(c["kops_per_s"], "headroom for 5-Jacobian path")

    print()
    print("Summary")
    print(f"  Per-tick budget @ 100 Hz = 10000 us")
    print(f"  Per-tick budget @ 200 Hz =  5000 us")
    print(f"  Bench A p99 = {a1['p99_us']:.2f} us  ({a1['p99_us']/5000*100:.2f}% of 200 Hz tick)")
    print(f"  Bench B p99 = {b['p99_us']:.2f} us  ({b['p99_us']/5000*100:.2f}% of 200 Hz tick)")
    print(f"  Bench C p99 = {c['p99_us']:.2f} us  ({c['p99_us']/5000*100:.2f}% of 200 Hz tick)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
