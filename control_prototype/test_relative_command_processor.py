"""Dry-run tests for RelativeCommandProcessor. Pure numpy, no hardware.

Run directly:   python control_prototype/test_relative_command_processor.py
Or via pytest:  pytest control_prototype/test_relative_command_processor.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from reachability_limiter import LimiterConfig
from relative_command_processor import (
    ProcessorConfig,
    RelativeCommandProcessor,
)
from excavator_fixture import REAL_JOINT_LIMITS_RAD, SAFE_ANGLES_RAD


JOINT_LIMITS_RAD = REAL_JOINT_LIMITS_RAD
SAFE_SIGNALS = dict(
    joint_angles_rad=SAFE_ANGLES_RAD,
    joint_limits_rad=JOINT_LIMITS_RAD,
    yoshikawa=0.5,
    cond=10.0,
)


def _approx(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_step_without_sync_raises():
    proc = RelativeCommandProcessor()
    try:
        proc.step([0.0, 0.0, 0.0], 0.0, 0.01, **SAFE_SIGNALS)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when step() is called before sync_to_measured()")


def test_sync_sets_target():
    proc = RelativeCommandProcessor()
    proc.sync_to_measured([0.55, -0.10, -0.20], 7.5)
    t = proc.target
    assert _approx(t.x, 0.55)
    assert _approx(t.y, -0.10)
    assert _approx(t.z, -0.20)
    assert _approx(t.rot_y_deg, 7.5)


def test_reset_drops_target():
    proc = RelativeCommandProcessor()
    proc.sync_to_measured([0.0, 0.0, 0.0], 0.0)
    proc.reset()
    assert proc.target is None


def test_target_property_returns_copy():
    proc = RelativeCommandProcessor()
    proc.sync_to_measured([0.0, 0.0, 0.0], 0.0)
    snapshot = proc.target
    snapshot.x = 99.0
    assert _approx(proc.target.x, 0.0), "target property must return a copy, not the live object"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_step_safe_pose_integrates_linearly():
    cfg = ProcessorConfig(limiter=LimiterConfig(max_ee_speed_mps=10.0, max_yaw_rate_dps=1000.0))
    proc = RelativeCommandProcessor(cfg)
    proc.sync_to_measured([0.50, 0.0, 0.0], 0.0)
    target, dbg = proc.step([0.10, 0.0, -0.05], 4.0, 0.10, **SAFE_SIGNALS)
    assert _approx(dbg["scale"], 1.0)
    assert _approx(target.x, 0.510)       # 0.50 + 0.10 m/s * 0.10 s
    assert _approx(target.y, 0.000)
    assert _approx(target.z, -0.005)      # 0.0 + (-0.05) * 0.10
    assert _approx(target.rot_y_deg, 0.4)  # 0.0 + 4.0 dps * 0.10 s


def test_step_at_joint_limit_freezes_target():
    proc = RelativeCommandProcessor()
    proc.sync_to_measured([0.50, 0.0, 0.0], 0.0)
    angles = SAFE_ANGLES_RAD.copy()
    angles[1] = JOINT_LIMITS_RAD[1][1]  # boom at upper limit -> scale 0
    target, dbg = proc.step(
        [0.10, 0.0, 0.0], 4.0, 0.10,
        joint_angles_rad=angles,
        joint_limits_rad=JOINT_LIMITS_RAD,
        yoshikawa=0.5,
        cond=10.0,
    )
    assert _approx(dbg["scale"], 0.0)
    assert dbg["dominant"] == "joint"
    assert _approx(target.x, 0.50)
    assert _approx(target.rot_y_deg, 0.0)


def test_step_partial_scale_still_moves():
    """In the slow zone, the target should still advance but at reduced rate."""
    cfg = ProcessorConfig(limiter=LimiterConfig(max_ee_speed_mps=10.0))
    proc = RelativeCommandProcessor(cfg)
    proc.sync_to_measured([0.50, 0.0, 0.0], 0.0)
    angles = SAFE_ANGLES_RAD.copy()
    # 4 deg inside the 8 deg slow zone -> raw_margin 0.5 -> smoothstep 0.5
    angles[1] = JOINT_LIMITS_RAD[1][1] - math.radians(4.0)
    target, dbg = proc.step(
        [0.10, 0.0, 0.0], 0.0, 0.10,
        joint_angles_rad=angles,
        joint_limits_rad=JOINT_LIMITS_RAD,
        yoshikawa=0.5,
        cond=10.0,
    )
    assert _approx(dbg["scale"], 0.5, tol=1e-6)
    # Half-speed: 0.10 m/s * 0.5 * 0.10 s = 0.005 m
    assert _approx(target.x, 0.505)


def test_step_clamps_huge_dt():
    cfg = ProcessorConfig(max_dt_s=0.05, limiter=LimiterConfig(max_ee_speed_mps=10.0))
    proc = RelativeCommandProcessor(cfg)
    proc.sync_to_measured([0.0, 0.0, 0.0], 0.0)
    target, dbg = proc.step([1.0, 0.0, 0.0], 0.0, 5.0, **SAFE_SIGNALS)
    assert _approx(dbg["dt"], 0.05)
    assert _approx(target.x, 0.05)  # 1.0 m/s * 0.05 s, scale=1


def test_step_negative_dt_clamped_to_zero():
    proc = RelativeCommandProcessor()
    proc.sync_to_measured([0.0, 0.0, 0.0], 0.0)
    target, dbg = proc.step([1.0, 0.0, 0.0], 0.0, -1.0, **SAFE_SIGNALS)
    assert _approx(dbg["dt"], 0.0)
    assert _approx(target.x, 0.0)


def test_yaw_wraps_in_range():
    cfg = ProcessorConfig(
        yaw_wrap_deg=180.0,
        max_dt_s=2.0,  # allow a full-second tick so the test wrap is reachable
        limiter=LimiterConfig(max_yaw_rate_dps=1000.0),
    )
    proc = RelativeCommandProcessor(cfg)
    proc.sync_to_measured([0.0, 0.0, 0.0], 170.0)
    target, _ = proc.step([0.0, 0.0, 0.0], 30.0, 1.0, **SAFE_SIGNALS)
    # 170 + 30 * 1.0 = 200 -> wraps to -160
    assert _approx(target.rot_y_deg, -160.0, tol=1e-4)


def test_multi_step_accumulates():
    cfg = ProcessorConfig(limiter=LimiterConfig(max_ee_speed_mps=10.0))
    proc = RelativeCommandProcessor(cfg)
    proc.sync_to_measured([0.0, 0.0, 0.0], 0.0)
    for _ in range(10):
        proc.step([0.10, 0.0, 0.0], 0.0, 0.10, **SAFE_SIGNALS)
    # 10 * (0.10 m/s * 0.10 s) = 0.10 m
    assert _approx(proc.target.x, 0.10)


def test_velocity_cap_limits_displacement():
    cfg = ProcessorConfig(limiter=LimiterConfig(max_ee_speed_mps=0.20))
    proc = RelativeCommandProcessor(cfg)
    proc.sync_to_measured([0.0, 0.0, 0.0], 0.0)
    # Command 1 m/s on X, scale=1, but cap=0.2 m/s -> 0.2 * 0.10 = 0.02 m
    target, _ = proc.step([1.0, 0.0, 0.0], 0.0, 0.10, **SAFE_SIGNALS)
    assert _approx(target.x, 0.02)


# ---------------------------------------------------------------------------
# Debug surface
# ---------------------------------------------------------------------------

def test_debug_dict_has_expected_keys():
    proc = RelativeCommandProcessor()
    proc.sync_to_measured([0.0, 0.0, 0.0], 0.0)
    _, dbg = proc.step([0.1, 0.0, 0.0], 0.0, 0.01, **SAFE_SIGNALS)
    for key in ("scale", "raw_margin", "dominant", "margins",
                "applied_v_xyz", "applied_v_yaw_dps", "dt"):
        assert key in dbg, f"missing debug key: {key}"
    assert dbg["dominant"] in ("joint", "yoshikawa", "cond")


def test_resync_recenters_after_drift():
    proc = RelativeCommandProcessor(ProcessorConfig(limiter=LimiterConfig(max_ee_speed_mps=10.0)))
    proc.sync_to_measured([0.50, 0.0, 0.0], 0.0)
    for _ in range(5):
        proc.step([0.10, 0.0, 0.0], 0.0, 0.10, **SAFE_SIGNALS)
    # Pretend a pause/resume happened and the measured pose is somewhere else.
    proc.sync_to_measured([0.30, 0.10, -0.20], 45.0)
    t = proc.target
    assert _approx(t.x, 0.30) and _approx(t.y, 0.10) and _approx(t.z, -0.20)
    assert _approx(t.rot_y_deg, 45.0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"  ERROR {fn.__name__}: {e!r}")
    print()
    print(f"{len(tests) - fails}/{len(tests)} passed")
    return 0 if fails == 0 else 1


def _print_demo() -> None:
    """Run a 20-tick simulated session and print the trajectory."""
    cfg = ProcessorConfig(limiter=LimiterConfig(max_ee_speed_mps=0.20))
    proc = RelativeCommandProcessor(cfg)
    proc.sync_to_measured([0.50, 0.0, 0.0], 0.0)
    print("\nSimulated trajectory (constant +X stick, boom walking toward upper limit):")
    print(f"  {'tick':>4} {'boom_deg':>10} {'scale':>7} {'dominant':>10} {'x':>8}")
    hi = JOINT_LIMITS_RAD[1][1]
    for tick in range(20):
        # Pretend the boom angle creeps toward its upper limit as we drive +X.
        boom = hi - math.radians(20.0 - tick * 1.2)
        angles = SAFE_ANGLES_RAD.copy()
        angles[1] = boom
        target, dbg = proc.step(
            [0.20, 0.0, 0.0], 0.0, 0.05,
            joint_angles_rad=angles,
            joint_limits_rad=JOINT_LIMITS_RAD,
            yoshikawa=0.5,
            cond=10.0,
        )
        print(f"  {tick:>4} {math.degrees(boom):>10.2f} "
              f"{dbg['scale']:>7.3f} {dbg['dominant']:>10} {target.x:>8.4f}")


if __name__ == "__main__":
    rc = _run_all()
    if rc == 0:
        _print_demo()
    sys.exit(rc)
