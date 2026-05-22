"""Dry-run tests for reachability_limiter. No hardware, no IK, just numpy.

Run directly:   python control_prototype/test_reachability_limiter.py
Or via pytest:  pytest control_prototype/test_reachability_limiter.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from reachability_limiter import (
    LimiterConfig,
    apply_scale,
    cond_margin,
    combine_margins,
    compute_scale,
    joint_limit_margin,
    smoothstep,
    yoshikawa_margin,
)
from excavator_fixture import REAL_JOINT_LIMITS_RAD, SAFE_ANGLES_RAD


# Joint limits and mid-pose come from the real robot config
# (configuration_files/profiles/rpi/control_config.yaml, mirrored in excavator_fixture.py).
# Slew is unbounded (None); boom/arm/bucket carry the real ranges.
JOINT_LIMITS_RAD = REAL_JOINT_LIMITS_RAD


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------
# smoothstep
# --------------------------------------------------------------------------

def test_smoothstep_endpoints():
    assert _approx(smoothstep(0.0), 0.0)
    assert _approx(smoothstep(1.0), 1.0)
    assert _approx(smoothstep(0.5), 0.5)


def test_smoothstep_clamps_out_of_range():
    assert _approx(smoothstep(-3.0), 0.0)
    assert _approx(smoothstep(7.5), 1.0)


def test_smoothstep_monotone():
    xs = np.linspace(0.0, 1.0, 50)
    ys = np.array([smoothstep(x) for x in xs])
    assert np.all(np.diff(ys) >= -1e-12), "smoothstep must be monotone non-decreasing"


# --------------------------------------------------------------------------
# joint_limit_margin
# --------------------------------------------------------------------------

def test_joint_margin_at_center_is_one():
    m, idx = joint_limit_margin(SAFE_ANGLES_RAD, JOINT_LIMITS_RAD, math.radians(8.0))
    assert _approx(m, 1.0), f"expected margin 1.0 at center, got {m}"
    assert idx == -1


def test_joint_margin_at_upper_limit_is_zero():
    angles = SAFE_ANGLES_RAD.copy()
    # Park boom exactly at its upper limit (30 deg).
    angles[1] = JOINT_LIMITS_RAD[1][1]
    m, idx = joint_limit_margin(angles, JOINT_LIMITS_RAD, math.radians(8.0))
    assert _approx(m, 0.0), f"expected margin 0.0 at limit, got {m}"
    assert idx == 1


def test_joint_margin_inside_slow_zone_is_linear():
    # 4 deg inside an 8 deg slow zone -> margin 0.5 (before smoothstep).
    angles = SAFE_ANGLES_RAD.copy()
    angles[2] = JOINT_LIMITS_RAD[2][1] - math.radians(4.0)
    m, idx = joint_limit_margin(angles, JOINT_LIMITS_RAD, math.radians(8.0))
    assert _approx(m, 0.5, tol=1e-6), f"expected 0.5, got {m}"
    assert idx == 2


def test_joint_margin_skips_unbounded_joints():
    angles = np.array([1e6, 0.0])
    limits = [None, (-1.0, 1.0)]
    m, idx = joint_limit_margin(angles, limits, math.radians(8.0))
    # First joint is unbounded; second is at center.
    assert _approx(m, 1.0)
    assert idx == -1


def test_joint_margin_violation_clamps_to_zero():
    angles = SAFE_ANGLES_RAD.copy()
    # Push boom past its upper limit -- margin must clamp to 0, not go negative.
    angles[1] = JOINT_LIMITS_RAD[1][1] + math.radians(5.0)
    m, idx = joint_limit_margin(angles, JOINT_LIMITS_RAD, math.radians(8.0))
    assert m == 0.0
    assert idx == 1


# --------------------------------------------------------------------------
# yoshikawa_margin
# --------------------------------------------------------------------------

def test_compute_scale_nonfinite_freezes_motion():
    """Sensor dropout / IK init: a NaN signal must clamp to scale 0, not propagate."""
    cfg = LimiterConfig()
    out = compute_scale(
        joint_angles_rad=SAFE_ANGLES_RAD,
        joint_limits_rad=JOINT_LIMITS_RAD,
        yoshikawa=float("nan"),
        cond=10.0,
        cfg=cfg,
    )
    assert out["scale"] == 0.0
    assert out["dominant"] == "nonfinite"

    out = compute_scale(
        joint_angles_rad=[float("nan"), 0.0, 0.0, 0.0],
        joint_limits_rad=JOINT_LIMITS_RAD,
        yoshikawa=0.5,
        cond=10.0,
        cfg=cfg,
    )
    assert out["scale"] == 0.0
    assert out["dominant"] == "nonfinite"


def test_compute_scale_inf_cond_still_safe():
    """Perfect singularity: cond -> inf must give scale 0 cleanly (not NaN)."""
    cfg = LimiterConfig()
    out = compute_scale(
        joint_angles_rad=SAFE_ANGLES_RAD,
        joint_limits_rad=JOINT_LIMITS_RAD,
        yoshikawa=0.5,
        cond=float("inf"),
        cfg=cfg,
    )
    assert out["scale"] == 0.0
    assert out["dominant"] == "nonfinite"


def test_yoshikawa_margin_endpoints():
    assert _approx(yoshikawa_margin(0.0, 0.1), 0.0)
    assert _approx(yoshikawa_margin(0.1, 0.1), 1.0)
    assert _approx(yoshikawa_margin(0.5, 0.1), 1.0)  # saturates


def test_yoshikawa_margin_midpoint():
    assert _approx(yoshikawa_margin(0.05, 0.1), 0.5)


# --------------------------------------------------------------------------
# cond_margin
# --------------------------------------------------------------------------

def test_cond_margin_endpoints():
    assert _approx(cond_margin(20.0, 20.0, 40.0), 1.0)
    assert _approx(cond_margin(40.0, 20.0, 40.0), 0.0)
    assert _approx(cond_margin(5.0, 20.0, 40.0), 1.0)   # well below cond_ok
    assert _approx(cond_margin(80.0, 20.0, 40.0), 0.0)  # well above cond_warn


def test_cond_margin_midpoint():
    assert _approx(cond_margin(30.0, 20.0, 40.0), 0.5)


# --------------------------------------------------------------------------
# combine_margins
# --------------------------------------------------------------------------

def test_combine_takes_min_and_reports_index():
    m, idx = combine_margins(0.9, 0.3, 0.8)
    assert _approx(m, 0.3)
    assert idx == 1


# --------------------------------------------------------------------------
# compute_scale (end-to-end)
# --------------------------------------------------------------------------

def test_compute_scale_safe_pose_is_full_speed():
    cfg = LimiterConfig()
    out = compute_scale(
        joint_angles_rad=SAFE_ANGLES_RAD,
        joint_limits_rad=JOINT_LIMITS_RAD,
        yoshikawa=0.5,   # well above nominal
        cond=10.0,       # well below cond_ok
        cfg=cfg,
    )
    assert _approx(out["scale"], 1.0), out


def test_compute_scale_joint_limit_kills_motion():
    cfg = LimiterConfig()
    angles = SAFE_ANGLES_RAD.copy()
    angles[1] = JOINT_LIMITS_RAD[1][1]  # boom at upper limit
    out = compute_scale(
        joint_angles_rad=angles,
        joint_limits_rad=JOINT_LIMITS_RAD,
        yoshikawa=0.5,
        cond=10.0,
        cfg=cfg,
    )
    assert _approx(out["scale"], 0.0)
    assert out["dominant"] == "joint"
    assert out["worst_joint_idx"] == 1


def test_compute_scale_singularity_kills_motion():
    cfg = LimiterConfig()
    out = compute_scale(
        joint_angles_rad=SAFE_ANGLES_RAD,
        joint_limits_rad=JOINT_LIMITS_RAD,
        yoshikawa=0.0,   # singular
        cond=10.0,
        cfg=cfg,
    )
    assert _approx(out["scale"], 0.0)
    assert out["dominant"] == "yoshikawa"


def test_compute_scale_condition_dominates():
    cfg = LimiterConfig()
    out = compute_scale(
        joint_angles_rad=SAFE_ANGLES_RAD,
        joint_limits_rad=JOINT_LIMITS_RAD,
        yoshikawa=0.5,
        cond=cfg.cond_warn,
        cfg=cfg,
    )
    assert _approx(out["scale"], 0.0)
    assert out["dominant"] == "cond"


def test_compute_scale_is_smooth_no_jumps():
    """Walking the boom joint toward its upper limit must produce a smooth scale curve."""
    cfg = LimiterConfig()
    hi = JOINT_LIMITS_RAD[1][1]
    slow_zone = math.radians(cfg.joint_slow_zone_deg)
    # Sweep from 2*slow_zone inside the limit up to the limit itself.
    sweep = np.linspace(hi - 2.0 * slow_zone, hi, 41)
    scales = []
    for theta in sweep:
        angles = SAFE_ANGLES_RAD.copy()
        angles[1] = theta
        out = compute_scale(
            joint_angles_rad=angles,
            joint_limits_rad=JOINT_LIMITS_RAD,
            yoshikawa=0.5,
            cond=10.0,
            cfg=cfg,
        )
        scales.append(out["scale"])
    scales = np.asarray(scales)
    # Monotone non-increasing.
    assert np.all(np.diff(scales) <= 1e-9), f"scale not monotone: {scales}"
    # Endpoints: full speed far from limit, zero at the limit.
    assert _approx(scales[0], 1.0)
    assert _approx(scales[-1], 0.0)
    # No step larger than 0.1 between adjacent samples -> smooth.
    assert float(np.max(np.abs(np.diff(scales)))) < 0.1, f"scale step too large: {np.max(np.abs(np.diff(scales)))}"


# --------------------------------------------------------------------------
# apply_scale
# --------------------------------------------------------------------------

def test_apply_scale_zero_freezes_motion():
    cfg = LimiterConfig()
    v, yaw = apply_scale([0.5, 0.0, -0.3], 20.0, scale=0.0, cfg=cfg)
    assert _approx(float(np.linalg.norm(v)), 0.0)
    assert _approx(yaw, 0.0)


def test_apply_scale_one_passes_through_under_cap():
    cfg = LimiterConfig(max_ee_speed_mps=10.0, max_yaw_rate_dps=1000.0)
    v, yaw = apply_scale([0.05, 0.0, -0.05], 12.0, scale=1.0, cfg=cfg)
    assert np.allclose(v, [0.05, 0.0, -0.05], atol=1e-6)
    assert _approx(yaw, 12.0)


def test_apply_scale_caps_linear_speed():
    cfg = LimiterConfig(max_ee_speed_mps=0.1)
    v, _ = apply_scale([1.0, 0.0, 0.0], 0.0, scale=1.0, cfg=cfg)
    assert _approx(float(np.linalg.norm(v)), 0.1, tol=1e-6)


def test_apply_scale_caps_yaw_rate():
    cfg = LimiterConfig(max_yaw_rate_dps=15.0)
    _, yaw = apply_scale([0.0, 0.0, 0.0], 100.0, scale=1.0, cfg=cfg)
    assert _approx(yaw, 15.0)
    _, yaw = apply_scale([0.0, 0.0, 0.0], -100.0, scale=1.0, cfg=cfg)
    assert _approx(yaw, -15.0)


# --------------------------------------------------------------------------
# Test runner
# --------------------------------------------------------------------------

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


def _print_sweep() -> None:
    """Visualize the smoothstep curve and the joint-walk sweep."""
    cfg = LimiterConfig()
    print("\nSmoothstep curve (margin -> scale):")
    for m in np.linspace(0.0, 1.0, 11):
        bar = "#" * int(round(smoothstep(float(m)) * 40))
        print(f"  margin={m:.2f} -> scale={smoothstep(float(m)):.3f}  |{bar}")

    print("\nBoom joint walk toward upper limit (deg below limit -> scale):")
    hi = JOINT_LIMITS_RAD[1][1]
    slow_zone = math.radians(cfg.joint_slow_zone_deg)
    for frac in np.linspace(0.0, 2.0, 21):
        angles = SAFE_ANGLES_RAD.copy()
        angles[1] = hi - frac * slow_zone
        out = compute_scale(
            joint_angles_rad=angles,
            joint_limits_rad=JOINT_LIMITS_RAD,
            yoshikawa=0.5,
            cond=10.0,
            cfg=cfg,
        )
        deg_to_limit = math.degrees(hi - angles[1])
        bar = "#" * int(round(out["scale"] * 40))
        print(f"  {deg_to_limit:5.2f} deg below limit -> scale={out['scale']:.3f}  |{bar}")


if __name__ == "__main__":
    rc = _run_all()
    if rc == 0:
        _print_sweep()
    sys.exit(rc)
