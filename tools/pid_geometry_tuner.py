#!/usr/bin/env python3
"""Geometry-aware PID gain helper for the excavator controller.

This is an offline estimator: it reads the configured robot geometry and PID
gains, then reports what each joint gain means in joint-angle and approximate
tool-motion terms. It does not connect to hardware.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.differential_ik import compute_jacobian_from_joint_angles, get_pose_from_joint_angles
from modules.differential_ik_cfg import load_excavator_robot_config


JOINTS = [
    ("joint0", "slew", "rotate"),
    ("joint1", "boom", "lift_boom"),
    ("joint2", "arm", "tilt_boom"),
    ("joint3", "bucket", "scoop"),
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} did not contain a YAML mapping")
    return data


def _fmt(value: float, width: int = 8, precision: int = 3) -> str:
    if math.isinf(value):
        return "inf".rjust(width)
    if math.isnan(value):
        return "nan".rjust(width)
    return f"{value:{width}.{precision}f}"


def _pulse_span_us(servo_cfg: dict[str, Any], channel_name: str) -> tuple[float, float, float] | None:
    channels = servo_cfg.get("CHANNEL_CONFIGS", {})
    cfg = channels.get(channel_name) if isinstance(channels, dict) else None
    if not isinstance(cfg, dict):
        return None

    center = float(cfg["center"])
    pos_edge = center + float(cfg.get("deadband_us_pos", 0.0))
    neg_edge = center - float(cfg.get("deadband_us_neg", 0.0))
    pos_span = float(cfg["pulse_max"]) - pos_edge
    neg_span = neg_edge - float(cfg["pulse_min"])
    return max(0.0, neg_span), max(0.0, pos_span), float(cfg.get("ramp_limit", 0.0))


def _recommended_kp(full_at_deg: float | None, full_at_mm: float | None, sensitivity_m_per_rad: float) -> float | None:
    if full_at_deg is not None:
        err_rad = math.radians(max(full_at_deg, 1e-6))
        return 1.0 / err_rad
    if full_at_mm is not None and sensitivity_m_per_rad > 1e-9:
        err_rad = (full_at_mm / 1000.0) / sensitivity_m_per_rad
        return 1.0 / max(err_rad, 1e-6)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show PID gains as joint-error and geometry-scaled tool-motion numbers."
    )
    parser.add_argument("--control-config", default="configuration_files/control_config.yaml")
    parser.add_argument("--servo-config", default="configuration_files/servo_config_200.yaml")
    parser.add_argument(
        "--angles-deg",
        nargs=4,
        type=float,
        metavar=("SLEW", "BOOM", "ARM", "BUCKET"),
        default=(0.0, 0.0, 90.0, -45.0),
        help="Sample joint angles used for geometry sensitivity estimates.",
    )
    parser.add_argument(
        "--full-at-deg",
        type=float,
        default=None,
        help="Recommend kp values that reach full output at this joint error in degrees.",
    )
    parser.add_argument(
        "--full-at-mm",
        type=float,
        default=None,
        help="Recommend kp values that reach full output after this approximate tool displacement.",
    )
    args = parser.parse_args()

    control_path = ROOT_DIR / args.control_config
    servo_path = ROOT_DIR / args.servo_config
    control_cfg = _load_yaml(control_path)
    servo_cfg = _load_yaml(servo_path)
    robot_cfg = load_excavator_robot_config(str(control_path))

    joint_angles = np.radians(np.asarray(args.angles_deg, dtype=np.float32))
    ee_pos, _ = get_pose_from_joint_angles(joint_angles, robot_cfg)
    jacobian = compute_jacobian_from_joint_angles(joint_angles, robot_cfg)
    pid_cfg = control_cfg.get("pid", {})

    print("PID geometry helper")
    print(f"  control: {control_path.relative_to(ROOT_DIR)}")
    print(f"  servo:   {servo_path.relative_to(ROOT_DIR)}")
    print(f"  sample joint angles deg: {[round(float(v), 3) for v in args.angles_deg]}")
    print(f"  sample tool position m:  [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}]")
    print()

    header = (
        "joint      kp      full_deg  tool_mm@full  mm_per_deg  "
        "neg_us  pos_us  ramp_us_s"
    )
    if args.full_at_deg is not None or args.full_at_mm is not None:
        header += "  kp_suggest"
    print(header)
    print("-" * len(header))

    for i, (pid_name, joint_name, servo_name) in enumerate(JOINTS):
        cfg = pid_cfg.get(pid_name, {}) if isinstance(pid_cfg, dict) else {}
        kp = float(cfg.get("kp", 0.0)) if isinstance(cfg, dict) else 0.0
        full_rad = 1.0 / kp if kp > 0.0 else math.inf
        full_deg = math.degrees(full_rad)

        linear_sensitivity = float(np.linalg.norm(jacobian[0:3, i]))
        mm_per_deg = linear_sensitivity * math.pi / 180.0 * 1000.0
        tool_mm_at_full = linear_sensitivity * full_rad * 1000.0 if math.isfinite(full_rad) else math.inf

        pulse = _pulse_span_us(servo_cfg, servo_name)
        if pulse is None:
            neg_us = pos_us = ramp = math.nan
        else:
            neg_us, pos_us, ramp = pulse

        line = (
            f"{joint_name:<8} "
            f"{_fmt(kp, 7, 3)} "
            f"{_fmt(full_deg, 9, 2)} "
            f"{_fmt(tool_mm_at_full, 12, 1)} "
            f"{_fmt(mm_per_deg, 10, 2)} "
            f"{_fmt(neg_us, 7, 1)} "
            f"{_fmt(pos_us, 7, 1)} "
            f"{_fmt(ramp, 9, 1)}"
        )
        suggested = _recommended_kp(args.full_at_deg, args.full_at_mm, linear_sensitivity)
        if suggested is not None:
            line += f"  {_fmt(suggested, 10, 3)}"
        print(line)

    print()
    print("Notes:")
    print("  full_deg = P-only joint error needed for output clamp (+/-1.0).")
    print("  tool_mm@full is a local linearized estimate from the Jacobian at the sample pose.")
    print("  For slew, mm_per_deg grows with reach, so tune using a realistic working radius.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
