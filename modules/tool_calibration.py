#!/usr/bin/env python3
"""
Tool IMU mounting-offset helper.

This helper reads a raw IMU quaternion from the Pico, assumes the tool is
freely hanging under gravity during capture, and prints a suggested mounting
offset quaternion for the selected IMU.

It does not write configuration files. It only prints results for manual copy.

Assumptions:
- The selected IMU is mounted on the tool being calibrated.
- During capture, the tool hangs at rest so one chosen local tool axis should
  point along world gravity.
- Tool length is handled elsewhere; this helper only solves orientation offset.

Example:
    python -m modules.tool_calibration --imu 4 --samples 200 --tool-axis 0,0,-1
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable

import numpy as np

try:
    from .imu_stream_reader import USBSerialReader
    from .quaternion_math import (
        quat_conjugate,
        quat_multiply,
        quat_normalize,
        quat_rotate_vector,
        euler_xyz_from_quat,
    )
except ImportError:  # pragma: no cover - script mode fallback
    from imu_stream_reader import USBSerialReader
    from quaternion_math import (
        quat_conjugate,
        quat_multiply,
        quat_normalize,
        quat_rotate_vector,
        euler_xyz_from_quat,
    )


def _parse_vec3(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    try:
        vec = np.asarray([float(p) for p in parts], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric x,y,z") from exc
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        raise argparse.ArgumentTypeError("vector must be non-zero")
    return vec / norm


def _quat_align_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Return the minimal quaternion rotating v_from onto v_to."""
    a = np.asarray(v_from, dtype=np.float32)
    b = np.asarray(v_to, dtype=np.float32)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if dot < -1.0 + 1e-6:
        # Pick an arbitrary stable orthogonal axis for the 180-degree case.
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(a, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = np.cross(a, ref)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)

    cross = np.cross(a, b).astype(np.float32)
    q = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float32)
    return quat_normalize(q)


def _average_quaternions(quats: Iterable[np.ndarray]) -> np.ndarray:
    quats = [quat_normalize(np.asarray(q, dtype=np.float32)) for q in quats]
    if not quats:
        raise ValueError("no quaternions to average")

    ref = quats[0]
    accum = np.zeros(4, dtype=np.float32)
    for q in quats:
        if float(np.dot(ref, q)) < 0.0:
            q = -q
        accum += q
    return quat_normalize(accum)


def _format_quat(q: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.7f}" for v in q) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a suggested tool IMU mounting offset quaternion.")
    parser.add_argument("--imu", type=int, required=True, help="Raw IMU index from firmware stream")
    parser.add_argument("--samples", type=int, default=200, help="Number of frames to average")
    parser.add_argument("--warmup", type=int, default=100, help="Frames to discard before capture")
    parser.add_argument("--timeout", type=float, default=10.0, help="Capture timeout in seconds")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--tool-axis", type=_parse_vec3, default=_parse_vec3("0,0,-1"),
                        help="Local IMU/tool axis expected to point along gravity while hanging, formatted x,y,z")
    parser.add_argument("--gravity", type=_parse_vec3, default=_parse_vec3("0,0,-1"),
                        help="World gravity direction, formatted x,y,z")
    parser.add_argument("--port", type=str, default=None, help="Serial port override")
    parser.add_argument("--sim", action="store_true", help="Use simulation mode")
    args = parser.parse_args()

    reader = USBSerialReader(
        baud_rate=args.baud,
        timeout=1.0,
        simulation_mode=args.sim,
        port=args.port,
    )

    try:
        reader.start_background_reader()
        print(f"Capturing IMU {args.imu} with tool-axis {args.tool_axis.tolist()} and gravity {args.gravity.tolist()}")

        warmup_remaining = int(max(0, args.warmup))
        samples: list[np.ndarray] = []
        start_t = time.time()

        while len(samples) < args.samples:
            if (time.time() - start_t) > args.timeout:
                print(f"Timed out after {args.timeout:.1f}s while waiting for IMU data", file=sys.stderr)
                return 1

            frame = reader.get_latest_imus(only_new=True)
            if not frame:
                time.sleep(0.002)
                continue
            if args.imu >= len(frame):
                print(
                    f"Requested IMU {args.imu}, but latest frame contains {len(frame)} sensor(s)",
                    file=sys.stderr,
                )
                return 1

            quat = quat_normalize(np.asarray(frame[args.imu][:4], dtype=np.float32))

            if warmup_remaining > 0:
                warmup_remaining -= 1
                continue

            samples.append(quat)

        avg_raw_quat = _average_quaternions(samples)
        measured_world_axis = quat_rotate_vector(avg_raw_quat, args.tool_axis)
        gravity_align = _quat_align_vectors(measured_world_axis, args.gravity)
        corrected_quat = quat_normalize(quat_multiply(gravity_align, avg_raw_quat))
        offset_quat = quat_normalize(quat_multiply(quat_conjugate(corrected_quat), avg_raw_quat))
        corrected_world_axis = quat_rotate_vector(corrected_quat, args.tool_axis)

        raw_euler_deg = np.degrees(euler_xyz_from_quat(avg_raw_quat))
        corrected_euler_deg = np.degrees(euler_xyz_from_quat(corrected_quat))

        print("")
        print("Suggested tool IMU mounting offset")
        print(f"imu_index: {args.imu}")
        print(f"samples: {len(samples)}")
        print(f"avg_raw_quat: {_format_quat(avg_raw_quat)}")
        print(f"suggested_offset_quat: {_format_quat(offset_quat)}")
        print(f"corrected_quat: {_format_quat(corrected_quat)}")
        print(f"measured_world_tool_axis: [{measured_world_axis[0]:.6f}, {measured_world_axis[1]:.6f}, {measured_world_axis[2]:.6f}]")
        print(f"corrected_world_tool_axis: [{corrected_world_axis[0]:.6f}, {corrected_world_axis[1]:.6f}, {corrected_world_axis[2]:.6f}]")
        print(f"raw_euler_deg_xyz: [{raw_euler_deg[0]:.3f}, {raw_euler_deg[1]:.3f}, {raw_euler_deg[2]:.3f}]")
        print(f"corrected_euler_deg_xyz: [{corrected_euler_deg[0]:.3f}, {corrected_euler_deg[1]:.3f}, {corrected_euler_deg[2]:.3f}]")
        print("")
        print("For runtime correction with q_corrected = q_raw * conj(q_offset), use:")
        print(_format_quat(offset_quat))
        print("")
        print("This solves a gravity-aligned offset for the chosen tool axis only.")
        print("If the tool can freely spin around its hanging axis, that remaining twist is not physically observable.")
        return 0
    finally:
        try:
            reader.stop_background_reader()
        except Exception:
            pass
        if getattr(reader, "ser", None) is not None:
            try:
                reader.ser.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
