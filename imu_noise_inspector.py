#!/usr/bin/env python3
"""
Stationary IMU noise inspector.

Run this with the robot/IMUs fully stationary to compare Pico AHRS settings,
especially gain values. Firmware settings are startup-only, so reset/power-cycle
the Pico between runs when changing --gain.

Examples:
    python -m tools.imu_noise_inspector --gain 5.0 --duration 30
    python -m tools.imu_noise_inspector --gain 4.5 --duration 30
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

from modules.quaternion_math import euler_xyz_from_quat, quat_normalize
from modules.usb_serial_reader import USBSerialReader


def _quat_array(quats: list[list[float]]) -> np.ndarray:
    out = np.asarray(quats, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 4:
        return np.empty((0, 4), dtype=np.float64)
    norms = np.linalg.norm(out, axis=1)
    valid = norms > 1e-12
    out = out[valid]
    norms = norms[valid]
    if out.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    return out / norms[:, None]


def _mean_quat(quats: np.ndarray) -> np.ndarray:
    ref = quats[0]
    aligned = quats.copy()
    signs = np.sign(aligned @ ref)
    signs[signs == 0.0] = 1.0
    aligned *= signs[:, None]
    mean = aligned.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        return ref
    return mean / norm


def _quat_noise_deg(quats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(quats) == 0:
        return np.empty(0, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mean = _mean_quat(quats)
    dots = np.abs(quats @ mean)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(2.0 * np.arccos(dots))
    return angles, mean


def _euler_std_deg(quats: np.ndarray) -> tuple[float, float, float]:
    if len(quats) == 0:
        return math.nan, math.nan, math.nan
    eulers = np.asarray([euler_xyz_from_quat(q.astype(np.float32)) for q in quats], dtype=np.float64)
    eulers = np.unwrap(eulers, axis=0)
    std = np.degrees(np.std(eulers, axis=0))
    return float(std[0]), float(std[1]), float(std[2])


def _fmt_vec(values: np.ndarray | tuple[float, ...], digits: int = 4) -> str:
    vals = [float(v) for v in values]
    return "[" + ", ".join(f"{v:.{digits}f}" for v in vals) + "]"


def _compute_stats(quats: list[list[float]], gyros: list[list[float]]) -> dict[str, object]:
    q = _quat_array(quats)
    g = np.asarray(gyros, dtype=np.float64)
    angles, mean_q = _quat_noise_deg(q)
    roll_std, pitch_std, yaw_std = _euler_std_deg(q)

    if g.ndim != 2 or g.shape[1] != 3 or len(g) == 0:
        gyro_mean = np.array([math.nan, math.nan, math.nan], dtype=np.float64)
        gyro_std = np.array([math.nan, math.nan, math.nan], dtype=np.float64)
        gyro_rms = math.nan
    else:
        gyro_mean = np.mean(g, axis=0)
        gyro_std = np.std(g, axis=0)
        gyro_rms = float(np.sqrt(np.mean(np.sum((g - gyro_mean) ** 2, axis=1))))

    return {
        "samples": int(len(q)),
        "mean_quat": mean_q,
        "quat_std_deg": float(np.std(angles)) if len(angles) else math.nan,
        "quat_p95_deg": float(np.percentile(angles, 95)) if len(angles) else math.nan,
        "quat_max_deg": float(np.max(angles)) if len(angles) else math.nan,
        "roll_std_deg": roll_std,
        "pitch_std_deg": pitch_std,
        "yaw_std_deg": yaw_std,
        "gyro_mean_dps": gyro_mean,
        "gyro_std_dps": gyro_std,
        "gyro_rms_dps": gyro_rms,
    }


def _print_calibration_report(reader: USBSerialReader) -> bool:
    report = reader.status().get("calibration_report")
    if not report:
        return False

    state = "accepted" if report.get("accepted") else "rejected"
    print(f"Startup calibration: {state}, duration={report.get('duration_s', 0.0):.1f}s")
    print("  gyro_bias values are subtracted in Pico firmware before streaming gx/gy/gz.")
    thresholds = report.get("thresholds", {})
    print(
        "  thresholds: "
        f"gyro_std<={thresholds.get('gyro_std_dps', math.nan):.4f} dps, "
        f"accel_std<={thresholds.get('accel_std_g', math.nan):.5f} g, "
        f"accel_norm={thresholds.get('accel_norm_min_g', math.nan):.3f}.."
        f"{thresholds.get('accel_norm_max_g', math.nan):.3f} g"
    )
    for sensor in report.get("sensors", []):
        failures = ",".join(sensor.get("failures", [])) or "ok"
        print(
            f"  cal[{sensor.get('index')}] samples={sensor.get('sample_count')} "
            f"gyro_bias={_fmt_vec(sensor.get('gyro_bias_dps', []), 4)} dps "
            f"gyro_std={_fmt_vec(sensor.get('gyro_std_dps', []), 4)} dps "
            f"accel_std={_fmt_vec(sensor.get('accel_std_g', []), 5)} g "
            f"accel_norm={sensor.get('accel_norm_min_g', math.nan):.4f}/"
            f"{sensor.get('accel_norm_mean_g', math.nan):.4f}/"
            f"{sensor.get('accel_norm_max_g', math.nan):.4f} result={failures}"
        )
    return True


def _print_summary(stats_by_imu: dict[int, dict[str, object]]) -> None:
    print("\nStationary noise summary")
    print(
        "IMU  N      quat_std  quat_p95  quat_max  "
        "euler_std[roll,pitch,yaw] deg      gyro_std[x,y,z] dps       gyro_rms"
    )
    for imu_index in sorted(stats_by_imu):
        st = stats_by_imu[imu_index]
        euler_std = (st["roll_std_deg"], st["pitch_std_deg"], st["yaw_std_deg"])
        print(
            f"{imu_index:>3}  {st['samples']:>5}  "
            f"{st['quat_std_deg']:>8.5f}  {st['quat_p95_deg']:>8.5f}  {st['quat_max_deg']:>8.5f}  "
            f"{_fmt_vec(euler_std, 5):<33} "
            f"{_fmt_vec(st['gyro_std_dps'], 5):<27} "
            f"{st['gyro_rms_dps']:>8.5f}"
        )


def _write_csv(path: Path, args: argparse.Namespace, stats_by_imu: dict[int, dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "gain", "sample_rate", "gyro_dps", "imu", "samples",
            "quat_std_deg", "quat_p95_deg", "quat_max_deg",
            "roll_std_deg", "pitch_std_deg", "yaw_std_deg",
            "gyro_mean_x_dps", "gyro_mean_y_dps", "gyro_mean_z_dps",
            "gyro_std_x_dps", "gyro_std_y_dps", "gyro_std_z_dps", "gyro_rms_dps",
        ])
        for imu_index in sorted(stats_by_imu):
            st = stats_by_imu[imu_index]
            gyro_mean = np.asarray(st["gyro_mean_dps"], dtype=np.float64)
            gyro_std = np.asarray(st["gyro_std_dps"], dtype=np.float64)
            writer.writerow([
                args.gain, args.sr, args.gyro_dps, imu_index, st["samples"],
                st["quat_std_deg"], st["quat_p95_deg"], st["quat_max_deg"],
                st["roll_std_deg"], st["pitch_std_deg"], st["yaw_std_deg"],
                gyro_mean[0], gyro_mean[1], gyro_mean[2],
                gyro_std[0], gyro_std[1], gyro_std[2], st["gyro_rms_dps"],
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure stationary quaternion and gyro noise from Pico IMUs.")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--duration", type=float, default=30.0, help="Capture duration in seconds")
    parser.add_argument("--warmup", type=float, default=2.0, help="Seconds to discard after streaming starts")
    parser.add_argument("--imu", type=int, default=-1, help="IMU index to inspect, or -1 for all")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path for summary stats")
    parser.add_argument("--live-period", type=float, default=1.0, help="Seconds between live progress updates (0=off)")

    parser.add_argument("--sr", type=int, default=USBSerialReader.DEFAULT_SAMPLE_RATE, help="Sample rate Hz")
    parser.add_argument("--gyro-dps", type=int, default=USBSerialReader.DEFAULT_GYRO_DPS,
                        help="Gyro range (125/250/500/1000/2000)")
    parser.add_argument("--gain", type=float, default=USBSerialReader.DEFAULT_GAIN, help="Pico AHRS gain")
    parser.add_argument("--accel-rejection", type=float, default=USBSerialReader.DEFAULT_ACCEL_REJ,
                        help="AHRS accel rejection threshold in degrees")
    parser.add_argument("--recovery-s", type=float, default=USBSerialReader.DEFAULT_RECOVERY_S,
                        help="AHRS recovery period in seconds")

    parser.add_argument("--sim", action="store_true", help="Use USB reader simulation mode")
    parser.add_argument("--debug", action="store_true", help="Log raw serial bytes")
    parser.add_argument("--no-wait", action="store_true", help="Skip waiting for Pico CFG_OK")
    args = parser.parse_args()

    if args.duration <= 0.0:
        parser.error("--duration must be > 0")
    if args.warmup < 0.0:
        parser.error("--warmup must be >= 0")

    print(
        f"Config: SR={args.sr} Hz, GYRO_DPS={args.gyro_dps}, GAIN={args.gain}, "
        f"ACC_REJ={args.accel_rejection}, RECOV_S={args.recovery_s}"
    )
    if not args.sim:
        print("Keep the machine fully stationary. Reset/power-cycle Pico between different gain runs.")

    reader = USBSerialReader(
        baud_rate=args.baud,
        port=args.port,
        simulation_mode=args.sim,
        debug=args.debug,
    )

    try:
        if not args.sim:
            reader.send_config(
                sample_rate=args.sr,
                gyro_dps=args.gyro_dps,
                gain=args.gain,
                accel_rejection=args.accel_rejection,
                recovery_s=args.recovery_s,
            )
            if not args.no_wait:
                if not reader.wait_for_cfg_ok(timeout_s=3.0, resend_s=0.3, calibration_timeout_s=120.0):
                    print("Warning: CFG_OK not received; continuing anyway", file=sys.stderr)

        calibration_report_printed = _print_calibration_report(reader)

        if args.warmup > 0.0:
            print(f"Warming up for {args.warmup:.1f}s...")
            warmup_end = time.time() + args.warmup
            while time.time() < warmup_end:
                reader.read_imus()
                if not calibration_report_printed:
                    calibration_report_printed = _print_calibration_report(reader)
                time.sleep(0.001)

        print(f"Capturing stationary noise for {args.duration:.1f}s...")
        quats_by_imu: dict[int, list[list[float]]] = {}
        gyros_by_imu: dict[int, list[list[float]]] = {}
        started = time.time()
        next_live = started + max(0.0, args.live_period)

        while (time.time() - started) < args.duration:
            frame = reader.read_imus()
            if not calibration_report_printed:
                calibration_report_printed = _print_calibration_report(reader)
            if frame is None:
                time.sleep(0.001)
                continue

            for imu_index, packet in enumerate(frame):
                if args.imu >= 0 and imu_index != args.imu:
                    continue
                if len(packet) < 7:
                    continue
                quat = quat_normalize(np.asarray(packet[:4], dtype=np.float32)).astype(float).tolist()
                gyro = [float(v) for v in packet[4:7]]
                quats_by_imu.setdefault(imu_index, []).append(quat)
                gyros_by_imu.setdefault(imu_index, []).append(gyro)

            now = time.time()
            if args.live_period > 0.0 and now >= next_live:
                elapsed = now - started
                stats = {
                    imu: _compute_stats(quats_by_imu.get(imu, []), gyros_by_imu.get(imu, []))
                    for imu in sorted(quats_by_imu)
                }
                total_samples = sum(int(st["samples"]) for st in stats.values())
                if stats:
                    worst_p95 = max(float(st["quat_p95_deg"]) for st in stats.values())
                    worst_gyro = max(float(st["gyro_rms_dps"]) for st in stats.values())
                    print(
                        f"  t={elapsed:5.1f}s samples={total_samples} "
                        f"worst_quat_p95={worst_p95:.5f} deg worst_gyro_rms={worst_gyro:.5f} dps"
                    )
                next_live = now + args.live_period

        stats_by_imu = {
            imu: _compute_stats(quats_by_imu.get(imu, []), gyros_by_imu.get(imu, []))
            for imu in sorted(quats_by_imu)
        }
        if not stats_by_imu:
            print("No IMU samples captured", file=sys.stderr)
            return 1

        if not calibration_report_printed and not args.sim:
            print("\nNo startup calibration report received from Pico during this run.", file=sys.stderr)

        _print_summary(stats_by_imu)
        if args.csv is not None:
            _write_csv(args.csv, args, stats_by_imu)
            print(f"\nWrote summary CSV: {args.csv}")
        return 0
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
