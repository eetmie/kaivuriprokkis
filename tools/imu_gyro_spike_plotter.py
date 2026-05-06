#!/usr/bin/env python3
"""Live plot Pico IMU gyro spikes from USB only.

This diagnostic connects directly to ``modules.usb_serial_reader.USBSerialReader``.
It does not initialize robot hardware, PWM, ADC, or the controller.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.usb_serial_reader import USBSerialReader


def _fmt_vec(values: list[float] | tuple[float, ...] | np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+7.3f}" for v in values) + "]"


def _write_csv_header(writer: csv.writer, max_imus: int) -> None:
    fields = ["t_s", "device_timestamp_us"]
    for i in range(max_imus):
        fields.extend([
            f"imu{i}_gx_dps",
            f"imu{i}_gy_dps",
            f"imu{i}_gz_dps",
            f"imu{i}_mag_dps",
        ])
    writer.writerow(fields)


def _write_csv_row(writer: csv.writer, t_s: float, device_ts: int | None, gyros: np.ndarray, max_imus: int) -> None:
    row: list[object] = [f"{t_s:.6f}", "" if device_ts is None else int(device_ts)]
    for i in range(max_imus):
        if i < len(gyros):
            gx, gy, gz = gyros[i]
            mag = float(np.linalg.norm(gyros[i]))
            row.extend([f"{gx:.6f}", f"{gy:.6f}", f"{gz:.6f}", f"{mag:.6f}"])
        else:
            row.extend(["", "", "", ""])
    writer.writerow(row)


def _setup_plot(args: argparse.Namespace, max_imus: int):
    if args.no_plot:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"matplotlib unavailable; continuing without plot: {exc}", file=sys.stderr)
        return None

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Pico IMU gyro tap/spike monitor")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("gyro (deg/s)")
    ax.grid(True, alpha=0.3)

    lines = []
    for i in range(max_imus):
        (mag_line,) = ax.plot([], [], label=f"IMU{i} |gyro|", linewidth=1.8)
        lines.append((mag_line, None, None, None))

    raw_lines = []
    if args.raw:
        colors = ["tab:red", "tab:green", "tab:blue"]
        for i in range(max_imus):
            imu_lines = []
            for axis, color in zip(("x", "y", "z"), colors):
                (line,) = ax.plot([], [], label=f"IMU{i} g{axis}", linewidth=0.8, alpha=0.45, color=color)
                imu_lines.append(line)
            raw_lines.append(tuple(imu_lines))
    else:
        raw_lines = [None] * max_imus

    ax.legend(loc="upper right", ncols=2 if args.raw else 1)
    fig.tight_layout()
    return plt, fig, ax, lines, raw_lines


def _update_plot(plot_state, times: deque[float], gyro_history: list[deque[np.ndarray]], args: argparse.Namespace) -> None:
    if plot_state is None or not times:
        return
    plt, fig, ax, mag_lines, raw_lines = plot_state
    x = np.asarray(times, dtype=np.float64)
    if len(x) == 0:
        return

    for i, history in enumerate(gyro_history):
        if not history:
            continue
        g = np.asarray(history, dtype=np.float64)
        mag_lines[i][0].set_data(x[-len(g):], np.linalg.norm(g, axis=1))
        if args.raw and raw_lines[i] is not None:
            for axis in range(3):
                raw_lines[i][axis].set_data(x[-len(g):], g[:, axis])

    now = x[-1]
    ax.set_xlim(max(0.0, now - args.window), max(args.window, now))
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    fig.canvas.draw_idle()
    plt.pause(0.001)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live plot Pico IMU gyro spikes over USB only")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration seconds, 0 = until Ctrl+C")
    parser.add_argument("--rate", type=float, default=100.0, help="Host plot/log update rate Hz")
    parser.add_argument("--window", type=float, default=10.0, help="Visible plot window seconds")
    parser.add_argument("--max-imus", type=int, default=4, help="Maximum IMUs to plot/log")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--raw", action="store_true", help="Also plot gx/gy/gz traces, not just vector magnitude")
    parser.add_argument("--no-plot", action="store_true", help="Do not open matplotlib plot; print/log only")
    parser.add_argument("--sim", action="store_true", help="Use USB reader simulation mode")
    parser.add_argument("--debug", action="store_true", help="Log raw serial bytes")
    parser.add_argument("--no-wait", action="store_true", help="Skip waiting for Pico CFG_OK")
    parser.add_argument("--sr", type=int, default=200, help="Pico sample rate Hz")
    parser.add_argument("--gyro-dps", type=int, default=USBSerialReader.DEFAULT_GYRO_DPS,
                        help="Pico gyro range dps")
    parser.add_argument("--gain", type=float, default=USBSerialReader.DEFAULT_GAIN, help="Pico AHRS gain")
    parser.add_argument("--accel-rejection", type=float, default=USBSerialReader.DEFAULT_ACCEL_REJ,
                        help="Pico AHRS accel rejection")
    parser.add_argument("--recovery-s", type=float, default=USBSerialReader.DEFAULT_RECOVERY_S,
                        help="Pico AHRS recovery period")
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be > 0")
    if args.window <= 0:
        parser.error("--window must be > 0")
    if args.max_imus <= 0:
        parser.error("--max-imus must be > 0")

    print("USB-only IMU gyro spike plotter. No robot hardware/PWM/ADC/controller is initialized.")
    print(
        f"Config: SR={args.sr} Hz, GYRO_DPS={args.gyro_dps}, GAIN={args.gain}, "
        f"ACC_REJ={args.accel_rejection}, RECOV_S={args.recovery_s}"
    )

    reader = USBSerialReader(
        baud_rate=args.baud,
        port=args.port,
        simulation_mode=args.sim,
        debug=args.debug,
    )

    csv_handle = None
    csv_writer = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_handle = args.csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_handle)
        _write_csv_header(csv_writer, args.max_imus)

    times: deque[float] = deque(maxlen=max(10, int(math.ceil(args.window * args.rate * 2.0))))
    gyro_history = [
        deque(maxlen=times.maxlen)
        for _ in range(args.max_imus)
    ]
    plot_state = _setup_plot(args, args.max_imus)

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
                reader.wait_for_cfg_ok(timeout_s=3.0, resend_s=0.3, calibration_timeout_s=120.0)

        print("Reading IMU stream. Tap the robot/IMUs and watch |gyro| spikes. Ctrl+C to stop.")
        started = time.monotonic()
        next_tick = started
        next_print = started
        end_time = None if args.duration <= 0.0 else started + args.duration

        while end_time is None or time.monotonic() < end_time:
            frame = reader.read_imus()
            now = time.monotonic()
            if frame is None:
                time.sleep(0.001)
                continue

            gyros = np.asarray([pkt[4:7] for pkt in frame[:args.max_imus] if len(pkt) >= 7], dtype=np.float64)
            if gyros.ndim != 2 or gyros.shape[1] != 3:
                continue

            t_s = now - started
            times.append(t_s)
            for i in range(args.max_imus):
                if i < len(gyros):
                    gyro_history[i].append(gyros[i].copy())
                elif gyro_history[i]:
                    gyro_history[i].append(np.full(3, np.nan, dtype=np.float64))

            if csv_writer is not None:
                _write_csv_row(csv_writer, t_s, reader.last_timestamp_us, gyros, args.max_imus)

            if now >= next_print:
                mags = [float(np.linalg.norm(g)) for g in gyros]
                parts = [f"IMU{i} |g|={mags[i]:7.3f} dps g={_fmt_vec(gyros[i])}" for i in range(len(gyros))]
                print("  " + " | ".join(parts), end="\r", flush=True)
                next_print = now + 0.2

            if now >= next_tick:
                _update_plot(plot_state, times, gyro_history, args)
                next_tick = now + (1.0 / args.rate)

        print()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    finally:
        if csv_handle is not None:
            csv_handle.close()
            print(f"Wrote CSV: {args.csv}")
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
