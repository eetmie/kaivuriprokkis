#!/usr/bin/env python3
"""USB IMU reader throughput stress test."""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from modules.rt_utils import apply_rt_to_thread, reset_to_normal, SCHED_FIFO  # noqa: E402
from modules.usb_serial_reader import USBSerialReader  # noqa: E402


def _percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * q))
    return ordered[idx]


def _cpu_affinity(core):
    return None if core is None else {int(core)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress-test the Pico USB IMU reader")
    parser.add_argument("--target-hz", type=int, default=200)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--fifo-priority", type=int, default=75)
    parser.add_argument("--lock-memory", action="store_true", help="Call mlockall() for RT threads")
    parser.add_argument("--cpu-core", type=int, default=3, help="CPU core for USB reader and main loop (default: 3)")
    parser.add_argument("--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    reader = None
    intervals_ms = deque(maxlen=4000)
    last_frame_time = None
    count = 0
    started = time.perf_counter()
    last_print = started

    try:
        reader = USBSerialReader(
            baud_rate=115200,
            timeout=1.0,
            simulation_mode=False,
            log_level=args.log_level.upper(),
            rt_priority=args.fifo_priority,
            rt_lock_memory=args.lock_memory,
            rt_cpu_core=args.cpu_core,
        )
        reader.send_config(sample_rate=args.target_hz)
        reader.wait_for_cfg_ok(timeout_s=3.0, calibration_timeout_s=30.0)
        reader.start_background_reader()

        apply_rt_to_thread(
            priority=args.fifo_priority,
            policy=SCHED_FIFO,
            lock_memory=args.lock_memory,
            cpu_affinity=_cpu_affinity(args.cpu_core),
            quiet=True,
        )

        print(
            f"[stress] usb reader start: target={args.target_hz}Hz duration={args.duration_s:.1f}s "
            f"fifo={args.fifo_priority} cpu_core={args.cpu_core} lock_memory={args.lock_memory}"
        )

        while (time.perf_counter() - started) < args.duration_s:
            frame = reader.get_latest_imus(only_new=True)
            if frame is not None:
                now = time.perf_counter()
                if last_frame_time is not None:
                    intervals_ms.append((now - last_frame_time) * 1000.0)
                last_frame_time = now
                count += 1
            else:
                time.sleep(0.0005)

            now = time.perf_counter()
            if now - last_print >= 1.0:
                status = reader.status()
                host_hz = count / max(1e-6, now - started)
                print(
                    "[stress] "
                    f"host={host_hz:.1f}Hz fw={status.get('timestamp_sps', 0.0):.1f}Hz raw={status.get('host_sps', 0.0):.1f}Hz "
                    f"p95={_percentile(list(intervals_ms), 0.95):.2f}ms p99={_percentile(list(intervals_ms), 0.99):.2f}ms "
                    f"sensors={status.get('num_sensors', 0)}"
                )
                last_print = now

        total_s = max(1e-6, time.perf_counter() - started)
        status = reader.status()
        print(
            "[stress] done | "
            f"host={count / total_s:.1f}Hz fw={status.get('timestamp_sps', 0.0):.1f}Hz raw={status.get('host_sps', 0.0):.1f}Hz "
            f"p95={_percentile(list(intervals_ms), 0.95):.2f}ms p99={_percentile(list(intervals_ms), 0.99):.2f}ms"
        )
        return 0
    finally:
        reset_to_normal(quiet=True)
        if reader is not None:
            try:
                reader.stop_background_reader()
            except Exception:
                pass
            try:
                reader.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
