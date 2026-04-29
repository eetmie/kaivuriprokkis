#!/usr/bin/env python3
"""Run the excavator stack at a fixed rate with explicit zero PWM commands.

This is a hardware stress/integration test, not a unit test.
It starts the real hardware interface and controller, enters direct mode,
and pushes explicit zero-valued named PWM commands every cycle.

The pump channel is left under the normal controller logic so valve zeros still
exercise the real PWM update path with the configured pump behavior.
"""

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from modules.control_protocol import (  # noqa: E402
    ControlCommand,
    ControlMode,
    DirectCommand,
    decode_command_message,
    encode_command_message,
    encode_telemetry_message,
)
from modules.excavator_controller import ControllerConfig, ExcavatorController  # noqa: E402
from modules.hardware_interface import HardwareInterface  # noqa: E402
from modules.robot_service import RobotService  # noqa: E402
from modules.rt_utils import apply_rt_to_thread, reset_to_normal, SCHED_FIFO  # noqa: E402


def _percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * q))
    return ordered[idx]


def _cpu_affinity(core):
    return None if core is None else {int(core)}


def _load_rate_config():
    cfg_path = ROOT / "configuration_files" / "control_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    rates = data.get("rates", {}) if isinstance(data, dict) else {}
    controller = data.get("controller", {}) if isinstance(data, dict) else {}
    imu = data.get("imu", {}) if isinstance(data, dict) else {}

    control_hz = float(rates.get("control_hz", 100.0))
    imu_hz = float(imu.get("sample_rate", control_hz))
    output_limits = (
        float(controller.get("output_limits_min", -1.0)),
        float(controller.get("output_limits_max", 1.0)),
    )
    return control_hz, imu_hz, output_limits


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-stack 200 Hz stress test with explicit zero PWM commands")
    parser.add_argument("--rate-hz", type=float, default=None,
                        help="Override control/IMU/ADC rates together; defaults to config values")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--ready-timeout-s", type=float, default=30.0)
    parser.add_argument("--fifo-priority", type=int, default=75)
    parser.add_argument("--lock-memory", action="store_true", help="Call mlockall() for RT threads")
    parser.add_argument("--control-core", type=int, default=2,
                        help="CPU core for sender + control loop (default: 2)")
    parser.add_argument("--io-core", type=int, default=3,
                        help="CPU core for USB reader + IMU + ADC threads (default: 3)")
    parser.add_argument("--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.WARNING))

    hardware = None
    service = None
    sender_intervals_ms = deque(maxlen=4000)
    sender_miss_count = 0
    sender_miss_window = deque(maxlen=4000)
    sender_cycle_count = 0

    try:
        config_control_hz, config_imu_hz, output_limits = _load_rate_config()
        target_hz = float(args.rate_hz) if args.rate_hz is not None else float(config_control_hz)
        target_imu_hz = float(args.rate_hz) if args.rate_hz is not None else float(config_imu_hz)

        hardware = HardwareInterface(
            config_file=str(ROOT / "configuration_files" / "servo_config_200.yaml"),
            log_level=args.log_level.upper(),
            pump_auto_mode=True,
            cleanup_disable_osc=False,
            enable_adc=False,
            start_adc_reader=False,
            imu_expected_hz=target_imu_hz,
            rt_lock_memory=args.lock_memory,
            usb_rt_priority=args.fifo_priority,
            imu_rt_priority=args.fifo_priority,
            adc_rt_priority=args.fifo_priority,
            usb_cpu_core=args.io_core,
            imu_cpu_core=args.io_core,
            adc_cpu_core=args.io_core,
        )

        deadline = time.time() + max(1.0, args.ready_timeout_s)
        while not hardware.is_hardware_ready():
            if time.time() >= deadline:
                raise TimeoutError("Hardware did not become ready before timeout")
            time.sleep(0.1)

        controller = ExcavatorController(
            hardware,
            config=ControllerConfig(output_limits=output_limits, control_frequency=target_hz),
            enable_perf_tracking=True,
            log_level=args.log_level.upper(),
            rt_priority=args.fifo_priority,
            rt_lock_memory=args.lock_memory,
            rt_cpu_core=args.control_core,
        )
        service = RobotService(controller, hardware)
        service.start()

        time.sleep(max(0.0, args.warmup_s))
        service.reset_perf_stats()

        apply_rt_to_thread(
            priority=args.fifo_priority,
            policy=SCHED_FIFO,
            lock_memory=args.lock_memory,
            cpu_affinity=_cpu_affinity(args.control_core),
            quiet=True,
        )

        zero_named_commands = {
            "rotate": 0.0,
            "lift_boom": 0.0,
            "tilt_boom": 0.0,
            "scoop": 0.0,
        }
        pwm = getattr(hardware, "pwm_controller", None)
        if pwm is not None and hasattr(pwm, "build_zero_commands"):
            zero_named_commands = pwm.build_zero_commands(include_toggleable=True, include_pump=False)

        period_s = 1.0 / max(1.0, target_hz)
        miss_threshold_s = period_s * 1.01
        started = time.perf_counter()
        next_run = started
        last_sender_start = None
        last_print = started
        sequence = 0

        print(
            f"[stress] full stack start: control={target_hz:.1f}Hz imu={target_imu_hz:.1f}Hz "
            f"adc={target_hz:.1f}Hz duration={args.duration_s:.1f}s fifo={args.fifo_priority} "
            f"control_core={args.control_core} io_core={args.io_core} lock_memory={args.lock_memory}"
        )

        while (time.perf_counter() - started) < args.duration_s:
            loop_start = time.perf_counter()
            if last_sender_start is not None:
                interval_s = loop_start - last_sender_start
                sender_intervals_ms.append(interval_s * 1000.0)
                missed = interval_s > miss_threshold_s
                sender_miss_window.append(1 if missed else 0)
                if missed:
                    sender_miss_count += 1
            last_sender_start = loop_start

            zero_cmd = ControlCommand(
                sequence=sequence,
                timestamp_ms=int(time.time() * 1000) & 0xFFFFFFFF,
                mode=ControlMode.DIRECT,
                direct=DirectCommand(0.0, 0.0, 0.0, 0.0),
            )
            decoded = decode_command_message(encode_command_message(zero_cmd))
            service.submit_command(decoded)
            controller.give_direct_commands(zero_named_commands)

            telemetry = service.get_state()
            encode_telemetry_message(telemetry)
            sender_cycle_count += 1
            sequence += 1

            now = time.perf_counter()
            if now - last_print >= 1.0:
                perf_stats = service.get_debug_state().get("perf_stats", {})
                sender_hz = sender_cycle_count / max(1e-6, now - started)
                sender_recent_miss_pct = (sum(sender_miss_window) / max(1, len(sender_miss_window))) * 100.0
                ctrl_recent_miss_count = perf_stats.get('deadline_miss_1pct_count_recent', 0)
                ctrl_recent_miss_pct = perf_stats.get('deadline_miss_1pct_pct_recent', 0.0)
                p95_ms = _percentile(list(sender_intervals_ms), 0.95)
                p99_ms = _percentile(list(sender_intervals_ms), 0.99)
                print(
                    "[stress] "
                    f"sender={sender_hz:.1f}Hz miss1%={sender_miss_count} ({sender_recent_miss_pct:.1f}% recent) "
                    f"p95={p95_ms:.2f}ms p99={p99_ms:.2f}ms | "
                    f"ctrl={perf_stats.get('actual_hz', 0.0):.1f}Hz procCPU={perf_stats.get('process_cpu_pct', 0.0):.1f}% "
                    f"loopUtil={perf_stats.get('loop_util_pct', perf_stats.get('cpu_usage_pct', 0.0)):.1f}% "
                    f"miss1%={ctrl_recent_miss_count} "
                    f"({ctrl_recent_miss_pct:.1f}%/{perf_stats.get('deadline_window_sec', 0.0):.1f}s)"
                )
                last_print = now

            next_run += period_s
            sleep_s = next_run - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_run = time.perf_counter()

        perf_stats = service.get_debug_state().get("perf_stats", {})
        sender_avg_hz = sender_cycle_count / max(1e-6, time.perf_counter() - started)
        print(
            "[stress] done | "
            f"sender={sender_avg_hz:.1f}Hz cycles={sender_cycle_count} | "
            f"ctrl={perf_stats.get('actual_hz', 0.0):.1f}Hz procCPU={perf_stats.get('process_cpu_pct', 0.0):.1f}% "
            f"loopUtil={perf_stats.get('loop_util_pct', perf_stats.get('cpu_usage_pct', 0.0)):.1f}% "
            f"sender_miss1%={sender_miss_count} ({(sender_miss_count / max(1, sender_cycle_count - 1)) * 100.0:.1f}% cumulative) "
            f"ctrl_miss1%={perf_stats.get('deadline_miss_1pct_count_recent', 0)} ({perf_stats.get('deadline_miss_1pct_pct_recent', 0.0):.1f}% recent)"
        )
        return 0

    finally:
        reset_to_normal(quiet=True)
        if service is not None:
            try:
                service.stop()
            except Exception:
                pass
        elif hardware is not None:
            try:
                hardware.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
