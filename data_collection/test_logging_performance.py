#!/usr/bin/env python3
"""Performance stress test for the data logging pipeline.

MANUAL HARDWARE TEST — requires real hardware. Not a unit test.

Verifies that 100Hz logging + sine computation can keep up alongside the
controller's background loop (200Hz PWM by default). Follows the pattern
from tests/stress_full_stack.py — reads control_config.yaml for default
rates, supports --rate-hz override, RT flags, and CPU pinning.

Key constraints that are always fixed:
    - Logging: 100Hz (always)
    - PWM commands: 200Hz (controller background loop)
    - IMU/ADC/controller rates: configurable via --rate-hz

No UDP connection needed — uses synthetic zero commands.
Writes results to data_collection/test_results/.

Usage:
    python data_collection/test_logging_performance.py
    python data_collection/test_logging_performance.py --duration-s 60 --with-sine
    python data_collection/test_logging_performance.py --rate-hz 100
    python data_collection/test_logging_performance.py --fifo-priority 75 --lock-memory --control-core 2 --io-core 3
"""

import argparse
import logging
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.hardware_interface import HardwareInterface, HardwareFaultError
from modules.excavator_controller import ExcavatorController, ControllerConfig
from modules.perf_tracker import LoopPerfTracker
from modules.rt_utils import apply_rt_to_thread, reset_to_normal, SCHED_FIFO

# Import logger classes from drive_logger
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drive_logger import DataLogger, SineExcitationGenerator, JOINT_NAMES

RESULTS_DIR = Path(__file__).parent / "test_results"
LOGGING_HZ = 100  # Fixed — always 100Hz


def _percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * q))
    return ordered[idx]


def _cpu_affinity(core):
    return None if core is None else {int(core)}


def _load_rate_config():
    """Load default rates from control_config.yaml (same pattern as stress_full_stack.py)."""
    cfg_path = _ROOT / "configuration_files" / "control_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    rates = data.get("rates", {}) if isinstance(data, dict) else {}
    controller = data.get("controller", {}) if isinstance(data, dict) else {}
    imu = data.get("imu", {}) if isinstance(data, dict) else {}

    control_hz = float(rates.get("control_hz", 200.0))
    imu_hz = float(imu.get("sample_rate", control_hz))
    output_limits = (
        float(controller.get("output_limits_min", -1.0)),
        float(controller.get("output_limits_max", 1.0)),
    )
    return control_hz, imu_hz, output_limits


def main():
    parser = argparse.ArgumentParser(
        description="Logging performance stress test. "
                    "Logging is always 100Hz, PWM always 200Hz. "
                    "--rate-hz overrides controller/IMU/ADC rates.")
    parser.add_argument("--rate-hz", type=float, default=None,
                        help="Override controller/IMU/ADC rate (default: from config)")
    parser.add_argument("--duration-s", type=float, default=30.0,
                        help="Test duration in seconds (default: 30)")
    parser.add_argument("--warmup-s", type=float, default=5.0,
                        help="Warmup before measurement (default: 5)")
    parser.add_argument("--with-sine", action="store_true",
                        help="Enable sine excitation during test")
    parser.add_argument("--fifo-priority", type=int, default=75,
                        help="RT FIFO priority (default: 75)")
    parser.add_argument("--lock-memory", action="store_true")
    parser.add_argument("--control-core", type=int, default=None,
                        help="CPU core for main + control loop")
    parser.add_argument("--io-core", type=int, default=None,
                        help="CPU core for USB/IMU/ADC threads")
    parser.add_argument("--log-level", type=str, default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.WARNING))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"logging_perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Resolve rates
    config_control_hz, config_imu_hz, output_limits = _load_rate_config()
    ctrl_hz = float(args.rate_hz) if args.rate_hz is not None else config_control_hz
    imu_hz = float(args.rate_hz) if args.rate_hz is not None else config_imu_hz
    adc_hz = 20.0  # pressure ADC is always 20Hz

    results = [
        "Data Logging Performance Test",
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "[config]",
        f"duration_s: {args.duration_s}",
        f"warmup_s: {args.warmup_s}",
        f"logging_hz: {LOGGING_HZ} (fixed)",
        f"controller_hz: {ctrl_hz}",
        f"imu_hz: {imu_hz}",
        f"adc_hz: {adc_hz}",
        f"with_sine: {args.with_sine}",
        f"fifo_priority: {args.fifo_priority}",
        f"lock_memory: {args.lock_memory}",
        f"control_core: {args.control_core}",
        f"io_core: {args.io_core}",
        "",
    ]

    hardware = None
    controller = None

    # Timing trackers
    log_loop_intervals_ms = deque(maxlen=int(LOGGING_HZ * args.duration_s) + 100)
    log_sample_times_ms = deque(maxlen=int(LOGGING_HZ * args.duration_s) + 100)
    sine_compute_times_ms = deque(maxlen=int(LOGGING_HZ * args.duration_s) + 100)
    give_cmd_times_ms = deque(maxlen=int(LOGGING_HZ * args.duration_s) + 100)
    miss_count = 0
    total_cycles = 0

    try:
        # ---- Hardware (same pattern as stress_full_stack.py) ----
        print(f"[test] Initializing hardware (ctrl={ctrl_hz}Hz imu={imu_hz}Hz adc={adc_hz}Hz)...")
        hardware = HardwareInterface(
            config_file=str(_ROOT / "configuration_files" / "servo_config_200.yaml"),
            pump_auto_mode=False,
            toggle_channels=True,
            stale_timeout_s=0.5,
            adc_channels=[
                "LiftBoom retract ps", "LiftBoom extend ps",
                "TiltBoom retract ps", "TiltBoom extend ps",
                "Scoop extend ps", "Scoop retract ps", "Pump ps",
            ],
            adc_sample_hz=adc_hz,
            enable_pwm=True,
            enable_imu=True,
            enable_adc=True,
            cleanup_disable_osc=False,
            log_level=args.log_level.upper(),
            rt_lock_memory=args.lock_memory,
            usb_rt_priority=args.fifo_priority,
            imu_rt_priority=args.fifo_priority,
            adc_rt_priority=args.fifo_priority,
            usb_cpu_core=args.io_core,
            imu_cpu_core=args.io_core,
            adc_cpu_core=args.io_core,
        )

        deadline = time.time() + 15.0
        while not hardware.is_hardware_ready():
            if time.time() >= deadline:
                raise TimeoutError("Hardware not ready within 15s")
            time.sleep(0.1)
        print("[test] Hardware ready")

        # ---- Controller (runs at ctrl_hz, handles 200Hz PWM internally) ----
        print(f"[test] Starting controller at {ctrl_hz}Hz (direct mode)...")
        controller = ExcavatorController(
            hardware,
            config=ControllerConfig(output_limits=output_limits, control_frequency=ctrl_hz),
            enable_perf_tracking=True,
            log_level=args.log_level.upper(),
            rt_priority=args.fifo_priority,
            rt_lock_memory=args.lock_memory,
            rt_cpu_core=args.control_core,
        )
        controller.start()
        time.sleep(max(1.0, args.warmup_s))
        controller.enter_direct_mode()

        # ---- Logger + Sine ----
        data_logger = DataLogger()
        sine_gen = SineExcitationGenerator()
        if args.with_sine:
            sine_gen.toggle()

        data_logger.start()
        main_loop_perf = LoopPerfTracker(enabled=True)

        # RT for this thread
        if args.fifo_priority > 0:
            apply_rt_to_thread(
                priority=args.fifo_priority,
                policy=SCHED_FIFO,
                lock_memory=args.lock_memory,
                cpu_affinity=_cpu_affinity(args.control_core),
                quiet=True,
            )

        zero_cmds = {name: 0.0 for name in JOINT_NAMES}

        # Main logging loop at fixed LOGGING_HZ
        log_period = 1.0 / LOGGING_HZ
        miss_threshold = log_period * 1.05
        next_run_time = time.perf_counter()
        started = time.perf_counter()
        last_loop_start = started
        last_print = started

        print(f"[test] Running for {args.duration_s}s: logging={LOGGING_HZ}Hz "
              f"controller={ctrl_hz}Hz sine={'ON' if args.with_sine else 'OFF'}")

        while (time.perf_counter() - started) < args.duration_s:
            loop_start = time.perf_counter()
            main_loop_perf.tick_start()

            # Track loop interval
            interval = loop_start - last_loop_start
            if total_cycles > 0:
                log_loop_intervals_ms.append(interval * 1000.0)
                if interval > miss_threshold:
                    miss_count += 1
            last_loop_start = loop_start

            # Sine computation
            sine_start = time.perf_counter()
            sine_cmds = sine_gen.get_all_signals(time.perf_counter())
            sine_elapsed_ms = (time.perf_counter() - sine_start) * 1000.0
            sine_compute_times_ms.append(sine_elapsed_ms)

            # Combine + clamp
            combined = {}
            for name in JOINT_NAMES:
                combined[name] = float(np.clip(zero_cmds[name] + sine_cmds[name], -1.0, 1.0))

            # Send to controller (timed)
            cmd_start = time.perf_counter()
            controller.give_direct_commands(combined)
            give_cmd_times_ms.append((time.perf_counter() - cmd_start) * 1000.0)

            # Log sample (timed)
            log_start = time.perf_counter()
            data_logger.log_sample(
                zero_cmds, sine_cmds, combined,
                controller, hardware,
                cmd_age_s=0.0, cmd_stale=False,
                sine_enabled=sine_gen.enabled,
            )
            log_sample_times_ms.append((time.perf_counter() - log_start) * 1000.0)

            total_cycles += 1
            main_loop_perf.tick_end(target_period_s=log_period)

            # Periodic status
            now = time.perf_counter()
            if now - last_print >= 5.0:
                elapsed = now - started
                actual_hz = total_cycles / elapsed if elapsed > 0 else 0
                miss_pct = (miss_count / max(1, total_cycles - 1)) * 100.0
                p95 = _percentile(list(log_loop_intervals_ms), 0.95)
                p99 = _percentile(list(log_loop_intervals_ms), 0.99)
                log_avg = sum(log_sample_times_ms) / max(1, len(log_sample_times_ms))
                ctrl_stats = controller.get_performance_stats()
                ctrl_hz_actual = ctrl_stats.get('actual_hz', 0.0)

                print(f"[test] t={elapsed:.0f}/{args.duration_s:.0f}s | "
                      f"log={actual_hz:.1f}Hz miss={miss_pct:.1f}% "
                      f"p95={p95:.2f}ms p99={p99:.2f}ms | "
                      f"log_sample={log_avg:.3f}ms | ctrl={ctrl_hz_actual:.1f}Hz")
                last_print = now

            # Sleep to maintain LOGGING_HZ
            next_run_time += log_period
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run_time = time.perf_counter()

        # ---- Collect results ----
        wall_elapsed = time.perf_counter() - started
        actual_hz = total_cycles / wall_elapsed if wall_elapsed > 0 else 0
        miss_pct = (miss_count / max(1, total_cycles - 1)) * 100.0

        ctrl_stats = controller.get_performance_stats()
        hw_stats = hardware.get_perf_stats()
        loop_stats = main_loop_perf.get_stats()

        intervals = list(log_loop_intervals_ms)
        logs = list(log_sample_times_ms)
        sines = list(sine_compute_times_ms)
        cmds = list(give_cmd_times_ms)

        results.append("[logging_loop]")
        results.append(f"total_cycles: {total_cycles}")
        results.append(f"actual_hz: {actual_hz:.2f}")
        results.append(f"target_hz: {LOGGING_HZ}")
        results.append(f"miss_count: {miss_count}")
        results.append(f"miss_pct: {miss_pct:.2f}")
        if intervals:
            results.append(f"interval_mean_ms: {sum(intervals)/len(intervals):.3f}")
            results.append(f"interval_std_ms: {np.std(intervals):.3f}")
            results.append(f"interval_p95_ms: {_percentile(intervals, 0.95):.3f}")
            results.append(f"interval_p99_ms: {_percentile(intervals, 0.99):.3f}")
            results.append(f"interval_max_ms: {max(intervals):.3f}")
        results.append("")

        results.append("[log_sample_timing]")
        if logs:
            results.append(f"mean_ms: {sum(logs)/len(logs):.4f}")
            results.append(f"p95_ms: {_percentile(logs, 0.95):.4f}")
            results.append(f"p99_ms: {_percentile(logs, 0.99):.4f}")
            results.append(f"max_ms: {max(logs):.4f}")
        results.append("")

        results.append("[sine_compute_timing]")
        if sines:
            results.append(f"mean_ms: {sum(sines)/len(sines):.4f}")
            results.append(f"max_ms: {max(sines):.4f}")
        results.append("")

        results.append("[give_direct_commands_timing]")
        if cmds:
            results.append(f"mean_ms: {sum(cmds)/len(cmds):.4f}")
            results.append(f"max_ms: {max(cmds):.4f}")
        results.append("")

        results.append("[controller_perf]")
        for k, v in ctrl_stats.items():
            results.append(f"{k}: {v}")
        results.append("")

        results.append("[hardware_perf]")
        for k, v in hw_stats.items():
            results.append(f"{k}: {v}")
        results.append("")

        results.append(f"samples_logged: {data_logger.get_sample_count()}")

        # ---- Summary ----
        imu_actual = hw_stats.get('imu', {}).get('hz', 0) if isinstance(hw_stats.get('imu'), dict) else 0

        print(f"\n{'='*60}")
        print(f"  LOGGING PERFORMANCE TEST RESULTS")
        print(f"{'='*60}")
        print(f"  Duration: {wall_elapsed:.1f}s")
        print(f"  Logging loop: {actual_hz:.1f} Hz (target {LOGGING_HZ})")
        print(f"  Controller:   {ctrl_stats.get('actual_hz', 0):.1f} Hz (target {ctrl_hz})")
        print(f"  IMU:          {imu_actual:.1f} Hz")
        print(f"  Deadline misses: {miss_count} ({miss_pct:.1f}%)")
        if intervals:
            print(f"  Interval p95/p99/max: {_percentile(intervals, 0.95):.2f} / "
                  f"{_percentile(intervals, 0.99):.2f} / {max(intervals):.2f} ms")
        if logs:
            print(f"  log_sample() mean/p99/max: {sum(logs)/len(logs):.3f} / "
                  f"{_percentile(logs, 0.99):.3f} / {max(logs):.3f} ms")
        print(f"  Samples logged: {data_logger.get_sample_count()}")

        ok = miss_pct < 5.0 and actual_hz > (LOGGING_HZ * 0.95)
        verdict = "PASS" if ok else "FAIL"
        print(f"\n  Verdict: {verdict}")
        results.append(f"\nverdict: {verdict}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        results.append(f"\nfailure: {e}")

    finally:
        reset_to_normal(quiet=True)
        if controller is not None:
            try:
                controller.exit_direct_mode()
                controller.stop()
            except Exception:
                pass
        elif hardware is not None:
            try:
                hardware.shutdown()
            except Exception:
                pass

    results_path.write_text("\n".join(results) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
