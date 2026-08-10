#!/usr/bin/env python3
"""Performance stress test for the data logging pipeline.

MANUAL HARDWARE TEST — requires real hardware. Not a unit test.

Verifies that 100Hz logging + sine computation can keep up alongside the
controller's background loop. Follows the pattern from
tools/stress_full_stack.py — resolves the board profile, reads that profile's
control_config.yaml for default rates, supports --rate-hz override, RT flags,
and CPU pinning.

Key constraints:
    - Logging: 100Hz (always — this is what simple_drive.py's main loop runs at)
    - Controller/IMU/ADC rates: from the profile, overridable via --rate-hz
      (rpi runs the control loop at 200Hz, jetson at 100Hz)

--input selects which command source the loop polls, matching simple_drive.py:

    none   — no source; synthetic zero commands (default, needs no peripherals)
    local  — gamepad wired to this machine
    udp    — remote client over the network

Under --input local/udp the polled axes are measured and then DISCARDED: the
valves are always commanded to zero. The point is to charge the loop for the
poll and for the source's background thread (the pad's monitor thread does
blocking reads on /dev/input/event*, which competes for wakeups), not to drive
the machine from a stick nobody is watching.

Writes results to data_collection/test_results/.

Usage:
    python data_collection/test_logging_performance.py
    python data_collection/test_logging_performance.py --input local --duration-s 60
    python data_collection/test_logging_performance.py --robot jetson --duration-s 60 --with-sine
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

from modules.board import PROFILES as ROBOT_PROFILES, resolve_profile
from modules.hardware_interface import HardwareInterface, HardwareFaultError
from modules.direct_controller import DirectController
from modules.excavator_controller import ExcavatorController, ControllerConfig
from modules.perf_tracker import LoopPerfTracker
from modules.rt_utils import apply_rt_to_thread, reset_to_normal, SCHED_FIFO

from simple_drive import (
    DataLogger,
    JOINT_NAMES,
    LocalGamepadInput,
    SineExcitationGenerator,
    UDPInput,
)

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


def _load_rate_config(control_config_file: str):
    """Load default rates from the profile's control_config.yaml.

    Rates are per-profile, not universal: rpi runs the control loop at 200Hz
    and jetson at 100Hz, so reading a hardcoded path would benchmark whichever
    board's numbers happened to be baked in rather than the one under test.
    """
    cfg_path = _ROOT / control_config_file
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
                    "Logging is always 100Hz; controller/IMU/ADC rates come from "
                    "the board profile and are overridden by --rate-hz.")
    parser.add_argument("--robot", choices=[*sorted(ROBOT_PROFILES), "auto"], default="auto",
                        help="Board profile (default: auto-detect)")
    parser.add_argument("--rate-hz", type=float, default=None,
                        help="Override controller/IMU/ADC rate (default: from config)")
    parser.add_argument("--duration-s", type=float, default=30.0,
                        help="Test duration in seconds (default: 30)")
    parser.add_argument("--warmup-s", type=float, default=5.0,
                        help="Warmup before measurement (default: 5)")
    parser.add_argument("--with-sine", action="store_true",
                        help="Enable sine excitation during test")
    parser.add_argument("--input", choices=["none", "local", "udp"], default="none",
                        help="Command source to poll each cycle; axes are measured "
                             "then discarded (valves stay at zero). Default: none")
    parser.add_argument("--ip", default="0.0.0.0:8080", metavar="HOST[:PORT]",
                        help="Listen address (--input udp only)")
    parser.add_argument("--miss-tolerance-ms", type=float, default=0.5,
                        help="How late an interval may run before it counts as a "
                             "deadline miss (default: 0.5). Without RT scheduling, "
                             "sleep() granularity alone costs several hundred us, so "
                             "a tight tolerance measures the timer, not the workload.")
    parser.add_argument("--fifo-priority", type=int, default=75,
                        help="RT FIFO priority (default: 75; needs CAP_SYS_NICE / "
                             "an rtprio limit, otherwise the run falls back to SCHED_OTHER)")
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

    # Resolve profile, then rates from that profile's control config
    profile = resolve_profile(args.robot)
    config_control_hz, config_imu_hz, output_limits = _load_rate_config(
        profile['control_config_file'])
    ctrl_hz = float(args.rate_hz) if args.rate_hz is not None else config_control_hz
    imu_hz = float(args.rate_hz) if args.rate_hz is not None else config_imu_hz
    adc_hz = 20.0  # pressure ADC is always 20Hz
    adc_on = bool(profile['enable_adc'])

    results = [
        "Data Logging Performance Test",
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "[config]",
        f"profile: {profile['profile_name']}",
        f"board: {profile['board']}",
        f"pwm_i2c_bus: {profile['pwm_i2c_bus']}",
        f"enable_adc: {adc_on}",
        f"duration_s: {args.duration_s}",
        f"warmup_s: {args.warmup_s}",
        f"logging_hz: {LOGGING_HZ} (fixed)",
        f"controller_hz: {ctrl_hz}",
        f"imu_hz: {imu_hz}",
        f"adc_hz: {adc_hz}",
        f"with_sine: {args.with_sine}",
        f"input: {args.input}",
        f"fifo_priority: {args.fifo_priority}",
        f"lock_memory: {args.lock_memory}",
        f"control_core: {args.control_core}",
        f"io_core: {args.io_core}",
        "",
    ]

    hardware = None
    controller = None
    source = None

    # Timing trackers
    log_loop_intervals_ms = deque(maxlen=int(LOGGING_HZ * args.duration_s) + 100)
    log_sample_times_ms = deque(maxlen=int(LOGGING_HZ * args.duration_s) + 100)
    sine_compute_times_ms = deque(maxlen=int(LOGGING_HZ * args.duration_s) + 100)
    give_cmd_times_ms = deque(maxlen=int(LOGGING_HZ * args.duration_s) + 100)
    input_poll_times_ms = deque(maxlen=int(LOGGING_HZ * args.duration_s) + 100)
    miss_count = 0
    total_cycles = 0
    input_stale_count = 0
    # Activity tracking. A pad nobody touches leaves its monitor thread parked
    # in a blocking read, so an idle run under-measures poll cost and thread
    # contention. Record proof of movement so a quiet run can be called out.
    input_active_cycles = 0
    input_axis_peak: dict = {}
    input_buttons_seen = 0

    try:
        # ---- Hardware (same pattern as tools/stress_full_stack.py) ----
        print(f"[test] Initializing hardware (profile={profile['profile_name']} "
              f"ctrl={ctrl_hz}Hz imu={imu_hz}Hz adc={adc_hz if adc_on else 'off'})...")
        hardware = HardwareInterface(
            config_file=profile['servo_config_file'],
            control_config_file=profile['control_config_file'],
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
            enable_adc=adc_on,
            pwm_i2c_bus=profile['pwm_i2c_bus'],
            pwm_i2c_addr=profile['pwm_i2c_addr'],
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
            control_config_file=profile['control_config_file'],
        )
        controller.start()
        time.sleep(max(1.0, args.warmup_s))
        direct = DirectController(hardware)
        controller.suspend_ik_output()

        # ---- Logger + Sine ----
        # Samples are held in memory and never written: this measures the cost
        # of log_sample(), not of the CSV flush.
        data_logger = DataLogger(RESULTS_DIR)
        sine_gen = SineExcitationGenerator()
        if args.with_sine:
            sine_gen.toggle()

        # ---- Command source (opened last: the pad's monitor thread should be
        # running for the whole measured window, but not during JIT warmup) ----
        if args.input == "local":
            source = LocalGamepadInput()
        elif args.input == "udp":
            host, port = (args.ip.rsplit(":", 1)[0], int(args.ip.rsplit(":", 1)[1])) \
                if ":" in args.ip else (args.ip, 8080)
            source = UDPInput(host, port)
        if source is not None and not source.open():
            raise RuntimeError(f"--input {args.input} source failed to open")

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
        miss_threshold = log_period + args.miss_tolerance_ms / 1000.0
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

            # Poll the command source. Axes are deliberately dropped — see the
            # module docstring; this charges the loop for the poll, not for
            # driving the machine.
            if source is not None:
                poll_start = time.perf_counter()
                axes, mask = source.poll()
                input_poll_times_ms.append((time.perf_counter() - poll_start) * 1000.0)
                if not source.is_live():
                    input_stale_count += 1
                if axes:
                    moved = False
                    for name, value in axes.items():
                        peak = abs(float(value))
                        if peak > input_axis_peak.get(name, 0.0):
                            input_axis_peak[name] = peak
                        if peak > 0.02:
                            moved = True
                    if moved:
                        input_active_cycles += 1
                input_buttons_seen |= int(mask)

            # Sine computation
            sine_start = time.perf_counter()
            sine_cmds = sine_gen.get_all(time.perf_counter())
            sine_elapsed_ms = (time.perf_counter() - sine_start) * 1000.0
            sine_compute_times_ms.append(sine_elapsed_ms)

            # Combine + clamp
            combined = {}
            for name in JOINT_NAMES:
                combined[name] = float(np.clip(zero_cmds[name] + sine_cmds[name], -1.0, 1.0))

            # Send to direct controller (timed)
            cmd_start = time.perf_counter()
            direct.give_commands(combined)
            direct.send_pending()
            give_cmd_times_ms.append((time.perf_counter() - cmd_start) * 1000.0)

            # Log sample (timed)
            log_start = time.perf_counter()
            data_logger.log_sample(
                zero_cmds, sine_cmds, combined,
                controller, hardware,
                cmd_age_s=0.0, cmd_stale=False,
                sine_enabled=sine_gen.enabled,
                sine_target=sine_gen.target_name,
                sine_seed=sine_gen.seed,
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
        polls = list(input_poll_times_ms)

        results.append("[logging_loop]")
        results.append(f"total_cycles: {total_cycles}")
        results.append(f"actual_hz: {actual_hz:.2f}")
        results.append(f"target_hz: {LOGGING_HZ}")
        results.append(f"miss_count: {miss_count}")
        results.append(f"miss_pct: {miss_pct:.2f}")
        results.append(f"miss_tolerance_ms: {args.miss_tolerance_ms}")
        if intervals:
            results.append(f"max_lateness_ms: {max(intervals) - log_period * 1000.0:.3f}")
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

        results.append("[direct_send_pending_timing]")
        if cmds:
            results.append(f"mean_ms: {sum(cmds)/len(cmds):.4f}")
            results.append(f"max_ms: {max(cmds):.4f}")
        results.append("")

        results.append("[input_poll_timing]")
        results.append(f"source: {args.input}")
        if polls:
            results.append(f"mean_ms: {sum(polls)/len(polls):.4f}")
            results.append(f"p99_ms: {_percentile(polls, 0.99):.4f}")
            results.append(f"max_ms: {max(polls):.4f}")
            results.append(f"not_live_cycles: {input_stale_count}")
            active_pct = (input_active_cycles / max(1, total_cycles)) * 100.0
            results.append(f"active_cycles: {input_active_cycles} ({active_pct:.1f}%)")
            results.append(f"buttons_seen_mask: 0b{input_buttons_seen:08b}")
            for name in sorted(input_axis_peak):
                results.append(f"axis_peak_{name}: {input_axis_peak[name]:.3f}")
        results.append("")

        results.append("[controller_perf]")
        for k, v in ctrl_stats.items():
            results.append(f"{k}: {v}")
        results.append("")

        results.append("[hardware_perf]")
        for k, v in hw_stats.items():
            results.append(f"{k}: {v}")
        results.append("")

        results.append(f"samples_logged: {data_logger.n_samples()}")

        # ---- Summary ----
        imu_actual = hw_stats.get('imu', {}).get('hz', 0) if isinstance(hw_stats.get('imu'), dict) else 0

        print(f"\n{'='*60}")
        print(f"  LOGGING PERFORMANCE TEST RESULTS")
        print(f"{'='*60}")
        print(f"  Duration: {wall_elapsed:.1f}s")
        print(f"  Logging loop: {actual_hz:.1f} Hz (target {LOGGING_HZ})")
        print(f"  Controller:   {ctrl_stats.get('actual_hz', 0):.1f} Hz (target {ctrl_hz})")
        print(f"  IMU:          {imu_actual:.1f} Hz")
        print(f"  Deadline misses: {miss_count} ({miss_pct:.1f}%) "
              f"at >{log_period*1000.0 + args.miss_tolerance_ms:.2f} ms")
        if intervals:
            print(f"  Interval p95/p99/max: {_percentile(intervals, 0.95):.2f} / "
                  f"{_percentile(intervals, 0.99):.2f} / {max(intervals):.2f} ms")
            print(f"  Worst sample ran {max(intervals) - log_period*1000.0:+.2f} ms late")
        if logs:
            print(f"  log_sample() mean/p99/max: {sum(logs)/len(logs):.3f} / "
                  f"{_percentile(logs, 0.99):.3f} / {max(logs):.3f} ms")
        if polls:
            print(f"  {args.input} poll() mean/p99/max: {sum(polls)/len(polls):.3f} / "
                  f"{_percentile(polls, 0.99):.3f} / {max(polls):.3f} ms"
                  + (f"  [{input_stale_count} cycles not live]" if input_stale_count else ""))
        print(f"  Samples logged: {data_logger.n_samples()}")

        ok = miss_pct < 5.0 and actual_hz > (LOGGING_HZ * 0.95)
        verdict = "PASS" if ok else "FAIL"
        print(f"\n  Verdict: {verdict}")

        # An untouched pad parks its monitor thread in a blocking read, so the
        # poll cost above is the idle case and the result says nothing about
        # what a pad being actively driven costs.
        if source is not None and input_active_cycles == 0 and input_buttons_seen == 0:
            warning = (f"input source '{args.input}' saw no stick or button activity — "
                       f"poll cost and thread contention are UNDER-MEASURED. "
                       f"Re-run while actually working the controller.")
            print(f"  *** {warning}")
            results.append(f"warning: {warning}")
        results.append(f"\nverdict: {verdict}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        results.append(f"\nfailure: {e}")

    finally:
        reset_to_normal(quiet=True)
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        if controller is not None:
            try:
                controller.resume_ik_output()
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
