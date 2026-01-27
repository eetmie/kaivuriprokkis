#!/usr/bin/env python3
"""
RT Control Loop Tester - Tests real control stack performance without waiting for goals.

Runs the actual HardwareInterface + ExcavatorController stack at different RT priority
levels to find optimal settings. Since no real servos/IMUs are connected, IK sees
static values and can never reach the goal - so we run for fixed duration and measure
timing performance.

Usage:
    sudo python3 test_rt_control_loop.py              # Full test
    sudo python3 test_rt_control_loop.py --quick      # Quick test
    sudo python3 test_rt_control_loop.py --duration 10  # Custom duration per test
"""

import argparse
import time
import sys
import os
import signal
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.hardware_interface import HardwareInterface
from modules.excavator_controller import ExcavatorController
from modules.rt_utils import (
    RTConfig, apply_rt_to_thread, reset_to_normal,
    get_system_info, is_rt_capable,
    SCHED_FIFO, SCHED_OTHER
)
from modules.perf_tracker import LoopPerfTracker


# Test parameters
DEFAULT_TEST_DURATION = 5.0   # seconds per RT config test
DEFAULT_LOOP_HZ = 100.0       # Target control loop frequency
WARMUP_DURATION = 1.0         # seconds of warmup before measuring


class ControlLoopTestResult:
    """Results from testing control loop at a specific RT configuration."""

    def __init__(self, config_name: str):
        self.config_name = config_name
        self.success = False
        self.error: Optional[str] = None

        # Main loop timing
        self.loop_hz: float = 0.0
        self.loop_jitter_ms: float = 0.0
        self.loop_max_ms: float = 0.0
        self.loop_proc_ms: float = 0.0
        self.loop_headroom_ms: float = 0.0
        self.loop_violations_pct: float = 0.0

        # Hardware subsystem timing
        self.imu_hz: float = 0.0
        self.imu_jitter_ms: float = 0.0
        self.adc_hz: float = 0.0
        self.adc_jitter_ms: float = 0.0
        self.pwm_hz: float = 0.0
        self.pwm_jitter_ms: float = 0.0

        # Controller internals
        self.ctrl_loop_hz: float = 0.0
        self.ctrl_proc_ms: float = 0.0
        self.ctrl_headroom_ms: float = 0.0

        self.samples: int = 0
        self.duration_s: float = 0.0

    @property
    def score(self) -> float:
        """Lower is better. Combines jitter metrics."""
        if not self.success:
            return float('inf')
        # Weight: loop jitter most important, then max latency, then violations
        return (self.loop_jitter_ms * 2.0 +
                self.loop_max_ms * 0.5 +
                self.loop_violations_pct * 0.1)


def run_control_loop_test(
    hardware: HardwareInterface,
    controller: ExcavatorController,
    rt_priority: int,
    duration: float,
    target_hz: float,
) -> ControlLoopTestResult:
    """Run control loop test at specified RT priority.

    Args:
        hardware: Initialized HardwareInterface
        controller: Initialized ExcavatorController
        rt_priority: RT priority (0 = normal, 1-89 = SCHED_FIFO)
        duration: Test duration in seconds
        target_hz: Target loop frequency

    Returns:
        ControlLoopTestResult with timing statistics
    """
    if rt_priority > 0:
        config_name = f"RT-FIFO-{rt_priority}"
    else:
        config_name = "NORMAL (no RT)"

    result = ControlLoopTestResult(config_name)

    # Apply RT settings to this thread
    if rt_priority > 0:
        success = apply_rt_to_thread(priority=rt_priority, policy=SCHED_FIFO, quiet=True)
        if not success:
            result.error = "Failed to apply RT settings"
            return result

    try:
        # Reset perf tracking
        hardware.reset_perf_stats()

        # Create a tracker for this test's main loop
        loop_tracker = LoopPerfTracker(enabled=True)

        target_period = 1.0 / target_hz

        # Fake target pose (doesn't matter - IK will see static sensors)
        fake_target_pos = np.array([0.3, 0.0, 0.2], dtype=np.float32)
        fake_target_rot = 0.0

        # Ensure controller is running
        controller.resume()

        # Warmup phase
        warmup_end = time.perf_counter() + WARMUP_DURATION
        next_run = time.perf_counter()
        while time.perf_counter() < warmup_end:
            controller.give_pose(fake_target_pos, fake_target_rot)
            next_run += target_period
            sleep_time = next_run - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run = time.perf_counter()

        # Reset after warmup
        loop_tracker.reset()
        hardware.reset_perf_stats()

        # Measurement phase
        test_start = time.perf_counter()
        test_end = test_start + duration
        next_run = test_start
        iterations = 0

        while time.perf_counter() < test_end:
            loop_tracker.tick_start()

            # This is the actual workload: send pose command to controller
            # Controller internally does: read IMU, FK, IK, send PWM
            controller.give_pose(fake_target_pos, fake_target_rot)

            loop_tracker.tick_end(target_period_s=target_period)
            iterations += 1

            # Maintain timing
            next_run += target_period
            sleep_time = next_run - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run = time.perf_counter()

        actual_duration = time.perf_counter() - test_start

        # Collect results
        loop_stats = loop_tracker.get_stats()
        hw_stats = hardware.get_perf_stats()
        ctrl_stats = controller.get_performance_stats()

        result.success = True
        result.duration_s = actual_duration
        result.samples = iterations

        # Main loop stats
        result.loop_hz = loop_stats.get('hz', 0.0)
        result.loop_jitter_ms = loop_stats.get('loop_std_ms', 0.0)
        result.loop_max_ms = loop_stats.get('loop_max_ms', 0.0)
        result.loop_proc_ms = loop_stats.get('proc_avg_ms', 0.0)
        result.loop_headroom_ms = loop_stats.get('headroom_avg_ms', 0.0)
        result.loop_violations_pct = loop_stats.get('violation_pct', 0.0)

        # Hardware subsystem stats
        imu = hw_stats.get('imu', {})
        adc = hw_stats.get('adc', {})
        pwm = hw_stats.get('pwm', {})

        result.imu_hz = imu.get('hz', 0.0)
        result.imu_jitter_ms = imu.get('std_interval_ms', 0.0)
        result.adc_hz = adc.get('hz', 0.0)
        result.adc_jitter_ms = adc.get('std_interval_ms', 0.0)
        result.pwm_hz = pwm.get('hz', 0.0)
        result.pwm_jitter_ms = pwm.get('loop_std_ms', 0.0)

        # Controller internal stats
        result.ctrl_loop_hz = ctrl_stats.get('avg_loop_time_ms', 0.0)
        if result.ctrl_loop_hz > 0:
            result.ctrl_loop_hz = 1000.0 / result.ctrl_loop_hz  # Convert ms to Hz
        result.ctrl_proc_ms = ctrl_stats.get('avg_loop_time_ms', 0.0)
        result.ctrl_headroom_ms = ctrl_stats.get('avg_headroom_ms', 0.0)

    except Exception as e:
        result.error = str(e)
    finally:
        # Always reset to normal scheduling
        reset_to_normal(quiet=True)
        controller.pause()

    return result


def format_results_report(
    results: List[ControlLoopTestResult],
    system_info: Dict[str, Any],
    test_params: Dict[str, Any],
) -> str:
    """Format test results as a text report."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("  RT CONTROL LOOP TEST RESULTS")
    lines.append("=" * 70)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # System info
    lines.append("-" * 70)
    lines.append("  SYSTEM INFORMATION")
    lines.append("-" * 70)
    lines.append(f"  Kernel: {system_info.get('kernel_version', 'unknown')[:60]}")
    lines.append(f"  RT Kernel: {'Yes' if system_info.get('is_rt_kernel') else 'No'}")
    lines.append(f"  RT Capable: {'Yes' if system_info.get('rt_capable') else 'No'}")
    lines.append(f"  Running as root: {'Yes' if system_info.get('is_root') else 'No'}")
    lines.append(f"  CPUs: {system_info.get('cpu_count', '?')}")
    lines.append("")

    # Test parameters
    lines.append("-" * 70)
    lines.append("  TEST PARAMETERS")
    lines.append("-" * 70)
    lines.append(f"  Target frequency: {test_params['target_hz']:.0f} Hz")
    lines.append(f"  Duration per test: {test_params['duration']:.1f} s")
    lines.append(f"  Priorities tested: {test_params['priorities']}")
    lines.append("")

    # Results table
    lines.append("-" * 70)
    lines.append("  RESULTS (sorted by stability score, lower = better)")
    lines.append("-" * 70)
    lines.append("")

    # Sort by score
    sorted_results = sorted(results, key=lambda r: r.score)

    # Header row
    lines.append(f"  {'Config':<20} {'Hz':>6} {'Jitter':>8} {'Max':>8} {'Proc':>8} {'Head':>8} {'Viol%':>6}")
    lines.append("  " + "-" * 66)

    for i, r in enumerate(sorted_results):
        if r.success:
            marker = " *" if i == 0 else "  "
            lines.append(
                f"{marker}{r.config_name:<20} {r.loop_hz:>6.1f} "
                f"{r.loop_jitter_ms:>7.3f}ms {r.loop_max_ms:>7.3f}ms "
                f"{r.loop_proc_ms:>7.3f}ms {r.loop_headroom_ms:>7.2f}ms "
                f"{r.loop_violations_pct:>5.1f}%"
            )
        else:
            lines.append(f"  {r.config_name:<20} FAILED: {r.error}")

    lines.append("")

    # Hardware subsystem details for best result
    best = next((r for r in sorted_results if r.success), None)
    if best:
        lines.append("-" * 70)
        lines.append(f"  BEST CONFIG DETAILS: {best.config_name}")
        lines.append("-" * 70)
        lines.append(f"  Main Loop:  {best.loop_hz:.1f} Hz, jitter={best.loop_jitter_ms:.3f}ms, max={best.loop_max_ms:.3f}ms")
        lines.append(f"  IMU:        {best.imu_hz:.1f} Hz, jitter={best.imu_jitter_ms:.3f}ms")
        lines.append(f"  ADC:        {best.adc_hz:.1f} Hz, jitter={best.adc_jitter_ms:.3f}ms")
        lines.append(f"  PWM:        {best.pwm_hz:.1f} Hz, jitter={best.pwm_jitter_ms:.3f}ms")
        lines.append(f"  Processing: {best.loop_proc_ms:.3f}ms avg")
        lines.append(f"  Headroom:   {best.loop_headroom_ms:.2f}ms avg")
        lines.append(f"  Violations: {best.loop_violations_pct:.1f}%")
        lines.append(f"  Samples:    {best.samples} over {best.duration_s:.1f}s")
        lines.append("")

    # Analysis and recommendations
    lines.append("-" * 70)
    lines.append("  ANALYSIS & RECOMMENDATIONS")
    lines.append("-" * 70)

    successful = [r for r in sorted_results if r.success]
    if not successful:
        lines.append("  [!] All tests failed. Check permissions (need root or CAP_SYS_NICE).")
    else:
        best = successful[0]
        baseline = next((r for r in results if "NORMAL" in r.config_name and r.success), None)

        lines.append(f"  Recommended config: {best.config_name}")
        lines.append("")

        if baseline and baseline != best:
            jitter_improvement = ((baseline.loop_jitter_ms - best.loop_jitter_ms) /
                                  baseline.loop_jitter_ms * 100) if baseline.loop_jitter_ms > 0 else 0
            max_improvement = ((baseline.loop_max_ms - best.loop_max_ms) /
                               baseline.loop_max_ms * 100) if baseline.loop_max_ms > 0 else 0
            lines.append(f"  vs baseline (NORMAL):")
            lines.append(f"    Jitter: {jitter_improvement:+.1f}% ({'better' if jitter_improvement > 0 else 'worse'})")
            lines.append(f"    Max latency: {max_improvement:+.1f}% ({'better' if max_improvement > 0 else 'worse'})")
            lines.append("")

        # Stability assessment
        if best.loop_jitter_ms < 0.5:
            lines.append("  Jitter assessment: EXCELLENT (<0.5ms)")
        elif best.loop_jitter_ms < 1.0:
            lines.append("  Jitter assessment: GOOD (<1.0ms)")
        elif best.loop_jitter_ms < 2.0:
            lines.append("  Jitter assessment: ACCEPTABLE (<2.0ms)")
        else:
            lines.append("  Jitter assessment: POOR (>2.0ms) - may affect control quality")

        # Headroom assessment
        target_period_ms = 1000.0 / test_params['target_hz']
        headroom_pct = (best.loop_headroom_ms / target_period_ms) * 100
        if headroom_pct > 80:
            lines.append(f"  Headroom assessment: EXCELLENT ({headroom_pct:.0f}% of period unused)")
        elif headroom_pct > 50:
            lines.append(f"  Headroom assessment: GOOD ({headroom_pct:.0f}% of period unused)")
        elif headroom_pct > 20:
            lines.append(f"  Headroom assessment: TIGHT ({headroom_pct:.0f}% of period unused)")
        else:
            lines.append(f"  Headroom assessment: CRITICAL ({headroom_pct:.0f}% of period unused)")

        # Violations
        if best.loop_violations_pct == 0:
            lines.append("  Timing violations: NONE")
        elif best.loop_violations_pct < 1:
            lines.append(f"  Timing violations: RARE ({best.loop_violations_pct:.2f}%)")
        else:
            lines.append(f"  Timing violations: CONCERNING ({best.loop_violations_pct:.1f}%)")

        lines.append("")

        # Code snippet
        lines.append("  To apply this config in your code:")
        lines.append("")
        if "NORMAL" in best.config_name:
            lines.append("    # No RT settings needed - default scheduling is optimal")
        else:
            priority = int(best.config_name.split("-")[-1])
            lines.append("    from modules.rt_utils import apply_rt_to_thread")
            lines.append(f"    apply_rt_to_thread(priority={priority})")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Test RT priority settings with actual control loop workload"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (shorter duration, fewer priorities)")
    parser.add_argument("--duration", type=float, default=DEFAULT_TEST_DURATION,
                        help=f"Test duration per config in seconds (default: {DEFAULT_TEST_DURATION})")
    parser.add_argument("--hz", type=float, default=DEFAULT_LOOP_HZ,
                        help=f"Target loop frequency (default: {DEFAULT_LOOP_HZ})")
    parser.add_argument("--priority", type=int, default=None,
                        help="Test specific priority only")
    parser.add_argument("--output", type=str, default="rt_test_results.txt",
                        help="Output file for results (default: rt_test_results.txt)")

    args = parser.parse_args()

    # Handle Ctrl+C
    def sigint_handler(sig, frame):
        print("\n\n[!] Interrupted - cleaning up...")
        reset_to_normal(quiet=True)
        sys.exit(1)
    signal.signal(signal.SIGINT, sigint_handler)

    # Get system info
    system_info = get_system_info()

    print("\n" + "=" * 60)
    print("  RT CONTROL LOOP TESTER")
    print("=" * 60)
    print(f"  RT Kernel: {'Yes' if system_info.get('is_rt_kernel') else 'No'}")
    print(f"  RT Capable: {'Yes' if system_info.get('rt_capable') else 'No'}")
    print(f"  Target: {args.hz:.0f}Hz, Duration: {args.duration:.1f}s per test")
    print()

    if not is_rt_capable():
        print("[!] RT scheduling not available. Run with sudo.")
        print()

    # Determine priorities to test
    if args.priority is not None:
        priorities = [0, args.priority]  # Always include baseline
    elif args.quick:
        priorities = [0, 50, 70]
    else:
        priorities = [0, 30, 50, 60, 70, 80]

    test_params = {
        'target_hz': args.hz,
        'duration': args.duration if not args.quick else 3.0,
        'priorities': priorities,
    }

    # Initialize hardware stack
    print("Initializing hardware interface...")
    try:
        hardware = HardwareInterface(
            perf_enabled=True,
            adc_channels=["Slew encoder rot"],
            adc_sample_hz=100.0,
        )
    except Exception as e:
        print(f"[!] Failed to initialize hardware: {e}")
        print("    This test requires hardware interface to be available.")
        return 1

    print("Initializing excavator controller...")
    try:
        controller = ExcavatorController(
            hardware_interface=hardware,
            enable_perf_tracking=True,
            log_level="WARNING",  # Reduce noise during test
        )
        controller.start()
    except Exception as e:
        print(f"[!] Failed to initialize controller: {e}")
        return 1

    print("Waiting for hardware to stabilize...")
    time.sleep(2.0)

    # Run tests
    results: List[ControlLoopTestResult] = []

    print()
    print("Running tests...")
    print()

    for priority in priorities:
        if priority == 0:
            desc = "NORMAL (baseline)"
        else:
            desc = f"RT-FIFO-{priority}"

        print(f"  Testing: {desc}...", end=" ", flush=True)

        result = run_control_loop_test(
            hardware=hardware,
            controller=controller,
            rt_priority=priority,
            duration=test_params['duration'],
            target_hz=test_params['target_hz'],
        )
        results.append(result)

        if result.success:
            print(f"jitter={result.loop_jitter_ms:.3f}ms, headroom={result.loop_headroom_ms:.2f}ms")
        else:
            print(f"FAILED: {result.error}")

        # Brief pause between tests
        time.sleep(0.5)

    # Cleanup
    print()
    print("Shutting down controller...")
    try:
        controller.stop()
    except Exception:
        pass

    # Generate report
    report = format_results_report(results, system_info, test_params)

    # Print summary
    print()
    print(report)

    # Save to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
