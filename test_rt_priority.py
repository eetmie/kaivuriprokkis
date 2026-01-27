#!/usr/bin/env python3
"""
RT Priority Tester - Find stable RT settings for your system.

Tests different priority levels and scheduling modes to find the best
stable configuration. Prioritizes stability over raw performance.

Usage:
    sudo python3 test_rt_priority.py              # Run all tests
    sudo python3 test_rt_priority.py --quick      # Quick test (fewer iterations)
    sudo python3 test_rt_priority.py --priority 50  # Test specific priority
    sudo python3 test_rt_priority.py --info       # Just show system info

Safety features:
    - Watchdog thread monitors for hangs
    - Automatic timeout on each test
    - Conservative priority range (max 89, leaves room for kernel)
    - Graceful degradation if RT unavailable
"""

import argparse
import time
import threading
import signal
import sys
import os
from typing import Dict, Any, List, Optional

# Add parent dir to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.rt_utils import (
    RTConfig, apply_rt_to_thread, reset_to_normal,
    get_system_info, is_rt_capable,
    SCHED_FIFO, SCHED_RR, SCHED_OTHER
)
from modules.perf_tracker import LoopPerfTracker


# Test parameters
DEFAULT_TEST_DURATION = 3.0      # seconds per test
DEFAULT_LOOP_HZ = 100            # Target loop frequency
WATCHDOG_TIMEOUT = 10.0          # Kill test if no heartbeat for this long
WARMUP_ITERATIONS = 50           # Iterations before measuring


class TestResult:
    """Results from a single RT configuration test."""

    def __init__(self, config_name: str):
        self.config_name = config_name
        self.success = False
        self.error: Optional[str] = None
        self.hz: float = 0.0
        self.jitter_ms: float = 0.0
        self.max_latency_ms: float = 0.0
        self.violations_pct: float = 0.0
        self.samples: int = 0

    def __repr__(self) -> str:
        if not self.success:
            return f"{self.config_name}: FAILED - {self.error}"
        return (f"{self.config_name}: {self.hz:.1f}Hz, "
                f"jitter={self.jitter_ms:.3f}ms, "
                f"max={self.max_latency_ms:.3f}ms, "
                f"violations={self.violations_pct:.1f}%")

    @property
    def score(self) -> float:
        """Lower is better. Combines jitter and max latency."""
        if not self.success:
            return float('inf')
        # Weight: jitter matters more than occasional spikes
        return self.jitter_ms * 2.0 + self.max_latency_ms * 0.5 + self.violations_pct * 0.1


class WatchdogThread(threading.Thread):
    """Watchdog thread that can terminate the process if main thread hangs."""

    def __init__(self, timeout: float = WATCHDOG_TIMEOUT):
        super().__init__(daemon=True, name="Watchdog")
        self.timeout = timeout
        self.last_heartbeat = time.time()
        self.running = True
        self._lock = threading.Lock()

    def heartbeat(self):
        """Call periodically from main thread to prevent watchdog trigger."""
        with self._lock:
            self.last_heartbeat = time.time()

    def stop(self):
        """Stop the watchdog."""
        self.running = False

    def run(self):
        while self.running:
            time.sleep(0.5)
            with self._lock:
                elapsed = time.time() - self.last_heartbeat
            if elapsed > self.timeout:
                print(f"\n[WATCHDOG] No heartbeat for {elapsed:.1f}s - test may be hung!")
                print("[WATCHDOG] Attempting to reset scheduler...")
                try:
                    reset_to_normal(quiet=True)
                except Exception:
                    pass
                print("[WATCHDOG] If system is unresponsive, use: sudo kill -9", os.getpid())
                # Don't force kill - let user decide
                self.running = False


def run_timing_test(
    priority: int,
    policy: int,
    duration: float,
    target_hz: float,
    lock_memory: bool = False,
    watchdog: Optional[WatchdogThread] = None
) -> TestResult:
    """Run a timing test with the given RT configuration.

    Args:
        priority: RT priority (0 = normal scheduling)
        policy: SCHED_FIFO, SCHED_RR, or SCHED_OTHER
        duration: Test duration in seconds
        target_hz: Target loop frequency
        lock_memory: Whether to lock memory
        watchdog: Watchdog thread to heartbeat

    Returns:
        TestResult with timing statistics
    """
    policy_names = {SCHED_FIFO: 'FIFO', SCHED_RR: 'RR', SCHED_OTHER: 'NORMAL'}
    policy_name = policy_names.get(policy, '?')

    if priority > 0:
        config_name = f"RT-{policy_name}-{priority}" + (" +mlock" if lock_memory else "")
    else:
        config_name = "NORMAL (no RT)"

    result = TestResult(config_name)

    # Apply RT settings
    if priority > 0:
        success = apply_rt_to_thread(
            priority=priority,
            policy=policy,
            lock_memory=lock_memory,
            quiet=True
        )
        if not success:
            result.error = "Failed to apply RT settings (need root/CAP_SYS_NICE?)"
            return result

    try:
        tracker = LoopPerfTracker(enabled=True)
        target_period = 1.0 / target_hz
        iterations = int(duration * target_hz)

        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            if watchdog:
                watchdog.heartbeat()
            time.sleep(target_period)

        tracker.reset()

        # Test loop
        next_run = time.perf_counter()
        for i in range(iterations):
            tracker.tick_start()

            # Simulate minimal work (just timing overhead)
            _ = time.perf_counter()

            tracker.tick_end(target_period_s=target_period)

            # Heartbeat watchdog
            if watchdog and i % 10 == 0:
                watchdog.heartbeat()

            # Maintain timing
            next_run += target_period
            sleep_time = next_run - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run = time.perf_counter()

        # Collect results
        stats = tracker.get_stats()
        result.success = True
        result.hz = stats.get('hz', 0.0)
        result.jitter_ms = stats.get('loop_std_ms', 0.0)
        result.max_latency_ms = stats.get('loop_max_ms', 0.0)
        result.violations_pct = stats.get('violation_pct', 0.0)
        result.samples = stats.get('samples', 0)

    except Exception as e:
        result.error = str(e)

    finally:
        # Always reset to normal scheduling after test
        reset_to_normal(quiet=True)
        if watchdog:
            watchdog.heartbeat()

    return result


def print_system_info():
    """Print system RT capability information."""
    info = get_system_info()

    print("\n" + "=" * 60)
    print("  SYSTEM RT INFORMATION")
    print("=" * 60)

    print(f"\n  Kernel: {info.get('kernel_version', 'unknown')[:60]}")
    print(f"  RT Kernel: {'Yes' if info.get('is_rt_kernel') else 'No'}")
    print(f"  RT Capable: {'Yes' if info.get('rt_capable') else 'No'}")
    print(f"  Running as root: {'Yes' if info.get('is_root') else 'No'}")
    print(f"  CAP_SYS_NICE: {'Yes' if info.get('has_cap_sys_nice') else 'No'}")
    print(f"  CPUs available: {info.get('available_cpus', [])}")
    print(f"  CPU count: {info.get('cpu_count', '?')}")

    if not info.get('rt_capable'):
        print("\n  [!] RT scheduling not available. Run as root or set CAP_SYS_NICE:")
        print("      sudo setcap cap_sys_nice+ep $(which python3)")

    print()


def run_test_suite(
    priorities: List[int],
    duration: float,
    target_hz: float,
    test_mlock: bool = True
) -> List[TestResult]:
    """Run a suite of RT configuration tests.

    Args:
        priorities: List of priorities to test
        duration: Duration per test in seconds
        target_hz: Target loop frequency
        test_mlock: Also test with memory locking

    Returns:
        List of TestResult objects
    """
    results = []

    # Start watchdog
    watchdog = WatchdogThread()
    watchdog.start()

    print("\n" + "=" * 60)
    print("  RT PRIORITY TEST SUITE")
    print("=" * 60)
    print(f"  Target: {target_hz}Hz ({1000/target_hz:.1f}ms period)")
    print(f"  Duration per test: {duration}s")
    print(f"  Priorities to test: {priorities}")
    print()

    # Test normal (non-RT) first as baseline
    print("  Testing: NORMAL (baseline)...", end=" ", flush=True)
    result = run_timing_test(0, SCHED_OTHER, duration, target_hz, False, watchdog)
    results.append(result)
    print(f"jitter={result.jitter_ms:.3f}ms")

    # Test each priority with SCHED_FIFO
    for priority in priorities:
        print(f"  Testing: FIFO priority {priority}...", end=" ", flush=True)
        result = run_timing_test(priority, SCHED_FIFO, duration, target_hz, False, watchdog)
        results.append(result)
        if result.success:
            print(f"jitter={result.jitter_ms:.3f}ms")
        else:
            print(f"FAILED: {result.error}")

    # Test with memory locking at mid-range priority
    if test_mlock and priorities:
        mid_priority = priorities[len(priorities) // 2]
        print(f"  Testing: FIFO priority {mid_priority} + mlock...", end=" ", flush=True)
        result = run_timing_test(mid_priority, SCHED_FIFO, duration, target_hz, True, watchdog)
        results.append(result)
        if result.success:
            print(f"jitter={result.jitter_ms:.3f}ms")
        else:
            print(f"FAILED: {result.error}")

    # Stop watchdog
    watchdog.stop()

    return results


def print_results(results: List[TestResult]):
    """Print test results with recommendations."""
    print("\n" + "=" * 60)
    print("  TEST RESULTS (sorted by stability)")
    print("=" * 60)

    # Sort by score (lower is better)
    sorted_results = sorted(results, key=lambda r: r.score)

    print(f"\n  {'Config':<30} {'Hz':>6} {'Jitter':>10} {'Max':>10} {'Violations':>10}")
    print("  " + "-" * 68)

    for i, result in enumerate(sorted_results):
        if result.success:
            marker = " *" if i == 0 else "  "
            print(f"{marker}{result.config_name:<30} {result.hz:>6.1f} "
                  f"{result.jitter_ms:>9.3f}ms {result.max_latency_ms:>9.3f}ms "
                  f"{result.violations_pct:>9.1f}%")
        else:
            print(f"  {result.config_name:<30} FAILED: {result.error}")

    # Recommendation
    successful = [r for r in sorted_results if r.success]
    if successful:
        best = successful[0]
        print(f"\n  RECOMMENDATION: {best.config_name}")
        print(f"  Jitter: {best.jitter_ms:.3f}ms, Max latency: {best.max_latency_ms:.3f}ms")

        # Compare to baseline
        baseline = next((r for r in results if "NORMAL" in r.config_name and r.success), None)
        if baseline and baseline != best:
            improvement = ((baseline.jitter_ms - best.jitter_ms) / baseline.jitter_ms) * 100
            print(f"  Improvement over baseline: {improvement:.1f}% lower jitter")
    else:
        print("\n  [!] All tests failed. Check permissions and system capabilities.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test RT priority settings for stability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    sudo python3 test_rt_priority.py           # Full test suite
    sudo python3 test_rt_priority.py --quick   # Quick test
    sudo python3 test_rt_priority.py --info    # Show system info only
    sudo python3 test_rt_priority.py --priority 50 --duration 10
        """
    )
    parser.add_argument("--info", action="store_true",
                        help="Show system RT info and exit")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (shorter duration, fewer priorities)")
    parser.add_argument("--priority", type=int, default=None,
                        help="Test specific priority only")
    parser.add_argument("--duration", type=float, default=DEFAULT_TEST_DURATION,
                        help=f"Test duration in seconds (default: {DEFAULT_TEST_DURATION})")
    parser.add_argument("--hz", type=float, default=DEFAULT_LOOP_HZ,
                        help=f"Target loop frequency (default: {DEFAULT_LOOP_HZ})")
    parser.add_argument("--no-mlock", action="store_true",
                        help="Skip memory locking tests")

    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    def sigint_handler(sig, frame):
        print("\n\n[!] Interrupted - resetting scheduler...")
        reset_to_normal(quiet=True)
        sys.exit(1)
    signal.signal(signal.SIGINT, sigint_handler)

    # Always show system info
    print_system_info()

    if args.info:
        return

    if not is_rt_capable():
        print("[!] RT scheduling not available. Tests will run but RT settings won't apply.")
        print("    Run with: sudo python3 test_rt_priority.py")
        print()

    # Determine priorities to test
    if args.priority is not None:
        priorities = [args.priority]
    elif args.quick:
        priorities = [30, 50, 70]
    else:
        priorities = [20, 30, 40, 50, 60, 70, 80]

    # Run tests
    results = run_test_suite(
        priorities=priorities,
        duration=args.duration if not args.quick else 1.5,
        target_hz=args.hz,
        test_mlock=not args.no_mlock
    )

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
