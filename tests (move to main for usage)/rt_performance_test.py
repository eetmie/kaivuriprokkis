#!/usr/bin/env python3
"""
Real-Time Performance Test for PREEMPT_RT Kernel

Tests PWM output at 200Hz and ADC reading at 20Hz while measuring:
- Jitter (deviation from target timing)
- Headroom (time remaining after each iteration)
- CPU core usage
- Latency statistics

Usage:
    python rt_performance_test.py [--duration SECONDS] [--pwm-rate HZ] [--adc-rate HZ]

For best RT performance, run with:
    sudo chrt -f 90 python rt_performance_test.py
"""

import argparse
import os
import sys
import time
import threading
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
from collections import deque

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from ADC import SimpleADC
from PCA9685_controller import PWMController


@dataclass
class LoopStats:
    """Statistics for a single control loop."""
    name: str
    target_hz: float

    # Timing data
    intervals: List[float] = field(default_factory=list)
    headrooms: List[float] = field(default_factory=list)  # Time remaining after work
    work_times: List[float] = field(default_factory=list)  # Actual work duration

    # Jitter tracking
    jitter_vals: List[float] = field(default_factory=list)

    # Error tracking
    missed_deadlines: int = 0
    total_iterations: int = 0

    @property
    def target_period(self) -> float:
        return 1.0 / self.target_hz

    def record_iteration(self, interval: float, work_time: float, headroom: float):
        """Record timing for one iteration."""
        self.intervals.append(interval)
        self.work_times.append(work_time)
        self.headrooms.append(headroom)
        self.total_iterations += 1

        # Calculate jitter (deviation from target)
        jitter = abs(interval - self.target_period)
        self.jitter_vals.append(jitter)

        # Check for missed deadline (>10% over target)
        if interval > self.target_period * 1.1:
            self.missed_deadlines += 1

    def get_stats(self) -> Dict:
        """Calculate comprehensive statistics."""
        if len(self.intervals) < 2:
            return {"error": "Not enough samples"}

        def calc_stats(data, to_us=True):
            mult = 1e6 if to_us else 1
            return {
                "mean": statistics.mean(data) * mult,
                "stdev": statistics.stdev(data) * mult,
                "min": min(data) * mult,
                "max": max(data) * mult,
                "p50": sorted(data)[len(data) // 2] * mult,
                "p95": sorted(data)[int(len(data) * 0.95)] * mult,
                "p99": sorted(data)[int(len(data) * 0.99)] * mult,
            }

        interval_stats = calc_stats(self.intervals)
        work_stats = calc_stats(self.work_times)
        headroom_stats = calc_stats(self.headrooms)
        jitter_stats = calc_stats(self.jitter_vals)

        return {
            "target_hz": self.target_hz,
            "target_period_us": self.target_period * 1e6,
            "total_iterations": self.total_iterations,
            "missed_deadlines": self.missed_deadlines,
            "missed_percent": (self.missed_deadlines / self.total_iterations * 100) if self.total_iterations > 0 else 0,
            "actual_hz": 1.0 / statistics.mean(self.intervals),
            "interval": interval_stats,
            "work_time": work_stats,
            "headroom": headroom_stats,
            "jitter": jitter_stats,
        }


class CPUMonitor:
    """Monitor CPU usage per core."""

    def __init__(self):
        self.num_cores = os.cpu_count() or 1
        self.samples: List[Dict] = []
        self._prev_times = self._read_cpu_times()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _read_cpu_times(self) -> Dict[str, List[int]]:
        """Read /proc/stat for CPU times."""
        times = {}
        try:
            with open('/proc/stat', 'r') as f:
                for line in f:
                    if line.startswith('cpu'):
                        parts = line.split()
                        name = parts[0]
                        values = [int(x) for x in parts[1:8]]  # user, nice, system, idle, iowait, irq, softirq
                        times[name] = values
        except Exception:
            pass
        return times

    def _calc_usage(self, prev: List[int], curr: List[int]) -> float:
        """Calculate CPU usage percentage."""
        prev_idle = prev[3] + prev[4]  # idle + iowait
        curr_idle = curr[3] + curr[4]

        prev_total = sum(prev)
        curr_total = sum(curr)

        total_diff = curr_total - prev_total
        idle_diff = curr_idle - prev_idle

        if total_diff == 0:
            return 0.0

        return ((total_diff - idle_diff) / total_diff) * 100

    def sample(self) -> Dict[str, float]:
        """Take a CPU usage sample."""
        curr_times = self._read_cpu_times()
        usage = {}

        for name in curr_times:
            if name in self._prev_times:
                usage[name] = self._calc_usage(self._prev_times[name], curr_times[name])

        self._prev_times = curr_times
        return usage

    def start_monitoring(self, interval: float = 0.5):
        """Start background CPU monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self._thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self._running:
            sample = self.sample()
            sample['timestamp'] = time.perf_counter()
            self.samples.append(sample)
            time.sleep(interval)

    def get_summary(self) -> Dict:
        """Get CPU usage summary."""
        if not self.samples:
            return {}

        summary = {}
        for key in self.samples[0]:
            if key == 'timestamp':
                continue
            values = [s[key] for s in self.samples if key in s]
            if values:
                summary[key] = {
                    "mean": statistics.mean(values),
                    "max": max(values),
                    "min": min(values),
                }
        return summary


class RTPerformanceTest:
    """Main real-time performance test coordinator."""

    def __init__(self, pwm_rate_hz: float = 200.0, adc_rate_hz: float = 20.0,
                 duration_s: float = 30.0, config_file: str = None):
        self.pwm_rate_hz = pwm_rate_hz
        self.adc_rate_hz = adc_rate_hz
        self.duration_s = duration_s
        self.config_file = config_file

        self.pwm_stats = LoopStats("PWM", pwm_rate_hz)
        self.adc_stats = LoopStats("ADC", adc_rate_hz)
        self.cpu_monitor = CPUMonitor()

        self.running = False
        self.start_time = 0.0

        self.adc: Optional[SimpleADC] = None
        self.pwm: Optional[PWMController] = None

        self.pwm_thread: Optional[threading.Thread] = None
        self.adc_thread: Optional[threading.Thread] = None

        # Live data for display
        self.last_adc_readings = {}
        self.last_pwm_value = 0.0

    def _init_hardware(self):
        """Initialize hardware interfaces."""
        if self.config_file:
            config_path = Path(self.config_file)
        else:
            config_path = Path(__file__).parent / "configuration_files" / "servo_config.yaml"

        print("Initializing ADC...")
        self.adc = SimpleADC(filter_alpha=1.0, log_level="WARNING")
        print(f"  ADC: {len(self.adc.config.sensors)} channels, {self.adc.config.bit_rate}-bit")

        print("Initializing PWM Controller...")
        self.pwm = PWMController(
            config_file=str(config_path),
            log_level="WARNING",
            input_rate_threshold=0
        )
        print(f"  PWM: {len(self.pwm.channel_configs)} channels")

    def _pwm_loop(self):
        """PWM control loop - runs at target rate with precise timing."""
        target_period = 1.0 / self.pwm_rate_hz

        # Get channel names for commands
        channel_names = [name for name in self.pwm.channel_configs.keys()
                        if not self.pwm.channel_configs[name].toggleable]

        # Phase for oscillating test signal
        phase = 0.0
        phase_increment = 2.0 * 3.14159 * 0.5 / self.pwm_rate_hz  # 0.5 Hz

        next_time = time.perf_counter() + target_period
        last_time = time.perf_counter()

        while self.running:
            loop_start = time.perf_counter()

            # Generate test command
            import math
            test_value = 0.3 * math.sin(phase)
            phase += phase_increment
            self.last_pwm_value = test_value

            commands = {name: test_value for name in channel_names}

            # Do the work
            work_start = time.perf_counter()
            try:
                self.pwm.update_named(commands, unset_to_zero=False)
            except Exception:
                pass
            work_end = time.perf_counter()

            work_time = work_end - work_start

            # Calculate headroom (time until next deadline)
            now = time.perf_counter()
            headroom = next_time - now

            # Record stats
            interval = now - last_time
            if last_time != loop_start:  # Skip first iteration
                self.pwm_stats.record_iteration(interval, work_time, max(0, headroom))

            last_time = now

            # Sleep until next iteration
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
                next_time += target_period
            else:
                # Missed deadline - reset timing
                next_time = time.perf_counter() + target_period

    def _adc_loop(self):
        """ADC reading loop - runs at target rate with precise timing."""
        target_period = 1.0 / self.adc_rate_hz

        next_time = time.perf_counter() + target_period
        last_time = time.perf_counter()

        while self.running:
            loop_start = time.perf_counter()

            # Do the work
            work_start = time.perf_counter()
            try:
                readings = self.adc.read_sensors()
                self.last_adc_readings = readings
            except Exception:
                pass
            work_end = time.perf_counter()

            work_time = work_end - work_start

            # Calculate headroom
            now = time.perf_counter()
            headroom = next_time - now

            # Record stats
            interval = now - last_time
            if last_time != loop_start:  # Skip first iteration
                self.adc_stats.record_iteration(interval, work_time, max(0, headroom))

            last_time = now

            # Sleep until next iteration
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
                next_time += target_period
            else:
                # Missed deadline
                next_time = time.perf_counter() + target_period

    def run(self):
        """Run the performance test."""
        print("\n" + "=" * 70)
        print("  PREEMPT_RT PERFORMANCE TEST")
        print("=" * 70)
        print(f"  PWM rate:      {self.pwm_rate_hz} Hz (period: {1e6/self.pwm_rate_hz:.0f} us)")
        print(f"  ADC rate:      {self.adc_rate_hz} Hz (period: {1e6/self.adc_rate_hz:.0f} us)")
        print(f"  Duration:      {self.duration_s} seconds")
        print(f"  CPU cores:     {self.cpu_monitor.num_cores}")
        print("=" * 70 + "\n")

        try:
            self._init_hardware()
        except Exception as e:
            print(f"Hardware initialization failed: {e}")
            return

        # Start CPU monitoring
        self.cpu_monitor.start_monitoring(interval=0.5)

        self.running = True
        self.start_time = time.perf_counter()

        # Start worker threads
        self.pwm_thread = threading.Thread(target=self._pwm_loop, name="PWM-RT")
        self.adc_thread = threading.Thread(target=self._adc_loop, name="ADC-RT")

        self.pwm_thread.start()
        self.adc_thread.start()

        # Progress display
        print("Running test... (Ctrl+C to stop early)\n")
        print(f"  {'Time':>6} | {'PWM iter':>10} | {'ADC iter':>10} | {'CPU':>6}")
        print("  " + "-" * 50)

        try:
            while self.running:
                elapsed = time.perf_counter() - self.start_time
                if elapsed >= self.duration_s:
                    break

                # Get current CPU usage
                cpu_sample = self.cpu_monitor.sample()
                cpu_total = cpu_sample.get('cpu', 0)

                print(f"\r  {elapsed:6.1f}s | {self.pwm_stats.total_iterations:10d} | "
                      f"{self.adc_stats.total_iterations:10d} | {cpu_total:5.1f}%",
                      end="", flush=True)

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")

        self.running = False
        print("\n\nStopping threads...")

        # Stop monitoring
        self.cpu_monitor.stop_monitoring()

        # Wait for threads
        if self.pwm_thread:
            self.pwm_thread.join(timeout=2.0)
        if self.adc_thread:
            self.adc_thread.join(timeout=2.0)

        # Reset PWM
        if self.pwm:
            self.pwm.reset()

        self._print_results()

    def _print_results(self):
        """Print detailed performance results."""
        print("\n" + "=" * 70)
        print("  RESULTS")
        print("=" * 70)

        for stats in [self.pwm_stats, self.adc_stats]:
            result = stats.get_stats()
            if "error" in result:
                continue

            print(f"\n  {stats.name} Loop Performance ({result['target_hz']:.0f} Hz target)")
            print("  " + "-" * 60)

            print(f"\n    Timing:")
            print(f"      Target period:     {result['target_period_us']:.0f} us")
            print(f"      Actual rate:       {result['actual_hz']:.1f} Hz")
            print(f"      Total iterations:  {result['total_iterations']}")

            iv = result['interval']
            print(f"\n    Interval (us):       mean={iv['mean']:.1f}, std={iv['stdev']:.1f}")
            print(f"                         min={iv['min']:.1f}, max={iv['max']:.1f}")
            print(f"                         p50={iv['p50']:.1f}, p95={iv['p95']:.1f}, p99={iv['p99']:.1f}")

            wt = result['work_time']
            print(f"\n    Work time (us):      mean={wt['mean']:.1f}, std={wt['stdev']:.1f}")
            print(f"                         min={wt['min']:.1f}, max={wt['max']:.1f}")

            hr = result['headroom']
            print(f"\n    Headroom (us):       mean={hr['mean']:.1f}, std={hr['stdev']:.1f}")
            print(f"                         min={hr['min']:.1f}, max={hr['max']:.1f}")
            headroom_pct = (hr['mean'] / result['target_period_us']) * 100
            print(f"                         ({headroom_pct:.1f}% of period available)")

            jt = result['jitter']
            print(f"\n    Jitter (us):         mean={jt['mean']:.1f}, max={jt['max']:.1f}")
            print(f"                         p95={jt['p95']:.1f}, p99={jt['p99']:.1f}")

            print(f"\n    Missed deadlines:    {result['missed_deadlines']} ({result['missed_percent']:.2f}%)")

        # CPU summary
        cpu_summary = self.cpu_monitor.get_summary()
        if cpu_summary:
            print(f"\n  CPU Usage")
            print("  " + "-" * 60)
            for core, usage in sorted(cpu_summary.items()):
                if core == 'cpu':
                    label = "Total"
                else:
                    label = core.upper()
                print(f"    {label:8}: mean={usage['mean']:.1f}%, max={usage['max']:.1f}%")

        # RT Quality Assessment
        print(f"\n  " + "=" * 60)
        print(f"  RT KERNEL QUALITY ASSESSMENT")
        print("  " + "=" * 60)

        for stats in [self.pwm_stats, self.adc_stats]:
            result = stats.get_stats()
            if "error" in result:
                continue

            max_jitter = result['jitter']['max']
            p99_jitter = result['jitter']['p99']
            missed = result['missed_percent']
            headroom_pct = (result['headroom']['mean'] / result['target_period_us']) * 100

            # Assess quality
            issues = []
            if max_jitter > 1000:
                issues.append(f"high max jitter ({max_jitter:.0f}us)")
            if p99_jitter > 500:
                issues.append(f"high p99 jitter ({p99_jitter:.0f}us)")
            if missed > 1.0:
                issues.append(f"missed deadlines ({missed:.1f}%)")
            if headroom_pct < 20:
                issues.append(f"low headroom ({headroom_pct:.0f}%)")

            if not issues:
                quality = "EXCELLENT"
                detail = "Suitable for hard real-time control"
            elif len(issues) == 1 and max_jitter < 2000 and missed < 1.0:
                quality = "GOOD"
                detail = "Suitable for soft real-time"
            elif missed < 5.0:
                quality = "FAIR"
                detail = f"Issues: {', '.join(issues)}"
            else:
                quality = "POOR"
                detail = f"Issues: {', '.join(issues)}"

            print(f"\n    {stats.name}: {quality}")
            print(f"         {detail}")

        print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="PREEMPT_RT performance test for PWM and ADC"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Test duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--pwm-rate", type=float, default=200.0,
        help="PWM update rate in Hz (default: 200)"
    )
    parser.add_argument(
        "--adc-rate", type=float, default=20.0,
        help="ADC read rate in Hz (default: 20)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="PWM config file path"
    )

    args = parser.parse_args()

    # Check for RT priority hint
    if os.geteuid() != 0:
        print("\nTip: For best RT performance, run with elevated priority:")
        print("  sudo chrt -f 90 python rt_performance_test.py\n")

    test = RTPerformanceTest(
        pwm_rate_hz=args.pwm_rate,
        adc_rate_hz=args.adc_rate,
        duration_s=args.duration,
        config_file=args.config
    )

    test.run()


if __name__ == "__main__":
    main()
