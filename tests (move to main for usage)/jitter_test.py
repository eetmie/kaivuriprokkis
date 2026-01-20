#!/usr/bin/env python3
"""
Jitter Test for RT Kernel Evaluation

Runs ADC reading and PWM control simultaneously at a set rate,
measuring timing jitter to evaluate real-time kernel performance.

Usage:
    python jitter_test.py [--rate HZ] [--duration SECONDS] [--adc-only] [--pwm-only]
"""

import argparse
import sys
import time
import threading
import statistics
import signal
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from ADC import SimpleADC
from PCA9685_controller import PWMController


@dataclass
class TimingStats:
    """Collected timing statistics for a single thread."""
    name: str
    intervals: List[float] = field(default_factory=list)
    target_interval: float = 0.0

    def add_interval(self, interval: float):
        self.intervals.append(interval)

    def compute_stats(self) -> dict:
        if len(self.intervals) < 2:
            return {"error": "Not enough samples"}

        intervals_us = [i * 1e6 for i in self.intervals]  # Convert to microseconds
        target_us = self.target_interval * 1e6

        mean = statistics.mean(intervals_us)
        stdev = statistics.stdev(intervals_us)
        min_val = min(intervals_us)
        max_val = max(intervals_us)

        # Jitter: deviation from target
        jitter_vals = [abs(i - target_us) for i in intervals_us]
        mean_jitter = statistics.mean(jitter_vals)
        max_jitter = max(jitter_vals)

        # Count missed deadlines (>10% over target)
        deadline_threshold = target_us * 1.1
        missed = sum(1 for i in intervals_us if i > deadline_threshold)

        # Percentiles
        sorted_intervals = sorted(intervals_us)
        p50 = sorted_intervals[len(sorted_intervals) // 2]
        p95 = sorted_intervals[int(len(sorted_intervals) * 0.95)]
        p99 = sorted_intervals[int(len(sorted_intervals) * 0.99)]

        return {
            "samples": len(self.intervals),
            "target_us": target_us,
            "mean_us": mean,
            "stdev_us": stdev,
            "min_us": min_val,
            "max_us": max_val,
            "mean_jitter_us": mean_jitter,
            "max_jitter_us": max_jitter,
            "p50_us": p50,
            "p95_us": p95,
            "p99_us": p99,
            "missed_deadlines": missed,
            "missed_percent": (missed / len(self.intervals)) * 100,
        }

    def print_histogram(self, bins: int = 20):
        if len(self.intervals) < 2:
            print("Not enough samples for histogram")
            return

        intervals_us = [i * 1e6 for i in self.intervals]
        min_val = min(intervals_us)
        max_val = max(intervals_us)
        bin_width = (max_val - min_val) / bins

        if bin_width == 0:
            print("All intervals identical")
            return

        histogram = [0] * bins
        for val in intervals_us:
            idx = min(int((val - min_val) / bin_width), bins - 1)
            histogram[idx] += 1

        max_count = max(histogram)
        bar_width = 40

        print(f"\n  Histogram ({self.name}):")
        print(f"  {'Range (us)':>20} | Count | Distribution")
        print("  " + "-" * 70)

        for i, count in enumerate(histogram):
            low = min_val + i * bin_width
            high = low + bin_width
            bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
            bar = "█" * bar_len
            print(f"  {low:8.1f} - {high:8.1f} | {count:5d} | {bar}")


class JitterTest:
    """Main jitter test coordinator."""

    def __init__(self, rate_hz: float, duration_s: float,
                 run_adc: bool = True, run_pwm: bool = True):
        self.rate_hz = rate_hz
        self.target_interval = 1.0 / rate_hz
        self.duration_s = duration_s
        self.run_adc = run_adc
        self.run_pwm = run_pwm

        self.running = False
        self.start_time = 0.0

        self.adc_stats = TimingStats("ADC", target_interval=self.target_interval)
        self.pwm_stats = TimingStats("PWM", target_interval=self.target_interval)

        self.adc: Optional[SimpleADC] = None
        self.pwm: Optional[PWMController] = None

        self.adc_thread: Optional[threading.Thread] = None
        self.pwm_thread: Optional[threading.Thread] = None

        # Sample data for display
        self.last_adc_readings = {}
        self.last_pwm_commands = {}

    def _init_hardware(self):
        """Initialize hardware interfaces."""
        config_path = Path(__file__).parent / "configuration_files" / "servo_config.yaml"

        if self.run_adc:
            print("Initializing ADC...")
            self.adc = SimpleADC(log_level="WARNING")
            print(f"  ADC initialized: {len(self.adc.config.sensors)} sensors")

        if self.run_pwm:
            print("Initializing PWM Controller...")
            self.pwm = PWMController(
                config_file=str(config_path),
                log_level="WARNING",
                input_rate_threshold=0  # Disable rate checking for test
            )
            print(f"  PWM initialized: {len(self.pwm.channel_configs)} channels")

    def _adc_loop(self):
        """ADC reading loop with timing measurement."""
        last_time = time.perf_counter()
        next_time = last_time + self.target_interval

        while self.running:
            # Read all sensors
            try:
                readings = self.adc.read_sensors()
                self.last_adc_readings = readings
            except Exception as e:
                pass  # Continue timing measurements even on read errors

            # Measure actual interval
            now = time.perf_counter()
            interval = now - last_time
            self.adc_stats.add_interval(interval)
            last_time = now

            # Sleep until next iteration
            next_time += self.target_interval
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Missed deadline, reset next_time
                next_time = time.perf_counter() + self.target_interval

    def _pwm_loop(self):
        """PWM control loop with timing measurement."""
        last_time = time.perf_counter()
        next_time = last_time + self.target_interval

        # Generate test commands (oscillating pattern)
        phase = 0.0
        phase_increment = 2.0 * 3.14159 * 0.5 / self.rate_hz  # 0.5 Hz oscillation

        channel_names = [name for name in self.pwm.channel_configs.keys()
                        if not self.pwm.channel_configs[name].toggleable]

        while self.running:
            # Generate oscillating test commands
            import math
            test_value = 0.3 * math.sin(phase)  # 30% amplitude sine wave
            phase += phase_increment

            commands = {name: test_value for name in channel_names}
            self.last_pwm_commands = commands

            try:
                self.pwm.update_named(commands, unset_to_zero=False)
            except Exception as e:
                pass  # Continue timing measurements even on write errors

            # Measure actual interval
            now = time.perf_counter()
            interval = now - last_time
            self.pwm_stats.add_interval(interval)
            last_time = now

            # Sleep until next iteration
            next_time += self.target_interval
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter() + self.target_interval

    def run(self):
        """Run the jitter test."""
        print(f"\n{'='*60}")
        print(f"  JITTER TEST - RT KERNEL EVALUATION")
        print(f"{'='*60}")
        print(f"  Target rate:    {self.rate_hz} Hz")
        print(f"  Target period:  {self.target_interval * 1e6:.1f} us")
        print(f"  Duration:       {self.duration_s} seconds")
        print(f"  ADC enabled:    {self.run_adc}")
        print(f"  PWM enabled:    {self.run_pwm}")
        print(f"{'='*60}\n")

        try:
            self._init_hardware()
        except Exception as e:
            print(f"Hardware initialization failed: {e}")
            return

        self.running = True
        self.start_time = time.perf_counter()

        # Start worker threads
        if self.run_adc:
            self.adc_thread = threading.Thread(target=self._adc_loop, name="ADC-Loop")
            self.adc_thread.start()

        if self.run_pwm:
            self.pwm_thread = threading.Thread(target=self._pwm_loop, name="PWM-Loop")
            self.pwm_thread.start()

        # Progress display
        print("Running test... (Ctrl+C to stop early)\n")
        try:
            while self.running:
                elapsed = time.perf_counter() - self.start_time
                if elapsed >= self.duration_s:
                    break

                # Print progress every second
                adc_count = len(self.adc_stats.intervals)
                pwm_count = len(self.pwm_stats.intervals)
                print(f"\r  Elapsed: {elapsed:6.1f}s | ADC samples: {adc_count:6d} | PWM samples: {pwm_count:6d}",
                      end="", flush=True)
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")

        self.running = False
        print("\n\nStopping threads...")

        # Wait for threads
        if self.adc_thread:
            self.adc_thread.join(timeout=2.0)
        if self.pwm_thread:
            self.pwm_thread.join(timeout=2.0)

        # Reset PWM to safe state
        if self.pwm:
            self.pwm.reset()

        self._print_results()

    def _print_results(self):
        """Print timing statistics."""
        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")

        for stats in [self.adc_stats, self.pwm_stats]:
            if len(stats.intervals) < 2:
                continue

            result = stats.compute_stats()

            print(f"\n  {stats.name} Timing Statistics:")
            print(f"  {'-'*40}")
            print(f"    Samples:           {result['samples']}")
            print(f"    Target period:     {result['target_us']:.1f} us")
            print(f"    Mean period:       {result['mean_us']:.1f} us")
            print(f"    Std deviation:     {result['stdev_us']:.1f} us")
            print(f"    Min period:        {result['min_us']:.1f} us")
            print(f"    Max period:        {result['max_us']:.1f} us")
            print(f"    Mean jitter:       {result['mean_jitter_us']:.1f} us")
            print(f"    Max jitter:        {result['max_jitter_us']:.1f} us")
            print(f"    P50 (median):      {result['p50_us']:.1f} us")
            print(f"    P95:               {result['p95_us']:.1f} us")
            print(f"    P99:               {result['p99_us']:.1f} us")
            print(f"    Missed deadlines:  {result['missed_deadlines']} ({result['missed_percent']:.2f}%)")

            stats.print_histogram()

        # RT kernel quality assessment
        print(f"\n{'='*60}")
        print(f"  RT KERNEL ASSESSMENT")
        print(f"{'='*60}")

        for stats in [self.adc_stats, self.pwm_stats]:
            if len(stats.intervals) < 2:
                continue

            result = stats.compute_stats()
            max_jitter = result['max_jitter_us']
            missed = result['missed_percent']

            if max_jitter < 100 and missed < 0.1:
                quality = "EXCELLENT - Suitable for hard real-time"
            elif max_jitter < 500 and missed < 1.0:
                quality = "GOOD - Suitable for soft real-time"
            elif max_jitter < 2000 and missed < 5.0:
                quality = "FAIR - May have occasional timing issues"
            else:
                quality = "POOR - Not suitable for real-time"

            print(f"  {stats.name}: {quality}")

        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Jitter test for RT kernel evaluation"
    )
    parser.add_argument(
        "--rate", type=float, default=100.0,
        help="Target loop rate in Hz (default: 100)"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Test duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--adc-only", action="store_true",
        help="Run only ADC timing test"
    )
    parser.add_argument(
        "--pwm-only", action="store_true",
        help="Run only PWM timing test"
    )

    args = parser.parse_args()

    run_adc = not args.pwm_only
    run_pwm = not args.adc_only

    if args.adc_only and args.pwm_only:
        print("Error: Cannot specify both --adc-only and --pwm-only")
        sys.exit(1)

    test = JitterTest(
        rate_hz=args.rate,
        duration_s=args.duration,
        run_adc=run_adc,
        run_pwm=run_pwm
    )

    test.run()


if __name__ == "__main__":
    main()
