#!/usr/bin/env python3
"""
ADC Speed and Jitter Performance Test

Tests:
1. Maximum read speed from 1 to 8 channels
2. Jitter analysis (timing consistency)
3. Different bit rates (12, 14, 16, 18)

Run on Raspberry Pi 5 with ADCPi board connected.
"""

import sys
import time
import statistics
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, '/home/joel/masi/modules')

from ADCPi import ADCPi


@dataclass
class SpeedResult:
    """Result from a speed test."""
    channels: int
    bit_rate: int
    iterations: int
    total_time_s: float
    reads_per_second: float
    cycles_per_second: float  # Full cycles (all channels read once)
    avg_read_time_ms: float
    min_read_time_ms: float
    max_read_time_ms: float


@dataclass
class JitterResult:
    """Result from a jitter test."""
    channels: int
    bit_rate: int
    target_hz: float
    actual_hz: float
    samples: int
    mean_interval_ms: float
    std_dev_ms: float
    min_interval_ms: float
    max_interval_ms: float
    jitter_p99_ms: float
    missed_deadlines: int
    missed_percent: float


def test_max_speed(adc: ADCPi, num_channels: int, bit_rate: int,
                   duration_s: float = 2.0) -> SpeedResult:
    """Test maximum read speed for given number of channels."""

    adc.set_bit_rate(bit_rate)
    time.sleep(0.1)  # Let settings stabilize

    read_times: List[float] = []
    iterations = 0

    start = time.perf_counter()
    deadline = start + duration_s

    while time.perf_counter() < deadline:
        cycle_start = time.perf_counter()

        # Read all channels in this test
        for ch in range(1, num_channels + 1):
            adc.read_voltage(ch)

        cycle_time = time.perf_counter() - cycle_start
        read_times.append(cycle_time)
        iterations += 1

    total_time = time.perf_counter() - start
    total_reads = iterations * num_channels

    return SpeedResult(
        channels=num_channels,
        bit_rate=bit_rate,
        iterations=iterations,
        total_time_s=total_time,
        reads_per_second=total_reads / total_time,
        cycles_per_second=iterations / total_time,
        avg_read_time_ms=statistics.mean(read_times) * 1000,
        min_read_time_ms=min(read_times) * 1000,
        max_read_time_ms=max(read_times) * 1000,
    )


def test_jitter(adc: ADCPi, num_channels: int, bit_rate: int,
                target_hz: float, duration_s: float = 5.0) -> JitterResult:
    """Test jitter at a target sample rate."""

    adc.set_bit_rate(bit_rate)
    time.sleep(0.1)

    target_interval = 1.0 / target_hz
    intervals: List[float] = []
    missed_deadlines = 0

    start = time.perf_counter()
    last_time = start
    deadline = start + duration_s
    next_sample = start + target_interval

    while time.perf_counter() < deadline:
        # Wait for next sample time
        now = time.perf_counter()
        sleep_time = next_sample - now
        if sleep_time > 0:
            time.sleep(sleep_time * 0.9)  # Sleep most of the way
            while time.perf_counter() < next_sample:
                pass  # Spin for precise timing

        sample_start = time.perf_counter()

        # Read all channels
        for ch in range(1, num_channels + 1):
            adc.read_voltage(ch)

        sample_end = time.perf_counter()

        # Record interval from last sample
        interval = sample_start - last_time
        intervals.append(interval)
        last_time = sample_start

        # Check if we missed the deadline
        if sample_end > next_sample + target_interval:
            missed_deadlines += 1

        next_sample += target_interval

    # Calculate statistics (skip first interval as it's from setup)
    if len(intervals) > 1:
        intervals = intervals[1:]

    intervals_ms = [i * 1000 for i in intervals]
    sorted_intervals = sorted(intervals_ms)
    p99_idx = int(len(sorted_intervals) * 0.99)

    return JitterResult(
        channels=num_channels,
        bit_rate=bit_rate,
        target_hz=target_hz,
        actual_hz=len(intervals) / duration_s,
        samples=len(intervals),
        mean_interval_ms=statistics.mean(intervals_ms),
        std_dev_ms=statistics.stdev(intervals_ms) if len(intervals_ms) > 1 else 0,
        min_interval_ms=min(intervals_ms),
        max_interval_ms=max(intervals_ms),
        jitter_p99_ms=sorted_intervals[p99_idx] if p99_idx < len(sorted_intervals) else sorted_intervals[-1],
        missed_deadlines=missed_deadlines,
        missed_percent=(missed_deadlines / len(intervals) * 100) if intervals else 0,
    )


def print_speed_table(results: List[SpeedResult]):
    """Print speed results as a table."""
    print("\n" + "=" * 90)
    print("MAXIMUM READ SPEED RESULTS")
    print("=" * 90)
    print(f"{'Channels':<10} {'Bit Rate':<10} {'Reads/s':<12} {'Cycles/s':<12} "
          f"{'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10}")
    print("-" * 90)

    for r in results:
        print(f"{r.channels:<10} {r.bit_rate:<10} {r.reads_per_second:<12.1f} "
              f"{r.cycles_per_second:<12.1f} {r.avg_read_time_ms:<10.2f} "
              f"{r.min_read_time_ms:<10.2f} {r.max_read_time_ms:<10.2f}")


def print_jitter_table(results: List[JitterResult]):
    """Print jitter results as a table."""
    print("\n" + "=" * 100)
    print("JITTER ANALYSIS RESULTS")
    print("=" * 100)
    print(f"{'Ch':<4} {'Bits':<6} {'Target':<8} {'Actual':<8} {'Mean':<10} "
          f"{'StdDev':<10} {'Min':<10} {'Max':<10} {'P99':<10} {'Missed':<10}")
    print(f"{'':4} {'':6} {'(Hz)':<8} {'(Hz)':<8} {'(ms)':<10} "
          f"{'(ms)':<10} {'(ms)':<10} {'(ms)':<10} {'(ms)':<10} {'(%)':10}")
    print("-" * 100)

    for r in results:
        print(f"{r.channels:<4} {r.bit_rate:<6} {r.target_hz:<8.1f} {r.actual_hz:<8.1f} "
              f"{r.mean_interval_ms:<10.2f} {r.std_dev_ms:<10.3f} {r.min_interval_ms:<10.2f} "
              f"{r.max_interval_ms:<10.2f} {r.jitter_p99_ms:<10.2f} {r.missed_percent:<10.1f}")


def main():
    print("=" * 70)
    print("ADC Speed and Jitter Performance Test")
    print("=" * 70)
    print()

    # Initialize ADC on bus 3 (software I2C)
    print("Initializing ADC on I2C bus 3...")
    try:
        adc = ADCPi(address=0x68, address2=0x69, bus=3)
        adc.set_pga(1)
        adc.set_conversion_mode(1)  # Continuous mode
        print("ADC initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize ADC: {e}")
        return 1

    # Warm up - do a few reads
    print("Warming up...")
    for _ in range(10):
        for ch in range(1, 9):
            adc.read_voltage(ch)

    speed_results: List[SpeedResult] = []
    jitter_results: List[JitterResult] = []

    # =========================================================================
    # TEST 1: Maximum speed at 12-bit (fastest mode)
    # =========================================================================
    print("\n--- TEST 1: Maximum Speed (12-bit mode) ---")
    print("Testing channels 1-8...")

    for num_channels in range(1, 9):
        print(f"  Testing {num_channels} channel(s)...", end=" ", flush=True)
        result = test_max_speed(adc, num_channels, bit_rate=12, duration_s=2.0)
        speed_results.append(result)
        print(f"{result.cycles_per_second:.1f} cycles/s, {result.reads_per_second:.1f} reads/s")

    # =========================================================================
    # TEST 2: Speed comparison across bit rates (8 channels)
    # =========================================================================
    print("\n--- TEST 2: Speed vs Bit Rate (8 channels) ---")

    for bit_rate in [12, 14, 16, 18]:
        print(f"  Testing {bit_rate}-bit mode...", end=" ", flush=True)
        result = test_max_speed(adc, num_channels=8, bit_rate=bit_rate, duration_s=2.0)
        speed_results.append(result)
        print(f"{result.cycles_per_second:.1f} cycles/s")

    # =========================================================================
    # TEST 3: Jitter at 20 Hz target (typical control loop rate)
    # =========================================================================
    print("\n--- TEST 3: Jitter Analysis at 20 Hz (12-bit) ---")

    # Only test channel counts that can achieve 20 Hz
    for num_channels in [1, 2, 4, 7, 8]:
        print(f"  Testing {num_channels} channel(s) at 20 Hz...", end=" ", flush=True)
        result = test_jitter(adc, num_channels, bit_rate=12, target_hz=20.0, duration_s=5.0)
        jitter_results.append(result)
        print(f"actual {result.actual_hz:.1f} Hz, jitter σ={result.std_dev_ms:.2f}ms")

    # =========================================================================
    # TEST 4: Jitter at different rates (8 channels, 12-bit)
    # =========================================================================
    print("\n--- TEST 4: Jitter vs Target Rate (8 channels, 12-bit) ---")

    for target_hz in [10, 15, 20, 25, 30]:
        print(f"  Testing 8 channels at {target_hz} Hz...", end=" ", flush=True)
        result = test_jitter(adc, num_channels=8, bit_rate=12, target_hz=target_hz, duration_s=3.0)
        jitter_results.append(result)
        status = "OK" if result.missed_percent < 5 else "MISSED"
        print(f"actual {result.actual_hz:.1f} Hz, missed {result.missed_percent:.1f}% [{status}]")

    # =========================================================================
    # TEST 5: Single channel high-speed jitter
    # =========================================================================
    print("\n--- TEST 5: Single Channel High-Speed (12-bit) ---")

    for target_hz in [50, 100, 150, 200]:
        print(f"  Testing 1 channel at {target_hz} Hz...", end=" ", flush=True)
        result = test_jitter(adc, num_channels=1, bit_rate=12, target_hz=target_hz, duration_s=3.0)
        jitter_results.append(result)
        status = "OK" if result.missed_percent < 5 else "MISSED"
        print(f"actual {result.actual_hz:.1f} Hz, missed {result.missed_percent:.1f}% [{status}]")

    # =========================================================================
    # Print summary tables
    # =========================================================================
    print_speed_table(speed_results)
    print_jitter_table(jitter_results)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find max achievable rates
    max_8ch_12bit = next((r for r in speed_results if r.channels == 8 and r.bit_rate == 12), None)
    max_1ch_12bit = next((r for r in speed_results if r.channels == 1 and r.bit_rate == 12), None)

    if max_8ch_12bit:
        print(f"Max 8-channel cycle rate (12-bit): {max_8ch_12bit.cycles_per_second:.1f} Hz")
    if max_1ch_12bit:
        print(f"Max 1-channel read rate (12-bit):  {max_1ch_12bit.reads_per_second:.1f} Hz")

    # Jitter summary for 8 channels at 20 Hz
    jitter_20hz_8ch = next((r for r in jitter_results
                            if r.channels == 8 and abs(r.target_hz - 20) < 1), None)
    if jitter_20hz_8ch:
        print(f"\n8-channel @ 20 Hz jitter:")
        print(f"  Mean interval: {jitter_20hz_8ch.mean_interval_ms:.2f} ms")
        print(f"  Std deviation: {jitter_20hz_8ch.std_dev_ms:.3f} ms")
        print(f"  Max interval:  {jitter_20hz_8ch.max_interval_ms:.2f} ms")
        print(f"  P99 interval:  {jitter_20hz_8ch.jitter_p99_ms:.2f} ms")

    print("\nTest complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
