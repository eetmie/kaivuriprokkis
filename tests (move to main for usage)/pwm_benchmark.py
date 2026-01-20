#!/usr/bin/env python3
"""
PWM Benchmark - Adafruit PCA9685 Library Performance
=====================================================
Measures baseline performance of the current Adafruit library implementation
for comparison with direct I2C optimization.

Benchmarks:
- Single channel update latency
- Multi-channel (8 channel) update latency
- Statistics: mean, P50, P95, P99, max
- GC impact assessment
"""

import gc
import time
import argparse
import statistics
from typing import List, Tuple

from adafruit_pca9685 import PCA9685
import board
import busio


def benchmark_single_channel(pca: PCA9685, iterations: int = 1000) -> List[float]:
    """Benchmark single channel duty cycle updates."""
    channel = pca.channels[0]
    latencies = []

    for i in range(iterations):
        duty = (i % 100) * 655  # 0-65500 range
        start = time.perf_counter_ns()
        channel.duty_cycle = duty
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000.0)  # Convert to microseconds

    return latencies


def benchmark_multi_channel(pca: PCA9685, num_channels: int = 8,
                            iterations: int = 1000) -> List[float]:
    """Benchmark updating multiple channels sequentially."""
    channels = [pca.channels[i] for i in range(num_channels)]
    latencies = []

    for i in range(iterations):
        duty = (i % 100) * 655
        start = time.perf_counter_ns()
        for ch in channels:
            ch.duty_cycle = duty
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000.0)

    return latencies


def benchmark_with_gc_disabled(pca: PCA9685, num_channels: int = 8,
                                iterations: int = 1000) -> List[float]:
    """Benchmark with GC disabled to isolate I2C overhead."""
    channels = [pca.channels[i] for i in range(num_channels)]
    latencies = []

    gc.collect()
    gc.disable()
    try:
        for i in range(iterations):
            duty = (i % 100) * 655
            start = time.perf_counter_ns()
            for ch in channels:
                ch.duty_cycle = duty
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1000.0)
    finally:
        gc.enable()

    return latencies


def compute_stats(latencies: List[float]) -> dict:
    """Compute statistics from latency measurements."""
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    return {
        'count': n,
        'mean': statistics.mean(latencies),
        'stdev': statistics.stdev(latencies) if n > 1 else 0,
        'min': sorted_lat[0],
        'p50': sorted_lat[int(n * 0.50)],
        'p95': sorted_lat[int(n * 0.95)],
        'p99': sorted_lat[int(n * 0.99)],
        'max': sorted_lat[-1],
    }


def print_stats(name: str, stats: dict):
    """Print formatted statistics."""
    print(f"\n{name}")
    print("=" * len(name))
    print(f"  Count:  {stats['count']}")
    print(f"  Mean:   {stats['mean']:.1f} µs")
    print(f"  Stdev:  {stats['stdev']:.1f} µs")
    print(f"  Min:    {stats['min']:.1f} µs")
    print(f"  P50:    {stats['p50']:.1f} µs")
    print(f"  P95:    {stats['p95']:.1f} µs")
    print(f"  P99:    {stats['p99']:.1f} µs")
    print(f"  Max:    {stats['max']:.1f} µs")


def estimate_max_rate(latency_us: float, num_channels: int) -> float:
    """Estimate maximum achievable update rate."""
    if latency_us <= 0:
        return float('inf')
    return 1e6 / latency_us  # Hz


def main():
    parser = argparse.ArgumentParser(description='PWM Adafruit Library Benchmark')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of iterations per test (default: 1000)')
    parser.add_argument('--channels', type=int, default=8,
                        help='Number of channels for multi-channel test (default: 8)')
    parser.add_argument('--frequency', type=int, default=200,
                        help='PWM frequency in Hz (default: 200)')
    args = parser.parse_args()

    print("PWM Benchmark - Adafruit PCA9685 Library")
    print("========================================")
    print(f"Iterations: {args.iterations}")
    print(f"Channels:   {args.channels}")
    print(f"Frequency:  {args.frequency} Hz")

    # Initialize hardware
    print("\nInitializing I2C and PCA9685...")
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = args.frequency
    print(f"Actual frequency: {pca.frequency} Hz")

    # Warmup
    print("\nWarmup (100 iterations)...")
    for i in range(100):
        pca.channels[0].duty_cycle = i * 655

    # Run benchmarks
    print("\nRunning benchmarks...")

    # Single channel
    print("  [1/4] Single channel update...")
    single_latencies = benchmark_single_channel(pca, args.iterations)
    single_stats = compute_stats(single_latencies)

    # Multi-channel with GC enabled
    print(f"  [2/4] {args.channels}-channel update (GC enabled)...")
    multi_gc_latencies = benchmark_multi_channel(pca, args.channels, args.iterations)
    multi_gc_stats = compute_stats(multi_gc_latencies)

    # Multi-channel with GC disabled
    print(f"  [3/4] {args.channels}-channel update (GC disabled)...")
    multi_nogc_latencies = benchmark_with_gc_disabled(pca, args.channels, args.iterations)
    multi_nogc_stats = compute_stats(multi_nogc_latencies)

    # Extended run for GC spike detection
    print("  [4/4] Extended run for GC spike detection (5000 iterations)...")
    extended_latencies = benchmark_multi_channel(pca, args.channels, 5000)
    extended_stats = compute_stats(extended_latencies)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print_stats("Single Channel Update", single_stats)
    max_rate_single = estimate_max_rate(single_stats['mean'], 1)
    print(f"  Max rate: {max_rate_single:.0f} Hz (theoretical)")

    print_stats(f"{args.channels}-Channel Update (GC enabled)", multi_gc_stats)
    max_rate_multi = estimate_max_rate(multi_gc_stats['mean'], args.channels)
    print(f"  Max rate: {max_rate_multi:.0f} Hz (theoretical)")
    print(f"  Per-channel: {multi_gc_stats['mean'] / args.channels:.1f} µs")

    print_stats(f"{args.channels}-Channel Update (GC disabled)", multi_nogc_stats)
    gc_overhead = multi_gc_stats['mean'] - multi_nogc_stats['mean']
    print(f"  GC overhead: {gc_overhead:.1f} µs ({gc_overhead / multi_gc_stats['mean'] * 100:.1f}%)")

    print_stats("Extended Run (5000 iter, GC enabled)", extended_stats)
    gc_spike_ratio = extended_stats['max'] / extended_stats['mean']
    print(f"  Max/Mean ratio: {gc_spike_ratio:.1f}x (GC spike indicator)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nCurrent Adafruit library performance:")
    print(f"  Single channel:     {single_stats['mean']:.0f} µs mean, {single_stats['max']:.0f} µs max")
    print(f"  {args.channels} channels:        {multi_gc_stats['mean']:.0f} µs mean, {multi_gc_stats['max']:.0f} µs max")
    print(f"  Per-channel cost:   {multi_gc_stats['mean'] / args.channels:.0f} µs")
    print(f"  GC spike factor:    {gc_spike_ratio:.1f}x max/mean")

    # Feasibility for 100Hz control loop
    time_budget_us = 10000  # 10ms = 10% of 100Hz cycle
    feasible = multi_gc_stats['p99'] < time_budget_us
    print(f"\n100Hz feasibility (10ms budget): {'YES' if feasible else 'NO'}")
    print(f"  P99 latency: {multi_gc_stats['p99']:.0f} µs")
    print(f"  Budget usage: {multi_gc_stats['p99'] / time_budget_us * 100:.1f}%")

    # Cleanup
    pca.deinit()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
