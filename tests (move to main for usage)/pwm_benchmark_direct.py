#!/usr/bin/env python3
"""
PWM Benchmark - Direct I2C Writer Performance
==============================================
Compares DirectPWMWriter (single transaction) vs Adafruit library.
"""

import gc
import time
import argparse
import statistics
from typing import List

from adafruit_pca9685 import PCA9685
import board
import busio

# Import our DirectPWMWriter
import sys
sys.path.insert(0, '/home/joel/masi/modules')
from PCA9685_controller import DirectPWMWriter


def benchmark_adafruit_multi(pca: PCA9685, num_channels: int, iterations: int) -> List[float]:
    """Benchmark Adafruit library - multiple channels sequentially."""
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


def benchmark_direct_all16(writer: DirectPWMWriter, iterations: int) -> List[float]:
    """Benchmark DirectPWMWriter - writes all 16 channels in one transaction."""
    writer.set_channel_range(0, 15)  # Full range
    latencies = []

    for i in range(iterations):
        duty = (i % 100) * 655
        start = time.perf_counter_ns()
        for ch in range(16):
            writer.set_channel(ch, duty)
        writer.flush()
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000.0)

    return latencies


def benchmark_direct_5ch(writer: DirectPWMWriter, iterations: int) -> List[float]:
    """Benchmark DirectPWMWriter - channels 2-6 (5 channels, realistic config)."""
    writer.set_channel_range(2, 6)  # scoop(2), lift(3), tilt(4), rotate(5), pump(6)
    latencies = []

    for i in range(iterations):
        duty = (i % 100) * 655
        start = time.perf_counter_ns()
        for ch in range(2, 7):  # channels 2-6
            writer.set_channel(ch, duty)
        writer.flush()
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000.0)

    return latencies


def benchmark_direct_5ch_gc_disabled(writer: DirectPWMWriter, iterations: int) -> List[float]:
    """Benchmark DirectPWMWriter 5 channels with GC disabled."""
    writer.set_channel_range(2, 6)
    latencies = []

    gc.collect()
    gc.disable()
    try:
        for i in range(iterations):
            duty = (i % 100) * 655
            start = time.perf_counter_ns()
            for ch in range(2, 7):
                writer.set_channel(ch, duty)
            writer.flush()
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1000.0)
    finally:
        gc.enable()

    return latencies


def compute_stats(latencies: List[float]) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description='PWM Direct Writer Benchmark')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of iterations per test (default: 1000)')
    parser.add_argument('--frequency', type=int, default=200,
                        help='PWM frequency in Hz (default: 200)')
    args = parser.parse_args()

    print("PWM Benchmark - Direct I2C vs Adafruit Library")
    print("=" * 50)
    print(f"Iterations: {args.iterations}")
    print(f"Frequency:  {args.frequency} Hz")

    # Initialize hardware
    print("\nInitializing I2C and PCA9685...")
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = args.frequency
    print(f"Actual frequency: {pca.frequency} Hz")

    # Create DirectPWMWriter
    writer = DirectPWMWriter(i2c)

    # Warmup both
    print("\nWarmup...")
    for i in range(100):
        pca.channels[0].duty_cycle = i * 655
    for i in range(100):
        writer.set_channel(0, i * 655)
        writer.flush()

    print("\nRunning benchmarks...")

    # Adafruit 5-channel baseline (matching your config)
    print("  [1/6] Adafruit 5-channel (baseline)...")
    adafruit_lat = benchmark_adafruit_multi(pca, 5, args.iterations)
    adafruit_stats = compute_stats(adafruit_lat)

    # Direct writer - all 16 channels
    print("  [2/6] Direct writer - 16 channels (full buffer)...")
    direct_16_lat = benchmark_direct_all16(writer, args.iterations)
    direct_16_stats = compute_stats(direct_16_lat)

    # Direct writer - 5 channels (your actual config: ch 2-6)
    print("  [3/6] Direct writer - 5 channels (ch 2-6, partial write)...")
    direct_5_lat = benchmark_direct_5ch(writer, args.iterations)
    direct_5_stats = compute_stats(direct_5_lat)

    # Direct writer - 5 channels GC disabled
    print("  [4/6] Direct writer - 5 channels (GC disabled)...")
    direct_5_nogc_lat = benchmark_direct_5ch_gc_disabled(writer, args.iterations)
    direct_5_nogc_stats = compute_stats(direct_5_nogc_lat)

    # Extended run for spike detection
    print("  [5/6] Direct writer - 5ch extended 5000 iterations...")
    direct_ext_lat = benchmark_direct_5ch(writer, 5000)
    direct_ext_stats = compute_stats(direct_ext_lat)

    # Adafruit 8-channel for comparison
    print("  [6/6] Adafruit 8-channel (for comparison)...")
    adafruit_8_lat = benchmark_adafruit_multi(pca, 8, args.iterations)
    adafruit_8_stats = compute_stats(adafruit_8_lat)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print_stats("Adafruit Library - 5 channels (baseline)", adafruit_stats)

    print_stats("Direct Writer - 16 channels (full buffer)", direct_16_stats)
    speedup_16 = adafruit_stats['mean'] / direct_16_stats['mean']
    print(f"  Speedup vs Adafruit 5ch: {speedup_16:.1f}x")

    print_stats("Direct Writer - 5 channels (partial write)", direct_5_stats)
    speedup_5 = adafruit_stats['mean'] / direct_5_stats['mean']
    print(f"  Speedup vs Adafruit 5ch: {speedup_5:.1f}x")

    print_stats("Direct Writer - 5 channels (GC disabled)", direct_5_nogc_stats)
    gc_overhead = direct_5_stats['mean'] - direct_5_nogc_stats['mean']
    print(f"  GC overhead: {gc_overhead:.1f} µs ({abs(gc_overhead) / direct_5_stats['mean'] * 100:.1f}%)")

    print_stats("Direct Writer - Extended (5000 iter)", direct_ext_stats)
    spike_ratio = direct_ext_stats['max'] / direct_ext_stats['mean']
    print(f"  Max/Mean ratio: {spike_ratio:.1f}x")

    print_stats("Adafruit Library - 8 channels (comparison)", adafruit_8_stats)

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Adafruit 5ch':>15} {'Direct 16ch':>15} {'Direct 5ch':>15}")
    print("-" * 70)
    print(f"{'Mean latency (µs)':<25} {adafruit_stats['mean']:>15.1f} {direct_16_stats['mean']:>15.1f} {direct_5_stats['mean']:>15.1f}")
    print(f"{'P99 latency (µs)':<25} {adafruit_stats['p99']:>15.1f} {direct_16_stats['p99']:>15.1f} {direct_5_stats['p99']:>15.1f}")
    print(f"{'Max latency (µs)':<25} {adafruit_stats['max']:>15.1f} {direct_16_stats['max']:>15.1f} {direct_5_stats['max']:>15.1f}")
    print(f"{'I2C bytes per write':<25} {'5 × ~8':>15} {'65':>15} {'21':>15}")
    print(f"{'Allocations per call':<25} {'5':>15} {'0':>15} {'0':>15}")

    print(f"\n** FINAL SPEEDUP: {speedup_5:.1f}x faster than Adafruit (mean) **")
    print(f"** P99 SPEEDUP:   {adafruit_stats['p99'] / direct_5_stats['p99']:.1f}x faster than Adafruit **")

    # Comparison vs 16-channel
    improvement_vs_16 = direct_16_stats['mean'] / direct_5_stats['mean']
    print(f"\nPartial write (5ch) vs full buffer (16ch): {improvement_vs_16:.2f}x faster")
    print(f"  16ch: {direct_16_stats['mean']:.0f} µs ({1 + 4*16} bytes)")
    print(f"   5ch: {direct_5_stats['mean']:.0f} µs ({1 + 4*5} bytes)")

    pca.deinit()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
