#!/usr/bin/env python3
"""
ADC Performance Diagnostic

Analyzes where time is spent in ADC reads to identify optimization opportunities.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "modules"))

from ADC import SimpleADC, ADCConfig


def time_individual_channels():
    """Time each channel read individually."""
    print("\n" + "=" * 60)
    print("  ADC CHANNEL TIMING BREAKDOWN")
    print("=" * 60)

    adc = SimpleADC(filter_alpha=1.0, log_level="WARNING")
    print(f"\n  Bit rate: {adc.config.bit_rate}-bit")
    print(f"  Theoretical conversion time: {1000/240:.2f}ms per channel (12-bit)")
    print(f"  I2C bus: 3 (software bit-banged GPIO)")

    # Warm up
    adc.read_sensors()

    print("\n  Per-channel timing (10 samples each):\n")

    channel_times = {}

    for sensor_name, sensor_config in adc.config.sensors.items():
        board = sensor_config['input'][0]
        channel = sensor_config['input'][1]

        times = []
        for _ in range(10):
            start = time.perf_counter()
            adc.read_channel(board, channel)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        channel_times[sensor_name] = avg_time

        print(f"    Ch{channel} ({sensor_name:20}): avg={avg_time:.2f}ms  min={min_time:.2f}ms  max={max_time:.2f}ms")

    # Full scan timing
    print("\n  Full scan timing (10 samples):\n")

    scan_times = []
    for _ in range(10):
        start = time.perf_counter()
        adc.read_sensors()
        elapsed = time.perf_counter() - start
        scan_times.append(elapsed * 1000)

    avg_scan = sum(scan_times) / len(scan_times)
    sum_channels = sum(channel_times.values())

    print(f"    Full scan (all 8 channels): avg={avg_scan:.2f}ms")
    print(f"    Sum of individual channels: {sum_channels:.2f}ms")
    print(f"    Overhead per scan:          {avg_scan - sum_channels:.2f}ms")

    return channel_times, avg_scan


def test_reduced_channels():
    """Test read speed with fewer channels."""
    print("\n" + "=" * 60)
    print("  REDUCED CHANNEL CONFIGURATIONS")
    print("=" * 60)

    # Test different channel counts
    configs = [
        ("8 channels (all)", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("6 channels", [1, 2, 3, 4, 5, 6]),
        ("4 channels", [1, 2, 3, 4]),
        ("2 channels", [1, 2]),
        ("1 channel", [1]),
    ]

    adc = SimpleADC(filter_alpha=1.0, log_level="WARNING")
    board = "b1"

    print("\n  Read times for different channel counts:\n")

    for name, channels in configs:
        times = []
        for _ in range(20):
            start = time.perf_counter()
            for ch in channels:
                adc.read_channel(board, ch)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        avg_time = sum(times) / len(times)
        max_rate = 1000 / avg_time  # Hz

        print(f"    {name:20}: {avg_time:.1f}ms  -> max {max_rate:.1f} Hz")

    print("\n  Recommendation based on your 20Hz target (50ms budget):")
    print("  " + "-" * 50)

    for name, channels in configs:
        times = []
        for _ in range(10):
            start = time.perf_counter()
            for ch in channels:
                adc.read_channel(board, ch)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        avg_time = sum(times) / len(times)
        headroom = 50 - avg_time
        headroom_pct = (headroom / 50) * 100

        if headroom_pct > 30:
            status = "OK - good headroom"
        elif headroom_pct > 10:
            status = "TIGHT - minimal headroom"
        else:
            status = "FAIL - exceeds budget"

        print(f"    {name:20}: {avg_time:.1f}ms, headroom={headroom:.1f}ms ({headroom_pct:.0f}%) - {status}")


def test_bit_rates():
    """Test different ADC bit rates."""
    print("\n" + "=" * 60)
    print("  BIT RATE COMPARISON")
    print("=" * 60)

    # We can't easily change bit rate without reinitializing, so let's just show theoretical
    print("\n  MCP3424 conversion times (hardware limit):\n")

    bit_rates = [
        (12, 240, "Fast, lower precision"),
        (14, 60, "Balanced"),
        (16, 15, "High precision, slow"),
        (18, 3.75, "Highest precision, very slow"),
    ]

    print(f"    {'Bits':>4} | {'SPS':>6} | {'Per-ch':>8} | {'8-ch scan':>10} | {'Max rate':>10} | Note")
    print("    " + "-" * 70)

    for bits, sps, note in bit_rates:
        per_ch_ms = 1000 / sps
        scan_8ch_ms = per_ch_ms * 8
        max_rate = 1000 / scan_8ch_ms

        print(f"    {bits:>4} | {sps:>6} | {per_ch_ms:>6.1f}ms | {scan_8ch_ms:>8.1f}ms | {max_rate:>8.1f} Hz | {note}")

    print("\n  Note: You're already at 12-bit (fastest). The MCP3424 ADC chip")
    print("  has fixed conversion times - this is a hardware limitation.")


def suggest_optimizations():
    """Print optimization suggestions."""
    print("\n" + "=" * 60)
    print("  OPTIMIZATION OPTIONS")
    print("=" * 60)

    print("""
  1. REDUCE CHANNELS (easiest)
     - Drop channels you don't need at high rate
     - 4 channels fits comfortably in 50ms budget
     - Read less-critical sensors at lower rate (e.g., every 5th cycle)

  2. MULTI-RATE READING
     - Critical sensors: read every cycle (20 Hz)
     - Less critical: read every N cycles (4-10 Hz)
     - Example: pressure sensors at 20Hz, pump sensor at 5Hz

  3. HARDWARE I2C (if possible)
     - Currently using software I2C (bus 3, GPIO bit-banged)
     - Hardware I2C is faster but requires rewiring
     - May gain 10-20% speed improvement

  4. DIFFERENT ADC HARDWARE
     - MCP3424 is slow but precise (delta-sigma)
     - Faster options: MCP3008 (SPI, ~100kHz sample rate)
     - Would require new board/wiring

  5. INTERLEAVED READING
     - Start conversion on chip 2 while reading chip 1
     - MCP3424 supports this in continuous mode
     - Requires code changes to ADCPi library
""")


def main():
    print("\n" + "=" * 60)
    print("  ADC PERFORMANCE DIAGNOSTIC")
    print("=" * 60)

    time_individual_channels()
    test_reduced_channels()
    test_bit_rates()
    suggest_optimizations()


if __name__ == "__main__":
    main()
