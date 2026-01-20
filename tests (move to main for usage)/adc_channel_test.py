#!/usr/bin/env python3
"""
Test specific ADC channel configurations for timing.
"""

import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "modules"))

from ADC import SimpleADC


def test_7_channels_at_20hz(duration_s=10.0):
    """Test 7 channels (skip encoder) at 20Hz."""
    print("\n" + "=" * 60)
    print("  7 CHANNELS AT 20Hz (skip Slew encoder rot)")
    print("=" * 60)

    adc = SimpleADC(filter_alpha=1.0, log_level="WARNING")

    # Channels to read (skip channel 8 - Slew encoder rot)
    channels_to_read = [
        ("b1", 1, "LiftBoom retract ps"),
        ("b1", 2, "LiftBoom extend ps"),
        ("b1", 3, "TiltBoom retract ps"),
        ("b1", 4, "TiltBoom extend ps"),
        ("b1", 5, "Scoop extend ps"),
        ("b1", 6, "Scoop retract ps"),
        ("b1", 7, "Pump ps"),
    ]

    print(f"\n  Reading {len(channels_to_read)} channels, skipping Ch8 (encoder)")
    print(f"  Target: 20 Hz (50ms period)")
    print(f"  Duration: {duration_s}s")

    target_period = 1.0 / 20.0  # 50ms

    # Warm up
    for board, ch, _ in channels_to_read:
        adc.read_channel(board, ch)

    intervals = []
    work_times = []
    headrooms = []

    next_time = time.perf_counter() + target_period
    last_time = time.perf_counter()
    start_time = last_time
    iterations = 0
    missed = 0

    print(f"\n  Running...")

    while time.perf_counter() - start_time < duration_s:
        work_start = time.perf_counter()

        # Read all 7 channels
        for board, ch, _ in channels_to_read:
            adc.read_channel(board, ch)

        work_end = time.perf_counter()
        work_time = work_end - work_start
        work_times.append(work_time * 1000)

        now = time.perf_counter()
        headroom = next_time - now
        headrooms.append(max(0, headroom * 1000))

        interval = now - last_time
        if iterations > 0:
            intervals.append(interval * 1000)
            if interval > target_period * 1.1:
                missed += 1

        last_time = now
        iterations += 1

        sleep_time = next_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
            next_time += target_period
        else:
            next_time = time.perf_counter() + target_period

    # Results
    avg_work = statistics.mean(work_times)
    avg_headroom = statistics.mean(headrooms)
    avg_interval = statistics.mean(intervals) if intervals else 0
    actual_hz = 1000 / avg_interval if avg_interval > 0 else 0

    print(f"\n  RESULTS:")
    print(f"  {'-' * 50}")
    print(f"    Iterations:        {iterations}")
    print(f"    Actual rate:       {actual_hz:.1f} Hz")
    print(f"    Work time:         {avg_work:.1f}ms (min={min(work_times):.1f}, max={max(work_times):.1f})")
    print(f"    Headroom:          {avg_headroom:.1f}ms ({avg_headroom/50*100:.0f}% of period)")
    print(f"    Missed deadlines:  {missed} ({missed/iterations*100:.1f}%)")

    if avg_headroom > 15:
        print(f"\n    STATUS: GOOD - comfortable headroom")
    elif avg_headroom > 5:
        print(f"\n    STATUS: TIGHT - minimal headroom")
    else:
        print(f"\n    STATUS: FAIL - insufficient headroom")


def test_single_channel_at_100hz(duration_s=10.0):
    """Test single channel at 100Hz."""
    print("\n" + "=" * 60)
    print("  SINGLE CHANNEL AT 100Hz")
    print("=" * 60)

    adc = SimpleADC(filter_alpha=1.0, log_level="WARNING")

    print(f"\n  Reading Ch1 only (LiftBoom retract ps)")
    print(f"  Target: 100 Hz (10ms period)")
    print(f"  Duration: {duration_s}s")

    target_period = 1.0 / 100.0  # 10ms

    # Warm up
    adc.read_channel("b1", 1)

    intervals = []
    work_times = []
    headrooms = []

    next_time = time.perf_counter() + target_period
    last_time = time.perf_counter()
    start_time = last_time
    iterations = 0
    missed = 0

    print(f"\n  Running...")

    while time.perf_counter() - start_time < duration_s:
        work_start = time.perf_counter()

        # Read single channel
        adc.read_channel("b1", 1)

        work_end = time.perf_counter()
        work_time = work_end - work_start
        work_times.append(work_time * 1000)

        now = time.perf_counter()
        headroom = next_time - now
        headrooms.append(max(0, headroom * 1000))

        interval = now - last_time
        if iterations > 0:
            intervals.append(interval * 1000)
            if interval > target_period * 1.1:
                missed += 1

        last_time = now
        iterations += 1

        sleep_time = next_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
            next_time += target_period
        else:
            next_time = time.perf_counter() + target_period

    # Results
    avg_work = statistics.mean(work_times)
    avg_headroom = statistics.mean(headrooms)
    avg_interval = statistics.mean(intervals) if intervals else 0
    actual_hz = 1000 / avg_interval if avg_interval > 0 else 0
    max_work = max(work_times)

    print(f"\n  RESULTS:")
    print(f"  {'-' * 50}")
    print(f"    Iterations:        {iterations}")
    print(f"    Actual rate:       {actual_hz:.1f} Hz")
    print(f"    Work time:         {avg_work:.2f}ms (min={min(work_times):.2f}, max={max_work:.2f})")
    print(f"    Headroom:          {avg_headroom:.2f}ms ({avg_headroom/10*100:.0f}% of period)")
    print(f"    Missed deadlines:  {missed} ({missed/iterations*100:.2f}%)")

    # Jitter stats
    jitter = [abs(i - 10.0) for i in intervals]
    print(f"    Jitter:            mean={statistics.mean(jitter):.2f}ms, max={max(jitter):.2f}ms")

    if avg_headroom > 5 and missed == 0:
        print(f"\n    STATUS: EXCELLENT - easily handles 100Hz")
    elif avg_headroom > 2:
        print(f"\n    STATUS: GOOD - can handle 100Hz")
    elif missed < iterations * 0.01:
        print(f"\n    STATUS: TIGHT - marginal at 100Hz")
    else:
        print(f"\n    STATUS: FAIL - cannot sustain 100Hz")


if __name__ == "__main__":
    test_7_channels_at_20hz(duration_s=10.0)
    test_single_channel_at_100hz(duration_s=10.0)
