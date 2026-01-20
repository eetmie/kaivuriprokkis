#!/usr/bin/env python3
"""
Maximum Speed Test for ADC and PWM Modules

Runs each module separately (no rate limiting) to determine the maximum
throughput each can achieve.

Usage:
    python speed_test.py [--duration SECONDS] [--adc-only] [--pwm-only]
"""

import argparse
import sys
import time
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from ADC import SimpleADC
from PCA9685_controller import PWMController


def test_adc_speed(duration_s: float = 5.0):
    """Test ADC maximum read speed."""
    print("\n" + "=" * 60)
    print("  ADC MAXIMUM SPEED TEST")
    print("=" * 60)

    try:
        print("Initializing ADC...")
        adc = SimpleADC(filter_alpha=1.0, log_level="WARNING")  # No filtering for raw speed
        print(f"  Bit rate: {adc.config.bit_rate} bit")
        print(f"  Channels: {len(adc.config.sensors)}")
        print(f"  Theoretical max SPS: {adc.config.get_sampling_info()['max_samples_per_second']}")
    except Exception as e:
        print(f"Failed to initialize ADC: {e}")
        return

    print(f"\nRunning for {duration_s} seconds (no rate limiting)...")

    # Warm-up read
    adc.read_sensors()

    sample_count = 0
    errors = 0
    start_time = time.perf_counter()
    end_time = start_time + duration_s

    # Track per-iteration times for analysis
    iteration_times = []
    last_time = start_time

    while time.perf_counter() < end_time:
        try:
            readings = adc.read_sensors()
            sample_count += 1

            now = time.perf_counter()
            iteration_times.append(now - last_time)
            last_time = now
        except Exception as e:
            errors += 1

    elapsed = time.perf_counter() - start_time

    # Calculate stats
    if iteration_times:
        avg_time_ms = (sum(iteration_times) / len(iteration_times)) * 1000
        min_time_ms = min(iteration_times) * 1000
        max_time_ms = max(iteration_times) * 1000
        samples_per_sec = sample_count / elapsed
    else:
        avg_time_ms = min_time_ms = max_time_ms = 0
        samples_per_sec = 0

    print("\n  RESULTS:")
    print(f"  {'-' * 40}")
    print(f"    Total samples:     {sample_count}")
    print(f"    Elapsed time:      {elapsed:.2f} s")
    print(f"    Samples/second:    {samples_per_sec:.1f}")
    print(f"    Avg time/sample:   {avg_time_ms:.2f} ms")
    print(f"    Min time/sample:   {min_time_ms:.2f} ms")
    print(f"    Max time/sample:   {max_time_ms:.2f} ms")
    print(f"    Errors:            {errors}")

    # Compare to theoretical
    theoretical_max = adc.config.get_sampling_info()['max_samples_per_second']
    # With 8 channels, theoretical max is divided by 8
    num_channels = len(adc.config.sensors)
    theoretical_full_scan = theoretical_max / num_channels
    efficiency = (samples_per_sec / theoretical_full_scan) * 100 if theoretical_full_scan > 0 else 0

    print(f"\n  ANALYSIS:")
    print(f"  {'-' * 40}")
    print(f"    Theoretical max (per channel): {theoretical_max} SPS")
    print(f"    Theoretical max (full scan):   {theoretical_full_scan:.1f} SPS")
    print(f"    Achieved efficiency:           {efficiency:.1f}%")

    return samples_per_sec


def test_pwm_speed(duration_s: float = 5.0, config_file: str = None):
    """Test PWM maximum update speed."""
    print("\n" + "=" * 60)
    print("  PWM MAXIMUM SPEED TEST")
    print("=" * 60)

    if config_file:
        config_path = Path(config_file)
    else:
        config_path = Path(__file__).parent / "configuration_files" / "servo_config.yaml"

    try:
        print("Initializing PWM Controller...")
        pwm = PWMController(
            config_file=str(config_path),
            log_level="WARNING",
            input_rate_threshold=0  # Disable rate checking
        )
        print(f"  Channels: {len(pwm.channel_configs)}")
        print(f"  PWM frequency: {pwm.pca.frequency} Hz")
    except Exception as e:
        print(f"Failed to initialize PWM Controller: {e}")
        return

    print(f"\nRunning for {duration_s} seconds (no rate limiting)...")

    # Get non-toggleable channel names for test commands
    channel_names = [name for name in pwm.channel_configs.keys()
                    if not pwm.channel_configs[name].toggleable]

    # Create test commands (alternating between two values)
    commands_a = {name: 0.1 for name in channel_names}
    commands_b = {name: -0.1 for name in channel_names}

    update_count = 0
    errors = 0
    start_time = time.perf_counter()
    end_time = start_time + duration_s

    # Track per-iteration times for analysis
    iteration_times = []
    last_time = start_time
    use_a = True

    while time.perf_counter() < end_time:
        try:
            commands = commands_a if use_a else commands_b
            pwm.update_named(commands, unset_to_zero=False)
            update_count += 1
            use_a = not use_a

            now = time.perf_counter()
            iteration_times.append(now - last_time)
            last_time = now
        except Exception as e:
            errors += 1

    elapsed = time.perf_counter() - start_time

    # Reset to safe state
    pwm.reset()

    # Calculate stats
    if iteration_times:
        avg_time_ms = (sum(iteration_times) / len(iteration_times)) * 1000
        min_time_ms = min(iteration_times) * 1000
        max_time_ms = max(iteration_times) * 1000
        updates_per_sec = update_count / elapsed
    else:
        avg_time_ms = min_time_ms = max_time_ms = 0
        updates_per_sec = 0

    print("\n  RESULTS:")
    print(f"  {'-' * 40}")
    print(f"    Total updates:     {update_count}")
    print(f"    Elapsed time:      {elapsed:.2f} s")
    print(f"    Updates/second:    {updates_per_sec:.1f}")
    print(f"    Avg time/update:   {avg_time_ms:.3f} ms")
    print(f"    Min time/update:   {min_time_ms:.3f} ms")
    print(f"    Max time/update:   {max_time_ms:.3f} ms")
    print(f"    Errors:            {errors}")

    print(f"\n  ANALYSIS:")
    print(f"  {'-' * 40}")
    print(f"    PWM period:        {1000 / pwm.pca.frequency:.2f} ms")
    print(f"    Channels updated:  {len(channel_names)}")
    print(f"    Update overhead:   {avg_time_ms * 1000:.1f} us per call")

    return updates_per_sec


def main():
    parser = argparse.ArgumentParser(
        description="Maximum speed test for ADC and PWM modules"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Test duration in seconds (default: 5)"
    )
    parser.add_argument(
        "--adc-only", action="store_true",
        help="Run only ADC speed test"
    )
    parser.add_argument(
        "--pwm-only", action="store_true",
        help="Run only PWM speed test"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="PWM config file path (default: configuration_files/servo_config.yaml)"
    )

    args = parser.parse_args()

    if args.adc_only and args.pwm_only:
        print("Error: Cannot specify both --adc-only and --pwm-only")
        sys.exit(1)

    run_adc = not args.pwm_only
    run_pwm = not args.adc_only

    print("\n" + "=" * 60)
    print("  MAXIMUM SPEED TEST")
    print("=" * 60)
    print(f"  Duration per test: {args.duration} seconds")
    print(f"  Test ADC: {run_adc}")
    print(f"  Test PWM: {run_pwm}")

    adc_speed = None
    pwm_speed = None

    if run_adc:
        adc_speed = test_adc_speed(args.duration)

    if run_pwm:
        pwm_speed = test_pwm_speed(args.duration, args.config)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if adc_speed is not None:
        print(f"    ADC max speed:  {adc_speed:.1f} full scans/second")
    if pwm_speed is not None:
        print(f"    PWM max speed:  {pwm_speed:.1f} updates/second")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
