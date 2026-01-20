#!/usr/bin/env python3
"""
Robustness test for PCA9685_controller.py

Tests error handling with invalid inputs, edge cases, and weird values.
Verifies the controller doesn't crash and handles bad data gracefully.
"""

import sys
import time
import math
import traceback
import tempfile
import os
from typing import Callable, List

sys.path.insert(0, '/home/joel/masi/modules')

CONFIG_FILE = '/home/joel/masi/configuration_files/servo_config.yaml'


class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", duration_ms: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms


def run_test(name: str, test_fn: Callable, expect_exception: type = None) -> TestResult:
    """Run a single test. If expect_exception is set, test passes if that exception is raised."""
    start = time.perf_counter()
    try:
        test_fn()
        duration = (time.perf_counter() - start) * 1000
        if expect_exception:
            return TestResult(name, False, f"Expected {expect_exception.__name__} but no exception raised", duration)
        return TestResult(name, True, "OK", duration)
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        if expect_exception and isinstance(e, expect_exception):
            return TestResult(name, True, f"Correctly raised {expect_exception.__name__}", duration)
        elif expect_exception:
            return TestResult(name, False, f"Expected {expect_exception.__name__}, got {type(e).__name__}: {e}", duration)
        else:
            return TestResult(name, False, f"Unexpected {type(e).__name__}: {e}\n{traceback.format_exc()}", duration)


def print_result(result: TestResult):
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {result.name} ({result.duration_ms:.1f}ms)")
    if result.message and result.message != "OK":
        for line in result.message.split('\n')[:3]:  # Limit output
            print(f"         {line}")


def main():
    print("=" * 70)
    print("PCA9685 Controller ROBUSTNESS Test (Bad Inputs & Edge Cases)")
    print("=" * 70)
    print()

    results: List[TestResult] = []

    # Import
    from PCA9685_controller import PWMController, ChannelConfig, PumpConfig

    # Initialize controller
    print("Initializing controller...")
    controller = PWMController(CONFIG_FILE, pump_variable=True, log_level="ERROR")
    print("Controller ready.\n")

    # =========================================================================
    # SECTION 1: Crazy values in update_named()
    # =========================================================================
    print("--- SECTION 1: Extreme/Invalid Values in update_named() ---")

    def test_huge_positive():
        controller.update_named({'lift_boom': 999999999.0})
        # Should clamp to 1.0 and not crash

    results.append(run_test("Huge positive value (999999999)", test_huge_positive))
    print_result(results[-1])

    def test_huge_negative():
        controller.update_named({'lift_boom': -999999999.0})

    results.append(run_test("Huge negative value (-999999999)", test_huge_negative))
    print_result(results[-1])

    def test_infinity():
        controller.update_named({'lift_boom': float('inf'), 'tilt_boom': float('-inf')})

    results.append(run_test("Infinity values (+inf, -inf)", test_infinity))
    print_result(results[-1])

    def test_nan():
        controller.update_named({'lift_boom': float('nan')})

    results.append(run_test("NaN value", test_nan))
    print_result(results[-1])

    def test_string_value():
        controller.update_named({'lift_boom': "hello"})  # Should be caught by float() conversion

    results.append(run_test("String value ('hello')", test_string_value))
    print_result(results[-1])

    def test_none_value():
        controller.update_named({'lift_boom': None})

    results.append(run_test("None value", test_none_value))
    print_result(results[-1])

    def test_list_value():
        controller.update_named({'lift_boom': [1, 2, 3]})

    results.append(run_test("List value ([1,2,3])", test_list_value))
    print_result(results[-1])

    def test_dict_value():
        controller.update_named({'lift_boom': {'nested': 'dict'}})

    results.append(run_test("Dict value ({'nested': 'dict'})", test_dict_value))
    print_result(results[-1])

    def test_empty_string():
        controller.update_named({'lift_boom': ''})

    results.append(run_test("Empty string value", test_empty_string))
    print_result(results[-1])

    def test_boolean_values():
        controller.update_named({'lift_boom': True, 'tilt_boom': False})
        # bool is subclass of int, so True=1.0, False=0.0

    results.append(run_test("Boolean values (True/False)", test_boolean_values))
    print_result(results[-1])

    # =========================================================================
    # SECTION 2: Invalid channel names
    # =========================================================================
    print("\n--- SECTION 2: Invalid Channel Names ---")

    def test_nonexistent_channel():
        controller.update_named({'nonexistent_channel': 0.5})

    results.append(run_test("Nonexistent channel name", test_nonexistent_channel))
    print_result(results[-1])

    def test_empty_channel_name():
        controller.update_named({'': 0.5})

    results.append(run_test("Empty string channel name", test_empty_channel_name))
    print_result(results[-1])

    def test_numeric_channel_name():
        controller.update_named({123: 0.5})

    results.append(run_test("Numeric channel name (123)", test_numeric_channel_name))
    print_result(results[-1])

    def test_none_channel_name():
        controller.update_named({None: 0.5})

    results.append(run_test("None as channel name", test_none_channel_name))
    print_result(results[-1])

    def test_special_chars_channel():
        controller.update_named({'lift_boom\x00injection': 0.5})

    results.append(run_test("Null byte in channel name", test_special_chars_channel))
    print_result(results[-1])

    def test_unicode_channel():
        controller.update_named({'升降臂': 0.5, '🚀': 0.3})

    results.append(run_test("Unicode channel names", test_unicode_channel))
    print_result(results[-1])

    # =========================================================================
    # SECTION 3: Empty/weird command dicts
    # =========================================================================
    print("\n--- SECTION 3: Empty/Weird Command Dictionaries ---")

    def test_empty_dict():
        controller.update_named({})

    results.append(run_test("Empty command dict", test_empty_dict))
    print_result(results[-1])

    def test_all_invalid_channels():
        controller.update_named({'foo': 1.0, 'bar': -1.0, 'baz': 0.5})

    results.append(run_test("All invalid channel names", test_all_invalid_channels))
    print_result(results[-1])

    def test_mixed_valid_invalid():
        controller.update_named({'lift_boom': 0.3, 'invalid': 0.5, 'tilt_boom': -0.2})

    results.append(run_test("Mixed valid/invalid channels", test_mixed_valid_invalid))
    print_result(results[-1])

    def test_huge_dict():
        # 1000 fake channels
        commands = {f'channel_{i}': 0.5 for i in range(1000)}
        commands['lift_boom'] = 0.1  # One valid
        controller.update_named(commands)

    results.append(run_test("Huge dict (1000 fake + 1 valid)", test_huge_dict))
    print_result(results[-1])

    # =========================================================================
    # SECTION 4: Pump edge cases
    # =========================================================================
    print("\n--- SECTION 4: Pump Edge Cases ---")

    def test_pump_huge_value():
        controller.update_named({'pump': 1000000.0})

    results.append(run_test("Pump huge value (1000000)", test_pump_huge_value))
    print_result(results[-1])

    def test_pump_negative_huge():
        controller.update_named({'pump': -1000000.0})

    results.append(run_test("Pump huge negative (-1000000)", test_pump_negative_huge))
    print_result(results[-1])

    def test_pump_string():
        controller.update_named({'pump': 'fast'})

    results.append(run_test("Pump string value ('fast')", test_pump_string))
    print_result(results[-1])

    def test_pump_load_huge():
        for _ in range(100):
            controller.update_pump_load(999999)
        # Should be clamped
        assert controller.manual_pump_load <= 0.3
        controller.reset_pump_load()

    results.append(run_test("Pump load huge adjustment (100x)", test_pump_load_huge))
    print_result(results[-1])

    def test_pump_load_negative_huge():
        for _ in range(100):
            controller.update_pump_load(-999999)
        assert controller.manual_pump_load >= -1.0
        controller.reset_pump_load()

    results.append(run_test("Pump load huge negative (100x)", test_pump_load_negative_huge))
    print_result(results[-1])

    # =========================================================================
    # SECTION 5: compute_pulse() edge cases
    # =========================================================================
    print("\n--- SECTION 5: compute_pulse() Edge Cases ---")

    def test_compute_invalid_channel():
        result = controller.compute_pulse('does_not_exist', 0.5)
        assert result is None

    results.append(run_test("compute_pulse() invalid channel", test_compute_invalid_channel))
    print_result(results[-1])

    def test_compute_huge_value():
        result = controller.compute_pulse('lift_boom', 99999.0)
        cfg = controller.channel_configs['lift_boom']
        # Should clamp and return valid pulse
        assert cfg.pulse_min <= result <= cfg.pulse_max + 50  # +50 for dither headroom

    results.append(run_test("compute_pulse() huge value", test_compute_huge_value))
    print_result(results[-1])

    def test_compute_nan():
        result = controller.compute_pulse('lift_boom', float('nan'))
        # NaN behavior - may return NaN or handle it

    results.append(run_test("compute_pulse() NaN", test_compute_nan))
    print_result(results[-1])

    def test_compute_negative_time():
        result = controller.compute_pulse('lift_boom', 0.5, now=-1000.0)
        # Negative time shouldn't crash

    results.append(run_test("compute_pulse() negative timestamp", test_compute_negative_time))
    print_result(results[-1])

    # =========================================================================
    # SECTION 6: Invalid config files
    # =========================================================================
    print("\n--- SECTION 6: Invalid Configuration Files ---")

    def test_nonexistent_config():
        PWMController('/nonexistent/path/config.yaml')

    results.append(run_test("Nonexistent config file", test_nonexistent_config, expect_exception=FileNotFoundError))
    print_result(results[-1])

    def test_empty_config():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            PWMController(temp_path)
        finally:
            os.unlink(temp_path)

    results.append(run_test("Empty config file", test_empty_config, expect_exception=Exception))
    print_result(results[-1])

    def test_invalid_yaml():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("{{{{not valid yaml::::")
            temp_path = f.name
        try:
            PWMController(temp_path)
        finally:
            os.unlink(temp_path)

    results.append(run_test("Invalid YAML syntax", test_invalid_yaml, expect_exception=Exception))
    print_result(results[-1])

    def test_missing_required_fields():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
CHANNEL_CONFIGS:
  test_channel:
    output_channel: 0
    # Missing pulse_min, pulse_max, direction, deadband_us_pos, deadband_us_neg
""")
            temp_path = f.name
        try:
            PWMController(temp_path)
        finally:
            os.unlink(temp_path)

    results.append(run_test("Config missing required fields", test_missing_required_fields, expect_exception=Exception))
    print_result(results[-1])

    def test_invalid_direction():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
CHANNEL_CONFIGS:
  test_channel:
    output_channel: 0
    pulse_min: 1000
    pulse_max: 2000
    direction: 5
    deadband_us_pos: 50.0
    deadband_us_neg: 50.0
""")
            temp_path = f.name
        try:
            PWMController(temp_path)
        finally:
            os.unlink(temp_path)

    results.append(run_test("Config invalid direction (5)", test_invalid_direction, expect_exception=ValueError))
    print_result(results[-1])

    def test_duplicate_output_channel():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
CHANNEL_CONFIGS:
  channel_a:
    output_channel: 3
    pulse_min: 1000
    pulse_max: 2000
    direction: 1
    deadband_us_pos: 50.0
    deadband_us_neg: 50.0
  channel_b:
    output_channel: 3
    pulse_min: 1000
    pulse_max: 2000
    direction: -1
    deadband_us_pos: 50.0
    deadband_us_neg: 50.0
""")
            temp_path = f.name
        try:
            PWMController(temp_path)
        finally:
            os.unlink(temp_path)

    results.append(run_test("Config duplicate output channel", test_duplicate_output_channel, expect_exception=ValueError))
    print_result(results[-1])

    def test_pulse_min_gt_max():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
CHANNEL_CONFIGS:
  test_channel:
    output_channel: 0
    pulse_min: 2000
    pulse_max: 1000
    direction: 1
    deadband_us_pos: 50.0
    deadband_us_neg: 50.0
""")
            temp_path = f.name
        try:
            PWMController(temp_path)
        finally:
            os.unlink(temp_path)

    results.append(run_test("Config pulse_min > pulse_max", test_pulse_min_gt_max, expect_exception=ValueError))
    print_result(results[-1])

    def test_output_channel_out_of_range():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
CHANNEL_CONFIGS:
  test_channel:
    output_channel: 99
    pulse_min: 1000
    pulse_max: 2000
    direction: 1
    deadband_us_pos: 50.0
    deadband_us_neg: 50.0
""")
            temp_path = f.name
        try:
            PWMController(temp_path)
        finally:
            os.unlink(temp_path)

    results.append(run_test("Config output_channel=99 (out of range)", test_output_channel_out_of_range, expect_exception=ValueError))
    print_result(results[-1])

    def test_negative_deadband():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
CHANNEL_CONFIGS:
  test_channel:
    output_channel: 0
    pulse_min: 1000
    pulse_max: 2000
    direction: 1
    deadband_us_pos: -50.0
    deadband_us_neg: 50.0
""")
            temp_path = f.name
        try:
            PWMController(temp_path)
        finally:
            os.unlink(temp_path)

    results.append(run_test("Config negative deadband", test_negative_deadband, expect_exception=ValueError))
    print_result(results[-1])

    def test_huge_deadband():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
CHANNEL_CONFIGS:
  test_channel:
    output_channel: 0
    pulse_min: 1000
    pulse_max: 2000
    direction: 1
    deadband_us_pos: 9999.0
    deadband_us_neg: 50.0
""")
            temp_path = f.name
        try:
            PWMController(temp_path)
        finally:
            os.unlink(temp_path)

    results.append(run_test("Config huge deadband (9999us)", test_huge_deadband, expect_exception=ValueError))
    print_result(results[-1])

    # =========================================================================
    # SECTION 7: Reload config edge cases
    # =========================================================================
    print("\n--- SECTION 7: reload_config() Edge Cases ---")

    def test_reload_nonexistent():
        result = controller.reload_config('/nonexistent/config.yaml')
        assert result == False

    results.append(run_test("reload_config() nonexistent file", test_reload_nonexistent))
    print_result(results[-1])

    def test_reload_invalid_yaml():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("not: valid: yaml: {{{{")
            temp_path = f.name
        try:
            result = controller.reload_config(temp_path)
            assert result == False
        finally:
            os.unlink(temp_path)

    results.append(run_test("reload_config() invalid YAML", test_reload_invalid_yaml))
    print_result(results[-1])

    # Restore valid config
    controller.reload_config(CONFIG_FILE)

    # =========================================================================
    # SECTION 8: Timing edge cases
    # =========================================================================
    print("\n--- SECTION 8: Timing Edge Cases ---")

    def test_stale_timestamp():
        # command_ts way in the past
        controller.update_named({'lift_boom': 0.5}, command_ts=0.0)

    results.append(run_test("update_named() with stale timestamp", test_stale_timestamp))
    print_result(results[-1])

    def test_future_timestamp():
        # command_ts way in the future
        future_ts = time.monotonic() + 999999
        controller.update_named({'lift_boom': 0.5}, command_ts=future_ts)

    results.append(run_test("update_named() with future timestamp", test_future_timestamp))
    print_result(results[-1])

    def test_negative_timestamp():
        controller.update_named({'lift_boom': 0.5}, command_ts=-1000.0)

    results.append(run_test("update_named() with negative timestamp", test_negative_timestamp))
    print_result(results[-1])

    # =========================================================================
    # SECTION 9: Rapid state changes
    # =========================================================================
    print("\n--- SECTION 9: Rapid State Changes ---")

    def test_rapid_pump_toggle():
        for _ in range(100):
            controller.set_pump(True)
            controller.set_pump(False)
        controller.set_pump(True)

    results.append(run_test("Rapid pump enable/disable (100x)", test_rapid_pump_toggle))
    print_result(results[-1])

    def test_rapid_channel_toggle():
        for _ in range(100):
            controller.disable_channels(True)
            controller.disable_channels(False)

    results.append(run_test("Rapid channel enable/disable (100x)", test_rapid_channel_toggle))
    print_result(results[-1])

    def test_rapid_variable_toggle():
        for _ in range(100):
            controller.toggle_pump_variable(True)
            controller.toggle_pump_variable(False)
        controller.toggle_pump_variable(True)

    results.append(run_test("Rapid variable pump toggle (100x)", test_rapid_variable_toggle))
    print_result(results[-1])

    def test_rapid_reset():
        for _ in range(20):
            controller.reset(reset_pump=True)
            controller.reset(reset_pump=False)

    results.append(run_test("Rapid reset() calls (20x)", test_rapid_reset))
    print_result(results[-1])

    # =========================================================================
    # SECTION 10: DirectPWMWriter edge cases
    # =========================================================================
    print("\n--- SECTION 10: DirectPWMWriter Edge Cases ---")

    def test_writer_huge_duty():
        writer = controller._direct_writer
        writer.set_channel(2, 0xFFFFFFFF)  # Way over 16-bit
        writer.flush()

    results.append(run_test("DirectPWMWriter huge duty cycle", test_writer_huge_duty))
    print_result(results[-1])

    def test_writer_negative_duty():
        writer = controller._direct_writer
        writer.set_channel(2, -1000)
        writer.flush()

    results.append(run_test("DirectPWMWriter negative duty cycle", test_writer_negative_duty))
    print_result(results[-1])

    def test_writer_channel_out_of_range():
        writer = controller._direct_writer
        writer.set_channel(999, 32768)  # Channel 999 doesn't exist
        # This sets _duty_cycles[999] which will fail on access
        # But flush() only writes configured range, so may not crash

    results.append(run_test("DirectPWMWriter channel 999", test_writer_channel_out_of_range))
    print_result(results[-1])

    # =========================================================================
    # SECTION 11: Boundary values
    # =========================================================================
    print("\n--- SECTION 11: Boundary Values ---")

    def test_exactly_one():
        controller.update_named({'lift_boom': 1.0, 'tilt_boom': 1.0, 'scoop': 1.0, 'rotate': 1.0})

    results.append(run_test("All channels at exactly 1.0", test_exactly_one))
    print_result(results[-1])

    def test_exactly_negative_one():
        controller.update_named({'lift_boom': -1.0, 'tilt_boom': -1.0, 'scoop': -1.0, 'rotate': -1.0})

    results.append(run_test("All channels at exactly -1.0", test_exactly_negative_one))
    print_result(results[-1])

    def test_epsilon_values():
        eps = sys.float_info.epsilon
        controller.update_named({'lift_boom': eps, 'tilt_boom': -eps})

    results.append(run_test("Epsilon values (smallest float)", test_epsilon_values))
    print_result(results[-1])

    def test_denormal_values():
        denorm = sys.float_info.min / 2
        controller.update_named({'lift_boom': denorm})

    results.append(run_test("Denormalized float value", test_denormal_values))
    print_result(results[-1])

    # =========================================================================
    # Final cleanup
    # =========================================================================
    print("\nCleaning up...")
    controller.reset(reset_pump=True)
    time.sleep(0.1)

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("ROBUSTNESS TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration_ms for r in results)

    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Total time: {total_time:.1f}ms")
    print()

    if failed > 0:
        print("FAILED TESTS:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message[:80]}")
        return 1
    else:
        print("All robustness tests passed! Controller handles bad inputs gracefully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
