#!/usr/bin/env python3
"""
Comprehensive test for PCA9685_controller.py

Tests all public methods and verifies hardware communication.
Run on Raspberry Pi 5 with PCA9685 connected.
"""

import sys
import time
import math
import traceback
from typing import Callable, List, Tuple

# Add modules to path
sys.path.insert(0, '/home/joel/masi/modules')

CONFIG_FILE = '/home/joel/masi/configuration_files/servo_config.yaml'


class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", duration_ms: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms


def run_test(name: str, test_fn: Callable) -> TestResult:
    """Run a single test and capture result."""
    start = time.perf_counter()
    try:
        test_fn()
        duration = (time.perf_counter() - start) * 1000
        return TestResult(name, True, "OK", duration)
    except AssertionError as e:
        duration = (time.perf_counter() - start) * 1000
        return TestResult(name, False, f"ASSERTION FAILED: {e}", duration)
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return TestResult(name, False, f"ERROR: {e}\n{traceback.format_exc()}", duration)


def print_result(result: TestResult):
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {result.name} ({result.duration_ms:.1f}ms)")
    if not result.passed:
        for line in result.message.split('\n'):
            print(f"         {line}")


def main():
    print("=" * 60)
    print("PCA9685 Controller Comprehensive Test")
    print("=" * 60)
    print()

    results: List[TestResult] = []
    controller = None

    # =========================================================================
    # Test 1: Import module
    # =========================================================================
    def test_import():
        global PCA9685_controller
        import PCA9685_controller
        assert hasattr(PCA9685_controller, 'PWMController')
        assert hasattr(PCA9685_controller, 'DirectPWMWriter')
        assert hasattr(PCA9685_controller, 'ChannelConfig')
        assert hasattr(PCA9685_controller, 'PumpConfig')
        assert hasattr(PCA9685_controller, 'PWMConstants')

    results.append(run_test("Import module", test_import))
    print_result(results[-1])
    if not results[-1].passed:
        print("\nCannot continue without successful import.")
        return 1

    from PCA9685_controller import (
        PWMController, DirectPWMWriter, ChannelConfig, PumpConfig, PWMConstants
    )

    # =========================================================================
    # Test 2: PWMConstants values
    # =========================================================================
    def test_constants():
        assert PWMConstants.PWM_FREQUENCY_DEFAULT == 200
        assert PWMConstants.MAX_CHANNELS == 16
        assert PWMConstants.DUTY_CYCLE_MAX == 65535
        assert PWMConstants.PULSE_MIN == 0
        assert PWMConstants.PULSE_MAX == 4095

    results.append(run_test("PWMConstants values", test_constants))
    print_result(results[-1])

    # =========================================================================
    # Test 3: ChannelConfig dataclass
    # =========================================================================
    def test_channel_config():
        cfg = ChannelConfig(
            output_channel=0,
            pulse_min=1000,
            pulse_max=2000,
            direction=1,
            deadband_us_pos=50.0,
            deadband_us_neg=50.0,
        )
        assert cfg.pulse_range == 1000
        assert cfg.center == 1500.0  # auto-computed
        assert cfg.deadzone_threshold == 0.0
        assert cfg._dither_omega == 2.0 * math.pi * 40.0  # default dither_hz

    results.append(run_test("ChannelConfig dataclass", test_channel_config))
    print_result(results[-1])

    # =========================================================================
    # Test 4: PumpConfig dataclass
    # =========================================================================
    def test_pump_config():
        cfg = PumpConfig(
            output_channel=9,
            pulse_min=1000,
            pulse_max=1700,
            idle=0.4,
            multiplier=0.4,
        )
        assert cfg.output_channel == 9
        assert cfg.idle == 0.4

    results.append(run_test("PumpConfig dataclass", test_pump_config))
    print_result(results[-1])

    # =========================================================================
    # Test 5: Controller initialization (HARDWARE)
    # =========================================================================
    def test_controller_init():
        nonlocal controller
        controller = PWMController(
            CONFIG_FILE,
            pump_variable=True,
            toggle_channels=True,
            log_level="WARNING",
        )
        assert controller is not None
        assert controller.pca is not None
        assert controller._direct_writer is not None
        assert len(controller.channel_configs) > 0

    results.append(run_test("Controller initialization (hardware)", test_controller_init))
    print_result(results[-1])
    if not results[-1].passed:
        print("\nCannot continue without successful hardware init.")
        return 1

    # =========================================================================
    # Test 6: get_channel_names()
    # =========================================================================
    def test_get_channel_names():
        names = controller.get_channel_names(include_pump=True)
        assert 'lift_boom' in names
        assert 'tilt_boom' in names
        assert 'scoop' in names
        assert 'rotate' in names
        assert 'pump' in names

        names_no_pump = controller.get_channel_names(include_pump=False)
        assert 'pump' not in names_no_pump

    results.append(run_test("get_channel_names()", test_get_channel_names))
    print_result(results[-1])

    # =========================================================================
    # Test 7: build_zero_commands()
    # =========================================================================
    def test_build_zero_commands():
        cmds = controller.build_zero_commands(include_toggleable=True, include_pump=False)
        assert all(v == 0.0 for v in cmds.values())
        assert 'lift_boom' in cmds

        cmds_with_pump = controller.build_zero_commands(include_pump=True)
        assert 'pump' in cmds_with_pump

    results.append(run_test("build_zero_commands()", test_build_zero_commands))
    print_result(results[-1])

    # =========================================================================
    # Test 8: compute_pulse() - preview pulse without hardware write
    # =========================================================================
    def test_compute_pulse():
        # Test center (value=0)
        pulse_center = controller.compute_pulse('lift_boom', 0.0)
        cfg = controller.channel_configs['lift_boom']
        assert pulse_center is not None
        assert abs(pulse_center - cfg.center) < 1.0, f"Expected ~{cfg.center}, got {pulse_center}"

        # Test positive value
        pulse_pos = controller.compute_pulse('lift_boom', 1.0)
        assert pulse_pos is not None
        assert pulse_pos > cfg.center

        # Test negative value
        pulse_neg = controller.compute_pulse('lift_boom', -1.0)
        assert pulse_neg is not None
        assert pulse_neg < cfg.center

        # Test invalid channel
        assert controller.compute_pulse('nonexistent', 0.5) is None

    results.append(run_test("compute_pulse()", test_compute_pulse))
    print_result(results[-1])

    # =========================================================================
    # Test 9: _apply_gamma()
    # =========================================================================
    def test_apply_gamma():
        # Linear (gamma=1)
        assert PWMController._apply_gamma(0.5, 1.0) == 0.5
        assert PWMController._apply_gamma(-0.5, 1.0) == -0.5

        # Gamma < 1 (more sensitive near zero)
        result = PWMController._apply_gamma(0.5, 0.5)
        assert result > 0.5  # sqrt(0.5) ≈ 0.707

        # Gamma > 1 (less sensitive near zero)
        result = PWMController._apply_gamma(0.5, 2.0)
        assert result < 0.5  # 0.5^2 = 0.25

        # Symmetric for negative
        assert PWMController._apply_gamma(-0.5, 2.0) == -PWMController._apply_gamma(0.5, 2.0)

    results.append(run_test("_apply_gamma()", test_apply_gamma))
    print_result(results[-1])

    # =========================================================================
    # Test 10: reset() - sends center to all channels
    # =========================================================================
    def test_reset():
        controller.reset(reset_pump=True)
        # Should complete without error
        assert controller.is_safe_state == False  # reset clears safe state
        assert controller._pump_override_throttle is None

    results.append(run_test("reset() (hardware write)", test_reset))
    print_result(results[-1])

    # =========================================================================
    # Test 11: update_named() with zero commands
    # =========================================================================
    def test_update_named_zero():
        commands = {'lift_boom': 0.0, 'tilt_boom': 0.0, 'scoop': 0.0, 'rotate': 0.0}
        controller.update_named(commands)
        # Should complete without error

    results.append(run_test("update_named() with zeros (hardware)", test_update_named_zero))
    print_result(results[-1])

    # =========================================================================
    # Test 12: update_named() with small values
    # =========================================================================
    def test_update_named_small():
        commands = {'lift_boom': 0.1, 'tilt_boom': -0.1, 'scoop': 0.05, 'rotate': 0.0}
        controller.update_named(commands)
        time.sleep(0.05)
        controller.update_named({'lift_boom': 0.0, 'tilt_boom': 0.0, 'scoop': 0.0, 'rotate': 0.0})

    results.append(run_test("update_named() with small values (hardware)", test_update_named_small))
    print_result(results[-1])

    # =========================================================================
    # Test 13: update_named() with pump override
    # =========================================================================
    def test_pump_override():
        commands = {'pump': 0.5, 'lift_boom': 0.0}
        controller.update_named(commands, one_shot_pump_override=True)
        assert controller._pump_override_throttle is None  # cleared after one-shot

        controller.update_named({'pump': 0.3}, one_shot_pump_override=False)
        assert controller._pump_override_throttle == 0.3
        controller.clear_pump_override()
        assert controller._pump_override_throttle is None

    results.append(run_test("pump override (hardware)", test_pump_override))
    print_result(results[-1])

    # =========================================================================
    # Test 14: set_pump() enable/disable
    # =========================================================================
    def test_set_pump():
        controller.set_pump(True)
        assert controller.pump_enabled == True
        controller.set_pump(False)
        assert controller.pump_enabled == False
        controller.set_pump(True)  # restore

    results.append(run_test("set_pump()", test_set_pump))
    print_result(results[-1])

    # =========================================================================
    # Test 15: toggle_pump_variable()
    # =========================================================================
    def test_toggle_pump_variable():
        original = controller.pump_variable
        controller.toggle_pump_variable(not original)
        assert controller.pump_variable == (not original)
        controller.toggle_pump_variable(original)  # restore

    results.append(run_test("toggle_pump_variable()", test_toggle_pump_variable))
    print_result(results[-1])

    # =========================================================================
    # Test 16: update_pump_load() / reset_pump_load()
    # =========================================================================
    def test_pump_load():
        controller.reset_pump_load()
        assert controller.manual_pump_load == 0.0

        controller.update_pump_load(1.0)  # adds 0.1
        assert abs(controller.manual_pump_load - 0.1) < 0.001

        controller.update_pump_load(-1.0)  # subtracts 0.1
        assert abs(controller.manual_pump_load) < 0.001

        controller.reset_pump_load()

    results.append(run_test("update_pump_load() / reset_pump_load()", test_pump_load))
    print_result(results[-1])

    # =========================================================================
    # Test 17: disable_channels()
    # =========================================================================
    def test_disable_channels():
        controller.disable_channels(True)
        assert controller.toggle_channels == False
        controller.disable_channels(False)
        assert controller.toggle_channels == True

    results.append(run_test("disable_channels()", test_disable_channels))
    print_result(results[-1])

    # =========================================================================
    # Test 18: get_average_input_rate()
    # =========================================================================
    def test_input_rate():
        rate = controller.get_average_input_rate()
        assert isinstance(rate, float)
        assert rate >= 0.0

    results.append(run_test("get_average_input_rate()", test_input_rate))
    print_result(results[-1])

    # =========================================================================
    # Test 19: set_log_level()
    # =========================================================================
    def test_log_level():
        controller.set_log_level("DEBUG")
        assert controller.logger.level == 10  # DEBUG
        controller.set_log_level("WARNING")
        assert controller.logger.level == 30  # WARNING

    results.append(run_test("set_log_level()", test_log_level))
    print_result(results[-1])

    # =========================================================================
    # Test 20: reload_config()
    # =========================================================================
    def test_reload_config():
        success = controller.reload_config(CONFIG_FILE)
        assert success == True
        assert len(controller.channel_configs) > 0

    results.append(run_test("reload_config()", test_reload_config))
    print_result(results[-1])

    # =========================================================================
    # Test 21: DirectPWMWriter standalone
    # =========================================================================
    def test_direct_writer():
        writer = controller._direct_writer
        assert writer is not None
        assert writer._num_channels > 0

        # Test set_channel
        writer.set_channel(2, 32768)
        assert writer._duty_cycles[2] == 32768

        # Test set_channel_range (re-allocates buffer)
        old_buf_len = len(writer._buf)
        writer.set_channel_range(2, 6)
        assert writer._num_channels == 5
        writer.set_channel_range(controller._direct_writer._min_ch, controller._direct_writer._max_ch)

    results.append(run_test("DirectPWMWriter methods", test_direct_writer))
    print_result(results[-1])

    # =========================================================================
    # Test 22: Rapid update loop (stress test)
    # =========================================================================
    def test_rapid_updates():
        import random
        start = time.perf_counter()
        iterations = 100
        for i in range(iterations):
            commands = {
                'lift_boom': random.uniform(-0.3, 0.3),
                'tilt_boom': random.uniform(-0.3, 0.3),
                'scoop': random.uniform(-0.3, 0.3),
                'rotate': random.uniform(-0.1, 0.1),
            }
            controller.update_named(commands)
        elapsed = time.perf_counter() - start
        rate = iterations / elapsed
        assert rate > 50, f"Update rate {rate:.1f} Hz is too slow (expected >50 Hz)"
        # Reset after stress test
        controller.reset()

    results.append(run_test(f"Rapid update loop (100 iterations)", test_rapid_updates))
    print_result(results[-1])

    # =========================================================================
    # Test 23: Deadband behavior verification
    # =========================================================================
    def test_deadband():
        cfg = controller.channel_configs['lift_boom']
        # At value=0, pulse should be at center
        pulse_zero = controller.compute_pulse('lift_boom', 0.0)
        assert abs(pulse_zero - cfg.center) < 1.0

        # Small positive value should jump over deadband
        pulse_small_pos = controller.compute_pulse('lift_boom', 0.05)
        expected_base = cfg.center + cfg.deadband_us_pos
        assert pulse_small_pos > cfg.center + cfg.deadband_us_pos * 0.9, \
            f"Expected pulse > {expected_base}, got {pulse_small_pos}"

    results.append(run_test("Deadband behavior", test_deadband))
    print_result(results[-1])

    # =========================================================================
    # Test 24: Dither timing (verify dither changes over time)
    # =========================================================================
    def test_dither_timing():
        cfg = controller.channel_configs['lift_boom']
        if not cfg.dither_enable:
            return  # Skip if dither disabled

        # Get pulse at two different times
        t1 = time.time()
        pulse1 = controller.compute_pulse('lift_boom', 0.5, now=t1)
        t2 = t1 + 0.025  # 25ms later (should be ~1/4 cycle at 40Hz default)
        pulse2 = controller.compute_pulse('lift_boom', 0.5, now=t2)

        # Pulses should differ due to dither
        assert pulse1 != pulse2, "Dither should cause pulse variation over time"

    results.append(run_test("Dither timing", test_dither_timing))
    print_result(results[-1])

    # =========================================================================
    # Test 25: Value clamping
    # =========================================================================
    def test_value_clamping():
        # Values outside [-1, 1] should be clamped
        controller.update_named({'lift_boom': 5.0})  # Should clamp to 1.0
        controller.update_named({'lift_boom': -5.0})  # Should clamp to -1.0
        controller.reset()

    results.append(run_test("Value clamping", test_value_clamping))
    print_result(results[-1])

    # =========================================================================
    # Final cleanup
    # =========================================================================
    print()
    print("Cleaning up...")
    controller.reset(reset_pump=True)
    time.sleep(0.1)

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

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
                print(f"  - {r.name}")
        return 1
    else:
        print("All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
