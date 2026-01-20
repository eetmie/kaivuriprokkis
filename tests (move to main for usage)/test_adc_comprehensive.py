#!/usr/bin/env python3
"""
Comprehensive ADC Test - Function Tests and Robustness/Edge Cases

Tests all functions in ADC.py and ADCPi.py, plus edge cases and error handling.
"""

import sys
import time
import traceback
from typing import Callable, List

sys.path.insert(0, '/home/joel/masi/modules')


class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", duration_ms: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms


def run_test(name: str, test_fn: Callable, expect_exception: type = None) -> TestResult:
    """Run a single test."""
    start = time.perf_counter()
    try:
        test_fn()
        duration = (time.perf_counter() - start) * 1000
        if expect_exception:
            return TestResult(name, False, f"Expected {expect_exception.__name__} but none raised", duration)
        return TestResult(name, True, "OK", duration)
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        if expect_exception and isinstance(e, expect_exception):
            return TestResult(name, True, f"Correctly raised {expect_exception.__name__}", duration)
        elif expect_exception:
            return TestResult(name, False, f"Expected {expect_exception.__name__}, got {type(e).__name__}: {e}", duration)
        else:
            return TestResult(name, False, f"{type(e).__name__}: {e}", duration)


def print_result(result: TestResult):
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {result.name} ({result.duration_ms:.1f}ms)")
    if result.message and result.message != "OK":
        for line in result.message.split('\n')[:2]:
            print(f"         {line}")


def main():
    print("=" * 70)
    print("ADC Comprehensive Test (Functions + Robustness)")
    print("=" * 70)
    print()

    results: List[TestResult] = []

    # =========================================================================
    # PART 1: ADCPi Low-Level Library Tests
    # =========================================================================
    print("=== PART 1: ADCPi Library (Low-Level) ===\n")

    # --- Imports ---
    def test_import_adcpi():
        from ADCPi import ADCPi, ADCTimeoutError, Error
        assert ADCPi is not None
        assert ADCTimeoutError is not None

    results.append(run_test("Import ADCPi", test_import_adcpi))
    print_result(results[-1])

    from ADCPi import ADCPi, ADCTimeoutError

    # --- Initialization ---
    adc_low = None

    def test_adcpi_init():
        nonlocal adc_low
        adc_low = ADCPi(address=0x68, address2=0x69, bus=3)
        assert adc_low is not None

    results.append(run_test("ADCPi initialization (bus 3)", test_adcpi_init))
    print_result(results[-1])

    if not results[-1].passed:
        print("\nCannot continue without ADCPi init.")
        return 1

    # --- Address methods ---
    def test_get_addresses():
        addr1 = adc_low.get_i2c_address1()
        addr2 = adc_low.get_i2c_address2()
        assert addr1 == 0x68
        assert addr2 == 0x69

    results.append(run_test("get_i2c_address1/2()", test_get_addresses))
    print_result(results[-1])

    def test_set_address1():
        adc_low.set_i2c_address1(0x6A)
        assert adc_low.get_i2c_address1() == 0x6A
        adc_low.set_i2c_address1(0x68)  # Restore

    results.append(run_test("set_i2c_address1()", test_set_address1))
    print_result(results[-1])

    def test_set_address2():
        adc_low.set_i2caddress2(0x6B)
        assert adc_low.get_i2c_address2() == 0x6B
        adc_low.set_i2caddress2(0x69)  # Restore

    results.append(run_test("set_i2caddress2()", test_set_address2))
    print_result(results[-1])

    # --- Configuration methods ---
    def test_set_pga():
        for gain in [1, 2, 4, 8]:
            adc_low.set_pga(gain)
        adc_low.set_pga(1)  # Restore default

    results.append(run_test("set_pga() all values", test_set_pga))
    print_result(results[-1])

    def test_set_bit_rate():
        for rate in [12, 14, 16, 18]:
            adc_low.set_bit_rate(rate)
        adc_low.set_bit_rate(12)  # Restore to fastest

    results.append(run_test("set_bit_rate() all values", test_set_bit_rate))
    print_result(results[-1])

    def test_set_bit_mode():
        adc_low.set_bit_mode(14)
        adc_low.set_bit_mode(12)

    results.append(run_test("set_bit_mode()", test_set_bit_mode))
    print_result(results[-1])

    def test_set_conversion_mode():
        adc_low.set_conversion_mode(0)  # One-shot
        adc_low.set_conversion_mode(1)  # Continuous

    results.append(run_test("set_conversion_mode()", test_set_conversion_mode))
    print_result(results[-1])

    # --- Read methods ---
    def test_read_voltage_all_channels():
        for ch in range(1, 9):
            v = adc_low.read_voltage(ch)
            assert isinstance(v, float)
            assert 0.0 <= v <= 5.5  # Reasonable voltage range

    results.append(run_test("read_voltage() channels 1-8", test_read_voltage_all_channels))
    print_result(results[-1])

    def test_read_raw_all_channels():
        for ch in range(1, 9):
            raw = adc_low.read_raw(ch)
            assert isinstance(raw, int)

    results.append(run_test("read_raw() channels 1-8", test_read_raw_all_channels))
    print_result(results[-1])

    # =========================================================================
    # PART 2: SimpleADC High-Level Library Tests
    # =========================================================================
    print("\n=== PART 2: SimpleADC Library (High-Level) ===\n")

    def test_import_adc():
        from ADC import SimpleADC, ADCConfig, Error
        assert SimpleADC is not None
        assert ADCConfig is not None

    results.append(run_test("Import ADC module", test_import_adc))
    print_result(results[-1])

    from ADC import SimpleADC, ADCConfig

    # --- ADCConfig tests ---
    def test_adcconfig_init():
        cfg = ADCConfig()
        assert cfg.bit_rate == 12
        assert cfg.pga_gain == 1
        assert cfg.conversion_mode == 1
        assert len(cfg.sensors) == 8

    results.append(run_test("ADCConfig initialization", test_adcconfig_init))
    print_result(results[-1])

    def test_adcconfig_validate():
        cfg = ADCConfig()
        cfg.validate()  # Should not raise

    results.append(run_test("ADCConfig.validate()", test_adcconfig_validate))
    print_result(results[-1])

    def test_adcconfig_get_needed_boards():
        cfg = ADCConfig()
        boards = cfg.get_needed_boards()
        assert "b1" in boards

    results.append(run_test("ADCConfig.get_needed_boards()", test_adcconfig_get_needed_boards))
    print_result(results[-1])

    def test_adcconfig_get_sensor_by_input():
        cfg = ADCConfig()
        name = cfg.get_sensor_by_input("b1", 1)
        assert name is not None
        assert "LiftBoom" in name or name is not None

    results.append(run_test("ADCConfig.get_sensor_by_input()", test_adcconfig_get_sensor_by_input))
    print_result(results[-1])

    def test_adcconfig_get_channel_mapping():
        cfg = ADCConfig()
        mapping = cfg.get_channel_mapping("b1")
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    results.append(run_test("ADCConfig.get_channel_mapping()", test_adcconfig_get_channel_mapping))
    print_result(results[-1])

    def test_adcconfig_get_board_addresses():
        cfg = ADCConfig()
        addr1, addr2 = cfg.get_board_addresses("b1")
        assert addr1 == 0x68
        assert addr2 == 0x69

    results.append(run_test("ADCConfig.get_board_addresses()", test_adcconfig_get_board_addresses))
    print_result(results[-1])

    def test_adcconfig_get_sampling_info():
        cfg = ADCConfig()
        info = cfg.get_sampling_info()
        assert "bit_rate" in info
        assert "max_samples_per_second" in info
        assert info["max_samples_per_second"] == 240  # 12-bit default

    results.append(run_test("ADCConfig.get_sampling_info()", test_adcconfig_get_sampling_info))
    print_result(results[-1])

    # --- SimpleADC tests ---
    simple_adc = None

    def test_simpleadc_init():
        nonlocal simple_adc
        simple_adc = SimpleADC(log_level="ERROR")
        assert simple_adc is not None
        assert simple_adc.initialized

    results.append(run_test("SimpleADC initialization", test_simpleadc_init))
    print_result(results[-1])

    if not results[-1].passed:
        print("\nCannot continue without SimpleADC init.")
        return 1

    def test_simpleadc_read_sensors():
        readings = simple_adc.read_sensors()
        assert isinstance(readings, dict)
        assert len(readings) == 8
        for name, value in readings.items():
            assert isinstance(value, float)

    results.append(run_test("SimpleADC.read_sensors()", test_simpleadc_read_sensors))
    print_result(results[-1])

    def test_simpleadc_read_channel():
        v = simple_adc.read_channel("b1", 1)
        assert isinstance(v, float)
        assert 0.0 <= v <= 5.5

    results.append(run_test("SimpleADC.read_channel()", test_simpleadc_read_channel))
    print_result(results[-1])

    def test_simpleadc_get_board_info():
        info = simple_adc.get_board_info()
        assert "b1" in info
        assert "address1" in info["b1"]

    results.append(run_test("SimpleADC.get_board_info()", test_simpleadc_get_board_info))
    print_result(results[-1])

    def test_simpleadc_get_config_summary():
        summary = simple_adc.get_config_summary()
        assert "bit_rate" in summary
        assert "total_sensors" in summary
        assert summary["total_sensors"] == 8

    results.append(run_test("SimpleADC.get_config_summary()", test_simpleadc_get_config_summary))
    print_result(results[-1])

    def test_simpleadc_set_log_level():
        simple_adc.set_log_level("DEBUG")
        simple_adc.set_log_level("ERROR")

    results.append(run_test("SimpleADC.set_log_level()", test_simpleadc_set_log_level))
    print_result(results[-1])

    def test_simpleadc_context_manager():
        with SimpleADC(log_level="ERROR") as adc:
            readings = adc.read_sensors()
            assert len(readings) == 8

    results.append(run_test("SimpleADC context manager", test_simpleadc_context_manager))
    print_result(results[-1])

    def test_simpleadc_filter_alpha():
        adc_filtered = SimpleADC(filter_alpha=0.5, log_level="ERROR")
        assert adc_filtered.filter_alpha == 0.5
        # Read twice to exercise filter
        adc_filtered.read_sensors()
        adc_filtered.read_sensors()

    results.append(run_test("SimpleADC EMA filter", test_simpleadc_filter_alpha))
    print_result(results[-1])

    # =========================================================================
    # PART 3: ADCPi Robustness Tests (Edge Cases)
    # =========================================================================
    print("\n=== PART 3: ADCPi Robustness (Edge Cases) ===\n")

    # --- Invalid addresses ---
    def test_invalid_address1_low():
        ADCPi(address=0x50, address2=0x69, bus=3)

    results.append(run_test("ADCPi address=0x50 (too low)", test_invalid_address1_low, expect_exception=ValueError))
    print_result(results[-1])

    def test_invalid_address1_high():
        ADCPi(address=0x80, address2=0x69, bus=3)

    results.append(run_test("ADCPi address=0x80 (too high)", test_invalid_address1_high, expect_exception=ValueError))
    print_result(results[-1])

    def test_invalid_address2_low():
        ADCPi(address=0x68, address2=0x50, bus=3)

    results.append(run_test("ADCPi address2=0x50 (too low)", test_invalid_address2_low, expect_exception=ValueError))
    print_result(results[-1])

    def test_set_invalid_address1():
        adc_low.set_i2c_address1(0x50)

    results.append(run_test("set_i2c_address1(0x50)", test_set_invalid_address1, expect_exception=ValueError))
    print_result(results[-1])

    def test_set_invalid_address2():
        adc_low.set_i2caddress2(0x50)

    results.append(run_test("set_i2caddress2(0x50)", test_set_invalid_address2, expect_exception=ValueError))
    print_result(results[-1])

    # --- Invalid PGA ---
    def test_invalid_pga_zero():
        adc_low.set_pga(0)

    results.append(run_test("set_pga(0)", test_invalid_pga_zero, expect_exception=ValueError))
    print_result(results[-1])

    def test_invalid_pga_3():
        adc_low.set_pga(3)

    results.append(run_test("set_pga(3)", test_invalid_pga_3, expect_exception=ValueError))
    print_result(results[-1])

    def test_invalid_pga_16():
        adc_low.set_pga(16)

    results.append(run_test("set_pga(16)", test_invalid_pga_16, expect_exception=ValueError))
    print_result(results[-1])

    # --- Invalid bit rate ---
    def test_invalid_bitrate_10():
        adc_low.set_bit_rate(10)

    results.append(run_test("set_bit_rate(10)", test_invalid_bitrate_10, expect_exception=ValueError))
    print_result(results[-1])

    def test_invalid_bitrate_20():
        adc_low.set_bit_rate(20)

    results.append(run_test("set_bit_rate(20)", test_invalid_bitrate_20, expect_exception=ValueError))
    print_result(results[-1])

    def test_invalid_bitmode_13():
        adc_low.set_bit_mode(13)

    results.append(run_test("set_bit_mode(13)", test_invalid_bitmode_13, expect_exception=ValueError))
    print_result(results[-1])

    # --- Invalid conversion mode ---
    def test_invalid_conversion_mode_2():
        adc_low.set_conversion_mode(2)

    results.append(run_test("set_conversion_mode(2)", test_invalid_conversion_mode_2, expect_exception=ValueError))
    print_result(results[-1])

    def test_invalid_conversion_mode_neg():
        adc_low.set_conversion_mode(-1)

    results.append(run_test("set_conversion_mode(-1)", test_invalid_conversion_mode_neg, expect_exception=ValueError))
    print_result(results[-1])

    # --- Invalid channels ---
    def test_read_voltage_channel_0():
        adc_low.read_voltage(0)

    results.append(run_test("read_voltage(0)", test_read_voltage_channel_0, expect_exception=ValueError))
    print_result(results[-1])

    def test_read_voltage_channel_9():
        adc_low.read_voltage(9)

    results.append(run_test("read_voltage(9)", test_read_voltage_channel_9, expect_exception=ValueError))
    print_result(results[-1])

    def test_read_voltage_channel_neg():
        adc_low.read_voltage(-1)

    results.append(run_test("read_voltage(-1)", test_read_voltage_channel_neg, expect_exception=ValueError))
    print_result(results[-1])

    def test_read_raw_channel_0():
        adc_low.read_raw(0)

    results.append(run_test("read_raw(0)", test_read_raw_channel_0, expect_exception=ValueError))
    print_result(results[-1])

    def test_read_raw_channel_100():
        adc_low.read_raw(100)

    results.append(run_test("read_raw(100)", test_read_raw_channel_100, expect_exception=ValueError))
    print_result(results[-1])

    # =========================================================================
    # PART 4: SimpleADC Robustness Tests
    # =========================================================================
    print("\n=== PART 4: SimpleADC Robustness (Edge Cases) ===\n")

    # --- Invalid board/channel in read_channel ---
    def test_read_channel_invalid_board():
        simple_adc.read_channel("nonexistent_board", 1)

    results.append(run_test("read_channel() invalid board", test_read_channel_invalid_board, expect_exception=ValueError))
    print_result(results[-1])

    def test_read_channel_channel_0():
        simple_adc.read_channel("b1", 0)

    results.append(run_test("read_channel() channel 0", test_read_channel_channel_0, expect_exception=ValueError))
    print_result(results[-1])

    def test_read_channel_channel_9():
        simple_adc.read_channel("b1", 9)

    results.append(run_test("read_channel() channel 9", test_read_channel_channel_9, expect_exception=ValueError))
    print_result(results[-1])

    def test_read_channel_channel_neg():
        simple_adc.read_channel("b1", -5)

    results.append(run_test("read_channel() channel -5", test_read_channel_channel_neg, expect_exception=ValueError))
    print_result(results[-1])

    # --- ADCConfig validation ---
    def test_config_invalid_pga():
        cfg = ADCConfig()
        cfg.pga_gain = 3  # Invalid
        cfg.validate()

    results.append(run_test("ADCConfig invalid pga_gain=3", test_config_invalid_pga, expect_exception=ValueError))
    print_result(results[-1])

    def test_config_invalid_bitrate():
        cfg = ADCConfig()
        cfg.bit_rate = 15  # Invalid
        cfg.validate()

    results.append(run_test("ADCConfig invalid bit_rate=15", test_config_invalid_bitrate, expect_exception=ValueError))
    print_result(results[-1])

    def test_config_invalid_conversion_mode():
        cfg = ADCConfig()
        cfg.conversion_mode = 5  # Invalid
        cfg.validate()

    results.append(run_test("ADCConfig invalid conversion_mode=5", test_config_invalid_conversion_mode, expect_exception=ValueError))
    print_result(results[-1])

    def test_config_invalid_i2c_address():
        cfg = ADCConfig()
        cfg.i2c_addresses["b1"] = [0x05, 0x69]  # 0x05 is invalid
        cfg.validate()

    results.append(run_test("ADCConfig invalid I2C address 0x05", test_config_invalid_i2c_address, expect_exception=ValueError))
    print_result(results[-1])

    def test_config_wrong_address_count():
        cfg = ADCConfig()
        cfg.i2c_addresses["b1"] = [0x68]  # Only 1 address
        cfg.validate()

    results.append(run_test("ADCConfig only 1 I2C address", test_config_wrong_address_count, expect_exception=ValueError))
    print_result(results[-1])

    def test_config_sensor_invalid_channel():
        cfg = ADCConfig()
        cfg.sensors["test"] = {"input": ["b1", 10]}  # Channel 10 invalid
        cfg.validate()

    results.append(run_test("ADCConfig sensor channel=10", test_config_sensor_invalid_channel, expect_exception=ValueError))
    print_result(results[-1])

    def test_config_sensor_unknown_board():
        cfg = ADCConfig()
        cfg.sensors["test"] = {"input": ["b99", 1]}  # Unknown board
        cfg.validate()

    results.append(run_test("ADCConfig sensor unknown board", test_config_sensor_unknown_board, expect_exception=ValueError))
    print_result(results[-1])

    def test_config_sensor_missing_input():
        cfg = ADCConfig()
        cfg.sensors["test"] = {"other": "value"}  # Missing 'input'
        cfg.validate()

    results.append(run_test("ADCConfig sensor missing 'input'", test_config_sensor_missing_input, expect_exception=ValueError))
    print_result(results[-1])

    # --- Filter alpha edge cases ---
    def test_filter_alpha_zero():
        adc = SimpleADC(filter_alpha=0.0, log_level="ERROR")
        adc.read_sensors()
        adc.read_sensors()  # Should work (no filtering, uses previous value)

    results.append(run_test("SimpleADC filter_alpha=0.0", test_filter_alpha_zero))
    print_result(results[-1])

    def test_filter_alpha_one():
        adc = SimpleADC(filter_alpha=1.0, log_level="ERROR")
        adc.read_sensors()

    results.append(run_test("SimpleADC filter_alpha=1.0", test_filter_alpha_one))
    print_result(results[-1])

    # --- Rapid reads ---
    def test_rapid_reads():
        for _ in range(50):
            simple_adc.read_sensors()

    results.append(run_test("Rapid read_sensors() (50x)", test_rapid_reads))
    print_result(results[-1])

    def test_rapid_channel_switching():
        for _ in range(20):
            for ch in range(1, 9):
                simple_adc.read_channel("b1", ch)

    results.append(run_test("Rapid channel switching (20x8)", test_rapid_channel_switching))
    print_result(results[-1])

    # --- get_board_addresses unknown board ---
    def test_get_board_addresses_unknown():
        cfg = ADCConfig()
        cfg.get_board_addresses("unknown_board")

    results.append(run_test("get_board_addresses() unknown board", test_get_board_addresses_unknown, expect_exception=ValueError))
    print_result(results[-1])

    # =========================================================================
    # PART 5: Type Confusion / Weird Inputs
    # =========================================================================
    print("\n=== PART 5: Type Confusion / Weird Inputs ===\n")

    def test_read_voltage_string_channel():
        adc_low.read_voltage("1")

    results.append(run_test("read_voltage('1') string", test_read_voltage_string_channel, expect_exception=Exception))
    print_result(results[-1])

    def test_read_voltage_float_channel():
        adc_low.read_voltage(1.5)  # Should convert to int or fail

    # This might work (int(1.5) = 1) or fail - let's see
    results.append(run_test("read_voltage(1.5) float", test_read_voltage_float_channel))
    print_result(results[-1])

    def test_read_voltage_none_channel():
        adc_low.read_voltage(None)

    results.append(run_test("read_voltage(None)", test_read_voltage_none_channel, expect_exception=Exception))
    print_result(results[-1])

    def test_set_pga_string():
        adc_low.set_pga("1")

    results.append(run_test("set_pga('1') string", test_set_pga_string, expect_exception=Exception))
    print_result(results[-1])

    def test_set_bit_rate_string():
        adc_low.set_bit_rate("12")

    results.append(run_test("set_bit_rate('12') string", test_set_bit_rate_string, expect_exception=Exception))
    print_result(results[-1])

    def test_read_channel_string_channel():
        simple_adc.read_channel("b1", "1")

    results.append(run_test("read_channel() string channel", test_read_channel_string_channel, expect_exception=Exception))
    print_result(results[-1])

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST SUMMARY")
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
                print(f"  - {r.name}: {r.message[:70]}")
        return 1
    else:
        print("All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
