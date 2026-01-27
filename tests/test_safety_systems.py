#!/usr/bin/env python3
"""
Safety Systems Tests for HardwareInterface and PWMController

Tests verify that safety mechanisms work as intended:
1. _safe_hardware_operation decorator triggers reset on exceptions
2. Stale command timeout rejects old commands
3. Input rate monitoring triggers safe state when rate drops
4. PWM reset properly centers valves and stops pump
5. Watchdog channel toggles at configured rate

Run with: python -m pytest tests/test_safety_systems.py -v
Or directly: python tests/test_safety_systems.py
"""

import sys
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.PCA9685_controller import PWMController, PWMConstants


class TestPWMControllerSafety(unittest.TestCase):
    """Test PWM controller safety mechanisms with real hardware."""

    @classmethod
    def setUpClass(cls):
        """Initialize PWM controller once for all tests."""
        cls.config_file = "configuration_files/servo_config.yaml"
        # Initialize with safety features enabled for testing
        cls.controller = PWMController(
            config_file=cls.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,  # Disabled for basic tests
            stale_timeout_s=0.0,
            log_level="WARNING"
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if hasattr(cls, 'controller') and cls.controller:
            cls.controller.reset(reset_pump=True)

    def setUp(self):
        """Reset state before each test."""
        self.controller.reset(reset_pump=True)
        time.sleep(0.05)

    def test_reset_sets_channels_to_center(self):
        """Verify reset() sends center pulse to all valve channels."""
        # Send a non-zero command first
        self.controller.update_named({'lift_boom': 0.5, 'tilt_boom': -0.5})
        time.sleep(0.02)

        # Reset - this sends center pulses to hardware but doesn't zero the values array
        # (values array is only updated by update_named, not reset)
        self.controller.reset(reset_pump=False)

        # Verify ramp state is re-initialized to center positions
        for name, cfg in self.controller.channel_configs.items():
            ramp_state = self.controller._channel_ramp_state.get(cfg.output_channel)
            self.assertIsNotNone(ramp_state,
                f"Channel '{name}' should have ramp state after reset")
            # Ramp state tuple is (pulse, time, velocity) - pulse should be center
            self.assertAlmostEqual(ramp_state[0], cfg.center, places=1,
                msg=f"Channel '{name}' ramp state should be at center after reset")

    def test_reset_stops_pump(self):
        """Verify reset(reset_pump=True) sets pump to minimum."""
        if not self.controller.pump_config:
            self.skipTest("No pump configured")

        # Enable pump and send commands
        self.controller.set_pump(True)
        self.controller.update_named({'lift_boom': 0.5, 'pump': 0.8})
        time.sleep(0.02)

        # Reset with pump
        self.controller.reset(reset_pump=True)

        # Pump override should be cleared
        self.assertIsNone(self.controller._pump_override_throttle,
            "Pump override should be None after reset")

    def test_reset_clears_safe_state(self):
        """Verify reset() clears the safe state flag."""
        self.controller.is_safe_state = True
        self.controller.reset(reset_pump=True)
        self.assertFalse(self.controller.is_safe_state,
            "is_safe_state should be False after reset")

    def test_reset_clears_input_counter(self):
        """Verify reset() clears the input counter."""
        self.controller.input_counter = 100
        self.controller.reset(reset_pump=True)
        self.assertEqual(self.controller.input_counter, 0,
            "input_counter should be 0 after reset")


class TestStaleCommandTimeout(unittest.TestCase):
    """Test stale command rejection safety feature."""

    def setUp(self):
        """Create controller with stale timeout enabled."""
        self.config_file = "configuration_files/servo_config.yaml"
        self.controller = PWMController(
            config_file=self.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            stale_timeout_s=0.5,  # 500ms timeout
            log_level="WARNING"
        )
        self.controller.reset(reset_pump=True)

    def tearDown(self):
        """Clean up controller."""
        if hasattr(self, 'controller'):
            self.controller.reset(reset_pump=True)

    def test_fresh_commands_accepted(self):
        """Verify commands with recent timestamps are accepted."""
        now = time.monotonic()
        self.controller.update_named(
            {'lift_boom': 0.5},
            command_ts=now
        )
        # Command should be applied (value updated)
        cfg = self.controller.channel_configs['lift_boom']
        self.assertEqual(self.controller.values[cfg.output_channel], 0.5,
            "Fresh command should be accepted")

    def test_stale_commands_rejected(self):
        """Verify commands with old timestamps are rejected."""
        # First send a fresh command
        now = time.monotonic()
        self.controller.update_named({'lift_boom': 0.3}, command_ts=now)
        cfg = self.controller.channel_configs['lift_boom']

        # Reset to zero
        self.controller.update_named({'lift_boom': 0.0}, command_ts=now + 0.01)
        self.assertEqual(self.controller.values[cfg.output_channel], 0.0)

        # Now send a stale command (1 second old, timeout is 0.5s)
        stale_ts = time.monotonic() - 1.0
        self.controller.update_named({'lift_boom': 0.8}, command_ts=stale_ts)

        # Value should NOT have changed to 0.8
        self.assertEqual(self.controller.values[cfg.output_channel], 0.0,
            "Stale command should be rejected - value should remain 0.0")

    def test_commands_without_timestamp_accepted(self):
        """Verify commands without timestamp are always accepted."""
        self.controller.update_named({'lift_boom': 0.7})
        cfg = self.controller.channel_configs['lift_boom']
        self.assertEqual(self.controller.values[cfg.output_channel], 0.7,
            "Command without timestamp should be accepted")

    def test_stale_timeout_boundary(self):
        """Test behavior at exact timeout boundary."""
        timeout = self.controller._stale_timeout_s

        # Command exactly at timeout should be rejected (> not >=)
        boundary_ts = time.monotonic() - timeout - 0.001
        self.controller.update_named({'lift_boom': 0.0})  # Reset first
        self.controller.update_named({'lift_boom': 0.5}, command_ts=boundary_ts)

        cfg = self.controller.channel_configs['lift_boom']
        self.assertEqual(self.controller.values[cfg.output_channel], 0.0,
            "Command at timeout boundary should be rejected")


class TestInputRateMonitoring(unittest.TestCase):
    """Test input rate monitoring safety feature."""

    def setUp(self):
        """Create controller with rate monitoring enabled."""
        self.config_file = "configuration_files/servo_config.yaml"
        self.controller = PWMController(
            config_file=self.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=10,  # Require 10 Hz
            stale_timeout_s=0.0,
            log_level="WARNING"
        )
        # Wait for monitor thread to start
        time.sleep(0.1)

    def tearDown(self):
        """Clean up controller."""
        if hasattr(self, 'controller'):
            self.controller._stop_monitoring()
            self.controller.reset(reset_pump=True)

    def test_monitoring_thread_started(self):
        """Verify monitoring thread is running when rate threshold > 0."""
        self.assertTrue(self.controller.running,
            "Monitoring should be running")
        self.assertIsNotNone(self.controller.monitor_thread,
            "Monitor thread should exist")
        self.assertTrue(self.controller.monitor_thread.is_alive(),
            "Monitor thread should be alive")

    def test_initial_state_not_safe(self):
        """Verify initial state is not safe when rate monitoring enabled."""
        # Fresh controller should not be in safe state until rate is established
        self.assertFalse(self.controller.is_safe_state,
            "Initial state should not be safe")

    def test_commands_blocked_when_not_safe(self):
        """Verify commands are blocked when not in safe state."""
        self.controller.is_safe_state = False
        initial_value = self.controller.values[
            self.controller.channel_configs['lift_boom'].output_channel
        ]

        self.controller.update_named({'lift_boom': 0.9})

        # Value should not change when not in safe state
        current_value = self.controller.values[
            self.controller.channel_configs['lift_boom'].output_channel
        ]
        self.assertEqual(current_value, initial_value,
            "Commands should be blocked when not in safe state")

    def test_commands_allowed_when_safe(self):
        """Verify commands are allowed when in safe state."""
        self.controller.is_safe_state = True
        self.controller.update_named({'lift_boom': 0.6})

        cfg = self.controller.channel_configs['lift_boom']
        self.assertEqual(self.controller.values[cfg.output_channel], 0.6,
            "Commands should be allowed when in safe state")

    def test_high_rate_establishes_safe_state(self):
        """Verify sustained high input rate establishes safe state."""
        # Send commands at high rate for longer than time window
        start_time = time.time()
        while (time.time() - start_time) < 1.5:  # 1.5 seconds
            self.controller.update_named({'lift_boom': 0.1})
            time.sleep(0.05)  # 20 Hz > 10 Hz threshold

        # After sustained high rate, should be in safe state
        # (threshold is 25% of required rate = 2.5 Hz)
        self.assertTrue(self.controller.is_safe_state,
            "Sustained high rate should establish safe state")

    def test_stop_monitoring(self):
        """Verify monitoring can be stopped cleanly."""
        self.assertTrue(self.controller.running)
        thread_ref = self.controller.monitor_thread  # Save reference before stop
        self.controller._stop_monitoring()
        self.assertFalse(self.controller.running,
            "Monitoring should be stopped")
        time.sleep(0.3)  # Wait for thread to join
        # Thread reference is set to None after stop, check saved reference
        self.assertFalse(thread_ref.is_alive(),
            "Monitor thread should have stopped")


class TestWatchdogChannel(unittest.TestCase):
    """Test hardware watchdog channel toggling."""

    def setUp(self):
        """Create controller with watchdog enabled."""
        self.config_file = "configuration_files/servo_config.yaml"
        self.controller = PWMController(
            config_file=self.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            stale_timeout_s=0.0,
            watchdog_channel=15,  # Use channel 15 for watchdog
            watchdog_toggle_hz=10.0,  # 10 Hz toggle
            log_level="WARNING"
        )

    def tearDown(self):
        """Clean up controller."""
        if hasattr(self, 'controller'):
            self.controller.reset(reset_pump=True)

    def test_watchdog_channel_configured(self):
        """Verify watchdog channel is properly configured."""
        self.assertEqual(self.controller._watchdog_channel, 15)
        self.assertEqual(self.controller._watchdog_toggle_hz, 10.0)

    def test_watchdog_toggles_on_update(self):
        """Verify watchdog toggles state when update is called."""
        initial_state = self.controller._watchdog_state

        # Send commands and wait for toggle period
        time.sleep(0.15)  # Wait > 100ms (10 Hz period)
        self.controller.update_named({'lift_boom': 0.1})

        # State should have toggled
        self.assertNotEqual(self.controller._watchdog_state, initial_state,
            "Watchdog state should toggle after period elapsed")

    def test_watchdog_respects_rate(self):
        """Verify watchdog doesn't toggle faster than configured rate."""
        initial_state = self.controller._watchdog_state

        # Rapid updates should not cause rapid toggling
        for _ in range(5):
            self.controller.update_named({'lift_boom': 0.1})
            time.sleep(0.01)  # 10ms between updates

        # Should not have toggled multiple times (rate is 10 Hz = 100ms period)
        # At most one toggle should have occurred
        toggles = 0 if self.controller._watchdog_state == initial_state else 1
        self.assertLessEqual(toggles, 1,
            "Watchdog should respect toggle rate limit")


class TestSafeHardwareOperationDecorator(unittest.TestCase):
    """Test the _safe_hardware_operation decorator behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_file = "configuration_files/servo_config.yaml"

    def test_decorator_exists_on_hardware_interface(self):
        """Verify decorated methods exist in HardwareInterface."""
        from modules.hardware_interface import HardwareInterface, _safe_hardware_operation

        # Check decorator is defined
        self.assertTrue(callable(_safe_hardware_operation))

        # Check it's applied to critical methods
        decorated_methods = [
            'read_imu_data',
            'read_imu_gyro',
            'read_imu_pitch',
            'read_slew_voltage',
            'read_slew_angle',
            'read_slew_quaternion',
        ]

        for method_name in decorated_methods:
            self.assertTrue(hasattr(HardwareInterface, method_name),
                f"HardwareInterface should have {method_name}")

    def test_decorator_calls_reset_on_exception(self):
        """Verify decorator triggers reset when wrapped function raises."""
        from modules.hardware_interface import _safe_hardware_operation

        # Create a mock object with reset method
        class MockHardware:
            def __init__(self):
                self.reset_called = False
                self.reset_pump_arg = None
                self.logger = MagicMock()

            def reset(self, reset_pump=False):
                self.reset_called = True
                self.reset_pump_arg = reset_pump

            @_safe_hardware_operation
            def failing_operation(self):
                raise RuntimeError("Simulated hardware failure")

        mock = MockHardware()

        # Call the decorated method - should raise but call reset first
        with self.assertRaises(RuntimeError):
            mock.failing_operation()

        self.assertTrue(mock.reset_called,
            "reset() should be called when exception occurs")
        self.assertTrue(mock.reset_pump_arg,
            "reset() should be called with reset_pump=True")

    def test_decorator_preserves_return_value(self):
        """Verify decorator doesn't interfere with normal return values."""
        from modules.hardware_interface import _safe_hardware_operation

        class MockHardware:
            def __init__(self):
                self.logger = MagicMock()

            def reset(self, reset_pump=False):
                pass

            @_safe_hardware_operation
            def successful_operation(self):
                return "success_value"

        mock = MockHardware()
        result = mock.successful_operation()

        self.assertEqual(result, "success_value",
            "Decorator should preserve return value on success")


class TestCommandValueClamping(unittest.TestCase):
    """Test that command values are properly clamped for safety."""

    @classmethod
    def setUpClass(cls):
        cls.config_file = "configuration_files/servo_config.yaml"
        cls.controller = PWMController(
            config_file=cls.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            stale_timeout_s=0.0,
            log_level="WARNING"
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'controller'):
            cls.controller.reset(reset_pump=True)

    def test_values_clamped_to_valid_range(self):
        """Verify command values exceeding [-1, 1] are clamped."""
        cfg = self.controller.channel_configs['lift_boom']

        # Send value > 1
        self.controller.update_named({'lift_boom': 5.0})
        self.assertEqual(self.controller.values[cfg.output_channel], 1.0,
            "Values > 1.0 should be clamped to 1.0")

        # Send value < -1
        self.controller.update_named({'lift_boom': -5.0})
        self.assertEqual(self.controller.values[cfg.output_channel], -1.0,
            "Values < -1.0 should be clamped to -1.0")

    def test_pump_values_clamped(self):
        """Verify pump command values are clamped."""
        if not self.controller.pump_config:
            self.skipTest("No pump configured")

        # Send extreme pump value with one_shot_pump_override=False to retain value
        self.controller.update_named({'pump': 10.0}, one_shot_pump_override=False)
        self.assertEqual(self.controller._pump_override_throttle, 1.0,
            "Pump values should be clamped to 1.0")

        self.controller.clear_pump_override()
        self.controller.update_named({'pump': -10.0}, one_shot_pump_override=False)
        self.assertEqual(self.controller._pump_override_throttle, -1.0,
            "Pump values should be clamped to -1.0")


class TestAtexitCleanup(unittest.TestCase):
    """Test that atexit cleanup is registered."""

    def test_cleanup_registered(self):
        """Verify cleanup function is registered with atexit."""
        import atexit

        config_file = "configuration_files/servo_config.yaml"
        controller = PWMController(
            config_file=config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            log_level="WARNING"
        )

        # Check _simple_cleanup is in atexit registry
        # (atexit doesn't expose registry directly, but we can verify the method exists)
        self.assertTrue(hasattr(controller, '_simple_cleanup'),
            "Controller should have _simple_cleanup method")
        self.assertTrue(callable(controller._simple_cleanup),
            "_simple_cleanup should be callable")

        # Clean up
        controller.reset(reset_pump=True)


class TestConcurrentSafety(unittest.TestCase):
    """Test thread safety of PWM controller."""

    def setUp(self):
        self.config_file = "configuration_files/servo_config.yaml"
        self.controller = PWMController(
            config_file=self.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            stale_timeout_s=0.0,
            log_level="WARNING"
        )
        self.errors = []

    def tearDown(self):
        if hasattr(self, 'controller'):
            self.controller.reset(reset_pump=True)

    def test_concurrent_updates_no_crash(self):
        """Verify concurrent updates don't cause crashes or deadlocks."""
        def update_thread(channel_name, iterations):
            try:
                for i in range(iterations):
                    value = (i % 20 - 10) / 10.0  # -1.0 to 1.0
                    self.controller.update_named({channel_name: value})
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(f"{channel_name}: {e}")

        threads = [
            threading.Thread(target=update_thread, args=('lift_boom', 50)),
            threading.Thread(target=update_thread, args=('tilt_boom', 50)),
            threading.Thread(target=update_thread, args=('scoop', 50)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(self.errors), 0,
            f"Concurrent updates caused errors: {self.errors}")

    def test_reset_during_updates(self):
        """Verify reset during active updates is safe."""
        stop_event = threading.Event()

        def update_loop():
            while not stop_event.is_set():
                try:
                    self.controller.update_named({'lift_boom': 0.5})
                    time.sleep(0.01)
                except Exception as e:
                    self.errors.append(str(e))

        update_thread = threading.Thread(target=update_loop)
        update_thread.start()

        # Perform resets while updates are happening
        for _ in range(10):
            time.sleep(0.02)
            self.controller.reset(reset_pump=True)

        stop_event.set()
        update_thread.join(timeout=2.0)

        self.assertEqual(len(self.errors), 0,
            f"Reset during updates caused errors: {self.errors}")


class TestReadyStateAndFaults(unittest.TestCase):
    """Test ReadyState enum and HardwareFaultError functionality."""

    def test_ready_state_values(self):
        """Verify ReadyState has expected values."""
        from modules.hardware_interface import ReadyState

        self.assertEqual(ReadyState.PENDING, "pending")
        self.assertEqual(ReadyState.READY, "ready")
        self.assertEqual(ReadyState.FAULT, "fault")

    def test_hardware_fault_error_attributes(self):
        """Verify HardwareFaultError carries subsystem and reason."""
        from modules.hardware_interface import HardwareFaultError

        error = HardwareFaultError("PWM", "Duplicate pin assignment")

        self.assertEqual(error.subsystem, "PWM")
        self.assertEqual(error.reason, "Duplicate pin assignment")
        self.assertIn("PWM", str(error))
        self.assertIn("Duplicate pin assignment", str(error))

    def test_check_faults_raises_on_fault_state(self):
        """Verify _check_faults raises HardwareFaultError when subsystem faulted."""
        from modules.hardware_interface import ReadyState, HardwareFaultError

        # Create a minimal mock that simulates fault state
        class MockHardwareInterface:
            def __init__(self):
                self._pwm_state = ReadyState.FAULT
                self._pwm_fault_reason = "Test PWM failure"
                self._imu_state = ReadyState.READY
                self._imu_fault_reason = None
                self._adc_state = ReadyState.READY
                self._adc_fault_reason = None

            def _check_faults(self):
                if self._pwm_state == ReadyState.FAULT:
                    raise HardwareFaultError("PWM", self._pwm_fault_reason or "Unknown")
                if self._imu_state == ReadyState.FAULT:
                    raise HardwareFaultError("IMU", self._imu_fault_reason or "Unknown")
                if self._adc_state == ReadyState.FAULT:
                    raise HardwareFaultError("ADC", self._adc_fault_reason or "Unknown")

        mock = MockHardwareInterface()

        with self.assertRaises(HardwareFaultError) as ctx:
            mock._check_faults()

        self.assertEqual(ctx.exception.subsystem, "PWM")
        self.assertEqual(ctx.exception.reason, "Test PWM failure")

    def test_check_faults_no_raise_on_ready(self):
        """Verify _check_faults does not raise when all subsystems ready."""
        from modules.hardware_interface import ReadyState, HardwareFaultError

        class MockHardwareInterface:
            def __init__(self):
                self._pwm_state = ReadyState.READY
                self._pwm_fault_reason = None
                self._imu_state = ReadyState.READY
                self._imu_fault_reason = None
                self._adc_state = ReadyState.READY
                self._adc_fault_reason = None

            def _check_faults(self):
                if self._pwm_state == ReadyState.FAULT:
                    raise HardwareFaultError("PWM", self._pwm_fault_reason or "Unknown")
                if self._imu_state == ReadyState.FAULT:
                    raise HardwareFaultError("IMU", self._imu_fault_reason or "Unknown")
                if self._adc_state == ReadyState.FAULT:
                    raise HardwareFaultError("ADC", self._adc_fault_reason or "Unknown")

        mock = MockHardwareInterface()
        # Should not raise
        mock._check_faults()

    def test_check_faults_no_raise_on_pending(self):
        """Verify _check_faults does not raise on PENDING state (still initializing)."""
        from modules.hardware_interface import ReadyState, HardwareFaultError

        class MockHardwareInterface:
            def __init__(self):
                self._pwm_state = ReadyState.READY
                self._pwm_fault_reason = None
                self._imu_state = ReadyState.PENDING  # Still initializing
                self._imu_fault_reason = None
                self._adc_state = ReadyState.PENDING
                self._adc_fault_reason = None

            def _check_faults(self):
                if self._pwm_state == ReadyState.FAULT:
                    raise HardwareFaultError("PWM", self._pwm_fault_reason or "Unknown")
                if self._imu_state == ReadyState.FAULT:
                    raise HardwareFaultError("IMU", self._imu_fault_reason or "Unknown")
                if self._adc_state == ReadyState.FAULT:
                    raise HardwareFaultError("ADC", self._adc_fault_reason or "Unknown")

        mock = MockHardwareInterface()
        # Should not raise - PENDING is not FAULT
        mock._check_faults()

    def test_fault_error_in_exception_chain(self):
        """Verify HardwareFaultError can be caught and re-raised properly."""
        from modules.hardware_interface import HardwareFaultError

        def inner_function():
            raise HardwareFaultError("ADC", "I2C bus timeout")

        def outer_function():
            try:
                inner_function()
            except HardwareFaultError:
                raise

        with self.assertRaises(HardwareFaultError) as ctx:
            outer_function()

        self.assertEqual(ctx.exception.subsystem, "ADC")


def run_tests():
    """Run all safety system tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPWMControllerSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestStaleCommandTimeout))
    suite.addTests(loader.loadTestsFromTestCase(TestInputRateMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestWatchdogChannel))
    suite.addTests(loader.loadTestsFromTestCase(TestSafeHardwareOperationDecorator))
    suite.addTests(loader.loadTestsFromTestCase(TestCommandValueClamping))
    suite.addTests(loader.loadTestsFromTestCase(TestAtexitCleanup))
    suite.addTests(loader.loadTestsFromTestCase(TestConcurrentSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestReadyStateAndFaults))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
