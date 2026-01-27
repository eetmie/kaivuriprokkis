#!/usr/bin/env python3
"""
PWM Controller Functional Tests - Oscilloscope Verification

These tests target the 'scoop' channel for visual verification on an oscilloscope.
Each test pauses to allow observation of the PWM signal.

Scoop channel config (from servo_config.yaml):
- output_channel: 2
- center: 1395 us
- pulse_min: 1170 us (extend)
- pulse_max: 1620 us (retract)
- direction: -1

Run individual tests:
    python -m pytest tests/test_pwm_functional.py::TestScoopChannel::test_center_pulse -v -s

Run all with pauses for observation:
    python tests/test_pwm_functional.py

Run quick automated tests (no pauses):
    python -m pytest tests/test_pwm_functional.py -v
"""

import sys
import time
import math
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.PCA9685_controller import PWMController, PWMConstants, DirectPWMWriter


# Set to True for interactive testing with oscilloscope observation pauses
INTERACTIVE_MODE = False
PAUSE_DURATION = 2.0  # seconds to hold each state for observation


def observe(message: str, duration: float = PAUSE_DURATION):
    """Pause for oscilloscope observation in interactive mode."""
    if INTERACTIVE_MODE:
        print(f"\n>>> OBSERVE: {message}")
        print(f"    (holding for {duration}s...)")
        time.sleep(duration)


class TestScoopChannel(unittest.TestCase):
    """Functional tests for scoop channel - verify on oscilloscope."""

    @classmethod
    def setUpClass(cls):
        cls.config_file = "configuration_files/servo_config.yaml"
        cls.controller = PWMController(
            config_file=cls.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            stale_timeout_s=0.0,
            log_level="INFO"
        )
        cls.scoop_cfg = cls.controller.channel_configs['scoop']
        print(f"\nScoop channel config:")
        print(f"  output_channel: {cls.scoop_cfg.output_channel}")
        print(f"  center: {cls.scoop_cfg.center} us")
        print(f"  pulse_min: {cls.scoop_cfg.pulse_min} us")
        print(f"  pulse_max: {cls.scoop_cfg.pulse_max} us")
        print(f"  direction: {cls.scoop_cfg.direction}")
        print(f"  PWM frequency: {cls.controller._direct_writer.frequency} Hz")
        print(f"  PWM period: {cls.controller._pwm_period_us:.1f} us")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'controller'):
            cls.controller.reset(reset_pump=True)

    def setUp(self):
        self.controller.reset(reset_pump=True)
        time.sleep(0.05)

    def test_center_pulse(self):
        """Output center pulse - should see ~1395us on scope."""
        self.controller.update_named({'scoop': 0.0})
        expected_pulse = self.scoop_cfg.center
        observe(f"CENTER pulse: ~{expected_pulse:.0f}us expected")

        # Verify internal state
        computed = self.controller.compute_pulse('scoop', 0.0)
        self.assertAlmostEqual(computed, expected_pulse, delta=1.0,
            msg=f"Center pulse should be {expected_pulse}us, got {computed}us")

    def test_min_pulse_full_negative(self):
        """Output minimum pulse (full negative command)."""
        # direction=-1, so positive input gives pulse_min
        self.controller.update_named({'scoop': 1.0})
        expected_pulse = self.scoop_cfg.pulse_min
        observe(f"MIN pulse (cmd=+1.0): ~{expected_pulse:.0f}us expected")

        computed = self.controller.compute_pulse('scoop', 1.0)
        # With deadband and dither, actual pulse will be above pulse_min
        # Account for dither amplitude which can swing the pulse
        dither_margin = self.scoop_cfg.dither_amp_us if self.scoop_cfg.dither_enable else 0
        self.assertLessEqual(computed, expected_pulse + 10 + dither_margin,
            msg=f"Full negative should approach pulse_min ({expected_pulse}us)")

    def test_max_pulse_full_positive(self):
        """Output maximum pulse (full positive command)."""
        # direction=-1, so negative input gives pulse_max
        self.controller.update_named({'scoop': -1.0})
        expected_pulse = self.scoop_cfg.pulse_max
        observe(f"MAX pulse (cmd=-1.0): ~{expected_pulse:.0f}us expected")

        computed = self.controller.compute_pulse('scoop', -1.0)
        # With deadband and dither, actual pulse will be below pulse_max
        # Account for dither amplitude which can swing the pulse
        dither_margin = self.scoop_cfg.dither_amp_us if self.scoop_cfg.dither_enable else 0
        self.assertGreaterEqual(computed, expected_pulse - 10 - dither_margin,
            msg=f"Full positive should approach pulse_max ({expected_pulse}us)")

    def test_half_positive_command(self):
        """Output 50% positive command."""
        self.controller.update_named({'scoop': -0.5})
        observe("HALF positive (cmd=-0.5): between center and max")

        computed = self.controller.compute_pulse('scoop', -0.5)
        self.assertGreater(computed, self.scoop_cfg.center,
            msg="Half positive should be above center")
        self.assertLess(computed, self.scoop_cfg.pulse_max,
            msg="Half positive should be below max")

    def test_half_negative_command(self):
        """Output 50% negative command."""
        self.controller.update_named({'scoop': 0.5})
        observe("HALF negative (cmd=+0.5): between center and min")

        computed = self.controller.compute_pulse('scoop', 0.5)
        self.assertLess(computed, self.scoop_cfg.center,
            msg="Half negative should be below center")
        self.assertGreater(computed, self.scoop_cfg.pulse_min,
            msg="Half negative should be above min")

    def test_reset_returns_to_center(self):
        """After reset, output should return to center."""
        # First move to max
        self.controller.update_named({'scoop': -1.0})
        observe("At MAX before reset")

        # Reset
        self.controller.reset(reset_pump=False)
        observe(f"After RESET: should be at center ~{self.scoop_cfg.center:.0f}us")

        # Ramp state should be at center
        ramp_state = self.controller._channel_ramp_state.get(self.scoop_cfg.output_channel)
        self.assertIsNotNone(ramp_state)
        self.assertAlmostEqual(ramp_state[0], self.scoop_cfg.center, delta=1.0)

    def test_command_sweep(self):
        """Sweep from min to max - visual verification of smooth transition."""
        print("\n  Sweeping scoop from -1.0 to +1.0...")
        steps = 20
        for i in range(steps + 1):
            cmd = -1.0 + (2.0 * i / steps)  # -1.0 to +1.0
            self.controller.update_named({'scoop': cmd})
            if INTERACTIVE_MODE:
                pulse = self.controller.compute_pulse('scoop', cmd)
                print(f"    cmd={cmd:+.2f} -> ~{pulse:.0f}us")
            time.sleep(0.05)

        observe("Sweep complete - should have seen smooth transition")

    def test_rapid_updates(self):
        """Rapid command updates - verify no glitches on scope."""
        print("\n  Rapid updates (100 iterations)...")
        for i in range(100):
            cmd = math.sin(i * 0.1) * 0.5  # Oscillate between -0.5 and +0.5
            self.controller.update_named({'scoop': cmd})
            time.sleep(0.005)  # 200 Hz update rate

        self.controller.update_named({'scoop': 0.0})
        observe("After rapid updates - should be at center, no glitches observed")


class TestDirectPWMWriter(unittest.TestCase):
    """Test DirectPWMWriter chip control features."""

    @classmethod
    def setUpClass(cls):
        cls.config_file = "configuration_files/servo_config.yaml"
        cls.controller = PWMController(
            config_file=cls.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            log_level="WARNING"
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'controller'):
            # Ensure chip is awake before cleanup
            cls.controller._direct_writer.wake()
            cls.controller.reset(reset_pump=True)

    def setUp(self):
        # Ensure chip is awake
        self.controller._direct_writer.wake()
        self.controller.reset(reset_pump=True)
        time.sleep(0.05)

    def test_sleep_stops_output(self):
        """Sleep mode should stop PWM oscillator - output goes low."""
        # First output a signal
        self.controller.update_named({'scoop': 0.5})
        observe("Signal active before sleep")

        # Sleep
        self.controller._direct_writer.sleep()
        observe("SLEEP mode: output should go LOW (no PWM)")

        # Wake
        self.controller._direct_writer.wake()
        self.controller.update_named({'scoop': 0.5})
        observe("After WAKE: signal should be active again")

    def test_wake_restores_output(self):
        """Wake from sleep should restore PWM output."""
        self.controller.update_named({'scoop': -0.5})

        # Sleep and wake cycle
        self.controller._direct_writer.sleep()
        time.sleep(0.1)
        self.controller._direct_writer.wake()

        # Re-send command (registers may need refresh after wake)
        self.controller.update_named({'scoop': -0.5})
        observe("After sleep/wake cycle: output should be restored")

    def test_frequency_setting(self):
        """Verify PWM frequency is set correctly."""
        expected_freq = PWMConstants.PWM_FREQUENCY_DEFAULT
        actual_freq = self.controller._direct_writer.frequency

        self.assertEqual(actual_freq, expected_freq,
            f"PWM frequency should be {expected_freq}Hz, got {actual_freq}Hz")

        # Verify period calculation
        expected_period = 1e6 / expected_freq
        actual_period = self.controller._pwm_period_us

        self.assertAlmostEqual(actual_period, expected_period, delta=1.0,
            msg=f"PWM period should be {expected_period}us, got {actual_period}us")


class TestPulseCalculation(unittest.TestCase):
    """Test pulse width calculations match expected values."""

    @classmethod
    def setUpClass(cls):
        cls.config_file = "configuration_files/servo_config.yaml"
        cls.controller = PWMController(
            config_file=cls.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            log_level="WARNING"
        )
        cls.scoop_cfg = cls.controller.channel_configs['scoop']

    def test_pulse_range_coverage(self):
        """Verify commands cover the full pulse range."""
        min_pulse = self.controller.compute_pulse('scoop', 1.0)
        max_pulse = self.controller.compute_pulse('scoop', -1.0)
        center_pulse = self.controller.compute_pulse('scoop', 0.0)

        # Min pulse should be near pulse_min (accounting for deadband)
        self.assertLess(min_pulse, center_pulse,
            "min pulse should be less than center")

        # Max pulse should be near pulse_max (accounting for deadband)
        self.assertGreater(max_pulse, center_pulse,
            "max pulse should be greater than center")

        # Range should be reasonable
        actual_range = max_pulse - min_pulse
        config_range = self.scoop_cfg.pulse_max - self.scoop_cfg.pulse_min
        self.assertGreater(actual_range, config_range * 0.5,
            f"Actual range {actual_range}us should cover >50% of config range {config_range}us")

    def test_deadband_applied(self):
        """Verify deadband jumps over dead zone."""
        cfg = self.scoop_cfg
        center = cfg.center
        # Use a value clearly outside deadzone (deadzone is typically 1-2%)
        test_cmd = 0.1  # 10% - well above any reasonable deadzone

        # Small positive command should jump past positive deadband
        small_pos = self.controller.compute_pulse('scoop', -test_cmd)
        if cfg.deadband_us_pos > 0:
            self.assertGreater(small_pos, center + cfg.deadband_us_pos * 0.5,
                "Small positive should jump past deadband")

        # Small negative command should jump past negative deadband
        small_neg = self.controller.compute_pulse('scoop', test_cmd)
        if cfg.deadband_us_neg > 0:
            self.assertLess(small_neg, center - cfg.deadband_us_neg * 0.5,
                "Small negative should jump past deadband")

    def test_direction_inversion(self):
        """Verify direction=-1 inverts command mapping."""
        # With direction=-1:
        # positive input -> pulse_min (extend)
        # negative input -> pulse_max (retract)
        self.assertEqual(self.scoop_cfg.direction, -1,
            "Scoop should have direction=-1")

        pos_pulse = self.controller.compute_pulse('scoop', 1.0)
        neg_pulse = self.controller.compute_pulse('scoop', -1.0)

        self.assertLess(pos_pulse, self.scoop_cfg.center,
            "Positive command with dir=-1 should give pulse < center")
        self.assertGreater(neg_pulse, self.scoop_cfg.center,
            "Negative command with dir=-1 should give pulse > center")


class TestDutyCycleConversion(unittest.TestCase):
    """Test duty cycle conversion accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.config_file = "configuration_files/servo_config.yaml"
        cls.controller = PWMController(
            config_file=cls.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            log_level="WARNING"
        )

    def test_duty_cycle_range(self):
        """Verify duty cycle stays within valid range."""
        for cmd in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            self.controller.update_named({'scoop': cmd})

            # Check queued duty cycle
            channel = self.controller.channel_configs['scoop'].output_channel
            duty = self.controller._direct_writer._duty_cycles[channel]

            self.assertGreaterEqual(duty, 0,
                f"Duty cycle for cmd={cmd} should be >= 0")
            self.assertLessEqual(duty, PWMConstants.DUTY_CYCLE_MAX,
                f"Duty cycle for cmd={cmd} should be <= {PWMConstants.DUTY_CYCLE_MAX}")

    def test_center_duty_cycle(self):
        """Verify center pulse produces expected duty cycle."""
        self.controller.update_named({'scoop': 0.0})

        cfg = self.controller.channel_configs['scoop']
        channel = cfg.output_channel
        duty = self.controller._direct_writer._duty_cycles[channel]

        # Expected duty = (center_us / period_us) * 65535
        period_us = self.controller._pwm_period_us
        expected_duty = int((cfg.center / period_us) * PWMConstants.DUTY_CYCLE_MAX)

        # Allow some tolerance for rounding
        self.assertAlmostEqual(duty, expected_duty, delta=100,
            msg=f"Center duty should be ~{expected_duty}, got {duty}")


class TestDeadzoneRescaling(unittest.TestCase):
    """Test deadzone rescaling behavior - values should use full range after threshold."""

    @classmethod
    def setUpClass(cls):
        cls.config_file = "configuration_files/servo_config.yaml"
        cls.controller = PWMController(
            config_file=cls.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            log_level="WARNING"
        )

    def test_deadzone_threshold_calculation(self):
        """Verify deadzone threshold is calculated as percentage per direction."""
        cfg = self.controller.channel_configs['scoop']
        # deadzone in config is percentage, threshold should be deadzone/100
        expected_threshold = cfg.deadzone / 100.0
        self.assertAlmostEqual(cfg.deadzone_threshold, expected_threshold, places=6,
            msg=f"Deadzone threshold should be {expected_threshold}, got {cfg.deadzone_threshold}")

    def test_deadzone_cuts_small_values(self):
        """Values within deadzone should map to center pulse."""
        cfg = self.controller.channel_configs['scoop']
        threshold = cfg.deadzone_threshold

        # Value just inside deadzone should give center
        inside_value = threshold * 0.5
        pulse = self.controller.compute_pulse('scoop', inside_value)
        self.assertEqual(pulse, cfg.center,
            f"Value {inside_value} inside deadzone should give center pulse")

        pulse_neg = self.controller.compute_pulse('scoop', -inside_value)
        self.assertEqual(pulse_neg, cfg.center,
            f"Value {-inside_value} inside deadzone should give center pulse")

    def test_deadzone_rescales_to_full_range(self):
        """Values outside deadzone should rescale to use full output range."""
        cfg = self.controller.channel_configs['scoop']
        threshold = cfg.deadzone_threshold

        # Full command (1.0) should still reach full pulse range
        pulse_max = self.controller.compute_pulse('scoop', -1.0)  # dir=-1, so -1 gives max
        pulse_min = self.controller.compute_pulse('scoop', 1.0)   # dir=-1, so +1 gives min

        # With rescaling, full command should approach pulse limits
        # (accounting for deadband offset)
        self.assertGreater(pulse_max, cfg.center + cfg.deadband_us_pos,
            "Full positive should exceed deadband edge")
        self.assertLess(pulse_min, cfg.center - cfg.deadband_us_neg,
            "Full negative should be below deadband edge")

    def test_deadzone_rescale_small_command(self):
        """Small command just above threshold should give small output."""
        cfg = self.controller.channel_configs['scoop']
        threshold = cfg.deadzone_threshold

        # Value just above threshold
        just_above = threshold + 0.05  # 5% above threshold
        pulse = self.controller.compute_pulse('scoop', just_above)

        # After rescaling: (0.05) / (1 - threshold) ≈ 0.05 (small)
        # This should give a pulse just past the deadband edge, not a big jump
        center = cfg.center
        deadband_edge = center - cfg.deadband_us_neg  # positive input with dir=-1 goes negative

        # The pulse should be between deadband edge and pulse_min
        self.assertLess(pulse, deadband_edge,
            "Small command should be past deadband edge")
        self.assertGreater(pulse, cfg.pulse_min,
            "Small command should not reach pulse_min")


class TestRampSkipDeadband(unittest.TestCase):
    """Test ramp_skip_deadband feature - deadband jump should be instant."""

    @classmethod
    def setUpClass(cls):
        cls.config_file = "configuration_files/servo_config.yaml"
        cls.controller = PWMController(
            config_file=cls.config_file,
            pump_variable=False,
            toggle_channels=True,
            input_rate_threshold=0,
            log_level="WARNING"
        )
        cls.scoop_cfg = cls.controller.channel_configs['scoop']

    def setUp(self):
        self.controller.reset(reset_pump=True)
        time.sleep(0.01)

    def test_ramp_skip_deadband_config_loaded(self):
        """Verify ramp_skip_deadband config is loaded correctly."""
        cfg = self.scoop_cfg
        # Should be a boolean attribute
        self.assertIsInstance(cfg.ramp_skip_deadband, bool,
            "ramp_skip_deadband should be a boolean")

    def test_ramp_with_skip_deadband_enabled(self):
        """When ramp_skip_deadband is True, deadband jump should be instant."""
        cfg = self.scoop_cfg
        center = cfg.center
        pos_edge = center + cfg.deadband_us_pos
        neg_edge = center - cfg.deadband_us_neg

        # Temporarily enable ramp and skip_deadband for this test
        original_ramp_enable = cfg.ramp_enable
        original_skip = cfg.ramp_skip_deadband
        original_limit = cfg.ramp_limit

        try:
            # Enable ramp with skip_deadband
            cfg.ramp_enable = True
            cfg.ramp_skip_deadband = True
            cfg.ramp_limit = 100.0  # Slow ramp to make the effect visible

            # Use a fixed timestamp to avoid timing issues
            now = 1000.0  # arbitrary fixed time
            self.controller._channel_ramp_state[cfg.output_channel] = (center, now, 0.0)

            # Compute pulse (this triggers the ramp logic)
            now2 = now + 0.001  # 1ms later
            self.controller._pulse_from_value(cfg, 0.5, now2, apply_ramp=True)

            # Check the ramp state directly (avoids dither interference)
            # With skip_deadband, should have jumped from center to neg_edge, then ramped slightly
            ramp_state = self.controller._channel_ramp_state[cfg.output_channel]
            ramped_pulse = ramp_state[0]

            # With 100us/s ramp and 1ms dt, max move from edge is 0.1us
            # So ramped_pulse should be very close to neg_edge (within 1us)
            self.assertLessEqual(ramped_pulse, neg_edge + 0.5,
                f"With skip_deadband, ramp state should be at/past deadband edge ({neg_edge}), got {ramped_pulse}")

        finally:
            # Restore original config
            cfg.ramp_enable = original_ramp_enable
            cfg.ramp_skip_deadband = original_skip
            cfg.ramp_limit = original_limit

    def test_ramp_without_skip_deadband(self):
        """When ramp_skip_deadband is False, deadband jump should be slew-limited."""
        cfg = self.scoop_cfg
        center = cfg.center

        # Temporarily enable ramp without skip_deadband
        original_ramp_enable = cfg.ramp_enable
        original_skip = cfg.ramp_skip_deadband
        original_limit = cfg.ramp_limit

        try:
            cfg.ramp_enable = True
            cfg.ramp_skip_deadband = False
            cfg.ramp_limit = 100.0  # Very slow ramp: 100 us/s

            # Use fixed timestamps for reproducibility
            now = 1000.0
            self.controller._channel_ramp_state[cfg.output_channel] = (center, now, 0.0)

            # Small time step: 10ms
            now2 = now + 0.01
            pulse = self.controller._pulse_from_value(cfg, 0.5, now2, apply_ramp=True)

            # With slow ramp (100 us/s) and no skip, should only move 1us in 10ms
            # max step = 100 us/s * 0.01s = 1us
            actual_move = abs(pulse - center)
            self.assertLess(actual_move, 2.0,
                f"Without skip_deadband, move should be slew-limited to ~1us (moved {actual_move}us)")

        finally:
            cfg.ramp_enable = original_ramp_enable
            cfg.ramp_skip_deadband = original_skip
            cfg.ramp_limit = original_limit


def run_interactive():
    """Run tests in interactive mode with pauses for oscilloscope observation."""
    global INTERACTIVE_MODE, PAUSE_DURATION
    INTERACTIVE_MODE = True
    PAUSE_DURATION = 3.0

    print("=" * 60)
    print("PWM FUNCTIONAL TESTS - INTERACTIVE MODE")
    print("Monitoring: SCOOP channel (output_channel=2)")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Run scoop channel tests
    suite.addTests(loader.loadTestsFromTestCase(TestScoopChannel))
    suite.addTests(loader.loadTestsFromTestCase(TestDirectPWMWriter))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


def run_automated():
    """Run all tests in automated mode (no pauses)."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestScoopChannel))
    suite.addTests(loader.loadTestsFromTestCase(TestDirectPWMWriter))
    suite.addTests(loader.loadTestsFromTestCase(TestPulseCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestDutyCycleConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestDeadzoneRescaling))
    suite.addTests(loader.loadTestsFromTestCase(TestRampSkipDeadband))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PWM Functional Tests')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode with pauses for oscilloscope observation')
    parser.add_argument('--pause', '-p', type=float, default=3.0,
                        help='Pause duration in seconds for interactive mode (default: 3.0)')
    args = parser.parse_args()

    if args.interactive:
        PAUSE_DURATION = args.pause
        sys.exit(run_interactive())
    else:
        sys.exit(run_automated())
