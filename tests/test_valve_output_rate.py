"""Why the valve writer must tick at the control rate, not the producer's rate.

These tests exercise the real PWMController (with a fake I2C bus) to pin down
the two failure modes that made 30 Hz VLA recording and inference misbehave:

  1. the input-rate gate silently discards every command once the caller drops
     below its threshold, and
  2. the dither is sampled at the caller's rate, so a 33 Hz dither driven at
     30 Hz aliases into a ~3 Hz wobble worth a large fraction of the channel's
     command authority.

Both are properties of *call rate*, which is exactly what
ExcavatorController's direct-command mode fixes by writing every control tick
regardless of when setpoints arrive.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import modules.PCA9685_controller  # noqa: F401  (patch target lives here)
from modules.pwm.controller import PWMController

JETSON_CONFIG = ROOT_DIR / "configuration_files" / "profiles" / "jetson" / "servo_config.yaml"


class _FakeSmbus:
    def __init__(self):
        self.block_writes = []

    def write_byte_data(self, addr, reg, val):
        pass

    def write_i2c_block_data(self, addr, reg, data):
        self.block_writes.append((reg, list(data)))


def _make_pwm(**kw):
    kw.setdefault('input_rate_threshold', 0)
    kw.setdefault('stale_timeout_s', 0.0)
    kw.setdefault('log_level', 'CRITICAL')
    kw.setdefault('cleanup_disable_osc', False)
    bus = _FakeSmbus()
    with patch("modules.PCA9685_controller._open_smbus", return_value=bus):
        pwm = PWMController(config_file=str(JETSON_CONFIG), bus=7, **kw)
    return pwm, bus


# ---------------------------------------------------------------------------
# 1. the input-rate gate
# ---------------------------------------------------------------------------

class TestInputRateGate(unittest.TestCase):
    """The gate is the silent one: no log, no exception, commands just vanish."""

    def test_unsafe_state_discards_commands_without_error(self):
        pwm, bus = _make_pwm(input_rate_threshold=20)
        try:
            pwm.is_safe_state = False
            before = len(bus.block_writes)
            pwm.update_named({'boom': 1.0})       # returns normally...
            self.assertEqual(len(bus.block_writes), before)  # ...and writes nothing
        finally:
            pwm._stop_monitoring()

    def test_safe_state_writes(self):
        pwm, bus = _make_pwm(input_rate_threshold=20)
        try:
            pwm.is_safe_state = True
            before = len(bus.block_writes)
            pwm.update_named({'boom': 1.0})
            self.assertGreater(len(bus.block_writes), before)
        finally:
            pwm._stop_monitoring()

    def test_threshold_zero_disables_the_gate(self):
        """The escape hatch used by the legacy direct-write path."""
        pwm, _ = _make_pwm(input_rate_threshold=0)
        try:
            self.assertTrue(pwm.skip_rate_checking)
        finally:
            pwm._stop_monitoring()

    @pytest.mark.slow
    def test_100hz_caller_stays_safe_but_15hz_caller_does_not(self):
        """Real-timer check of the gate against a 20 Hz threshold."""
        for rate_hz, expect_safe in ((100.0, True), (15.0, False)):
            pwm, _ = _make_pwm(input_rate_threshold=20)
            try:
                period = 1.0 / rate_hz
                deadline = time.perf_counter() + 1.3   # > 2 monitor windows
                next_t = time.perf_counter()
                while time.perf_counter() < deadline:
                    pwm.update_named({'boom': 0.0})
                    next_t += period
                    sleep = next_t - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                self.assertEqual(
                    pwm.is_safe_state, expect_safe,
                    msg=f"{rate_hz:.0f} Hz caller: is_safe_state="
                        f"{pwm.is_safe_state}, expected {expect_safe}")
            finally:
                pwm._stop_monitoring()


# ---------------------------------------------------------------------------
# 2. dither aliasing
# ---------------------------------------------------------------------------

def _dominant_freq_hz(samples, sample_rate_hz):
    """Frequency of the largest non-DC FFT bin."""
    x = np.asarray(samples, dtype=np.float64)
    x = x - x.mean()
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate_hz)
    return float(freqs[int(np.argmax(spectrum[1:])) + 1])


class TestDitherSamplingRate(unittest.TestCase):
    """boom/arm/bucket run dither_enable with dither_hz=33 in the Jetson config.

    _pulse_from_value evaluates that sine at whatever timestamp the caller
    passes, so the effective sample rate is the call rate.
    """

    def setUp(self):
        self.pwm, _ = _make_pwm()
        self.cfg = self.pwm.channel_configs['boom']

    def tearDown(self):
        self.pwm._stop_monitoring()

    def _pulses_at(self, rate_hz, n=256, value=0.5):
        return [self.pwm.compute_pulse('boom', value, now=k / rate_hz)
                for k in range(n)]

    def test_config_still_has_dither_on_boom(self):
        """Guards the premise of the rest of this class."""
        self.assertTrue(self.cfg.dither_enable)
        self.assertAlmostEqual(self.cfg.dither_hz, 33.0)
        self.assertAlmostEqual(self.cfg.dither_amp_us, 25.0)

    def test_at_100hz_dither_reads_as_its_true_frequency(self):
        pulses = self._pulses_at(100.0)
        self.assertAlmostEqual(_dominant_freq_hz(pulses, 100.0), 33.0, delta=1.5)

    def test_at_30hz_dither_aliases_to_a_few_hz(self):
        """33 Hz sampled at 30 Hz folds to |33-30| = 3 Hz — a slow wobble the
        hydraulics will follow, not a stiction-breaking dither."""
        pulses = self._pulses_at(30.0)
        self.assertAlmostEqual(_dominant_freq_hz(pulses, 30.0), 3.0, delta=1.0)

    def test_alias_amplitude_is_a_large_share_of_command_authority(self):
        """Quantifies the damage: the aliased swing against the full stroke."""
        pulses = np.asarray(self._pulses_at(30.0))
        swing = pulses.max() - pulses.min()
        center = float(self.cfg.center)
        authority = float(self.cfg.pulse_max) - (center + float(self.cfg.deadband_us_pos))
        self.assertGreater(swing / authority, 0.15)

    def test_100hz_swing_matches_configured_amplitude(self):
        pulses = np.asarray(self._pulses_at(100.0))
        swing = pulses.max() - pulses.min()
        self.assertLessEqual(swing, 2.0 * self.cfg.dither_amp_us + 1e-6)


# ---------------------------------------------------------------------------
# 3. ramp state clock consistency
# ---------------------------------------------------------------------------

class TestRampStateClock(unittest.TestCase):
    """_init_ramp_state must stamp the same clock _collect_channel_counts uses.

    Seeding from time.time() while ticking with time.monotonic() made the first
    ramped tick after every reset()/reload_config() compute a negative dt, clamp
    it to zero, and hold the channel at center for that tick.
    """

    def test_ramped_channel_moves_on_first_tick_after_reset(self):
        pwm, _ = _make_pwm()
        try:
            cfg = pwm.channel_configs['boom']
            cfg.ramp_enable = True
            cfg.ramp_limit = 5000.0      # µs/s: fast enough to reach target in one tick
            cfg.dither_enable = False
            pwm.reset()

            pwm.values[cfg.output_channel] = 1.0   # full command, away from center
            now = time.monotonic()
            pwm._collect_channel_counts(now + 0.01)
            pulse = pwm._current_pulse_us[cfg.output_channel]
            self.assertNotAlmostEqual(pulse, float(cfg.center), places=3)
        finally:
            pwm._stop_monitoring()

    def test_ramp_state_seeded_with_monotonic(self):
        pwm, _ = _make_pwm()
        try:
            _, stamped, _ = pwm._channel_ramp_state[pwm.channel_configs['boom'].output_channel]
            # Epoch timestamps are ~1.7e9; monotonic (uptime) is far smaller.
            self.assertLess(stamped, 1e8)
        finally:
            pwm._stop_monitoring()


if __name__ == '__main__':
    unittest.main(verbosity=2)
