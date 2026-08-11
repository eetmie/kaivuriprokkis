"""Tests for SetpointSchedule — the slow-producer -> fixed-rate-writer bridge.

Time is injected explicitly (every method takes t_now) so none of this depends
on wall-clock timing or sleeps.
"""

import sys
import threading
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.setpoint_schedule import SetpointSchedule

JOINTS = ['slew', 'boom', 'arm', 'bucket']


def _sched(**kw):
    kw.setdefault('hold_timeout_s', 0.25)
    kw.setdefault('decay_s', 0.25)
    return SetpointSchedule(JOINTS, **kw)


# ---------------------------------------------------------------------------
# empty schedule
# ---------------------------------------------------------------------------

class TestEmptySchedule(unittest.TestCase):
    def test_no_data_returns_zeros(self):
        s = _sched()
        out = s.sample(t_now=10.0)
        self.assertEqual(out.commands, {n: 0.0 for n in JOINTS})

    def test_no_data_is_exhausted(self):
        self.assertTrue(_sched().sample(t_now=10.0).exhausted)

    def test_no_data_age_is_infinite(self):
        self.assertEqual(_sched().sample(t_now=10.0).age_s, float('inf'))

    def test_no_data_decay_is_zero(self):
        self.assertEqual(_sched().sample(t_now=10.0).decay, 0.0)

    def test_has_data_false_when_empty(self):
        self.assertFalse(_sched().has_data())

    def test_clear_drops_data(self):
        s = _sched()
        s.set_point({'boom': 0.5}, t_now=10.0)
        s.clear()
        self.assertFalse(s.has_data())
        self.assertTrue(s.sample(t_now=10.0).exhausted)


# ---------------------------------------------------------------------------
# point mode / zero-order hold
# ---------------------------------------------------------------------------

class TestPointMode(unittest.TestCase):
    def test_point_held_between_updates(self):
        s = _sched()
        s.set_point({'boom': 0.5}, t_now=10.0)
        # Sampled at 100 Hz across a 30 Hz producer gap — value must not sag.
        for i in range(1, 4):
            out = s.sample(t_now=10.0 + i * 0.01)
            self.assertAlmostEqual(out.commands['boom'], 0.5, places=6)

    def test_point_is_fresh_immediately(self):
        s = _sched()
        s.set_point({'boom': 0.5}, t_now=10.0)
        out = s.sample(t_now=10.0)
        self.assertEqual(out.age_s, 0.0)
        self.assertEqual(out.decay, 1.0)

    def test_point_age_grows(self):
        s = _sched()
        s.set_point({'boom': 0.5}, t_now=10.0)
        self.assertAlmostEqual(s.sample(t_now=10.1).age_s, 0.1, places=6)

    def test_point_not_exhausted(self):
        s = _sched()
        s.set_point({'boom': 0.5}, t_now=10.0)
        self.assertFalse(s.sample(t_now=10.0).exhausted)

    def test_values_clipped(self):
        s = _sched()
        s.set_point({'boom': 5.0, 'arm': -5.0}, t_now=10.0)
        out = s.sample(t_now=10.0)
        self.assertAlmostEqual(out.commands['boom'], 1.0)
        self.assertAlmostEqual(out.commands['arm'], -1.0)

    def test_partial_setpoint_zeroes_other_channels(self):
        """Omitted channels must be emitted as explicit zeros, not left out —
        otherwise they depend on the PWM layer's unset_to_zero default."""
        s = _sched()
        s.set_point({'boom': 0.5}, t_now=10.0)
        out = s.sample(t_now=10.0)
        self.assertEqual(set(out.commands), set(JOINTS))
        self.assertAlmostEqual(out.commands['slew'], 0.0)
        self.assertAlmostEqual(out.commands['arm'], 0.0)
        self.assertAlmostEqual(out.commands['bucket'], 0.0)

    def test_extra_channel_names_pass_through(self):
        s = _sched()
        s.set_point({'boom': 0.5, 'trackL': 0.2}, t_now=10.0)
        out = s.sample(t_now=10.0)
        self.assertAlmostEqual(out.commands['trackL'], 0.2, places=5)
        self.assertEqual(set(out.commands), set(JOINTS) | {'trackL'})

    def test_stores_copy_not_reference(self):
        s = _sched()
        cmds = {'boom': 0.5}
        s.set_point(cmds, t_now=10.0)
        cmds['boom'] = 99.0
        self.assertAlmostEqual(s.sample(t_now=10.0).commands['boom'], 0.5)

    def test_new_point_replaces_old(self):
        s = _sched()
        s.set_point({'boom': 0.5}, t_now=10.0)
        s.set_point({'boom': -0.25}, t_now=10.03)
        self.assertAlmostEqual(s.sample(t_now=10.03).commands['boom'], -0.25)

    def test_new_point_resets_age(self):
        s = _sched()
        s.set_point({'boom': 0.5}, t_now=10.0)
        s.set_point({'boom': 0.5}, t_now=10.4)
        self.assertAlmostEqual(s.sample(t_now=10.4).age_s, 0.0, places=6)

    def test_setting_point_clears_chunk(self):
        s = _sched()
        s.set_chunk(np.ones((4, 4), dtype=np.float32), fps=30.0, t_now=10.0)
        s.set_point({'boom': 0.0, 'slew': 0.0, 'arm': 0.0, 'bucket': 0.0}, t_now=10.0)
        self.assertIsNone(s.sample(t_now=10.0).chunk_pos)


# ---------------------------------------------------------------------------
# staleness decay
# ---------------------------------------------------------------------------

class TestStalenessDecay(unittest.TestCase):
    def test_full_authority_within_hold(self):
        s = _sched(hold_timeout_s=0.25, decay_s=0.25)
        s.set_point({'boom': 1.0}, t_now=10.0)
        out = s.sample(t_now=10.24)
        self.assertEqual(out.decay, 1.0)
        self.assertAlmostEqual(out.commands['boom'], 1.0)

    def test_decay_starts_after_hold(self):
        s = _sched(hold_timeout_s=0.25, decay_s=0.25)
        s.set_point({'boom': 1.0}, t_now=10.0)
        out = s.sample(t_now=10.375)  # halfway through the decay window
        self.assertAlmostEqual(out.decay, 0.5, places=5)
        self.assertAlmostEqual(out.commands['boom'], 0.5, places=5)

    def test_fully_decayed_to_zero(self):
        s = _sched(hold_timeout_s=0.25, decay_s=0.25)
        s.set_point({'boom': 1.0}, t_now=10.0)
        out = s.sample(t_now=10.6)
        self.assertEqual(out.decay, 0.0)
        self.assertAlmostEqual(out.commands['boom'], 0.0)

    def test_decay_is_monotonic(self):
        s = _sched(hold_timeout_s=0.1, decay_s=0.4)
        s.set_point({'boom': 1.0}, t_now=10.0)
        prev = 1.1
        for i in range(20):
            d = s.sample(t_now=10.0 + i * 0.03).decay
            self.assertLessEqual(d, prev + 1e-9)
            prev = d

    def test_zero_decay_window_cuts_immediately(self):
        s = _sched(hold_timeout_s=0.1, decay_s=0.0)
        s.set_point({'boom': 1.0}, t_now=10.0)
        self.assertEqual(s.sample(t_now=10.05).decay, 1.0)
        self.assertEqual(s.sample(t_now=10.11).decay, 0.0)

    def test_decay_preserves_sign(self):
        s = _sched(hold_timeout_s=0.25, decay_s=0.25)
        s.set_point({'boom': -0.8}, t_now=10.0)
        out = s.sample(t_now=10.375)
        self.assertLess(out.commands['boom'], 0.0)
        self.assertAlmostEqual(out.commands['boom'], -0.4, places=5)

    def test_fresh_setpoint_restores_authority(self):
        s = _sched(hold_timeout_s=0.25, decay_s=0.25)
        s.set_point({'boom': 1.0}, t_now=10.0)
        self.assertLess(s.sample(t_now=10.4).decay, 1.0)
        s.set_point({'boom': 1.0}, t_now=10.4)
        self.assertEqual(s.sample(t_now=10.4).decay, 1.0)


# ---------------------------------------------------------------------------
# chunk mode
# ---------------------------------------------------------------------------

def _ramp_chunk(n=5):
    """(n, 4) chunk where boom ramps 0.0, 0.1, 0.2, ... and others are zero."""
    c = np.zeros((n, 4), dtype=np.float32)
    c[:, 1] = np.arange(n, dtype=np.float32) * 0.1
    return c


class TestChunkMode(unittest.TestCase):
    def test_first_step_at_t0(self):
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        self.assertAlmostEqual(s.sample(t_now=10.0).commands['boom'], 0.0, places=6)

    def test_lands_on_exact_steps(self):
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        for i in range(5):
            out = s.sample(t_now=10.0 + i * 0.1)
            self.assertAlmostEqual(out.commands['boom'], i * 0.1, places=5,
                                   msg=f'step {i}')

    def test_interpolates_between_steps(self):
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        # Halfway between step 1 (0.1) and step 2 (0.2)
        out = s.sample(t_now=10.15)
        self.assertAlmostEqual(out.commands['boom'], 0.15, places=5)

    def test_interpolation_is_smooth_at_writer_rate(self):
        """100 Hz sampling of a 10 Hz chunk must be monotonic, no staircase."""
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        vals = [s.sample(t_now=10.0 + i * 0.01).commands['boom'] for i in range(40)]
        for a, b in zip(vals, vals[1:]):
            self.assertGreater(b, a - 1e-9)
        # A staircase would repeat each value ~10x; interpolation must not.
        self.assertGreater(len(set(round(v, 6) for v in vals)), 30)

    def test_holds_last_step_past_end(self):
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        out = s.sample(t_now=10.45)  # chunk ends at 10.4
        self.assertAlmostEqual(out.commands['boom'], 0.4, places=5)

    def test_exhausted_past_end(self):
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        self.assertFalse(s.sample(t_now=10.2).exhausted)
        self.assertTrue(s.sample(t_now=10.45).exhausted)

    def test_age_zero_inside_chunk(self):
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        self.assertEqual(s.sample(t_now=10.25).age_s, 0.0)

    def test_age_grows_from_chunk_end(self):
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        # chunk ends at 10.4; at 10.5 the data is 0.1 s old
        self.assertAlmostEqual(s.sample(t_now=10.5).age_s, 0.1, places=6)

    def test_exhausted_chunk_decays(self):
        s = _sched(hold_timeout_s=0.1, decay_s=0.1)
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        self.assertAlmostEqual(s.sample(t_now=10.45).decay, 1.0)   # within hold
        self.assertAlmostEqual(s.sample(t_now=10.55).decay, 0.5, places=5)
        self.assertAlmostEqual(s.sample(t_now=10.7).decay, 0.0)

    def test_before_t0_clamps_to_first_step(self):
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        out = s.sample(t_now=9.9)
        self.assertAlmostEqual(out.commands['boom'], 0.0, places=6)
        self.assertFalse(out.exhausted)

    def test_chunk_pos_tracks_progress(self):
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        self.assertAlmostEqual(s.sample(t_now=10.15).chunk_pos, 1.5, places=5)

    def test_new_chunk_replaces_old(self):
        s = _sched()
        s.set_chunk(_ramp_chunk(), fps=10.0, t_now=10.0)
        other = np.zeros((3, 4), dtype=np.float32)
        other[:, 1] = -0.5
        s.set_chunk(other, fps=10.0, t_now=10.15)
        self.assertAlmostEqual(s.sample(t_now=10.15).commands['boom'], -0.5, places=5)

    def test_single_step_chunk_is_immediately_exhausted(self):
        s = _sched()
        s.set_chunk(np.full((1, 4), 0.3, dtype=np.float32), fps=10.0, t_now=10.0)
        out = s.sample(t_now=10.001)
        self.assertTrue(out.exhausted)
        self.assertAlmostEqual(out.commands['boom'], 0.3, places=5)

    def test_chunk_values_clipped(self):
        s = _sched()
        c = np.full((3, 4), 9.0, dtype=np.float32)
        s.set_chunk(c, fps=10.0, t_now=10.0)
        self.assertAlmostEqual(s.sample(t_now=10.0).commands['boom'], 1.0)

    def test_custom_joint_names(self):
        """A chunk may cover a subset of the schedule's channels; the rest are
        emitted as explicit zeros rather than dropped from the command."""
        s = _sched()
        c = np.zeros((2, 2), dtype=np.float32)
        c[:, 0] = 0.4
        s.set_chunk(c, fps=10.0, t_now=10.0, joint_names=['boom', 'arm'])
        out = s.sample(t_now=10.0)
        self.assertAlmostEqual(out.commands['boom'], 0.4, places=5)
        self.assertEqual(set(out.commands), set(JOINTS))
        self.assertAlmostEqual(out.commands['slew'], 0.0)

    def test_list_input_accepted(self):
        s = _sched()
        s.set_chunk([[0.0, 0.2, 0.0, 0.0], [0.0, 0.4, 0.0, 0.0]],
                    fps=10.0, t_now=10.0)
        self.assertAlmostEqual(s.sample(t_now=10.0).commands['boom'], 0.2, places=5)


class TestChunkValidation(unittest.TestCase):
    def test_rejects_1d(self):
        with self.assertRaises(ValueError):
            _sched().set_chunk(np.zeros(4, dtype=np.float32), fps=10.0, t_now=0.0)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            _sched().set_chunk(np.zeros((0, 4), dtype=np.float32), fps=10.0, t_now=0.0)

    def test_rejects_column_mismatch(self):
        with self.assertRaises(ValueError):
            _sched().set_chunk(np.zeros((3, 2), dtype=np.float32), fps=10.0, t_now=0.0)

    def test_rejects_nonpositive_fps(self):
        with self.assertRaises(ValueError):
            _sched().set_chunk(np.zeros((3, 4), dtype=np.float32), fps=0.0, t_now=0.0)


# ---------------------------------------------------------------------------
# blending
# ---------------------------------------------------------------------------

class TestBlend(unittest.TestCase):
    def test_blend_off_by_default_steps_immediately(self):
        s = _sched()
        s.set_point({'boom': 1.0}, t_now=10.0)
        s.sample(t_now=10.0)
        s.set_point({'boom': -1.0}, t_now=10.01)
        self.assertAlmostEqual(s.sample(t_now=10.01).commands['boom'], -1.0, places=5)

    def test_first_setpoint_fades_in_from_zero(self):
        """A fresh schedule has emitted nothing, so the first blend starts at 0."""
        s = _sched(blend_s=0.1)
        s.set_point({'boom': 1.0}, t_now=10.0)
        self.assertAlmostEqual(s.sample(t_now=10.0).commands['boom'], 0.0, places=5)
        self.assertAlmostEqual(s.sample(t_now=10.05).commands['boom'], 0.5, places=5)

    def test_blend_fades_from_last_emitted(self):
        s = _sched(blend_s=0.1)
        s.set_point({'boom': 1.0}, t_now=10.0)
        s.sample(t_now=10.2)              # first blend done, emitting +1.0
        s.set_point({'boom': -1.0}, t_now=10.2)
        # Halfway through the blend: midpoint of +1 and -1
        self.assertAlmostEqual(s.sample(t_now=10.25).commands['boom'], 0.0, places=5)

    def test_blend_completes(self):
        s = _sched(blend_s=0.1)
        s.set_point({'boom': 1.0}, t_now=10.0)
        s.sample(t_now=10.2)
        s.set_point({'boom': -1.0}, t_now=10.2)
        self.assertAlmostEqual(s.sample(t_now=10.35).commands['boom'], -1.0, places=5)

    def test_blend_softens_chunk_boundary(self):
        s = _sched(blend_s=0.1)
        s.set_chunk(np.full((5, 4), 0.8, dtype=np.float32), fps=10.0, t_now=10.0)
        s.sample(t_now=10.2)
        s.set_chunk(np.full((5, 4), -0.8, dtype=np.float32), fps=10.0, t_now=10.2)
        mid = s.sample(t_now=10.25).commands['boom']
        self.assertGreater(mid, -0.8)
        self.assertLess(mid, 0.8)


# ---------------------------------------------------------------------------
# thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):
    def test_concurrent_producer_and_consumer(self):
        """Producer swapping points/chunks while a consumer samples must not
        raise or ever emit a partially-updated command dict."""
        s = _sched()
        s.set_point({n: 0.0 for n in JOINTS}, t_now=0.0)
        errors = []
        stop = threading.Event()

        def produce():
            try:
                for i in range(2000):
                    if i % 2:
                        s.set_point({n: 0.5 for n in JOINTS})
                    else:
                        s.set_chunk(np.full((10, 4), -0.5, dtype=np.float32), fps=30.0)
            except Exception as e:      # pragma: no cover
                errors.append(e)
            finally:
                stop.set()

        def consume():
            try:
                while not stop.is_set():
                    out = s.sample()
                    self.assertEqual(set(out.commands), set(JOINTS))
                    for v in out.commands.values():
                        self.assertGreaterEqual(v, -1.0)
                        self.assertLessEqual(v, 1.0)
            except Exception as e:      # pragma: no cover
                errors.append(e)

        t1 = threading.Thread(target=produce)
        t2 = threading.Thread(target=consume)
        t1.start(); t2.start()
        t1.join(timeout=10.0); t2.join(timeout=10.0)
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
