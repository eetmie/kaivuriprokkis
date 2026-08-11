"""Tests for ExcavatorController's direct-command mode.

Uses the same thin-proxy trick as test_vel_ctrl_stack.py: the mode's methods are
bound onto a stand-in object carrying only the attributes they touch, so none of
this needs hardware, numba, robot config or a control thread.

The invariant under test is the one the mode exists to enforce: the control
thread writes the valves on EVERY tick, including when the command is zero.
Skipping the write is what starves the PWM watchdog and the input-rate gate.
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.excavator_controller import ExcavatorController, _DIRECT_CMD_JOINTS
from modules.setpoint_schedule import SetpointSchedule


class _DirectCmdProxy:
    """Proxy for direct-command mode methods only."""

    enter_direct_command_mode = ExcavatorController.enter_direct_command_mode
    exit_direct_command_mode = ExcavatorController.exit_direct_command_mode
    give_direct_commands = ExcavatorController.give_direct_commands
    give_direct_chunk = ExcavatorController.give_direct_chunk
    get_direct_status = ExcavatorController.get_direct_status
    _send_direct_commands = ExcavatorController._send_direct_commands

    def __init__(self):
        self._lock = threading.RLock()
        self._vel_cmd_lock = threading.Lock()
        self._output_suspended = False
        self._vel_cmd_mode = False
        self._vel_cmd_commands = {}
        self._vel_cmd_integrals = {'boom': 0.0, 'arm': 0.0, 'bucket': 0.0}
        self._direct_cmd_mode = False
        self._direct_schedule = SetpointSchedule(_DIRECT_CMD_JOINTS)
        self._direct_last_sample = None
        self._direct_decayed_warned = False
        self._outputs_zeroed = False
        self.logger = MagicMock()
        self.hardware = MagicMock()

    def sent(self):
        """The last command dict handed to hardware."""
        return self.hardware.send_named_pwm_commands.call_args[0][0]


# ---------------------------------------------------------------------------
# enter / exit
# ---------------------------------------------------------------------------

class TestEnterExit(unittest.TestCase):
    def setUp(self):
        self.ctrl = _DirectCmdProxy()

    def test_enter_sets_mode(self):
        self.ctrl.enter_direct_command_mode()
        self.assertTrue(self.ctrl._direct_cmd_mode)

    def test_enter_clears_output_suspended(self):
        self.ctrl._output_suspended = True
        self.ctrl.enter_direct_command_mode()
        self.assertFalse(self.ctrl._output_suspended)

    def test_enter_clears_velocity_mode(self):
        """The three output modes are mutually exclusive — two writers on the
        PWM bus would interleave commands."""
        self.ctrl._vel_cmd_mode = True
        self.ctrl.enter_direct_command_mode()
        self.assertFalse(self.ctrl._vel_cmd_mode)

    def test_enter_starts_with_no_data(self):
        self.ctrl.enter_direct_command_mode()
        self.assertFalse(self.ctrl._direct_schedule.has_data())

    def test_enter_discards_previous_setpoint(self):
        self.ctrl.enter_direct_command_mode()
        self.ctrl.give_direct_commands({'boom': 0.9})
        self.ctrl.enter_direct_command_mode()
        self.assertFalse(self.ctrl._direct_schedule.has_data())

    def test_enter_applies_timing_params(self):
        self.ctrl.enter_direct_command_mode(hold_timeout_s=0.4, decay_s=0.6,
                                            blend_s=0.05)
        s = self.ctrl._direct_schedule
        self.assertAlmostEqual(s.hold_timeout_s, 0.4)
        self.assertAlmostEqual(s.decay_s, 0.6)
        self.assertAlmostEqual(s.blend_s, 0.05)

    def test_enter_accepts_custom_joint_names(self):
        self.ctrl.enter_direct_command_mode(joint_names=['boom', 'arm'])
        self.assertEqual(self.ctrl._direct_schedule.joint_names, ['boom', 'arm'])

    def test_default_joint_names(self):
        self.ctrl.enter_direct_command_mode()
        self.assertEqual(self.ctrl._direct_schedule.joint_names,
                         list(_DIRECT_CMD_JOINTS))

    def test_exit_clears_mode(self):
        self.ctrl.enter_direct_command_mode()
        self.ctrl.exit_direct_command_mode()
        self.assertFalse(self.ctrl._direct_cmd_mode)

    def test_exit_resets_hardware(self):
        self.ctrl.enter_direct_command_mode()
        self.ctrl.exit_direct_command_mode()
        self.ctrl.hardware.reset.assert_called_once()

    def test_exit_clears_schedule(self):
        self.ctrl.enter_direct_command_mode()
        self.ctrl.give_direct_commands({'boom': 0.5})
        self.ctrl.exit_direct_command_mode()
        self.assertFalse(self.ctrl._direct_schedule.has_data())


# ---------------------------------------------------------------------------
# producer API
# ---------------------------------------------------------------------------

class TestGiveCommands(unittest.TestCase):
    def setUp(self):
        self.ctrl = _DirectCmdProxy()
        self.ctrl.enter_direct_command_mode()

    def test_give_commands_does_no_io(self):
        """The producer thread must not touch hardware — that is the point."""
        self.ctrl.give_direct_commands({'boom': 0.5})
        self.ctrl.hardware.send_named_pwm_commands.assert_not_called()

    def test_give_chunk_does_no_io(self):
        self.ctrl.give_direct_chunk(np.zeros((5, 4), dtype=np.float32), fps=30.0)
        self.ctrl.hardware.send_named_pwm_commands.assert_not_called()

    def test_command_reaches_hardware_on_tick(self):
        self.ctrl.give_direct_commands({'boom': 0.5})
        self.ctrl._send_direct_commands()
        self.assertAlmostEqual(self.ctrl.sent()['boom'], 0.5, places=5)

    def test_chunk_reaches_hardware_on_tick(self):
        chunk = np.zeros((5, 4), dtype=np.float32)
        chunk[:, 1] = 0.7
        self.ctrl.give_direct_chunk(chunk, fps=30.0)
        self.ctrl._send_direct_commands()
        self.assertAlmostEqual(self.ctrl.sent()['boom'], 0.7, places=5)

    def test_chunk_column_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.ctrl.give_direct_chunk(np.zeros((5, 2), dtype=np.float32), fps=30.0)

    def test_give_commands_clears_outputs_zeroed(self):
        self.ctrl._outputs_zeroed = True
        self.ctrl.give_direct_commands({'boom': 0.1})
        self.assertFalse(self.ctrl._outputs_zeroed)


# ---------------------------------------------------------------------------
# the core invariant: write every tick
# ---------------------------------------------------------------------------

class TestWritesEveryTick(unittest.TestCase):
    def setUp(self):
        self.ctrl = _DirectCmdProxy()
        self.ctrl.enter_direct_command_mode()

    def test_writes_without_any_setpoint(self):
        """No producer has spoken yet — still must write, to feed the watchdog."""
        self.ctrl._send_direct_commands()
        self.ctrl.hardware.send_named_pwm_commands.assert_called_once()

    def test_writes_zeros_when_no_setpoint(self):
        self.ctrl._send_direct_commands()
        for name, v in self.ctrl.sent().items():
            self.assertAlmostEqual(v, 0.0, msg=f'{name} not zero')

    def test_writes_repeatedly_for_one_setpoint(self):
        """One 30 Hz setpoint must produce many 100 Hz writes."""
        self.ctrl.give_direct_commands({'boom': 0.5})
        for _ in range(10):
            self.ctrl._send_direct_commands()
        self.assertEqual(self.ctrl.hardware.send_named_pwm_commands.call_count, 10)

    def test_never_calls_hardware_reset_on_zero(self):
        """Unlike velocity mode, a zero command is written, not short-circuited
        into a one-shot hardware.reset() that then stops writing."""
        self.ctrl.give_direct_commands({n: 0.0 for n in _DIRECT_CMD_JOINTS})
        for _ in range(5):
            self.ctrl._send_direct_commands()
        self.ctrl.hardware.reset.assert_not_called()
        self.assertEqual(self.ctrl.hardware.send_named_pwm_commands.call_count, 5)

    def test_keeps_writing_after_setpoint_fully_decays(self):
        self.ctrl.enter_direct_command_mode(hold_timeout_s=0.0, decay_s=0.0)
        self.ctrl.give_direct_commands({'boom': 1.0})
        self.ctrl._direct_schedule.set_point({'boom': 1.0}, t_now=0.0)
        for _ in range(5):
            self.ctrl._send_direct_commands()
        self.assertEqual(self.ctrl.hardware.send_named_pwm_commands.call_count, 5)
        self.assertAlmostEqual(self.ctrl.sent()['boom'], 0.0)

    def test_all_channels_present_in_every_write(self):
        self.ctrl.give_direct_commands({'boom': 0.5})
        self.ctrl._send_direct_commands()
        self.assertEqual(set(self.ctrl.sent()), set(_DIRECT_CMD_JOINTS))


# ---------------------------------------------------------------------------
# staleness behaviour at the controller level
# ---------------------------------------------------------------------------

class TestStaleness(unittest.TestCase):
    def setUp(self):
        self.ctrl = _DirectCmdProxy()
        self.ctrl.enter_direct_command_mode(hold_timeout_s=0.1, decay_s=0.1)

    def _tick_at(self, t):
        """Drive one control tick with the schedule evaluated at time t."""
        sample = self.ctrl._direct_schedule.sample(t_now=t)
        self.ctrl.hardware.send_named_pwm_commands(sample.commands)
        self.ctrl._direct_last_sample = sample
        return sample

    def test_held_at_full_authority_across_producer_gap(self):
        self.ctrl._direct_schedule.set_point({'boom': 0.8}, t_now=10.0)
        s = self._tick_at(10.05)
        self.assertAlmostEqual(s.commands['boom'], 0.8, places=5)

    def test_decays_when_producer_dies(self):
        self.ctrl._direct_schedule.set_point({'boom': 0.8}, t_now=10.0)
        self.assertAlmostEqual(self._tick_at(10.15).commands['boom'], 0.4, places=5)
        self.assertAlmostEqual(self._tick_at(10.25).commands['boom'], 0.0, places=5)

    def test_status_reports_decay(self):
        self.ctrl._direct_schedule.set_point({'boom': 0.8}, t_now=10.0)
        self._tick_at(10.15)
        status = self.ctrl.get_direct_status()
        self.assertTrue(status['active'])
        self.assertAlmostEqual(status['decay'], 0.5, places=5)
        self.assertAlmostEqual(status['age_s'], 0.15, places=5)

    def test_status_before_any_tick(self):
        status = self.ctrl.get_direct_status()
        self.assertEqual(status['age_s'], float('inf'))
        self.assertEqual(status['decay'], 0.0)
        self.assertTrue(status['exhausted'])

    def test_warns_once_when_decaying(self):
        self.ctrl.give_direct_commands({'boom': 0.8})
        self.ctrl._direct_schedule.set_point({'boom': 0.8}, t_now=0.0)
        for _ in range(5):
            self.ctrl._send_direct_commands()
        self.assertEqual(self.ctrl.logger.warning.call_count, 1)

    def test_warn_rearms_after_recovery(self):
        self.ctrl._direct_schedule.set_point({'boom': 0.8}, t_now=0.0)
        self.ctrl._send_direct_commands()          # stale -> warns
        self.ctrl.give_direct_commands({'boom': 0.8})   # fresh again
        self.ctrl._send_direct_commands()
        self.assertFalse(self.ctrl._direct_decayed_warned)
        self.ctrl._direct_schedule.set_point({'boom': 0.8}, t_now=0.0)
        self.ctrl._send_direct_commands()
        self.assertEqual(self.ctrl.logger.warning.call_count, 2)


# ---------------------------------------------------------------------------
# chunk playback through the controller
# ---------------------------------------------------------------------------

class TestChunkPlayback(unittest.TestCase):
    def setUp(self):
        self.ctrl = _DirectCmdProxy()
        self.ctrl.enter_direct_command_mode()

    def test_chunk_interpolates_across_ticks(self):
        """A 10 Hz chunk sampled at 100 Hz must move smoothly, not in steps."""
        chunk = np.zeros((5, 4), dtype=np.float32)
        chunk[:, 1] = np.arange(5, dtype=np.float32) * 0.2
        self.ctrl.give_direct_chunk(chunk, fps=10.0)
        sched = self.ctrl._direct_schedule
        t0 = sched._chunk.t0
        vals = [sched.sample(t_now=t0 + i * 0.01).commands['boom'] for i in range(40)]
        self.assertGreater(len(set(round(v, 6) for v in vals)), 30)
        self.assertAlmostEqual(vals[0], 0.0, places=5)
        self.assertAlmostEqual(vals[-1], 0.78, places=2)

    def test_chunk_position_in_status(self):
        self.ctrl.give_direct_chunk(np.zeros((10, 4), dtype=np.float32), fps=10.0)
        self.ctrl._send_direct_commands()
        self.assertIsNotNone(self.ctrl.get_direct_status()['chunk_pos'])

    def test_new_chunk_supersedes_running_one(self):
        first = np.full((10, 4), 0.5, dtype=np.float32)
        second = np.full((10, 4), -0.5, dtype=np.float32)
        self.ctrl.give_direct_chunk(first, fps=10.0)
        self.ctrl._send_direct_commands()
        self.ctrl.give_direct_chunk(second, fps=10.0)
        self.ctrl._send_direct_commands()
        self.assertAlmostEqual(self.ctrl.sent()['boom'], -0.5, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
