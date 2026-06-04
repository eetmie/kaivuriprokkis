"""Tests for velocity-control additions to ExcavatorController and JointVelocityController.

ExcavatorController methods are tested via a thin proxy (avoiding full hardware/numba init).
JointVelocityController is imported from simple_drive (the older velocity controller there,
not the root test_vel_ctrl.py hardware prototype) with hardware deps pre-mocked.
"""

import math
import sys
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.excavator_controller import ExcavatorController, _VEL_CMD_JOINTS


# ---------------------------------------------------------------------------
# Thin proxy — lets us call the new ExcavatorController methods without
# spinning up hardware, numba JIT, robot config, IK, or a control thread.
# Only the attributes each method actually touches are present.
# ---------------------------------------------------------------------------

class _ControllerProxy:
    _gyro_velocity_mode: str = 'fd_only'
    _last_joint_vel_radps = None
    _last_joint_vel_time = None
    _last_gyro_device_ts_us = 123
    _last_gyro_wall_t = 456.0
    _gyro_fallback_counter = 7
    _gyro_bias_radps = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    logger = MagicMock()

    set_velocity_mode             = ExcavatorController.set_velocity_mode
    get_joint_velocities_with_age = ExcavatorController.get_joint_velocities_with_age


class _VelCmdProxy:
    """Proxy for velocity-command mode methods only."""
    _output_suspended: bool = False
    _vel_cmd_mode: bool = False
    _vel_cmd_commands: dict = {}
    _vel_cmd_integrals: dict = {j: 0.0 for j in _VEL_CMD_JOINTS}
    _outputs_zeroed: bool = True
    _lock = threading.Lock()
    _vel_cmd_lock = threading.Lock()
    # PI config — default gains matching simple_drive defaults
    _vel_kp: float = 0.04
    _vel_ki: float = 0.008
    _vel_ki_max: float = 0.4
    _vel_deadband_radps: float = math.radians(2.0)
    _vel_stale_timeout_s: float = 0.15
    _last_joint_vel_radps = None
    _last_joint_vel_time = None
    logger = MagicMock()
    hardware = MagicMock()

    enter_velocity_command_mode = ExcavatorController.enter_velocity_command_mode
    exit_velocity_command_mode  = ExcavatorController.exit_velocity_command_mode
    give_velocity_commands      = ExcavatorController.give_velocity_commands
    _send_velocity_commands     = ExcavatorController._send_velocity_commands

    def _fresh_vel(self, *radps_values):
        """Inject a fresh velocity reading."""
        self._last_joint_vel_radps = np.array(radps_values, dtype=np.float32)
        self._last_joint_vel_time = time.perf_counter()


# ---------------------------------------------------------------------------
# set_velocity_mode()
# ---------------------------------------------------------------------------

class TestSetVelocityMode(unittest.TestCase):
    def setUp(self):
        self.ctrl = _ControllerProxy()

    def test_fd_only(self):
        self.ctrl.set_velocity_mode('fd_only')
        self.assertEqual(self.ctrl._gyro_velocity_mode, 'fd_only')

    def test_gyro_only(self):
        self.ctrl.set_velocity_mode('gyro_only')
        self.assertEqual(self.ctrl._gyro_velocity_mode, 'gyro_only')

    def test_fused(self):
        self.ctrl.set_velocity_mode('fused')
        self.assertEqual(self.ctrl._gyro_velocity_mode, 'fused')

    def test_strips_whitespace_and_lowercases(self):
        self.ctrl.set_velocity_mode('  GYRO_ONLY  ')
        self.assertEqual(self.ctrl._gyro_velocity_mode, 'gyro_only')

    def test_invalid_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.ctrl.set_velocity_mode('bad_mode')

    def test_error_message_lists_all_valid_options(self):
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.set_velocity_mode('banana')
        msg = str(ctx.exception)
        self.assertIn('fd_only', msg)
        self.assertIn('gyro_only', msg)
        self.assertIn('fused', msg)

    def test_mode_unchanged_after_invalid(self):
        self.ctrl._gyro_velocity_mode = 'fused'
        try:
            self.ctrl.set_velocity_mode('bad')
        except ValueError:
            pass
        self.assertEqual(self.ctrl._gyro_velocity_mode, 'fused')

    def test_switching_modes_resets_gyro_state(self):
        self.ctrl._gyro_velocity_mode = 'fd_only'
        self.ctrl._last_gyro_device_ts_us = 123
        self.ctrl._last_gyro_wall_t = 456.0
        self.ctrl._gyro_fallback_counter = 7
        self.ctrl._gyro_bias_radps = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        self.ctrl.set_velocity_mode('gyro_only')

        self.assertIsNone(self.ctrl._last_gyro_device_ts_us)
        self.assertIsNone(self.ctrl._last_gyro_wall_t)
        self.assertEqual(self.ctrl._gyro_fallback_counter, 0)
        self.assertTrue(np.array_equal(self.ctrl._gyro_bias_radps, np.zeros(3, dtype=np.float32)))

    def test_same_mode_is_noop(self):
        bias = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.ctrl._gyro_velocity_mode = 'fd_only'
        self.ctrl._last_gyro_device_ts_us = 123
        self.ctrl._last_gyro_wall_t = 456.0
        self.ctrl._gyro_fallback_counter = 7
        self.ctrl._gyro_bias_radps = bias.copy()

        self.ctrl.set_velocity_mode('fd_only')

        self.assertEqual(self.ctrl._last_gyro_device_ts_us, 123)
        self.assertEqual(self.ctrl._last_gyro_wall_t, 456.0)
        self.assertEqual(self.ctrl._gyro_fallback_counter, 7)
        self.assertTrue(np.array_equal(self.ctrl._gyro_bias_radps, bias))


# ---------------------------------------------------------------------------
# get_joint_velocities_with_age()
# ---------------------------------------------------------------------------

class TestGetJointVelocitiesWithAge(unittest.TestCase):
    def setUp(self):
        self.ctrl = _ControllerProxy()

    def test_never_computed_returns_none_and_inf(self):
        vels, age = self.ctrl.get_joint_velocities_with_age()
        self.assertIsNone(vels)
        self.assertEqual(age, float('inf'))

    def test_radps_none_with_time_set_returns_inf(self):
        self.ctrl._last_joint_vel_time = time.perf_counter()
        vels, age = self.ctrl.get_joint_velocities_with_age()
        self.assertIsNone(vels)
        self.assertEqual(age, float('inf'))

    def test_time_none_with_radps_set_returns_inf(self):
        self.ctrl._last_joint_vel_radps = np.zeros(4, dtype=np.float32)
        vels, age = self.ctrl.get_joint_velocities_with_age()
        self.assertIsNone(vels)
        self.assertEqual(age, float('inf'))

    def test_converts_radps_to_degps(self):
        radps = np.array([0.0, 1.0, -0.5, 0.25], dtype=np.float32)
        self.ctrl._last_joint_vel_radps = radps
        self.ctrl._last_joint_vel_time = time.perf_counter()
        vels, _ = self.ctrl.get_joint_velocities_with_age()
        self.assertIsNotNone(vels)
        self.assertAlmostEqual(vels[1], math.degrees(1.0), places=3)
        self.assertAlmostEqual(vels[2], math.degrees(-0.5), places=3)
        self.assertAlmostEqual(vels[3], math.degrees(0.25), places=3)

    def test_age_is_small_when_just_stamped(self):
        self.ctrl._last_joint_vel_radps = np.zeros(4, dtype=np.float32)
        self.ctrl._last_joint_vel_time = time.perf_counter()
        _, age = self.ctrl.get_joint_velocities_with_age()
        self.assertLess(age, 0.05)

    def test_age_grows_over_time(self):
        self.ctrl._last_joint_vel_radps = np.zeros(4, dtype=np.float32)
        self.ctrl._last_joint_vel_time = time.perf_counter()
        time.sleep(0.025)
        _, age = self.ctrl.get_joint_velocities_with_age()
        self.assertGreater(age, 0.015)

    def test_returns_list_not_array(self):
        self.ctrl._last_joint_vel_radps = np.zeros(4, dtype=np.float32)
        self.ctrl._last_joint_vel_time = time.perf_counter()
        vels, _ = self.ctrl.get_joint_velocities_with_age()
        self.assertIsInstance(vels, list)


# ---------------------------------------------------------------------------
# JointVelocityController (from simple_drive)
# Pre-mock hardware imports so simple_drive can be loaded on non-RPi hosts.
# ---------------------------------------------------------------------------

_MOCK = MagicMock()
for _mod in ('modules.udp_socket', 'modules.hardware_interface',
             'tools.linkage_rate_compensation'):
    sys.modules.setdefault(_mod, _MOCK)

_jvc_import_err = None
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location('simple_drive', ROOT_DIR / 'simple_drive.py')
    _sd = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_sd)
    JointVelocityController = _sd.JointVelocityController
except Exception as _e:
    _jvc_import_err = _e


@unittest.skipIf(_jvc_import_err, f'simple_drive import failed: {_jvc_import_err}')
class TestJointVelocityController(unittest.TestCase):

    def _make(self, kp=0.1, ki=0.01, ki_max=0.5, deadband=2.0, max_degps=20.0):
        return JointVelocityController(
            kp=kp, ki=ki, ki_max=ki_max,
            deadband_degps=deadband, max_degps=max_degps,
        )

    @staticmethod
    def _vels(slew=0.0, boom=0.0, arm=0.0, bucket=0.0):
        return [slew, boom, arm, bucket]

    # -- deadband / centre reset --

    def test_zero_joystick_zeroes_output_and_resets_integral(self):
        ctrl = self._make()
        ctrl._integral['lift_boom'] = 0.4
        ctrl._integral['scoop'] = -0.3
        out = ctrl.apply({'lift_boom': 0.0, 'tilt_boom': 0.0, 'scoop': 0.0},
                         self._vels(), dt=0.01)
        self.assertAlmostEqual(out['lift_boom'], 0.0)
        self.assertAlmostEqual(out['scoop'], 0.0)
        self.assertAlmostEqual(ctrl._integral['lift_boom'], 0.0)
        self.assertAlmostEqual(ctrl._integral['scoop'], 0.0)

    def test_small_joystick_within_deadband_zeroes_output(self):
        # deadband=2.0 deg/s, max_degps=20 → joystick 0.09 → desired 1.8 deg/s < deadband
        ctrl = self._make(deadband=2.0, max_degps=20.0)
        out = ctrl.apply({'lift_boom': 0.09, 'tilt_boom': 0.0, 'scoop': 0.0},
                         self._vels(boom=0.0), dt=0.01)
        self.assertAlmostEqual(out['lift_boom'], 0.0)

    # -- proportional response --

    def test_positive_joystick_at_rest_gives_positive_command(self):
        ctrl = self._make(kp=0.1, ki=0.0)
        out = ctrl.apply({'lift_boom': 1.0, 'tilt_boom': 0.0, 'scoop': 0.0},
                         self._vels(boom=0.0), dt=0.01)
        self.assertGreater(out['lift_boom'], 0.0)

    def test_negative_joystick_at_rest_gives_negative_command(self):
        ctrl = self._make(kp=0.1, ki=0.0)
        out = ctrl.apply({'lift_boom': -1.0, 'tilt_boom': 0.0, 'scoop': 0.0},
                         self._vels(boom=0.0), dt=0.01)
        self.assertLess(out['lift_boom'], 0.0)

    def test_zero_error_gives_zero_command_with_ki_zero(self):
        ctrl = self._make(kp=0.1, ki=0.0, max_degps=20.0)
        # half joystick → desired 10 deg/s, actual = 10 deg/s → error = 0
        out = ctrl.apply({'lift_boom': 0.5, 'tilt_boom': 0.0, 'scoop': 0.0},
                         self._vels(boom=10.0), dt=0.01)
        self.assertAlmostEqual(out['lift_boom'], 0.0, places=5)

    def test_larger_error_gives_larger_command(self):
        ctrl = self._make(kp=0.05, ki=0.0, max_degps=20.0)
        cmds = {'lift_boom': 1.0, 'tilt_boom': 0.0, 'scoop': 0.0}
        out_slow = ctrl.apply(cmds, self._vels(boom=15.0), dt=0.01)  # err=5
        ctrl.reset()
        out_fast = ctrl.apply(cmds, self._vels(boom=5.0), dt=0.01)   # err=15
        self.assertGreater(out_fast['lift_boom'], out_slow['lift_boom'])

    # -- integral --

    def test_integral_accumulates_over_ticks(self):
        ctrl = self._make(kp=0.0, ki=0.1, ki_max=1.0, max_degps=20.0)
        cmds = {'lift_boom': 1.0, 'tilt_boom': 0.0, 'scoop': 0.0}
        ctrl.apply(cmds, self._vels(boom=0.0), dt=0.01)
        i_after_1 = ctrl._integral['lift_boom']
        ctrl.apply(cmds, self._vels(boom=0.0), dt=0.01)
        i_after_2 = ctrl._integral['lift_boom']
        self.assertGreater(i_after_2, i_after_1)

    def test_integral_anti_windup(self):
        ctrl = self._make(kp=0.0, ki=1.0, ki_max=0.4, max_degps=20.0)
        cmds = {'lift_boom': 1.0, 'tilt_boom': 0.0, 'scoop': 0.0}
        for _ in range(500):
            ctrl.apply(cmds, self._vels(boom=0.0), dt=0.01)
        self.assertLessEqual(ctrl._integral['lift_boom'], 0.4 + 1e-6)

    # -- output clamping --

    def test_output_clamped_to_plus_one(self):
        ctrl = self._make(kp=999.0, ki=0.0)
        out = ctrl.apply({'lift_boom': 1.0, 'tilt_boom': 0.0, 'scoop': 0.0},
                         self._vels(boom=0.0), dt=0.01)
        self.assertLessEqual(out['lift_boom'], 1.0)

    def test_output_clamped_to_minus_one(self):
        ctrl = self._make(kp=999.0, ki=0.0)
        out = ctrl.apply({'lift_boom': -1.0, 'tilt_boom': 0.0, 'scoop': 0.0},
                         self._vels(boom=0.0), dt=0.01)
        self.assertGreaterEqual(out['lift_boom'], -1.0)

    # -- passthrough of uncontrolled keys --

    def test_rotate_passes_through_unchanged(self):
        ctrl = self._make()
        out = ctrl.apply({'rotate': 0.7, 'lift_boom': 0.0, 'scoop': 0.0},
                         self._vels(), dt=0.01)
        self.assertAlmostEqual(out['rotate'], 0.7)

    def test_track_keys_pass_through_unchanged(self):
        ctrl = self._make()
        out = ctrl.apply({'trackR': 0.5, 'trackL': -0.3, 'lift_boom': 0.0, 'scoop': 0.0},
                         self._vels(), dt=0.01)
        self.assertAlmostEqual(out['trackR'], 0.5)
        self.assertAlmostEqual(out['trackL'], -0.3)

    # -- reset --

    def test_reset_zeroes_all_integrals(self):
        ctrl = self._make()
        for k in ctrl._integral:
            ctrl._integral[k] = 0.9
        ctrl.reset()
        for k, v in ctrl._integral.items():
            self.assertAlmostEqual(v, 0.0, msg=f'integral[{k!r}] not zeroed')


# ---------------------------------------------------------------------------
# enter/exit velocity command mode
# ---------------------------------------------------------------------------

class TestEnterExitVelocityCommandMode(unittest.TestCase):
    def setUp(self):
        self.ctrl = _VelCmdProxy()

    def test_enter_sets_vel_cmd_mode(self):
        self.ctrl.enter_velocity_command_mode()
        self.assertTrue(self.ctrl._vel_cmd_mode)

    def test_enter_clears_output_suspended(self):
        self.ctrl._output_suspended = True
        self.ctrl.enter_velocity_command_mode()
        self.assertFalse(self.ctrl._output_suspended)

    def test_enter_resets_integrals(self):
        for k in self.ctrl._vel_cmd_integrals:
            self.ctrl._vel_cmd_integrals[k] = 9.9
        self.ctrl.enter_velocity_command_mode()
        for k, v in self.ctrl._vel_cmd_integrals.items():
            self.assertAlmostEqual(v, 0.0, msg=f'integral[{k!r}] not reset')

    def test_enter_clears_commands(self):
        self.ctrl._vel_cmd_commands = {'lift_boom': 0.5}
        self.ctrl.enter_velocity_command_mode()
        self.assertEqual(self.ctrl._vel_cmd_commands, {})

    def test_exit_clears_vel_cmd_mode(self):
        self.ctrl._vel_cmd_mode = True
        self.ctrl.exit_velocity_command_mode()
        self.assertFalse(self.ctrl._vel_cmd_mode)

    def test_exit_resets_integrals(self):
        self.ctrl._vel_cmd_mode = True
        for k in self.ctrl._vel_cmd_integrals:
            self.ctrl._vel_cmd_integrals[k] = 3.0
        self.ctrl.exit_velocity_command_mode()
        for k, v in self.ctrl._vel_cmd_integrals.items():
            self.assertAlmostEqual(v, 0.0, msg=f'integral[{k!r}] not reset on exit')


# ---------------------------------------------------------------------------
# give_velocity_commands
# ---------------------------------------------------------------------------

class TestGiveVelocityCommands(unittest.TestCase):
    def setUp(self):
        self.ctrl = _VelCmdProxy()

    def test_stores_commands(self):
        cmds = {'lift_boom': 0.3, 'rotate': 0.5}
        self.ctrl.give_velocity_commands(cmds)
        with self.ctrl._vel_cmd_lock:
            stored = dict(self.ctrl._vel_cmd_commands)
        self.assertEqual(stored, cmds)

    def test_clears_outputs_zeroed(self):
        self.ctrl._outputs_zeroed = True
        self.ctrl.give_velocity_commands({'lift_boom': 0.1})
        self.assertFalse(self.ctrl._outputs_zeroed)

    def test_stores_copy_not_reference(self):
        cmds = {'lift_boom': 0.1}
        self.ctrl.give_velocity_commands(cmds)
        cmds['lift_boom'] = 99.0
        with self.ctrl._vel_cmd_lock:
            self.assertNotEqual(self.ctrl._vel_cmd_commands['lift_boom'], 99.0)


# ---------------------------------------------------------------------------
# _send_velocity_commands — PI behaviour
# ---------------------------------------------------------------------------

class TestSendVelocityCommands(unittest.TestCase):
    def setUp(self):
        self.ctrl = _VelCmdProxy()
        self.ctrl.enter_velocity_command_mode()

    def _set_cmds(self, **kw):
        self.ctrl.give_velocity_commands(kw)

    def test_empty_commands_calls_hardware_reset(self):
        self.ctrl._outputs_zeroed = False
        self.ctrl._vel_cmd_commands = {}
        self.ctrl._send_velocity_commands(0.01)
        self.ctrl.hardware.reset.assert_called_once()

    def test_empty_commands_sets_outputs_zeroed(self):
        self.ctrl._outputs_zeroed = False
        self.ctrl._vel_cmd_commands = {}
        self.ctrl._send_velocity_commands(0.01)
        self.assertTrue(self.ctrl._outputs_zeroed)

    def test_pass_through_joint_clipped_positive(self):
        self._set_cmds(rotate=0.7)
        self.ctrl._send_velocity_commands(0.01)
        sent = self.ctrl.hardware.send_named_pwm_commands.call_args[0][0]
        self.assertAlmostEqual(sent['rotate'], 0.7, places=5)

    def test_pass_through_joint_clipped_to_one(self):
        self._set_cmds(rotate=5.0)
        self.ctrl._send_velocity_commands(0.01)
        sent = self.ctrl.hardware.send_named_pwm_commands.call_args[0][0]
        self.assertLessEqual(sent['rotate'], 1.0)

    def test_pass_through_negative_clipped(self):
        self._set_cmds(trackL=-2.5)
        self.ctrl._send_velocity_commands(0.01)
        sent = self.ctrl.hardware.send_named_pwm_commands.call_args[0][0]
        self.assertGreaterEqual(sent['trackL'], -1.0)

    def test_stale_velocity_zeros_controlled_joint(self):
        # No velocity reading → stale → output must be 0
        self._set_cmds(lift_boom=0.5)
        self.ctrl._send_velocity_commands(0.01)
        sent = self.ctrl.hardware.send_named_pwm_commands.call_args[0][0]
        self.assertAlmostEqual(sent['lift_boom'], 0.0)

    def test_stale_velocity_resets_integral(self):
        self.ctrl._vel_cmd_integrals['lift_boom'] = 0.3
        self._set_cmds(lift_boom=0.5)
        self.ctrl._send_velocity_commands(0.01)
        self.assertAlmostEqual(self.ctrl._vel_cmd_integrals['lift_boom'], 0.0)

    def test_desired_below_deadband_zeroes_output(self):
        self.ctrl._fresh_vel(0.0, 0.0, 0.0, 0.0)
        # deadband = 2 deg/s ≈ 0.035 rad/s; send 0.01 rad/s
        self._set_cmds(lift_boom=0.01)
        self.ctrl._send_velocity_commands(0.01)
        sent = self.ctrl.hardware.send_named_pwm_commands.call_args[0][0]
        self.assertAlmostEqual(sent['lift_boom'], 0.0)

    def test_proportional_response_at_rest(self):
        self.ctrl._fresh_vel(0.0, 0.0, 0.0, 0.0)
        self._set_cmds(lift_boom=0.3)   # 0.3 rad/s desired, 0.0 actual → error = 0.3
        self.ctrl._vel_ki = 0.0
        self.ctrl._send_velocity_commands(0.01)
        sent = self.ctrl.hardware.send_named_pwm_commands.call_args[0][0]
        expected = np.clip(self.ctrl._vel_kp * 0.3, -1.0, 1.0)
        self.assertAlmostEqual(sent['lift_boom'], expected, places=5)

    def test_zero_error_gives_zero_output(self):
        # actual == desired → error = 0, no integral (ki=0)
        self.ctrl._fresh_vel(0.0, 0.3, 0.0, 0.0)  # lift idx=1 at 0.3 rad/s
        self._set_cmds(lift_boom=0.3)
        self.ctrl._vel_ki = 0.0
        self.ctrl._send_velocity_commands(0.01)
        sent = self.ctrl.hardware.send_named_pwm_commands.call_args[0][0]
        self.assertAlmostEqual(sent['lift_boom'], 0.0, places=5)

    def test_integral_accumulates(self):
        self.ctrl._fresh_vel(0.0, 0.0, 0.0, 0.0)
        self._set_cmds(tilt_boom=0.5)
        self.ctrl._vel_kp = 0.0
        self.ctrl._send_velocity_commands(0.01)
        i_after_1 = self.ctrl._vel_cmd_integrals['tilt_boom']
        self.ctrl._fresh_vel(0.0, 0.0, 0.0, 0.0)
        self.ctrl._send_velocity_commands(0.01)
        i_after_2 = self.ctrl._vel_cmd_integrals['tilt_boom']
        self.assertGreater(i_after_2, i_after_1)

    def test_integral_anti_windup(self):
        self.ctrl._vel_ki = 1.0
        self.ctrl._vel_ki_max = 0.4
        self.ctrl._vel_kp = 0.0
        for _ in range(500):
            self.ctrl._fresh_vel(0.0, 0.0, 0.0, 0.0)
            self._set_cmds(lift_boom=1.0)
            self.ctrl._send_velocity_commands(0.01)
        self.assertLessEqual(self.ctrl._vel_cmd_integrals['lift_boom'], 0.4 + 1e-6)

    def test_output_clamped_to_one(self):
        self.ctrl._fresh_vel(0.0, 0.0, 0.0, 0.0)
        self.ctrl._vel_kp = 999.0
        self._set_cmds(scoop=1.0)
        self.ctrl._send_velocity_commands(0.01)
        sent = self.ctrl.hardware.send_named_pwm_commands.call_args[0][0]
        self.assertLessEqual(sent['scoop'], 1.0)

    def test_output_clamped_to_minus_one(self):
        self.ctrl._fresh_vel(0.0, 0.0, 0.0, 0.0)
        self.ctrl._vel_kp = 999.0
        self._set_cmds(scoop=-1.0)
        self.ctrl._send_velocity_commands(0.01)
        sent = self.ctrl.hardware.send_named_pwm_commands.call_args[0][0]
        self.assertGreaterEqual(sent['scoop'], -1.0)

    def test_vel_cmd_joints_constant_covers_expected_joints(self):
        self.assertIn('lift_boom', _VEL_CMD_JOINTS)
        self.assertIn('tilt_boom', _VEL_CMD_JOINTS)
        self.assertIn('scoop', _VEL_CMD_JOINTS)
        self.assertNotIn('rotate', _VEL_CMD_JOINTS)
        self.assertNotIn('trackL', _VEL_CMD_JOINTS)
        self.assertNotIn('trackR', _VEL_CMD_JOINTS)


if __name__ == '__main__':
    unittest.main(verbosity=2)
