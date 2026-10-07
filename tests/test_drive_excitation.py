"""Hardware-free excitation, recording lifecycle and replay checks."""

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simple_drive as drive


def test_sine_enters_from_zero():
    gen = drive.SineExcitationGenerator(enabled=True, seed=42)
    assert all(value == 0 for value in gen.get_all(10.0).values())
    assert max(abs(value) for value in gen.get_all(10.01).values()) < 0.001


def test_reseed_restarts_phase_for_repeated_recordings():
    gen = drive.SineExcitationGenerator(enabled=True, seed=42)
    gen.get_all(0.0)
    gen.get_all(1.0)
    gen.reseed(42)
    assert gen.start_time is None
    assert all(value == 0 for value in gen.get_all(20.0).values())


def test_sampling_does_not_change_next_parameter_draw():
    a = drive.SineExcitationGenerator(enabled=True, seed=42)
    b = drive.SineExcitationGenerator(enabled=True, seed=42)
    a.get_all(0.0)
    for t in np.arange(0.01, 2, 0.01):
        a.get_all(float(t))
    a.randomize()
    b.randomize()
    assert a._params == b._params


def test_same_elapsed_time_replays_despite_different_polling():
    a = drive.SineExcitationGenerator(enabled=True, seed=42)
    b = drive.SineExcitationGenerator(enabled=True, seed=42)
    a.get_all(0.0)
    b.get_all(100.0)
    for t in np.arange(0.01, 2, 0.01):
        a.get_all(float(t))
    assert a.get_all(2.0) == b.get_all(102.0)
    assert a.metadata(2.0)['noise_tick'] == 200


def test_target_change_starts_new_joint_at_zero_and_disable_is_immediate():
    gen = drive.SineExcitationGenerator(enabled=True, seed=5)
    gen.get_all(0.0)
    gen.get_all(2.0)
    gen.step_target(1)
    assert gen.target_name == 'lift'
    assert all(value == 0 for value in gen.get_all(3.0).values())
    assert gen.get_all(4.0)['boom'] != 0
    gen.disable()
    assert all(value == 0 for value in gen.get_all(4.01).values())


def test_chirp_phase_matches_frequency_integral():
    gen = drive.ChirpExcitationGenerator(seed=1, start_hz=.05, end_hz=.9, sweep_s=60)
    for t in (1.0, 30.0, 59.0, 61.0, 90.0, 119.0):
        h = 1e-4
        derivative = (gen.chirp_phase(t + h) - gen.chirp_phase(t - h)) / (2 * h)
        expected = .05 * 18 ** (t / 60) if t < 60 else .9 * (1 / 18) ** ((t - 60) / 60)
        assert derivative / (2 * np.pi) == pytest.approx(expected, rel=1e-7)
    assert gen.chirp_phase(60 - 1e-7) == pytest.approx(gen.chirp_phase(60 + 1e-7))


def test_chirp_limits_rest_interval_and_slew_isolation():
    gen = drive.ChirpExcitationGenerator(enabled=True, seed=3, amplitude=.25)
    gen.start_time = 0.0
    values = [gen.get_all(float(t)) for t in np.arange(0, 131, .1)]
    assert max(abs(v) for row in values for v in row.values()) <= .25
    assert all(row['slew'] == 0 for row in values)
    for t in (120.0, 125.0, 129.99, 130.0):
        assert all(value == 0 for value in gen.get_all(t).values())
    assert gen.metadata(125.0)['stage'] == 'rest'
    gen = drive.ChirpExcitationGenerator(enabled=True, seed=3, enable_slew=True)
    gen.target_idx = len(gen.modes) - 1
    gen.start_time = 0.0
    signal = gen.get_all(2.0)
    assert signal['slew'] != 0
    assert all(signal[j] == 0 for j in ('boom', 'arm', 'bucket'))


@pytest.mark.parametrize('argv', [
    ['--chirp-start-hz', 'nan'], ['--chirp-end-hz', 'inf'],
    ['--chirp-start-hz', '0'], ['--chirp-end-hz', '20'],
    ['--excitation', 'stroke', '--stroke-margin-deg', 'nan'],
    ['--excitation', 'stroke', '--stroke-seconds', '0'],
    ['--chirp-start-hz', '.8', '--chirp-end-hz', '.1'],
    ['--chirp-seconds', '1'], ['--excitation-amplitude', '0'],
    ['--excitation-amplitude', '1.1'], ['--excitation-seed', '-1'],
    ['--excitation-target', 'slew'],
])
def test_invalid_options_rejected_before_hardware(argv):
    with patch.object(sys, 'argv', ['simple_drive.py', *argv]), pytest.raises(SystemExit):
        drive._parse_args()


def test_stop_recording_disables_excitation_even_for_empty_capture(tmp_path):
    gen = drive.ChirpExcitationGenerator(enabled=True, seed=3)
    logger = drive.DataLogger(tmp_path)
    logger.start()
    direct = Mock()
    logger.stop_and_save(direct, excitation=gen)
    assert not gen.enabled
    assert not logger.is_logging
    direct.clear.assert_called_once()
    direct.send_pending.assert_called_once()
    assert all(value == 0 for value in gen.get_all(1.0).values())


def test_stale_input_clears_manual_tracks_and_excitation_without_auto_resume():
    gen = drive.ChirpExcitationGenerator(enabled=True, seed=3)
    manual, signal, combined = drive.build_drive_commands(
        {'boom': .3, 'trackR': .8}, gen, 1.0, cmd_stale=True)
    assert all(value == 0 for value in combined.values())
    assert not gen.enabled
    _, signal, combined = drive.build_drive_commands({'boom': .2}, gen, 2.0, cmd_stale=False)
    assert all(value == 0 for value in signal.values())
    assert combined['boom'] == .2


def test_udp_liveness_uses_packet_age_even_with_cached_commands():
    source = drive.UDPInput('0.0.0.0', 8080)
    source._sock = Mock()
    source._sock.get_connection_stats.return_value = {'data_age_seconds': .6}
    assert source.command_age_s() == .6
    assert not source.is_live()
    source._sock.get_connection_stats.return_value = {'data_age_seconds': .1}
    assert source.is_live()


def test_cli_builds_chirp_with_requested_target_and_seed():
    with patch.object(sys, 'argv', ['simple_drive.py', '--excitation', 'chirp',
                                  '--excitation-target', 'scoop', '--excitation-seed', '42']):
        gen = drive.make_excitation(drive._parse_args())
    assert isinstance(gen, drive.ChirpExcitationGenerator)
    assert gen.target_joints == ('bucket',)
    assert gen.seed == 42
    assert gen.amplitude == .35
    assert not gen.enabled


@pytest.mark.parametrize('stop_reason', ['button', 'timeout', 'disconnect'])
def test_main_records_then_keeps_excitation_off(tmp_path, stop_reason):
    """Exercise real main/DirectController/logger with an entirely mocked device."""
    clock = [100.0]
    sent = []
    hardware = Mock()
    hardware.pwm_controller = None
    hardware.try_read_imu_gyro.return_value = None
    hardware.send_named_pwm_commands.side_effect = lambda commands: sent.append(dict(commands))
    profile = dict(drive._resolve_board_profile('rpi'))
    profile['enable_imu'] = False

    class Input:
        name = 'local'
        tick = 0

        def open(self):
            return True

        def poll(self):
            self.tick += 1
            clock[0] += .1
            if self.tick > 25:
                raise KeyboardInterrupt
            mask = (1 << drive.BTN_A) if self.tick == 1 else 0
            if self.tick == 3:
                mask = 1 << drive.BTN_B
            if self.tick == 12 and stop_reason == 'button':
                mask = 1 << drive.BTN_A
            return dict.fromkeys(('right_rl', 'right_ud', 'left_rl', 'left_ud',
                                  'right_paddle', 'left_paddle'), 0.0), mask

        def is_live(self):
            return not (stop_reason == 'disconnect' and self.tick >= 12)

        def command_age_s(self):
            return 0.0 if self.is_live() else float('inf')

        def close(self):
            pass

    with ExitStack() as patches:
        patches.enter_context(patch.object(sys, 'argv', ['simple_drive.py', '--robot', 'rpi',
                                                        '--excitation', 'chirp', '--excitation-seed', '42']))
        patches.enter_context(patch.object(drive, '_resolve_profile', return_value=profile))
        patches.enter_context(patch.object(drive, 'make_input_source', return_value=Input()))
        patches.enter_context(patch.object(drive, 'LOG_OUTPUT_DIR', tmp_path))
        patches.enter_context(patch.object(drive, 'RECORD_MINUTES', 1 / 60 if stop_reason == 'timeout' else 10))
        patches.enter_context(patch.object(drive, 'wait_for_hardware_ready'))
        patches.enter_context(patch('modules.hardware_interface.HardwareInterface', return_value=hardware))
        patches.enter_context(patch('modules.rt_utils.apply_rt_to_thread'))
        for name in ('time', 'perf_counter', 'monotonic'):
            patches.enter_context(patch.object(drive.time, name, side_effect=lambda: clock[0]))
        patches.enter_context(patch.object(drive.time, 'sleep'))
        drive.main()
    assert any(abs(v) > .001 for row in sent[:12] for v in row.values())
    assert all(v == 0 for row in sent[-10:] for v in row.values())
    assert len(list(tmp_path.glob('drive_log_*.csv'))) == 1
    assert len(list(tmp_path.glob('excitation_*.json'))) == 1
    hardware.shutdown.assert_called_once()


def test_log_records_clipping_and_companion_parameters(tmp_path):
    gen = drive.ChirpExcitationGenerator(enabled=True, seed=42)
    gen.get_all(0.0)
    logger = drive.DataLogger(tmp_path, imu_roles=['base'])
    logger.start()
    hardware = Mock()
    hardware.try_read_imu_gyro.return_value = None
    for t in (1.0, 2.0):
        logger.log_sample({'boom': .8}, {'boom': .4}, {'boom': 1.0},
                          ([0] * 4, None, None), hardware, 0.0, False, True,
                          gen.target_name, gen.seed, excitation_meta=gen.metadata(t))
    assert logger._cols['command_clipped_lift'] == [1, 1]
    assert logger._cols['effective_excitation_cmd_lift'] == pytest.approx([.2, .2])
    assert logger._cols['excitation_mode'] == ['chirp', 'chirp']
    assert len(logger._excitation_blocks) == 1
    logger.log_imu_raw([(100, [[1.0] + [0.0] * 9])], 1)
    out = logger.save()
    suffix = out.name.removeprefix('drive_log_').removesuffix('.csv')
    meta_path = tmp_path / f'excitation_{suffix}.json'
    meta = json.loads(meta_path.read_text())
    assert meta['drive_log'] == out.name
    assert len(meta['blocks']) == 1
    assert meta['blocks'][0]['parameters']['seed'] == 42
    assert (tmp_path / f'imu_raw_{suffix}.csv').is_file()
    assert logger.n_samples() == 2


LIMITS = {'boom': (-50.0, 30.0), 'arm': (28.0, 144.0), 'bucket': (-120.0, 20.0)}


def _drive_stroke(gen, seconds, q0=(0.0, -20.0, 90.0, -50.0), rate_per_u=40.0):
    """Close the stroke loop on an integrator joint model [deg/s per unit valve]."""
    q = np.array(q0, dtype=float)
    rows = []
    for t in np.arange(0, seconds, .01):
        gen.observe(q)
        signal = gen.get_all(float(t))
        for joint, index in gen.JOINT_INDEX.items():
            q[index] += rate_per_u * signal[joint] * .01
        rows.append((float(t), q.copy(), dict(signal), gen.metadata(float(t))['stage']))
    return rows


def test_stroke_sweeps_inside_margins_and_chirps_every_other_stroke():
    gen = drive.StrokeExcitationGenerator(enabled=True, seed=7, joint_limits_deg=LIMITS, stroke_s=4.0)
    gen.target_idx = [name for name, _ in gen.modes].index('lift')
    rows = _drive_stroke(gen, 30.0)
    boom = np.array([q[1] for _, q, _, _ in rows])
    assert boom.min() > LIMITS['boom'][0] + 5 and boom.max() < LIMITS['boom'][1] - 5
    assert boom.max() - boom.min() > 40          # it really strokes
    assert all(signal['arm'] == 0 and signal['bucket'] == 0 for _, _, signal, _ in rows)
    stages = {stage for *_, stage in rows}
    assert stages == {'stroke', 'stroke+chirp'}
    assert gen._strokes['boom'] >= 3
    assert max(abs(signal['boom']) for _, _, signal, _ in rows) <= 1.0


def test_stroke_outputs_nothing_without_angles_and_stops_outside_limits():
    gen = drive.StrokeExcitationGenerator(enabled=True, seed=1, joint_limits_deg=LIMITS)
    gen.observe(None)
    assert all(v == 0 for v in gen.get_all(0.0).values())
    gen.observe([0.0, 35.0, 90.0, -50.0])          # boom beyond its +30 deg limit
    gen.get_all(0.01)
    assert not gen.enabled
    assert all(v == 0 for v in gen.get_all(0.02).values())


def test_cli_builds_stroke_from_profile_limits(tmp_path):
    config = tmp_path / 'control_config.yaml'
    config.write_text('ik:\n  joint_limits_relative: [null, [-50, 30], [28, 144], [-120, 20]]\n')
    with patch.object(sys, 'argv', ['simple_drive.py', '--excitation', 'stroke', '--excitation-target', 'tilt']):
        args = drive._parse_args()
    gen = drive.make_excitation(args, drive.load_joint_limits_deg(config))
    assert isinstance(gen, drive.StrokeExcitationGenerator)
    assert (gen.start_hz, gen.end_hz, gen.sweep_s, gen.amplitude) == (.5, 12.0, 20.0, .4)
    assert gen.target_name == 'tilt' and gen.limits['arm'] == (28.0, 144.0)
