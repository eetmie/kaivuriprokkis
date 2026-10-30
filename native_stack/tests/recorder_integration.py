"""Dry-only recorder lifecycle, action attribution and failure tests with synthetic IR."""
import json
import math
from pathlib import Path
import subprocess
import resource
import signal
import sys
import tempfile
import time

build, source = map(Path, sys.argv[1:])
sys.path.insert(0, str(source / 'tools'))
from client import Client, DRIVE, DEADMAN
from export_episodes import validate_episode, frame_rows


def wait_until(fn, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = fn()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError('timed out')


with tempfile.TemporaryDirectory(prefix='rec-test-') as temp:
    root = Path(temp)
    control = subprocess.Popen([str(build/'kaivuri-control'), '--profile', str(source.parent/'configuration_files/profiles/jetson'),
        '--safety', str(source/'config/safety.yaml'), '--socket', str(root/'control.sock'), '--simulate-imu', '--control-hz', '200'])
    camera = None
    recorder = None
    try:
        wait_until(lambda: (root/'control.sock').exists())
        # Applied command is a previous successfully written tick, rather than newly received intent.
        with Client(root/'control.sock') as client:
            from client import DISARM
            client.arm()
            client.send(DRIVE, [(1, .2, -.1, .3, 1, -1)], flags=DEADMAN)
            sent_sequence = client.sequence
            time.sleep(.03)
            state = client.status()
            assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(state['applied_action'], [0, .2, -.1, .3, 0, 0]))  # disabled slew/tracks
            assert state['applied_command_sequence'] == sent_sequence
            assert state['applied_host_monotonic_ns'] <= state['host_monotonic_ns']
            client.send(DISARM)
        camera = subprocess.Popen([str(build/'kaivuri-camera'), '--synthetic', '--camera', 'first:1', '--camera', 'second:2',
                                   '--width', '64', '--height', '48', '--output-dir', str(root/'cameras')])
        wait_until(lambda: (root/'cameras/second-ir2.frame').exists())
        base = [str(build/'kaivuri-record'), '--socket', str(root/'control.sock'), '--task', 'test', '--scripted',
                '--camera-file', str(root/'cameras/first-ir1.frame'), '--camera-file', str(root/'cameras/second-ir2.frame')]
        def launch(name, *extra):
            return subprocess.Popen([*base, '--root', str(root/name), *extra], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def command(text):
            recorder.stdin.write(text+'\n'); recorder.stdin.flush()
        recorder = launch('save', '--auto-record-s', '1')
        out, err = recorder.communicate(timeout=6)
        assert recorder.returncode == 0, (out, err)
        episode = next((root/'save').glob('*.ready'))
        session = json.loads((root/'save/session.json').read_text())
        shape, count, hz = validate_episode(episode, session)
        assert shape == [(48, 64), (48, 64)] and 28 <= count <= 32
        rows = list(frame_rows(episode))
        assert rows[0]['cam1']['ir_index'] == 1 and rows[0]['cam2']['ir_index'] == 2
        assert len(rows[0]['status']['fk_joint_xyz_m']) == 4
        assert rows[0]['status']['simulated_imu'] and not rows[0]['status']['live_pwm']
        # Depth: same-exposure pairing by sensor timestamp, lossless zstd spool, export checks.
        depth_camera = subprocess.Popen([str(build/'kaivuri-camera'), '--synthetic', '--camera', 'dfirst:1', '--camera', 'dsecond:2',
            '--depth', 'dfirst', '--depth', 'dsecond', '--rotate-180', 'dfirst', '--width', '64', '--height', '48',
            '--output-dir', str(root/'dcameras')])
        try:
            wait_until(lambda: (root/'dcameras/dsecond-depth.frame').exists())
            recorder = subprocess.Popen([str(build/'kaivuri-record'), '--socket', str(root/'control.sock'), '--task', 'test',
                '--scripted', '--root', str(root/'depth'), '--auto-record-s', '1',
                '--camera-file', str(root/'dcameras/dfirst-ir1.frame'), '--camera-file', str(root/'dcameras/dsecond-ir2.frame'),
                '--cam1-depth-file', str(root/'dcameras/dfirst-depth.frame'), '--cam2-depth-file', str(root/'dcameras/dsecond-depth.frame')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = recorder.communicate(timeout=6)
            assert recorder.returncode == 0, (out, err)
        finally:
            depth_camera.terminate(); depth_camera.wait(timeout=5)
        from export_episodes import depth_frames
        episode = next((root/'depth').glob('*.ready'))
        session = json.loads((root/'depth/session.json').read_text())
        assert session['depth'] == ['cam1', 'cam2'] and json.loads((root/'depth/cam1.depth.json').read_text())['synthetic']
        shape, count, hz = validate_episode(episode, session)
        assert 28 <= count <= 32
        rows = list(frame_rows(episode))
        for key in ('cam1', 'cam2'):
            frames = list(depth_frames(episode/f'{key}.depth.zst'))
            assert len(frames) == count
            for row, data in zip(rows, frames):
                d = row[f'{key}_depth']
                assert d['sensor_timestamp_us'] == row[key]['sensor_timestamp_us'] and d['ir_index'] == 0
                z, level = memoryview(data).cast('H'), 400 + row[key]['sequence'] % 600
                ramp = [level + x for x in range(64)]  # synthetic row; camera 1 is rotated 180 degrees
                assert list(z[:64]) == (ramp[::-1] if key == 'cam1' else ramp), list(z[:4])
        raw = episode/'cam2.depth.zst'
        data = raw.read_bytes(); raw.write_bytes(data[:-3])
        try: validate_episode(episode, session)
        except ValueError as e: assert 'depth' in str(e), e
        else: raise AssertionError('truncated depth accepted')
        raw.write_bytes(data)
        # Export must reject truncated pixels and inaccurate timebases before creating a dataset.
        raw = episode/'cam1.y8'
        data = raw.read_bytes(); raw.write_bytes(data[:-1])
        try: validate_episode(episode, session)
        except ValueError as e: assert 'pixels' in str(e)
        else: raise AssertionError('truncation accepted')
        raw.write_bytes(data)
        try: validate_episode(episode, dict(session, fps=15))
        except ValueError as e: assert 'declared' in str(e)
        else: raise AssertionError('wrong timebase accepted')
        recorder = launch('discard', '--episode-offset', '21')
        command('arm'); time.sleep(.4); command('start'); time.sleep(.4); command('discard')
        time.sleep(.1)
        assert not list((root/'discard').glob('episode-*')), 'discarded take leaked'
        command('start'); time.sleep(.4); command('save'); command('quit')
        out, err = recorder.communicate(timeout=5)
        assert recorder.returncode == 0, (out, err)
        assert 'Discarded ep 22:' in out and 'Saved ep 22:' in out, out
        assert 'Saved episodes in session: 22' in out, out
        episode = next((root/'discard').glob('*.ready'))
        assert json.loads((episode/'episode.json').read_text())['episode_number'] == 22
        recorder = launch('interrupt')
        command('arm'); time.sleep(.4); command('start'); time.sleep(.4)
        command('arm'); time.sleep(.05)  # Repeated Y must not end a recording or change its pump/control state.
        recorder.send_signal(signal.SIGINT)
        out, err = recorder.communicate(timeout=5)
        assert recorder.returncode == 0 and list((root/'interrupt').glob('*.ready')), (out, err)
        def disk_limit():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (8192, 8192))
        recorder = subprocess.Popen([*base, '--root', str(root/'disk-failed'), '--auto-record-s', '3'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=disk_limit)
        out, err = recorder.communicate(timeout=5)
        assert recorder.returncode == 1 and ('write failed' in err or 'incomplete episode' in err), (out, err)
        assert not list((root/'disk-failed').glob('*.ready')) and list((root/'disk-failed').glob('*.partial'))
        recorder = launch('stalled', '--auto-record-s', '5')
        wait_until(lambda: list((root/'stalled').glob('*.partial')))
        time.sleep(.3)
        camera.terminate(); camera.wait(timeout=5)
        out, err = recorder.communicate(timeout=5)
        assert recorder.returncode == 1 and 'camera stalled' in err, (out, err)
        assert not list((root/'stalled').glob('*.ready')) and list((root/'stalled').glob('*.partial'))
        with Client(root/'control.sock') as client:
            state = wait_until(lambda: (s if (s := client.status())['state'] == 'disarmed' else None))
            assert not state['pump_on']
        recorder = launch('flags-differ', '--auto-record-s', '.5', '--enable-tracks')
        out, err = recorder.communicate(timeout=5)
        assert recorder.returncode == 1 and 'flags differ' in err
        # A controller fault mid-take keeps that take as .partial, re-arms without operator
        # input and the session continues.
        camera = subprocess.Popen([str(build/'kaivuri-camera'), '--synthetic', '--camera', 'first:1', '--camera', 'second:2',
                                   '--width', '64', '--height', '48', '--output-dir', str(root/'cameras2')])
        wait_until(lambda: (root/'cameras2/second-ir2.frame').exists())
        recorder = subprocess.Popen([str(build/'kaivuri-record'), '--socket', str(root/'control.sock'), '--task', 'test',
            '--scripted', '--root', str(root/'recover'), '--camera-file', str(root/'cameras2/first-ir1.frame'),
            '--camera-file', str(root/'cameras2/second-ir2.frame')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with Client(root/'control.sock') as client:
            from client import ESTOP
            wait_until(lambda: client.status()['state'] == 'armed')
            command('start')
            wait_until(lambda: list((root/'recover').glob('*.partial')))
            time.sleep(.3)
            client.send(ESTOP)
            wait_until(lambda: client.status()['state'] == 'fault', timeout=1)
            wait_until(lambda: client.status()['state'] == 'armed', timeout=4)
        command('start'); time.sleep(.4); command('save'); command('quit')
        out, err = recorder.communicate(timeout=5)
        assert recorder.returncode == 0, (out, err)
        assert 'kept incomplete' in err and 're-arming automatically' in err, err
        assert len(list((root/'recover').glob('*.partial'))) == 1 and len(list((root/'recover').glob('*.ready'))) == 1
    finally:
        for proc in (recorder, camera, control):
            if proc is not None and proc.poll() is None:
                proc.terminate(); proc.wait(timeout=5)
print('Recorder lifecycle, camera failure, fault recovery, action attribution and export validation passed')
