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
        recorder = launch('discard')
        command('arm'); time.sleep(.4); command('start'); time.sleep(.4); command('discard'); command('quit')
        out, err = recorder.communicate(timeout=5)
        assert recorder.returncode == 0, (out, err)
        assert not list((root/'discard').glob('episode-*')), 'discarded take leaked'
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
        recorder = launch('live-refused', '--allow-live', '--auto-record-s', '.5', '--enable-tracks')
        out, err = recorder.communicate(timeout=5)
        assert recorder.returncode == 1 and 'flags differ' in err
    finally:
        for proc in (recorder, camera, control):
            if proc is not None and proc.poll() is None:
                proc.terminate(); proc.wait(timeout=5)
print('Recorder lifecycle, camera failure, action attribution and export validation passed')
