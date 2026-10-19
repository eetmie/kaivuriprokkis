"""Collector lifecycle with real PWM register bytes redirected to a test file."""

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import pty
import resource
import signal
import struct
import subprocess
import tempfile
import threading
import time

from mlp_integration import packet


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError('Timed out waiting for collector state')


@contextlib.contextmanager
def collector(args, root, extra=(), pad=None, full_disk=False):
    root.mkdir()
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    preload = ':'.join(filter(None, (os.environ.get('KAIVURI_TEST_ASAN_RUNTIME'), str(args.device))))
    env = dict(os.environ, LD_PRELOAD=preload, KAIVURI_TEST_PWM_TRACE=str(root/'pwm.bin'))
    command = [str(args.binary), '--profile', str(args.profile), '--serial', os.ttyname(slave),
               '--out', str(root), '--duration-s', '15', *extra]
    command += ['--gamepad', str(pad)] if pad else ['--scripted']

    def file_limit():
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, env=env,
                               preexec_fn=file_limit if full_disk else None)
    stop = threading.Event()

    def feed():
        index, next_pair = 0, time.monotonic()
        while not stop.is_set():
            payload = bytearray(packet(1000+index*5000, index*.005) +
                                packet(6000+index*5000, (index+1)*.005))
            index += 2
            while payload and not stop.is_set():
                try:
                    del payload[:os.write(master, payload)]
                except BlockingIOError:
                    stop.wait(.001)
                except OSError:
                    return
            next_pair += .01
            stop.wait(max(0, next_pair-time.monotonic()))

    thread = threading.Thread(target=feed)
    try:
        line = process.stdout.readline()
        assert 'PWM LIVE' in line, (line, process.stderr.read())
        thread.start()
        yield process
    finally:
        stop.set()
        if thread.is_alive():
            thread.join(timeout=2)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        out, err = process.communicate(timeout=10)
        os.close(master)
        os.close(slave)
        assert process.returncode == 0, (out, err)
    check_neutral(root/'pwm.bin')


def send(process, text):
    process.stdin.write(text+'\n')
    process.stdin.flush()


def read_rows(root):
    rows = []
    for path in sorted(root.glob('drive_log_*.csv')):
        with path.open() as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def check_neutral(path):
    # Register address distinguishes two-byte setup writes from the PWM burst.
    data, bursts = path.read_bytes(), []
    index = 0
    while index < len(data):
        register = data[index]
        length = 37 if register == 6 else 2  # Jetson channels 0..8 inclusive
        assert register in (0, 1, 0xfe, 6), (index, register)
        payload = data[index:index+length]
        assert len(payload) == length
        if register == 6:
            bursts.append(payload)
        index += length
    assert len(bursts) > 3
    assert bursts[-1] == bursts[0] == bursts[-2], 'Exit did not repeat startup neutral and pump minimum'
    return bursts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'profile', 'device'):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mlp-lifecycle-') as temp:
        root = Path(temp)
        repeated = root/'repeated'
        with collector(args, repeated) as process:
            second = subprocess.run([str(args.binary), '--dry', '--scripted', '--out', str(root/'excluded'),
                                     '--profile', str(args.profile), '--serial', '/dev/null'], capture_output=True, text=True)
            assert second.returncode and 'another native controller' in second.stderr
            send(process, 'pump\ncommand 0 .3 -.2 .1 0 0')
            for count in range(1, 4):
                send(process, 'start\nexcitation')
                time.sleep(.18)
                send(process, 'save')
                wait_until(lambda: len(list(repeated.glob('session_*.json'))) == count)
            send(process, 'start')
            time.sleep(.18)  # SIGINT saves this fourth strip.
        assert len(list(repeated.glob('session_*.json'))) == 4
        rows = read_rows(repeated)
        assert len(rows) >= 60 and all(row['pump_on'] == '1' for row in rows)
        assert all(float(row['manual_cmd_lift']) == .3 for row in rows)
        assert any(float(row['sine_cmd_lift']) != 0 for row in rows)

        timeout = root/'timeout'
        with collector(args, timeout, ['--record-on-start', '--record-seconds', '.2']) as process:
            wait_until(lambda: list(timeout.glob('session_*.json')))
            time.sleep(.08)
        assert 15 <= len(read_rows(timeout)) <= 24

        for flag, value in [('--excitation-amplitude','nan'), ('--duration-s','1junk'),
                            ('--queue-records','17'), ('--excitation-seed','-1')]:
            bad = subprocess.run([str(args.binary), '--dry', '--scripted', '--out', str(root/'invalid'), flag, value],
                                 capture_output=True, text=True)
            assert bad.returncode != 0 and not (root/'invalid').exists()

        disk = root/'disk-full'
        with collector(args, disk, ['--record-on-start'], full_disk=True):
            time.sleep(.3)
        assert list(disk.glob('*.partial')) and not list(disk.glob('session_*.json'))

        fifo = root/'pad.fifo'
        os.mkfifo(fifo)
        pad_fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        disconnected = root/'disconnect'
        with collector(args, disconnected, ['--record-on-start'], pad=fifo):
            events = [(3, 4, -16000), (1, 305, 1), (1, 307, 1), (3, 17, -1)]
            os.write(pad_fd, b''.join(struct.pack('llHHi', 0, 0, *event) for event in events))
            time.sleep(.2)
            os.close(pad_fd)
            time.sleep(.2)
        rows = read_rows(disconnected)
        assert any(row['input_valid'] == '1' and float(row['manual_cmd_lift']) > .2 for row in rows)
        assert any(row['sine_target'] == 'lift' for row in rows), 'Xbox hat D-pad did not select lift'
        lost = [row for row in rows if row['input_valid'] == '0']
        assert len(lost) >= 10
        assert all(row['cmd_stale'] == '1' and row['sine_enabled'] == '0' and
                   all(float(row[f'written_cmd_{name}']) == 0 for name in ('slew','boom','arm','bucket','trackL','trackR'))
                   for row in lost)
    print('MLP repeated strips, SIGINT, timeout, full disk, pad loss and neutral register bytes passed')


if __name__ == '__main__':
    main()
