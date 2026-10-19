"""Dry-only byte-stream replay, capture lifecycle and training compatibility checks."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pty
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time

BINARY = PROFILE = None


def packet(timestamp, t=0.0, invalid=False, packet_type=4):
    # Physical indices: arm, base, boom, bucket. Raw quaternions retain mounting.
    pitches = [.9 + .03 * math.cos(.7*t) + .010681414,
               .002 * math.sin(.8*t),
               -.3 + .05 * math.sin(1.1*t) + .24173176,
               .2 + .04 * math.sin(.9*t)]
    rates = [-.021 * math.sin(.7*t), .0016 * math.cos(.8*t),
             .055 * math.cos(1.1*t), .036 * math.cos(.9*t)]
    body = bytearray(struct.pack('<BBI', packet_type, 4, timestamp & 0xffffffff))
    for i, (angle, rate) in enumerate(zip(pitches, rates)):
        values = [math.nan if invalid and i == 0 else math.cos(angle/2),
                  0, math.sin(angle/2), 0, 0, math.degrees(rate), 0, 0, 0, 1]
        body.extend(struct.pack('<10f' if packet_type == 4 else '<7f',
                                *(values if packet_type == 4 else values[:7])))
    return b'\xaa\x55' + body + struct.pack('<H', sum(body) & 65535)


def descriptor():
    body = struct.pack('<BBHff8B', 6, 4, 200, 2000, 16, 0, 0x6a, 0, 0x6b, 1, 0x6a, 1, 0x6b)
    return b'\xaa\x55' + body + struct.pack('<H', sum(body) & 65535)


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify(root):
    manifests = sorted(root.glob('session_*.json'))
    assert manifests, (list(root.iterdir()), 'no committed recording')
    rows, raw = [], []
    for path in manifests:
        m = json.loads(path.read_text())
        assert m['format'] == 'kaivuri-native-mlp-v1' and m['complete'] and m['dry_pwm']
        assert m['dropped'] == 0 and m['queue_high_water'] < m['queue_capacity']
        for name, expected in m['outputs'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
        stem = path.name.removeprefix('session_').removesuffix('.json')
        d, r = read_csv(root/f'drive_log_{stem}.csv'), read_csv(root/f'imu_raw_{stem}.csv')
        assert len(d) == m['rows'] and len(r) == m['raw_packets']
        assert all(int(b['sample_idx']) - int(a['sample_idx']) ==
                   1 + int(b['missed_periods']) - int(a['missed_periods']) for a,b in zip(d,d[1:]))
        assert all(int(b['device_ts_us']) > int(a['device_ts_us']) for a,b in zip(r,r[1:]))
        lookup = {int(p['device_ts_us']): p for p in r}
        for row in d:
            if row['imu_valid'] != '1':
                assert row['cmd_stale'] == '1'
                continue
            p = lookup[int(row['state_imu_ts_us'])]
            assert int(p['host_read_ns']) <= int(row['post_write_ns'])
            assert 0 <= float(row['state_age_s']) <= .03
            assert float(row['timestamp']) == int(row['post_write_ns'])/1e9
            assert row['write_ok'] == '1'
        for row in d:
            for channel in ('lift','tilt','scoop'):
                assert -1 <= float(row[f'combined_cmd_{channel}']) <= 1
        rows.extend(d); raw.extend(r)
    return rows, raw


def run(root, seconds=3.0, reset=False, corruption=False, waveform='chirp', chunk=7, packet_type=4):
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    process = subprocess.Popen([str(BINARY), '--dry', '--scripted', '--record-on-start',
        '--profile', str(PROFILE), '--serial', os.ttyname(slave), '--out', str(root),
        '--duration-s', str(seconds), '--read-chunk-bytes', str(chunk),
        '--excitation', waveform, '--excitation-target', 'lift', '--excitation-seed', '42'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stop = threading.Event()
    sent = []
    def feed():
        index, base, did_reset = 0, 2**32-500000, False
        next_pair = time.monotonic()
        while not stop.is_set():
            if reset and not did_reset and index >= 240:
                os.write(master, descriptor()); base = 1000-index*5000; did_reset=True
            payload = bytearray()
            for _ in range(2):
                bad = corruption and index == 120
                checksum = corruption and index == 180
                gap = corruption and 320 <= index < 340
                data = bytearray(packet(base+index*5000, index*.005, invalid=bad, packet_type=packet_type))
                if checksum: data[-1] ^= 1
                if not gap: payload.extend(data)
                if not (bad or checksum or gap): sent.append((base+index*5000)&0xffffffff)
                index += 1
            while payload and not stop.is_set():
                try:
                    written = os.write(master, payload)
                    del payload[:written]
                except BlockingIOError:
                    stop.wait(.001)
                except OSError:
                    return
            next_pair += .01
            stop.wait(max(0, next_pair-time.monotonic()))
    thread = None
    try:
        line = process.stdout.readline()
        assert 'MLP collector' in line, (line, process.stderr.read())
        process.stdin.write('pump\nexcitation\n'); process.stdin.flush()
        thread = threading.Thread(target=feed); thread.start()
        out, err = process.communicate(timeout=seconds+10)
        assert process.returncode == 0, (line+out, err)
    finally:
        stop.set()
        if thread: thread.join(timeout=2)
        if process.poll() is None: process.kill(); process.wait()
        os.close(master); os.close(slave)
    rows, raw = verify(root)
    assert len(rows) > seconds*85, (len(rows), out, err)
    assert len(raw) > seconds*170, (len(raw), out, err)
    assert len({int(x['sequence']) for x in raw}) == len(raw), 'duplicated raw packet'
    # Only the final packets still in the kernel at process exit may be outside capture.
    recorded = [int(x['raw_device_ts_us']) for x in raw]
    assert recorded == sent[:len(recorded)], (recorded[:5], sent[:5], len(recorded),len(sent))
    # File flush/hash/fsync after capture can outlast several transmitter periods.
    # The prefix equality above proves there are no holes within the captured span.
    assert any(float(x['sine_cmd_lift']) != 0 for x in rows)
    assert all(float(x['sine_cmd_tilt']) == 0 and float(x['sine_cmd_scoop']) == 0 for x in rows)
    assert any(int(x['device_ts_us']) > 2**32 for x in raw)
    return rows, raw


def suite(root):
    run(root/'burst-wrap', seconds=3, chunk=7)
    rows, raw = run(root/'reset-invalid', seconds=3, reset=True, corruption=True, chunk=512)
    assert any(row['imu_valid'] == '0' for row in rows), 'deliberate stream gap was not marked stale'
    assert any(float(row['gyro_range_dps']) == 2000 and float(row['accel_range_g']) == 16 for row in raw)
    assert len(list((root/'reset-invalid').glob('session_*.json'))) >= 2
    run(root/'stroke', seconds=2, waveform='stroke')
    rows, raw = run(root/'type1', seconds=1, packet_type=1)
    assert all(row['packet_type'] == '1' and math.isnan(float(row['imu_arm_ax_g'])) for row in raw)
    valid = [row for row in rows if row['imu_valid'] == '1']
    assert all(math.isnan(float(row['imu_ax_arm'])) and math.isfinite(float(row['imu_gy_arm'])) for row in valid)
    assert all(math.isnan(float(row['joint_vel_arm'])) for row in valid)


def main():
    global BINARY, PROFILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--artifacts', type=Path, help='Keep a six-second round-trip fixture here')
    args = parser.parse_args()
    BINARY, PROFILE = args.binary.resolve(), args.profile.resolve()
    if args.artifacts:
        run(args.artifacts, seconds=6, chunk=256)
    else:
        with tempfile.TemporaryDirectory(prefix='mlp-integration-') as temp:
            suite(Path(temp))
    print('MLP byte replay, raw capture and waveform routing passed')


if __name__ == '__main__':
    main()
