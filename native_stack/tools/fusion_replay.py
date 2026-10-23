"""Replay the Pico's x-io Fusion AHRS offline over a recorded episode's gyro/accel.

    fusion_replay.py EPISODE [--gain 0.5] [--rejection 5] [--recovery-s 2] [--compare-camera]

Recordings carry each IMU's gyro (dps) and accel (g) as Fusion received them (after the sensor's
low-pass filters) at 100 Hz; the Pico runs Fusion at 416 Hz, so a replay reproduces the firmware's
joint angles to about 0.2-0.3 deg p99. That is close enough to compare settings on identical motion
before reflashing. Joint angles use the controller's extraction (gravity-pitch deltas after the
mounting offsets from the session's saved profile). --compare-camera scores firmware and replay
against the side camera (sidecam_audit.py, needs tag detections).
The C driver (fusion_replay/replay.c) is compiled with `cc` against pico_imu_reader's vendored Fusion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np

from export_episodes import samples

SOURCE = Path(__file__).resolve().parents[1]
FUSION = SOURCE.parent / 'pico_imu_reader/src/imu_reader/Fusion/Fusion'
ROLES = ('base', 'boom', 'arm', 'bucket')  # status rows follow [slew, boom, arm, bucket]
GYRO_SATURATION_DPS = 573.4                # ISM330 at the 500 dps setting


def binary():
    sources = [Path(__file__).with_name('fusion_replay') / 'replay.c', FUSION / 'FusionAhrs.c', FUSION / 'FusionOffset.c']
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:12]
    out = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / f'kaivuri/fusion_replay-{digest}'
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cc', '-O2', f'-I{FUSION}', *map(str, sources), '-lm', '-o', str(out)], check=True)
    return out


def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def gravity_pitch(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(-2 * (x * z - w * y), 1 - 2 * (x * x + y * y))


def joint_angles(quats, offsets):
    """quats (n, 4 roles, 4): boom/arm/bucket joint angles as in core.cpp (corrected = raw * conj(offset))."""
    conj = np.asarray(offsets, float) * [1, -1, -1, -1]
    p = [gravity_pitch(qmul(quats[:, k], np.broadcast_to(conj[k], (len(quats), 4)))) for k in range(4)]
    return np.stack([np.angle(np.exp(1j * (p[k] - p[k - 1]))) for k in (1, 2, 3)], 1)


def session_offsets(episode):
    """Mounting offsets from the profile the controller ran with (saved under provenance/)."""
    import yaml
    session = episode.parent
    manifest = json.loads((session / 'provenance/manifest.json').read_text())
    args = manifest['controller_arguments']; profile = Path(args[args.index('--profile') + 1])
    for path in (session / 'provenance/configuration_files/profiles' / profile.name / 'control_config.yaml',
                 session / 'provenance/profile/control_config.yaml'):
        if path.exists():
            m = yaml.safe_load(path.read_text())['imu']['mounting_offsets_quat']
            return np.array([m[r] for r in ROLES], float)
    raise SystemExit('no saved profile in the session provenance')


def load(episode):
    st = [row['status'] for row in samples(episode)]
    return dict(dev=np.array([s['imu_device_us'] for s in st]) * 1e-6, q=np.array([s['imu_quat_wxyz'] for s in st]),
                g=np.array([s['imu_gyro_dps'] for s in st]), a=np.array([s['imu_accel_g'] for s in st]),
                ang=np.array([s['angles_rad'] for s in st]))


def replay(d, role, gain, rejection, recovery_s, exe=None):
    dt = np.clip(np.diff(d['dev'], prepend=d['dev'][0] - .01), .001, .05)
    rec = np.c_[dt, d['g'][:, role], d['a'][:, role]].astype(np.float32)
    blob = struct.pack('<i4f4f', len(rec), gain, rejection, recovery_s / np.median(dt), GYRO_SATURATION_DPS, *d['q'][0, role]) + rec.tobytes()
    out = subprocess.run([str(exe or binary())], input=blob, capture_output=True, check=True).stdout
    return np.frombuffer(out[:16 * len(rec)], np.float32).reshape(-1, 4).astype(float)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('episode', type=Path)
    p.add_argument('--gain', type=float, default=.5)
    p.add_argument('--rejection', type=float, default=5., help='acceleration rejection, degrees')
    p.add_argument('--recovery-s', type=float, default=2.)
    p.add_argument('--compare-camera', action='store_true')
    a = p.parse_args()
    episode = a.episode.resolve(); d = load(episode); offsets = session_offsets(episode); exe = binary()
    firmware = joint_angles(d['q'], offsets)
    check = np.degrees(np.abs(firmware - d['ang'][:, 1:])).max()
    q = np.stack([replay(d, k, a.gain, a.rejection, a.recovery_s, exe) for k in range(4)], 1)
    replayed = joint_angles(q, offsets)
    diff = np.degrees(np.percentile(np.abs(replayed - firmware), 99, axis=0))
    print(f'{episode.parent.name}/{episode.name}: extraction matches angles_rad to {check:.3f} deg')
    print(f'replay gain {a.gain} / rejection {a.rejection} deg / recovery {a.recovery_s} s vs firmware joint angles, p99 deg: '
          f'boom {diff[0]:.3f}, arm {diff[1]:.3f}, bucket {diff[2]:.3f}')
    if a.compare_camera:
        import sidecam_audit as sc
        cam = sc.load(episode); sc.geometry(cam)
        for label, ang in (('firmware', d['ang']), ('replay', np.c_[d['ang'][:, :1], replayed])):
            r = sc.compare(cam, ang)
            print(f'  {label:8s} camera-IMU rms/p99 deg: ' + ', '.join(f'{k} {v["rms_deg"]:.2f}/{v["p99_deg"]:.2f}' for k, v in r.items()))


if __name__ == '__main__':
    main()
