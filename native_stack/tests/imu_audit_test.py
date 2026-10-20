"""imu_audit.py recovers a known hinge-axis tilt, roll offset, angle gain and lag from synthetic samples."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

source = Path(sys.argv[1])
sys.path.insert(0, str(source / 'tools'))
import imu_audit  # noqa: E402

rng = np.random.default_rng(1)
dt, n = .01, 12000
t = np.arange(n) * dt
# One joint at a time: 20 s of +-40 deg sine per joint (slew, boom, arm, bucket), still gaps between.
rates = np.zeros((n, 4))
for k in range(4):
    s, e = 1000 + k * 2500, 1000 + k * 2500 + 2000
    rates[s:e, k] = 40 * 2 * math.pi * .25 * np.cos(2 * math.pi * .25 * t[s:e])  # dps
# Each IMU's absolute rate: slew about the cabin vertical; pitch joints add up along the chain.
arm_tilt = math.radians(1.5)  # arm IMU's hinge axis leans 1.5 deg toward sensor X
axis = [np.array([0, 0, 1.]), np.array([0, 1., 0]), np.array([math.sin(arm_tilt), math.cos(arm_tilt), 0]), np.array([0, 1., 0])]
pitch = np.cumsum(rates[:, 1:], axis=1)  # absolute pitch rate of boom, arm, bucket IMUs
gyro = np.zeros((n, 4, 3))
gyro[:, 0] = np.outer(rates[:, 0], axis[0])
for k in (1, 2, 3):
    gyro[:, k] = np.outer(pitch[:, k - 1], axis[k])
gyro += rng.normal(0, .05, gyro.shape)
# Recorded angles: gain 0.98 on the arm, 30 ms lag on the bucket.
gain, lag = np.ones(4), np.zeros(4, int)
gain[2], lag[3] = .98, 3
angles = np.zeros((n, 4))
for k in range(4):
    r = np.roll(rates[:, k] * gain[k], lag[k])
    angles[:, k] = np.radians((np.cumsum(r) - r / 2) * dt)  # trapezoid: a point sample, not a right-end sum
roll_bucket = math.radians(.6)  # bucket IMU rolled 0.6 deg relative to the cabin
accel = np.tile([0, 0, 1.], (n, 4, 1)); accel[:, 3] = [0, math.sin(roll_bucket), math.cos(roll_bucket)]
accel += rng.normal(0, .0005, accel.shape)

with tempfile.TemporaryDirectory(prefix='imu-audit-') as d:
    ep = Path(d) / 'episode-1.ready'; ep.mkdir()
    with open(ep / 'samples.jsonl', 'w') as f:
        for i in range(n):
            f.write(json.dumps({'sample_ns': int(t[i] * 1e9), 'status': {
                'fk_valid': True, 'simulated_imu': False, 'imu_gyro_dps': gyro[i].tolist(),
                'imu_accel_g': accel[i].tolist(), 'angles_rad': angles[i].tolist()}}) + '\n')
    report = imu_audit.audit(imu_audit.load([d]))
    cli = subprocess.run([sys.executable, '-B', str(source / 'tools/imu_audit.py'), d, '--json', str(Path(d) / 'r.json')],
                         capture_output=True, text=True)
    assert cli.returncode == 0 and json.loads((Path(d) / 'r.json').read_text())['samples'] == n, cli.stderr

h = report['hinge_axis']
assert abs(h['arm']['tilt_deg'] - 1.5) < .1, h['arm']
assert all(h[k]['tilt_deg'] < .2 for k in ('cabin', 'boom', 'bucket')), h
lo, hi = h['arm']['tilt_ci95']; assert lo <= 1.5 <= hi + .05, h['arm']
r = report['relative_roll']
assert abs(r['bucket']['median_deg'] - .6) < .05 and abs(r['arm']['median_deg']) < .05, r
a = report['angle_rate']
assert abs(a['arm']['gain'] - .98) < .005 and abs(a['boom']['gain'] - 1) < .005, a
assert a['bucket']['lag_ms'] == 30 and a['arm']['lag_ms'] == 0, a
print('IMU audit recovers hinge tilt, relative roll, angle gain and lag')
