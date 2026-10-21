"""fusion_replay.py builds the vendored Fusion AHRS, holds attitude when still, converges on a small
tilt at its gain, ignores a large acceleration, and extracts joint angles like core.cpp."""
import math
from pathlib import Path
import shutil
import sys

import numpy as np

source = Path(sys.argv[1])
sys.path.insert(0, str(source / 'tools'))
import fusion_replay as fr  # noqa: E402

if not shutil.which('cc'):
    print('no C compiler; fusion replay test skipped')
    sys.exit(0)
exe = fr.binary()
n = 2000  # 20 s at 100 Hz
still = dict(dev=np.arange(n) * .01, q=np.tile([1., 0, 0, 0], (n, 4, 1)), g=np.zeros((n, 4, 3)), a=np.tile([0, 0, 1.], (n, 4, 1)))
q = fr.replay(still, 0, .5, 5, 2, exe)
assert np.abs(q[:, 1:]).max() < 1e-5, q[-1]  # attitude held; Fusion's fast inverse sqrt leaves |q| about 1.0001

# 3 deg of accelerometer tilt about Y, starting level: gain 0.5 is a 2 s time constant.
tilted = dict(still); tilted['a'] = np.tile([-math.sin(math.radians(3)), 0, math.cos(math.radians(3))], (n, 4, 1))
pitch = np.degrees(fr.gravity_pitch(fr.replay(tilted, 0, .5, 5, 2, exe)))
assert abs(abs(pitch[-1]) - 3) < .05, pitch[-1]
assert .4 < abs(pitch[200]) / 3 < .8, pitch[200]  # about 1 - exp(-1) after 2 s

# 20 deg of apparent tilt for 1 s (a shake) is rejected at 5 deg and accepted at 30 deg.
shake = dict(still); a = np.tile([0, 0, 1.], (n, 4, 1)); a[500:600] = [-math.sin(math.radians(20)), 0, math.cos(math.radians(20))]; shake['a'] = a
for rejection, moved in ((5, False), (30, True)):
    p = np.degrees(fr.gravity_pitch(fr.replay(shake, 0, .5, rejection, 2, exe)))
    assert (abs(p[599]) > 2) == moved, (rejection, p[599])

# Joint extraction: boom IMU pitched 30 deg, arm 30 + 40, offsets identity except the arm (+1 deg about Y).
def about_y(deg):
    return [math.cos(math.radians(deg) / 2), 0, math.sin(math.radians(deg) / 2), 0]
quats = np.array([[about_y(0), about_y(30), about_y(70), about_y(70)]])
offsets = np.array([about_y(0), about_y(0), about_y(1), about_y(0)])
ang = np.degrees(fr.joint_angles(quats, offsets))[0]
assert np.allclose(ang, [30, 39, 1], atol=1e-6), ang  # arm corrected to 69: arm 69-30, bucket 70-69
print('Fusion replay holds still attitude, converges on tilt at its gain, rejects shakes, and extracts joint angles')
