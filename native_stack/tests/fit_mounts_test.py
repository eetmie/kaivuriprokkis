"""fit_mounts.py: the ground plane through tip touches, and the track-end tilt matching core.cpp."""
import math
from pathlib import Path
import sys

import numpy as np

source = Path(sys.argv[1])
sys.path.insert(0, str(source / 'tools'))
import fit_mounts as fm  # noqa: E402

rng = np.random.default_rng(0)
xy = rng.uniform(-.5, .8, (12, 2))
n, d, r = fm.ground(np.c_[xy, .01 * xy[:, 0] - .002 * xy[:, 1] + .003])
assert np.allclose(n, np.array([-.01, .002, 1]) / np.linalg.norm([-.01, .002, 1])) and abs(d + .003 * n[2]) < 1e-12
assert np.abs(r).max() < 1e-12

# Front up by 5 deg (pitch about +y lifts +x): the rear track end stays down, so the body rises.
a = math.radians(5)
st = {'cabin_tilt_quat_wxyz': [math.cos(a / 2), 0, -math.sin(a / 2), 0],
      'fk_joint_xyz_m': [[0, 0, .1045]] * 4, 'fk_xyz_m': [.6, 0, 0]}
assert fm.sh.quat_matrix(st['cabin_tilt_quat_wxyz'])[2, 0] > 0
up = fm.lifted(st, .16)
lift = np.array(up['fk_xyz_m']) - st['fk_xyz_m']
assert np.allclose(lift, [-.16 + .16 * math.cos(a), 0, .16 * math.sin(a)]), lift
assert np.allclose(np.array(up['fk_joint_xyz_m'][0]) - st['fk_joint_xyz_m'][0], lift)
print('fit_mounts ok')
