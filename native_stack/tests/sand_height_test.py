"""sand_height.py: the depth correction flattens synthetically domed planes; the arm model flags points
on the boom and bucket and leaves far ones."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

source = Path(sys.argv[1])
sys.path.insert(0, str(source / 'tools'))
import sand_height as sh  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    i = {'width': 640, 'height': 480, 'fx': 384.4, 'fy': 384.4, 'ppx': 325.5, 'ppy': 236.2}
    path = Path(tmp) / 'cam.depth.json'
    path.write_text(json.dumps({'depth_scale_m': .0001, 'depth_intrinsics': i}))
    rays = sh.Rays(path)
true_k = sh.depth_basis(rays.u, rays.v, rays.w, rays.h, 2) @ [-.03, 0, .2, 0, 0, .25]  # a dome, 1/m
rng = np.random.default_rng(0); frames = []
for _ in range(40):  # tilted planes 0.25-0.6 m away
    n = np.array([rng.uniform(-.4, .4), rng.uniform(-.8, -.3), -1.]); n /= np.linalg.norm(n); d = rng.uniform(.25, .45)
    z = -d / (rays.x * n[0] + rays.y * n[1] + n[2])  # ray (x, y, 1) z meets n.p + d = 0
    measured = z / (1 + z * true_k)  # inverse of z' = z / (1 - z k)
    frames.append(np.where((z > .15) & (z < .9), np.rint(measured / .0001), 0).astype('<u2').tobytes())
correction, before, after, _ = sh.fit_depth_correction(frames, rays, degree=2, every=1)
assert before[1] > 2 and after[1] < .3, (before, after)
rays.set_correction(correction)
P, _ = rays.points(frames[0], 4, 1.)
assert len(P) > 1000

status = {'angles_rad': [.49, -.88, 1.62, -2.08], 'fk_xyz_m': [.568, .304, .322],
          'fk_joint_xyz_m': [[0, 0, .07], [.0145, .0078, .1345], [.2783, .1491, .4944], [.4402, .2358, .3248]],
          'fk_link_quat_wxyz': [[.9699, 0, 0, .2434], [.8781, .1033, -.4119, .2203], [.9033, -.0886, .3532, .2267], [.7606, .151, -.6019, .1908]]}
arm = sh.ArmModel(source / 'config/arm_model.npz')
pivots = np.array(status['fk_joint_xyz_m'][1:] + [status['fk_xyz_m']])
assert arm.near(pivots, status).all(), arm.near(pivots, status)
assert not arm.near(np.array([[1.5, 0, 0], [0, 0, -.5], [.3, -.4, .05]]), status).any()
print('sand_height ok: planarity p90', before[1], '->', after[1], 'mm')
