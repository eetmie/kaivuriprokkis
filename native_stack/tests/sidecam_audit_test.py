"""sidecam_audit.py recovers joint angles and pin-to-pin lengths from a synthetic tagged machine."""
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

source = Path(sys.argv[1])
sys.path.insert(0, str(source / 'tools'))
import sidecam_audit as sc  # noqa: E402

rng = np.random.default_rng(3)
K = np.array([[640., 0, 640], [0, 640, 360], [0, 0, 1]]); DEPTH = .8
BOOM, ARM = .468, .240
P1 = np.array([-.05, .05])  # boom pin in the image plane (metres; x right, y down), cabin tags at the origin
# Tag centres in each link's frame (x along the link from its pin, y toward image-down).
LOCAL = {14: (1, (.30, -.03)), 22: (1, (.38, -.06)), 18: (2, (.06, .02)), 12: (2, (.16, .01)), 23: (3, (.09, .03))}
fps, n = 30, 900
t = np.arange(n) / fps
theta = np.c_[np.zeros(n), np.radians(-30 + 25 * np.sin(2 * math.pi * t / 30)),       # boom
              np.radians(90 + 55 * np.sin(2 * math.pi * t / 23)),                      # arm
              np.radians(-90 + 65 * np.sin(2 * math.pi * t / 17))]                     # bucket


def poses(th):
    """Pin position and image-plane angle of each link; a joint angle theta turns the image angle by -theta."""
    out = {0: (np.zeros(2), 0.)}; a = math.pi - th[1]  # boom points to the image left
    out[1] = (P1, a); p2 = P1 + BOOM * np.array([math.cos(a), math.sin(a)])
    a2 = a - th[2]; out[2] = (p2, a2); p3 = p2 + ARM * np.array([math.cos(a2), math.sin(a2)])
    out[3] = (p3, a2 - th[3])
    return out


def corners(centre, angle, size):
    c, s = math.cos(angle), math.sin(angle); R = np.array([[c, -s], [s, c]])
    sq = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * size / 2
    pts = centre + sq @ R.T
    uv = np.c_[pts[:, 0] / DEPTH * K[0, 0] + K[0, 2], pts[:, 1] / DEPTH * K[1, 1] + K[1, 2]]
    return uv + rng.normal(0, .03, uv.shape)


rows = []
for f in range(n):
    P = poses(theta[f])
    for i, (x, y) in ((16, (0., 0.)), (20, (.25, .005))):
        rows.append([f, i, 0, 90, 0, 0, *corners(np.array([x, y]), 0., .0278).ravel()])
    for i, (link, (lx, ly)) in LOCAL.items():
        o, a = P[link]; c, s = math.cos(a), math.sin(a)
        rows.append([f, i, 0, 60, 0, 0, *corners(o + np.array([c * lx - s * ly, s * lx + c * ly]), a, sc.SIZE.get(i, .0278)).ravel()])
tags = np.array(rows, float)
tags[:, 4:6] = tags[:, 6:14].reshape(-1, 4, 2).mean(1)

with tempfile.TemporaryDirectory(prefix='sidecam-') as d:
    session = Path(d) / 'session'; episode = session / 'episode-1.ready'; episode.mkdir(parents=True)
    (session / 'cam2.depth.json').write_text(json.dumps({'ir_intrinsics': {'fx': K[0, 0], 'fy': K[1, 1], 'ppx': K[0, 2], 'ppy': K[1, 2]}}))
    t0 = 1000.
    with open(episode / 'samples.jsonl', 'w') as out:
        for j in range(n * 100 // fps):  # 100 Hz samples; IMU angle at t + CAMERA_LAG_S matches the frame at t
            ts = j / 100; f = min(int(ts * fps), n - 1)
            ang = [float(np.interp(ts + sc.CAMERA_LAG_S, t, theta[:, k])) for k in range(4)]
            out.write(json.dumps({'frame_index': f, 'status': {'imu_host_monotonic_ns': int((t0 + ts) * 1e9), 'angles_rad': ang},
                                  'cam2': {'sensor_timestamp_us': f / fps * 1e6, 'host_monotonic_ns': int((t0 + f / fps) * 1e9),
                                           'actual_exposure_us': 0}}) + '\n')
    np.save(Path(d) / 'tags.npy', tags)
    data = sc.load(episode, Path(d) / 'tags.npy'); sc.geometry(data)
    joints = sc.compare(data)
    piv = sc.pivots(data)

# The hinge axis comes from a narrow boom sweep; its small tilt bends angles slightly with pose
# (the same effect limits gains to about +-2% on the machine).
assert math.degrees(math.acos(abs(data['n'][2]))) < 2, data['n']
for name, r in joints.items():
    assert abs(r['gain'] - 1) < .01 and r['rms_deg'] < .25, (name, r)
assert abs(piv['boom_mm'] - 1e3 * BOOM) < 1.5 and abs(piv['arm_mm'] - 1e3 * ARM) < 1.5, piv
assert abs(piv['arm_pinline_minus_imu_deg']) < .2, piv
print(f'side-camera audit recovers joints (rms {max(r["rms_deg"] for r in joints.values()):.3f} deg) '
      f'and lengths (boom {piv["boom_mm"]:.1f}, arm {piv["arm_mm"]:.1f} mm)')
