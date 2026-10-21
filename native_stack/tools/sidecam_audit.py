"""Side-camera FK/IMU audit from AprilTag detections (sidecam_tags.py) and recorded controller status.

    sidecam_audit.py joints  EPISODE [--tags F] [--axis-from EPISODE2]   camera vs IMU joint angles
    sidecam_audit.py pivots  EPISODE [--tags F] [--axis-from EPISODE2]   pin positions, link lengths
    sidecam_audit.py slew    EPISODE [--tags F]                          camera vs IMU slew
    sidecam_audit.py touches EPISODE [EPISODE ...]                       FK tip height at ground touches

Calibration-only setup (2026-10-21; tags are removed for normal recording): the former cabin D435i on
a tripod beside the machine as cam2, recorded with `record_native.py --resolution 1280x720 --depth cam2`.
tagStandard41h12, border edge 27.8 mm (50 mm print) or 22.2 mm (40 mm). Single-tag PnP orientation is
ambiguous at 20-30 px, so tag corners are intersected with planes perpendicular to the hinge axis (one
offset per tag) and link directions come from tag-to-tag baselines (TAGS below). The hinge axis is the
least-variance direction of the boom tags' sweep; reuse another episode's (--axis-from) when the boom
barely moved. Results are relative to the camera, so the camera must stay put (a wall tag shows it).
`touches` needs no camera: still periods with the tip near the episode's lowest point.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Tag layout on the machine. Each link direction is the baseline between two tags on that link
# (first -> second); the bucket uses tag 23's edge direction.
TAGS = {'cabin': (16, 20), 'boom': (14, 22), 'arm': (18, 12), 'bucket': (23,), 'wall': (17, 19), 'track': (21,)}
SIZE = {12: .0222, 14: .0222}  # border edge, metres; others .0278
CAMERA_LAG_S = .012            # IMU sample time relative to mid-exposure (fitted 6-18 ms)


def load(episode, tags=None):
    episode = Path(episode).resolve()
    k = json.loads((episode.parent / 'cam2.depth.json').read_text())['ir_intrinsics']
    rows = [json.loads(l) for l in open(episode / 'samples.jsonl')]
    st = [r['status'] for r in rows]
    d = dict(name=f'{episode.parent.name}/{episode.name}',
             K=np.array([[k['fx'], 0, k['ppx']], [0, k['fy'], k['ppy']], [0, 0, 1]]),
             t=np.array([s['imu_host_monotonic_ns'] for s in st]) * 1e-9,
             ang=np.array([s['angles_rad'] for s in st]), status=st)
    ft = {}
    for r in rows:
        c = r.get('cam2')
        if c:
            ft[r['frame_index']] = (c['sensor_timestamp_us'] * 1e-6, c['host_monotonic_ns'] * 1e-9, c['actual_exposure_us'] * 1e-6)
    if ft:
        frames = np.array(sorted(ft)); sens, host, expo = np.array([ft[f] for f in frames]).T
        # Device clock -> host clock: rate from a fit, offset from the least-delayed arrival.
        a = np.polyfit(sens, host, 1)[0]; b = np.min(host - a * sens)
        d['frames'], d['T'] = frames, a * (sens + expo / 2) + b
    if tags is not None or ft:
        path = Path(tags) if tags else episode.parent.with_name(episode.parent.name + '.sidecam') / (episode.name.removesuffix('.ready') + '.tags.npy')
        if path.exists():
            d['tags'] = np.load(path)
    return d


def geometry(d, axis=None):
    """Rectified tag centres C[id] (n_frames, 2) and edge angles A[id] in the plane perpendicular to the hinge axis."""
    import cv2
    tags = d['tags']; ids = tags[:, 1].astype(int); px = tags[:, 6:14].reshape(-1, 4, 2); K = d['K']
    cen = np.zeros((len(tags), 3))
    for j in range(len(tags)):
        s = SIZE.get(ids[j], .0278); obj = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * s / 2
        cen[j] = cv2.solvePnP(obj, px[j], K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)[2].ravel()
    sweep = np.concatenate([cen[ids == i] - cen[ids == i].mean(0) for i in TAGS['boom'] if (ids == i).any()])
    w, V = np.linalg.eigh(np.cov(sweep.T)); n = V[:, 0] * np.sign(V[2, 0])
    if axis is not None:
        n = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.array([1., 0, 0]) - n[0] * n; u /= np.linalg.norm(u); v = np.cross(n, u)
    plane = {i: np.median(cen[ids == i] @ n) for i in np.unique(ids)}
    ray = np.einsum('ij,nkj->nki', np.linalg.inv(K), np.concatenate([px, np.ones((len(px), 4, 1))], 2))
    X = ray * (np.array([plane[i] for i in ids])[:, None] / (ray @ n))[..., None]
    q = np.stack([X @ u, X @ v], -1)
    index = {f: j for j, f in enumerate(d['frames'])}; nf = len(d['frames']); C, A = {}, {}
    for i in np.unique(ids):
        m = ids == i; r = [index[f] for f in tags[m, 0].astype(int)]
        c = np.full((nf, 2), np.nan); a = np.full(nf, np.nan)
        c[r] = q[m].mean(1); e = (q[m, 1] - q[m, 0]) + (q[m, 2] - q[m, 3]); a[r] = np.arctan2(e[:, 1], e[:, 0])
        C[int(i)], A[int(i)] = c, a
    d.update(n=n, u=u, v=v, C=C, A=A, plane=plane, sweep_mm=1e3 * np.sqrt(w[::-1]))


def direction(d, link):
    a, b = TAGS[link]; x = d['C'][b] - d['C'][a]
    return np.arctan2(x[:, 1], x[:, 0])


def unwrap_nan(x):
    m = np.isfinite(x); y = np.full_like(x, np.nan); y[m] = np.unwrap(x[m]); return y


def camera_joints(d):
    """Camera joint angles (image v points down, so signs flip to match the controller)."""
    cab, boom, arm = direction(d, 'cabin'), direction(d, 'boom'), direction(d, 'arm')
    bucket = d['A'][TAGS['bucket'][0]]
    return {j: -unwrap_nan(np.angle(np.exp(1j * x))) for j, x in (('boom', boom - cab), ('arm', arm - boom), ('bucket', bucket - arm))}


def compare(d, angles=None, mask=None):
    """Camera minus IMU per joint after a linear fit; the 0.5 s average removes camera noise."""
    angles = d['ang'] if angles is None else angles; out = {}
    for j, (name, cam) in enumerate(camera_joints(d).items(), start=1):
        m = np.isfinite(cam) & (np.ones(len(cam), bool) if mask is None else mask)
        if m.sum() < 30:
            continue
        x = np.interp(d['T'] - CAMERA_LAG_S, d['t'], angles[:, j]); A = np.c_[x[m], np.ones(m.sum())]
        (g, o), *_ = np.linalg.lstsq(A, cam[m], rcond=None)
        e = np.zeros(len(cam)); e[m] = cam[m] - A @ [g, o]
        smooth = (np.convolve(e, np.ones(15), 'same') / np.maximum(np.convolve(m, np.ones(15), 'same'), 1))[m]
        out[name] = dict(n=int(m.sum()), span_deg=np.degrees(np.ptp(x[m])), gain=g, offset_deg=np.degrees(np.median(cam[m] - x[m])),
                         rms_deg=np.degrees(np.sqrt(np.mean(smooth ** 2))), p99_deg=np.degrees(np.percentile(np.abs(smooth), 99)))
    return out


def circle(p):
    p = p[np.isfinite(p).all(1)]; A = np.c_[2 * p, np.ones(len(p))]
    c, *_ = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None); centre = c[:2]
    for _ in range(5):
        dd = p - centre; r = np.linalg.norm(dd, axis=1); J = np.c_[-dd / r[:, None], -np.ones(len(p))]
        step, *_ = np.linalg.lstsq(J, -(r - r.mean()), rcond=None); centre = centre + step[:2]
    r = np.linalg.norm(p - centre, axis=1)
    return centre, r.mean(), np.std(r)


def to_frame(origin, angle, p):
    c, s = np.cos(angle), np.sin(angle); x = p - origin
    return np.c_[c * x[:, 0] + s * x[:, 1], -s * x[:, 0] + c * x[:, 1]]


def from_frame(origin, angle, p):
    c, s = np.cos(angle), np.sin(angle)
    return origin + np.c_[c * p[0] - s * p[1], s * p[0] + c * p[1]]


def pivots(d):
    """Each tag circles its link's pin as seen from the parent link's tags. A circle fit needs a wide
    sweep of that joint: under about 90 degrees the radius and centre trade off."""
    cab = direction(d, 'cabin'); origin = d['C'][TAGS['cabin'][0]]
    fits = {'boom pin (cabin frame)': [circle(to_frame(origin, cab, d['C'][i])) for i in TAGS['boom']]}
    P1 = np.mean([f[0] for f in fits['boom pin (cabin frame)']], 0); P1w = from_frame(origin, cab, P1)
    boom = direction(d, 'boom')
    fits['arm pin (boom frame)'] = [circle(to_frame(P1w, boom, d['C'][i])) for i in TAGS['arm']]
    P2 = np.mean([f[0] for f in fits['arm pin (boom frame)']], 0); P2w = from_frame(P1w, boom, P2)
    arm = direction(d, 'arm')
    fits['bucket pin (arm frame)'] = [circle(to_frame(P2w, arm, d['C'][TAGS['bucket'][0]]))]
    P3 = fits['bucket pin (arm frame)'][0][0]; P3w = from_frame(P2w, arm, P3)
    boom_line = np.arctan2(*(P2w - P1w).T[::-1]); arm_line = np.arctan2(*(P3w - P2w).T[::-1])
    cam_arm = -np.angle(np.exp(1j * (arm_line - boom_line)))
    imu_arm = np.interp(d['T'] - CAMERA_LAG_S, d['t'], d['ang'][:, 2]); m = np.isfinite(cam_arm)
    return dict(fits=fits, boom_mm=1e3 * np.linalg.norm(P2), arm_mm=1e3 * np.linalg.norm(P3),
                arm_pinline_minus_imu_deg=np.degrees(np.median(np.angle(np.exp(1j * (cam_arm[m] - imu_arm[m]))))))


def slew(d):
    """Rigid rotation of all upper-structure tag centres (camera as reference) vs the IMU slew."""
    import cv2
    from scipy.spatial.transform import Rotation
    upper = TAGS['cabin'] + TAGS['boom'] + TAGS['arm'] + TAGS['bucket']; tags = d['tags']; K = d['K']; cen = {}
    for row in tags:
        f, i = int(row[0]), int(row[1])
        if i in upper:
            s = SIZE.get(i, .0278); obj = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * s / 2
            cen[f, i] = cv2.solvePnP(obj, row[6:14].reshape(4, 2), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)[2].ravel()
    frames = sorted({f for f, _ in cen}); seen = {f: [i for i in upper if (f, i) in cen] for f in frames}
    ref = max(frames, key=lambda f: (len(seen[f]), -f)); fr, rv = [], []
    for f in frames:
        common = [i for i in seen[f] if i in seen[ref]]
        if len(common) < 3:
            continue
        P = np.array([cen[ref, i] for i in common]); Q = np.array([cen[f, i] for i in common])
        H = (P - P.mean(0)).T @ (Q - Q.mean(0)); U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        fr.append(f); rv.append(Rotation.from_matrix(R).as_rotvec())
    rv = np.array(rv); big = np.linalg.norm(rv, axis=1) > np.radians(5)
    if big.sum() < 10:
        raise SystemExit('the slew moved less than 5 degrees')
    ax = rv[big] / np.linalg.norm(rv[big], axis=1)[:, None]; ax = np.mean(ax * np.sign(ax @ ax[0])[:, None], 0); ax /= np.linalg.norm(ax)
    cam = np.degrees(rv @ ax); index = {f: j for j, f in enumerate(d['frames'])}; T = d['T'][[index[f] for f in fr]]
    imu = np.degrees(np.unwrap(d['ang'][:, 0])); best = None
    for lag in np.arange(-.1, .1001, .002):
        x = np.interp(T - lag, d['t'], imu); A = np.c_[x, np.ones(len(x))]; c, *_ = np.linalg.lstsq(A, cam, rcond=None); e = cam - A @ c
        if best is None or e @ e < best[0]:
            best = (e @ e, lag, c, e, x)
    _, lag, (g, _), e, x = best
    return dict(frames=len(fr), camera_span_deg=np.ptp(cam), imu_span_deg=np.ptp(x), gain=abs(g), lag_ms=1e3 * lag,
                rms_deg=np.sqrt(np.mean(e ** 2)), p95_deg=np.percentile(np.abs(e), 95))


def touches(d, max_above_mm=15., min_still_s=.8):
    """Still periods (all joint rates < 1 deg/s) with the level-frame tip within max_above_mm of the
    episode's lowest 1%: one row per touch (t, x mm, z mm, bucket level pitch deg, cabin pitch deg)."""
    st = d['status']; t = d['t'] - d['t'][0]; ang = d['ang']
    tip = np.array([s['fk_xyz_m'] for s in st]) * 1e3; cp = np.degrees([s['cabin_pitch_rad'] for s in st])
    bucket = np.degrees([s['bucket_pitch_rad'] for s in st]) + cp
    rate = np.degrees(np.abs(np.gradient(np.unwrap(ang, axis=0), t, axis=0))).max(1)
    still = np.convolve(rate < 1, np.ones(15) / 15, 'same') > .99
    edges = np.diff(np.r_[0, still.astype(int), 0]); floor = np.percentile(tip[:, 2], 1); rows = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if t[b - 1] - t[a] < min_still_s:
            continue
        s = slice(a + 10, b - 10); z = np.median(tip[s, 2])
        if z < floor + max_above_mm:
            rows.append((t[a], np.median(tip[s, 0]), z, np.median(bucket[s]), np.median(cp[s])))
    return np.array(rows).reshape(-1, 5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('joints', 'pivots', 'slew', 'touches'))
    p.add_argument('episodes', type=Path, nargs='+')
    p.add_argument('--tags', type=Path, help='detections from sidecam_tags.py (default: SESSION.sidecam/EPISODE.tags.npy)')
    p.add_argument('--axis-from', type=Path, help='episode whose boom sweep defines the hinge axis')
    a = p.parse_args()
    axis = None
    if a.axis_from:
        ref = load(a.axis_from); geometry(ref); axis = ref['n']
    for ep in a.episodes:
        d = load(ep, a.tags if len(a.episodes) == 1 else None)
        if a.command == 'touches':
            r = touches(d)
            print(f'{d["name"]}: {len(r)} touches')
            if len(r):
                fit = np.polyfit(r[:, 1], r[:, 2], 1) if len(r) > 2 else (0, r[:, 2].mean())
                print(f'  tip z mean {r[:, 2].mean():+.2f} mm, std {r[:, 2].std():.2f}, range {r[:, 2].min():+.1f}..{r[:, 2].max():+.1f}; '
                      f'reach {r[:, 1].min():.0f}..{r[:, 1].max():.0f} mm, slope {100 * fit[0]:+.2f} mm/100 mm; bucket {r[:, 3].min():+.0f}..{r[:, 3].max():+.0f} deg')
                for row in r:
                    print(f'    t {row[0]:6.1f} s  x {row[1]:6.1f}  z {row[2]:+6.1f} mm  bucket {row[3]:+6.1f}  cabin pitch {row[4]:+5.2f} deg')
            continue
        if 'tags' not in d:
            raise SystemExit(f'{d["name"]}: no tag detections; run sidecam_tags.py first or pass --tags')
        if a.command == 'slew':
            r = slew(d)
            print(f'{d["name"]}: {r["frames"]} frames, camera slew span {r["camera_span_deg"]:.1f} deg (IMU {r["imu_span_deg"]:.1f}); gain {r["gain"]:.4f}, '
                  f'lag {r["lag_ms"]:+.0f} ms, residual rms {r["rms_deg"]:.2f} deg, p95 {r["p95_deg"]:.2f}')
            continue
        geometry(d, axis)
        print(f'{d["name"]}: {len(d["frames"])} frames, hinge axis {np.round(d["n"], 4)}, boom sweep {d["sweep_mm"][0]:.0f} x {d["sweep_mm"][1]:.0f} mm')
        wall = d['C'].get(TAGS['wall'][0])
        if wall is not None and np.isfinite(wall).all(1).sum() > 10:
            w = wall[np.isfinite(wall).all(1)]
            print(f'  wall tag {TAGS["wall"][0]} spread p1..p99 {1e3 * np.ptp(np.percentile(w[:, 0], [1, 99])):.2f} x {1e3 * np.ptp(np.percentile(w[:, 1], [1, 99])):.2f} mm (camera motion)')
        if a.command == 'joints':
            for name, r in compare(d).items():
                print(f'  {name:6s} n={r["n"]:5d} span {r["span_deg"]:4.0f} deg  gain {r["gain"]:+.4f}  camera-IMU rms {r["rms_deg"]:.2f} deg, p99 {r["p99_deg"]:.2f}  '
                      f'offset {r["offset_deg"]:+.2f} deg (includes tag placement)')
        else:
            r = pivots(d)
            for k, fits in r['fits'].items():
                print(f'  {k}: ' + '; '.join(f'({1e3 * c[0]:.1f}, {1e3 * c[1]:.1f}) mm, radius {1e3 * rad:.1f}, rms {1e3 * s:.2f}' for c, rad, s in fits))
            print(f'  boom pin-to-pin {r["boom_mm"]:.1f} mm, arm pin-to-pin {r["arm_mm"]:.1f} mm; '
                  f'arm pin-line angle minus IMU arm angle {r["arm_pinline_minus_imu_deg"]:+.2f} deg (pose dependent; a plumb line is the absolute reference)')


if __name__ == '__main__':
    main()
