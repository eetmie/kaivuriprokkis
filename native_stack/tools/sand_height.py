"""Sand height from recorded RealSense depth: camera-on-link calibration and Rerun view.

Each camera rides on one FK link. A depth point p (camera frame) lands in the FK base frame as
    p_base = R_BL(t) (R_X p + t_X) + t_BL(t)
with the link pose from that frame's recorded FK and the fixed mount X = (R_X, t_X). Height is
the signed distance above the flat-sand plane n . p_base + d = 0, so it stays level as the
cabin slews and the arm moves. Positions on the plane are shared by both cameras, which fuses
them into one top-down height map.

  calibrate SESSION --flat EP [--sweep EP:START:END] [--overlap EP:START:END] [-o FILE]
      EP is an episode index in the session. --flat is a take of flattened sand (arm may move;
      more slew/arm variety is better). --sweep is a static stretch where the cabin slews over
      features (finished holes); it fixes the cabin camera's offset from the slew axis, which
      flat sand cannot show. --overlap is a static stretch both cameras see; it fixes the
      cabin camera's rotation about the slew axis and the wrist camera's sideways offset.
  view SESSION [--calibration FILE] [--episode EP] [--save FILE.rrd]
      Per-camera level height images, a fused top-down height map, the change since the take
      started, litres dug/piled and the dug hole's bottom below the flat sand.

Calibration needs SciPy; viewing needs only NumPy and rerun-sdk 0.26.2. Camera mounts are
fixed per machine, so one calibration serves later sessions (pass --calibration).
Slew angle comes from a gyro-only IMU and can drift: the viewer re-registers the cabin camera
against the calibration's reference map every frame and applies that turn to both cameras.
Chassis tilt (digging forces tip the machine) comes from the base IMU and levels all FK poses;
recordings from a controller with robot.level_frame.cabin_tilt already carry it in their FK.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_episodes import committed_episodes, depth_frames, frame_rows  # noqa: E402

LINKS = ('slew', 'boom', 'arm', 'bucket')
CELL = .005  # fused map resolution, m


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotvec_matrix(r):
    a = np.linalg.norm(r)
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(r) / a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Chassis tilt: the FK assumes a level base, but digging forces tip the machine about a track
# edge (2 degrees and a 16 mm cabin lift seen when the bucket hit the container bottom). The
# 'base' IMU rides the cabin (its yaw is the slew angle), so its AHRS gravity is in cabin axes;
# the turn back onto the reference (median over the flat take), turned by the slew into base
# axes and applied about the calibrated pivot (a chassis point), levels every FK pose.
GRAVITY_REFERENCE = None
TILT_YAW = 0.          # cabin IMU axes relative to the upper-carriage axes, about vertical (rad)
TILT_PIVOT = np.zeros(3)  # a point on the tipping axis, base frame
TILT_DEADBAND = math.radians(.75)  # slewing rocks the base IMU by about 0.5 degrees; only real tipping counts


def load_tilt(cal):
    """Take the chassis tilt correction from a calibration that has one."""
    global GRAVITY_REFERENCE, TILT_YAW, TILT_PIVOT, TILT_DEADBAND
    if cal.get('base_gravity_reference'):
        GRAVITY_REFERENCE = np.asarray(cal['base_gravity_reference'])
        TILT_YAW = math.radians(cal.get('tilt_imu_yaw_deg', 0.))
        TILT_PIVOT = np.asarray(cal.get('tilt_pivot_base_m', [0, 0, 0.]))
        TILT_DEADBAND = math.radians(cal.get('tilt_deadband_deg', math.degrees(TILT_DEADBAND)))


def base_gravity(status):
    return quat_matrix(status['imu_corrected_quat_wxyz'][0]).T @ [0, 0, 1.]


def tilt_correction(status, yaw=None, pivot=None):
    """(T, u): level a base-frame point p as T p + u. Recordings whose controller already
    levels its base frame (fk_cabin_tilt) need nothing here."""
    if GRAVITY_REFERENCE is None or status.get('fk_cabin_tilt') or not status.get('imu_corrected_quat_wxyz'):
        return np.eye(3), np.zeros(3)
    yaw = TILT_YAW if yaw is None else yaw; pivot = TILT_PIVOT if pivot is None else pivot
    g = base_gravity(status); v = np.cross(g, GRAVITY_REFERENCE); s = np.linalg.norm(v)
    angle = math.asin(min(s, 1.)) - TILT_DEADBAND
    T = np.eye(3) if angle <= 0 else rotvec_matrix(v / s * angle)
    a = yaw + status['angles_rad'][0]  # cabin IMU axes -> base axes
    T = rz(a) @ T @ rz(-a)
    return T, pivot - T @ pivot


def link_pose(status, link):
    T, u = tilt_correction(status)
    return T @ quat_matrix(status['fk_link_quat_wxyz'][link]), T @ np.asarray(status['fk_joint_xyz_m'][link], float) + u


def fk_chain(status, first=0):
    """FK pivots from `first` plus the tool tip, levelled like the camera points."""
    T, u = tilt_correction(status)
    return np.asarray([*status['fk_joint_xyz_m'][first:], status['fk_xyz_m']]) @ T.T + u


def level_rotation(n):
    """Rotation taking plane normal n to +Z by the shortest arc."""
    v = np.cross(n, [0, 0, 1]); s = np.linalg.norm(v)
    return np.eye(3) if s < 1e-12 else rotvec_matrix(v / s * math.asin(min(s, 1.)))


class Rays:
    """Per stored pixel ray in SDK camera coordinates (stored pixels may be turned 180 degrees).
    From a recorded session's camN.depth.json (session, key) or any camera depth JSON (path)."""
    def __init__(self, session, key=None):
        self.key, path = key, session if key is None else session / f'{key}.depth.json'
        c = json.loads(Path(path).read_text()); i = c['depth_intrinsics']
        self.h, self.w, self.scale = i['height'], i['width'], c['depth_scale_m']
        v, u = np.mgrid[0:self.h, 0:self.w].astype(float)
        if c.get('rotation_deg', 0) == 180:
            u, v = self.w - 1 - u, self.h - 1 - v
        self.x, self.y = (u - i['ppx']) / i['fx'], (v - i['ppy']) / i['fy']

    def points(self, data, stride, max_range):
        """Camera-frame points of valid pixels, and their flat pixel indices on the strided grid."""
        z = np.frombuffer(data, '<u2').reshape(self.h, self.w)[::stride, ::stride] * self.scale
        P = np.stack([self.x[::stride, ::stride] * z, self.y[::stride, ::stride] * z, z], -1).reshape(-1, 3)
        keep = np.flatnonzero((z.reshape(-1) > 0) & (np.einsum('ij,ij->i', P, P) < max_range ** 2))
        return P[keep], keep


class Mount:
    """One camera's calibrated mount and sand plane."""
    def __init__(self, link, R, t, n, d, cabin_shift=(0., 0.)):
        self.link, self.R, self.t = link, np.asarray(R, float), np.asarray(t, float)
        self.cabin_shift = np.asarray(cabin_shift, float)
        self.n, self.d = np.asarray(n, float), float(d)
        self.level = level_rotation(self.n)

    def base(self, P, status):
        R, t = link_pose(status, self.link)
        Pb = (P @ self.R.T + self.t) @ R.T + t
        if self.cabin_shift.any():  # residual chain mismatch, fixed in the slewing upper frame
            Pb[:, :2] += rz(status['angles_rad'][0])[:2, :2] @ self.cabin_shift
        return Pb

    def world(self, Pb):
        """Level frame: x, y along the sand plane, z = height above it."""
        W = Pb @ self.level.T; W[:, 2] = Pb @ self.n + self.d
        return W

    def to_json(self):
        return {'link': self.link, 'link_name': LINKS[self.link], 'rotation_camera_to_link': self.R.tolist(),
                'translation_camera_to_link_m': self.t.tolist(), 'sand_normal_base': self.n.tolist(),
                'sand_offset_m': self.d, 'cabin_shift_xy_m': self.cabin_shift.tolist()}

    @classmethod
    def from_json(cls, c):
        return cls(c['link'], c['rotation_camera_to_link'], c['translation_camera_to_link_m'],
                   c['sand_normal_base'], c['sand_offset_m'], c.get('cabin_shift_xy_m', (0., 0.)))


def episode_path(session, index):
    episodes = committed_episodes(session)
    if not 0 <= index < len(episodes):
        raise SystemExit(f'{session}: no episode {index} ({len(episodes)} saved)')
    return episodes[index]


def load(episode, key, wanted):
    wanted, rows, out = set(wanted), list(frame_rows(episode)), {}
    for i, data in enumerate(depth_frames(episode / f'{key}.depth.zst')):
        if i in wanted:
            out[i] = (data, rows[i]['status'])
    return out


def raster(xy, values, lo, shape):
    ij = np.floor((xy - lo) / CELL).astype(int)
    k = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    flat = ij[k, 0] * shape[1] + ij[k, 1]; size = shape[0] * shape[1]
    total, count = np.bincount(flat, values[k], size), np.bincount(flat, None, size)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (total / count).reshape(shape), count.reshape(shape)


def lookup(grid, lo, xy):
    ij = np.floor((xy - lo) / CELL).astype(int)
    k = (ij >= 0).all(1) & (ij[:, 0] < grid.shape[0]) & (ij[:, 1] < grid.shape[1])
    g = np.full(len(xy), np.nan); g[k] = grid[ij[k, 0], ij[k, 1]]
    return g


def height_map(W):
    """Mean height of level points W on a grid around them with a 5 cm margin: (float32 map, lower corner)."""
    lo = np.floor(W[:, :2].min(0) / CELL) * CELL - .05
    shape = tuple(np.ceil((W[:, :2].max(0) + .05 - lo) / CELL).astype(int))
    return raster(W[:, :2], W[:, 2], lo, shape)[0].astype(np.float32), lo


def near_flat(W, band=.08):
    """Level points within band of the flat-sand plane."""
    return W[np.abs(W[:, 2]) < band]


# ---------------------------------------------------------------- calibration
def plane_ransac(P, rng, iters=150, tol=.006):
    best = (0, None, 0.)
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm; d = -n @ s[0]; k = int((np.abs(P @ n + d) < tol).sum())
        if k > best[0]:
            best = (k, n, d)
    k, n, d = best
    return (n, d, k / len(P)) if d >= 0 else (-n, -d, k / len(P))  # normal toward the camera above


def flat_samples(episode, rays, every=3, per_frame=150, seed=0):
    """Sand points (camera frame) from each sampled frame of the flat take, with the frame status."""
    rng = np.random.default_rng(seed); out = []
    rows = list(frame_rows(episode))
    for i, data in enumerate(depth_frames(episode / f'{rays.key}.depth.zst')):
        if i % every or not rows[i]['status'].get('fk_valid'):
            continue
        P, _ = rays.points(data, 8, 1.)
        if len(P) < 300:
            continue
        n, d, frac = plane_ransac(P, rng)
        if frac < .4:
            continue
        inl = P[np.abs(P @ n + d) < .015]
        out.append((i, rows[i]['status'], inl[rng.choice(len(inl), min(per_frame, len(inl)), replace=False)], n, d))
    return out


def fit_mount(samples, link, start=None):
    """Robust fit of X and the base-frame sand plane to flat-sand points of many link poses.
    With start (a Mount), keep its rotation and in-plane offset (what the sweep set; flat sand
    cannot see them) and refit only its height and the plane."""
    from scipy.optimize import least_squares
    poses = [link_pose(s[1], link) for s in samples]
    up = np.array([0, 0, 1.])
    # Initial rotation: camera plane normals should map to (roughly) up in the base (Kabsch).
    A = np.array([s[3] for s in samples]); B = np.array([R.T @ up for R, _ in poses])
    U, _, Vt = np.linalg.svd(A.T @ B); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    RX = (U @ D @ Vt).T
    # Initial offset: d_cam = up . (R t_X + t_link) + d, linear in (t_X, d).
    M = np.array([[*(R.T @ up), 1] for R, _ in poses]); y = np.array([s[4] - t[2] for s, (_, t) in zip(samples, poses)])
    sol, *_ = np.linalg.lstsq(M, y, rcond=1e-6)
    tX = sol[:3]
    from scipy.spatial.transform import Rotation
    from_vec = lambda v: Rotation.from_rotvec(v).as_matrix()  # noqa: E731
    if start is not None:
        RX, tX = start.R, start.t
    r0 = Rotation.from_matrix(RX).as_rotvec()
    x0 = np.r_[r0, tX, 0, 0, sol[3]]
    if start is not None:  # plane as a tilt (rotation vector about x, y) of +Z, and offset
        tilt = np.cross([0, 0, 1], start.n); sn = np.linalg.norm(tilt)
        x0[6:8] = (tilt / sn * math.asin(min(sn, 1.)))[:2] if sn > 1e-12 else 0
        x0[8] = start.d
    free = np.ones(9, bool)
    if start is not None:  # refit height and plane only; rotation and in-plane offset are set
        free[:5] = False

    def unpack(x):
        n = rotvec_matrix([x[6], x[7], 0]) @ up
        return from_vec(x[:3]), x[3:6], n, x[8]

    def residual(v):
        x = x0.copy(); x[free] = v
        RXv, tXv, n, d = unpack(x)
        r = [(s[2] @ RXv.T + tXv) @ R.T @ n + t @ n + d for s, (R, t) in zip(samples, poses)]
        # Directions the data cannot see stay at the initial guess (weak prior).
        return np.concatenate(r + [1e-3 * (v - x0[free])])

    out = least_squares(residual, x0[free], loss='soft_l1', f_scale=.004, x_scale='jac')
    x = x0.copy(); x[free] = out.x
    RXv, tXv, n, d = unpack(x)
    r = residual(out.x)[:-free.sum()]
    return Mount(link, RXv, tXv, n, d), r


def frame_set(episode, start, end, key='cam2'):
    rows = list(frame_rows(episode))[start:end]
    slew = np.degrees([r['status']['angles_rad'][0] for r in rows])
    mode = np.median(slew)
    ref = [start + i for i in range(len(rows)) if abs(slew[i] - mode) < 2][::5]
    off = [start + i for i in range(len(rows)) if abs(slew[i] - mode) > 8][::4]
    return ref, off, mode


def correlation(W, G, lo, min_points=2000):
    """Zero-mean correlation of point heights with the map heights under them (None: too little overlap)."""
    g = lookup(G, lo, W[:, :2]); ok = np.isfinite(g)
    if ok.sum() < min_points:
        return None
    x, y = W[ok, 2] - W[ok, 2].mean(), g[ok] - g[ok].mean()
    return float(x @ y / math.sqrt(max((x @ x) * (y @ y), 1e-18)))


def mean_correlation(sets, G, lo):
    c = [v for v in (correlation(W, G, lo) for W in sets) if v is not None]
    return float(np.mean(c)) if len(c) > len(sets) // 2 else -1.


def search(score, x0, steps):
    """Coordinate grid refinement maximising score; steps = [(per-axis step, half-width in steps), ...]."""
    best = (score(*x0), tuple(x0))
    for step, n in steps:
        centre = np.array(best[1])
        axes = [c + np.arange(-n, n + 1) * st for c, st in zip(centre, step)]
        for x in np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, len(x0)):
            v = score(*x)
            if v > best[0]:
                best = (v, tuple(x))
    return best


def sweep_offset(episode, rays, mount, start, end):
    """Cabin camera offset from the slew axis (link x, y): finished features must stay put as it slews.
    Scored by height correlation, which rewards lining up holes and piles rather than flat overlap."""
    ref, off, mode = frame_set(episode, start, end)
    if len(off) < 5:
        raise SystemExit(f'--sweep {start}:{end}: needs frames slewed >8 degrees from {mode:.0f}')
    frames = load(episode, 'cam2', ref + off)
    pre = {}
    for i, (data, status) in frames.items():
        P, _ = rays.points(data, 4, .9); R, t = link_pose(status, mount.link)
        pre[i] = (P @ mount.R.T + mount.t, R, t)

    def world(i, o):
        PL, R, t = pre[i]
        return near_flat(mount.world((PL + o) @ R.T + t))
    lo, shape = np.array([-1.2, -1.2]), (480, 480)

    def score(ox, oy):
        o = np.array([ox, oy, 0]); W = np.concatenate([world(i, o) for i in ref])
        G, _ = raster(W[:, :2], W[:, 2], lo, shape)
        return mean_correlation([world(i, o) for i in off], G, lo)
    c0 = score(0, 0)
    c1, (ox, oy) = search(score, (0., 0.), [((.04, .04), 6), ((.01, .01), 3), ((.0025, .0025), 3)])
    return np.array([ox, oy]), c0, c1, len(ref), len(off), mode


def ncc_shift(A, B):
    """Integer shift s maximising masked zero-mean correlation of B(x) with A(x + s)."""
    fa, fb = np.isfinite(A), np.isfinite(B)
    a = np.where(fa, A - np.nanmean(A), 0); b = np.where(fb, B - np.nanmean(B), 0)
    P = [2 * s for s in A.shape]; F = np.fft.rfft2; I = np.fft.irfft2
    ma, mb = F(fa.astype(float), P), F(fb.astype(float), P)
    num = I(F(a, P) * np.conj(F(b, P)), P); overlap = I(ma * np.conj(mb), P)
    ea = I(F(a * a, P) * np.conj(mb), P); eb = I(ma * np.conj(F(b * b, P)), P)
    score = np.where(overlap > 2000, num / np.sqrt(np.maximum(ea * eb, 1e-12)), -1)
    k = np.unravel_index(np.argmax(score), score.shape)
    return score[k], np.array([(x + P[j] // 2) % P[j] - P[j] // 2 for j, x in enumerate(k)])


def overlap_register(episode, rays, mounts, start, end):
    """Turn the cabin camera about the slew axis and shift the wrist camera in the cabin frame
    until both cameras' height maps of the same static sand line up."""
    ref, _, mode = frame_set(episode, start, end)
    rows = list(frame_rows(episode))
    wrist_frames = [i for i in range(start, end, 10)
                    if abs(math.degrees(rows[i]['status']['angles_rad'][0]) - mode) < 2]
    cab, wr = mounts['cam2'], mounts['cam1']
    F2, F1 = load(episode, 'cam2', ref), load(episode, 'cam1', wrist_frames)
    lo, shape = np.array([-1.2, -1.2]), (480, 480)
    W = near_flat(np.concatenate([cab.world(cab.base(rays['cam2'].points(F2[i][0], 4, .9)[0], F2[i][1])) for i in ref]))
    G, _ = raster(W[:, :2], W[:, 2], lo, shape)
    wrist = {}
    for i in wrist_frames:
        data, status = F1[i]; W1 = near_flat(wr.world(wr.base(rays['cam1'].points(data, 4, .9)[0], status)))
        if len(W1) >= 3000:
            wrist[i] = W1
    # Coarse: per-frame full-circle turn + FFT shift; keep confident frames.
    est = []
    for i in list(wrist)[::3]:
        W1 = wrist[i]; best, cen = None, W1[:, :2].mean(0)
        for th in np.radians(np.arange(-180, 180, 4)):
            R2 = rz(th)[:2, :2]  # turn about the patch centre so it stays inside the grid
            B, _ = raster((W1[:, :2] - cen) @ R2.T + cen, W1[:, 2], lo, shape); sc, sh = ncc_shift(G, B)
            if best is None or sc > best[0]:
                best = (sc, th, cen - R2 @ cen + sh * CELL)
        if best[0] > .75:
            est.append(best)
    if len(est) < 3:
        raise SystemExit(f'--overlap {start}:{end}: too few confident wrist/cabin matches ({len(est)})')
    # Model: p_cabin = Rot(psi) (p_wrist + Rz(slew) c), c fixed in the upper-carriage frame.
    th = np.array([e[1] for e in est])
    near = np.abs(np.angle(np.exp(1j * (th[:, None] - th[None, :])))) < math.radians(10)
    agree = near[np.argmax(near.sum(1))]  # largest cluster of per-frame turns
    psi0 = float(np.angle(np.exp(1j * th[agree]).mean()))
    u = np.median([e[2] for e, a in zip(est, agree) if a], 0)
    print(f'overlap coarse: {len(est)} frames, turns {np.degrees(th).round().astype(int).tolist()} deg', flush=True)
    s = math.radians(mode); c0 = rz(-s)[:2, :2] @ rz(-psi0)[:2, :2] @ u
    shift = rz(s)[:2, :2]

    def moved(W1, psi, cx, cy):
        Q = W1.copy(); Q[:, :2] = (W1[:, :2] + shift @ [cx, cy]) @ rz(psi)[:2, :2].T
        return Q

    def score(psi, cx, cy):
        return mean_correlation([moved(W1, psi, cx, cy) for W1 in wrist.values()], G, lo)
    first = score(psi0, *c0)
    best, (psi, cx, cy) = search(score, (psi0, *c0), [((math.radians(1), .01, .01), 3),
                                                       ((math.radians(.25), .0025, .0025), 3)])
    r = np.concatenate([(lambda Q: Q[:, 2] - lookup(G, lo, Q[:, :2]))(moved(W1, psi, cx, cy)) for W1 in wrist.values()])
    return psi, np.array([cx, cy]), r[np.isfinite(r)], (first, best), len(est), mode


def reference_map(episode, rays, mount, start, end):
    """Cabin-camera height map of a static stretch at one slew (arm masked): the reference that
    later frames are registered against to correct slew drift, and the source of the field mask."""
    ref, _, mode = frame_set(episode, start, end)
    W = []
    for data, status in load(episode, 'cam2', ref).values():
        P, _ = rays.points(data, 2, .9); Pb = mount.base(P, status); w = mount.world(Pb)
        W.append(w[(np.abs(w[:, 2]) < .08) & ~near_arm(Pb, status, .1)])
    return (*height_map(np.concatenate(W)), mode)


def sand_rectangle(G, lo, margin=.05):
    """Minimum-area rectangle around the largest connected near-flat region, shrunk by margin
    (walls and the sand slope against them read wrong at grazing angles)."""
    from scipy import ndimage
    from scipy.spatial import ConvexHull
    flat = ndimage.binary_opening(np.isfinite(G) & (np.abs(np.nan_to_num(G, nan=1.)) < .06), iterations=3)
    labels, n = ndimage.label(flat)
    big = labels == 1 + np.argmax(ndimage.sum(flat, labels, range(1, n + 1)))
    xy = lo + (np.argwhere(big) + .5) * CELL; hull = xy[ConvexHull(xy).vertices]
    best = None
    for e in np.diff(np.vstack([hull, hull[:1]]), axis=0):
        u = e / np.linalg.norm(e); v = np.array([-u[1], u[0]]); a, b = hull @ u, hull @ v
        area = (a.max() - a.min()) * (b.max() - b.min())
        if best is None or area < best[0]:
            best = (area, u, v, a.min() + margin, a.max() - margin, b.min() + margin, b.max() - margin)
    _, u, v, a0, a1, b0, b1 = best
    return [list(u * a + v * b) for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))]  # counter-clockwise


def tilt_pivot(session, episodes, rays, mount, every=6):
    """Base IMU yaw and tipping-axis pivot that keep the cabin camera's undisturbed sand level.
    Needs a take where the machine tips (bucket pushing on the bottom); otherwise skipped."""
    data = []
    for ep in episodes:
        rows = list(frame_rows(ep))
        for i, d in enumerate(depth_frames(ep / 'cam2.depth.zst')):
            if i % every or not rows[i]['status'].get('fk_valid'):
                continue
            st = rows[i]['status']; P, _ = rays.points(d, 6, .8); Pb = mount.base(P, st)  # untilted
            data.append((Pb[~near_arm(Pb, st, .1)], st))
    tilts = np.array([math.degrees(math.asin(min(np.linalg.norm(np.cross(base_gravity(st), GRAVITY_REFERENCE)), 1.))) for _, st in data])
    tipped = tilts > math.degrees(TILT_DEADBAND) + .25
    if tipped.sum() < 5:
        return None, float(tilts.max())

    def level(Pb, st, yaw, pivot):
        T, u = tilt_correction(st, yaw, pivot); W = mount.world(Pb @ T.T + u); h = W[np.abs(W[:, 2]) < .04, 2]
        return np.median(h) if len(h) > 200 else np.nan
    target = np.nanmedian([level(Pb, st, 0., np.zeros(3)) for (Pb, st), t in zip(data, tipped) if not t])
    data = [d for d, t in zip(data, tipped) if t]

    def spread(yaw, cx, cy):  # tipped frames should sit at the level of untipped ones
        lv = np.array([level(Pb, st, yaw, np.array([cx, cy, 0.])) for Pb, st in data])
        return -float(np.nanmean(np.abs(lv - target)))
    best = None
    for yaw in np.radians([0, 90, 180, 270]):
        b = search(lambda cx, cy: spread(yaw, cx, cy), (0., 0.), [((.1, .1), 12)])
        if best is None or b[0] > best[0]:
            best = (b[0], yaw, *b[1])
    b = search(lambda cx, cy: spread(best[1], cx, cy), best[2:], [((.025, .025), 4)])
    global GRAVITY_REFERENCE
    keep, GRAVITY_REFERENCE = GRAVITY_REFERENCE, None; none = spread(0., 0., 0.); GRAVITY_REFERENCE = keep
    return (best[1], np.array([*b[1], 0.]), -none * 1000, -b[0] * 1000, int(tipped.sum())), float(tilts.max())


def p50_p90_mm(r):
    return [round(float(np.median(np.abs(r))) * 1000, 2), round(float(np.percentile(np.abs(r), 90)) * 1000, 2)]


def calibrate(args):
    session = args.session
    path = args.output or session / 'sand-calibration.json'
    rays = {k: Rays(session, k) for k in ('cam1', 'cam2')}
    flat = episode_path(session, args.flat)
    global GRAVITY_REFERENCE
    flat_rows = [r['status'] for r in frame_rows(flat) if r['status'].get('fk_valid')]
    g = np.median([base_gravity(st) for st in flat_rows], 0)
    levelled = all(st.get('fk_cabin_tilt') for st in flat_rows)  # the controller's FK is already level
    GRAVITY_REFERENCE = None if args.no_tilt or levelled else g / np.linalg.norm(g)
    report, mounts = {}, {}
    for key, links in (('cam2', (0,)), ('cam1', (1, 2, 3))):
        samples = flat_samples(flat, rays[key])
        fits = [(np.median(np.abs(r)), link, m, r) for link in links for m, r in [fit_mount(samples, link)]]
        med, link, mount, r = min(fits, key=lambda f: f[0])
        mounts[key] = mount
        report[key] = {'flat_frames': len(samples), 'link': LINKS[link],
                       'flat_residual_mm_p50_p90': [round(med * 1000, 2), round(float(np.percentile(np.abs(r), 90)) * 1000, 2)]}
        print(f'{key}: on {LINKS[link]} link, flat-sand residual p50 {med * 1000:.1f} mm p90 '
              f'{np.percentile(np.abs(r), 90) * 1000:.1f} mm over {len(samples)} frames', flush=True)
    global TILT_YAW, TILT_PIVOT
    if GRAVITY_REFERENCE is not None:
        eps = sorted({args.flat, *(x[0] for x in (args.sweep, args.overlap) if x)})
        fit, worst = tilt_pivot(session, [episode_path(session, e) for e in eps], rays['cam2'], mounts['cam2'])
        if fit is None:
            print(f'tilt: chassis never tipped over {worst:.2f} deg; tilt correction about the base origin only', flush=True)
        else:
            TILT_YAW, TILT_PIVOT, before, after, n = fit
            report['tilt'] = {'max_tilt_deg': round(worst, 2), 'imu_yaw_deg': round(math.degrees(TILT_YAW)),
                              'pivot_base_xy_m': TILT_PIVOT[:2].round(3).tolist(),
                              'tipped_frames': n, 'tipped_level_error_mm_before_after': [round(before, 2), round(after, 2)]}
            print(f'tilt: up to {worst:.1f} deg; IMU yaw {math.degrees(TILT_YAW):.0f} deg, pivot {TILT_PIVOT[:2].round(2)} m; '
                  f'level error on {n} tipped frames {before:.1f} -> {after:.1f} mm', flush=True)
    if args.sweep:
        ep, start, end = args.sweep
        o, c0, c1, nref, noff, mode = sweep_offset(episode_path(session, ep), rays['cam2'], mounts['cam2'], start, end)
        xy = mounts['cam2'].t[:2] + o
        samples = flat_samples(flat, rays['cam2'])
        cab = mounts['cam2']; moved = Mount(0, cab.R, np.r_[xy, cab.t[2]], cab.n, cab.d)
        mounts['cam2'], r = fit_mount(samples, 0, start=moved)
        report['cam2'].update(sweep_offset_m=o.round(4).tolist(), sweep_correlation=[round(c0, 3), round(c1, 3)],
                              flat_residual_mm_p50_p90=p50_p90_mm(r))
        print(f'cam2: slew-axis offset {o.round(3)} m from {nref} frames at {mode:.0f} deg vs {noff} slewed; '
              f'correlation {c0:.2f} -> {c1:.2f}; flat residual p50 {np.median(np.abs(r)) * 1000:.1f} mm', flush=True)
    if args.overlap:
        ep, start, end = args.overlap
        psi, shift, r, (c0, c1), n, mode = overlap_register(episode_path(session, ep), rays, mounts, start, end)
        cab, wr = mounts['cam2'], mounts['cam1']
        Rz = rz(-psi)  # bring the cabin camera into the wrist camera's (FK-consistent) base frame
        mounts['cam2'] = Mount(cab.link, Rz @ cab.R, Rz @ cab.t, Rz @ cab.n, cab.d)
        TILT_PIVOT = Rz @ TILT_PIVOT  # the pivot was fitted in the cabin camera's base frame
        # IMU axes vs the upper carriage: the gauge turn also turns that frame's carriage axes
        TILT_YAW = TILT_YAW - psi
        if 'tilt' in report:
            report['tilt'].update(imu_yaw_deg=round(math.degrees(TILT_YAW) % 360, 1), pivot_base_xy_m=TILT_PIVOT[:2].round(3).tolist())
        mounts['cam1'] = Mount(wr.link, wr.R, wr.t, wr.n, wr.d, shift)
        report['overlap'] = {'cabin_yaw_deg': round(-math.degrees(psi), 2), 'wrist_cabin_shift_m': shift.round(4).tolist(),
                             'correlation_coarse_fine': [round(c0, 3), round(c1, 3)],
                             'confident_coarse_frames': n, 'slew_deg': round(mode, 1),
                             'height_diff_mm_p50_p90': p50_p90_mm(r),
                             'height_bias_mm': round(float(np.median(r)) * 1000, 2)}
        print(f'overlap: cabin camera yaw {-math.degrees(psi):+.1f} deg, wrist shift {shift.round(3)} m (cabin frame); '
              f'correlation {c0:.2f} -> {c1:.2f}; wrist minus cabin height p50 |d| {np.median(np.abs(r)) * 1000:.1f} mm, '
              f'bias {np.median(r) * 1000:+.1f} mm', flush=True)
    extra = {}
    if args.overlap:
        G, lo, mode = reference_map(episode_path(session, args.overlap[0]), rays['cam2'], mounts['cam2'], *args.overlap[1:])
        poly = sand_rectangle(G, lo)
        reference = path.with_suffix('.reference.npz')
        np.savez_compressed(reference, height=G, lo=lo)
        extra = {'reference_map': reference.name, 'reference_slew_deg': round(float(mode), 2)}
    else:
        poly = sand_rectangle(*reference_map(flat, rays['cam2'], mounts['cam2'], 0, 10 ** 9)[:2])
    out = {'format': 'kaivuri-sand-calibration-v1', 'session': str(session),
           'serials': {k: json.loads((session / f'{k}.depth.json').read_text())['serial'] for k in rays},
           'arguments': {'flat': args.flat, 'sweep': args.sweep, 'overlap': args.overlap},
           'model': 'p_base = R_link (R p_camera + t) + t_link; height = n . p_base + d',
           'base_gravity_reference': None if GRAVITY_REFERENCE is None else GRAVITY_REFERENCE.tolist(),
           'tilt_imu_yaw_deg': math.degrees(TILT_YAW), 'tilt_pivot_base_m': TILT_PIVOT.tolist(),
           'tilt_deadband_deg': math.degrees(TILT_DEADBAND),
           'cameras': {k: m.to_json() for k, m in mounts.items()}, 'sand_polygon_level_xy_m': poly, **extra, 'report': report}
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {path}')


# ---------------------------------------------------------------- viewing
def diverging(h, limit):
    """Blue (below) - white - red (above); NaN black."""
    t = np.clip(np.nan_to_num(h, nan=0.) / limit, -1, 1)[..., None]
    white, blue, red = np.array([247, 247, 247.]), np.array([33, 102, 172.]), np.array([178, 24, 43.])
    rgb = np.where(t < 0, white + (blue - white) * -t, white + (red - white) * t)
    rgb[np.isnan(h)] = 0
    return rgb.astype(np.uint8)


def inside(poly, xy):
    """Points inside a convex counter-clockwise polygon."""
    a, b = poly, np.roll(poly, -1, 0); e = b - a
    cross = e[None, :, 0] * (xy[:, None, 1] - a[None, :, 1]) - e[None, :, 1] * (xy[:, None, 0] - a[None, :, 0])
    return (cross >= 0).all(1)


def near_arm(Pb, status, radius):
    """Points within radius of the boom/arm/bucket segments (FK base frame, tilt corrected)."""
    chain = fk_chain(status, 1)
    hit = np.zeros(len(Pb), bool)
    for a, b in zip(chain[:-1], chain[1:]):
        ab = b - a; t = np.clip((Pb - a) @ ab / max(ab @ ab, 1e-9), 0, 1)
        hit |= np.einsum('ij,ij->i', Pb - a - t[:, None] * ab, Pb - a - t[:, None] * ab) < radius ** 2
    return hit


class SlewTracker:
    """Slew drift correction: turn the cabin camera's view about the slew axis until it lines up
    with the reference map. The same turn applies to every camera (all use the slew angle)."""
    def __init__(self, G, lo, mount, rays):
        self.G, self.lo, self.mount, self.rays, self.delta, self.score = G, lo, mount, rays, 0., None

    def update(self, data, status):
        P, _ = self.rays.points(data, 4, .9); Pb = self.mount.base(P, status)
        Pb = Pb[~near_arm(Pb, status, .1)]

        def score(d):
            return correlation(near_flat(self.mount.world(Pb @ rz(d).T)), self.G, self.lo) or -1.

        def best(centre, half, step):
            return max(((score(d), d) for d in centre + np.arange(-half, half + 1e-9, step)))
        c, d = best(self.delta, math.radians(4), math.radians(.5))
        if c < .5:  # lost: wide search
            c, d = max((c, d), best(0., math.radians(40), math.radians(1)))
        c, d = best(d, math.radians(.5), math.radians(.1))
        if c >= .4:
            self.delta = d
        self.score = c
        return self.delta


class FieldMap:
    """Level height map of the sand field on 5 mm cells: each frame's cameras fused by point count,
    the change since each cell was first seen this take, and litres moved."""
    def __init__(self, poly):
        self.poly = poly
        self.lo = poly.min(0) - .02; self.shape = tuple(np.ceil((poly.max(0) + .02 - self.lo) / CELL).astype(int))
        cells = np.stack(np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing='ij'), -1)
        centres = self.lo + (cells.reshape(-1, 2) + .5) * CELL
        self.in_field = inside(poly, centres).reshape(self.shape)
        middle = poly.mean(0); inner = poly + (middle - poly) * (.03 * math.sqrt(2) / np.linalg.norm(middle - poly, axis=1))[:, None]
        self.in_metric = inside(inner, centres).reshape(self.shape)  # metrics keep 3 more cm off the walls
        self.fused = np.full(self.shape, np.nan); self.start = np.full(self.shape, np.nan); self.change = self.fused

    def points(self, mount, rays, data, status, turn, a):
        """Level points of one depth frame, which to keep (inside the field, within a.below_m/a.above_m
        of the sand and off the arm) and their flat pixel indices."""
        P, idx = rays.points(data, a.stride, a.max_range_m)
        Pb = mount.base(P, status); arm = near_arm(Pb, status, a.arm_radius_m)
        W = mount.world(Pb @ turn.T)
        good = (W[:, 2] > -a.below_m) & (W[:, 2] < a.above_m) & inside(self.poly, W[:, :2]) & ~arm
        return W, good, idx

    def begin(self):
        self.update, self.weight = np.zeros(self.shape), np.zeros(self.shape)

    def add(self, key, W, fusion):
        """Add one camera's kept points to this frame, cabin camera (cam2) first."""
        mean, count = raster(W[:, :2], W[:, 2], self.lo, self.shape)
        seen = count > 0
        if key == 'cam1' and fusion == 'fill':
            seen &= self.weight == 0  # wrist camera only where the cabin camera sees nothing this frame
        if key == 'cam1' and fusion == 'cabin':
            seen[:] = False
        self.update[seen] += mean[seen] * count[seen]; self.weight[seen] += count[seen]

    def commit(self):
        seen = self.weight > 0; self.fused[seen] = self.update[seen] / self.weight[seen]
        new = seen & np.isnan(self.start); self.start[new] = self.fused[new]  # first look at each cell this take
        self.change = self.fused - self.start

    def metrics(self, deadband_mm):
        known = np.isfinite(self.change) & self.in_metric; c, h = self.change[known], self.fused[known]
        area = CELL * CELL * 1000; dead = deadband_mm / 1000  # litres per metre; ignore noise
        dug = h[c < -dead]  # bottom of what was dug, against the flat sand (1st percentile)
        return {'dug_l': float(-c[c < -dead].sum() * area), 'piled_l': float(c[c > dead].sum() * area),
                'hole_bottom_mm': float(-np.percentile(dug, 1) * 1000) if len(dug) > 50 else 0.}

    def image(self, grid, limit):
        """Field cells of grid as a colour image, x right and y up."""
        return diverging(np.where(self.in_field, grid, np.nan).T[::-1], limit)


def view(args):
    import rerun as rr
    import rerun.blueprint as rrb
    session = args.session
    cal_path = args.calibration or session / 'sand-calibration.json'
    cal = json.loads(cal_path.read_text())
    if not args.no_tilt:
        load_tilt(cal)
    mounts = {k: Mount.from_json(c) for k, c in cal['cameras'].items()}
    rays = {k: Rays(session, k) for k in mounts}
    tracker = None
    if 'reference_map' in cal and 'cam2' in mounts and not args.no_slew_correction:
        ref = np.load(cal_path.parent / cal['reference_map'])
        tracker = SlewTracker(ref['height'], ref['lo'], mounts['cam2'], rays['cam2'])
    field = FieldMap(np.asarray(cal['sand_polygon_level_xy_m']))
    episode = episode_path(session, args.episode)
    rows = list(frame_rows(episode))
    ir = {k: np.memmap(episode / f'{k}.y8', np.uint8, 'r').reshape(-1, rays[k].h, rays[k].w) for k in mounts}
    streams = {k: depth_frames(episode / f'{k}.depth.zst') for k in mounts}
    lim = args.color_mm / 1000
    rr.init('kaivuri-sand-height', spawn=False)
    order = [k for k in ('cam2', 'cam1') if k in mounts]
    blueprint = rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(*[v for k in order for v in (rrb.Spatial2DView(origin=f'ir/{k}', name=f'{k} IR'),
                                                     rrb.Spatial2DView(origin=f'height/{k}', name=f'{k} height'))]),
        rrb.Horizontal(rrb.Spatial2DView(origin='map', name=f'Fused height above flat sand (+-{args.color_mm:.0f} mm, 5 mm cells)'),
                       rrb.Spatial2DView(origin='change', name=f'Change since start of take (+-{args.color_mm:.0f} mm)'),
                       rrb.Spatial3DView(origin='world', name='Level sand frame (m)', line_grid=False)),
        rrb.Horizontal(rrb.TimeSeriesView(origin='volume', name='Moved since start (litres)'),
                       rrb.TimeSeriesView(origin='depth', name='Dug hole bottom below flat sand (mm)'),
                       rrb.TimeSeriesView(origin='slew', name='Visual slew correction (deg) and match')),
        row_shares=[2, 3, 1]))
    if args.save:
        rr.save(args.save, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    ring = np.vstack([field.poly, field.poly[:1]])
    rr.log('world/sand_field', rr.LineStrips3D([np.c_[ring, np.zeros(len(ring))]], radii=.002, colors=[[90, 90, 90]]), static=True)
    s = args.stride; t0 = time.monotonic()
    for i, row in enumerate(rows):
        data = {k: next(streams[k]) for k in mounts}
        if i % args.every:
            continue
        rr.set_time('frame_index', sequence=i); status = row['status']
        if not status.get('fk_valid'):
            continue
        turn = np.eye(3)
        if tracker is not None:
            delta = tracker.update(data['cam2'], status); turn = rz(delta)
            rr.log('slew/correction_deg', rr.Scalars(math.degrees(delta)))
            rr.log('slew/match', rr.Scalars(tracker.score))
        field.begin(); cloud = []
        for k in order:  # cabin camera first
            r = rays[k]; W, good, idx = field.points(mounts[k], r, data[k], status, turn, args)
            gh, gw = len(range(0, r.h, s)), len(range(0, r.w, s))
            img = np.full(gh * gw, np.nan); img[idx[good]] = W[good, 2]
            rr.log(f'height/{k}', rr.Image(diverging(img.reshape(gh, gw), lim)))
            rr.log(f'ir/{k}', rr.Image(np.asarray(ir[k][i][::s, ::s])).compress(jpeg_quality=80))
            field.add(k, W[good], args.fusion)
            cloud.append(W[good][::2])
        field.commit()
        rr.log('map', rr.Image(field.image(field.fused, lim)))
        rr.log('change', rr.Image(field.image(field.change, lim)))
        C = np.concatenate(cloud)
        rr.log('world/points', rr.Points3D(C, colors=diverging(C[:, 2], lim), radii=.0015))
        m = mounts.get('cam1') or mounts['cam2']
        rr.log('world/arm', rr.LineStrips3D([m.world(fk_chain(status) @ turn.T)], radii=.004, colors=[[255, 200, 0]]))
        metrics = field.metrics(args.deadband_mm)
        rr.log('volume/dug_l', rr.Scalars(metrics['dug_l']))
        rr.log('volume/piled_l', rr.Scalars(metrics['piled_l']))
        rr.log('depth/hole_bottom_mm', rr.Scalars(metrics['hole_bottom_mm']))
    rr.get_global_data_recording().flush(timeout_sec=20)
    print(f'{episode.name}: {len(rows)} frames, 1 in {args.every} shown, {time.monotonic() - t0:.0f} s', flush=True)


def span(text):
    ep, start, end = (int(x) for x in text.split(':'))
    return ep, start, end


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('calibrate'); c.add_argument('session', type=Path)
    c.add_argument('--flat', type=int, required=True, help='episode index of the flattened-sand take')
    c.add_argument('--sweep', type=span, help='EP:START:END static frames where the cabin slews over features')
    c.add_argument('--overlap', type=span, help='EP:START:END static frames both cameras see')
    c.add_argument('-o', '--output', type=Path)
    c.add_argument('--no-tilt', action='store_true', help='ignore base IMU chassis tilt')
    v = sub.add_parser('view'); v.add_argument('session', type=Path)
    v.add_argument('--calibration', type=Path); v.add_argument('--episode', type=int, default=0)
    v.add_argument('--every', type=int, default=3, help='show every Nth frame')
    v.add_argument('--stride', type=int, default=2, help='depth pixel step')
    v.add_argument('--max-range-m', type=float, default=.8, help='drop depth farther than this from the camera')
    v.add_argument('--above-m', type=float, default=.12, help='drop points this far above the flat sand')
    v.add_argument('--below-m', type=float, default=.15, help='drop points this far below the flat sand')
    v.add_argument('--arm-radius-m', type=float, default=.11, help='drop points this close to the FK arm')
    v.add_argument('--color-mm', type=float, default=50)
    v.add_argument('--deadband-mm', type=float, default=8, help='changes smaller than this count as unchanged')
    v.add_argument('--fusion', choices=('fill', 'average', 'cabin'), default='cabin',
                   help='cabin: cabin camera only (default; wrist FK heights are 5-10 mm off at some poses); '
                        'fill: wrist camera where the cabin camera is blocked; '
                        'average: both weighted by point count')
    v.add_argument('--no-tilt', action='store_true', help='ignore base IMU chassis tilt')
    v.add_argument('--no-slew-correction', action='store_true', help='trust the recorded (gyro) slew angle')
    v.add_argument('--save', type=Path, help='write an .rrd instead of spawning a viewer')
    args = p.parse_args()
    (calibrate if args.command == 'calibrate' else view)(args)


if __name__ == '__main__':
    main()
