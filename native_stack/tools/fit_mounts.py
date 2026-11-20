#!/usr/bin/env python3
"""Camera mounts for the height map, from a flat take, checked on objects of known height.

  fit_mounts.py SESSION --flat EP [--holdout EP] [--object EP:MM ...] [--start FILE] [-o FILE]

The ground plane comes from the bucket tip touching it (FK tip held still at its lowest, in the
--touches takes), so heights share the FK's frame. Then, on the flat take:
  cabin camera (slew link): its two tilts and its height. These set every height value; its yaw
      and x/y only move the map sideways and flat ground cannot see them, so they stay at the
      --start mount (or --cabin-xy, tape measured from the slew axis).
  wrist camera (arm link): all but its offset along the hinge axis (link y), which flat ground
      cannot see either (--wrist-y, default from --start).
No depth correction: the stereo dome is ~0.1 mm with the close-range settings (flat-calib-001).
Writes the sand_height.py calibration format, so `sand_height.py view --calibration` and the
ROS bridge (kaivuri_bridge.camera_transforms) read it unchanged.

Recordings made before level_frame.tilt_pivot_dist was in the controller profile get the same
track-end tilt here (--tilt-pivot-dist), so old and new takes level alike.
"""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.optimize import least_squares

import sand_height as sh

KEYS = {'cam2': 0, 'cam1': 2}  # camera -> FK link (slew, arm), as in sand_height.py
DEFAULT_START = Path.home() / 'depth-live-bench/sand-calibration.json'


def pivot_in_fk(episode):
    """True when the controller that recorded this take already tilted about the track end."""
    cfg = episode.parent / 'provenance/configuration_files/profiles/jetson/control_config.yaml'
    return cfg.is_file() and re.search(r'^\s*tilt_pivot_dist:\s*[0-9.]', cfg.read_text(), re.M) is not None


def lifted(status, dist):
    """Status with the controller's track-end tilt applied (core.cpp forward_kinematic_state)."""
    x = sh.quat_matrix(status['cabin_tilt_quat_wxyz'])[:, 0]
    end = -dist if x[2] > 0 else dist
    shift = np.array([end * (1 - x[0]), -end * x[1], -end * x[2]])
    return {**status, 'fk_joint_xyz_m': (np.asarray(status['fk_joint_xyz_m']) + shift).tolist(),
            'fk_xyz_m': (np.asarray(status['fk_xyz_m']) + shift).tolist()}


def samples(session, ep, key, dist):
    episode = sh.episode_path(session, ep)
    out = sh.flat_samples(episode, sh.Rays(session, key))
    if dist and not pivot_in_fk(episode):
        out = [(i, lifted(st, dist), P, n, d) for i, st, P, n, d in out]
    return out, episode


def touches(episode, dist, still_rows=20, band=.015):
    """FK tip positions where the bucket rests on the ground: lowest stretches held >= 0.2 s."""
    st = [s for s in (json.loads(l).get('status') for l in open(episode / 'samples.jsonl')) if s and s.get('fk_valid')]
    if dist and not pivot_in_fk(episode):
        st = [lifted(s, dist) for s in st]
    tip = np.array([s['fk_xyz_m'] for s in st])
    low = np.r_[tip[:, 2] < tip[:, 2].min() + band, False]
    out, s0 = [], None
    for j, b in enumerate(low):
        if b and s0 is None:
            s0 = j
        if not b and s0 is not None:
            if j - s0 >= still_rows:
                out.append(np.median(tip[s0:j], 0))
            s0 = None
    return np.array(out)


def ground(points):
    """Plane n . p + d = 0 (n up) through the touch points, least squares on z = a x + b y + c."""
    (a, b, c), *_ = np.linalg.lstsq(np.c_[points[:, :2], np.ones(len(points))], points[:, 2], rcond=None)
    n = np.array([-a, -b, 1.]); k = np.linalg.norm(n)
    return n / k, -c / k, points[:, 2] - points[:, :2] @ [a, b] - c


def fit(flat, start, n, d, wrist_y):
    """Least squares with the ground fixed: cabin tilts + height (3), wrist mount without link y (5)."""
    poses = {k: [sh.link_pose(s[1], KEYS[k]) for s in flat[k]] for k in flat}
    cab, wr = start['cam2'], start['cam1']

    def unpack(x):
        # cabin: turn about its link's x and y (both horizontal), never about the vertical
        return {'cam2': (sh.rotvec_matrix([x[0], x[1], 0]) @ cab.R, np.r_[cab.t[:2], x[2]]),
                'cam1': (sh.rotvec_matrix(x[3:6]) @ wr.R, np.r_[x[6], wrist_y, x[7]])}

    x0 = np.r_[0, 0, cab.t[2], 0, 0, 0, wr.t[0], wr.t[2]]

    def residual(x):
        r = []
        for k, (R, t) in unpack(x).items():
            r += [(s[2] @ R.T + t) @ RL.T @ n + tL @ n + d for s, (RL, tL) in zip(flat[k], poses[k])]
        return np.concatenate(r)

    out = least_squares(residual, x0, loss='soft_l1', f_scale=.004, x_scale='jac')
    return {k: sh.Mount(KEYS[k], R, t, n, d) for k, (R, t) in unpack(out.x).items()}


def flat_residual(m, flat):
    return np.concatenate([m.world(m.base(s[2], s[1]))[:, 2] for s in flat])


def object_height(session, ep, key, mount, dist, mm, every=15):
    """Median top height of an object of about mm over the frames where it shows as a tight cluster."""
    episode = sh.episode_path(session, ep); rays = sh.Rays(session, key)
    fix = dist and not pivot_in_fk(episode)
    tops = []
    for i, (data, row) in enumerate(zip(sh.depth_frames(episode / f'{key}.depth.zst'), sh.frame_rows(episode))):
        st = row['status']
        if i % every or not st.get('fk_valid'):
            continue
        P, _ = rays.points(data, 4, 1.)
        h = mount.world(mount.base(P, lifted(st, dist) if fix else st))[:, 2]
        top = h[(h > mm * .6e-3) & (h < mm * 1.4e-3)]
        if len(top) > 150 and np.subtract(*np.percentile(top, [75, 25])) < .006:
            tops.append(np.median(top))
    return (round(float(np.median(tops)) * 1e3, 1), len(tops)) if tops else (None, 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('session', type=Path)
    p.add_argument('--flat', type=int, required=True, help='episode index of the flat take')
    p.add_argument('--holdout', type=int, help='another flat episode, scored only')
    p.add_argument('--touches', type=int, nargs='+', help='episodes with tip touches on the ground (default: --flat and --holdout)')
    p.add_argument('--object', action='append', default=[], metavar='EP:MM', help='episode with an object MM tall (scored only)')
    p.add_argument('--start', type=Path, default=DEFAULT_START, help='calibration with the starting mounts (cabin yaw, x/y)')
    p.add_argument('--cabin-xy', type=float, nargs=2, metavar=('X', 'Y'), help='tape-measured cabin camera x, y on the slew link (m)')
    p.add_argument('--wrist-y', type=float, help='wrist camera offset along the arm hinge axis (m)')
    p.add_argument('--tilt-pivot-dist', type=float, default=.16, help='track-end tilt for takes recorded without it (m; 0: off)')
    p.add_argument('-o', '--output', type=Path)
    a = p.parse_args()
    cal = json.loads(a.start.read_text())
    start = {k: sh.Mount.from_json(cal['cameras'][k]) for k in KEYS}
    if a.cabin_xy:
        start['cam2'].t[:2] = a.cabin_xy
    # The old tool's wrist mount carries most of its sideways offset in cabin_shift_xy_m.
    wrist_y = a.wrist_y if a.wrist_y is not None else start['cam1'].t[1] + start['cam1'].cabin_shift[1]
    eps = a.touches or [e for e in (a.flat, a.holdout) if e is not None]
    tips = np.concatenate([touches(sh.episode_path(a.session, e), a.tilt_pivot_dist) for e in eps])
    n, d, rt = ground(tips)
    report = {'ground': {'touches': len(tips), 'tilt_deg': round(math.degrees(math.acos(min(n[2], 1.))), 2),
                         'height_at_origin_mm': round(-d / n[2] * 1e3, 1), 'touch_residual_mm_p50_p90': sh.p50_p90_mm(rt)}}
    print('ground', json.dumps(report['ground']))
    flat = {k: samples(a.session, a.flat, k, a.tilt_pivot_dist)[0] for k in KEYS}
    mounts = fit(flat, start, n, d, wrist_y)
    for k, m in mounts.items():
        rep = report[k] = {'link': sh.LINKS[m.link], 'translation_m': m.t.round(4).tolist(),
                           'flat_mm_p50_p90': sh.p50_p90_mm(flat_residual(m, flat[k]))}
        if a.holdout is not None:
            rep['holdout_mm_p50_p90'] = sh.p50_p90_mm(flat_residual(m, samples(a.session, a.holdout, k, a.tilt_pivot_dist)[0]))
        for spec in a.object:
            ep, mm = spec.split(':')
            got, frames = object_height(a.session, int(ep), k, m, a.tilt_pivot_dist, float(mm))
            rep[f'object_ep{ep}_{mm}mm'] = {'measured_mm': got, 'frames': frames}
        print(k, json.dumps(rep))
    out = {'format': 'kaivuri-sand-calibration-v1', 'tool': 'fit_mounts.py', 'session': str(a.session),
           'arguments': {'flat': a.flat, 'touches': eps, 'start': str(a.start), 'cabin_xy': a.cabin_xy,
                         'wrist_y': wrist_y, 'tilt_pivot_dist': a.tilt_pivot_dist},
           'model': 'p_base = R_link (R p_camera + t) + t_link; height = n . p_base + d',
           'serials': {k: json.loads((a.session / f'{k}.depth.json').read_text())['serial'] for k in KEYS},
           'cameras': {k: m.to_json() for k, m in mounts.items()}, 'depth_correction': {}, 'report': report}
    path = a.output or a.session / 'mounts.json'
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
