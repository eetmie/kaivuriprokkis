"""IMU chain audit from recorded native sessions (no hardware, NumPy only).

  imu_audit.py SESSION_OR_EPISODE [...] [--json FILE]

Pools every 100 Hz sample row of the given sessions/episodes and reports:

  1. Hinge axis per IMU, from its gyro while that joint moves alone: the angle between the
     measured rotation axis and the axis the extraction assumes (sensor Y for boom/arm/bucket,
     sensor Z for the cabin IMU's slew), split into its two tilt directions. A constant tilt is
     mounting misalignment; off-axis rate that grows with load or speed points at hinge play.
  2. Gravity roll of each link IMU relative to the cabin IMU while everything is still. On
     parallel hinges this is zero at every joint angle; a slope against the joint angle is
     mounting yaw, scatter is play or flex.
  3. Angle extraction gain and latency per joint: the recorded joint angle rate against the
     gyro hinge rate (child minus parent). Gain off 1.0 is scale error; the lag is the AHRS/
     extraction delay.
  4. Cabin rocking: roll/pitch rate spectrum and ringing frequencies of the cabin IMU, and the
     cabin's static roll/pitch from gravity.

Every result carries its sample count and a block-bootstrap 95% interval (1 s blocks), and is
marked LOW DATA under the thresholds below. Best data: free-air back-and-forth of one joint at a
time (3-4 cycles each, slow and fast), a slew back-and-forth, and still periods between moves.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from export_episodes import committed_episodes, samples

NAMES = ('cabin', 'boom', 'arm', 'bucket')  # IMU rows and joints share this order (slew = cabin IMU)
JOINTS = ('slew', 'boom', 'arm', 'bucket')
MIN_MOVING, MIN_STILL = 1000, 2000  # samples (10 s / 20 s at 100 Hz)


def episodes(paths):
    for p in paths:
        p = Path(p)
        if (p / 'samples.jsonl').exists():
            yield p
        else:
            found = committed_episodes(p)
            if not found:
                raise SystemExit(f'{p}: no saved episodes')
            yield from found


def load(paths):
    """Per-episode arrays: time (s), gyro (n,4,3) dps, accel (n,4,3) g, joint angles (n,4) rad."""
    out = []
    for ep in episodes(paths):
        t, g, a, q = [], [], [], []
        for row in samples(ep):
            st = row['status']
            if not st.get('fk_valid') or st.get('simulated_imu') or st.get('imu_gyro_dps') is None:
                continue
            t.append(row['sample_ns'] / 1e9); g.append(st['imu_gyro_dps']); a.append(st['imu_accel_g']); q.append(st['angles_rad'])
        if len(t) > 50:
            out.append(dict(name=ep.name, t=np.array(t), gyro=np.array(g), accel=np.array(a), angle=np.unwrap(np.array(q), axis=0)))
    if not out:
        raise SystemExit('no usable IMU samples')
    return out


def rate(t, x, half=3):
    """Centred difference over +-half samples (smooths 100 Hz quantisation)."""
    r = np.zeros_like(x); r[half:-half] = (x[2 * half:] - x[:-2 * half]) / (t[2 * half:] - t[:-2 * half])[:, None]
    return r


def blocks(n, size=100):
    return [np.arange(i, min(i + size, n)) for i in range(0, n, size)]


def bootstrap(stat, idx, reps=200, seed=0):
    """95% interval of stat(sample indices) resampling blocks of the selected samples (1 s, or
    a tenth of them when fewer, so short selections still get an interval)."""
    rng = np.random.default_rng(seed); bl = blocks(len(idx), max(5, min(100, len(idx) // 10)))
    vals = [stat(idx[np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))])]) for _ in range(reps)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def principal_axis(w, nominal):
    _, _, vt = np.linalg.svd(w, full_matrices=False); ax = vt[0]
    return ax if ax @ nominal >= 0 else -ax


def flag(n, need):
    return '' if n >= need else f'  LOW DATA ({n} < {need})'


def audit(data):
    t = np.concatenate([d['t'] for d in data]); G = np.concatenate([d['gyro'] for d in data])
    A = np.concatenate([d['accel'] for d in data]); Q = np.concatenate([d['angle'] for d in data])
    R = np.degrees(np.concatenate([rate(d['t'], d['angle']) for d in data]))  # joint rates, dps
    edge = np.concatenate([np.r_[np.ones(3, bool), np.zeros(len(d['t']) - 6, bool), np.ones(3, bool)] for d in data])
    report = {'episodes': [d['name'] for d in data], 'samples': int(len(t))}
    still = np.all(np.linalg.norm(G, axis=2) < 2, axis=1) & np.all(np.abs(R) < 1, axis=1) & ~edge
    print(f"{len(data)} episodes, {len(t)} samples ({len(t) / 100:.0f} s), {still.sum()} still")

    # 1. hinge axes
    print('\n1) Hinge axis in each IMU (that joint alone moving > 15 dps):')
    report['hinge_axis'] = {}; axes = {}
    for k in range(4):
        nominal = np.array([0, 0, 1.]) if k == 0 else np.array([0, 1., 0])
        others = np.delete(np.abs(R), k, axis=1).max(1)
        moving = (np.abs(R[:, k]) > 15) & (others < .2 * np.abs(R[:, k])) & ~edge
        idx = np.flatnonzero(moving); w = G[idx, k]
        if len(idx) < 20:
            print(f'  {NAMES[k]:6s} ({JOINTS[k]}): no isolated motion'); continue
        ax = principal_axis(w, nominal); axes[k] = ax
        tilt = lambda i: math.degrees(math.acos(min(abs(principal_axis(G[i, k], nominal) @ nominal), 1.)))
        off = np.linalg.norm(w - np.outer(w @ ax, ax), axis=1) / np.linalg.norm(w, axis=1)
        # tilt components: rotate nominal toward the measured axis about the two other sensor axes
        comp = np.degrees(np.arcsin(np.clip(ax, -1, 1))); comp[np.argmax(nominal)] = 0
        ci = bootstrap(tilt, idx)
        report['hinge_axis'][NAMES[k]] = dict(n=int(len(idx)), axis_sensor=ax.round(4).tolist(), tilt_deg=round(tilt(idx), 3),
                                              tilt_ci95=[round(c, 3) for c in ci], off_axis_rate_p50_p90=[round(float(np.median(off)), 4), round(float(np.percentile(off, 90)), 4)])
        print(f'  {NAMES[k]:6s} ({JOINTS[k]}): n={len(idx):5d}  axis {ax.round(3)}  {tilt(idx):.2f} deg off nominal '
              f'(95% {ci[0]:.2f}..{ci[1]:.2f}; toward x/y/z {comp.round(2)}), off-axis rate p50 {np.median(off) * 100:.0f}% '
              f'p90 {np.percentile(off, 90) * 100:.0f}%{flag(len(idx), MIN_MOVING)}')

    # 2. relative roll
    print('\n2) Gravity roll relative to the cabin IMU (all still):')
    roll = np.degrees(np.arcsin(np.clip(A[:, :, 1] / np.linalg.norm(A, axis=2), -1, 1)))
    idx = np.flatnonzero(still); report['relative_roll'] = {}
    for k in (1, 2, 3):
        rel = roll[:, k] - roll[:, 0]; ci = bootstrap(lambda i: np.median(rel[i]), idx)
        slope = np.polyfit(np.degrees(Q[idx, k]), rel[idx], 1)[0] * 100 if len(idx) > 10 else float('nan')
        span = np.ptp(np.degrees(Q[idx, k])) if len(idx) else 0
        report['relative_roll'][NAMES[k]] = dict(n=int(len(idx)), median_deg=round(float(np.median(rel[idx])), 3), ci95=[round(c, 3) for c in ci],
                                                 p5_p95=[round(float(np.percentile(rel[idx], q)), 3) for q in (5, 95)],
                                                 slope_deg_per_100deg=round(float(slope), 3), joint_span_deg=round(float(span), 1))
        print(f'  {NAMES[k]:6s} median {np.median(rel[idx]):+.2f} deg (95% {ci[0]:+.2f}..{ci[1]:+.2f}), scatter p5..p95 '
              f'{np.percentile(rel[idx], 5):+.2f}..{np.percentile(rel[idx], 95):+.2f}, slope {slope:+.2f} deg/100 deg over a '
              f'{span:.0f} deg span{flag(len(idx), MIN_STILL)}')

    # 3. extraction gain and latency
    print('\n3) Joint angle rate vs gyro hinge rate (gain 1.00, lag 0 ms is ideal):')
    report['angle_rate'] = {}
    y = np.array([0, 1., 0])
    for k in range(4):
        if k not in axes:
            continue
        # Slew is the cabin IMU's own twist; each pitch joint is child pitch minus parent pitch.
        parent = 0. if k == 0 else G[:, k - 1] @ (axes.get(k - 1, y) if k > 1 else y)
        hinge = G[:, k] @ axes[k] - parent
        # Same +-3 sample window as rate(), so vibration above ~7 Hz is attenuated equally on both sides.
        hinge = np.convolve(hinge, np.r_[.5, np.ones(5), .5] / 6, 'same')
        # Select on the gyro side: selecting on the noisier angle rate biases the gain low
        # (shaking spikes the angle rate without matching hinge motion).
        moving = np.flatnonzero((np.abs(hinge) > 15) & ~edge)
        if len(moving) < 50:
            continue
        lags = np.arange(-20, 21); best = (-2, 0)
        for L in lags:  # angle rate at t vs gyro at t - L samples
            i = moving[(moving - L >= 0) & (moving - L < len(t))]
            c = np.corrcoef(R[i, k], hinge[i - L])[0, 1]
            if abs(c) > best[0]:
                best = (abs(c), L, c)
        L = best[1]; i = moving[(moving - L >= 0) & (moving - L < len(t))]
        gain = float(np.polyfit(hinge[i - L], R[i, k], 1)[0])
        ci = bootstrap(lambda j: np.polyfit(hinge[j - L], R[j, k], 1)[0], i)
        report['angle_rate'][JOINTS[k]] = dict(n=int(len(i)), gain=round(gain, 4), gain_ci95=[round(c, 4) for c in ci],
                                               lag_ms=int(L * 10), correlation=round(float(best[2]), 4))
        print(f'  {JOINTS[k]:6s} n={len(i):5d}  gain {gain:+.3f} (95% {ci[0]:+.3f}..{ci[1]:+.3f}), lag {L * 10:+d} ms, '
              f'corr {best[2]:+.3f}{flag(len(i), MIN_MOVING)}')

    # 4. cabin rocking
    print('\n4) Cabin IMU rocking:')
    report['cabin'] = {}
    for c, name in ((0, 'roll rate (sensor X)'), (1, 'pitch rate (sensor Y)')):
        peaks = []
        for d in data:
            x = d['gyro'][:, 0, c] - d['gyro'][:, 0, c].mean(); f = np.fft.rfftfreq(len(x), np.median(np.diff(d['t'])))
            P = np.abs(np.fft.rfft(x)) ** 2; band = (f > .5) & (f < 25)
            peaks.append(f[band][np.argmax(P[band])])
        x = G[:, 0, c]
        report['cabin'][name] = dict(rms_dps=round(float(x.std()), 3), p99_dps=round(float(np.percentile(np.abs(x - x.mean()), 99)), 2),
                                     peak_hz_per_episode=[round(float(p), 2) for p in peaks])
        print(f'  {name}: rms {x.std():.2f} dps, p99 {np.percentile(np.abs(x - x.mean()), 99):.1f} dps, spectral peak per episode {np.round(peaks, 1).tolist()} Hz')
    g = A[still, 0]; r0 = np.degrees(np.arctan2(g[:, 1], g[:, 2])); p0 = np.degrees(np.arctan2(-g[:, 0], np.hypot(g[:, 1], g[:, 2])))
    report['cabin']['static_roll_pitch_deg'] = dict(roll_p5_p50_p95=np.percentile(r0, [5, 50, 95]).round(2).tolist(),
                                                    pitch_p5_p50_p95=np.percentile(p0, [5, 50, 95]).round(2).tolist(), n=int(still.sum()))
    print(f'  static (still samples): roll p5/p50/p95 {np.percentile(r0, [5, 50, 95]).round(2).tolist()} deg, pitch '
          f'{np.percentile(p0, [5, 50, 95]).round(2).tolist()} deg (sensor axes; includes mounting){flag(int(still.sum()), MIN_STILL)}')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', nargs='+', help='session directories or episode directories')
    p.add_argument('--json', type=Path, help='also write the report as JSON')
    args = p.parse_args()
    report = audit(load(args.paths))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n')
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
