#!/usr/bin/env python3
"""Live sand height next to pi0.5: shared depth frames + control status -> level height map,
litres moved and visual slew (gyro yaw) correction, using sand_height.py's calibration model.

  replay SESSION --episode N    time the per-update work on a recorded take (no hardware)
  live --frames DIR --socket S  same work on live kaivuri-camera depth; status reads only

Never sends commands. Writes updates.jsonl (per-update timings and metrics) and map PNG/NPZ
snapshots to --out.
"""
import argparse
import collections
import json
import math
import signal
from pathlib import Path
import sys
import threading
import time

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import sand_height as sh  # noqa: E402
from client import Client  # noqa: E402
from export_episodes import depth_frames, frame_rows  # noqa: E402
from frame_reader import FrameReader  # noqa: E402
from record_native import CAM1, CAM2  # noqa: E402


class LiveTracker(sh.SlewTracker):
    """Slew tracker whose reference is built on the spot: the first frames (machine still) seed it
    at the gyro heading, and each confidently matched frame fills cells the map has not seen yet.
    Known cells are never overwritten, so the heading frame cannot creep; digging inside the
    seeded area does lower the match, which is what the score reports."""
    def __init__(self, mount, rays, seed_frames, grow_score, half=1.5):
        self.lo = np.array([-half, -half]); self.shape = (int(2 * half / sh.CELL),) * 2
        self.sum = np.zeros(self.shape); self.count = np.zeros(self.shape)
        self.G = np.full(self.shape, np.nan, np.float32)
        self.mount, self.rays, self.delta, self.score = mount, rays, 0., None
        self.seed, self.grow_score, self.seen = seed_frames, grow_score, 0

    def add(self, data, status, delta, only_new):
        P, _ = self.rays.points(data, 4, .9); Pb = self.mount.base(P, status)
        W = sh.near_flat(self.mount.world(Pb[~sh.near_arm(Pb, status, .1)] @ sh.rz(delta).T))
        mean, count = sh.raster(W[:, :2], W[:, 2], self.lo, self.shape)
        new = (count > 0) & ((self.count == 0) if only_new else True)
        self.sum[new] += np.nan_to_num(mean[new]) * count[new]; self.count[new] += count[new]
        k = self.count > 0; self.G[k] = self.sum[k] / self.count[k]
        return int(new.sum())

    def update(self, data, status):
        if self.seen < self.seed:  # seeding: trust the gyro
            self.seen += 1; self.add(data, status, 0., False); self.score = None
            return 0.
        delta = super().update(data, status)
        if self.score is not None and self.score >= self.grow_score:
            self.grown = self.add(data, status, delta, True)
        return delta


class Engine:
    def __init__(self, cal_path, rays, args):
        cal = json.loads(Path(cal_path).read_text())
        if not args.no_tilt:
            sh.load_tilt(cal)
        self.args = args
        self.keys = ['cam2'] + (['cam1'] if args.fusion != 'cabin' else [])
        self.mounts = {k: sh.Mount.from_json(cal['cameras'][k]) for k in self.keys}
        self.rays = rays
        self.tracker = None
        if args.reference == 'live':
            self.tracker = LiveTracker(self.mounts['cam2'], rays['cam2'], args.seed_frames, args.grow_score)
        elif args.reference != 'none':
            ref = np.load(Path(cal_path).parent / cal['reference_map'])
            self.tracker = sh.SlewTracker(ref['height'], ref['lo'], self.mounts['cam2'], rays['cam2'])
        self.fresh = []  # frames collected for a fresh reference
        self.n = -1  # index of the current map update
        poly = np.asarray(cal['sand_polygon_level_xy_m'])
        if args.field == 'none':  # the calibration's field outline needs the calibration day's slew zero
            h = 1.2; poly = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        self.field = sh.FieldMap(poly)

    def _refresh_reference(self, data, status, delta):
        """Collect cabin frames at the corrected slew, then swap them in as the tracker reference
        (current terrain, heading anchored by the stored reference)."""
        m, r = self.mounts['cam2'], self.rays['cam2']
        P, _ = r.points(data, 2, .9); Pb = m.base(P, status); w = m.world(Pb @ sh.rz(delta).T)
        self.fresh.append(w[(np.abs(w[:, 2]) < .08) & ~sh.near_arm(Pb, status, .1)])
        if len(self.fresh) < self.args.fresh_frames:
            return False
        self.tracker.G, self.tracker.lo = sh.height_map(np.concatenate(self.fresh))
        self.fresh = None
        return True

    def update(self, data, status):
        a, t, out = self.args, {}, {}
        c0 = time.perf_counter(); p0 = time.process_time()
        turn = np.eye(3)
        self.n += 1
        if self.tracker is not None and self.n % a.track_every:
            # Between camera fixes: heading = gyro + last correction.
            turn = sh.rz(self.tracker.delta)
            out.update(slew_correction_deg=math.degrees(self.tracker.delta), tracked=False)
        elif self.tracker is not None:
            delta = self.tracker.update(data['cam2'], status); turn = sh.rz(delta)
            out.update(slew_correction_deg=math.degrees(delta), slew_match=self.tracker.score,
                       reference_grown=getattr(self.tracker, 'grown', None), tracked=True)
            self.tracker.grown = None
            t['track_ms'] = (time.perf_counter() - c0) * 1e3
            if (a.reference == 'fresh' and self.fresh is not None and self.tracker.score is not None
                    and self.tracker.score >= .5):
                c1 = time.perf_counter()
                out['reference_refreshed'] = self._refresh_reference(data['cam2'], status, delta)
                t['refresh_ms'] = (time.perf_counter() - c1) * 1e3
        field = self.field
        field.begin()
        for k in self.keys:
            c1 = time.perf_counter()
            W, good, _ = field.points(self.mounts[k], self.rays[k], data[k], status, turn, a)
            field.add(k, W[good], a.fusion)
            t[f'{k}_ms'] = (time.perf_counter() - c1) * 1e3
            out[f'{k}_points'] = int(good.sum())
        c1 = time.perf_counter()
        field.commit()
        out.update(**field.metrics(a.deadband_mm), field_seen=float(np.isfinite(field.fused)[field.in_field].mean()))
        t['metrics_ms'] = (time.perf_counter() - c1) * 1e3
        t['total_ms'] = (time.perf_counter() - c0) * 1e3
        t['cpu_ms'] = (time.process_time() - p0) * 1e3
        out['timing'] = t
        return out

    def snapshot(self, path):
        field = self.field
        np.savez_compressed(path.with_suffix('.npz'), fused=field.fused, change=field.change, lo=field.lo,
                            in_field=field.in_field, poly=field.poly)
        try:
            from PIL import Image
            lim = self.args.color_mm / 1000
            a, b = field.image(field.fused, lim), field.image(field.change, lim)
            gap = np.zeros((a.shape[0], 8, 3), np.uint8)
            parts = [a, gap, b]
            if isinstance(self.tracker, LiveTracker):  # reference map, same cells as the field when --field none
                g = self.tracker.G
                if self.args.field == 'none' and g.shape[0] >= a.shape[0]:
                    o = (g.shape[0] - a.shape[0]) // 2; g = g[o:o + a.shape[1], o:o + a.shape[0]]
                if g.T.shape[:2] == a.shape[:2]:
                    parts += [gap, sh.diverging(g.T[::-1], lim)]
                np.savez_compressed(path.with_name(path.stem + '-reference.npz'), height=self.tracker.G, lo=self.tracker.lo)
            img = Image.fromarray(np.concatenate(parts, 1))
            img.resize((img.width * 2, img.height * 2), Image.NEAREST).save(path.with_suffix('.png'))
        except ImportError:
            pass


def common(p):
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fusion', choices=('cabin', 'fill', 'average'), default='cabin')
    p.add_argument('--reference', choices=('stored', 'fresh', 'live', 'none'), default='stored',
                   help='slew tracker reference: calibration map; fresh = re-taken once the stored map locks; '
                        'live = seeded at start (machine still) and grown on confident matches')
    p.add_argument('--fresh-frames', type=int, default=4)
    p.add_argument('--track-every', type=int, default=1, help='run the slew tracker on every Nth map update')
    p.add_argument('--seed-frames', type=int, default=4, help='live reference: frames at start, gyro trusted')
    p.add_argument('--grow-score', type=float, default=.7, help='live reference: match needed to add new cells')
    p.add_argument('--field', choices=('calibration', 'none'), default='calibration',
                   help='none: ignore the calibration sand outline (its heading needs the same slew zero)')
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--max-range-m', type=float, default=.8)
    p.add_argument('--above-m', type=float, default=.12)
    p.add_argument('--below-m', type=float, default=.15)
    p.add_argument('--arm-radius-m', type=float, default=.11)
    p.add_argument('--deadband-mm', type=float, default=8)
    p.add_argument('--color-mm', type=float, default=50)
    p.add_argument('--no-tilt', action='store_true')
    p.add_argument('--snapshot-s', type=float, default=5, help='map PNG/NPZ every N seconds (0: end only)')


def replay(args):
    session = args.session
    rays = {k: sh.Rays(session, k) for k in ('cam1', 'cam2')}
    engine = Engine(args.calibration, rays, args)
    episode = sh.episode_path(session, args.episode)
    rows = list(frame_rows(episode))
    streams = {k: depth_frames(episode / f'{k}.depth.zst') for k in engine.keys}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'updates.jsonl').open('w') as log:
        for i, row in enumerate(rows):
            data = {k: next(streams[k]) for k in engine.keys}
            if i % args.every or not row['status'].get('fk_valid') or (args.limit and i // args.every >= args.limit):
                continue
            r = engine.update(data, row['status'])
            r.update(frame=i, slew_gyro_deg=math.degrees(row['status']['angles_rad'][0]))
            log.write(json.dumps(r) + '\n')
    engine.snapshot(args.out / 'final')
    report(args.out)


class StatusRing(threading.Thread):
    """Poll control status at a fixed rate; keep the recent ones for time pairing with depth."""
    def __init__(self, socket_path, hz):
        super().__init__(daemon=True, name='status')
        self.client, self.period = Client(socket_path), 1 / hz
        self.ring, self.lock, self.stop, self.error = collections.deque(maxlen=int(hz * 2)), threading.Lock(), threading.Event(), None

    def run(self):
        try:
            nxt = time.monotonic()
            while not self.stop.is_set():
                s = self.client.status()
                with self.lock:
                    self.ring.append(s)
                nxt += self.period
                if nxt < time.monotonic():
                    nxt = time.monotonic()
                self.stop.wait(max(0, nxt - time.monotonic()))
        except BaseException as e:  # noqa: BLE001
            self.error = repr(e)

    def nearest(self, t_ns):
        with self.lock:
            ring = list(self.ring)
        if not ring:
            return None
        return min(ring, key=lambda s: abs(s.get('imu_host_monotonic_ns', s['host_monotonic_ns']) - t_ns))


def live(args):
    frames = args.frames
    serial = {'cam1': args.cam1, 'cam2': args.cam2}
    deadline = time.monotonic() + 20
    while not all((frames / f'{serial[k]}-depth.json').exists() for k in ('cam1', 'cam2')):
        if time.monotonic() > deadline:
            raise SystemExit('depth JSON not found in ' + str(frames))
        time.sleep(.1)
    rays = {k: sh.Rays(frames / f'{serial[k]}-depth.json') for k in ('cam1', 'cam2')}
    engine = Engine(args.calibration, rays, args)
    lib = args.library
    readers = {k: FrameReader(frames / f'{serial[k]}-depth.frame', lib, capacity=640 * 480 * 2, max_age_s=.5)
               for k in engine.keys}
    ring = StatusRing(args.socket, args.status_hz); ring.start()
    args.out.mkdir(parents=True, exist_ok=True)
    period = 1 / args.hz if args.hz > 0 else 0
    end = time.monotonic() + args.duration_s
    nxt = time.monotonic(); last_snap = time.monotonic(); n = 0; skipped = collections.Counter()
    stop = threading.Event(); dumps = {}
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    with (args.out / 'updates.jsonl').open('w') as log:
        while time.monotonic() < end and not stop.is_set():
            if ring.error:
                raise SystemExit('status reader failed: ' + ring.error)
            if period:
                wait = nxt - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                nxt += period
                if nxt < time.monotonic() - period:
                    nxt = time.monotonic()
            c0 = time.perf_counter()
            data, meta = {}, {}
            for k, rd in readers.items():
                snap = rd.read()
                if snap is None:
                    break
                meta[k], data[k] = snap[0], bytes(snap[1])  # copy: the view is reused by the next read
            if len(data) != len(readers):
                skipped['stale_depth'] += 1; continue
            t_ns = meta['cam2'].host_monotonic_ns - int(args.depth_latency_ms * 1e6)
            status = ring.nearest(t_ns)
            if status is None or not status.get('fk_valid'):
                skipped['no_fk'] += 1; continue
            read_ms = (time.perf_counter() - c0) * 1e3
            r = engine.update(data, status)
            now = time.monotonic_ns()
            r['timing']['read_ms'] = read_ms
            r.update(t_ns=now, depth_age_ms=(now - meta['cam2'].host_monotonic_ns) / 1e6,
                     status_offset_ms=(status.get('imu_host_monotonic_ns', status['host_monotonic_ns']) - t_ns) / 1e6,
                     depth_seq=meta['cam2'].sequence, slew_gyro_deg=math.degrees(status['angles_rad'][0]),
                     state=status['state'])
            log.write(json.dumps(r) + '\n'); log.flush(); n += 1
            if args.dump:
                if not dumps:  # open once
                    dumps = {k: (args.out / f'{k}.depth.raw').open('wb') for k in engine.keys}
                    dumps['status'] = (args.out / 'status.jsonl').open('w')
                for k in engine.keys:
                    dumps[k].write(data[k])
                dumps['status'].write(json.dumps(status) + '\n')
            if args.snapshot_s and time.monotonic() - last_snap >= args.snapshot_s:
                engine.snapshot(args.out / 'latest'); last_snap = time.monotonic()
    ring.stop.set()
    for f in dumps.values():
        f.close()
    engine.snapshot(args.out / 'final')
    (args.out / 'skipped.json').write_text(json.dumps(skipped) + '\n')
    report(args.out)


def report(out):
    rows = [json.loads(l) for l in (out / 'updates.jsonl').read_text().splitlines()]
    if not rows:
        print('no updates'); return
    def dist(v):
        v = np.asarray(v, float)
        return {'n': len(v), 'p50': round(float(np.percentile(v, 50)), 2), 'p90': round(float(np.percentile(v, 90)), 2),
                'p99': round(float(np.percentile(v, 99)), 2), 'max': round(float(v.max()), 2)}
    keys = sorted({k for r in rows for k in r['timing']})
    rep = {k: dist([r['timing'][k] for r in rows if k in r['timing']]) for k in keys}
    if 't_ns' in rows[0] and len(rows) > 1:
        rep['achieved_hz'] = round((len(rows) - 1) / ((rows[-1]['t_ns'] - rows[0]['t_ns']) / 1e9), 3)
        rep['depth_age_ms'] = dist([r['depth_age_ms'] for r in rows])
        rep['status_offset_ms'] = dist([r['status_offset_ms'] for r in rows])
    tracked = [r for r in rows if r.get('slew_match') is not None]
    if tracked:
        rep['slew_match'] = dist([r['slew_match'] for r in tracked])
        rep['slew_correction_deg'] = dist([r['slew_correction_deg'] for r in tracked])
        rep['tracker_updates'] = len(tracked)
    rep['last'] = {k: rows[-1][k] for k in ('dug_l', 'piled_l', 'hole_bottom_mm', 'field_seen') if k in rows[-1]}
    (out / 'report.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('replay'); r.add_argument('session', type=Path); common(r)
    r.add_argument('--episode', type=int, default=1); r.add_argument('--every', type=int, default=15)
    r.add_argument('--limit', type=int, default=0)
    v = sub.add_parser('live'); common(v)
    v.add_argument('--frames', type=Path, required=True); v.add_argument('--socket', required=True)
    v.add_argument('--library', default=None)
    v.add_argument('--cam1', default=CAM1); v.add_argument('--cam2', default=CAM2)
    v.add_argument('--hz', type=float, default=2, help='update rate cap (0: as fast as possible)')
    v.add_argument('--status-hz', type=float, default=50)
    v.add_argument('--depth-latency-ms', type=float, default=0, help='pair status this long before depth arrival')
    v.add_argument('--duration-s', type=float, default=120)
    v.add_argument('--dump', action='store_true', help='also save the depth frames and status used (for replay)')
    args = p.parse_args()
    (replay if args.command == 'replay' else live)(args)


if __name__ == '__main__':
    main()
