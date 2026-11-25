#!/usr/bin/env python3
"""Run dual IR, dry control, optional FastDEM heightmaps and optional shadow π0.5."""
import argparse
from bisect import bisect_left
import csv
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
import time

import numpy as np

from bench_pi05 import distribution, summarize_camera
from client import stop
from frame_reader import FrameReader
from mapper import (ANNULUS, KEYS, DepthProjector, FastDEM, pipeline_defaults, pose_at,
                    projection_arguments, render, track_pivot, update)
from measure import process_snapshot, system_snapshot
from record_native import CAM1, CAM2
from sand_height_live import StatusRing
import sand_height as sh


def wait_for(predicate, children, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check_processes(children)
        if predicate():
            return
        time.sleep(.05)
    raise TimeoutError('startup timed out; see process logs')


def check_processes(children):
    for name, process in children.items():
        if process.poll() is not None:
            raise RuntimeError(f'{name} exited ({process.returncode}); see {name}.log')


def check_status(ring):
    if ring.error:
        raise RuntimeError(ring.error)
    with ring.lock:
        recent = list(ring.ring)
    if not recent:
        return []
    s = recent[-1]
    if s['state'] != 'disarmed' or s['live_pwm'] or s['pump_on']:
        raise RuntimeError('expected dry, disarmed, pump-off control')
    if time.monotonic_ns() - s['host_monotonic_ns'] > 500_000_000:
        raise RuntimeError('control status stalled')
    return [p for p in recent if p['fk_valid'] and p['imu_healthy'] and p['fk_cabin_tilt']]


def observe(readers, previous, poses, max_age_ms, pivot):
    snapshots = [reader.read() for reader in readers]
    if any(s is None for s in snapshots):
        return 'stale_depth'
    if any(s[0].sequence <= seq for s, seq in zip(snapshots, previous)):
        return 'duplicate_depth'
    if not poses:
        return 'no_healthy_fk'
    timestamps = [p['host_monotonic_ns'] for p in poses]
    matched = []
    for meta, data in snapshots:
        stamp = meta.host_monotonic_ns
        after = bisect_left(timestamps, stamp)
        neighbours = timestamps[max(0, after - 1):after + 1]
        if any(abs(t - stamp) > max_age_ms * 1e6 for t in neighbours):
            return 'stale_fk'
        pose = pose_at(stamp, timestamps, poses, pivot)
        matched.append((meta, data, {**pose, 'fk_source_ns': neighbours}))
    return matched


def run(args, stack, children):
    out, native = args.out.resolve(), args.native.resolve()
    out.mkdir(parents=True)
    (out / 'settings.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    build = native / 'build'
    profile = native.parent / 'configuration_files/profiles/jetson'
    work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='hm-bench-', dir='/dev/shm')))
    frames, sock = work / 'frames', work / 'control.sock'
    serials = {'cam2': CAM2, 'cam1': CAM1}
    keys = KEYS if args.mode == 'fused' else KEYS[:1]
    ir = [frames / f'{CAM1}-ir2.frame', frames / f'{CAM2}-ir1.frame']

    def launch(name, *cmd):
        log = stack.enter_context((out / f'{name}.log').open('w'))
        children[name] = subprocess.Popen([str(x) for x in cmd], cwd=native, stdout=log,
                                          stderr=subprocess.STDOUT)

    # The isolated controller runs dry. Inference has no actuator output path.
    launch('control', build / 'kaivuri-control', '--profile', profile, '--safety',
           native / 'config/safety.yaml', '--socket', sock, '--serial', args.serial,
           '--enable-slew', '--control-hz', 200, '--duration-s', args.seconds + 300,
           '--timing-csv', out / 'control-timing.csv')
    depth_flags = [flag for serial in serials.values() for flag in ('--depth', serial)]
    launch('camera', build / 'kaivuri-camera', '--output-dir', frames, '--fps', args.ir_fps,
           '--metadata-csv', out / 'camera-timestamps.csv', '--camera', f'{CAM1}:2',
           '--camera', f'{CAM2}:1', '--rotate-180', CAM1, *depth_flags,
           '--depth-units-um', args.depth_units_um,
           '--disparity-shift', f'{CAM1}:{args.arm_disparity_shift}',
           '--disparity-shift', f'{CAM2}:{args.cabin_disparity_shift}')
    wait_for(lambda: sock.exists() and all(p.exists() for p in ir), children, 25)
    ring = StatusRing(str(sock), 100)
    ring.start()

    def close_status():
        ring.stop.set()
        ring.join(timeout=2)
        ring.client.close()

    stack.callback(close_status)
    wait_for(lambda: bool(check_status(ring)), children, 15)

    if args.inference:
        launch('inference', args.inference, '--bundle', args.bundle, '--cache', args.cache,
               '--fixtures', args.fixtures, '--task', args.task, '--control-socket', sock,
               '--camera-frame', ir[0], '--camera-frame', ir[1], '--rate-hz', args.policy_hz,
               '--duration-s', args.seconds + 10, '--out', out / 'inference.json',
               '--parity-cosine', .9998, '--parity-max-abs', .02)
        wait_for(lambda: '[bench] READY' in (out / 'inference.log').read_text(), children, 160)

    if shutil.which('tegrastats'):
        launch('tegrastats', 'tegrastats', '--interval', 1000)

    if args.mode != 'off':
        sh.load_arm(args)
        cal = json.loads(args.calibration.read_text())
        pivot = track_pivot(cal, profile / 'control_config.yaml')
        camera_files, readers = {}, []
        for key in keys:
            serial = serials[key]
            intr = frames / f'{serial}-depth.json'
            path = frames / f'{serial}-depth.frame'
            wait_for(lambda: intr.exists() and path.exists(), children, 10)
            camera_files[key] = intr
            readers.append(stack.enter_context(FrameReader(path, build / 'libkaivuri_frames.so',
                                                            capacity=640 * 480 * 2, max_age_s=.1)))
        projector = DepthProjector(cal, camera_files, args)
        (out / 'depth-profiles.json').write_text(json.dumps(projector.metadata, indent=2) + '\n')
        engine = stack.enter_context(FastDEM(args.fastdem, args.config))

    period = round(1e9 / args.map_hz)
    start = nxt = time.monotonic_ns()
    end = start + round(args.seconds * 1e9)
    skipped, rows, resources = Counter(), [], []
    previous = [-1] * len(keys)
    latest = None
    sample_after = start
    with ((out / 'updates.jsonl').open('w', buffering=1) as log,
          (out / 'skips.jsonl').open('w', buffering=1) as skip_log):
        while nxt < end:
            time.sleep(max(0, (nxt - time.monotonic_ns()) / 1e9))
            tick = time.monotonic_ns()
            missed = max(0, (tick - nxt) // period)
            skipped['overrun_slots'] += missed
            nxt += (1 + missed) * period
            if tick >= end:
                break
            check_processes(children)
            poses = check_status(ring)
            if tick >= sample_after:
                processes = {k: p.pid for k, p in children.items()}
                processes['mapper_python'] = os.getpid()
                if args.mode != 'off':
                    processes['fastdem'] = engine.proc.pid
                resources.append({'ns': tick, 'system': system_snapshot(),
                                  'processes': {k: process_snapshot(pid) for k, pid in processes.items()}})
                sample_after = tick + 1_000_000_000
            if args.mode == 'off':
                continue
            before = time.monotonic_ns()
            obs = observe(readers, previous, poses, args.max_fk_age_ms, pivot)
            if isinstance(obs, str):
                skipped[obs] += 1
                skip_log.write(json.dumps({'tick_ns': tick, 'reason': obs}) + '\n')
                continue
            previous = [meta.sequence for meta, _, _ in obs]
            scans = [projector.scan(key, data, pose) for key, (_, data, pose) in zip(keys, obs)]
            projected = time.monotonic_ns()
            front, surround, age = update(engine, scans, obs[0][2]['slew_yaw_rad'])
            latest = front, surround, age
            finish = time.monotonic_ns()
            stamps = [meta.host_monotonic_ns for meta, _, _ in obs]
            row = dict(finish_ns=finish, sequence=previous, compute_ms=(finish-before)/1e6,
                       cameras={key: {'depth_host_ns': meta.host_monotonic_ns,
                                      'fk_source_ns': pose['fk_source_ns'],
                                      'slew_deg': float(np.rad2deg(pose['slew_yaw_rad'])),
                                      'joint_deg': np.rad2deg(pose['angles_rad']).tolist(),
                                      'cabin_pitch_deg': float(np.rad2deg(pose['cabin_pitch_rad'])),
                                      'cabin_roll_deg': float(np.rad2deg(pose['cabin_roll_rad']))}
                                for key, (meta, _, pose) in zip(keys, obs)},
                       projection_ms=(projected-before)/1e6, map_ms=(finish-projected)/1e6,
                       oldest_depth_ms=(finish-min(stamps))/1e6, skew_ms=(max(stamps)-min(stamps))/1e6,
                       arm_rejected_pct={key: 100 * np.count_nonzero(mask == 2) / max(1, np.count_nonzero(mask))
                                         for key, (_, _, _, _, mask) in zip(keys, scans)},
                       front_coverage=float(np.isfinite(front).mean()),
                       surround_coverage=float(np.isfinite(surround).sum()/ANNULUS.sum()))
            log.write(json.dumps(row) + '\n')
            rows.append(row)
    if latest is not None:
        front, surround, age = latest
        diagnostics = {}
        for key, (_, _, _, depth, mask) in zip(keys, scans):
            diagnostics[f'{key}_depth_m'] = depth
            diagnostics[f'{key}_arm_mask'] = mask
        np.savez_compressed(out / 'latest.npz', front_m=front, surround_m=surround,
                            surround_age_updates=age, **diagnostics)
        from PIL import Image
        for name, heights in [('front', front), ('surround', surround)]:
            Image.fromarray(render(heights)).save(out / f'{name}.png')

    # Let native inference finish normally so its timing CSV and report are flushed.
    if args.inference and children['inference'].wait(timeout=35):
        raise RuntimeError('inference failed; see inference.log')
    return start, end, rows, skipped, resources


def resource_summary(samples):
    if len(samples) < 2:
        return {}
    elapsed = (samples[-1]['ns'] - samples[0]['ns']) / 1e9
    return {
        name: {
            'rss_mib_peak': max(s['processes'][name]['rss_mib'] for s in samples),
            'cpu_percent_one_core': 100 * (
                samples[-1]['processes'][name]['cpu_seconds'] -
                samples[0]['processes'][name]['cpu_seconds']) / elapsed,
        }
        for name in samples[0]['processes']
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('native', 'out', 'calibration', 'fastdem', 'config', 'inference', 'bundle', 'cache', 'fixtures'):
        p.add_argument(f'--{option}', type=lambda s: Path(s).expanduser().resolve(),
                       required=option in ('native', 'out'))
    p.add_argument('--mode', choices=('off', 'cabin', 'fused'), default='fused')
    p.add_argument('--serial', default='/dev/ttyACM0')
    p.add_argument('--depth-units-um', type=int, choices=(100, 1000), default=100)
    p.add_argument('--arm-disparity-shift', type=int, default=25)
    p.add_argument('--cabin-disparity-shift', type=int, default=0)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--map-hz', type=float, default=5)
    p.add_argument('--policy-hz', type=float, default=2)
    p.add_argument('--task', default='scoop the sand and dump it into the container')
    p.add_argument('--ir-fps', type=int, choices=(30,), help='native camera profile')
    p.add_argument('--max-fk-age-ms', type=float)
    projection_arguments(p)
    pipeline_defaults(p, ("capture", "runtime", "projection"))
    a = p.parse_args()
    if not all(np.isfinite(x) for x in (a.seconds, a.map_hz, a.policy_hz)):
        p.error('duration and rates must be finite')
    if any(not 0 <= shift <= 255 for shift in (a.arm_disparity_shift, a.cabin_disparity_shift)):
        p.error('disparity shifts must be between 0 and 255')
    if not np.isfinite(a.arm_radius_m) or a.arm_radius_m < 0:
        p.error('--arm-radius-m must be finite and nonnegative')
    if not np.isfinite(a.max_fk_age_ms) or a.max_fk_age_ms <= 0:
        p.error("--max-fk-age-ms must be finite and positive")
    if a.ir_fps != 30 or a.depth_units_um not in (100, 1000):
        p.error("unsupported camera profile")
    if a.seconds <= 0 or not 0 < a.map_hz <= 30 or a.policy_hz < 0:
        p.error('invalid duration or rate')
    if a.mode != 'off' and not all((a.calibration, a.fastdem, a.config)):
        p.error('mapping requires --calibration, --fastdem and --config')
    if a.inference and not all((a.bundle, a.cache, a.fixtures)):
        p.error('inference requires --bundle, --cache and --fixtures')
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGINT))
    children = {}
    with ExitStack() as stack:
        # Register after run's resources via finally, to stop native writers before unmapping.
        try:
            start, end, rows, skipped, resources = run(a, stack, children)
        finally:
            stop(list(children.values()))
    cameras = summarize_camera(a.out / 'camera-timestamps.csv', (start, end))
    in_window = [r for r in rows if r['finish_ns'] <= end]
    map_rate = len(in_window) * 1e9 / (end - start)
    finishes = [start, *(r['finish_ns'] for r in in_window), end]
    longest_gap_ms = max(b - a for a, b in zip(finishes, finishes[1:])) / 1e6
    with (a.out / 'control-timing.csv').open() as f:
        control = [r for r in csv.DictReader(f) if start <= int(r['start_ns']) <= end]
    report = {'mode': a.mode, 'seconds': a.seconds, 'target_map_hz': a.map_hz,
              'map_completed_hz': map_rate,
              'longest_map_gap_ms': longest_gap_ms if a.mode != 'off' else None,
              'skipped': dict(skipped), 'cameras': cameras,
              'ir_near_30_both': len(cameras) == 2 and all(c['observed_fps'] >= 29.5 for c in cameras.values()),
              'map_at_least_5_hz': map_rate >= 5 if a.mode != 'off' else None,
              'control_completed_hz': len(control) / a.seconds,
              'control_lateness_ms': distribution([float(r['lateness_ms']) for r in control]),
              'system_used_mib': distribution([r['system']['used_mib'] for r in resources]),
              'processes': resource_summary(resources),
              'timing': {k: distribution([r[k] for r in in_window]) for k in
                         ('compute_ms', 'projection_ms', 'map_ms', 'oldest_depth_ms', 'skew_ms')}}
    if a.inference:
        with (a.out / 'inference.csv').open() as f:
            policy = [r for r in csv.DictReader(f) if start <= int(r['start_ns']) and int(r['finish_ns']) <= end]
        report['policy_completed_hz'] = len(policy) / a.seconds
        report['policy_inference_ms'] = distribution([float(r['inference_ms']) for r in policy])
        report['policy_target_hz'] = a.policy_hz
    (a.out / 'resources.json').write_text(json.dumps(resources, indent=2) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
