#!/usr/bin/env python3
"""Full-stack test: dry/disarmed control + dual IR (+ depth) camera + pi0.5 at 2 Hz + live sand height.

Nothing drives hardware: kaivuri-control runs dry (no I2C), stays disarmed (checked every 0.5 s),
and kaivuri-inference runs without --drive. Live sand height starts once inference is READY so
both share the same measured window. Reuses bench_pi05.py's summary; adds the sand process's CPU/RSS.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
PROJECT = SOURCE.parent
import bench_pi05 as bench  # noqa: E402
from client import Client  # noqa: E402

CAM1, CAM2 = '233522073919', '141722076647'  # record_native.py defaults: cam1 (wrist, upside down), cam2 (cabin)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--build', type=Path, default=SOURCE / 'build')
    p.add_argument('--profile', type=Path, default=PROJECT / 'configuration_files/profiles/jetson')
    p.add_argument('--safety', type=Path, default=SOURCE / 'config/safety.yaml')
    p.add_argument('--bundle', type=Path, required=True, help='pi05_openpi_state bundle')
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True, help='prepare_pi05.py output')
    p.add_argument('--task', default='scoop sand and dump it to the left')
    p.add_argument('--duration-s', type=float, default=120)
    p.add_argument('--rate-hz', type=float, default=2)
    p.add_argument('--no-depth', action='store_true', help='IR only (baseline)')
    p.add_argument('--sand-hz', type=float, default=-1, help='-1: no sand process; 0: as fast as possible')
    p.add_argument('--sand-args', default='', help='extra live_sand.py arguments')
    p.add_argument('--calibration', type=Path, help='sand_height.py calibrate output (needed with --sand-hz)')
    a = p.parse_args()
    if a.sand_hz >= 0 and not a.calibration:
        p.error('--sand-hz needs --calibration')
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    build = a.build.resolve()
    sock = f'/tmp/kaivuri-native-{os.geteuid()}/fullstack.sock'
    frames = out / 'frames'
    children, logs = {}, []

    def start(name, cmd, env=None):
        log = (out / f'{name}.log').open('w'); logs.append(log)
        children[name] = subprocess.Popen([str(c) for c in cmd], cwd=SOURCE, stdout=log,
                                          stderr=subprocess.STDOUT, env=env)
        return children[name]

    control = [build / 'kaivuri-control', '--profile', a.profile, '--safety', a.safety, '--socket', sock, '--control-hz', 200,
               '--duration-s', a.duration_s + 600, '--timing-csv', out / 'control-timing.csv',
               '--serial', '/dev/ttyACM0', '--enable-slew']  # dry PWM: no --live-pwm
    camera = [build / 'kaivuri-camera', '--output-dir', frames, '--metadata-csv', out / 'camera-timestamps.csv',
              '--camera', f'{CAM1}:2', '--camera', f'{CAM2}:1', '--rotate-180', CAM1]
    if not a.no_depth:
        camera += ['--depth', CAM1, '--depth', CAM2]
    ir = [frames / f'{CAM1}-ir2.frame', frames / f'{CAM2}-ir1.frame']
    infer = [build / 'kaivuri-inference', '--bundle', a.bundle, '--cache', a.cache, '--fixtures', a.fixtures,
             '--control-socket', sock, '--task', a.task, '--duration-s', a.duration_s, '--warmup', 5,
             '--out', out / 'inference.json', '--rate-hz', a.rate_hz,
             '--parity-cosine', 0.9998, '--parity-max-abs', 0.02, '--camera-frame', ir[0], '--camera-frame', ir[1]]
    samples, client, ok = [], None, False
    try:
        start('control', control); start('camera', camera)
        deadline = time.monotonic() + 20
        while not (all(f.exists() for f in ir) and Path(sock).exists()):
            for n, c in children.items():
                if c.poll() is not None:
                    raise RuntimeError(f'{n} exited during startup; see {n}.log')
            if time.monotonic() > deadline:
                raise TimeoutError('camera/control startup')
            time.sleep(.1)
        client = Client(sock)
        s = client.status()
        if s['state'] != 'disarmed' or s['live_pwm']:
            raise RuntimeError(f'expected dry disarmed control, got {s["state"]} live_pwm={s["live_pwm"]}')
        inference = start('inference', infer)
        if shutil.which('tegrastats'):
            start('tegrastats', ['tegrastats', '--interval', '500'])
        tele = (out / 'telemetry.jsonl').open('w'); logs.append(tele)
        sand_started = False
        while inference.poll() is None:
            if not sand_started and a.sand_hz >= 0 and '[bench] READY' in (out / 'inference.log').read_text():
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
                start('sand', [sys.executable, '-B', Path(__file__).resolve().parent / 'sand_height_live.py', 'live', '--frames', frames, '--socket', sock,
                               '--library', build / 'libkaivuri_frames.so', '--calibration', a.calibration,
                               '--out', out / 'sand', '--hz', a.sand_hz, '--duration-s', a.duration_s + 30,
                               *a.sand_args.split()], env=env)
                sand_started = True
            rec = {'monotonic_ns': time.monotonic_ns(), 'system': bench.system_snapshot(), 'processes': {}}
            for n in ('control', 'camera', 'inference', 'sand'):
                if n in children and children[n].poll() is None:
                    try:
                        rec['processes'][n] = bench.process_snapshot(children[n].pid)
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                elif n in ('control', 'camera'):
                    raise RuntimeError(f'{n} exited; see {n}.log')
            s = client.status(); rec['control_status'] = s
            if s['state'] != 'disarmed' or s['live_pwm'] or s['pump_on']:
                raise RuntimeError('control left dry/disarmed/pump-off')
            samples.append(rec); tele.write(json.dumps(rec) + '\n'); tele.flush()
            time.sleep(.5)
        if inference.returncode:
            raise RuntimeError(f'inference failed ({inference.returncode}); see inference.log')
        ok = True
    finally:
        if client:
            client.close()
        for c in children.values():
            if c.poll() is None:
                c.terminate()
        for c in children.values():
            try:
                c.wait(timeout=15)
            except subprocess.TimeoutExpired:
                c.kill(); c.wait()
        for f in logs:
            f.close()
        (out / 'session.json').write_text(json.dumps({'args': {k: str(v) for k, v in vars(a).items()},
            'control': [str(c) for c in control], 'camera': [str(c) for c in camera], 'inference': [str(c) for c in infer],
            'exit_codes': {n: c.returncode for n, c in children.items()}, 'completed': ok}, indent=1) + '\n')
    rep = bench.summarize(out, samples)
    win = [s for s in samples if 'sand' in s['processes']]
    if len(win) > 1:
        el = (win[-1]['monotonic_ns'] - win[0]['monotonic_ns']) / 1e9
        rep['sand_process'] = {'rss_mib_max': max(s['processes']['sand']['rss_mib'] for s in win),
                               'cpu_percent_one_core': 100 * (win[-1]['processes']['sand']['cpu_seconds'] - win[0]['processes']['sand']['cpu_seconds']) / el}
        (out / 'summary.json').write_text(json.dumps(rep, indent=2) + '\n')
    m = rep['metrics']
    brief = {'inference_ms': {k: round(m['inference_ms'][k], 1) for k in ('p50', 'p99', 'max')},
             'rate_overruns': rep['inference'].get('rate_overruns'),
             'control_lateness_ms_p99': round(m['control_lateness_ms']['p99'], 3),
             'system_used_mib_max': round(rep['system_used_mib']['max']),
             'cpu_percent_one_core': {n: round(v['cpu_percent_one_core'], 1) for n, v in rep['processes'].items()},
             'sand_process': rep.get('sand_process'),
             'camera_gaps': {k: v['device_counter_gaps'] for k, v in rep['cameras'].items()}}
    print(json.dumps(brief))


if __name__ == '__main__':
    main()
