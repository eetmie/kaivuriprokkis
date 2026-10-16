"""Launch native controller, dual-IR camera and recorder; no Python capture loop.

A=start/save, B=discard, X=pump, Y=centered arm/recover. Export is a separate
post-session step, after controller shutdown. Defaults never open physical PWM.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[1]
PROJECT = SOURCE.parent


def wait_for(path, process, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'{process.args[0]} exited during startup: {process.returncode}')
        if path.exists():
            return
        time.sleep(.02)
    raise TimeoutError(f'timed out waiting for {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True, help='new raw session directory')
    p.add_argument('--task', required=True)
    p.add_argument('--build', type=Path, default=SOURCE / 'build')
    p.add_argument('--profile', type=Path, default=PROJECT / 'configuration_files/profiles/jetson')
    p.add_argument('--safety', type=Path, default=SOURCE / 'config/safety.yaml')
    p.add_argument('--gamepad', help='Xbox evdev path, e.g. /dev/input/by-id/...-event-joystick')
    p.add_argument('--camera1', default='233522073919:2')
    p.add_argument('--camera2', default='141722076647:1')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--control-hz', type=int, default=200)
    p.add_argument('--max-episode-s', type=float, default=180)
    p.add_argument('--max-pair-skew-ms', type=float, default=40)
    p.add_argument('--live-view', action='store_true', help='optional separate Rerun preview; off by default')
    p.add_argument('--rotate-camera1', action=argparse.BooleanOptionalAction, default=True,
                   help='rotate camera 1 capture by 180 degrees for its upside-down mount')
    p.add_argument('--view-rotate-camera1', action=argparse.BooleanOptionalAction, default=False,
                   help='camera 1 preview rotation for the upside-down mount; raw recording is unchanged')
    p.add_argument('--view-fps', type=float, default=10, help='preview rate cap; recording FPS is unchanged')
    p.add_argument('--view-python', type=Path, default=PROJECT / '.venv-rerun/bin/python')
    p.add_argument('--serial', default='/dev/ttyACM0')
    p.add_argument('--live-pwm', action='store_true')
    p.add_argument('--enable-slew', action='store_true')
    p.add_argument('--enable-tracks', action='store_true')
    p.add_argument('--exposure-us', type=float)
    p.add_argument('--gain', type=float)
    p.add_argument('--synthetic', action='store_true', help='simulated cameras and IMUs, dry only')
    p.add_argument('--bench-seconds', type=float, help='automatic zero-command recording, dry only')
    args = p.parse_args()
    if args.session.exists():
        p.error('--session must be a new directory')
    if args.live_pwm and (args.synthetic or args.bench_seconds is not None):
        p.error('synthetic/automatic bench recording cannot use live PWM')
    if not args.bench_seconds and not args.gamepad:
        p.error('--gamepad required outside --bench-seconds mode')
    if args.bench_seconds is not None and args.bench_seconds <= 0:
        p.error('--bench-seconds must be positive')
    if args.synthetic and not args.bench_seconds:
        p.error('--synthetic requires --bench-seconds')
    specs = [args.camera1, args.camera2]
    try:
        parsed = [spec.split(':') for spec in specs]
        if any(len(s) != 2 or not s[0].replace('-', '').replace('_', '').isalnum() or s[1] not in ('1', '2') for s in parsed):
            raise ValueError()
        if parsed[0][0] == parsed[1][0]:
            raise ValueError()
    except ValueError:
        p.error('select two different physical cameras as SERIAL:IR_INDEX')
    args.session = args.session.resolve()
    args.session.parent.mkdir(parents=True, exist_ok=True)
    # Capture the exact calibration and source bytes before launching the services.
    snapshots = {str(path.relative_to(PROJECT)): path.read_bytes() for path in args.profile.rglob('*') if path.is_file() and path.is_relative_to(PROJECT)}
    if not args.profile.is_relative_to(PROJECT):
        snapshots.update({f'profile/{path.relative_to(args.profile)}': path.read_bytes() for path in args.profile.rglob('*') if path.is_file()})
    snapshots['safety.yaml'] = args.safety.read_bytes()
    for name in ('src/core.cpp', 'src/control_main.cpp', 'src/camera_main.cpp', 'src/record_main.cpp',
                 'src/frame_store.cpp', 'include/kaivuri/core.hpp', 'include/kaivuri/frame_store.hpp', 'CMakeLists.txt'):
        snapshots['native_stack/' + name] = (SOURCE / name).read_bytes()
    executable_hashes = {name: hashlib.sha256((args.build / name).read_bytes()).hexdigest()
                         for name in ('kaivuri-control', 'kaivuri-camera', 'kaivuri-record')}
    revision = subprocess.run(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    revision = revision.stdout.strip() if revision.returncode == 0 else 'uncommitted-deployment'
    if args.live_view and not args.view_python.exists():
        p.error('Rerun environment missing; install rerun-sdk==0.26.2 in .venv-rerun or use --view-python')
    if not 0 < args.view_fps <= 90:
        p.error('--view-fps must be in (0, 90]')
    processes = []
    viewer = None
    rc = 1
    try:
        with tempfile.TemporaryDirectory(prefix='kvr-') as tmp:
            runtime = Path(tmp)
            flags = [flag for flag, enabled in (('--enable-slew', args.enable_slew), ('--enable-tracks', args.enable_tracks)) if enabled]
            control_args = [str(args.build / 'kaivuri-control'), '--profile', str(args.profile), '--safety', str(args.safety),
                            '--socket', str(runtime / 'control.sock'), '--control-hz', str(args.control_hz), *flags]
            if args.bench_seconds is not None:
                control_args += ['--duration-s', str(args.bench_seconds + 60), '--timing-csv', str(runtime / 'control-timing.csv')]
            control_args += ['--simulate-imu'] if args.synthetic else ['--serial', args.serial]
            if args.live_pwm:
                control_args.append('--live-pwm')
            controller = subprocess.Popen(control_args, start_new_session=True)
            processes.append(controller)
            wait_for(runtime / 'control.sock', controller)
            camera_args = [str(args.build / 'kaivuri-camera'), '--camera', specs[0], '--camera', specs[1],
                           '--output-dir', str(runtime / 'cameras'), '--fps', str(args.fps)]
            if args.rotate_camera1:
                camera_args += ['--rotate-180', parsed[0][0]]
            if args.synthetic:
                camera_args.append('--synthetic')
            if args.exposure_us is not None:
                camera_args += ['--exposure-us', str(args.exposure_us)]
            if args.gain is not None:
                camera_args += ['--gain', str(args.gain)]
            cameras = subprocess.Popen(camera_args, start_new_session=True)
            processes.append(cameras)
            files = [runtime / 'cameras' / f'{serial}-ir{index}.frame' for serial, index in parsed]
            for path in files:
                wait_for(path, cameras)
            recorder_args = [str(args.build / 'kaivuri-record'), '--socket', str(runtime / 'control.sock'),
                             '--root', str(args.session), '--task', args.task, '--revision', revision,
                             '--fps', str(args.fps), '--max-episode-s', str(args.max_episode_s),
                             '--max-pair-skew-ms', str(args.max_pair_skew_ms), *flags]
            for file in files:
                recorder_args += ['--camera-file', str(file)]
            if args.live_pwm:
                recorder_args.append('--allow-live')
            if args.bench_seconds is not None:
                recorder_args += ['--scripted', '--auto-record-s', str(args.bench_seconds)]
            else:
                recorder_args += ['--gamepad', args.gamepad]
            recorder = subprocess.Popen(recorder_args, stdin=subprocess.DEVNULL, start_new_session=True)
            processes.append(recorder)
            if args.live_view:
                viewer = subprocess.Popen([str(args.view_python), str(SOURCE / 'tools/view_native.py'),
                    '--camera-file', str(files[0]), '--camera-file', str(files[1]),
                    '--socket', str(runtime / 'control.sock'), '--library', str(args.build / 'libkaivuri_frames.so'),
                    '--max-fps', str(args.view_fps),
                    *(['--rotate-camera1'] if args.view_rotate_camera1 else [])], start_new_session=True)
            try:
                rc = recorder.wait()
            except KeyboardInterrupt:
                recorder.send_signal(signal.SIGINT)
                rc = recorder.wait(timeout=20)
            finally:
                # Stop recorder before controller and cameras, including startup failures.
                for proc in reversed(processes):
                    if proc.poll() is None:
                        proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                if args.session.exists() and (runtime / 'control-timing.csv').exists():
                    (args.session / 'control-timing.csv').write_bytes((runtime / 'control-timing.csv').read_bytes())
                if viewer is not None and args.bench_seconds is not None and rc == 0:
                    if viewer.poll() is None:
                        viewer.terminate(); viewer.wait(timeout=10)
                    viewer = subprocess.Popen([str(args.view_python), str(SOURCE / 'tools/view_native.py'),
                        str(args.session), '--max-fps', str(args.view_fps),
                        *(['--rotate-camera1'] if args.view_rotate_camera1 else [])], start_new_session=True)
                    print('Capture and hardware processes stopped. Rerun remains available; Ctrl+C closes it.', flush=True)
                    try:
                        viewer.wait()
                    except KeyboardInterrupt:
                        pass
    finally:
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
            try:
                viewer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                viewer.kill(); viewer.wait()
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
        if args.session.exists():
            provenance = args.session / 'provenance'
            provenance.mkdir(exist_ok=True)
            for name, data in snapshots.items():
                dest = provenance / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            (provenance / 'manifest.json').write_text(json.dumps({'revision': revision,
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()},
                'executables': executable_hashes, 'camera_order': specs, 'controller_arguments': control_args,
                'camera_arguments': camera_args, 'camera_rotations_deg': [180 if args.rotate_camera1 else 0, 0],
                'live_view': args.live_view, 'view_fps': args.view_fps if args.live_view else None,
                'view_camera1_rotation_deg': 180 if args.view_rotate_camera1 else 0}, indent=2) + '\n')
    if rc == 0:
        print(f'Native recording complete: {args.session}\nExport with export_episodes.py after recording.')
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
