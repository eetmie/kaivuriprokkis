"""Optional Rerun playback/live tail of a native episode. No hardware imports.

Live preview is capped at --max-fps (10 by default); full recording stays intact.
Uses only NumPy + rerun-sdk 0.26.2, never LeRobot/Torch or controller IPC.
"""
import argparse
import json
import math
from pathlib import Path
import time


def wait_episode(session, index, live):
    while True:
        episodes = sorted([*session.glob('episode-*.ready'), *session.glob('episode-*.partial')])
        if not live:
            episodes = [ep for ep in episodes if ep.suffix == '.ready']
        if len(episodes) > index:
            return episodes[index]
        if not live:
            raise ValueError(f'committed episode {index} not found')
        time.sleep(.1)


def rows(episode, live, max_fps=10):
    # Open handles survive the writer's .partial -> .ready rename.
    while True:
        try:
            samples = (episode / 'samples.jsonl').open()
            streams = [(episode / f'cam{i}.y8').open('rb') for i in (1, 2)]
            break
        except FileNotFoundError:
            if 'samples' in locals():
                samples.close()
            if not live:
                raise
            if episode.with_suffix('.ready').exists():
                episode = episode.with_suffix('.ready')
            time.sleep(.05)
    next_preview = -math.inf
    try:
        while True:
            offset = samples.tell()
            line = samples.readline()
            complete = episode.suffix == '.ready' or episode.with_suffix('.ready').exists()
            if not line or not line.endswith('\n'):
                samples.seek(offset)
                if complete:
                    if line:
                        raise ValueError('incomplete final JSONL row')
                    return
                if not live:
                    raise ValueError('episode is incomplete; use --live to preview an active take')
                time.sleep(.03)
                continue
            row = json.loads(line)
            if 'frame_index' not in row:
                continue
            elapsed = row['episode_ns'] / 1e9
            if elapsed < next_preview:
                continue
            pixels = []
            for key, stream in zip(('cam1', 'cam2'), streams):
                m = row[key]
                if m['pixel_bytes'] != m['width'] * m['height'] or m['stride'] != m['width']:
                    raise ValueError('unsupported Y8 geometry')
                stream.seek(row['frame_index'] * m['pixel_bytes'])
                data = stream.read(m['pixel_bytes'])
                if len(data) != m['pixel_bytes']:
                    if complete:
                        raise ValueError('truncated pixel file')
                    samples.seek(offset)
                    time.sleep(.03)
                    break
                pixels.append(data)
            if len(pixels) == 2:
                next_preview = elapsed + 1 / max_fps
                yield row, pixels
    finally:
        samples.close()
        for stream in streams:
            stream.close()


def camera_rows(paths, socket_path, library, max_fps):
    """Sample native frame stores/status without driving or writing an episode."""
    from frame_reader import FrameReader, Metadata
    from client import Client
    with FrameReader(paths[0], library) as cam1, FrameReader(paths[1], library) as cam2, Client(socket_path) as control:
        begin = time.monotonic_ns()
        while True:
            start = time.monotonic()
            snapshots = [cam1.read(), cam2.read()]
            if all(item is not None for item in snapshots):
                status = control.status()
                now = time.monotonic_ns()
                row = {'sample_ns': now, 'episode_ns': now - begin,
                       'frame_index': snapshots[0][0].sequence, 'status': status}
                pixels = []
                for key, snapshot in zip(('cam1', 'cam2'), snapshots):
                    meta, data = snapshot
                    row[key] = {name: getattr(meta, name) for name, _ in Metadata._fields_}
                    pixels.append(bytes(data))
                yield row, pixels
            time.sleep(max(0, 1 / max_fps - (time.monotonic() - start)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path, nargs='?')
    p.add_argument('--camera-file', type=Path, action='append', default=[])
    p.add_argument('--socket', help='status-only native controller connection for direct live preview')
    p.add_argument('--library', type=Path)
    p.add_argument('--episode-index', type=int, default=0)
    p.add_argument('--live', action='store_true')
    p.add_argument('--max-fps', type=float, default=10)
    p.add_argument('--rotate-camera1', action='store_true', help='rotate camera 1 display by 180 degrees')
    p.add_argument('--jpeg-quality', type=int, default=80)
    p.add_argument('--grpc-port', type=int, default=9876)
    p.add_argument('--web-port', type=int, default=9090)
    p.add_argument('--save', type=Path, help='write an .rrd instead of serving a viewer')
    args = p.parse_args()
    if args.episode_index < 0 or not math.isfinite(args.max_fps) or not 0 < args.max_fps <= 90 or not 1 <= args.jpeg_quality <= 100:
        p.error('invalid preview limits')
    if args.camera_file:
        if len(args.camera_file) != 2 or not args.socket or not args.library:
            p.error('direct preview requires two --camera-file paths, --socket and --library')
    elif args.session is None:
        p.error('session or direct camera preview arguments required')
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb
    rr.init('kaivuri-native-episode', spawn=False)
    blueprint = rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(rrb.Spatial2DView(origin='cameras/cam1', name='Camera 1'),
                       rrb.Spatial2DView(origin='cameras/cam2', name='Camera 2'),
                       rrb.Spatial3DView(origin='robot/base', name='Full FK (meters)')),
        rrb.Horizontal(rrb.TimeSeriesView(origin='actions', name='Actions'),
                       rrb.TimeSeriesView(origin='joints', name='Joint angles (rad)'),
                       rrb.TimeSeriesView(origin='timing', name='Timing (ms)'))))
    if args.save:
        rr.save(args.save, default_blueprint=blueprint)
    else:
        uri = rr.serve_grpc(grpc_port=args.grpc_port, server_memory_limit='256MB', default_blueprint=blueprint)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=uri)
        print(f'Browser: http://192.168.0.108:{args.web_port}/?url=rerun%2Bhttp://192.168.0.108:{args.grpc_port}/proxy', flush=True)
    if args.camera_file:
        print(f'Direct camera/status preview active; camera 1 display rotation={180 if args.rotate_camera1 else 0} degrees; recording is independent.', flush=True)
        source = camera_rows(args.camera_file, args.socket, args.library, args.max_fps)
    else:
        print(f'Waiting for episode {args.episode_index}: {args.session}', flush=True)
        episode = wait_episode(args.session, args.episode_index, args.live)
        source = rows(episode, args.live, args.max_fps)
    count = 0
    for row, pixels in source:
        elapsed = row['episode_ns'] / 1e9
        rr.set_time('elapsed', duration=elapsed)
        rr.set_time('frame_index', sequence=row['frame_index'])
        for key, data in zip(('cam1', 'cam2'), pixels):
            m = row[key]
            image = np.frombuffer(data, dtype=np.uint8).reshape(m['height'], m['width'])
            if key == 'cam1' and args.rotate_camera1:
                image = np.rot90(image, 2).copy()
            rr.log(f'cameras/{key}', rr.Image(image).compress(jpeg_quality=args.jpeg_quality))
            age = (row['sample_ns'] - m['host_monotonic_ns']) / 1e6
            rr.log(f'timing/{key}_arrival_age', rr.Scalars(age))
        status = row['status']
        for i, joint in enumerate(('slew', 'boom', 'arm', 'bucket')):
            rr.log(f'joints/{joint}', rr.Scalars(status['angles_rad'][i]))
        for i, joint in enumerate(('slew', 'boom', 'arm', 'bucket', 'trackL', 'trackR')):
            if 'requested_action' in row:
                rr.log(f'actions/{joint}/requested', rr.Scalars(row['requested_action'][i]))
            rr.log(f'actions/{joint}/applied', rr.Scalars(status['applied_action'][i]))
        if status['fk_valid']:
            for frame, pivots, tip in (('base', 'fk_joint_xyz_m', 'fk_xyz_m'), ('upper', 'fk_joint_xyz_upper_m', 'fk_upper_xyz_m')):
                points = np.asarray([*status[pivots], status[tip]])
                rr.log(f'robot/{frame}/pivots', rr.Points3D(points, radii=.012, labels=['slew','boom','arm','bucket','tool']))
                rr.log(f'robot/{frame}/links', rr.LineStrips3D([points], radii=.004))
            for i, axis in enumerate('xyz'):
                rr.log(f'fk/upper_{axis}_m', rr.Scalars(status['fk_upper_xyz_m'][i]))
        for key, value in {'teleop_lateness': row.get('teleop_lateness_ms'),
                           'controller_worst_lateness': status['worst_lateness_s'] * 1000,
                           'imu_age': status['imu_age_s'] * 1000,
                           'pair_arrival_skew': abs(row['cam1']['host_monotonic_ns'] - row['cam2']['host_monotonic_ns']) / 1e6}.items():
            if value is not None:
                rr.log(f'timing/{key}', rr.Scalars(value))
        if count == 0:
            print(f"Preview streaming: cam1 IR{row['cam1']['ir_index']}, cam2 IR{row['cam2']['ir_index']}", flush=True)
        count += 1
    rr.get_global_data_recording().flush(timeout_sec=10)
    print(f'Preview loaded: {count} sampled frame pairs. Full-rate recording is unchanged.', flush=True)
    if not args.save:
        print('Server stays available for playback; Ctrl+C stops it.', flush=True)
        while True:
            time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
