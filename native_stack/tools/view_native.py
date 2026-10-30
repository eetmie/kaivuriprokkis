"""Optional Rerun playback/live tail of a native episode. No hardware imports.

Live preview is capped at --max-fps (10 by default); full recording stays intact.
Uses only NumPy + rerun-sdk 0.26.2, never LeRobot/Torch or controller IPC.

Committed takes with depth also show each camera's colormapped depth and a 3D point
cloud colored by its IR image (camera 1's IR2 via the recorded depth-to-IR extrinsics).
Live tails (--live, direct preview) show IR only.
"""
import argparse
import json
import math
from pathlib import Path
import time

from native_sessions import recording_paths


def wait_episode(session, index, live):
    while True:
        runs = recording_paths(session) if (session / 'session.json').is_file() else [session]
        episodes = [ep for run in runs
                    for ep in sorted([*run.glob('episode-*.ready'), *run.glob('episode-*.partial')])]
        if not live:
            episodes = [ep for ep in episodes if ep.suffix == '.ready']
        if len(episodes) > index:
            return episodes[index]
        if not live:
            raise ValueError(f'committed episode {index} not found')
        time.sleep(.1)


def rows(episode, live, max_fps=10, depth_keys=()):
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
    from export_episodes import depth_frames
    depth = {key: [depth_frames(episode / f'{key}.depth.zst'), 0] for key in depth_keys}
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
                frames = {}
                for key, state in depth.items():  # sequential zstd frames; skip unsampled ones
                    while state[1] < row['frame_index']:
                        next(state[0]); state[1] += 1
                    frames[key] = next(state[0]); state[1] += 1
                yield row, pixels, frames
    finally:
        samples.close()
        for stream in streams:
            stream.close()


def camera_rows(paths, socket_path, library, max_fps):
    """Sample native frame stores/status without driving or writing an episode."""
    from frame_reader import FrameReader, Metadata
    from client import Client
    capacity = 1280 * 800  # largest D435 IR mode; record_native --resolution goes up to 1280x720
    with FrameReader(paths[0], library, capacity) as cam1, FrameReader(paths[1], library, capacity) as cam2, \
            Client(socket_path) as control:
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
                yield row, pixels, {}
            time.sleep(max(0, 1 / max_fps - (time.monotonic() - start)))


def point_cloud(g, z, image, depth_range_mm):
    """Depth points (m, displayed camera frame) and their gray from the IR image; g from main()'s geometry."""
    import numpy as np
    raw_z, raw_ir = (np.rot90(z, 2), np.rot90(image, 2)) if g['turned'] else (z, image)
    zm = raw_z[::g['step'], ::g['step']].astype(np.float32) * g['scale']
    # The colormap range also bounds the cloud: sparse far matches would otherwise
    # dominate the 3D view's automatic framing.
    lo, hi = (r / 1000 for r in depth_range_mm)
    ok = (zm >= lo) & (zm <= hi)
    # Drop speckle: keep points with at least 4 of 8 neighbours within 3% depth.
    padded = np.pad(np.where(ok, zm, np.nan), 1, constant_values=np.nan)
    agree = sum(np.abs(padded[1 + dy:padded.shape[0] - 1 + dy, 1 + dx:padded.shape[1] - 1 + dx] - zm) < .03 * zm
                for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    ok &= agree >= 4
    pts = np.concatenate([g['rays'][ok] * zm[ok, None], zm[ok, None]], 1)
    # Color from this camera's recorded IR imager (IR2 is 50 mm from depth's IR1).
    q = pts @ g['R'].T + g['t']
    ii = g['ii']
    pu = np.round(q[:, 0] / q[:, 2] * ii['fx'] + ii['ppx']).astype(int)
    pv = np.round(q[:, 1] / q[:, 2] * ii['fy'] + ii['ppy']).astype(int)
    inside = (pu >= 0) & (pu < ii['width']) & (pv >= 0) & (pv < ii['height'])
    gray = np.zeros(len(pts), np.uint8)
    gray[inside] = raw_ir[pv[inside], pu[inside]]
    if g['turned']:
        pts[:, :2] *= -1
    return pts, gray


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
    p.add_argument('--no-depth', action='store_true', help='skip recorded depth')
    p.add_argument('--cloud-stride', type=int, default=2, help='depth pixel step for the 3D point cloud')
    p.add_argument('--depth-range-mm', type=float, nargs=2, default=(150, 1500), help='depth colormap range')
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
    depth_keys, calibration, episode = [], {}, None
    if args.session is not None and not args.live and not args.no_depth:
        episode = wait_episode(args.session, args.episode_index, False)
        depth_keys = json.loads((episode.parent / 'session.json').read_text()).get('depth', [])
        calibration = {key: json.loads((episode.parent / f'{key}.depth.json').read_text()) for key in depth_keys}
    rr.init('kaivuri-native-episode', spawn=False)
    series = rrb.Horizontal(rrb.TimeSeriesView(origin='actions', name='Actions'),
                            rrb.TimeSeriesView(origin='joints', name='Joint angles (rad)'),
                            rrb.TimeSeriesView(origin='timing', name='Timing (ms)'))
    if depth_keys:
        images, clouds = [], []
        for key in ('cam1', 'cam2'):
            images.append(rrb.Spatial2DView(origin=f'cameras/{key}', name=f'Camera {key[-1]} IR'))
            if key in depth_keys:
                images.append(rrb.Spatial2DView(origin=f'scene/{key}/camera', name=f'Camera {key[-1]} depth'))
                clouds.append(rrb.Spatial3DView(origin=f'scene/{key}', name=f'Camera {key[-1]} point cloud (m)',
                                                contents=['+ $origin/**', '- $origin/camera/depth'], line_grid=False))
        blueprint = rrb.Blueprint(rrb.Vertical(rrb.Horizontal(*images),
            rrb.Horizontal(*clouds, rrb.Spatial3DView(origin='robot/base', name='Full FK (meters)')),
            series, row_shares=[2, 3, 1]))
    else:
        blueprint = rrb.Blueprint(rrb.Vertical(
            rrb.Horizontal(rrb.Spatial2DView(origin='cameras/cam1', name='Camera 1'),
                           rrb.Spatial2DView(origin='cameras/cam2', name='Camera 2'),
                           rrb.Spatial3DView(origin='robot/base', name='Full FK (meters)')),
            series))
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
        episode = episode or wait_episode(args.session, args.episode_index, args.live)
        source = rows(episode, args.live, args.max_fps, depth_keys)
    # Depth intrinsics/extrinsics are SDK values for unrotated sensor pixels. Unrotate stored
    # pixels for geometry; a 180 degree image turn equals flipping camera x and y.
    geometry = {}
    for key in depth_keys:
        c = calibration[key]
        turned = (c.get('rotation_deg', 0) + (180 if key == 'cam1' and args.rotate_camera1 else 0)) % 360 == 180
        if c.get('synthetic'):
            geometry[key] = None
            continue
        di, ii, e = c['depth_intrinsics'], c['ir_intrinsics'], c['depth_to_ir_extrinsics']
        w, h, step = di['width'], di['height'], args.cloud_stride
        v, u = np.mgrid[0:h:step, 0:w:step]
        rays = np.stack([(u - di['ppx']) / di['fx'], (v - di['ppy']) / di['fy']], -1).astype(np.float32)
        geometry[key] = dict(rays=rays, step=step, ii=ii, turned=turned, scale=c['depth_scale_m'],
                             R=np.asarray(e['rotation_column_major'], np.float32).reshape(3, 3).T,
                             t=np.asarray(e['translation_m'], np.float32))
        ppx, ppy = (w - 1 - di['ppx'], h - 1 - di['ppy']) if turned else (di['ppx'], di['ppy'])
        rr.log(f'scene/{key}', rr.ViewCoordinates.RDF, static=True)
        rr.log(f'scene/{key}/camera', rr.Pinhole(focal_length=[di['fx'], di['fy']], principal_point=[ppx, ppy],
               width=w, height=h, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=.05), static=True)
    count = 0
    for row, pixels, depth in source:
        elapsed = row['episode_ns'] / 1e9
        rr.set_time('elapsed', duration=elapsed)
        rr.set_time('frame_index', sequence=row['frame_index'])
        for key, data in zip(('cam1', 'cam2'), pixels):
            m = row[key]
            rotate = key == 'cam1' and args.rotate_camera1
            image = np.frombuffer(data, dtype=np.uint8).reshape(m['height'], m['width'])
            if rotate:
                image = np.rot90(image, 2).copy()
            rr.log(f'cameras/{key}', rr.Image(image).compress(jpeg_quality=args.jpeg_quality))
            if key in depth:
                z = np.frombuffer(depth[key], dtype='<u2').reshape(m['height'], m['width'])
                if rotate:
                    z = np.rot90(z, 2).copy()
                rr.log(f'scene/{key}/camera/depth', rr.DepthImage(z, meter=1000.0, colormap='turbo',
                       depth_range=args.depth_range_mm))
                if geometry.get(key):
                    pts, gray = point_cloud(geometry[key], z, image, args.depth_range_mm)
                    rr.log(f'scene/{key}/points', rr.Points3D(pts, colors=np.repeat(gray[:, None], 3, 1),
                                                              radii=rr.Radius.ui_points(1.5)))
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
