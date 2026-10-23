"""Convert committed native Y8 episodes to a LeRobot v3 dataset, outside capture/control.

Written with LeRobot 0.6.1 (0.5.1 also works). One or more sessions go into one dataset;
each session keeps its own task. Keeps original high-rate samples, camera metadata and
calibration snapshots beside LeRobot's nominal FPS timeline. Refuses interrupted takes
and inaccurate rates.

Only `observation.state` and `observation.images.*` are policy inputs. FK and IMU go
under `feedback.*`, like the Python recorder's `feedback.imu_quaternion`: LeRobot types
every other `observation.*` key as a STATE input feature too, so a policy would silently
pick them up. Build a state from them on purpose instead.

Optional depth (`session.json` "depth": ["cam1", ...]) is not a LeRobot feature: H.264
would quantize Z16. Each take's `camN.depth.zst` (one zstd frame of packed little-endian
uint16 per recorded frame, paired to that frame's IR exposure by sensor timestamp) is
validated and copied losslessly to `native/episode-*/`, with the session's
`camN.depth.json` calibration in `native/session-*/`. Read it with `depth_frames()`.
"""
import argparse
import ctypes
import ctypes.util
import functools
import hashlib
import json
import math
import mmap
from pathlib import Path
import shutil

# Shapes are flattened so numeric features remain simple vectors in LeRobot/Parquet.
POSE_FIELDS = {
    'fk_xyz_m': 3, 'fk_quat_wxyz': 4, 'fk_upper_xyz_m': 3,
    'fk_upper_xz_m': 2, 'fk_upper_quat_wxyz': 4,
    'slew_yaw_rad': 1, 'bucket_pitch_rad': 1,
    'fk_joint_xyz_m': 12, 'fk_joint_xyz_upper_m': 12,
    'fk_joint_quat_wxyz': 16, 'fk_link_quat_wxyz': 16,
    'fk_link_quat_upper_wxyz': 16,
    'imu_quat_wxyz': 16, 'imu_corrected_quat_wxyz': 16,
    'imu_gyro_dps': 12, 'imu_accel_g': 12,
}
CAM_INT_FIELDS = ('sequence', 'host_monotonic_ns', 'sensor_timestamp_us',
                  'frame_timestamp_us', 'frame_counter', 'actual_exposure_us',
                  'gain_level', 'valid_metadata', 'timestamp_domain', 'ir_index')
STATUS_INT_FIELDS = ('imu_sequence', 'imu_device_us', 'imu_host_monotonic_ns',
                     'host_monotonic_ns', 'ticks', 'applied_command_sequence',
                     'applied_host_monotonic_ns')
ROW_INT_FIELDS = ('sample_ns', 'episode_ns', 'requested_sequence')
CLOCK_FLOAT_FIELDS = ('pair_skew_s', 'imu_age_s', 'teleop_lateness_ms', 'controller_worst_lateness_s', 'fk_work_us', 'loop')
CAMERAS = ('cam1', 'cam2')
JOINTS = {'slew': 0, 'lift': 1, 'tilt': 2, 'scoop': 3}


@functools.cache
def _zstd():
    lib = ctypes.CDLL(ctypes.util.find_library('zstd') or 'libzstd.so.1')
    size, ptr = ctypes.c_size_t, ctypes.c_void_p
    for name, res, args in (('ZSTD_findFrameCompressedSize', size, [ptr, size]),
                            ('ZSTD_getFrameContentSize', ctypes.c_ulonglong, [ptr, size]),
                            ('ZSTD_decompress', size, [ptr, size, ptr, size]),
                            ('ZSTD_isError', ctypes.c_uint, [size])):
        getattr(lib, name).restype, getattr(lib, name).argtypes = res, args
    return lib


def depth_frames(path):
    """Yield each recorded depth frame of a camN.depth.zst as bytes (packed uint16 LE, mm * depth_scale)."""
    lib = _zstd()
    with open(path, 'rb') as stream:
        if not Path(path).stat().st_size:
            return
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as view:
            buffer = (ctypes.c_char * len(view)).from_buffer(view)
            base, offset, total = ctypes.addressof(buffer), 0, len(view)
            try:
                while offset < total:
                    frame = lib.ZSTD_findFrameCompressedSize(base + offset, total - offset)
                    content = lib.ZSTD_getFrameContentSize(base + offset, total - offset)
                    if lib.ZSTD_isError(frame) or content >= 2**63 or content > 64 * 2**20:
                        raise ValueError(f'{path}: corrupt zstd depth frame at byte {offset}')
                    out = ctypes.create_string_buffer(content)
                    got = lib.ZSTD_decompress(out, content, base + offset, frame)
                    if lib.ZSTD_isError(got) or got != content:
                        raise ValueError(f'{path}: corrupt zstd depth frame at byte {offset}')
                    yield out.raw
                    offset += frame
            finally:
                del buffer


def feedback_key(field):
    return f"feedback.{'imu' if field.startswith('imu_') else 'fk'}.{field}"


def samples(episode):
    """Every row of an episode's samples.jsonl: frame rows and the high-rate status rows between them."""
    with (Path(episode) / 'samples.jsonl').open() as stream:
        for line in stream:
            yield json.loads(line)


def frame_rows(episode):
    return (row for row in samples(episode) if 'frame_index' in row)


def committed_episodes(session):
    """A session's saved takes in order; .partial (interrupted) takes are left out."""
    return sorted(Path(session).glob('episode-*.ready'))


def validate_episode(episode, session, rate_tolerance=.03):
    summary = json.loads((episode / 'episode.json').read_text())
    depth_keys = session.get('depth', [])
    if summary.get('depth', []) != depth_keys or any(k not in CAMERAS for k in depth_keys):
        raise ValueError(f'{episode}: depth cameras differ from session')
    count, first, previous, shape, gaps = 0, None, None, None, []
    for row in frame_rows(episode):
        if row['frame_index'] != count:
            raise ValueError(f'{episode}: non-contiguous frame indices')
        status = row['status']
        if not status['fk_valid'] or not status['imu_healthy'] or status['state'] != 'armed':
            raise ValueError(f'{episode}: invalid FK/controller state')
        if len(status['applied_action']) != 6 or not all(math.isfinite(x) and abs(x) <= 1 for x in status['applied_action']):
            raise ValueError(f'{episode}: invalid applied action')
        shapes = []
        for key in CAMERAS:
            m = row[key]
            if m['stride'] != m['width'] or m['pixel_bytes'] != m['width'] * m['height']:
                raise ValueError(f'{episode}: invalid packed Y8 geometry')
            shapes.append((m['height'], m['width']))
            if previous and (m['sequence'] <= previous[key]['sequence'] or m['host_monotonic_ns'] <= previous[key]['host_monotonic_ns']):
                raise ValueError(f'{episode}: duplicate/restarted camera stream')
        for key in depth_keys:
            d, m = row[f'{key}_depth'], row[key]
            if (d['width'], d['height']) != (m['width'], m['height']) or d['stride'] != 2 * d['width'] or d['pixel_bytes'] != d['stride'] * d['height'] or d['ir_index'] != 0:
                raise ValueError(f'{episode}: invalid packed Z16 depth geometry')
            if not (d['valid_metadata'] & m['valid_metadata'] & 1) or d['sensor_timestamp_us'] != m['sensor_timestamp_us']:
                raise ValueError(f'{episode}: depth not paired with its IR exposure')
            if previous and d['sequence'] <= previous[f'{key}_depth']['sequence']:
                raise ValueError(f'{episode}: duplicate/restarted depth stream')
        shape = shape or shapes
        if shape != shapes:
            raise ValueError(f'{episode}: camera resolution changed')
        if previous:
            gaps.append((row['cam1']['host_monotonic_ns'] - previous['cam1']['host_monotonic_ns']) / 1e9)
        first = first or row
        previous = row
        count += 1
    if not count or count != summary['frames']:
        raise ValueError(f'{episode}: frame count mismatch')
    for key, (height, width) in zip(CAMERAS, shape):
        if (episode / f'{key}.y8').stat().st_size != count * height * width:
            raise ValueError(f'{episode}: truncated/extra pixels in {key}')
        if key in depth_keys:
            frames = 0
            for data in depth_frames(episode / f'{key}.depth.zst'):
                if len(data) != 2 * height * width:
                    raise ValueError(f'{episode}: wrong depth frame size in {key}')
                frames += 1
            if frames != count:
                raise ValueError(f'{episode}: {frames} depth frames for {count} samples in {key}')
    if count < 2:
        raise ValueError(f'{episode}: at least two frames required for timing validation')
    elapsed = (previous['cam1']['host_monotonic_ns'] - first['cam1']['host_monotonic_ns']) / 1e9
    hz = (count - 1) / elapsed
    fps = session['fps']
    if abs(hz / fps - 1) > rate_tolerance or max(gaps) > 1.5 / fps:
        raise ValueError(f'{episode}: capture achieved {hz:.3f} Hz for declared {fps} FPS, max gap {max(gaps)*1000:.1f} ms; retain raw data and re-record')
    return shape, count, hz


def build_features(shapes, tracks=False, state_joints=('lift', 'tilt', 'scoop')):
    def spec(dtype, size=1, names=None):
        return {'dtype': dtype, 'shape': (size,), 'names': names}
    width = 6 if tracks else 4
    features = {
        'observation.state': spec('float32', len(state_joints), list(state_joints)),
        'action': spec('float32', width, ['slew', 'lift', 'tilt', 'scoop', 'trackL', 'trackR'][:width]),
        'diagnostics.requested_action': spec('float32', width),
        'diagnostics.pulses_us': spec('float32', 9),
    }
    for key, (height, width) in zip(CAMERAS, shapes):
        features[f'observation.images.{key}'] = {'dtype': 'video', 'shape': (height, width, 3), 'names': ['height', 'width', 'channels']}
        features.update({f'clock.{key}.{field}': spec('int64') for field in CAM_INT_FIELDS})
        features.update({f'clock.{key}.{field}': spec('float64') for field in ('age_s', 'sdk_timestamp_ms')})
    features.update({feedback_key(key): spec('float32', size) for key, size in POSE_FIELDS.items()})
    features.update({f'clock.{key}': spec('int64') for key in (*STATUS_INT_FIELDS, *ROW_INT_FIELDS)})
    features.update({f'clock.{key}': spec('float64') for key in CLOCK_FLOAT_FIELDS})
    features['diagnostics.pump_on'] = spec('int64')
    return features


def convert_frame(row, pixels, session, state_joints):
    import numpy as np
    s = row['status']
    width = 6 if session['enable_tracks'] else 4
    frame = {'task': session['task'],
             'observation.state': np.rad2deg(np.array([s['angles_rad'][JOINTS[j]] for j in state_joints], dtype=np.float32)),
             'action': np.array(s['applied_action'][:width], dtype=np.float32),
             'diagnostics.requested_action': np.array(row['requested_action'][:width], dtype=np.float32),
             'diagnostics.pulses_us': np.array(s['pulses_us'], dtype=np.float32)}

    def scalars(dtype, values):
        frame.update({key: np.array([value], dtype=dtype) for key, value in values})
    for key, data in zip(CAMERAS, pixels):
        m = row[key]
        gray = np.frombuffer(data, dtype=np.uint8).reshape(m['height'], m['width'])
        frame[f'observation.images.{key}'] = np.repeat(gray[..., None], 3, axis=2)
        scalars(np.int64, ((f'clock.{key}.{field}', m[field]) for field in CAM_INT_FIELDS))
        scalars(np.float64, ((f'clock.{key}.sdk_timestamp_ms', m['sdk_timestamp_ms']),
                             (f'clock.{key}.age_s', (row['sample_ns'] - m['host_monotonic_ns']) / 1e9)))
    for key in POSE_FIELDS:
        frame[feedback_key(key)] = np.asarray(s[key], dtype=np.float32).reshape(-1)
    scalars(np.int64, ((f'clock.{key}', s[key]) for key in STATUS_INT_FIELDS))
    scalars(np.int64, ((f'clock.{key}', row[key]) for key in ROW_INT_FIELDS))
    clock = {'pair_skew_s': (row['cam2']['host_monotonic_ns'] - row['cam1']['host_monotonic_ns']) / 1e9,
             'imu_age_s': s['imu_age_s'], 'teleop_lateness_ms': row['teleop_lateness_ms'],
             'controller_worst_lateness_s': s['worst_lateness_s'], 'fk_work_us': s['fk_work_us'],
             'loop': row['episode_ns'] / 1e9}  # same keys and order as CLOCK_FLOAT_FIELDS
    scalars(np.float64, ((f'clock.{key}', value) for key, value in clock.items()))
    scalars(np.int64, [('diagnostics.pump_on', int(s['pump_on']))])
    return frame


def load_session(session_path):
    session = json.loads((session_path / 'session.json').read_text())
    if session['format'] != 'kaivuri-native-episode-v1':
        raise ValueError(f'{session_path}: unsupported native episode format')
    episodes = committed_episodes(session_path)
    if not episodes:
        raise ValueError(f'{session_path}: no committed episodes; .partial takes are intentionally excluded')
    return session, episodes


def create_dataset(repo_id, root, fps, features):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    common = dict(repo_id=repo_id, root=root, fps=fps, features=features,
                  robot_type='masi_excavator_native', use_videos=True, streaming_encoding=True)
    try:  # LeRobot 0.6: the encoder is a config object
        from lerobot.configs.video import RGBEncoderConfig
    except ImportError:  # LeRobot 0.5
        return LeRobotDataset.create(**common, vcodec='h264')
    return LeRobotDataset.create(**common, rgb_encoder=RGBEncoderConfig(vcodec='h264'))


def export(session_paths, root, repo_id, state_joints=('lift', 'tilt', 'scoop')):
    if isinstance(session_paths, (str, Path)):
        session_paths = [session_paths]
    session_paths = [Path(p) for p in session_paths]
    if root.exists():
        raise ValueError(f'dataset root already exists: {root}; choose a new destination')
    sessions = [load_session(p) for p in session_paths]
    first = sessions[0][0]
    for path, (session, _) in zip(session_paths, sessions):
        if session['fps'] != first['fps'] or session['enable_tracks'] != first['enable_tracks']:
            raise ValueError(f'{path}: fps/track layout differs from {session_paths[0]}')
    checked = [[validate_episode(ep, session) for ep in episodes] for session, episodes in sessions]
    shapes = {tuple(item[0]) for group in checked for item in group}
    if len(shapes) != 1:
        raise ValueError('camera shapes differ between episodes')
    # Heavy dependencies are imported only after all takes pass validation.
    features = build_features(checked[0][0][0], first['enable_tracks'], state_joints)
    dataset = create_dataset(repo_id, root, first['fps'], features)
    index = 0
    try:
        native = root / 'native'
        native.mkdir()
        for number, (session_path, (session, episodes), group) in enumerate(zip(session_paths, sessions, checked)):
            target_session = native / f'session-{number:03d}'
            target_session.mkdir()
            shutil.copy2(session_path / 'session.json', target_session / 'session.json')
            if (session_path / 'provenance').exists():
                shutil.copytree(session_path / 'provenance', target_session / 'provenance')
            if (session_path / 'control-timing.csv').exists():
                shutil.copy2(session_path / 'control-timing.csv', target_session / 'control-timing.csv')
            for key in session.get('depth', []):
                shutil.copy2(session_path / f'{key}.depth.json', target_session / f'{key}.depth.json')
            for episode, (_, count, hz) in zip(episodes, group):
                with (episode / 'cam1.y8').open('rb') as c1, (episode / 'cam2.y8').open('rb') as c2:
                    for row in frame_rows(episode):
                        pixels = [stream.read(row[key]['pixel_bytes']) for stream, key in zip((c1, c2), CAMERAS)]
                        dataset.add_frame(convert_frame(row, pixels, session, state_joints))
                dataset.save_episode()
                target = native / f'episode-{index:06d}'
                target.mkdir()
                for name in ('samples.jsonl', 'episode.json', *(f'{k}.depth.zst' for k in session.get('depth', []))):
                    shutil.copy2(episode / name, target / name)
                (target / 'source.json').write_text(json.dumps({'source': str(episode.resolve()), 'session': number,
                    'task': session['task'], 'frames': count, 'achieved_hz': hz,
                    'samples_sha256': hashlib.sha256((episode/'samples.jsonl').read_bytes()).hexdigest()}) + '\n')
                print(f'Exported episode {index}: {count} frames, {hz:.3f} Hz, task {session["task"]!r}')
                index += 1
    finally:
        dataset.finalize()
    (root / 'native' / 'EXPORT_COMPLETE').write_text('kaivuri-native-episode-v1\n')
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path, nargs='+', help='one or more native sessions, in episode order')
    p.add_argument('--root', type=Path)
    p.add_argument('--repo-id')
    p.add_argument('--state-joints', default='lift,tilt,scoop')
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    joints = tuple(args.state_joints.split(','))
    if not joints or len(set(joints)) != len(joints) or any(j not in JOINTS for j in joints):
        p.error('--state-joints must be a unique subset of slew,lift,tilt,scoop')
    if args.validate_only:
        for path in args.session:
            session, episodes = load_session(path)
            for ep in episodes:
                shape, count, hz = validate_episode(ep, session)
                print(f'{ep.name}: {count} frames, {hz:.3f} Hz, shapes={shape}, depth={session.get("depth", [])}')
    else:
        if not args.root or not args.repo_id:
            p.error('--root and --repo-id required for export')
        export(args.session, args.root, args.repo_id, joints)


if __name__ == '__main__':
    main()
