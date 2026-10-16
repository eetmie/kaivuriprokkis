"""Convert committed native Y8 episodes to a LeRobot v3 dataset, outside capture/control.

Written with LeRobot 0.6.1 (0.5.1 also works). One or more sessions go into one dataset;
each session keeps its own task. Keeps original high-rate samples, camera metadata and
calibration snapshots beside LeRobot's nominal FPS timeline. Refuses interrupted takes
and inaccurate rates.

Only `observation.state` and `observation.images.*` are policy inputs. FK and IMU go
under `feedback.*`, like the Python recorder's `feedback.imu_quaternion`: LeRobot types
every other `observation.*` key as a STATE input feature too, so a policy would silently
pick them up. Build a state from them on purpose instead.
"""
import argparse
import hashlib
import json
import math
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
JOINTS = {'slew': 0, 'lift': 1, 'tilt': 2, 'scoop': 3}


def feedback_key(field):
    return f"feedback.{'imu' if field.startswith('imu_') else 'fk'}.{field}"


def frame_rows(episode):
    with (episode / 'samples.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if 'frame_index' in row:
                yield row


def validate_episode(episode, session, rate_tolerance=.03):
    summary = json.loads((episode / 'episode.json').read_text())
    count, first, last, previous = 0, None, None, None
    shape = None
    gaps = []
    for row in frame_rows(episode):
        if row['frame_index'] != count:
            raise ValueError(f'{episode}: non-contiguous frame indices')
        status = row['status']
        if not status['fk_valid'] or not status['imu_healthy'] or status['state'] != 'armed':
            raise ValueError(f'{episode}: invalid FK/controller state')
        if len(status['applied_action']) != 6 or not all(math.isfinite(x) and abs(x) <= 1 for x in status['applied_action']):
            raise ValueError(f'{episode}: invalid applied action')
        shapes = []
        for key in ('cam1', 'cam2'):
            m = row[key]
            if m['stride'] != m['width'] or m['pixel_bytes'] != m['width'] * m['height']:
                raise ValueError(f'{episode}: invalid packed Y8 geometry')
            shapes.append((m['height'], m['width']))
            if previous and (m['sequence'] <= previous[key]['sequence'] or m['host_monotonic_ns'] <= previous[key]['host_monotonic_ns']):
                raise ValueError(f'{episode}: duplicate/restarted camera stream')
        if shape is None:
            shape = shapes
        if shape != shapes:
            raise ValueError(f'{episode}: camera resolution changed')
        if previous:
            delta = (row['cam1']['host_monotonic_ns'] - previous['cam1']['host_monotonic_ns']) / 1e9
            gaps.append(delta)
        first = row if first is None else first
        last = row
        previous = row
        count += 1
    if not count or count != summary['frames']:
        raise ValueError(f'{episode}: frame count mismatch')
    for key, (height, width) in zip(('cam1', 'cam2'), shape):
        if (episode / f'{key}.y8').stat().st_size != count * height * width:
            raise ValueError(f'{episode}: truncated/extra pixels in {key}')
    if count < 2:
        raise ValueError(f'{episode}: at least two frames required for timing validation')
    elapsed = (last['cam1']['host_monotonic_ns'] - first['cam1']['host_monotonic_ns']) / 1e9
    hz = (count - 1) / elapsed
    fps = session['fps']
    if abs(hz / fps - 1) > rate_tolerance or max(gaps) > 1.5 / fps:
        raise ValueError(f'{episode}: capture achieved {hz:.3f} Hz for declared {fps} FPS, max gap {max(gaps)*1000:.1f} ms; retain raw data and re-record')
    return shape, count, hz


def build_features(shapes, tracks=False, state_joints=('lift', 'tilt', 'scoop')):
    width = 6 if tracks else 4
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (len(state_joints),), 'names': list(state_joints)},
        'action': {'dtype': 'float32', 'shape': (width,), 'names': ['slew', 'lift', 'tilt', 'scoop', 'trackL', 'trackR'][:width]},
        'diagnostics.requested_action': {'dtype': 'float32', 'shape': (width,), 'names': None},
        'diagnostics.pulses_us': {'dtype': 'float32', 'shape': (9,), 'names': None},
    }
    for key, (height, width) in zip(('cam1', 'cam2'), shapes):
        features[f'observation.images.{key}'] = {'dtype': 'video', 'shape': (height, width, 3), 'names': ['height', 'width', 'channels']}
        for field in CAM_INT_FIELDS:
            features[f'clock.{key}.{field}'] = {'dtype': 'int64', 'shape': (1,), 'names': None}
        for field in ('age_s', 'sdk_timestamp_ms'):
            features[f'clock.{key}.{field}'] = {'dtype': 'float64', 'shape': (1,), 'names': None}
    for key, size in POSE_FIELDS.items():
        features[feedback_key(key)] = {'dtype': 'float32', 'shape': (size,), 'names': None}
    for key in STATUS_INT_FIELDS:
        features[f'clock.{key}'] = {'dtype': 'int64', 'shape': (1,), 'names': None}
    for key in ('sample_ns', 'episode_ns', 'requested_sequence'):
        features[f'clock.{key}'] = {'dtype': 'int64', 'shape': (1,), 'names': None}
    for key in ('pair_skew_s', 'imu_age_s', 'teleop_lateness_ms', 'controller_worst_lateness_s', 'fk_work_us', 'loop'):
        features[f'clock.{key}'] = {'dtype': 'float64', 'shape': (1,), 'names': None}
    features['diagnostics.pump_on'] = {'dtype': 'int64', 'shape': (1,), 'names': None}
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
    def scalar(key, value, dtype):
        frame[key] = np.array([value], dtype=dtype)
    for key, data in zip(('cam1', 'cam2'), pixels):
        m = row[key]
        gray = np.frombuffer(data, dtype=np.uint8).reshape(m['height'], m['width'])
        frame[f'observation.images.{key}'] = np.repeat(gray[..., None], 3, axis=2)
        for field in CAM_INT_FIELDS:
            scalar(f'clock.{key}.{field}', m[field], np.int64)
        scalar(f'clock.{key}.sdk_timestamp_ms', m['sdk_timestamp_ms'], np.float64)
        scalar(f'clock.{key}.age_s', (row['sample_ns'] - m['host_monotonic_ns']) / 1e9, np.float64)
    for key in POSE_FIELDS:
        frame[feedback_key(key)] = np.asarray(s[key], dtype=np.float32).reshape(-1)
    for key in STATUS_INT_FIELDS:
        scalar(f'clock.{key}', s[key], np.int64)
    for key in ('sample_ns', 'episode_ns', 'requested_sequence'):
        scalar(f'clock.{key}', row[key], np.int64)
    values = {'pair_skew_s': (row['cam2']['host_monotonic_ns'] - row['cam1']['host_monotonic_ns']) / 1e9,
              'imu_age_s': s['imu_age_s'], 'teleop_lateness_ms': row['teleop_lateness_ms'],
              'controller_worst_lateness_s': s['worst_lateness_s'], 'fk_work_us': s['fk_work_us'],
              'loop': row['episode_ns'] / 1e9}
    for key, value in values.items():
        scalar(f'clock.{key}', value, np.float64)
    scalar('diagnostics.pump_on', int(s['pump_on']), np.int64)
    return frame


def load_session(session_path):
    session = json.loads((session_path / 'session.json').read_text())
    if session['format'] != 'kaivuri-native-episode-v1':
        raise ValueError(f'{session_path}: unsupported native episode format')
    episodes = sorted(session_path.glob('episode-*.ready'))
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
            for episode, (_, count, hz) in zip(episodes, group):
                with (episode / 'cam1.y8').open('rb') as c1, (episode / 'cam2.y8').open('rb') as c2:
                    for row in frame_rows(episode):
                        pixels = [stream.read(row[key]['pixel_bytes']) for stream, key in zip((c1, c2), ('cam1', 'cam2'))]
                        dataset.add_frame(convert_frame(row, pixels, session, state_joints))
                dataset.save_episode()
                target = native / f'episode-{index:06d}'
                target.mkdir()
                for name in ('samples.jsonl', 'episode.json'):
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
                print(f'{ep.name}: {count} frames, {hz:.3f} Hz, shapes={shape}')
    else:
        if not args.root or not args.repo_id:
            p.error('--root and --repo-id required for export')
        export(args.session, args.root, args.repo_id, joints)


if __name__ == '__main__':
    main()
