"""Native recording runs inside a session, including legacy single-run sessions."""
import json
from pathlib import Path


def recording_paths(session):
    """Original run first, followed by explicitly named resumed runs in order."""
    session = Path(session)
    if not (session / 'session.json').is_file():
        raise ValueError(f'{session}: missing session.json')
    paths = [session]
    runs = session / 'recordings'
    if runs.exists():
        for path in sorted(runs.glob('run-*')):
            if not path.is_dir():
                continue
            if (path / 'session.json').is_file():
                paths.extend(recording_paths(path))
            elif list(path.glob('episode-*.ready')):
                raise ValueError(f'{path}: saved episodes have no session.json')
    return paths


def committed_episodes(session):
    """Saved takes across all runs; never include interrupted .partial takes."""
    session = Path(session)
    # Diagnostics also accept bare episode collections without recording metadata.
    if not (session / 'session.json').is_file():
        return sorted(session.glob('episode-*.ready'))
    return [ep for run in recording_paths(session) for ep in sorted(run.glob('episode-*.ready'))]


def export_recording_paths(sessions):
    """Expand session containers and deduplicate runs, skipping runs with no saved takes."""
    paths, seen = [], set()
    for session in sessions:
        for path in recording_paths(session):
            path = path.resolve()
            if path not in seen and any(path.glob('episode-*.ready')):
                paths.append(path)
                seen.add(path)
    if not paths:
        raise ValueError('no committed episodes; .partial takes are intentionally excluded')
    return paths


def recording_contract(session, provenance):
    """Settings that must stay consistent when appending training demonstrations."""
    args = provenance['camera_arguments']
    def argument(flag, default):
        return args[args.index(flag) + 1] if flag in args else default
    # Before --resolution existed, camera capture defaulted to 640x480.
    return {
        'fps': session['fps'], 'enable_slew': bool(session['enable_slew']),
        'enable_tracks': bool(session['enable_tracks']), 'depth': session.get('depth', []),
        'live_pwm': bool(session['live_pwm']), 'simulated_imu': bool(session['simulated_imu']),
        'control_hz': session['initial_status']['control_hz'],
        'camera_order': provenance['camera_order'],
        'camera_rotations_deg': provenance['camera_rotations_deg'],
        'resolution': f"{argument('--width', '640')}x{argument('--height', '480')}",
        'calibration': {name: value for name, value in provenance['files'].items()
                        if name == 'safety.yaml' or name.startswith(('configuration_files/profiles/', 'profile/'))},
    }


def check_resume(session_path, expected):
    """Check every prior run before creating directories or opening hardware."""
    paths = recording_paths(session_path)
    for path in paths:
        session = json.loads((path / 'session.json').read_text())
        if session.get('format') != 'kaivuri-native-episode-v1':
            raise ValueError(f'{path}: unsupported native episode format')
        manifest = path / 'provenance/manifest.json'
        if not manifest.is_file():
            raise ValueError(f'{path}: provenance is missing; use a new session directory')
        contract = recording_contract(session, json.loads(manifest.read_text()))
        changed = [key for key, value in expected.items() if contract.get(key) != value]
        if changed:
            raise ValueError(f'{path}: resume settings differ: {", ".join(changed)}; '
                             'use matching settings or a new session directory')
    return len(committed_episodes(session_path))
