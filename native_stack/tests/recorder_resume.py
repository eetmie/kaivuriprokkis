"""Dry synthetic resume: immutable originals, task attribution, compatibility and numbering."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

build, source = map(Path, sys.argv[1:])
sys.path.insert(0, str(source / 'tools'))
from native_sessions import committed_episodes, export_recording_paths, recording_paths
from export_episodes import load_session, validate_episode
from view_native import wait_episode


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


with tempfile.TemporaryDirectory(prefix='recorder-resume-') as tmp:
    root = Path(tmp) / 'session'
    base = [sys.executable, '-B', str(source / 'tools/record_native.py'), '--session', str(root),
            '--build', str(build), '--synthetic', '--bench-seconds', '1', '--depth', 'none']
    first_task, second_task = 'scoop "sand" into the bin', 'level the sand → left'
    def run(*args, error=None):
        result = subprocess.run([*base, *args], capture_output=True, text=True, timeout=35)
        if error:
            assert result.returncode != 0 and error in result.stderr, (result.stdout, result.stderr)
        else:
            assert result.returncode == 0, (result.stdout, result.stderr)
        return result

    run('--task', first_task, '--bench-episodes', '2')
    original_episodes = committed_episodes(root)
    assert len(original_episodes) == 2
    # Simulate a pre-resume recorder session: older summaries lack display numbers.
    for ep in original_episodes:
        path = ep / 'episode.json'
        summary = json.loads(path.read_text())
        summary.pop('episode_number')
        path.write_text(json.dumps(summary) + '\n')
    partial = root / 'episode-0.partial'
    partial.mkdir()
    (partial / 'samples.jsonl').write_text('interrupted take\n')
    original = hashes(root)

    run('--task', second_task, error='use --resume')
    assert hashes(root) == original
    for flags, key in ((['--fps', '15'], 'fps'), (['--resolution', '848x480'], 'resolution'),
                       (['--enable-slew'], 'enable_slew'), (['--enable-tracks'], 'enable_tracks'),
                       (['--depth', 'cam2'], 'depth'), (['--control-hz', '100'], 'control_hz'),
                       (['--no-rotate-camera1'], 'camera_rotations_deg'),
                       (['--camera2', 'different:1'], 'camera_order')):
        run('--resume', '--task', second_task, *flags, error=f'resume settings differ: {key}')
        assert hashes(root) == original
    with (root.parent / f'.{root.name}.recording.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run('--resume', '--task', second_task, error='already recording or being backed up')
    run('--resume', '--session', str(root.parent / 'missing'), '--task', second_task,
        error='requires an existing native session')
    assert hashes(root) == original

    result = run('--resume', '--task', second_task)
    assert 'Saved episodes: 2; next ep 3' in result.stdout
    assert 'Recording ep 3:' in result.stdout and 'Saved ep 3:' in result.stdout
    result = run('--resume', '--task', first_task, '--exposure-us', '2000', '--gain', '16')
    assert 'Saved episodes: 3; next ep 4' in result.stdout
    assert 'Recording ep 4:' in result.stdout and 'Saved ep 4:' in result.stdout
    current = hashes(root)
    assert {key: current[key] for key in original} == original
    assert (partial / 'samples.jsonl').read_text() == 'interrupted take\n'

    paths = recording_paths(root)
    episodes = committed_episodes(root)
    assert len(paths) == 3 and len(episodes) == 4
    assert export_recording_paths([root, paths[1]]) == paths
    assert [json.loads((path/'session.json').read_text())['task'] for path in paths] == [first_task, second_task, first_task]
    assert '--exposure-us' in json.loads((paths[-1]/'provenance/manifest.json').read_text())['camera_arguments']
    assert [json.loads((ep/'episode.json').read_text()).get('episode_number') for ep in episodes] == [None, None, 3, 4]
    for index, ep in enumerate(episodes):
        assert wait_episode(root, index, False) == ep
        assert ep.name != partial.name
    for path in paths:
        session, group = load_session(path)
        for ep in group:
            validate_episode(ep, session)
    # Check per-frame task attribution through the actual exporter, without heavy video deps.
    import export_episodes
    original_create = export_episodes.create_dataset
    captured, saved_tasks = [], []
    class Dataset:
        def add_frame(self, frame):
            captured.append(frame['task'])
        def save_episode(self):
            saved_tasks.append(captured[-1])
        def finalize(self):
            pass
    # Real LeRobot export/readback is tested separately in recorder_export.py.
    export_episodes.create_dataset = lambda repo_id, target, fps, features: (target.mkdir(), Dataset())[1]
    # convert_frame imports NumPy; keep this CPU test runnable with the stdlib interpreter.
    original_convert = export_episodes.convert_frame
    export_episodes.convert_frame = lambda row, pixels, session, joints: {'task': session['task']}
    target = root.parent / 'export'
    try:
        export_episodes.export(root, target, 'test/resume')
    finally:
        export_episodes.create_dataset = original_create
        export_episodes.convert_frame = original_convert
    assert saved_tasks == [first_task, first_task, second_task, first_task], saved_tasks
    assert (target/'native/EXPORT_COMPLETE').exists()
    assert all((target/f'native/session-{i:03d}/provenance/manifest.json').exists() for i in range(3))
    for i, ep in enumerate(episodes):
        assert (target/f'native/episode-{i:06d}/samples.jsonl').read_bytes() == (ep/'samples.jsonl').read_bytes()
print('Resume opt-in, immutable legacy takes, per-run tasks/provenance, compatibility, lock and numbering passed')
