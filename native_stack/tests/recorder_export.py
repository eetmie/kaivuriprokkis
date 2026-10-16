"""Optional LeRobot 0.6.1 round trip of a recorded fixture (requires export environment).

python tests/recorder_export.py /path/to/native-session
"""
import json
from pathlib import Path
import sys
import tempfile
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from export_episodes import export, frame_rows

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

session_path = Path(sys.argv[1])
session = json.loads((session_path / 'session.json').read_text())
episodes = sorted(session_path.glob('*.ready'))
rows = [row for ep in episodes for row in frame_rows(ep)]
with tempfile.TemporaryDirectory(prefix='kvr-export-roundtrip-') as temp:
    root = Path(temp) / 'dataset'
    export(session_path, root, 'masi/native_recorder_roundtrip')
    assert (root / 'native/EXPORT_COMPLETE').exists()
    d = LeRobotDataset('masi/native_recorder_roundtrip', root=root, video_backend='pyav')
    assert len(d) == len(rows)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for index in (0, len(rows)//2, len(rows)-1):
            frame, row = d[index], rows[index]
            for key in ('cam1', 'cam2'):
                m = row[key]
                assert tuple(frame[f'observation.images.{key}'].shape) == (3, m['height'], m['width'])
                assert frame[f'clock.{key}.ir_index'].item() == m['ir_index']
                assert frame[f'clock.{key}.host_monotonic_ns'].item() == m['host_monotonic_ns']
            width = 6 if session['enable_tracks'] else 4
            np.testing.assert_allclose(frame['action'].numpy(), row['status']['applied_action'][:width], atol=1e-7)
            for name, prefix in (('fk_joint_xyz_m', 'fk'), ('fk_upper_quat_wxyz', 'fk'), ('imu_corrected_quat_wxyz', 'imu')):
                np.testing.assert_allclose(frame[f'feedback.{prefix}.{name}'].numpy(),
                                           np.asarray(row['status'][name]).reshape(-1), atol=1e-7)
    assert (root / 'native/episode-000000/samples.jsonl').read_bytes() == (episodes[0] / 'samples.jsonl').read_bytes()
print(f'LeRobot round trip passed: {len(rows)} frames; both videos, timestamps, applied actions and FK/IMU')
