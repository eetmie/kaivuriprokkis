"""Offline acceptance: native byte replay -> training.drive_logs.prepare -> known q/v/u.

Requires NumPy/SciPy and an explicit Isaac-hydraulic-actuator checkout. It never
loads a model or drives hardware. Keep --out outside real training datasets.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import mlp_integration as replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    sys.path.insert(0, str(args.training_root.resolve()))
    from training import drive_logs
    from hydraulic_controller.sensors import CARRIAGE_PITCH_ZERO, MOUNT_PITCH

    replay.BINARY, replay.PROFILE = args.binary.resolve(), args.profile.resolve()
    rows, _ = replay.run(args.out/'replay', seconds=6, chunk=7)
    (path,) = (args.out/'replay').glob('drive_log_*.csv')
    drive_logs.prepare(SimpleNamespace(logs=[path], split='dev', out=args.out/'prepared',
                                      min_seconds=3., target_v='gyro'))
    manifest = json.loads((args.out/'prepared/drive_logs_dev.json').read_text())
    (entry,) = manifest['recordings']
    assert entry['source'] == 'native_mlp' and entry['excited_rows'] > 500
    assert entry['source_session_id'] and entry['excitation']['blocks']
    stamps = np.array([float(row['timestamp']) for row in rows])
    command = np.array([[float(row[f'combined_cmd_{name}']) for name in ('lift','tilt','scoop')] for row in rows])
    device = np.array([float(row['state_imu_ts_us']) for row in rows])
    maxima = dict(q=0., v=0., u=0.)
    count = 0
    for filename in entry['paths']:
        with np.load(args.out/'prepared'/filename) as chunk:
            held = np.searchsorted(stamps, chunk['t'], side='right') - 1
            assert (stamps[held] <= chunk['t']).all()
            t = (device[held] - (2**32 - 500000)) / 1e6
            pitch = np.stack([.002*np.sin(.8*t), -.3+.05*np.sin(1.1*t)+.24173176,
                              .9+.03*np.cos(.7*t)+.010681414, .2+.04*np.sin(.9*t)], axis=-1) - MOUNT_PITCH
            q = np.c_[np.diff(pitch, axis=-1), pitch[:,0]-CARRIAGE_PITCH_ZERO]
            rate = np.stack([.0016*np.cos(.8*t), .055*np.cos(1.1*t),
                             -.021*np.sin(.7*t), .036*np.cos(.9*t)], axis=-1)
            v = np.c_[np.diff(rate, axis=-1), rate[:,0]]
            for name, expected in (('q',q), ('v',v), ('u',command[held])):
                error = float(np.max(np.abs(chunk[name]-expected)))
                maxima[name] = max(maxima[name], error)
                np.testing.assert_allclose(chunk[name], expected, rtol=0, atol=4e-7)
            np.testing.assert_array_equal(chunk['target_v'], chunk['v'])
            count += len(chunk['t'])
    assert count > 500
    report = {'samples': count, 'dt_s': .01, 'max_absolute_error': maxima,
              'source': entry['source'], 'source_session_id': entry['source_session_id'],
              'sensor_contract': manifest['sensors']}
    (args.out/'roundtrip_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
