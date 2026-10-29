from pathlib import Path
import json, time
import numpy as np
from lerobot_vla.camera import D435iCamera, check_frames
from lerobot_vla.record_episodes import build_features, clock_fields
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig
encoder = RGBEncoderConfig(vcodec="h264", crf=18, preset="veryfast")

root = Path('results/dual-ir-check-20261027')
root.mkdir(parents=True, exist_ok=True)
dataset_root = root / 'camera-only-test-dataset'
keys = ['observation.images.cam1', 'observation.images.cam2']
cam = D435iCamera()
summary = {'purpose': 'Camera and encoder test only. State/actions are dummy zeros; no robot controller is opened.', 'episodes': []}
try:
    cam.start()
    previous = (0., 0.)
    master = 0.
    for run, duration in enumerate([30, 10]):
        ds = (LeRobotDataset.create('test/dual_ir_camera_only', root=dataset_root,
                robot_type='camera_test_only', fps=30, features=build_features(480, 640),
                use_videos=True, rgb_encoder=encoder, streaming_encoding=True) if run == 0
              else LeRobotDataset.resume('test/dual_ir_camera_only', root=dataset_root,
                rgb_encoder=encoder, streaming_encoding=True))
        start = time.perf_counter(); count = skips = 0
        add_ms, skews, gaps, master_gaps = [], [], [], []
        try:
            while time.perf_counter() - start < duration:
                new = cam.wait_for_next(master, .2)
                if not new:
                    raise RuntimeError('No new cam1 frame')
                a, ta, b, tb = cam.get_frames()
                master = ta
                obs = {keys[0]: a, keys[1]: b, 'img_ts': ta, 'cam2_ts': tb,
                       'state_ts': None, 'imu_device_us': None}
                check_frames(obs, keys)
                if ta <= previous[0] or tb <= previous[1]:
                    skips += 1
                    continue
                tick = time.perf_counter()
                frame = {keys[0]: a, keys[1]: b,
                         'observation.state': np.zeros(3, np.float32),
                         'action': np.zeros(4, np.float32),
                         'task': 'camera-only-test-' + str(run)}
                frame.update(clock_fields(obs, tick, start))
                t = time.perf_counter(); ds.add_frame(frame)
                add_ms.append((time.perf_counter()-t)*1000)
                skews.append(abs(ta-tb)*1000)
                if count:
                    gaps.append((ta-previous[0])*1000)
                previous = ta,tb
                count += 1
                if count % 150 == 0:
                    print(f'[test] run {run}: {count} frames, {count/(time.perf_counter()-start):.2f} Hz', flush=True)
            elapsed = time.perf_counter()-start
            ds.save_episode()
            summary['episodes'].append({'frames': count, 'elapsed_s': elapsed, 'fps': count/elapsed,
                'skipped_repeated_pairs': skips, 'add_ms_p50_p99_max': np.percentile(add_ms,[50,99,100]).tolist(),
                'pair_skew_ms_max': max(skews), 'cam1_gap_ms_max': max(gaps)})
            from PIL import Image
            Image.fromarray(np.concatenate([a,b], axis=1)).save(root / f'views-{run}.jpg')
        finally:
            ds.finalize()
    import av
    videos = []
    for path in dataset_root.rglob('*.mp4'):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = sum(1 for f in container.decode(video=0))
            videos.append({'path': str(path.relative_to(dataset_root)), 'codec': stream.codec_context.name,
                           'frames': frames, 'width': stream.width, 'height': stream.height})
    summary['videos'] = videos
    expected_frames = sum(e['frames'] for e in summary['episodes'])
    for key in ['cam1', 'cam2']:
        assert sum(v['frames'] for v in videos if f'images.{key}/' in v['path']) == expected_frames
    import pyarrow.parquet as pq
    tables = [pq.read_table(p).to_pydict() for p in dataset_root.glob('data/**/*.parquet')]
    assert sum(len(t['index']) for t in tables) == sum(e['frames'] for e in summary['episodes'])
    summary['status'] = 'passed'
finally:
    cam.stop()
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
