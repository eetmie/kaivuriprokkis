"""Detect tagStandard41h12 AprilTags in a recorded episode's side-camera IR stream (cam2.y8).

    sidecam_tags.py SESSION/episode-N.ready [--out FILE] [--workers 8]

Writes one row per detection: frame, id, hamming, decision_margin, centre x/y, then the four
corners (x, y) in pupil-apriltags order. The default output is ../sidecam/SESSION/episode-N.tags.npy
beside the sessions directory, so the raw session and the session list stay untouched.
Needs pupil-apriltags (on the board: ~/kaivuriprokkis/.venv-tags).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np

from export_episodes import samples

FAMILY = 'tagStandard41h12'


def default_tags(episode):
    session = episode.parent
    return session.parent.parent / 'sidecam' / session.name / (episode.name.removesuffix('.ready') + '.tags.npy')


def frame_shape(episode):
    for row in samples(episode):
        if row.get('cam2'):
            return row['cam2']['height'], row['cam2']['width']
    raise SystemExit('no cam2 metadata in samples.jsonl')


def detect(args):
    path, shape, frames = args
    import pupil_apriltags
    detector = pupil_apriltags.Detector(families=FAMILY, quad_decimate=1.0, refine_edges=True, nthreads=1)
    images = np.memmap(path, np.uint8, 'r').reshape(-1, *shape)
    rows = []
    for f in frames:
        for t in detector.detect(np.ascontiguousarray(images[f])):
            rows.append([f, t.tag_id, t.hamming, t.decision_margin, *t.center, *t.corners.ravel()])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('episode', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    episode = a.episode.resolve(); shape = frame_shape(episode); video = episode / 'cam2.y8'
    n = video.stat().st_size // (shape[0] * shape[1])
    out = a.out or default_tags(episode)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    chunks = [(str(video), shape, range(i, min(n, i + 100))) for i in range(0, n, 100)]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = np.array([r for chunk in pool.map(detect, chunks) for r in chunk]).reshape(-1, 14)
    np.save(out, rows)
    print(f'{n} frames in {time.time() - start:.0f} s -> {out}')
    for i in np.unique(rows[:, 1]).astype(int):
        m = rows[:, 1] == i
        print(f'  id {i:3d}: {m.sum():5d} frames ({100 * m.sum() / n:3.0f}%), decision margin median {np.median(rows[m, 3]):.0f}')


if __name__ == '__main__':
    main()
