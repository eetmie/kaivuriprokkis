"""Record 5 Hz fused heightmaps beside the native recorder's committed IR takes."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from bench_live import observe
from frame_reader import FrameReader
from mapper import (KEYS, UNKNOWN_MM, DepthProjector, FastDEM, pipeline_defaults,
                    projection_arguments, track_pivot, update)
from sand_height_live import StatusRing
import sand_height as sh


def millimetres(height):
    return np.where(np.isfinite(height), np.round(height * 1000), UNKNOWN_MM).astype(np.int16)


def save_episode(episode, rows, args):
    info = json.loads((episode / "episode.json").read_text())
    rows = [row for row in rows if info["start_ns"] <= min(row[1]) and row[0] <= info["end_ns"]]
    if not rows:
        raise RuntimeError(f"{episode.name}: no in-window heightmaps")
    publish, stamps, sequence, front, surround = map(np.stack, zip(*rows))
    data = episode / "heightmaps.npz"
    with data.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, publish_host_ns=publish, depth_host_ns=stamps,
                            depth_sequence=sequence, front_mm=front, surround_mm=surround)
        f.flush()
        os.fsync(f.fileno())
    data.with_suffix(".tmp").replace(data)
    fd = os.open(episode, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return len(rows)


def run(args):
    calibration = json.loads(args.calibration.read_text())
    serials = dict(zip(KEYS, args.serials))
    sh.load_arm(args)
    projector = DepthProjector(
        calibration, {k: args.frames / f"{serials[k]}-depth.json" for k in KEYS}, args)
    pivot = track_pivot(calibration, args.profile / "control_config.yaml")
    alive = True

    def stop(*_):
        nonlocal alive
        alive = False

    signal.signal(signal.SIGTERM, stop)
    period = round(1e9 / args.map_hz)
    with ExitStack() as resources:
        readers = [resources.enter_context(FrameReader(args.frames / f"{serials[k]}-depth.frame",
                                                       args.library, 640 * 480 * 2, .1)) for k in KEYS]
        ring = StatusRing(str(args.socket), 100)
        ring.start()

        def close_status():
            ring.stop.set()
            ring.join(timeout=2)
            ring.client.close()

        resources.callback(close_status)
        engine, active, rows = None, None, []
        recorded = set()
        args.ready.touch()
        next_map = 0
        try:
            while alive:
                if ring.error:
                    raise RuntimeError(ring.error)
                candidates = sorted(args.root.glob("episode-*.partial"))
                # An old interrupted take can remain .partial. A new take is always last.
                partial = candidates[-1] if candidates and candidates[-1] not in recorded else None
                if active is not None and (not active.exists() or (partial and partial != active)):
                    ready = active.with_suffix(".ready")
                    if ready.is_dir():
                        count = save_episode(ready, rows, args)
                        print(f"{ready.name}: saved {count} fused map observations", flush=True)
                    recorded.add(active)
                    engine.__exit__(None, None, None)
                    engine, active, rows = None, None, []
                if partial is not None and active is None and partial not in recorded:
                    active = partial
                    previous = [-1] * len(KEYS)
                    engine = FastDEM(args.fastdem, args.config).__enter__()
                    next_map = time.monotonic_ns()
                now = time.monotonic_ns()
                if active is not None and now >= next_map:
                    next_map = now + period
                    with ring.lock:
                        poses = [p for p in ring.ring if p["fk_valid"] and p["imu_healthy"]
                                 and p["fk_cabin_tilt"]]
                    obs = observe(readers, previous, poses, args.max_fk_age_ms, pivot)
                    if not isinstance(obs, str):
                        scans = [projector.scan(k, data, pose) for k, (_, data, pose) in zip(KEYS, obs)]
                        front, surround, _ = update(engine, scans, obs[0][2]["slew_yaw_rad"])
                        previous = [meta.sequence for meta, _, _ in obs]
                        rows.append((time.monotonic_ns(), [meta.host_monotonic_ns for meta, _, _ in obs],
                                     previous, millimetres(front), millimetres(surround)))
                time.sleep(.01 if active is None else min(.01, max(0, (next_map-time.monotonic_ns()) / 1e9)))
        finally:
            if active is not None:
                ready = active.with_suffix(".ready")
                if ready.is_dir():
                    count = save_episode(ready, rows, args)
                    print(f"{ready.name}: saved {count} fused map observations", flush=True)
                engine.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "frames", "socket", "library", "profile", "calibration", "fastdem", "config", "ready"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--serials", nargs=2, required=True, metavar=("CABIN", "ARM"))
    parser.add_argument("--map-hz", type=float)
    parser.add_argument("--max-fk-age-ms", type=float)
    projection_arguments(parser)
    pipeline_defaults(parser, ("runtime", "projection"))
    args = parser.parse_args()
    if not 0 < args.map_hz <= 5:
        parser.error("recording map rate must be in (0, 5] Hz")
    run(args)


if __name__ == "__main__":
    main()
