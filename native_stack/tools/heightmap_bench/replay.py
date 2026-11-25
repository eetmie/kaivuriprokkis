"""Export the complete native session as two synchronized offline LeRobot datasets."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from export_episodes import depth_frames, frame_rows, samples
from native_sessions import committed_episodes
import sand_height as sh


from mapper import (
    ANNULUS, FAR_RADIUS, FRONT_CELL, FRONT_SIZE, HEIGHT_LIMIT, KEYS, MODES, NEAR_RADIUS,
    SURROUND_CELL, SURROUND_SIZE, UNKNOWN_MM, DepthProjector, FastDEM, pipeline_defaults, pose_at, projection_arguments,
    render, track_pivot, update,
)


def pose_series(episode):
    poses = {}
    for row in samples(episode):
        st = row["status"]
        if st["fk_valid"] and st["imu_healthy"]:
            poses[st["host_monotonic_ns"]] = st
    times = sorted(poses)
    if not times:
        raise ValueError(f"{episode}: no valid poses")
    return np.asarray(times), [poses[t] for t in times]


def fk_fresh(stamp, times, max_age_ms):
    """Same test as bench_live.observe: both FK neighbours within max_age_ms of the depth stamp."""
    after = int(np.searchsorted(times, stamp))
    return bool(np.all(np.abs(times[max(0, after - 1):after + 1] - stamp) <= max_age_ms * 1e6))


def uncompensated(world, status, ground_offset, pivot):
    """Undo the recorded tilt for comparison with the same points and FK mask."""
    tilt = sh.quat_matrix(status["cabin_tilt_quat_wxyz"])
    joint = np.asarray(status["fk_joint_xyz_m"][0])
    anchor = joint + [0, 0, ground_offset]
    axis = tilt[:, 0]
    end = -pivot if axis[2] > 0 else pivot
    lift = end * (np.array([1, 0, 0]) - axis)
    level_anchor = (joint - lift) @ tilt
    return (world - anchor) @ tilt + level_anchor + [0, 0, ground_offset]


def millimetres(height):
    result = np.full(height.shape, UNKNOWN_MM, dtype=np.int16)
    known = np.isfinite(height)
    result[known] = np.round(1000 * height[known]).astype(np.int16)
    return result


def inspection(scan_data, st, counts, uncomp, comp):
    canvas = Image.new("RGB", (640, 256), (14, 19, 26))
    draw = ImageDraw.Draw(canvas)
    for col, (name, (depth, mask)) in enumerate(zip(("Cabin depth", "Arm depth"), scan_data)):
        x = col * 320
        gray = np.clip(242 - (depth - .12) * 250, 18, 242).astype(np.uint8)
        gray[depth <= 0] = 0
        rgb = np.repeat(gray[..., None], 3, axis=2)
        rgb[mask == 2] = [238, 82, 57]
        canvas.paste(Image.fromarray(rgb).resize((320, 210), Image.Resampling.NEAREST), (x, 22))
        draw.text((x + 7, 5), f"{name}  /  {counts[col]:.0f}% FK arm rejects", fill="white")
    yaw, roll, pitch = np.rad2deg([st["slew_yaw_rad"], st["cabin_roll_rad"], st["cabin_pitch_rad"]])
    draw.text((7, 237), f"Slew {yaw:+.1f} deg  /  roll {roll:+.1f}  pitch {pitch:+.1f}  /  "
                        f"cab median |height| {uncomp:.1f} -> {comp:.1f} mm", fill="white")
    return np.asarray(canvas)


class ReviewVideo:
    def __init__(self, path, fps):
        self.file = av.open(str(path), mode="w")
        self.stream = self.file.add_stream("libx264", rate=fps, options={"crf": "18", "preset": "veryfast"})
        self.stream.width, self.stream.height = 800, 528
        self.stream.pix_fmt = "yuv420p"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for packet in self.stream.encode():
            self.file.mux(packet)
        self.file.close()

    def add(self, diagnostic, maps, yaw):
        page = Image.new("RGB", (800, 528), (14, 19, 26))
        draw = ImageDraw.Draw(page)
        for i, (title, height) in enumerate(zip(
                ("Cabin forward", "Fused forward", "Cabin FastDEM", "Fused FastDEM"), maps)):
            x = 200*i + 4
            draw.text((x + 2, 8), title, fill="white")
            image = Image.fromarray(render(height)).resize((192, 192), Image.Resampling.NEAREST)
            page.paste(image, (x, 30))
            if i >= 2:
                centre = np.array([x + 96, 126])
                end = centre - 22 * np.array([np.sin(yaw), np.cos(yaw)])
                draw.line((tuple(centre), tuple(end)), fill=(100, 215, 255), width=2)
            draw.text((x + 2, 227), f"{np.isfinite(height).sum():,} observed cells", fill="white")
        page.paste(Image.fromarray(diagnostic), (80, 252))
        draw.text((84, 511), "Black = unknown   /   mid-gray = flat ground   /   cyan = cabin heading (review only)",
                  fill=(210, 223, 234))
        frame = av.VideoFrame.from_ndarray(np.asarray(page), format="rgb24")
        for packet in self.stream.encode(frame):
            self.file.mux(packet)


def create_dataset(root, fps, mode, action_size):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    def vector(dtype, names):
        return {"dtype": dtype, "shape": (len(names),), "names": names}

    def video(height, width):
        return {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}

    features = {
        "observation.state": vector("float32", ["slew", "lift", "tilt", "scoop"]),
        "action": vector("float32", ["slew", "lift", "tilt", "scoop", "trackL", "trackR"][:action_size]),
        "observation.images.heightmap_front": video(64, 64),
        "observation.images.heightmap_surround": video(160, 160),
        "diagnostics.mapping": vector("float32", [
            "cabin_roll_deg", "cabin_pitch_deg", "cabin_reject_pct", "arm_reject_pct",
            "front_observed_fraction", "surround_ring_observed_fraction",
            "cabin_raw_tilt_p50_abs_mm", "cabin_compensated_p50_abs_mm",
        ]),
        "clock.depth_host_ns": vector("int64", ["cabin", "arm"]),
        "clock.native_frame": vector("int64", ["index"]),
    }
    return LeRobotDataset.create(
        repo_id=f"local/{mode}", root=root, fps=fps, robot_type="masi_excavator_native",
        features=features, use_videos=True, streaming_encoding=True, encoder_threads=1,
        video_backend="pyav", rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=0),
    )


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "settings.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    cal = json.loads(args.calibration.read_text())
    sh.load_arm(args)
    sources = [ep for session in args.session for ep in committed_episodes(session)]
    if args.episode is not None:
        sources = [sources[args.episode]]
    first = json.loads((sources[0].parent / "session.json").read_text())
    fps = first["fps"]
    action_size = 6 if first["enable_tracks"] else 4
    ds = [create_dataset(args.out / mode, fps, mode, action_size) for mode in MODES]
    report = []
    try:
        for ep_idx, episode in enumerate(sources):
            print(f"Replay {episode.name}", flush=True)
            ep = json.loads((episode / "episode.json").read_text())
            session = json.loads((episode.parent / "session.json").read_text())
            if session["fps"] != fps or (6 if session["enable_tracks"] else 4) != action_size:
                raise ValueError(f"{episode}: incompatible sample rate or actions")
            projector = DepthProjector(cal, {k: episode.parent / f"{k}.depth.json" for k in KEYS}, args)
            times, poses = pose_series(episode)
            profile = episode.parent / "provenance/configuration_files/profiles/jetson/control_config.yaml"
            pivot = track_pivot(cal, profile)
            buffer = [[], []]
            stats = []
            last_maps = None
            period, next_map, updates = round(1e9 / args.map_hz) if args.map_hz else 0, 0, 0
            unknown = lambda n: np.full((n, n), np.nan, np.float32)
            held = (unknown(FRONT_SIZE), unknown(SURROUND_SIZE), np.full((SURROUND_SIZE,) * 2, 65535, np.uint16)) * 2
            map_stamps = np.zeros(len(KEYS), np.int64)  # depth host ns the held maps came from; 0 = none yet
            with ExitStack() as stack:
                engines = [stack.enter_context(FastDEM(args.fastdem, args.config)) for _ in MODES]
                review = stack.enter_context(ReviewVideo(args.out / f"episode-{ep_idx:03d}-review.mp4", fps))
                streams = [depth_frames(episode / f"{k}.depth.zst") for k in KEYS]
                for i, (row, cab, arm) in enumerate(zip(frame_rows(episode), *streams, strict=True)):
                    if args.limit and i == args.limit:
                        break
                    timestamps = [row[f"{k}_depth"]["host_monotonic_ns"] for k in KEYS]
                    status = [pose_at(t, times, poses, pivot) for t in timestamps]
                    if not all(st["fk_cabin_tilt"] for st in status):
                        raise ValueError(f"{episode}: expected gravity-aligned FK")
                    pc, wc, tc, dc, mc = projector.scan("cam2", cab, status[0])
                    pa, wa, ta, da, ma = projector.scan("cam1", arm, status[1])
                    st = status[0]
                    yaw = st["slew_yaw_rad"]
                    scans = [(pc, wc, tc, dc, mc), (pa, wa, ta, da, ma)]
                    # Live cadence: log_heightmaps.py maps the newest frame every 1/map_hz and
                    # skips a tick (waiting a full period) when FK around either depth stamp is stale.
                    if not args.map_hz or timestamps[0] >= next_map:
                        next_map = timestamps[0] + period
                        if all(fk_fresh(t, times, args.max_fk_age_ms) for t in timestamps):
                            held = (*update(engines[0], scans[:1], yaw), *update(engines[1], scans, yaw, args.arm_min_view_deg, args.arm_min_forward_m))
                            updates += 1
                            map_stamps = np.asarray(timestamps, np.int64)
                    front, cabin_ring, cabin_age, fused, fused_ring, fused_age = held
                    surrounds = [(cabin_ring, cabin_age), (fused_ring, fused_age)]
                    counts = [100 * np.count_nonzero(m == 2) / max(1, np.count_nonzero(m)) for m in (mc, ma)]
                    no_tilt = uncompensated(wc, st, projector.mounts["cam2"].d, pivot)
                    comp = np.median(np.abs(wc[:, 2])) * 1000 if len(wc) else 0
                    uncomp = np.median(np.abs(no_tilt[:, 2])) * 1000 if len(wc) else 0
                    screen = inspection(((dc, mc), (da, ma)), st, counts, uncomp, comp)
                    review.add(screen, (front, fused, surrounds[0][0], surrounds[1][0]), yaw)
                    values = []
                    for j, (current, (remembered, age)) in enumerate(zip((front, fused), surrounds)):
                        metrics = [
                            np.rad2deg(st["cabin_roll_rad"]), np.rad2deg(st["cabin_pitch_rad"]), *counts,
                            np.isfinite(current).mean(), np.isfinite(remembered).sum() / ANNULUS.sum(), uncomp, comp,
                        ]
                        values.append(metrics)
                        ds[j].add_frame({
                            "task": session["task"],
                            "observation.state": np.rad2deg(row["status"]["angles_rad"]).astype(np.float32),
                            "action": np.asarray(row["status"]["applied_action"][:action_size], dtype=np.float32),
                            "observation.images.heightmap_front": render(current),
                            "observation.images.heightmap_surround": render(remembered),
                            "diagnostics.mapping": np.asarray(metrics, dtype=np.float32),
                            "clock.depth_host_ns": np.asarray(timestamps, dtype=np.int64),
                            "clock.native_frame": np.asarray([row["frame_index"]], dtype=np.int64),
                        })
                        buffer[j].append((millimetres(current), millimetres(remembered), age, map_stamps))
                        if len(buffer[j]) == 150:
                            save_chunk(args.out / MODES[j], ep_idx, i + 1 - len(buffer[j]), buffer[j])
                    stats.append(values)
                    last_maps = surrounds
                    if (i + 1) % 250 == 0:
                        print(f"  {i+1}/{ep['frames']} frames", flush=True)
            count = len(stats)
            if not args.limit and count != ep["frames"]:
                raise ValueError(f"{episode}: expected {ep['frames']} frames, got {count}")
            for j, dataset in enumerate(ds):
                if buffer[j]:
                    save_chunk(args.out / MODES[j], ep_idx, count - len(buffer[j]), buffer[j])
                dataset.save_episode()
            arr = np.asarray(stats)
            record = {"source": episode.name, "frames": count, "map_updates": updates, "track_end_pivot_m": pivot,
                      "median_diagnostics": {mode: np.median(arr[:, j], axis=0).tolist() for j, mode in enumerate(MODES)},
                      "final_surround_cells": {m: int(np.isfinite(last_maps[j][0]).sum()) for j,m in enumerate(MODES)},
                      "arm_reject_pct_p50_p90_max": np.percentile(arr[:, 0, 2:4], [50, 90, 100], axis=0).tolist()}
            report.append(record)
            print(json.dumps(record), flush=True)
    finally:
        for dataset in ds:
            dataset.finalize()
    info = {"fps": fps, "map_hz": args.map_hz, "max_fk_age_ms": args.max_fk_age_ms, "fastdem_config": args.config.read_text(), "calibration": cal,
            "height_color_limit_m": HEIGHT_LIMIT, "ground_z_m": 0, "unknown_mm": UNKNOWN_MM,
            "front": {"x_m": [.10, .90], "y_m": [-.40, .40], "cell_m": FRONT_CELL, "frame": "level_cabin_yaw"},
            "surround": {"xy_m": [-.8, .8], "cell_m": SURROUND_CELL, "radius_m": [NEAR_RADIUS, FAR_RADIUS],
                         "frame": "level_undercarriage", "unknown_age_frames": 65535},
            "axes": "row decreases X, column decreases Y", "episodes": report}
    for mode in MODES:
        (args.out / mode / "meta" / "heightmaps.json").write_text(json.dumps(info, indent=2) + "\n")
    (args.out / "report.json").write_text(json.dumps(info, indent=2) + "\n")


def save_chunk(root, ep, start, buffer):
    directory = root / "heightmaps" / f"episode-{ep:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    front, surround, age, stamps = (np.stack(a) for a in zip(*buffer))
    np.savez_compressed(directory / f"frame-{start:06d}.npz", front_mm=front, surround_mm=surround,
                        surround_age_frames=age, map_depth_host_ns=stamps)
    buffer.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, nargs="+", help="one or more native sessions, in episode order")
    parser.add_argument("--calibration", type=Path, required=True)
    projection_arguments(parser)
    parser.add_argument("--fastdem", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--map-hz", type=float, help="live map rate; 0 maps every depth frame")
    parser.add_argument("--max-fk-age-ms", type=float)
    pipeline_defaults(parser, ("projection", "runtime"))
    args = parser.parse_args()
    if not 0 <= args.map_hz <= 30:
        parser.error("--map-hz must be in [0, 30]")
    if not np.isfinite(args.arm_radius_m) or args.arm_radius_m < 0:
        parser.error("--arm-radius-m must be finite and nonnegative")
    run(args)
