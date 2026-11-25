"""Export an interactive comparison of two FK mask margin settings."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np


def extract(args):
    from export_episodes import depth_frames, frame_rows
    from native_sessions import committed_episodes
    from mapper import DepthProjector, FastDEM, KEYS, pose_at, render, track_pivot, update
    from replay import pose_series
    import sand_height as sh

    episodes = {json.loads((ep / "episode.json").read_text())["episode_number"]: ep
                for ep in committed_episodes(args.session)}
    ep = episodes[args.episode]
    calibration = json.loads(args.calibration.read_text())
    profile = ep.parent / "provenance/configuration_files/profiles/jetson/control_config.yaml"
    pivot = track_pivot(calibration, profile)
    projector = DepthProjector(calibration, {k: ep.parent / f"{k}.depth.json" for k in KEYS}, args)
    times, poses = pose_series(ep)
    models = [sh.ArmModel(args.arm_model, margins) for margins in (args.baseline, args.wider)]
    rows = frame_rows(ep)
    chunks = []
    args.out.mkdir(parents=True)
    with ExitStack() as stack:
        engines = [stack.enter_context(FastDEM(args.fastdem, args.config)) for _ in range(3)]
        streams = [depth_frames(ep / f"{k}.depth.zst") for k in KEYS]
        for i, (row, *data) in enumerate(zip(rows, *streams, strict=True)):
            if i % args.every:
                continue
            status = [pose_at(row[f"{k}_depth"]["host_monotonic_ns"], times, poses, pivot) for k in KEYS]
            yaw = status[0]["slew_yaw_rad"]
            variants = []
            for model in models:
                sh.ARM = model
                variants.append([projector.scan(k, d, s) for k, d, s in zip(KEYS, data, status)])
            baseline, widened = variants
            result = {"frame": i, "slew_deg": np.rad2deg(yaw),
                      "pitch_deg": np.rad2deg(status[0]["cabin_pitch_rad"])}
            for name, scans, engine in zip(("baseline", "wider", "cabin"), (baseline, widened, widened[:1]), engines):
                front, surround, _ = update(engine, scans, yaw)
                result[f"{name}_front"] = render(front)[:, :, 0].copy()
                result[f"{name}_surround"] = render(surround)[:, :, 0].copy()
            for k, before, after in zip(KEYS, baseline, widened):
                result[f"{k}_depth"] = np.round(after[3] * 1000).clip(0, 65535).astype(np.uint16)
                result[f"{k}_mask"] = np.where((after[4] == 2) & (before[4] != 2), 3, after[4]).astype(np.uint8)
            chunks.append(result)
            if len(chunks) == 60:
                save_chunk(args.out, chunks)
        if chunks:
            save_chunk(args.out, chunks)
    (args.out / "settings.json").write_text(json.dumps({
        "source": ep.name, "episode": args.episode, "every": args.every,
        "baseline_m": args.baseline, "wider_m": args.wider, "calibration": calibration,
    }, indent=2) + "\n")


def save_chunk(out, rows):
    np.savez_compressed(out / f"{rows[0]['frame']:06d}.npz",
                        **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
    rows.clear()


def record(args):
    import rerun as rr
    import rerun.blueprint as bp

    names = [("mask/cam2", "Cabin depth and FK mask"), ("mask/cam1", "Arm depth and FK mask"),
             ("baseline/front", "Fused front · original mask"), ("wider/front", "Fused front · wider mask"),
             ("baseline/surround", "Fused surround · original mask"),
             ("wider/surround", "Fused surround · wider mask")]
    blueprint = bp.Blueprint(bp.Vertical(
        bp.Grid(*(bp.Spatial2DView(origin=path, name=name) for path, name in names), grid_columns=2),
        bp.Horizontal(bp.TimeSeriesView(origin="pose", name="Slew and cabin tilt"),
                      bp.TimeSeriesView(origin="reject", name="Depth returns masked (%)")),
        row_shares=[3, 1]), bp.TimePanel(timeline="frame"))
    rr.init(args.name)
    rr.save(args.out, default_blueprint=blueprint)
    settings = json.loads((args.chunks / "settings.json").read_text())
    rr.log("settings", rr.TextDocument(
        f"Episode {settings['episode']}. Red is the original FK mask. Amber is the added rejection. "
        f"Original margins (m): {settings['baseline_m']}. Wider margins (m): {settings['wider_m']}."), static=True)
    rejected = {k: [] for k in ("cam2", "cam1")}
    for path in sorted(args.chunks.glob("*.npz")):
        with np.load(path) as data:
            for j, frame in enumerate(data["frame"]):
                rr.set_time("frame", sequence=int(frame))
                rr.set_time("recording_seconds", duration=float(frame) / 30)
                for k in rejected:
                    mask = data[f"{k}_mask"][j]
                    depth = data[f"{k}_depth"][j]
                    gray = np.where(depth == 0, 0, np.clip(255 - depth.astype(float) * .22, 25, 230)).astype(np.uint8)
                    rgb = np.repeat(gray[:, :, None], 3, axis=2)
                    rgb[mask == 2] = (230, 48, 48)
                    rgb[mask == 3] = (255, 170, 0)
                    rr.log(f"mask/{k}", rr.Image(rgb))
                    denominator = max(1, np.count_nonzero(mask))
                    before = 100 * np.count_nonzero(mask == 2) / denominator
                    after = 100 * np.count_nonzero(mask >= 2) / denominator
                    rr.log(f"reject/{k}/original", rr.Scalars(before))
                    rr.log(f"reject/{k}/wider", rr.Scalars(after))
                    rejected[k].append([before, after])
                for mode in ("baseline", "wider", "cabin"):
                    for view in ("front", "surround"):
                        rr.log(f"{mode}/{view}", rr.Image(data[f"{mode}_{view}"][j]))
                for key in ("slew_deg", "pitch_deg"):
                    rr.log(f"pose/{key}", rr.Scalars(data[key][j]))
    rr.disconnect()
    print(json.dumps({k: {"median_pct": np.median(v, axis=0).tolist(), "frames": len(v)}
                      for k, v in rejected.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    a = sub.add_parser("extract")
    a.add_argument("session", type=Path)
    for key in ("calibration", "fastdem", "config", "arm-model", "out"):
        a.add_argument(f"--{key}", type=Path, required=True)
    a.add_argument("--episode", type=int, required=True)
    a.add_argument("--every", type=int, default=3)
    a.add_argument("--stride", type=int, default=4)
    a.add_argument("--arm-radius-m", type=float, default=.11)
    a.add_argument("--baseline", type=float, nargs=3, default=(.025, .025, .080))
    a.add_argument("--wider", type=float, nargs=3, default=(.045, .045, .100))
    a.set_defaults(run=extract)
    b = sub.add_parser("record")
    b.add_argument("chunks", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--name", required=True)
    b.set_defaults(run=record)
    args = parser.parse_args()
    args.run(args)
