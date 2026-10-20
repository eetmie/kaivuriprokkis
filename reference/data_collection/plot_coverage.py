#!/usr/bin/env python3
"""Offline pose-coverage PNGs from drive logs; never imports robot hardware.

Requires numpy, matplotlib and PyYAML. See COVERAGE.md for frame and data rules.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

JOINTS = ("boom", "arm", "bucket")
POSITIONS = tuple(f"joint_pos_{j}" for j in JOINTS)
VELOCITIES = tuple(f"joint_vel_{j}" for j in JOINTS)
COMMANDS = tuple(f"combined_cmd_{j}" for j in ("lift", "tilt", "scoop"))
QUALITY = ("cmd_stale", "cmd_age_s", "state_age_s", "sample_idx")


def load_profile(path):
    """Accept a control YAML or the robot_geometry.json exported with a bundle."""
    with Path(path).open() as stream:
        profile = yaml.safe_load(stream)
    joints = profile["robot"]["joints"]
    if [j["name"] for j in joints] != ["slew", *JOINTS]:
        raise ValueError("Expected a slew/boom/arm/bucket chain")
    expected = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    if not np.allclose([j["axis"] for j in joints], expected):
        raise ValueError("The X/Z plot requires positive Z slew and aligned positive Y arm axes")
    offsets = np.asarray([j["parent_to_joint_xyz"] for j in joints], dtype=float)
    tip = np.asarray(profile["robot"]["tool"]["parent_to_tip_xyz"], dtype=float)
    limits = np.asarray(profile["ik"]["joint_limits_relative"][1:4], dtype=float)
    if offsets.shape != (4, 3) or tip.shape != (3,) or limits.shape != (3, 2):
        raise ValueError("Expected four XYZ offsets, an XYZ tip and three joint limits")
    if not all(np.isfinite(x).all() for x in (offsets, tip, limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Geometry and ascending joint limits must be finite")
    pitch_offset = float(profile["robot"]["tool"].get("pitch_offset_rad", 0))
    if not np.isfinite(pitch_offset):
        raise ValueError("Bucket pitch offset must be finite")
    return {
        "offsets": offsets, "tip": tip, "limits_deg": limits,
        "pitch_offset": pitch_offset,
        "pitch_reference": "cutting lip" if "pitch_offset_rad" in profile["robot"]["tool"] else "joint frame",
        "origin": profile["robot"].get("coordinate_origin", "profile origin"),
    }


def forward_kinematics(q, geometry):
    """Vectorized robot URDF semantics, with slew zero in the carriage frame.

    Return XYZ [m] and wrapped bucket pitch [deg, positive about Y]. Base rocking
    is intentionally excluded: this maps the driven arm's relative configuration.
    """
    q = np.asarray(q, dtype=float).reshape(-1, 3)
    offsets, tip = geometry["offsets"], geometry["tip"]
    point = np.broadcast_to(offsets[0] + offsets[1], (len(q), 3)).copy()
    angle = np.zeros(len(q))
    for joint, offset in zip(range(3), (offsets[2], offsets[3], tip)):
        angle += q[:, joint]
        c, s = np.cos(angle), np.sin(angle)
        point[:, 0] += c * offset[0] + s * offset[2]
        point[:, 1] += offset[1]
        point[:, 2] += -s * offset[0] + c * offset[2]
    pitch = (np.degrees(angle + geometry["pitch_offset"]) + 180) % 360 - 180
    return point, pitch


def resolve_logs(values):
    paths = set()
    for value in values:
        matches = [Path(value)] if Path(value).exists() else [Path(p) for p in glob.glob(value)]
        if not matches:
            raise ValueError(f"No files match {value!r}")
        for path in matches:
            files = [*path.glob("drive_log_*.csv"), *path.glob("drive_log_*.csv.gz")] if path.is_dir() else [path]
            if not files:
                raise ValueError(f"No drive logs in {path}")
            for file in files:
                if not file.name.startswith("drive_log_") or not file.name.endswith((".csv", ".csv.gz")):
                    raise ValueError(f"Expected drive_log_*.csv[.gz], got {file}")
                paths.add(file.resolve())
    return sorted(paths)


def read_log(path, motion_only=False):
    required = ("timestamp", *POSITIONS, *COMMANDS)
    if motion_only:
        required += VELOCITIES
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} lacks {', '.join(sorted(missing))}")
        names = [*required, *(n for n in (*QUALITY, "vel_age_s") if n in reader.fieldnames)]
        rows = []
        for line, row in enumerate(reader, 2):
            try:
                rows.append([float(row[name]) if row[name] else np.nan for name in names])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}:{line}: malformed numeric field") from exc
    matrix = np.asarray(rows, dtype=float).reshape(-1, len(names))
    return dict(zip(names, matrix.T))


def usable_intervals(data, max_gap, max_age, motion_only=False, min_speed=1.0):
    """Conservative dwell: count row i -> i+1 only if both endpoints are usable.

    Invalid/stale rows, nonpositive time and skipped sample indices never bridge.
    The last row has no known duration. Missing legacy quality columns are
    reported separately by the caller, rather than invented.
    """
    t = data["timestamp"]
    finite = np.isfinite(np.column_stack([data[n] for n in ("timestamp", *POSITIONS, *COMMANDS)])).all(1)
    valid = finite.copy()
    reasons = {"invalid_values": int((~finite).sum())}
    for name in ("cmd_stale", "cmd_age_s", "state_age_s"):
        if name not in data:
            continue
        values = data[name]
        bad = ~np.isfinite(values) | (values != 0 if name == "cmd_stale" else ((values < 0) | (values > max_age)))
        valid &= ~bad
        reasons[name] = int(bad.sum())
    if motion_only:
        v = np.column_stack([data[n] for n in VELOCITIES])
        moving = np.isfinite(v).all(1) & (np.abs(v).max(1) >= np.radians(min_speed))
        if "vel_age_s" in data:
            age = data["vel_age_s"]
            moving &= np.isfinite(age) & (age >= 0) & (age <= max_age)
        valid &= moving
        reasons["not_fresh_motion"] = int((~moving).sum())
    dt = np.diff(t)
    continuous = np.isfinite(dt) & (dt > 0) & (dt <= max_gap)
    if "sample_idx" in data:
        continuous &= np.diff(data["sample_idx"]) == 1
    interval = valid[:-1] & valid[1:] & continuous
    ids = np.flatnonzero(interval)
    return ids, dt[ids], {
        "rows": len(t), "usable_rows": int(valid.sum()), "accepted_intervals": len(ids),
        "accepted_seconds": float(dt[ids].sum()), "broken_intervals": int((~continuous).sum()),
        "rejected_rows_by_reason_nonexclusive": reasons,
        "unavailable_quality_columns": [n for n in QUALITY if n not in data],
    }


def edges_for_range(lo, hi, step):
    start = math.floor(lo / step) * step
    stop = max(start + step, math.ceil(hi / step) * step)
    return np.arange(start, stop + step * 0.5, step)


def pitch_edges(step):
    # Keep the periodic endpoints exact, including when step does not divide 360.
    return np.r_[np.arange(-180, 180, step), 180.0]


def sample_workspace(geometry, step_deg):
    axes = [np.linspace(lo, hi, int(np.ceil((hi - lo) / step_deg)) + 1) for lo, hi in geometry["limits_deg"]]
    size = math.prod(len(a) for a in axes)
    if size > 2_000_000:
        raise ValueError(f"Workspace sampling would create {size:,} poses; increase --workspace-step-deg")
    angles = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return forward_kinematics(np.radians(angles), geometry)


def histogram(xy, weights, edges):
    return np.histogram2d(xy[:, 0], xy[:, 1], bins=edges, weights=weights)[0]


def render_plots(out, q, points, pitch, weights, workspace, geometry, args):
    # Import only at rendering time; computation/tests need no plotting environment.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm
    from matplotlib.patches import Patch

    ref_points, ref_pitch = workspace
    xy, ref_xy = points[:, [0, 2]], ref_points[:, [0, 2]]
    union = np.vstack([xy, ref_xy])
    spatial = [edges_for_range(union[:, i].min(), union[:, i].max(), args.spatial_bin_mm / 1000) for i in range(2)]
    q_deg = np.degrees(q)
    joint_edges = [edges_for_range(min(limits[0], q_deg[:, i].min()), max(limits[1], q_deg[:, i].max()), args.joint_bin_deg)
                   for i, limits in enumerate(geometry["limits_deg"])]
    p_edges = pitch_edges(args.pitch_bin_deg)
    overall = histogram(xy, weights, spatial)
    # Pooled joint bins can hold more time than a spatial bin. Use one scale
    # covering all figures; each sliced bin is bounded by its pooled counterpart.
    maximum = max(float(overall.max()), *(float(histogram(q_deg[:, [a, b]], weights, [joint_edges[a], joint_edges[b]]).max())
                                         for a, b in ((0, 1), (0, 2), (1, 2))))
    minimum = float(weights.min())
    norm = LogNorm(vmin=minimum, vmax=max(maximum, minimum * 1.01))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((0, 0, 0, 0))
    mode = "motion only" if args.motion_only else "all accepted poses"
    caption = f"{len(args.resolved_logs)} recordings · {weights.sum() / 60:.1f} min · {mode}"
    frame = f"{geometry['origin']}, slew-relative X/Z; base rocking excluded"
    saved = []

    def draw(ax, counts, edges, possible=None, labels=("X [m]", "Z [m]")):
        background = np.ones_like(counts) if possible is None else possible.astype(float)
        extent = (edges[0][0], edges[0][-1], edges[1][0], edges[1][-1])
        ax.imshow(background.T, origin="lower", extent=extent, interpolation="nearest",
                  cmap=ListedColormap(["#d9d9d9", "white"]), vmin=0, vmax=1, aspect="auto")
        mesh = ax.pcolormesh(*edges, np.ma.masked_less_equal(counts.T, 0), cmap=cmap, norm=norm, shading="flat", rasterized=True)
        ax.set(xlabel=labels[0], ylabel=labels[1], xlim=extent[:2], ylim=extent[2:])
        ax.tick_params(labelsize=8)
        if labels[0] == "X [m]":
            ax.set_aspect("equal", adjustable="box")
        return mesh

    def save(fig, axes, mesh, name, title, spatial_plot=False):
        fig.suptitle(title + "\n" + caption, fontsize=13)
        fig.colorbar(mesh, ax=list(np.asarray(axes, dtype=object).flat), label="Recorded seconds per bin (log scale)", shrink=0.75)
        note = frame if spatial_plot else "Joint angles as recorded, plus configured offsets; dashed lines: profile limits"
        if spatial_plot:
            fig.legend(handles=[Patch(facecolor="white", edgecolor="gray", label="Sampled workspace, no data"),
                                Patch(facecolor="#d9d9d9", label="Outside sampled workspace")],
                       loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=2, fontsize=9)
            note += "; envelope uses joint limits only (no collision checks)"
        fig.text(0.5, 0.004, note, ha="center", fontsize=8)
        path = out / name
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        plt.close(fig)
        saved.append(name)

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.10, 1, 0.90))
    mesh = draw(ax, overall, spatial, histogram(ref_xy, None, spatial) > 0)
    save(fig, [ax], mesh, "workspace.png", "Bucket-tip position coverage", True)

    count = len(p_edges) - 1
    cols = min(4, count)
    fig, axes = plt.subplots(math.ceil(count / cols), cols, squeeze=False, figsize=(cols * 4, math.ceil(count / cols) * 3.3), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.065, 1, 0.92))
    for i, ax in enumerate(axes.flat):
        if i >= count:
            ax.set_visible(False)
            continue
        lo, hi = p_edges[i:i + 2]
        selected = (pitch >= lo) & (pitch < hi)
        ref = (ref_pitch >= lo) & (ref_pitch < hi)
        mesh = draw(ax, histogram(xy[selected], weights[selected], spatial), spatial, histogram(ref_xy[ref], None, spatial) > 0)
        ax.set_title(f"{lo:g}° to {hi:g}° · {weights[selected].sum():.1f} s", fontsize=10)
    save(fig, axes, mesh, "workspace_by_pitch.png", f"Bucket-tip coverage by {geometry['pitch_reference']} pitch (+Y)", True)

    def joint_plot(ax, first, second, selected):
        edges = [joint_edges[first], joint_edges[second]]
        counts = histogram(q_deg[selected][:, [first, second]], weights[selected], edges)
        mesh = draw(ax, counts, edges, labels=(f"{JOINTS[first].title()} [deg]", f"{JOINTS[second].title()} [deg]"))
        for limit in geometry["limits_deg"][first]:
            ax.axvline(limit, color="gray", ls="--", lw=0.7)
        for limit in geometry["limits_deg"][second]:
            ax.axhline(limit, color="gray", ls="--", lw=0.7)
        return mesh

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.90))
    for ax, (first, second) in zip(axes, ((0, 1), (0, 2), (1, 2))):
        mesh = joint_plot(ax, first, second, np.ones(len(q), dtype=bool))
    save(fig, axes, mesh, "joint_pairs.png", "Joint-pair pose coverage (third joint pooled)")

    for first, second, third in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        lo = min(geometry["limits_deg"][third, 0], q_deg[:, third].min())
        hi = max(geometry["limits_deg"][third, 1], q_deg[:, third].max())
        slices = edges_for_range(lo, hi, args.pitch_bin_deg)
        count = len(slices) - 1
        cols = min(4, count)
        fig, axes = plt.subplots(math.ceil(count / cols), cols, squeeze=False, figsize=(cols * 4, math.ceil(count / cols) * 3.3), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.92))
        for i, ax in enumerate(axes.flat):
            if i >= count:
                ax.set_visible(False)
                continue
            lo, hi = slices[i:i + 2]
            selected = (q_deg[:, third] >= lo) & ((q_deg[:, third] <= hi) if i == count - 1 else (q_deg[:, third] < hi))
            mesh = joint_plot(ax, first, second, selected)
            ax.set_title(f"{JOINTS[third]} {lo:g}° to {hi:g}° · {weights[selected].sum():.1f} s", fontsize=10)
        save(fig, axes, mesh, f"joint_{JOINTS[first]}_{JOINTS[second]}_by_{JOINTS[third]}.png", "Joint configuration coverage")
    return saved


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True, help="Files, quoted globs or directories (nonrecursive)")
    parser.add_argument("--profile", type=Path, required=True, help="Geometry/limits control YAML or robot_geometry.json")
    parser.add_argument("--out", type=Path, help="New output directory; default data_collection/coverage_plots/<timestamp>")
    parser.add_argument("--label", default="", help="Recording/calibration group label stored in summary")
    parser.add_argument("--spatial-bin-mm", type=positive_number, default=20)
    parser.add_argument("--joint-bin-deg", type=positive_number, default=10)
    parser.add_argument("--pitch-bin-deg", type=positive_number, default=15, help="Bucket pitch and third-joint slice width")
    parser.add_argument("--workspace-step-deg", type=positive_number, default=3)
    parser.add_argument("--max-gap-ms", type=positive_number, default=30)
    parser.add_argument("--max-age-ms", type=positive_number, default=30)
    parser.add_argument("--angle-offset-deg", type=float, nargs=3, default=[0, 0, 0], metavar=("BOOM", "ARM", "BUCKET"), help="Explicit logged-to-profile calibration correction")
    parser.add_argument("--motion-only", action="store_true", help="Require fresh logged joint velocity above --min-speed-deg-s")
    parser.add_argument("--min-speed-deg-s", type=positive_number, default=1)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if not np.isfinite(args.angle_offset_deg).all() or args.dpi <= 0:
        parser.error("Angle offsets must be finite and dpi positive")
    if args.pitch_bin_deg > 360:
        parser.error("--pitch-bin-deg must be at most 360")
    try:
        args.resolved_logs = resolve_logs(args.logs)
        geometry = load_profile(args.profile)
        workspace = sample_workspace(geometry, args.workspace_step_deg)
        qs, weights, reports, hashes = [], [], [], {}
        for path in args.resolved_logs:
            # Deduplicate exact copies even when different paths were selected.
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest in hashes:
                print(f"Skip duplicate {path.name} (same contents as {hashes[digest]})", flush=True)
                continue
            hashes[digest] = str(path)
            data = read_log(path, args.motion_only)
            ids, dwell, report = usable_intervals(data, args.max_gap_ms / 1000, args.max_age_ms / 1000, args.motion_only, args.min_speed_deg_s)
            q = np.column_stack([data[n][ids] for n in POSITIONS]) + np.radians(args.angle_offset_deg)
            report.update(path=str(path), sha256=digest)
            if len(q):
                degrees = np.degrees(q)
                report["joint_ranges_deg"] = np.stack([degrees.min(0), degrees.max(0)], 1).tolist()
                report["out_of_limit_intervals"] = int(((degrees < geometry["limits_deg"][:, 0]) | (degrees > geometry["limits_deg"][:, 1])).any(1).sum())
            qs.append(q)
            weights.append(dwell)
            reports.append(report)
            print(f"{path.name}: {report['accepted_seconds']:.1f} s accepted, {len(ids):,} intervals", flush=True)
        q, dwell = np.concatenate(qs), np.concatenate(weights)
        if not len(q):
            raise ValueError("No usable duration after filtering; check log quality, ages and motion threshold")
        points, pitch = forward_kinematics(q, geometry)
        out = args.out or Path(__file__).resolve().parent / "coverage_plots" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Output directory is not empty: {out}; choose a new --out")
        out.mkdir(parents=True, exist_ok=True)
        args.resolved_logs = [Path(r["path"]) for r in reports]
        figures = render_plots(out, q, points, pitch, dwell, workspace, geometry, args)
        summary = {
            "version": 1, "coverage_kind": "recorded_pose_not_MLP_training_window_coverage",
            "label": args.label, "profile": str(args.profile.resolve()),
            "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            "frame": "slew_relative_XZ_base_rocking_excluded", "origin": geometry["origin"],
            "bucket_pitch_reference": geometry["pitch_reference"], "bucket_pitch_positive_axis": "Y",
            "angle_offset_deg": args.angle_offset_deg, "motion_only": args.motion_only,
            "min_speed_deg_s": args.min_speed_deg_s, "max_gap_ms": args.max_gap_ms, "max_age_ms": args.max_age_ms,
            "spatial_bin_mm": args.spatial_bin_mm, "joint_bin_deg": args.joint_bin_deg, "pitch_bin_deg": args.pitch_bin_deg,
            "workspace_step_deg": args.workspace_step_deg, "workspace_collision_checked": False,
            "accepted_seconds": float(dwell.sum()), "accepted_intervals": len(dwell),
            "recordings": reports, "figures": figures,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {len(figures)} PNGs and summary.json to {out.resolve()}", flush=True)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Coverage error: {exc}\n")


if __name__ == "__main__":
    main()
