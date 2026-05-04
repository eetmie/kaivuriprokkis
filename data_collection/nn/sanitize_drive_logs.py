"""
sanitize_drive_logs.py

Clean and segment raw drive logs before training.

Main tasks:
- Validate required columns
- Sort and deduplicate timestamps
- Split on invalid/large time gaps
- Drop unusable rows (NaNs in core channels)
- Save each clean segment to its own CSV
- Write a manifest with per-segment stats
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import argparse
import json

import numpy as np
import pandas as pd


TIME_COL = "timestamp"
U_COLS = ["combined_cmd_lift", "combined_cmd_tilt", "combined_cmd_scoop"]
Q_COLS = ["joint_pos_boom", "joint_pos_arm", "joint_pos_bucket"]
QDOT_COLS = ["joint_vel_boom", "joint_vel_arm", "joint_vel_bucket"]
CORE_COLS = [*U_COLS, *Q_COLS, *QDOT_COLS]
MANDATORY_COLS = [*U_COLS, *Q_COLS]


def _finite_difference_velocity(seg: pd.DataFrame) -> pd.DataFrame:
    """Fill missing joint velocity columns from joint positions and timestamps."""
    seg = seg.copy()
    t = seg[TIME_COL].to_numpy(np.float64)
    if len(seg) < 2:
        for qdot_col in QDOT_COLS:
            if qdot_col not in seg.columns:
                seg[qdot_col] = np.nan
        return seg

    for q_col, qdot_col in zip(Q_COLS, QDOT_COLS):
        q = pd.to_numeric(seg[q_col], errors="coerce").to_numpy(np.float64)
        derived = np.gradient(q, t)
        if qdot_col not in seg.columns:
            seg[qdot_col] = derived
            continue

        existing = pd.to_numeric(seg[qdot_col], errors="coerce").to_numpy(np.float64)
        fill_mask = ~np.isfinite(existing)
        if np.any(fill_mask):
            existing[fill_mask] = derived[fill_mask]
            seg[qdot_col] = existing
    return seg


def _resolve_inputs(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob("*.csv")))
        else:
            out.extend(sorted(Path().glob(item)))
    # unique while preserving order
    seen = set()
    unique: List[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(rp)
    return unique


def _split_by_gaps(df: pd.DataFrame, dt: float, gap_factor: float) -> List[pd.DataFrame]:
    if len(df) < 2:
        return []
    t = df[TIME_COL].to_numpy(np.float64)
    gaps = np.diff(t)
    split_idx = np.where((gaps <= 0.0) | (gaps > gap_factor * dt))[0] + 1
    chunks = np.split(df, split_idx)
    return [c.copy() for c in chunks if len(c) >= 2]


def sanitize_file(
    csv_path: Path,
    dt: float,
    gap_factor: float,
    min_segment_sec: float,
    drop_long_idle_sec: float,
) -> List[pd.DataFrame]:
    df = pd.read_csv(csv_path)
    missing = [c for c in [TIME_COL] + MANDATORY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TIME_COL]).copy()
    df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="first")

    segments = _split_by_gaps(df, dt=dt, gap_factor=gap_factor)
    clean_segments: List[pd.DataFrame] = []
    min_rows = max(2, int(min_segment_sec / dt))
    max_idle_rows = int(drop_long_idle_sec / dt) if drop_long_idle_sec > 0 else 0

    for seg in segments:
        seg = _finite_difference_velocity(seg)
        seg = seg.dropna(subset=CORE_COLS).copy()
        if "cmd_stale" in seg.columns:
            cmd_stale = pd.to_numeric(seg["cmd_stale"], errors="coerce").fillna(0.0)
            seg = seg.loc[cmd_stale <= 0.5].copy()
        if len(seg) < min_rows:
            continue

        # Optionally trim very long all-zero command intervals.
        if max_idle_rows > 0:
            u = seg[U_COLS].to_numpy(np.float64)
            all_zero = (np.abs(u) < 1e-6).all(axis=1)
            keep = np.ones(len(seg), dtype=bool)
            start = None
            for i, z in enumerate(all_zero):
                if z and start is None:
                    start = i
                elif (not z) and start is not None:
                    run = i - start
                    if run > max_idle_rows:
                        keep[start + max_idle_rows : i] = False
                    start = None
            if start is not None:
                run = len(seg) - start
                if run > max_idle_rows:
                    keep[start + max_idle_rows :] = False
            seg = seg.iloc[np.where(keep)[0]].copy()
            if len(seg) < min_rows:
                continue

        # Rebase local time to make segments self-contained.
        t0 = float(seg[TIME_COL].iloc[0])
        seg[TIME_COL] = seg[TIME_COL] - t0
        clean_segments.append(seg)

    return clean_segments


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True, help="CSV file(s), directory, or glob pattern(s)")
    p.add_argument("--out", default="data_collection/nn/cleaned_logs", help="Output directory")
    p.add_argument("--dt", type=float, default=0.02, help="Target dt used for gap checks")
    p.add_argument("--gap-factor", type=float, default=3.0, help="Split when dt > gap_factor * dt")
    p.add_argument("--min-segment-sec", type=float, default=5.0, help="Drop segments shorter than this")
    p.add_argument(
        "--drop-long-idle-sec",
        type=float,
        default=0.0,
        help="If >0, trim all-zero command runs longer than this duration",
    )
    args = p.parse_args()

    files = _resolve_inputs(args.input)
    if not files:
        raise FileNotFoundError("No input CSV files found")

    out_dir = Path(args.out)
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    global_segment_id = 0

    for f in files:
        segments = sanitize_file(
            f,
            dt=args.dt,
            gap_factor=args.gap_factor,
            min_segment_sec=args.min_segment_sec,
            drop_long_idle_sec=args.drop_long_idle_sec,
        )
        for local_id, seg in enumerate(segments):
            seg = seg.copy()
            seg["segment_id"] = global_segment_id
            seg["sample_idx"] = np.arange(len(seg), dtype=np.int64)
            seg["source_file"] = str(f)

            out_csv = seg_dir / f"{f.stem}_seg{global_segment_id:04d}.csv"
            seg.to_csv(out_csv, index=False)

            duration = float(seg[TIME_COL].iloc[-1] - seg[TIME_COL].iloc[0]) if len(seg) > 1 else 0.0
            active_frac = float((np.abs(seg[U_COLS].to_numpy()) > 0.05).any(axis=1).mean())
            manifest_rows.append(
                {
                    "segment_id": global_segment_id,
                    "source_file": str(f),
                    "local_segment_index": local_id,
                    "rows": int(len(seg)),
                    "duration_s": duration,
                    "active_frac": active_frac,
                    "output_csv": str(out_csv),
                }
            )
            global_segment_id += 1

    if not manifest_rows:
        raise RuntimeError("No usable segments produced. Check thresholds and input quality.")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = out_dir / "manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    summary = {
        "num_input_files": len(files),
        "num_segments": int(len(manifest_df)),
        "total_rows": int(manifest_df["rows"].sum()),
        "total_duration_s": float(manifest_df["duration_s"].sum()),
        "mean_active_frac": float(manifest_df["active_frac"].mean()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Wrote {len(manifest_df)} segments to {seg_dir}")
    print(f"Manifest: {manifest_csv}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
