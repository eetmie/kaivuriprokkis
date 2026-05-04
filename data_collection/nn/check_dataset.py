"""
check_dataset.py

Quick QA tool for hydraulic drive logs (raw or cleaned).

Reports:
- timing quality (dt stats, large gaps, monotonicity)
- NaNs in core channels
- command activity/saturation/idle behavior
- optional stale signal ratios if available in logs
- per-file and aggregate summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd


TIME_COL = "timestamp"
U_COLS = ["combined_cmd_lift", "combined_cmd_tilt", "combined_cmd_scoop"]
Q_COLS = ["joint_pos_boom", "joint_pos_arm", "joint_pos_bucket"]
QDOT_COLS = ["joint_vel_boom", "joint_vel_arm", "joint_vel_bucket"]
CORE_COLS = [TIME_COL] + U_COLS + Q_COLS + QDOT_COLS


def resolve_inputs(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file() and p.suffix.lower() == ".csv":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("*.csv")))
        else:
            files.extend(sorted(Path().glob(item)))

    seen = set()
    out: List[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def contiguous_true_runs(mask: np.ndarray) -> List[int]:
    if mask.size == 0:
        return []
    runs: List[int] = []
    n = 0
    for v in mask:
        if v:
            n += 1
        elif n > 0:
            runs.append(n)
            n = 0
    if n > 0:
        runs.append(n)
    return runs


def analyze_file(path: Path, expected_dt: float, gap_factor: float, active_thr: float, sat_thr: float) -> Dict[str, Any]:
    df = pd.read_csv(path)
    out: Dict[str, Any] = {"file": str(path), "rows": int(len(df))}

    missing = [c for c in CORE_COLS if c not in df.columns]
    out["missing_required_columns"] = missing
    if missing:
        out["ok"] = False
        return out

    t = df[TIME_COL].to_numpy(np.float64)
    if len(t) < 2:
        out.update({"ok": False, "reason": "too_few_rows"})
        return out

    dt = np.diff(t)
    non_mono = int((dt <= 0.0).sum())
    gaps = dt > (expected_dt * gap_factor)

    out["ok"] = True
    out["duration_s"] = float(t[-1] - t[0])
    out["dt_mean"] = float(np.mean(dt))
    out["dt_std"] = float(np.std(dt))
    out["dt_min"] = float(np.min(dt))
    out["dt_max"] = float(np.max(dt))
    out["non_monotonic_steps"] = non_mono
    out["num_large_gaps"] = int(gaps.sum())
    out["large_gap_threshold_s"] = float(expected_dt * gap_factor)

    # NaN counts in core channels
    nan_counts = df[CORE_COLS].isna().sum().to_dict()
    out["nan_counts_core"] = {k: int(v) for k, v in nan_counts.items()}
    out["nan_rows_any_core"] = int(df[CORE_COLS].isna().any(axis=1).sum())

    # command stats
    u = df[U_COLS].to_numpy(np.float64)
    active = np.abs(u) > active_thr
    sat = np.abs(u) > sat_thr
    all_zero = (np.abs(u) < 1e-9).all(axis=1)
    all_active_any = active.any(axis=1)

    out["u_minmax"] = {c: [float(df[c].min()), float(df[c].max())] for c in U_COLS}
    out["active_frac_per_joint"] = {c: float(active[:, i].mean()) for i, c in enumerate(U_COLS)}
    out["sat_frac_per_joint"] = {c: float(sat[:, i].mean()) for i, c in enumerate(U_COLS)}
    out["all_joints_zero_frac"] = float(all_zero.mean())
    out["any_joint_active_frac"] = float(all_active_any.mean())
    counts = active.sum(axis=1)
    out["active_joint_count_frac"] = {str(k): float((counts == k).mean()) for k in range(len(U_COLS) + 1)}

    zero_runs = contiguous_true_runs(all_zero)
    if zero_runs:
        out["max_all_zero_run_s"] = float(max(zero_runs) * np.median(dt))
        out["num_all_zero_runs_gt_5s"] = int(sum((r * np.median(dt)) > 5.0 for r in zero_runs))
    else:
        out["max_all_zero_run_s"] = 0.0
        out["num_all_zero_runs_gt_5s"] = 0

    # optional stale metrics
    if "cmd_stale" in df.columns:
        out["cmd_stale_frac"] = float(pd.to_numeric(df["cmd_stale"], errors="coerce").fillna(0.0).mean())
    if "pressure_stale" in df.columns:
        out["pressure_stale_frac"] = float(pd.to_numeric(df["pressure_stale"], errors="coerce").fillna(0.0).mean())
    if "cmd_age_s" in df.columns:
        age = pd.to_numeric(df["cmd_age_s"], errors="coerce")
        out["cmd_age_s_p95"] = float(age.quantile(0.95)) if age.notna().any() else float("nan")
    if "sine_enabled" in df.columns:
        sine = pd.to_numeric(df["sine_enabled"], errors="coerce").fillna(0.0)
        out["sine_enabled_frac"] = float(sine.mean())
    if "pressure_age_s" in df.columns:
        age = pd.to_numeric(df["pressure_age_s"], errors="coerce")
        out["pressure_age_s_p95"] = float(age.quantile(0.95)) if age.notna().any() else float("nan")

    return out


def aggregate(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in reports if r.get("ok")]
    bad = [r for r in reports if not r.get("ok")]
    summary: Dict[str, Any] = {
        "num_files": len(reports),
        "num_ok_files": len(ok),
        "num_bad_files": len(bad),
        "bad_files": [r["file"] for r in bad],
    }
    if not ok:
        return summary

    summary["total_rows"] = int(sum(r["rows"] for r in ok))
    summary["total_duration_s"] = float(sum(r["duration_s"] for r in ok))
    summary["total_large_gaps"] = int(sum(r["num_large_gaps"] for r in ok))
    summary["total_non_monotonic_steps"] = int(sum(r["non_monotonic_steps"] for r in ok))
    summary["max_gap_s"] = float(max(r["dt_max"] for r in ok))
    summary["median_dt_mean_s"] = float(np.median([r["dt_mean"] for r in ok]))
    summary["median_any_joint_active_frac"] = float(np.median([r["any_joint_active_frac"] for r in ok]))
    summary["median_all_joints_zero_frac"] = float(np.median([r["all_joints_zero_frac"] for r in ok]))

    if any("cmd_stale_frac" in r for r in ok):
        vals = [r.get("cmd_stale_frac", np.nan) for r in ok]
        summary["mean_cmd_stale_frac"] = float(np.nanmean(vals))
    if any("pressure_stale_frac" in r for r in ok):
        vals = [r.get("pressure_stale_frac", np.nan) for r in ok]
        summary["mean_pressure_stale_frac"] = float(np.nanmean(vals))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset QA for hydraulic drive logs")
    parser.add_argument("--input", nargs="+", required=True, help="CSV file(s), directory, or glob pattern(s)")
    parser.add_argument("--dt", type=float, default=0.02, help="Expected sample period (s)")
    parser.add_argument("--gap-factor", type=float, default=3.0, help="Gap threshold factor (gap > dt*factor)")
    parser.add_argument("--active-thr", type=float, default=0.05, help="Command active threshold")
    parser.add_argument("--sat-thr", type=float, default=0.95, help="Command saturation threshold")
    parser.add_argument("--out-json", default=None, help="Optional path to write full report as JSON")
    args = parser.parse_args()

    files = resolve_inputs(args.input)
    if not files:
        raise FileNotFoundError("No CSV files found from --input")

    reports = [
        analyze_file(
            path=f,
            expected_dt=args.dt,
            gap_factor=args.gap_factor,
            active_thr=args.active_thr,
            sat_thr=args.sat_thr,
        )
        for f in files
    ]
    summary = aggregate(reports)

    print("Dataset QA Summary")
    print("- files:", summary.get("num_files"))
    print("- ok/bad:", f"{summary.get('num_ok_files')}/{summary.get('num_bad_files')}")
    if summary.get("num_ok_files", 0) > 0:
        print("- total rows:", summary.get("total_rows"))
        print("- total duration (min):", round(summary.get("total_duration_s", 0.0) / 60.0, 2))
        print("- large gaps:", summary.get("total_large_gaps"), "(max gap s:", round(summary.get("max_gap_s", 0.0), 4), ")")
        print("- median dt_mean:", round(summary.get("median_dt_mean_s", 0.0), 6))
        print("- median active frac (any joint):", round(summary.get("median_any_joint_active_frac", 0.0), 4))
        print("- median all-zero frac:", round(summary.get("median_all_joints_zero_frac", 0.0), 4))
        if "mean_cmd_stale_frac" in summary:
            print("- mean cmd stale frac:", round(summary["mean_cmd_stale_frac"], 4))
        if "mean_pressure_stale_frac" in summary:
            print("- mean pressure stale frac:", round(summary["mean_pressure_stale_frac"], 4))

    if summary.get("bad_files"):
        print("- bad files:")
        for f in summary["bad_files"]:
            print("  -", f)

    if args.out_json:
        payload = {"summary": summary, "files": reports}
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        print(f"Wrote report JSON: {out_path}")


if __name__ == "__main__":
    main()
