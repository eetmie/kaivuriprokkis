#!/usr/bin/env python3
"""Sanitize and plot linkage-rate logger output.

Reads a CSV produced by ``tools/linkage_rate_logger.py``, removes the obvious
end-stop / settle artifacts, and saves:

- a sanitized CSV with the retained sweep samples
- per-joint plots of joint velocity vs angle
- per-joint plots of normalized rate (deg/s per unit command) vs angle

Typical use:

  python tools/analyze_linkage_rates.py logs/linkage_rate_20260504_195552.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter  # type: ignore
except Exception:  # pragma: no cover - optional analysis dependency
    savgol_filter = None


ROOT_DIR = Path(__file__).resolve().parents[1]
JOINTS = ["boom", "arm", "bucket"]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data_collection" / "hydraulic_data" / "processed"
ANGLE_BIN_DEG = 2.0
MIN_SEGMENT_SAMPLES = 25
TRIM_FRACTION = 0.05
TRIM_MAX_SAMPLES = 25
ROLLING_WINDOW = 9


def _segment_id_table(df: pd.DataFrame) -> pd.Series:
    keys = ["joint", "pass_index", "amplitude", "direction"]
    changed = df[keys].ne(df[keys].shift()).any(axis=1)
    return changed.cumsum().astype(np.int32)


def sanitize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    work = df.copy()
    for col in ["command", "amplitude", "direction", "pass_index"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work[work["phase"] == "sweep"].copy()
    work = work[work["joint"].isin(JOINTS)].copy()
    work = work[np.isfinite(work["amplitude"]) & (work["amplitude"] > 0.0)].copy()
    work = work[np.isfinite(work["direction"]) & (work["direction"] != 0.0)].copy()
    work = work.reset_index(drop=True)
    work["segment_id"] = _segment_id_table(work)

    keep_segments: list[pd.DataFrame] = []
    stats = {
        "rows_input": int(len(df)),
        "rows_sweep": int(len(work)),
        "rows_output": 0,
        "segments_kept": 0,
        "segments_dropped": 0,
    }

    for segment_id, segment in work.groupby("segment_id", sort=False):
        segment = segment.copy().reset_index(drop=True)
        joint = str(segment.at[0, "joint"])
        angle_col = f"{joint}_deg"
        vel_col = f"{joint}_vel_deg_s"

        segment[angle_col] = pd.to_numeric(segment[angle_col], errors="coerce")
        segment[vel_col] = pd.to_numeric(segment[vel_col], errors="coerce")
        segment = segment[np.isfinite(segment[angle_col]) & np.isfinite(segment[vel_col])].copy()
        if len(segment) < MIN_SEGMENT_SAMPLES:
            stats["segments_dropped"] += 1
            continue

        trim_n = min(TRIM_MAX_SAMPLES, int(len(segment) * TRIM_FRACTION))
        if trim_n > 0 and len(segment) > 2 * trim_n:
            segment = segment.iloc[trim_n:-trim_n].copy()
        if len(segment) < MIN_SEGMENT_SAMPLES:
            stats["segments_dropped"] += 1
            continue

        segment["progress_vel_deg_s"] = segment[vel_col] * segment["direction"]
        rolling = segment["progress_vel_deg_s"].rolling(ROLLING_WINDOW, center=True, min_periods=1).median()
        segment["progress_vel_deg_s_med"] = rolling
        segment = segment[segment["progress_vel_deg_s_med"] > 0.0].copy()
        if len(segment) < MIN_SEGMENT_SAMPLES:
            stats["segments_dropped"] += 1
            continue

        tail_len = min(TRIM_MAX_SAMPLES, max(6, int(len(segment) * 0.08)))
        if len(segment) > tail_len:
            tail = segment.iloc[-tail_len:]
            tail_thresh = max(0.75, float(tail["progress_vel_deg_s_med"].quantile(0.2)))
            segment = segment[segment["progress_vel_deg_s_med"] >= tail_thresh].copy()
        if len(segment) < MIN_SEGMENT_SAMPLES:
            stats["segments_dropped"] += 1
            continue

        segment["joint_angle_deg"] = segment[angle_col]
        segment["joint_vel_deg_s"] = segment[vel_col]
        segment["forward_vel_deg_s"] = segment["progress_vel_deg_s"]
        segment["rate_per_command"] = segment["forward_vel_deg_s"] / segment["amplitude"]
        segment["source_segment_id"] = int(segment_id)
        keep_segments.append(segment)
        stats["segments_kept"] += 1

    if not keep_segments:
        raise RuntimeError("No usable sweep segments remained after sanitization")

    out = pd.concat(keep_segments, ignore_index=True)
    out["angle_bin_deg"] = np.round(out["joint_angle_deg"] / ANGLE_BIN_DEG) * ANGLE_BIN_DEG
    stats["rows_output"] = int(len(out))
    stats["kept_fraction"] = float(len(out) / max(1, len(work)))
    return out, stats


def summarize_bins(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["joint", "amplitude", "direction", "angle_bin_deg"], as_index=False)
    return grouped.agg(
        samples=("joint", "size"),
        angle_deg=("joint_angle_deg", "median"),
        forward_vel_deg_s=("forward_vel_deg_s", "median"),
        rate_per_command=("rate_per_command", "median"),
    )


def _jump_threshold(values: np.ndarray, factor: float, min_delta: float) -> float:
    deltas = np.abs(np.diff(values))
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return float("inf")
    typical = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - typical)))
    robust = typical + factor * max(mad, 1e-6)
    average = float(np.mean(deltas)) * factor
    return max(min_delta, min(robust, average))


def _large_jump_mask(values: np.ndarray, factor: float, min_delta: float) -> np.ndarray:
    if len(values) < 4:
        return np.zeros(len(values), dtype=bool)
    jumps = np.abs(np.diff(values)) > _jump_threshold(values, factor, min_delta)
    bad = np.zeros(len(values), dtype=bool)

    for i, is_jump in enumerate(jumps):
        if not is_jump:
            continue
        if i == 0:
            bad[0] = True
            continue
        if i == len(values) - 2:
            bad[-1] = True
            continue
        if i + 1 < len(jumps) and jumps[i + 1]:
            bad[i + 1] = True
            continue
        if i - 1 >= 0 and jumps[i - 1]:
            bad[i] = True
            continue

        left_slope = values[i] - values[i - 1]
        right_slope = values[i + 2] - values[i + 1]
        predicted_left = values[i - 1] + right_slope
        predicted_right = values[i + 2] - left_slope
        if abs(values[i] - predicted_left) >= abs(values[i + 1] - predicted_right):
            bad[i] = True
        else:
            bad[i + 1] = True
    return bad


def filter_summary_jumps(summary: pd.DataFrame, *, factor: float, min_delta: float) -> tuple[pd.DataFrame, dict[str, int]]:
    if factor <= 0:
        return summary, {"jump_filtered_points": 0, "jump_filtered_curves": 0}

    kept_parts: list[pd.DataFrame] = []
    removed_points = 0
    removed_curves = 0
    for _, curve in summary.groupby(["joint", "amplitude", "direction"], sort=False):
        curve = curve.sort_values("angle_deg").copy()
        values = curve["rate_per_command"].to_numpy(dtype=np.float64)
        bad = _large_jump_mask(values, factor, min_delta)
        if np.any(bad) and len(curve) - int(np.sum(bad)) >= 4:
            removed_points += int(np.sum(bad))
            removed_curves += 1
            curve = curve.loc[~bad].copy()
        kept_parts.append(curve)

    if not kept_parts:
        return summary, {"jump_filtered_points": 0, "jump_filtered_curves": 0}
    return pd.concat(kept_parts, ignore_index=True), {
        "jump_filtered_points": removed_points,
        "jump_filtered_curves": removed_curves,
    }


def _odd_window(window: int, length: int) -> int:
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window > length:
        window = length if length % 2 == 1 else length - 1
    return max(1, window)


def smooth_summary(summary: pd.DataFrame, *, window: int, polyorder: int) -> pd.DataFrame:
    if window <= 1:
        return summary
    if savgol_filter is None:
        raise RuntimeError("scipy is required for --smooth-window Savitzky-Golay smoothing")

    out = summary.copy()
    group_cols = ["joint", "amplitude", "direction"]
    for _, index in out.groupby(group_cols, sort=False).groups.items():
        group_index = list(index)
        ordered = out.loc[group_index].sort_values("angle_deg")
        if len(ordered) <= polyorder + 2:
            continue
        actual_window = _odd_window(window, len(ordered))
        if actual_window <= polyorder:
            continue
        for col in ["forward_vel_deg_s", "rate_per_command"]:
            values = ordered[col].to_numpy(dtype=np.float64)
            smoothed = savgol_filter(values, actual_window, polyorder, mode="interp")
            smoothed = np.maximum(smoothed, 1e-6)
            out.loc[ordered.index, col] = smoothed
    return out


def _parse_amplitudes(text: str) -> set[float]:
    return {round(float(item.strip()), 3) for item in text.split(",") if item.strip()}


def build_universal_shape(summary: pd.DataFrame, *, amplitudes: set[float], combine_directions: bool = False) -> pd.DataFrame:
    work = summary.copy()
    work["amplitude_key"] = work["amplitude"].round(3)
    work = work[work["amplitude_key"].isin(amplitudes)].copy()
    if work.empty:
        return pd.DataFrame()

    normalized_parts: list[pd.DataFrame] = []
    for _, curve in work.groupby(["joint", "amplitude_key", "direction"], sort=False):
        curve = curve.sort_values("angle_deg").copy()
        reference = float(curve["rate_per_command"].median())
        if not np.isfinite(reference) or reference <= 1e-6:
            continue
        curve["shape_factor"] = curve["rate_per_command"] / reference
        curve["curve_reference_rate"] = reference
        curve["source_curve"] = f"amp={float(curve['amplitude_key'].iloc[0]):.3f},dir={int(curve['direction'].iloc[0]):+d}"
        normalized_parts.append(curve)

    if not normalized_parts:
        return pd.DataFrame()

    normalized = pd.concat(normalized_parts, ignore_index=True)
    group_cols = ["joint", "angle_bin_deg"] if combine_directions else ["joint", "direction", "angle_bin_deg"]
    grouped = normalized.groupby(group_cols, as_index=False)
    universal = grouped.agg(
        samples=("samples", "sum"),
        source_curves=("source_curve", "nunique"),
        angle_deg=("angle_deg", "median"),
        shape_factor=("shape_factor", "median"),
        shape_factor_q25=("shape_factor", lambda s: float(s.quantile(0.25))),
        shape_factor_q75=("shape_factor", lambda s: float(s.quantile(0.75))),
        reference_rate_per_command=("curve_reference_rate", "median"),
    )
    if combine_directions:
        universal["direction"] = 0
    universal["universal_rate_per_command"] = universal["shape_factor"] * universal["reference_rate_per_command"]
    return universal


def smooth_universal_shape(universal: pd.DataFrame, *, sigma_deg: float) -> pd.DataFrame:
    if universal.empty or sigma_deg <= 0.0:
        return universal
    out = universal.copy()
    out["shape_factor_raw"] = out["shape_factor"]
    out["universal_rate_per_command_raw"] = out["universal_rate_per_command"]

    for _, index in out.groupby(["joint", "direction"], sort=False).groups.items():
        ordered = out.loc[list(index)].sort_values("angle_deg")
        if len(ordered) < 4:
            continue
        x = ordered["angle_deg"].to_numpy(dtype=np.float64)
        y = ordered["shape_factor_raw"].to_numpy(dtype=np.float64)
        samples = np.maximum(1.0, ordered["samples"].to_numpy(dtype=np.float64))
        smoothed = np.empty_like(y)
        for i, xi in enumerate(x):
            weights = np.exp(-0.5 * ((x - xi) / sigma_deg) ** 2) * np.sqrt(samples)
            weight_sum = float(np.sum(weights))
            smoothed[i] = y[i] if weight_sum <= 1e-9 else float(np.sum(weights * y) / weight_sum)
        smoothed = np.maximum(smoothed, 1e-6)
        out.loc[ordered.index, "shape_factor"] = smoothed
        out.loc[ordered.index, "universal_rate_per_command"] = smoothed * ordered["reference_rate_per_command"]
    return out


def filter_universal_jumps(universal: pd.DataFrame, *, factor: float, min_delta: float) -> tuple[pd.DataFrame, dict[str, int]]:
    if universal.empty or factor <= 0:
        return universal, {"universal_jump_filtered_points": 0, "universal_jump_filtered_curves": 0}

    kept_parts: list[pd.DataFrame] = []
    removed_points = 0
    removed_curves = 0
    for _, curve in universal.groupby(["joint", "direction"], sort=False):
        curve = curve.sort_values("angle_deg").copy()
        values = curve["shape_factor"].to_numpy(dtype=np.float64)
        bad = _large_jump_mask(values, factor, min_delta)
        if np.any(bad) and len(curve) - int(np.sum(bad)) >= 4:
            removed_points += int(np.sum(bad))
            removed_curves += 1
            curve = curve.loc[~bad].copy()
        kept_parts.append(curve)

    if not kept_parts:
        return universal, {"universal_jump_filtered_points": 0, "universal_jump_filtered_curves": 0}
    return pd.concat(kept_parts, ignore_index=True), {
        "universal_jump_filtered_points": removed_points,
        "universal_jump_filtered_curves": removed_curves,
    }


def plot_universal_shape(summary: pd.DataFrame, universal: pd.DataFrame, joint: str, output_dir: Path) -> list[Path]:
    joint_universal = universal[universal["joint"] == joint].copy()
    if joint_universal.empty:
        return []

    joint_summary = summary[summary["joint"] == joint].copy()
    output_paths: list[Path] = []
    combined = set(joint_universal["direction"].dropna().astype(int).unique()) == {0}

    if combined:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        for (amp, direction), curve in joint_summary.groupby(["amplitude", "direction"], sort=True):
            curve = curve.sort_values("angle_deg")
            reference = float(curve["rate_per_command"].median())
            if not np.isfinite(reference) or reference <= 1e-6:
                continue
            ax.plot(
                curve["angle_deg"],
                curve["rate_per_command"] / reference,
                color="0.7",
                linewidth=1.0,
                alpha=0.4,
            )

        curve = joint_universal.sort_values("angle_deg")
        shape_col = "shape_factor_raw" if "shape_factor_raw" in curve.columns else "shape_factor"
        ax.fill_between(
            curve["angle_deg"],
            curve["shape_factor_q25"],
            curve["shape_factor_q75"],
            color="tab:blue",
            alpha=0.18,
            label="25-75% source spread",
        )
        if shape_col != "shape_factor":
            ax.plot(
                curve["angle_deg"],
                curve[shape_col],
                color="tab:blue",
                linewidth=1.5,
                alpha=0.45,
                label="universal raw",
            )
        ax.plot(
            curve["angle_deg"],
            curve["shape_factor"],
            color="tab:blue",
            linewidth=3,
            label="combined universal smooth" if shape_col != "shape_factor" else "combined universal shape",
        )
        ax.axhline(1.0, color="0.25", linewidth=1, linestyle="--")
        ax.set_xlabel(f"{joint} angle (deg)")
        ax.set_ylabel("relative rate shape")
        ax.set_title(f"{joint} combined-direction universal normalized linkage shape")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = output_dir / f"{joint}_universal_shape_combined.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(path)
        return output_paths

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, direction in zip(axes, [1, -1], strict=True):
        sub_summary = joint_summary[joint_summary["direction"] == direction]
        for amp in sorted(float(v) for v in sub_summary["amplitude"].dropna().unique()):
            curve = sub_summary[np.isclose(sub_summary["amplitude"], amp)].sort_values("angle_deg")
            if curve.empty:
                continue
            reference = float(curve["rate_per_command"].median())
            if not np.isfinite(reference) or reference <= 1e-6:
                continue
            ax.plot(
                curve["angle_deg"],
                curve["rate_per_command"] / reference,
                color="0.7",
                linewidth=1.0,
                alpha=0.45,
            )

        curve = joint_universal[joint_universal["direction"] == direction].sort_values("angle_deg")
        if not curve.empty:
            shape_col = "shape_factor_raw" if "shape_factor_raw" in curve.columns else "shape_factor"
            ax.fill_between(
                curve["angle_deg"],
                curve["shape_factor_q25"],
                curve["shape_factor_q75"],
                color="tab:blue",
                alpha=0.18,
                label="25-75% source spread",
            )
            if shape_col != "shape_factor":
                ax.plot(
                    curve["angle_deg"],
                    curve[shape_col],
                    color="tab:blue",
                    linewidth=1.5,
                    alpha=0.45,
                    label="universal raw",
                )
            ax.plot(
                curve["angle_deg"],
                curve["shape_factor"],
                color="tab:blue",
                linewidth=3,
                label="universal smooth" if shape_col != "shape_factor" else "universal shape",
            )
        ax.axhline(1.0, color="0.25", linewidth=1, linestyle="--")
        ax.set_ylabel("relative rate shape")
        ax.set_title(f"{joint} direction {direction:+d}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel(f"{joint} angle (deg)")
    fig.suptitle(f"{joint} universal normalized linkage shape")
    fig.tight_layout()
    path = output_dir / f"{joint}_universal_shape.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)
    return output_paths


def plot_joint(summary: pd.DataFrame, joint: str, output_dir: Path) -> list[Path]:
    joint_df = summary[summary["joint"] == joint].copy()
    if joint_df.empty:
        return []

    output_paths: list[Path] = []
    amplitudes = sorted(float(v) for v in joint_df["amplitude"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    colors = {amp: cmap(i / max(1, len(amplitudes) - 1)) for i, amp in enumerate(amplitudes)}

    for metric, ylabel, suffix in [
        ("forward_vel_deg_s", "joint velocity (deg/s)", "velocity"),
        ("rate_per_command", "rate per command (deg/s / cmd)", "normalized_rate"),
    ]:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for ax, direction in zip(axes, [1, -1], strict=True):
            sub = joint_df[joint_df["direction"] == direction]
            for amp in amplitudes:
                curve = sub[sub["amplitude"] == amp].sort_values("angle_deg")
                if curve.empty:
                    continue
                ax.plot(
                    curve["angle_deg"],
                    curve[metric],
                    label=f"cmd={amp:.1f}",
                    color=colors[amp],
                    linewidth=2,
                )
            ax.set_ylabel(ylabel)
            ax.set_title(f"{joint} direction {direction:+d}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        axes[-1].set_xlabel(f"{joint} angle (deg)")
        fig.suptitle(f"{joint} linkage rates")
        fig.tight_layout()
        path = output_dir / f"{joint}_{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(path)
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanitize and plot linkage-rate logger output")
    parser.add_argument("csv_path", help="Input CSV from tools/linkage_rate_logger.py")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smooth-window", type=int, default=1, help="Odd Savitzky-Golay window in binned samples; 1 disables smoothing")
    parser.add_argument("--smooth-polyorder", type=int, default=2)
    parser.add_argument("--universal-amplitudes", default="0.6,0.8,1.0", help="Command amplitudes used for universal normalized shape")
    parser.add_argument("--combine-directions-for-universal", action="store_true", help="Build one geometry-only universal shape per joint by overlaying both directions after per-curve normalization")
    parser.add_argument("--jump-filter-factor", type=float, default=0.0, help="Drop curve bins whose point-to-point delta is this many typical deltas; 0 disables")
    parser.add_argument("--jump-filter-min-delta", type=float, default=6.0, help="Minimum rate-per-command jump required before a bin can be removed")
    parser.add_argument("--universal-jump-filter-factor", type=float, default=0.0, help="Drop universal-shape bins whose relative point-to-point delta is too large; 0 disables")
    parser.add_argument("--universal-jump-filter-min-delta", type=float, default=0.35)
    parser.add_argument("--universal-smooth-sigma-deg", type=float, default=0.0, help="Gaussian smoothing sigma for universal shape in degrees; 0 disables")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = ROOT_DIR / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path)
    sanitized, stats = sanitize(raw)
    summary = summarize_bins(sanitized)
    summary, jump_stats = filter_summary_jumps(
        summary,
        factor=args.jump_filter_factor,
        min_delta=args.jump_filter_min_delta,
    )
    stats.update(jump_stats)
    summary = smooth_summary(summary, window=args.smooth_window, polyorder=args.smooth_polyorder)
    universal = build_universal_shape(
        summary,
        amplitudes=_parse_amplitudes(args.universal_amplitudes),
        combine_directions=args.combine_directions_for_universal,
    )
    universal, universal_jump_stats = filter_universal_jumps(
        universal,
        factor=args.universal_jump_filter_factor,
        min_delta=args.universal_jump_filter_min_delta,
    )
    stats.update(universal_jump_stats)
    universal = smooth_universal_shape(universal, sigma_deg=args.universal_smooth_sigma_deg)

    base = csv_path.stem
    sanitized_path = output_dir / f"{base}_sanitized.csv"
    summary_path = output_dir / f"{base}_summary.csv"
    universal_path = output_dir / f"{base}_universal_shape.csv"
    sanitized.to_csv(sanitized_path, index=False)
    summary.to_csv(summary_path, index=False)
    if not universal.empty:
        universal.to_csv(universal_path, index=False)

    plot_paths: list[Path] = []
    for joint in JOINTS:
        plot_paths.extend(plot_joint(summary, joint, output_dir))
        if not universal.empty:
            plot_paths.extend(plot_universal_shape(summary, universal, joint, output_dir))

    print(f"Input:      {csv_path}")
    print(f"Sanitized:  {sanitized_path}")
    print(f"Summary:    {summary_path}")
    if not universal.empty:
        print(f"Universal:  {universal_path}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    for path in plot_paths:
        print(f"Plot:       {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
