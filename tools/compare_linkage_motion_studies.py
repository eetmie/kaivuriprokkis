#!/usr/bin/env python3
"""Compare logger-derived linkage shapes with SolidWorks motion studies.

The motion studies export cylinder displacement vs joint angle.  For comparison
to logger joint velocity per command, this uses the normalized inverse slope
``d(angle) / d(cylinder displacement)`` as the geometry-only shape reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOGGER_SHAPE = ROOT_DIR / "data_collection" / "hydraulic_data" / "processed_universal_combined_sigma8" / "linkage_rate_20260504_195552_universal_shape.csv"


STUDIES = {
    "boom": {
        "file": "liftboom.csv",
        "columns": ["time_s", "cylinder_a_mm", "angle_deg", "cylinder_mm", "linear4_mm", "linear5_mm"],
        "skiprows": 1,
        "displacement_col": "cylinder_mm",
    },
    "arm": {
        "file": "tiltboom.csv",
        "columns": ["time_s", "cylinder_mm", "angle_deg"],
        "skiprows": 1,
        "displacement_col": "cylinder_mm",
    },
    "bucket": {
        "file": "scoop.csv",
        "columns": ["time_s", "cylinder_mm", "angle_deg"],
        "skiprows": 1,
        "displacement_col": "cylinder_mm",
    },
}


def _normalized_inverse_slope(df: pd.DataFrame, displacement_col: str) -> pd.DataFrame:
    work = df[["angle_deg", displacement_col]].copy()
    work = work.apply(pd.to_numeric, errors="coerce").dropna()
    work = work.sort_values("angle_deg").drop_duplicates("angle_deg")
    angle = work["angle_deg"].to_numpy(dtype=np.float64)
    displacement = work[displacement_col].to_numpy(dtype=np.float64)
    if len(angle) < 5:
        raise RuntimeError("Not enough motion-study samples")

    d_length_d_angle = np.gradient(displacement, angle)
    joint_speed_shape = 1.0 / np.maximum(np.abs(d_length_d_angle), 1e-9)
    reference = float(np.median(joint_speed_shape[np.isfinite(joint_speed_shape)]))
    work["shape_factor"] = joint_speed_shape / reference
    return work[["angle_deg", "shape_factor"]]


def load_motion_shapes(study_dir: Path) -> dict[str, pd.DataFrame]:
    shapes: dict[str, pd.DataFrame] = {}
    for joint, spec in STUDIES.items():
        path = study_dir / str(spec["file"])
        raw = pd.read_csv(path, skiprows=int(spec["skiprows"]))
        raw.columns = list(spec["columns"])
        shapes[joint] = _normalized_inverse_slope(raw, str(spec["displacement_col"]))
    return shapes


def compare_shapes(logger_shape: pd.DataFrame, motion_shapes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for joint, motion in motion_shapes.items():
        observed = logger_shape[logger_shape["joint"] == joint].sort_values("angle_deg")
        if observed.empty:
            continue
        metrics = _compare_one_transform(observed, motion)
        mirrored = motion.copy()
        mirrored["angle_deg"] = -mirrored["angle_deg"]
        mirrored = mirrored.sort_values("angle_deg")
        mirror_metrics = _compare_one_transform(observed, mirrored)
        if metrics is None:
            if mirror_metrics is None:
                rows.append({"joint": joint, "samples": 0})
            else:
                mirror_metrics["joint"] = joint
                mirror_metrics["angle_transform"] = "mirrored"
                rows.append(mirror_metrics)
            continue
        metrics["joint"] = joint
        metrics["angle_transform"] = "as_is"
        if mirror_metrics is not None:
            metrics["mirror_corr"] = mirror_metrics["corr"]
            metrics["mirror_mean_abs_delta"] = mirror_metrics["mean_abs_delta"]
        rows.append(metrics)
    return pd.DataFrame(rows)


def _compare_one_transform(observed: pd.DataFrame, motion: pd.DataFrame) -> dict[str, float | int] | None:
    lo = max(float(observed["angle_deg"].min()), float(motion["angle_deg"].min()))
    hi = min(float(observed["angle_deg"].max()), float(motion["angle_deg"].max()))
    common = observed[(observed["angle_deg"] >= lo) & (observed["angle_deg"] <= hi)].copy()
    if len(common) < 5:
        return None
    motion_interp = np.interp(common["angle_deg"], motion["angle_deg"], motion["shape_factor"])
    observed_shape = common["shape_factor"].to_numpy(dtype=np.float64)
    motion_interp = motion_interp / np.median(motion_interp)
    observed_shape = observed_shape / np.median(observed_shape)
    delta = observed_shape - motion_interp
    corr = float(np.corrcoef(observed_shape, motion_interp)[0, 1]) if len(common) > 2 else float("nan")
    return {
        "samples": int(len(common)),
        "overlap_min_deg": lo,
        "overlap_max_deg": hi,
        "corr": corr,
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "rms_delta": float(np.sqrt(np.mean(delta**2))),
        "max_abs_delta": float(np.max(np.abs(delta))),
    }


def plot_comparison(logger_shape: pd.DataFrame, motion_shapes: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for ax, joint in zip(axes, ["boom", "arm", "bucket"], strict=True):
        observed = logger_shape[logger_shape["joint"] == joint].sort_values("angle_deg")
        motion = motion_shapes[joint].sort_values("angle_deg")
        lo = max(float(observed["angle_deg"].min()), float(motion["angle_deg"].min()))
        hi = min(float(observed["angle_deg"].max()), float(motion["angle_deg"].max()))
        common = observed[(observed["angle_deg"] >= lo) & (observed["angle_deg"] <= hi)].copy()
        if len(common) >= 5:
            motion_interp = np.interp(common["angle_deg"], motion["angle_deg"], motion["shape_factor"])
            observed_shape = common["shape_factor"].to_numpy(dtype=np.float64)
            ax.plot(
                common["angle_deg"],
                observed_shape / np.median(observed_shape),
                label="logger universal",
                linewidth=3,
            )
            ax.plot(
                common["angle_deg"],
                motion_interp / np.median(motion_interp),
                label="SolidWorks inverse slope",
                linewidth=2,
            )
        ax.axhline(1.0, color="0.25", linewidth=1, linestyle="--")
        ax.set_title(joint)
        ax.set_ylabel("relative joint-speed shape")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("joint angle (deg)")
    fig.suptitle("Logger linkage shape vs SolidWorks motion-study geometry")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare logger-derived linkage shape with SolidWorks motion studies")
    parser.add_argument("study_dir", help="Directory containing liftboom.csv, tiltboom.csv, and scoop.csv")
    parser.add_argument("--logger-shape", default=str(DEFAULT_LOGGER_SHAPE))
    parser.add_argument("--output", default=str(DEFAULT_LOGGER_SHAPE.parent / "motion_study_comparison.png"))
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    logger_path = Path(args.logger_shape)
    if not logger_path.is_absolute():
        logger_path = ROOT_DIR / logger_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path

    logger_shape = pd.read_csv(logger_path)
    motion_shapes = load_motion_shapes(study_dir)
    comparison = compare_shapes(logger_shape, motion_shapes)
    plot_comparison(logger_shape, motion_shapes, output_path)

    print(comparison.to_string(index=False))
    print(f"Plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
