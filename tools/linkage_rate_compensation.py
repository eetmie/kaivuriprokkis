#!/usr/bin/env python3
"""Prototype linkage-rate command compensation.

The compensation table is the ``*_summary.csv`` produced by
``tools/analyze_linkage_rates.py``.  For each boom/arm/bucket command, this
looks up the measured rate-per-command at the current joint angle and scales the
open-loop command toward that joint/direction/amplitude's median rate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT_DIR / "data_collection" / "hydraulic_data" / "processed_universal_combined_sigma8" / "linkage_rate_20260504_195552_summary.csv"
DEFAULT_UNIVERSAL_TABLE = DEFAULT_TABLE.parent / DEFAULT_TABLE.name.replace("_summary.csv", "_universal_shape.csv")

COMMAND_TO_JOINT = {
    "boom": ("boom", 1),
    "arm": ("arm", 2),
    "bucket": ("bucket", 3),
}


@dataclass(frozen=True)
class RateCurve:
    angles_deg: np.ndarray
    rates_deg_s_per_cmd: np.ndarray
    reference_rate: float


class LinkageRateCompensator:
    """Angle-dependent open-loop valve command scaler."""

    def __init__(
        self,
        table_path: str | Path = DEFAULT_TABLE,
        *,
        min_factor: float = 0.35,
        max_factor: float = 2.25,
        min_command: float = 0.05,
    ) -> None:
        self.table_path = Path(table_path)
        if not self.table_path.is_absolute():
            self.table_path = ROOT_DIR / self.table_path
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)
        self.min_command = float(min_command)
        self.curves = self._load_curves(self.table_path)

    @staticmethod
    def _nearest_amplitude(amplitude: float, available: list[float]) -> float:
        return min(available, key=lambda value: abs(value - amplitude))

    @staticmethod
    def _direction(command: float) -> int:
        return 1 if command >= 0.0 else -1

    def _load_curves(self, table_path: Path) -> dict[tuple[str, float, int], RateCurve]:
        if not table_path.exists():
            raise FileNotFoundError(table_path)

        rows: dict[tuple[str, float, int], list[tuple[float, float]]] = {}
        with table_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                joint = row.get("joint", "")
                if joint not in {"boom", "arm", "bucket"}:
                    continue
                try:
                    amplitude = round(float(row["amplitude"]), 3)
                    direction = int(float(row["direction"]))
                    angle = float(row["angle_deg"])
                    rate = float(row["rate_per_command"])
                except (KeyError, TypeError, ValueError):
                    continue
                if direction == 0 or not np.isfinite(angle) or not np.isfinite(rate) or rate <= 1e-6:
                    continue
                rows.setdefault((joint, amplitude, direction), []).append((angle, rate))

        curves: dict[tuple[str, float, int], RateCurve] = {}
        for key, points in rows.items():
            points = sorted(points, key=lambda item: item[0])
            angles = np.asarray([p[0] for p in points], dtype=np.float32)
            rates = np.asarray([p[1] for p in points], dtype=np.float32)
            if len(angles) < 2:
                continue
            curves[key] = RateCurve(
                angles_deg=angles,
                rates_deg_s_per_cmd=rates,
                reference_rate=float(np.median(rates)),
            )
        if not curves:
            raise RuntimeError(f"No usable rate curves in {table_path}")
        return curves

    def available_amplitudes(self, joint: str, direction: int) -> list[float]:
        return sorted({amp for (curve_joint, amp, curve_dir) in self.curves if curve_joint == joint and curve_dir == direction})

    def correction_factor(self, joint: str, command: float, angle_deg: float) -> float:
        if abs(command) < self.min_command:
            return 1.0

        direction = self._direction(command)
        amplitudes = self.available_amplitudes(joint, direction)
        if not amplitudes:
            return 1.0

        amplitude = self._nearest_amplitude(abs(command), amplitudes)
        curve = self.curves[(joint, amplitude, direction)]
        local_rate = float(np.interp(angle_deg, curve.angles_deg, curve.rates_deg_s_per_cmd))
        if local_rate <= 1e-6:
            return 1.0

        factor = curve.reference_rate / local_rate
        return float(np.clip(factor, self.min_factor, self.max_factor))

    def apply(self, commands: dict[str, float], joint_angles_deg) -> dict[str, float]:
        corrected = dict(commands)
        angles = np.asarray(joint_angles_deg, dtype=np.float32)
        for command_name, (joint, joint_index) in COMMAND_TO_JOINT.items():
            if command_name not in corrected or joint_index >= len(angles):
                continue
            command = float(corrected[command_name])
            factor = self.correction_factor(joint, command, float(angles[joint_index]))
            corrected[command_name] = float(np.clip(command * factor, -1.0, 1.0))
        return corrected


class UniversalShapeCompensator:
    """Smoothed universal-shape compensator loaded from *_universal_shape.csv.

    Uses the combined-direction ``shape_factor`` column (Gaussian-smoothed over
    all amplitudes and directions) so there is no per-amplitude or per-direction
    lookup.  Correction = 1 / shape_factor, clamped to [min_factor, max_factor].
    """

    def __init__(
        self,
        table_path: str | Path = DEFAULT_UNIVERSAL_TABLE,
        *,
        min_factor: float = 0.35,
        max_factor: float = 2.25,
        min_command: float = 0.05,
    ) -> None:
        self.table_path = Path(table_path)
        if not self.table_path.is_absolute():
            self.table_path = ROOT_DIR / self.table_path
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)
        self.min_command = float(min_command)
        self.curves = self._load_curves(self.table_path)

    def _load_curves(self, table_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not table_path.exists():
            raise FileNotFoundError(table_path)
        rows: dict[str, list[tuple[float, float]]] = {}
        with table_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                joint = row.get("joint", "")
                if joint not in {"boom", "arm", "bucket"}:
                    continue
                try:
                    angle = float(row["angle_deg"])
                    shape = float(row["shape_factor"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isfinite(angle) or not np.isfinite(shape) or shape <= 1e-6:
                    continue
                rows.setdefault(joint, []).append((angle, shape))

        curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for joint, points in rows.items():
            points = sorted(points, key=lambda p: p[0])
            angles = np.asarray([p[0] for p in points], dtype=np.float32)
            shapes = np.asarray([p[1] for p in points], dtype=np.float32)
            if len(angles) < 2:
                continue
            curves[joint] = (angles, shapes)
        if not curves:
            raise RuntimeError(f"No usable universal shape curves in {table_path}")
        return curves

    def correction_factor(self, joint: str, command: float, angle_deg: float) -> float:
        if abs(command) < self.min_command:
            return 1.0
        curve = self.curves.get(joint)
        if curve is None:
            return 1.0
        angles, shapes = curve
        shape = float(np.interp(angle_deg, angles, shapes))
        if shape <= 1e-6:
            return 1.0
        return float(np.clip(1.0 / shape, self.min_factor, self.max_factor))

    def apply(self, commands: dict[str, float], joint_angles_deg) -> dict[str, float]:
        corrected = dict(commands)
        angles = np.asarray(joint_angles_deg, dtype=np.float32)
        for command_name, (joint, joint_index) in COMMAND_TO_JOINT.items():
            if command_name not in corrected or joint_index >= len(angles):
                continue
            command = float(corrected[command_name])
            factor = self.correction_factor(joint, command, float(angles[joint_index]))
            corrected[command_name] = float(np.clip(command * factor, -1.0, 1.0))
        return corrected

    def pump_correction_factor(self, commands: dict[str, float], joint_angles_deg) -> float:
        """Return a command-weighted pump gain correction factor across active joints.

        Each active joint contributes 1/shape_factor (so slow linkage positions boost
        pump, fast positions reduce it). Weights are proportional to command magnitude.
        Returns 1.0 when no joints are active.
        """
        angles = np.asarray(joint_angles_deg, dtype=np.float32)
        weighted_sum = 0.0
        total_weight = 0.0
        for command_name, (joint, joint_index) in COMMAND_TO_JOINT.items():
            command = float(commands.get(command_name, 0.0))
            weight = abs(command)
            if weight < self.min_command or joint_index >= len(angles):
                continue
            curve = self.curves.get(joint)
            if curve is None:
                continue
            curve_angles, shapes = curve
            shape = float(np.interp(float(angles[joint_index]), curve_angles, shapes))
            if shape <= 1e-6:
                continue
            weighted_sum += weight / shape
            total_weight += weight
        if total_weight < 1e-6:
            return 1.0
        return float(np.clip(weighted_sum / total_weight, self.min_factor, self.max_factor))


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview prototype linkage-rate command compensation")
    parser.add_argument("--table", default=str(DEFAULT_TABLE), help="Path to *_summary.csv")
    parser.add_argument("--joint", choices=["boom", "arm", "bucket"], required=True)
    parser.add_argument("--command", type=float, required=True)
    parser.add_argument("--angle-deg", type=float, required=True)
    args = parser.parse_args()

    command_name = args.joint
    joint_index = COMMAND_TO_JOINT[command_name][1]
    angles = np.zeros(4, dtype=np.float32)
    angles[joint_index] = args.angle_deg

    compensator = LinkageRateCompensator(args.table)
    before = {command_name: args.command}
    after = compensator.apply(before, angles)
    print(f"{command_name}: {args.command:+.3f} -> {after[command_name]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
