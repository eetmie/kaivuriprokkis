#!/usr/bin/env python3
"""Scoop-only hydraulic velocity tester.

This is a hardware test tool for the hardest joint first: bucket/scoop.
It drives the scoop valve back and forth inside a local IMU-derived angle
window, logs velocity response, and recommends a small set of smooth speed
bins for later feed-forward + PI velocity control.

Typical robot-side use:

    python tools/scoop_velocity_tester.py --i-understand --enable-pressure

The tool uses IMU joint angles, not external ground truth. That is enough for
relative velocity mapping and valve smoothness testing. Keep the first runs in
free air and use a small window.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.ik import load_excavator_robot_config, canonical_joint_angles_from_imus
from modules.hardware_interface import HardwareInterface


JOINT_NAME = "bucket"
JOINT_INDEX = 3
PWM_NAME = "bucket"
DEFAULT_OUTPUT_DIR = "data_collection/hydraulic_data"
DEFAULT_COMMANDS = "0.08,0.12,0.16,0.22,0.30,0.42,0.55"


@dataclass(frozen=True)
class Sample:
    t_s: float
    phase: str
    pass_index: int
    amplitude: float
    direction: int
    command: float
    bucket_deg: float
    bucket_vel_deg_s: float
    pressure_age_s: float
    pressure_json: str


@dataclass(frozen=True)
class SweepSummary:
    amplitude: float
    direction: int
    command: float
    samples: int
    median_progress_vel_deg_s: float
    mean_progress_vel_deg_s: float
    std_progress_vel_deg_s: float
    ripple_ratio: float
    start_delay_s: float
    angle_travel_deg: float
    score: float
    usable: bool


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _parse_command_ladder(value: str) -> list[float]:
    commands = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        command = abs(float(part))
        if command <= 0.0 or command > 1.0:
            raise argparse.ArgumentTypeError("command amplitudes must be in (0, 1]")
        commands.append(command)
    if not commands:
        raise argparse.ArgumentTypeError("at least one command amplitude is required")
    return sorted(set(commands))


def _read_bucket_angle_deg(hardware: HardwareInterface, robot_config: Any) -> float:
    quats, _ = hardware.read_all_imu_quaternions()
    if quats is None:
        raise RuntimeError("IMU quaternions unavailable")
    angles = canonical_joint_angles_from_imus(np.asarray(quats, dtype=np.float32), robot_config)
    return float(math.degrees(float(angles[JOINT_INDEX])))


def summarize_sweeps(
    samples: Iterable[Sample],
    *,
    settle_ignore_s: float,
    min_vel_deg_s: float,
    max_ripple_ratio: float,
    min_samples: int = 8,
) -> list[SweepSummary]:
    """Summarize sweep rows into per-command velocity quality metrics."""
    groups: dict[tuple[int, float, int], list[Sample]] = {}
    starts: dict[tuple[int, float, int], float] = {}
    for sample in samples:
        if sample.phase not in {"sweep_settle", "sweep"}:
            continue
        key = (sample.pass_index, round(sample.amplitude, 6), sample.direction)
        starts.setdefault(key, sample.t_s)
        groups.setdefault(key, []).append(sample)

    summaries: list[SweepSummary] = []
    for key, rows in sorted(groups.items()):
        _, amplitude, direction = key
        start_t = starts[key]
        steady = [row for row in rows if row.phase == "sweep" and row.t_s - start_t >= settle_ignore_s]
        if not steady:
            steady = [row for row in rows if row.phase == "sweep"]

        progress_vel = np.asarray(
            [float(direction) * row.bucket_vel_deg_s for row in steady],
            dtype=np.float64,
        )
        if progress_vel.size:
            median_vel = float(np.median(progress_vel))
            mean_vel = float(np.mean(progress_vel))
            std_vel = float(np.std(progress_vel))
        else:
            median_vel = mean_vel = std_vel = 0.0
        ripple = std_vel / max(abs(median_vel), 1e-6)

        start_delay = math.inf
        moving_threshold = max(0.5, min_vel_deg_s * 0.25)
        for row in rows:
            if float(direction) * row.bucket_vel_deg_s >= moving_threshold:
                start_delay = row.t_s - start_t
                break

        angles = [row.bucket_deg for row in rows]
        angle_travel = float(abs(angles[-1] - angles[0])) if len(angles) >= 2 else 0.0
        usable = (
            progress_vel.size >= min_samples
            and median_vel >= min_vel_deg_s
            and ripple <= max_ripple_ratio
            and math.isfinite(start_delay)
        )
        score = median_vel / (1.0 + ripple + 0.5 * min(start_delay, 2.0))
        summaries.append(SweepSummary(
            amplitude=float(amplitude),
            direction=int(direction),
            command=float(direction) * float(amplitude),
            samples=int(progress_vel.size),
            median_progress_vel_deg_s=median_vel,
            mean_progress_vel_deg_s=mean_vel,
            std_progress_vel_deg_s=std_vel,
            ripple_ratio=float(ripple),
            start_delay_s=float(start_delay),
            angle_travel_deg=angle_travel,
            score=float(score),
            usable=bool(usable),
        ))
    return summaries


def recommend_speed_bins(summaries: Iterable[SweepSummary], *, bins_per_direction: int) -> dict[str, list[dict[str, float]]]:
    """Pick evenly spaced smooth speed bins from usable summary rows."""
    out: dict[str, list[dict[str, float]]] = {"positive": [], "negative": []}
    for direction, label in ((1, "positive"), (-1, "negative")):
        rows = [s for s in summaries if s.direction == direction and s.usable]
        rows.sort(key=lambda s: s.median_progress_vel_deg_s)
        if not rows:
            continue
        if len(rows) <= bins_per_direction:
            chosen = rows
        else:
            idxs = np.linspace(0, len(rows) - 1, bins_per_direction)
            chosen = [rows[int(round(i))] for i in idxs]
        out[label] = [
            {
                "target_deg_s": float(row.median_progress_vel_deg_s),
                "command": float(row.command),
                "ripple_ratio": float(row.ripple_ratio),
                "start_delay_s": float(row.start_delay_s),
            }
            for row in chosen
        ]
    return out


class ScoopVelocityTester:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.log = logging.getLogger("scoop_velocity_tester")
        self.robot_config = load_excavator_robot_config(args.control_config_file)
        self.hardware = HardwareInterface(
            config_file=args.config_file,
            control_config_file=args.control_config_file,
            pump_auto_mode=args.pump_auto,
            cleanup_disable_osc=False,
            enable_adc=args.enable_pressure,
            start_adc_reader=args.enable_pressure,
            adc_sample_hz=args.pressure_hz,
            log_level=args.log_level,
        )
        self.start_time = time.monotonic()
        self.prev_angle_deg: float | None = None
        self.prev_time: float | None = None
        self.last_vel_deg_s = 0.0
        self.samples: list[Sample] = []

    def wait_ready(self) -> None:
        self.log.info("Waiting for hardware")
        while not self.hardware.is_hardware_ready():
            time.sleep(0.1)
        self.hardware.set_pump_enabled(True)
        self.log.info("Hardware ready; pump enabled")

    def shutdown(self) -> None:
        try:
            self.hardware.reset(reset_pump=True)
        finally:
            self.hardware.shutdown()

    def read_state(self) -> tuple[float, float, dict[str, Any], float]:
        now = time.monotonic()
        angle = _read_bucket_angle_deg(self.hardware, self.robot_config)
        if self.prev_angle_deg is None or self.prev_time is None:
            vel = 0.0
        else:
            dt = max(1e-6, now - self.prev_time)
            vel = (angle - self.prev_angle_deg) / dt
        self.prev_angle_deg = angle
        self.prev_time = now

        pressure: dict[str, Any] = {}
        pressure_age = math.inf
        if self.args.enable_pressure:
            snapshot = self.hardware.get_latest_adc_snapshot()
            pressure = dict(snapshot.get("readings") or {})
            ts = snapshot.get("timestamp")
            if ts is not None:
                pressure_age = max(0.0, now - float(ts))
        self.last_vel_deg_s = float(vel)
        return float(angle), float(vel), pressure, float(pressure_age)

    def write_sample(
        self,
        writer: csv.DictWriter,
        *,
        phase: str,
        pass_index: int,
        amplitude: float,
        direction: int,
        command: float,
    ) -> None:
        angle, vel, pressure, pressure_age = self.read_state()
        sample = Sample(
            t_s=time.monotonic() - self.start_time,
            phase=phase,
            pass_index=pass_index,
            amplitude=float(amplitude),
            direction=int(direction),
            command=float(command),
            bucket_deg=angle,
            bucket_vel_deg_s=vel,
            pressure_age_s=pressure_age,
            pressure_json=json.dumps(pressure, sort_keys=True, separators=(",", ":")),
        )
        self.samples.append(sample)
        writer.writerow(asdict(sample))

    def hold_zero(self, writer: csv.DictWriter, duration_s: float) -> None:
        end_t = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < end_t:
            self.hardware.reset(reset_pump=False)
            self.write_sample(
                writer,
                phase="hold_zero",
                pass_index=-1,
                amplitude=0.0,
                direction=0,
                command=0.0,
            )
            time.sleep(1.0 / self.args.rate)

    def sweep(
        self,
        writer: csv.DictWriter,
        *,
        pass_index: int,
        amplitude: float,
        direction: int,
        target_angle_deg: float,
    ) -> None:
        command = self.args.bucket_command_sign * float(direction) * float(amplitude)
        start_t = time.monotonic()
        settle_until = start_t + self.args.settle_ignore_s
        stall_since: float | None = None
        stop_reason = "timeout"
        while time.monotonic() - start_t < self.args.sweep_timeout_s:
            self.hardware.send_named_pwm_commands({PWM_NAME: command}, unset_to_zero=True)
            phase = "sweep_settle" if time.monotonic() < settle_until else "sweep"
            self.write_sample(
                writer,
                phase=phase,
                pass_index=pass_index,
                amplitude=amplitude,
                direction=direction,
                command=command,
            )

            angle = self.samples[-1].bucket_deg
            progress_remaining = float(direction) * (target_angle_deg - angle)
            if progress_remaining <= 0.0:
                stop_reason = "target_angle"
                break

            elapsed = time.monotonic() - start_t
            progress_vel = float(direction) * self.last_vel_deg_s
            if elapsed >= self.args.min_sweep_s and time.monotonic() >= settle_until:
                if progress_vel <= self.args.stall_vel_deg_s:
                    if stall_since is None:
                        stall_since = time.monotonic()
                    elif time.monotonic() - stall_since >= self.args.stall_time_s:
                        stop_reason = "stall"
                        break
                else:
                    stall_since = None
            time.sleep(1.0 / self.args.rate)

        self.hardware.reset(reset_pump=False)
        self.write_sample(
            writer,
            phase=f"sweep_stop:{stop_reason}",
            pass_index=pass_index,
            amplitude=amplitude,
            direction=direction,
            command=0.0,
        )

    def run(self) -> tuple[Path, Path]:
        out_dir = ROOT_DIR / self.args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"scoop_velocity_{timestamp}.csv"
        json_path = out_dir / f"scoop_velocity_{timestamp}.json"

        self.wait_ready()
        center = _read_bucket_angle_deg(self.hardware, self.robot_config)
        if self.args.min_angle_deg is None or self.args.max_angle_deg is None:
            half = 0.5 * self.args.window_deg
            min_angle = center - half
            max_angle = center + half
        else:
            min_angle = float(self.args.min_angle_deg)
            max_angle = float(self.args.max_angle_deg)
        if min_angle >= max_angle:
            raise RuntimeError("min angle must be smaller than max angle")
        self.log.info(
            "Scoop test window: %.2f .. %.2f deg (start %.2f deg)",
            min_angle, max_angle, center,
        )

        fields = list(Sample.__dataclass_fields__)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            self.hold_zero(writer, self.args.pause_s)
            pass_index = 0
            for amplitude in self.args.commands:
                for repeat in range(self.args.passes):
                    for direction, target in ((1, max_angle), (-1, min_angle)):
                        self.log.info(
                            "amp=%.3f pass=%d direction=%+d target=%.2f deg",
                            amplitude, repeat + 1, direction, target,
                        )
                        self.sweep(
                            writer,
                            pass_index=pass_index,
                            amplitude=amplitude,
                            direction=direction,
                            target_angle_deg=target,
                        )
                        pass_index += 1
                        self.hold_zero(writer, self.args.pause_s)

        summaries = summarize_sweeps(
            self.samples,
            settle_ignore_s=self.args.settle_ignore_s,
            min_vel_deg_s=self.args.min_usable_vel_deg_s,
            max_ripple_ratio=self.args.max_ripple_ratio,
        )
        result = {
            "joint": JOINT_NAME,
            "pwm": PWM_NAME,
            "angle_window_deg": [min_angle, max_angle],
            "commands": self.args.commands,
            "summaries": [asdict(s) for s in summaries],
            "recommended_speed_bins": recommend_speed_bins(
                summaries,
                bins_per_direction=self.args.speed_bins,
            ),
        }
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scoop-only valve velocity mapper")
    parser.add_argument("--i-understand", action="store_true", help="Required safety acknowledgement")
    parser.add_argument("--config-file", default="configuration_files/profiles/rpi/servo_config.yaml")
    parser.add_argument("--control-config-file", default="configuration_files/profiles/rpi/control_config.yaml")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--commands", type=_parse_command_ladder, default=_parse_command_ladder(DEFAULT_COMMANDS))
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--window-deg", type=float, default=24.0, help="Relative test window around starting bucket angle")
    parser.add_argument("--min-angle-deg", type=float, default=None, help="Absolute lower bucket angle limit")
    parser.add_argument("--max-angle-deg", type=float, default=None, help="Absolute upper bucket angle limit")
    parser.add_argument("--bucket-command-sign", type=float, default=1.0, help="Set -1 if positive command decreases bucket angle")
    parser.add_argument("--pump-auto", action="store_true", help="Use activity-based auto pump instead of fixed pump")
    parser.add_argument("--enable-pressure", action="store_true", help="Enable ADC pressure logging")
    parser.add_argument("--pressure-hz", type=float, default=20.0)
    parser.add_argument("--pause-s", type=float, default=0.5)
    parser.add_argument("--settle-ignore-s", type=float, default=0.45)
    parser.add_argument("--min-sweep-s", type=float, default=0.5)
    parser.add_argument("--sweep-timeout-s", type=float, default=5.0)
    parser.add_argument("--stall-vel-deg-s", type=float, default=0.5)
    parser.add_argument("--stall-time-s", type=float, default=0.45)
    parser.add_argument("--min-usable-vel-deg-s", type=float, default=1.0)
    parser.add_argument("--max-ripple-ratio", type=float, default=0.45)
    parser.add_argument("--speed-bins", type=int, default=5)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    if not args.i_understand:
        parser.error("Refusing to move hardware without --i-understand")
    if args.passes <= 0:
        parser.error("--passes must be positive")
    if args.rate <= 0.0:
        parser.error("--rate must be positive")
    if args.window_deg <= 0.0:
        parser.error("--window-deg must be positive")
    if args.min_angle_deg is not None and args.max_angle_deg is None:
        parser.error("--min-angle-deg requires --max-angle-deg")
    if args.max_angle_deg is not None and args.min_angle_deg is None:
        parser.error("--max-angle-deg requires --min-angle-deg")
    if args.bucket_command_sign == 0.0:
        parser.error("--bucket-command-sign must be nonzero")
    args.bucket_command_sign = 1.0 if args.bucket_command_sign > 0.0 else -1.0
    if args.speed_bins <= 0:
        parser.error("--speed-bins must be positive")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(name)s: %(message)s")
    tester = ScoopVelocityTester(args)
    try:
        csv_path, json_path = tester.run()
        logging.getLogger("scoop_velocity_tester").info("Saved %s", csv_path)
        logging.getLogger("scoop_velocity_tester").info("Saved %s", json_path)
    finally:
        tester.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
