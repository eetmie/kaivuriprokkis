#!/usr/bin/env python3
"""Auto logger for linkage-rate compensation data.

This script is intentionally separate from excv_gui.py. It drives one joint at a
time with fixed open-loop valve commands, logs IMU-derived joint angles and
finite-difference velocities, then shuttles that joint end-to-end between both
stops for multiple passes and command amplitudes.

Typical use on the robot:

  python tools/linkage_rate_logger.py --i-understand

Each tested joint is first homed to one end stop at full command so the data
passes start from a known boundary. It then sweeps across the full available
range in both directions using the fixed amplitude ladder ``0.2, 0.4, 0.6, 0.8,
1.0``.

Sweeps stop when the selected joint no longer advances, i.e. hydraulic end stop
or mechanical stall is reached. ``--sweep-timeout-s`` is only a fault guard.

The fixed-command ``sweep`` rows are the useful identification data. Exclude
``seek_endstop``, ``sweep_settle``, and ``sweep_stop`` rows when building the
rate table.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.ik import load_excavator_robot_config, canonical_joint_angles_from_imus
from modules.hardware_interface import HardwareInterface


JOINTS = {
    "boom": {"index": 1, "pwm": "lift_boom"},
    "arm": {"index": 2, "pwm": "tilt_boom"},
    "bucket": {"index": 3, "pwm": "scoop"},
}
DEFAULT_ORDER = ["bucket", "arm", "boom"]
COMMAND_AMPLITUDES = [0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_OUTPUT_DIR = "data_collection/hydraulic_data"
ADVANCE_EPS_DEG = 0.35


@dataclass(frozen=True)
class JointPlan:
    name: str
    index: int
    pwm_name: str
    command_sign: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


class LinkageRateLogger:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.log = logging.getLogger("linkage_rate_logger")
        self.robot_config = load_excavator_robot_config("configuration_files/profiles/rpi/control_config.yaml")
        self.hardware = HardwareInterface(
            config_file=args.config_file,
            control_config_file="configuration_files/profiles/rpi/control_config.yaml",
            # Intentional: linkage-rate logging needs activity-correlated pump
            # behaviour, so this tool stays on auto pump even though the
            # project default is fixed pump everywhere else.
            pump_auto_mode=True,
            cleanup_disable_osc=False,
            enable_adc=False,
            start_adc_reader=False,
            log_level=args.log_level,
        )
        self.prev_angles: np.ndarray | None = None
        self.prev_time: float | None = None
        self.start_time = time.monotonic()
        self.last_angles = np.zeros(4, dtype=np.float32)
        self.last_velocities = np.zeros(4, dtype=np.float32)

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

    def read_state(self) -> tuple[np.ndarray, np.ndarray]:
        quats = self.hardware.read_all_imu_quaternions()
        if quats is None:
            raise RuntimeError("IMU quaternions unavailable")
        angles = canonical_joint_angles_from_imus(np.asarray(quats, dtype=np.float32), self.robot_config)
        now = time.monotonic()
        if self.prev_angles is None or self.prev_time is None:
            velocities = np.zeros(4, dtype=np.float32)
        else:
            dt = max(1e-6, now - self.prev_time)
            delta = np.arctan2(np.sin(angles - self.prev_angles), np.cos(angles - self.prev_angles))
            velocities = delta / dt
        self.prev_angles = angles.copy()
        self.prev_time = now
        self.last_angles = np.degrees(angles).astype(np.float32)
        self.last_velocities = np.degrees(velocities).astype(np.float32)
        return self.last_angles.copy(), self.last_velocities.copy()

    def write_row(
        self,
        writer: csv.DictWriter,
        *,
        phase: str,
        joint: str,
        command: float,
        pass_index: int,
        amplitude: float,
        direction: int,
        target_endpoint_deg: float,
        stop_reason: str = "",
    ) -> None:
        angles, velocities = self.read_state()
        row = {
            "t_s": f"{time.monotonic() - self.start_time:.6f}",
            "phase": phase,
            "joint": joint,
            "joint_index": JOINTS[joint]["index"],
            "command": f"{command:.6f}",
            "pass_index": pass_index,
            "amplitude": f"{amplitude:.6f}",
            "direction": direction,
            "target_endpoint_deg": f"{target_endpoint_deg:.6f}",
            "stop_reason": stop_reason,
            "slew_deg": f"{angles[0]:.6f}",
            "boom_deg": f"{angles[1]:.6f}",
            "arm_deg": f"{angles[2]:.6f}",
            "bucket_deg": f"{angles[3]:.6f}",
            "slew_vel_deg_s": f"{velocities[0]:.6f}",
            "boom_vel_deg_s": f"{velocities[1]:.6f}",
            "arm_vel_deg_s": f"{velocities[2]:.6f}",
            "bucket_vel_deg_s": f"{velocities[3]:.6f}",
        }
        writer.writerow(row)

    def _hold_zero(self, duration_s: float) -> None:
        end_t = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < end_t:
            self.hardware.reset(reset_pump=False)
            self.read_state()
            time.sleep(1.0 / self.args.rate)

    def seek_endstop(
        self,
        plan: JointPlan,
        writer: csv.DictWriter,
        *,
        direction: int,
        command_scale: float = 1.0,
    ) -> None:
        command = plan.command_sign * int(np.sign(direction)) * _clamp(command_scale, 0.05, 1.0)
        self.log.info("Seeking %s end stop direction=%+d command=%.2f", plan.name, direction, command)
        self.sweep_until_stall(
            plan,
            writer,
            command=command,
            pass_index=-1,
            amplitude=abs(command),
            direction=int(np.sign(direction)),
            phase_sweep="seek_endstop",
            phase_settle="seek_endstop",
        )

    def sweep_until_stall(
        self,
        plan: JointPlan,
        writer: csv.DictWriter,
        *,
        command: float,
        pass_index: int,
        amplitude: float,
        direction: int,
        phase_sweep: str = "sweep",
        phase_settle: str = "sweep_settle",
    ) -> None:
        start_t = time.monotonic()
        settle_until = start_t + self.args.settle_ignore_s
        stall_since: float | None = None
        stop_reason = "timeout"
        best_progress_deg: float | None = None
        while time.monotonic() - start_t < self.args.sweep_timeout_s:
            self.hardware.send_named_pwm_commands({plan.pwm_name: command}, unset_to_zero=True)
            phase = phase_settle if time.monotonic() < settle_until else phase_sweep
            self.write_row(
                writer,
                phase=phase,
                joint=plan.name,
                command=command,
                pass_index=pass_index,
                amplitude=amplitude,
                direction=direction,
                target_endpoint_deg=float("nan"),
            )
            current_angle_deg = float(self.last_angles[plan.index])
            signed_progress_deg = float(direction) * current_angle_deg
            if best_progress_deg is None or signed_progress_deg > best_progress_deg + ADVANCE_EPS_DEG:
                best_progress_deg = signed_progress_deg
                stall_since = None

            elapsed = time.monotonic() - start_t
            progress_vel = float(direction) * float(self.last_velocities[plan.index])
            if elapsed >= self.args.min_sweep_s and time.monotonic() >= settle_until:
                if best_progress_deg is None:
                    best_progress_deg = signed_progress_deg
                if progress_vel <= self.args.stall_vel_deg_s:
                    if stall_since is None:
                        stall_since = time.monotonic()
                    elif time.monotonic() - stall_since >= self.args.stall_time_s:
                        stop_reason = "stall"
                        break
                elif signed_progress_deg <= best_progress_deg + ADVANCE_EPS_DEG:
                    if stall_since is None:
                        stall_since = time.monotonic()
                    elif time.monotonic() - stall_since >= self.args.stall_time_s:
                        stop_reason = "no_progress"
                        break
                else:
                    stall_since = None
            time.sleep(1.0 / self.args.rate)
        self.hardware.reset(reset_pump=False)
        self.write_row(
            writer,
            phase="sweep_stop",
            joint=plan.name,
            command=0.0,
            pass_index=pass_index,
            amplitude=amplitude,
            direction=direction,
            target_endpoint_deg=float("nan"),
            stop_reason=stop_reason,
        )

    def run(self) -> Path:
        command_signs = {
            "boom": float(self.args.boom_command_sign),
            "arm": float(self.args.arm_command_sign),
            "bucket": float(self.args.bucket_command_sign),
        }
        order = [name.strip() for name in self.args.order.split(",") if name.strip()]
        for name in order:
            if name not in JOINTS:
                raise RuntimeError(f"Unknown joint in order: {name}")

        plans = []
        for name in order:
            sign = 1.0 if command_signs[name] >= 0.0 else -1.0
            plans.append(JointPlan(name, JOINTS[name]["index"], JOINTS[name]["pwm"], sign))

        out_dir = ROOT_DIR / self.args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"linkage_rate_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        fields = [
            "t_s", "phase", "joint", "joint_index", "command", "pass_index", "amplitude",
            "direction", "target_endpoint_deg", "stop_reason", "slew_deg", "boom_deg", "arm_deg", "bucket_deg",
            "slew_vel_deg_s", "boom_vel_deg_s", "arm_vel_deg_s", "bucket_vel_deg_s",
        ]

        self.wait_ready()
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for plan in plans:
                self.log.info("Homing %s to initial end stop", plan.name)
                self.seek_endstop(plan, writer, direction=-1, command_scale=1.0)
                self._hold_zero(self.args.pause_s)

                for amplitude in COMMAND_AMPLITUDES:
                    for pass_index in range(self.args.passes):
                        for direction in (1, -1):
                            command = plan.command_sign * direction * amplitude
                            self.log.info(
                                "%s pass=%d amp=%.2f direction=%+d until stall",
                                plan.name, pass_index + 1, amplitude, direction,
                            )
                            self.sweep_until_stall(
                                plan,
                                writer,
                                command=command,
                                pass_index=pass_index,
                                amplitude=amplitude,
                                direction=direction,
                            )
                            self._hold_zero(self.args.pause_s)
        return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto linkage-rate logger using fixed full-range valve shuttles")
    parser.add_argument("--i-understand", action="store_true", help="Required safety acknowledgement")
    parser.add_argument("--config-file", default="configuration_files/profiles/rpi/servo_config.yaml")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--boom-command-sign", type=float, default=1.0, help="Set -1 if positive boom command decreases boom angle")
    parser.add_argument("--arm-command-sign", type=float, default=1.0, help="Set -1 if positive arm command decreases arm angle")
    parser.add_argument("--bucket-command-sign", type=float, default=1.0, help="Set -1 if positive bucket command decreases bucket angle")
    parser.add_argument("--order", default=",".join(DEFAULT_ORDER), help="Comma-separated order, default bucket,arm,boom")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--sweep-timeout-s", type=float, default=25.0, help="Fault guard only; normal sweep stop is stall detection")
    parser.add_argument("--min-sweep-s", type=float, default=1.0, help="Minimum command time before stall detection can stop a sweep")
    parser.add_argument("--stall-vel-deg-s", type=float, default=0.8, help="Advancing velocity below this counts as no advancement")
    parser.add_argument("--stall-time-s", type=float, default=0.8, help="No-advancement duration required to stop sweep")
    parser.add_argument("--settle-ignore-s", type=float, default=0.35)
    parser.add_argument("--pause-s", type=float, default=0.5)
    args = parser.parse_args()

    if not args.i_understand:
        parser.error("Refusing to move hardware without --i-understand")
    if args.passes <= 0:
        parser.error("--passes must be positive")
    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.min_sweep_s < 0:
        parser.error("--min-sweep-s must be non-negative")
    if args.stall_vel_deg_s < 0:
        parser.error("--stall-vel-deg-s must be non-negative")
    if args.stall_time_s <= 0:
        parser.error("--stall-time-s must be positive")

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(name)s: %(message)s")
    logger = LinkageRateLogger(args)
    try:
        path = logger.run()
        logging.getLogger("linkage_rate_logger").info("Saved %s", path)
    finally:
        logger.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
