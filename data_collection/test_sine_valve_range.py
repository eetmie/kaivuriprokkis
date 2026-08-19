#!/usr/bin/env python3
"""Simple IMU-based sine speed sweep for one hydraulic joint at a time.

MANUAL HARDWARE TEST — requires real hardware. Not a unit test.

The script captures the current joint angle as the center, then tries to move
that joint around ``center +/- target_swing_deg``. A simple direct-mode valve
controller tracks a sine target while the sine frequency is increased step by
step until the IMU reports basically no movement.

Results are saved as a plain text file.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simple_drive import JOINT_NAMES


RESULTS_DIR = Path(__file__).parent / "test_results"

# controller.get_joint_angles() returns ([slew, boom, arm, bucket], state_ts, imu_us)
JOINT_TO_IMU_IDX = {
    "slew": 0,
    "boom": 1,
    "arm": 2,
    "bucket": 3,
}

DEFAULT_JOINTS = ["boom", "arm", "bucket"]
DEFAULT_TARGET_SWING_DEG = 20.0
DEFAULT_START_FREQ_HZ = 0.10
DEFAULT_FREQ_STEP_HZ = 0.05
DEFAULT_MAX_FREQ_HZ = 2.00
DEFAULT_MOVEMENT_THRESHOLD_DEG = 3.0
DEFAULT_CONTROL_HZ = 100.0
DEFAULT_KP = 0.06
DEFAULT_KD = 0.002
DEFAULT_MIN_COMMAND = 0.08
DEFAULT_MAX_COMMAND = 1.0
DEFAULT_SETTLE_S = 1.0
MIN_TRACKING_ERROR_FOR_COMMAND = 1.0


def make_zero_commands() -> dict[str, float]:
    return {name: 0.0 for name in JOINT_NAMES}


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sign_or_zero(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def get_joint_angle_deg(controller, joint_name: str) -> float:
    angles, _, _ = controller.get_joint_angles()
    return float(angles[JOINT_TO_IMU_IDX[joint_name]])


def wait_for_hardware_ready(hardware, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    while not hardware.is_hardware_ready():
        if time.time() >= deadline:
            raise TimeoutError(f"Hardware not ready within {timeout_s:.0f}s")
        time.sleep(0.1)


def settle_joint(direct, settle_s: float) -> None:
    direct.give_commands(make_zero_commands())
    direct.send_pending()
    time.sleep(settle_s)


def format_step_line(result: dict[str, float | bool]) -> str:
    status = "INTERRUPTED" if result["interrupted"] else (
        "STOP" if result["movement_detected"] is False else "OK"
    )
    return (
        f"  freq={result['freq_hz']:.2f} Hz | duration={result['duration_s']:.1f}s | "
        f"angle_range={result['angle_range_deg']:.2f} deg | "
        f"max_cmd={result['max_abs_command']:.3f} | {status}"
    )


def run_frequency_step(
    controller,
    direct,
    joint_name: str,
    center_deg: float,
    freq_hz: float,
    target_swing_deg: float,
    movement_threshold_deg: float,
    control_hz: float,
    kp: float,
    kd: float,
    min_command: float,
    max_command: float,
) -> dict[str, float | bool]:
    loop_period = 1.0 / control_hz
    duration_s = min(max(2.0 / freq_hz, 6.0), 15.0)
    angles: list[float] = []
    commands: list[float] = []
    last_error_deg = 0.0
    interrupted = False

    start = time.perf_counter()
    next_run = start

    try:
        while (time.perf_counter() - start) < duration_s:
            elapsed = time.perf_counter() - start
            target_angle_deg = center_deg + target_swing_deg * math.sin(2.0 * math.pi * freq_hz * elapsed)
            current_angle_deg = get_joint_angle_deg(controller, joint_name)
            error_deg = target_angle_deg - current_angle_deg
            error_rate_deg_s = (error_deg - last_error_deg) / loop_period
            command = kp * error_deg + kd * error_rate_deg_s

            if abs(error_deg) >= MIN_TRACKING_ERROR_FOR_COMMAND and 0.0 < abs(command) < min_command:
                command = sign_or_zero(command) * min_command

            command = clip(command, -max_command, max_command)
            cmd_dict = make_zero_commands()
            cmd_dict[joint_name] = command
            direct.give_commands(cmd_dict)
            direct.send_pending()

            angles.append(current_angle_deg)
            commands.append(command)
            last_error_deg = error_deg

            next_run += loop_period
            sleep_time = next_run - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_run = time.perf_counter()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        direct.give_commands(make_zero_commands())
        direct.send_pending()

    if not angles:
        raise RuntimeError(f"No IMU samples collected for {joint_name} at {freq_hz:.2f} Hz")

    angle_min = min(angles)
    angle_max = max(angles)
    angle_range_deg = angle_max - angle_min
    max_abs_command = max(abs(cmd) for cmd in commands) if commands else 0.0

    return {
        "freq_hz": freq_hz,
        "duration_s": time.perf_counter() - start,
        "angle_min_deg": angle_min,
        "angle_max_deg": angle_max,
        "angle_range_deg": angle_range_deg,
        "max_abs_command": max_abs_command,
        "movement_detected": angle_range_deg >= movement_threshold_deg,
        "interrupted": interrupted,
    }


def run_joint_sweep(
    controller,
    direct,
    joint_name: str,
    target_swing_deg: float,
    start_freq_hz: float,
    freq_step_hz: float,
    max_freq_hz: float,
    movement_threshold_deg: float,
    control_hz: float,
    kp: float,
    kd: float,
    min_command: float,
    max_command: float,
    settle_s: float,
) -> tuple[list[str], bool]:
    settle_joint(direct, settle_s)
    center_deg = get_joint_angle_deg(controller, joint_name)

    lines = []
    lines.append("")
    lines.append(f"Joint: {joint_name}")
    lines.append(f"  Center angle at start: {center_deg:+.2f} deg")
    lines.append(f"  Target motion: center +/- {target_swing_deg:.1f} deg")
    lines.append(f"  Movement threshold: {movement_threshold_deg:.1f} deg peak-to-peak")
    lines.append(f"  Sweep: {start_freq_hz:.2f} Hz -> {max_freq_hz:.2f} Hz in {freq_step_hz:.2f} Hz steps")
    lines.append("")

    freq_hz = start_freq_hz
    largest_command_overall = 0.0
    largest_command_at_stop = 0.0
    stop_reason = "reached_max_frequency"
    stop_freq_hz = max_freq_hz
    interrupted = False

    while freq_hz <= max_freq_hz + 1e-9:
        result = run_frequency_step(
            controller=controller,
            direct=direct,
            joint_name=joint_name,
            center_deg=center_deg,
            freq_hz=freq_hz,
            target_swing_deg=target_swing_deg,
            movement_threshold_deg=movement_threshold_deg,
            control_hz=control_hz,
            kp=kp,
            kd=kd,
            min_command=min_command,
            max_command=max_command,
        )

        largest_command_overall = max(largest_command_overall, float(result["max_abs_command"]))
        print(format_step_line(result))
        lines.append(format_step_line(result))

        if result["interrupted"]:
            stop_reason = "keyboard_interrupt"
            stop_freq_hz = freq_hz
            largest_command_at_stop = float(result["max_abs_command"])
            interrupted = True
            break

        if result["movement_detected"] is False:
            stop_reason = "imu_movement_below_threshold"
            stop_freq_hz = freq_hz
            largest_command_at_stop = float(result["max_abs_command"])
            break

        largest_command_at_stop = float(result["max_abs_command"])
        freq_hz += freq_step_hz

    lines.append("")
    lines.append("Summary:")
    lines.append(f"  stop_reason: {stop_reason}")
    lines.append(f"  stop_freq_hz: {stop_freq_hz:.2f}")
    lines.append(f"  largest_command_at_stop: {largest_command_at_stop:.3f}")
    lines.append(f"  largest_command_overall: {largest_command_overall:.3f}")
    lines.append("")

    return lines, interrupted


def run_hardware_test(args) -> list[str]:
    from modules.direct_controller import DirectController
    from modules.excavator_controller import ExcavatorController
    from modules.hardware_interface import HardwareInterface

    lines = []
    lines.append("=" * 70)
    lines.append("SIMPLE SINE SPEED SWEEP")
    lines.append("=" * 70)
    lines.append(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"joints: {', '.join(args.joints)}")
    lines.append(f"target_swing_deg: {args.target_swing_deg:.1f}")
    lines.append(f"start_freq_hz: {args.start_freq_hz:.2f}")
    lines.append(f"freq_step_hz: {args.freq_step_hz:.2f}")
    lines.append(f"max_freq_hz: {args.max_freq_hz:.2f}")
    lines.append(f"movement_threshold_deg: {args.movement_threshold_deg:.1f}")
    lines.append(f"control_hz: {args.control_hz:.1f}")
    lines.append(f"kp: {args.kp:.3f}")
    lines.append(f"kd: {args.kd:.4f}")
    lines.append(f"min_command: {args.min_command:.3f}")
    lines.append("")

    hardware = None
    controller = None
    direct = None

    try:
        print("[hw] Initializing hardware...")
        hardware = HardwareInterface(
            config_file="configuration_files/profiles/rpi/servo_config.yaml",
            control_config_file="configuration_files/profiles/rpi/control_config.yaml",
            pump_auto_mode=False,
            toggle_channels=True,
            stale_timeout_s=0.5,
            adc_channels=[
                "LiftBoom retract ps", "LiftBoom extend ps",
                "TiltBoom retract ps", "TiltBoom extend ps",
                "Scoop extend ps", "Scoop retract ps", "Pump ps",
            ],
            adc_sample_hz=20,
            enable_pwm=True,
            enable_imu=True,
            enable_adc=False,
            cleanup_disable_osc=False,
            log_level="WARNING",
        )
        wait_for_hardware_ready(hardware)
        print("[hw] Hardware ready")

        controller = ExcavatorController(
            hardware,
            config=None,
            enable_perf_tracking=False,
            log_level="WARNING",
            control_config_file="configuration_files/profiles/rpi/control_config.yaml",
        )
        controller.start()
        time.sleep(2.0)
        direct = DirectController(hardware)
        controller.suspend_ik_output()
        print("[hw] Controller ready; IK output suspended (DirectController owns bus)")

        for joint_name in args.joints:
            print(f"\n[joint] {joint_name}")
            joint_lines, interrupted = run_joint_sweep(
                controller=controller,
                direct=direct,
                joint_name=joint_name,
                target_swing_deg=args.target_swing_deg,
                start_freq_hz=args.start_freq_hz,
                freq_step_hz=args.freq_step_hz,
                max_freq_hz=args.max_freq_hz,
                movement_threshold_deg=args.movement_threshold_deg,
                control_hz=args.control_hz,
                kp=args.kp,
                kd=args.kd,
                min_command=args.min_command,
                max_command=DEFAULT_MAX_COMMAND,
                settle_s=args.settle_s,
            )
            lines.extend(joint_lines)
            if interrupted:
                break

    finally:
        if controller is not None:
            try:
                if direct is not None:
                    direct.give_commands(make_zero_commands())
                    direct.send_pending()
                time.sleep(0.3)
                controller.resume_ik_output()
                controller.stop()
            except Exception:
                pass
        elif hardware is not None:
            try:
                hardware.shutdown()
            except Exception:
                pass

    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple IMU-based sine speed sweep.")
    parser.add_argument(
        "--joint",
        dest="joints",
        action="append",
        choices=JOINT_NAMES,
        help="Joint to test. Repeat to test multiple joints. Default: boom, arm, bucket",
    )
    parser.add_argument("--target-swing-deg", type=float, default=DEFAULT_TARGET_SWING_DEG)
    parser.add_argument("--start-freq-hz", type=float, default=DEFAULT_START_FREQ_HZ)
    parser.add_argument("--freq-step-hz", type=float, default=DEFAULT_FREQ_STEP_HZ)
    parser.add_argument("--max-freq-hz", type=float, default=DEFAULT_MAX_FREQ_HZ)
    parser.add_argument("--movement-threshold-deg", type=float, default=DEFAULT_MOVEMENT_THRESHOLD_DEG)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--kp", type=float, default=DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=DEFAULT_KD)
    parser.add_argument("--min-command", type=float, default=DEFAULT_MIN_COMMAND)
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    args = parser.parse_args()

    if args.joints is None:
        args.joints = list(DEFAULT_JOINTS)
    if args.target_swing_deg <= 0.0:
        parser.error("--target-swing-deg must be > 0")
    if args.start_freq_hz <= 0.0:
        parser.error("--start-freq-hz must be > 0")
    if args.freq_step_hz <= 0.0:
        parser.error("--freq-step-hz must be > 0")
    if args.max_freq_hz < args.start_freq_hz:
        parser.error("--max-freq-hz must be >= --start-freq-hz")
    if args.movement_threshold_deg <= 0.0:
        parser.error("--movement-threshold-deg must be > 0")
    if args.control_hz <= 0.0:
        parser.error("--control-hz must be > 0")
    if args.min_command < 0.0:
        parser.error("--min-command must be >= 0")
    if args.settle_s < 0.0:
        parser.error("--settle-s must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = run_hardware_test(args)

    results_path = RESULTS_DIR / f"sine_valve_range_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
