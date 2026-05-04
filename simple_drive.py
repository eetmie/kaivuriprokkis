#!/usr/bin/env python3
"""Bench driving helper — same UDP/gamepad input as drive_logger.py, no recording.

Usage:
    sudo python simple_drive.py
    sudo python simple_drive.py --linkage-rate-correction

Buttons (remote gamepad, same wire format as drive_logger.py):
    Button 0: toggle live mounting-corrected IMU and joint angle print
    Button 1: reload PWM/servo config
    Button 2: toggle hydraulic pump
    Button 3: cycle compensation mode — OFF → raw summary → universal smooth

Left paddle (bench mode): slowly trims pump auto-mode activity_gain_us base level while held.
    Push forward = more gain, pull back = less. Prints current base value when it changes.
    In mode 2 (pump gain), the base is further modulated each tick by the universal linkage
    shape so slow positions get a boost and fast positions are reduced — valve commands stay
    completely untouched. Also drives left track as normal — fine on the bench.

The controller runs in DIRECT mode — joystick axes go straight to valves,
IK/PID are bypassed. Useful for sanity-checking joint readouts (e.g., after
removing the slew joint limit) without touching the data-logger codepath.
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.udp_socket import UDPSocket
from modules.hardware_interface import HardwareInterface, HardwareFaultError
from modules.excavator_controller import ExcavatorController
from tools.linkage_rate_compensation import (
    DEFAULT_TABLE,
    DEFAULT_UNIVERSAL_TABLE,
    LinkageRateCompensator,
    UniversalShapeCompensator,
)


SAMPLING_FREQUENCY = 100              # main loop Hz
PRINT_DECIMATION = 10                 # print every Nth iteration when enabled (~10Hz)
PUMP_GAIN_ADJUST_RATE = 80.0         # µs/s per unit of paddle input for bench gain trim
PUMP_GAIN_PRINT_THRESHOLD = 5.0      # µs change before printing updated gain
CONTROL_JOINT_NAMES = ['slew', 'lift', 'arm', 'bucket']
IMU_ROLE_ORDER = ['base', 'boom', 'arm', 'bucket']


def euler_pry_deg_from_quat(quat) -> tuple[float, float, float]:
    """Euler pitch/roll/yaw from a [w, x, y, z] quaternion, in degrees."""
    q = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q = q / norm
    w, x, y, z = q

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    if abs(sin_pitch) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sin_pitch)
    else:
        pitch = np.arcsin(sin_pitch)

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(sin_yaw, cos_yaw)

    return tuple(float(v) for v in np.degrees([pitch, roll, yaw]))


def get_control_joint_names(controller) -> list[str]:
    """Return control-stack joint labels in get_joint_angles() output order."""
    names = list(CONTROL_JOINT_NAMES)
    chain = getattr(getattr(controller, 'robot_config', None), 'imu_chain', None) or []
    for item in chain:
        if not isinstance(item, dict) or 'output_index' not in item:
            continue
        output_index = int(item['output_index'])
        if 0 <= output_index < len(names) and item.get('joint'):
            names[output_index] = str(item['joint'])
    return names


def get_imu_role_order(controller) -> list[str]:
    """Return configured IMU role order for link-ordered debug output."""
    robot_config = getattr(controller, 'robot_config', None)
    roles = getattr(robot_config, 'imu_sensor_roles', None) if robot_config is not None else None
    return list(roles) if roles else list(IMU_ROLE_ORDER)


def format_imu_debug_line(payload, joint_angles_deg=None, joint_names=None, imu_role_order=None) -> str:
    """Format mounting-corrected IMU Euler angles and controller joint angles."""
    if not payload:
        return "[IMU] waiting for raw IMU data"

    corrected_quats = payload.get('corrected_quats') or []
    role_by_index = payload.get('role_by_index') or {}
    index_by_role = {role: idx for idx, role in role_by_index.items()}
    descriptors = payload.get('descriptors') or []

    parts = []
    ordered_indices = []
    for role in imu_role_order or IMU_ROLE_ORDER:
        idx = index_by_role.get(role)
        if idx is not None and idx < len(corrected_quats) and idx not in ordered_indices:
            ordered_indices.append(idx)
    ordered_indices.extend(i for i in range(len(corrected_quats)) if i not in ordered_indices)

    for i in ordered_indices:
        corr_q = corrected_quats[i]
        role = role_by_index.get(i, '-')
        label = descriptors[i].get('label', '') if i < len(descriptors) else ''
        label_text = f" {label}" if label else ""
        pitch, roll, yaw = euler_pry_deg_from_quat(corr_q)
        parts.append(
            f"imu{i}({role}{label_text}) P/R/Y={pitch:+7.2f}/{roll:+7.2f}/{yaw:+7.2f}"
        )

    joint_text = "joints waiting"
    if joint_angles_deg is not None:
        joint_names = joint_names or CONTROL_JOINT_NAMES
        joint_text = " ".join(
            f"{joint_names[i] if i < len(joint_names) else f'j{i}'}={float(angle):+7.2f}"
            for i, angle in enumerate(joint_angles_deg)
        )
    return (
        "[IMU mount-corrected P/R/Y] "
        + " | ".join(parts)
        + f" deg || [control joints] {joint_text} deg"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Bench driving helper with optional linkage-rate correction")
    parser.add_argument(
        "--linkage-rate-correction",
        action="store_true",
        help="Start with prototype linkage-rate command correction enabled",
    )
    parser.add_argument(
        "--linkage-rate-table",
        default=str(DEFAULT_TABLE),
        help="Path to processed linkage-rate *_summary.csv",
    )
    parser.add_argument(
        "--linkage-rate-universal-table",
        default=str(DEFAULT_UNIVERSAL_TABLE),
        help="Path to processed linkage-rate *_universal_shape.csv",
    )
    parser.add_argument("--linkage-rate-min-factor", type=float, default=0.35)
    parser.add_argument("--linkage-rate-max-factor", type=float, default=2.25)
    return parser.parse_args()


def main():
    args = parse_args()
    # ---- UDP (mirror drive_logger.py wire format exactly) ----
    server = UDPSocket(local_id=2)
    server.setup("192.168.0.132", 8080, num_inputs=10, num_outputs=0, is_server=True)

    # ---- Hardware ----
    print("Initializing hardware...")
    hardware = HardwareInterface(
        pump_auto_mode=True,
        toggle_channels=True,
        stale_timeout_s=0.5,
        enable_pwm=True,
        enable_imu=True,
        enable_adc=False,           # not needed for plain driving
        cleanup_disable_osc=False,
    )

    print("Waiting for hardware to be ready...")
    try:
        while not hardware.is_hardware_ready():
            time.sleep(0.1)
        print("Hardware ready.")
    except HardwareFaultError as e:
        print(f"\n*** HARDWARE FAULT ({e.subsystem}): {e.reason} ***")
        hardware.shutdown()
        raise SystemExit(1)

    # ---- Controller (direct mode, IK/PID bypassed) ----
    print("Starting controller...")
    controller = ExcavatorController(hardware, config=None, enable_perf_tracking=False)
    controller.start()
    time.sleep(2.0)                 # numba JIT warmup
    controller.enter_direct_mode()
    control_joint_names = get_control_joint_names(controller)
    imu_role_order = get_imu_role_order(controller)
    print("Controller in DIRECT mode.")

    linkage_compensator = None
    universal_compensator = None
    # comp_mode: 0=OFF, 1=raw summary, 2=universal smooth
    comp_mode = 1 if args.linkage_rate_correction else 0
    COMP_LABELS = ["OFF", "valve scale (raw)", "pump gain (universal shape)"]

    try:
        linkage_compensator = LinkageRateCompensator(
            args.linkage_rate_table,
            min_factor=args.linkage_rate_min_factor,
            max_factor=args.linkage_rate_max_factor,
        )
        print(f"Raw linkage-rate table loaded: {linkage_compensator.table_path}")
    except Exception as e:
        print(f"Raw linkage-rate correction unavailable: {e}")

    try:
        universal_compensator = UniversalShapeCompensator(
            args.linkage_rate_universal_table,
            min_factor=args.linkage_rate_min_factor,
            max_factor=args.linkage_rate_max_factor,
        )
        print(f"Universal shape table loaded: {universal_compensator.table_path}")
    except Exception as e:
        print(f"Universal shape correction unavailable: {e}")

    print(f"Compensation mode: {COMP_LABELS[comp_mode]}")

    # ---- Pump gain trim state ----
    pwm = hardware.pwm_controller
    pump_gain_available = pwm is not None and pwm.pump_config is not None and pwm.pump_auto_mode
    pump_gain_us = pwm.get_pump_activity_gain_us() if pump_gain_available else 0.0
    pump_gain_last_printed = pump_gain_us
    if pump_gain_available:
        print(f"Pump auto gain trim enabled (left paddle). Initial activity_gain_us={pump_gain_us:.1f} µs")
    else:
        print("Pump gain trim unavailable (no pump config or auto mode off).")

    # ---- UDP handshake ----
    print("Waiting for remote controller...")
    if not server.handshake(timeout=30.0):
        print("UDP handshake failed.")
        controller.exit_direct_mode()
        controller.stop()
        hardware.shutdown()
        raise SystemExit(1)
    server.start_receiving()
    print(
        "Connected. Drive with the gamepad. "
        "Button 0 prints IMU/joint angles, Button 1 reloads config, "
        "Button 3 cycles compensation mode (OFF / raw / universal smooth).\n"
    )

    # ---- Loop state ----
    loop_period = 1.0 / SAMPLING_FREQUENCY
    next_run_time = time.perf_counter()

    right_rl = right_ud = left_rl = left_ud = 0.0
    right_paddle = left_paddle = 0.0
    button_prev = [0.0, 0.0, 0.0, 0.0]
    button_threshold = 0.5

    print_angles = False
    iter_count = 0

    try:
        while True:
            float_data = server.get_latest_floats()
            if float_data:
                right_rl = float_data[9]   # scoop
                right_ud = float_data[8]   # lift
                left_rl = float_data[7]    # rotate (slew)
                left_ud = float_data[6]    # tilt
                right_paddle = float_data[5]
                left_paddle = float_data[4]
                buttons = [float_data[0], float_data[1], float_data[2], float_data[3]]

                # Button 0: toggle mounting-corrected IMU + relative joint angle printing.
                if buttons[0] > button_threshold and button_prev[0] <= button_threshold:
                    print_angles = not print_angles
                    state = "ON" if print_angles else "OFF"
                    print(f"\n[Button 0] IMU/joint angle print {state}")

                # Button 1: reload PWM/servo config for live valve tuning.
                if buttons[1] > button_threshold and button_prev[1] <= button_threshold:
                    ok = hardware.reload_config()
                    state = "OK" if ok else "FAILED"
                    print(f"\n[Button 1] reload config {state}")

                # Button 2: pump toggle (handy on the bench).
                if buttons[2] > button_threshold and button_prev[2] <= button_threshold:
                    if hardware.pwm_controller is not None:
                        new_state = not hardware.pwm_controller.pump_enabled
                        hardware.pwm_controller.set_pump_enabled(new_state)
                        print(f"\n[Button 2] pump {'ON' if new_state else 'OFF'}")

                # Button 3: cycle compensation mode OFF → raw → universal smooth.
                if buttons[3] > button_threshold and button_prev[3] <= button_threshold:
                    comp_mode = (comp_mode + 1) % 3
                    if comp_mode == 1 and linkage_compensator is None:
                        comp_mode = (comp_mode + 1) % 3
                    if comp_mode == 2 and universal_compensator is None:
                        comp_mode = (comp_mode + 1) % 3
                    print(f"\n[Button 3] compensation mode: {COMP_LABELS[comp_mode]}")

                button_prev = buttons

            # Direct-mode commands straight from the joystick.
            commands = {
                'rotate':    left_rl,
                'lift_boom': right_ud,
                'tilt_boom': left_ud,
                'scoop':     right_rl,
                'trackR':    right_paddle,
                'trackL':    left_paddle,
            }
            # TODO: replace open-loop shape compensation with closed-loop joint velocity
            # control. Joystick → desired deg/s → PID → valve command, with gyro angular
            # rates as feedback. IMU gyros give rate directly (no differentiation noise),
            # but will need a low-pass (e.g. exponential smoothing) to suppress vibration
            # before feeding the PID error. Would make feel consistent across full range
            # regardless of linkage geometry, pressure, or temperature drift.
            # NOTE: valve response is slow, so PID needs oscillation prevention —
            # derivative damping and/or a command deadband are likely necessary to avoid
            # chatter at steady state while waiting for pressure to build.
            joint_angles = controller.get_joint_angles()

            # Mode 1: scale valve commands per-joint (valve curves are affected).
            # Mode 2: valve commands untouched — correction goes to pump gain only.
            if comp_mode == 1 and linkage_compensator is not None:
                commands = linkage_compensator.apply(commands, joint_angles)
            controller.give_direct_commands(commands)

            # Left paddle: trim the pump activity_gain_us base level.
            # In mode 2 this base is then further modulated by linkage position below.
            if pump_gain_available and abs(left_paddle) > 0.05:
                raw = pump_gain_us + left_paddle * PUMP_GAIN_ADJUST_RATE * loop_period
                pump_gain_us = float(np.clip(raw, 0.0, pwm.pump_config.pulse_max - pwm.pump_config.pulse_min))
                if abs(pump_gain_us - pump_gain_last_printed) >= PUMP_GAIN_PRINT_THRESHOLD:
                    print(f"\n[pump gain base] activity_gain_us={pump_gain_us:.1f} µs")
                    pump_gain_last_printed = pump_gain_us

            # Apply pump gain: base level, optionally modulated by linkage shape (mode 2).
            if pump_gain_available:
                if comp_mode == 2 and universal_compensator is not None:
                    factor = universal_compensator.pump_correction_factor(commands, joint_angles)
                else:
                    factor = 1.0
                pwm.set_pump_activity_gain_us(pump_gain_us * factor)

            # Live IMU readout — only when toggled on, decimated to ~10Hz.
            iter_count += 1
            if print_angles and (iter_count % PRINT_DECIMATION == 0):
                line = format_imu_debug_line(
                    hardware.read_imu_debug_quaternions(),
                    controller.get_joint_angles(),
                    control_joint_names,
                    imu_role_order,
                )
                # \r keeps the readout on a single line; the toggle prints
                # above use a leading newline so they don't get overwritten.
                print(line, end="\r", flush=True)

            # Tight 100Hz pacing.
            next_run_time += loop_period
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\n\nInterrupted (Ctrl+C).")
    finally:
        print("Shutting down...")
        try:
            controller.give_direct_commands({})       # zero outputs
            time.sleep(0.2)
            controller.exit_direct_mode()
            controller.stop()
        except Exception:
            pass
        try:
            hardware.reset(reset_pump=True)
            hardware.shutdown()
        except Exception:
            pass
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
