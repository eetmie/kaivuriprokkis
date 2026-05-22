#!/usr/bin/env python3
"""VERY MESSY TEST SCRIPT, unify the usage some day!!!!

Bench driving helper — same UDP/gamepad input as drive_logger.py, no recording.

Usage:
    sudo python simple_drive.py
    sudo python simple_drive.py --disable
    sudo python simple_drive.py --linkage-rate-correction
    sudo python simple_drive.py --robot jetson

Jetson mode uses the Jetson PCA9685 profile and the same USB IMU stack as the
Raspberry Pi profile. Use --disable-imu for PCA9685-only valve config tuning.

Buttons (remote gamepad, same wire format as drive_logger.py):
    Button 0: toggle live mounting-corrected IMU and joint angle print
    Button 1: reload PWM/servo config
    Button 2: toggle hydraulic pump
    Button 3: cycle compensation mode — OFF → raw summary → universal smooth → velocity PID

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

from modules.board import PROFILES as ROBOT_PROFILES, resolve_profile as _resolve_board_profile
from modules.udp_socket import UDPSocket
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


# Velocity PID: joystick command → desired deg/s → PI → valve command.
# Slew (rotate) has no gyro so it stays open-loop; only boom/arm/bucket are controlled.
_VEL_CTRL_JOINTS = {'lift_boom': 1, 'tilt_boom': 2, 'scoop': 3}


class JointVelocityController:
    def __init__(self, kp: float, ki: float, ki_max: float, deadband_degps: float, max_degps: float):
        self.kp = kp
        self.ki = ki
        self.ki_max = ki_max
        self.deadband = deadband_degps
        self.max_degps = max_degps
        self._integral: dict[str, float] = {name: 0.0 for name in _VEL_CTRL_JOINTS}

    def apply(self, commands: dict, joint_vels_degps, dt: float) -> dict:
        out = dict(commands)
        for name, vel_idx in _VEL_CTRL_JOINTS.items():
            if name not in out or vel_idx >= len(joint_vels_degps):
                continue
            desired = float(out[name]) * self.max_degps
            if abs(desired) < self.deadband:
                self._integral[name] = 0.0
                out[name] = 0.0
                continue
            actual = float(joint_vels_degps[vel_idx])
            err = desired - actual
            self._integral[name] = float(np.clip(
                self._integral[name] + err * dt, -self.ki_max, self.ki_max,
            ))
            cmd = self.kp * err + self.ki * self._integral[name]
            out[name] = float(np.clip(cmd, -1.0, 1.0))
        return out

    def reset(self) -> None:
        for k in self._integral:
            self._integral[k] = 0.0


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
        "--robot",
        choices=[*sorted(ROBOT_PROFILES), "auto"],
        default="auto",
        help="Robot profile. 'auto' detects only the board profile at startup (default).",
    )
    parser.add_argument(
        "--ip",
        default="192.168.0.132:8080",
        metavar="HOST[:PORT]",
        help="UDP remote IP and optional port (default: 192.168.0.132:8080).",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Override PWM servo config path selected by --robot.",
    )
    parser.add_argument(
        "--control-config-file",
        default=None,
        help="Override controller/IK/IMU config path selected by --robot.",
    )
    parser.add_argument(
        "--pwm-i2c-bus",
        type=int,
        default=None,
        help="Override PCA9685 Linux I2C bus selected by --robot.",
    )
    parser.add_argument(
        "--pwm-i2c-addr",
        type=lambda value: int(value, 0),
        default=None,
        help="Override PCA9685 I2C address selected by --robot, e.g. 0x40.",
    )
    parser.add_argument(
        "--disable-imu",
        action="store_true",
        help="Disable IMU startup (both profiles enable IMU by default).",
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable toggleable PWM channels so they are left out of control.",
    )
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
    parser.add_argument("--vel-kp",        type=float, default=0.04,  help="Velocity PI — proportional gain (cmd units / deg_s error)")
    parser.add_argument("--vel-ki",        type=float, default=0.008, help="Velocity PI — integral gain")
    parser.add_argument("--vel-ki-max",    type=float, default=0.4,   help="Velocity PI — I-term anti-windup clamp")
    parser.add_argument("--vel-deadband",  type=float, default=2.0,   help="Velocity PI — error deadband (deg/s)")
    parser.add_argument("--vel-max-degps", type=float, default=30.0,  help="Velocity PI — joystick full deflection = this many deg/s")
    return parser.parse_args()


def resolve_robot_profile(args) -> dict:
    profile = _resolve_board_profile(args.robot)   # handles 'auto' detection
    if args.config_file is not None:
        profile['servo_config_file'] = args.config_file
        profile['config_file'] = args.config_file
    if args.control_config_file is not None:
        profile['control_config_file'] = args.control_config_file
    if args.pwm_i2c_bus is not None:
        profile['pwm_i2c_bus'] = args.pwm_i2c_bus
    if args.pwm_i2c_addr is not None:
        profile['pwm_i2c_addr'] = args.pwm_i2c_addr
    if args.disable_imu:
        profile['enable_imu'] = False
    return profile


class PWMOnlyController:
    """Small direct-mode controller facade used when IMUs are disabled."""

    def __init__(self, hardware):
        self.hardware = hardware

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def enter_direct_mode(self) -> None:
        return

    def exit_direct_mode(self) -> None:
        return

    def set_velocity_mode(self, _mode: str) -> None:
        return

    def get_joint_angles(self):
        return np.zeros(4, dtype=np.float32)

    def get_joint_velocities_with_age(self) -> tuple:
        return None, float('inf')

    def give_direct_commands(self, commands: dict) -> None:
        self.hardware.send_named_pwm_commands(commands)


def main():
    args = parse_args()
    robot_profile = resolve_robot_profile(args)
    imu_enabled = bool(robot_profile['enable_imu'])

    # ---- UDP (mirror drive_logger.py wire format exactly) ----
    _ip_arg = args.ip
    if ":" in _ip_arg:
        _host, _port_str = _ip_arg.rsplit(":", 1)
        try:
            _port = int(_port_str)
        except ValueError:
            print(f"Invalid port in --ip {_ip_arg!r}; using 8080", file=sys.stderr)
            _host, _port = _ip_arg, 8080
    else:
        _host, _port = _ip_arg, 8080

    server = UDPSocket(local_id=2)
    server.setup(_host, _port, inputs='10b', outputs='', is_server=True)

    # ---- Hardware ----
    _selected = args.robot
    _resolved = robot_profile.get('_resolved_board', _selected)
    _auto_note = f" (auto-detected as '{_resolved}')" if _selected == "auto" else ""
    print("Initializing hardware...")
    print(
        f"Robot: {_selected}{_auto_note} | servo_config={robot_profile['servo_config_file']} | "
        f"control_config={robot_profile['control_config_file']} | "
        f"I2C bus={robot_profile['pwm_i2c_bus']} addr=0x{robot_profile['pwm_i2c_addr']:02X} | "
        f"IMU={'on' if imu_enabled else 'off'} | "
        f"UDP={_host}:{_port} | "
        f"toggleable channels={'disabled' if args.disable else 'enabled'}"
    )
    from modules.hardware_interface import HardwareFaultError, HardwareInterface

    hardware = HardwareInterface(
        config_file=robot_profile['servo_config_file'],
        control_config_file=robot_profile['control_config_file'],
        pump_auto_mode=True,
        toggle_channels=not args.disable,
        stale_timeout_s=0.5,
        enable_pwm=True,
        enable_imu=imu_enabled,
        enable_adc=False,           # not needed for plain driving
        start_imu_reader=imu_enabled,
        start_adc_reader=False,
        cleanup_disable_osc=False,
        pwm_i2c_bus=robot_profile['pwm_i2c_bus'],
        pwm_i2c_addr=robot_profile['pwm_i2c_addr'],
    )

    print("Waiting for hardware to be ready...")
    try:
        while not hardware.is_hardware_ready():
            time.sleep(0.1)
        print("Hardware ready.")
    except HardwareFaultError as e:
        subsystem = getattr(e, 'subsystem', 'hardware')
        reason = getattr(e, 'reason', str(e))
        print(f"\n*** HARDWARE FAULT ({subsystem}): {reason} ***")
        hardware.shutdown()
        raise SystemExit(1)

    # ---- Controller (direct mode, IK/PID bypassed) ----
    print("Starting controller...")
    if imu_enabled:
        from modules.excavator_controller import ExcavatorController

        controller = ExcavatorController(
            hardware,
            config=None,
            enable_perf_tracking=False,
            control_config_file=robot_profile['control_config_file'],
        )
    else:
        controller = PWMOnlyController(hardware)
    controller.start()
    if imu_enabled:
        time.sleep(2.0)             # numba JIT warmup
    controller.enter_direct_mode()
    controller.set_velocity_mode('gyro_only')
    control_joint_names = get_control_joint_names(controller)
    imu_role_order = get_imu_role_order(controller)
    print(
        "Controller in DIRECT mode "
        f"(velocity feedback: {'gyro_only' if imu_enabled else 'disabled, no IMU'})."
    )

    linkage_compensator = None
    universal_compensator = None
    # comp_mode: 0=OFF, 1=raw summary, 2=universal smooth, 3=velocity PID
    comp_mode = 1 if args.linkage_rate_correction else 0
    COMP_LABELS = ["OFF", "valve scale (raw)", "pump gain (universal shape)", "velocity PID"]
    vel_controller = JointVelocityController(
        kp=args.vel_kp,
        ki=args.vel_ki,
        ki_max=args.vel_ki_max,
        deadband_degps=args.vel_deadband,
        max_degps=args.vel_max_degps,
    )

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
    prev_loop_time = time.perf_counter()

    right_rl = right_ud = left_rl = left_ud = 0.0
    right_paddle = left_paddle = 0.0
    button_prev = [0.0, 0.0, 0.0, 0.0]
    button_threshold = 0.5

    print_angles = False
    iter_count = 0

    try:
        while True:
            now = time.perf_counter()
            actual_dt = now - prev_loop_time
            prev_loop_time = now

            float_data = UDPSocket.ints_to_floats(server.get_latest() or [])
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
                    if imu_enabled:
                        print_angles = not print_angles
                        state = "ON" if print_angles else "OFF"
                        print(f"\n[Button 0] IMU/joint angle print {state}")
                    else:
                        print("\n[Button 0] IMU/joint angle print unavailable (IMU is disabled)")

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

                # Button 3: cycle compensation mode OFF → raw → universal smooth → velocity PID.
                if buttons[3] > button_threshold and button_prev[3] <= button_threshold:
                    old_comp_mode = comp_mode
                    comp_mode = (comp_mode + 1) % 4
                    if comp_mode == 1 and linkage_compensator is None:
                        comp_mode = (comp_mode + 1) % 4
                    if comp_mode == 2 and universal_compensator is None:
                        comp_mode = (comp_mode + 1) % 4
                    if comp_mode == 3 and not imu_enabled:
                        comp_mode = (comp_mode + 1) % 4
                    # reset integrator when leaving velocity PID mode
                    if old_comp_mode == 3 and comp_mode != 3:
                        vel_controller.reset()
                    print(f"\n[Button 3] compensation mode: {COMP_LABELS[comp_mode]}")

                button_prev = buttons

            # Direct-mode commands straight from the joystick.
            canonical_commands = {
                'rotate':    left_rl,
                'lift_boom': right_ud,
                'tilt_boom': left_ud,
                'scoop':     right_rl,
                'trackR':    right_paddle,
                'trackL':    left_paddle,
            }
            joint_angles = controller.get_joint_angles()

            # Mode 1: scale valve commands per-joint (valve curves are affected).
            # Mode 2: valve commands untouched — correction goes to pump gain only.
            # Mode 3: closed-loop velocity PI — joystick sets desired deg/s, gyro is feedback.
            if comp_mode == 1 and linkage_compensator is not None:
                canonical_commands = linkage_compensator.apply(canonical_commands, joint_angles)
            elif comp_mode == 3 and imu_enabled:
                # TODO: add linkage feed-forward — apply linkage_compensator to commands first,
                # then run vel_controller on top. Reduces initial tracking error at positions
                # where valve effectiveness drops (e.g. full arm extension), without relying
                # solely on the integrator to catch up.
                joint_vels, vel_age = controller.get_joint_velocities_with_age()
                if vel_age < 0.05:
                    canonical_commands = vel_controller.apply(canonical_commands, joint_vels, actual_dt)
                else:
                    vel_controller.reset()
            controller.give_direct_commands(canonical_commands)

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
                    factor = universal_compensator.pump_correction_factor(canonical_commands, joint_angles)
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
