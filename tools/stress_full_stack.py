#!/usr/bin/env python3
"""Run the excavator stack at a fixed rate while forcing zero PWM commands.

This is a hardware stress/integration test, not a unit test.
It starts the real hardware interface and controller, submits IK commands through
the service/protocol boundary, and forces every named PWM valve command to zero
before it reaches hardware.

The pump channel is left under the normal controller logic so valve zeros still
exercise the real PWM update path with the configured pump behavior.
"""

import argparse
import csv
import logging
import math
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from modules.control_protocol import (  # noqa: E402
    ControlCommand,
    ControlMode,
    DirectCommand,
    PoseTarget,
    decode_command_message,
    encode_command_message,
    encode_telemetry_message,
)
from modules.excavator_controller import ControllerConfig, ExcavatorController  # noqa: E402
from modules.hardware_interface import HardwareInterface  # noqa: E402
from modules.robot_service import RobotService  # noqa: E402
from modules.rt_utils import apply_rt_to_thread, reset_to_normal, SCHED_FIFO  # noqa: E402


TRAJECTORY_CSV_COLUMNS = [
    "x_g", "y_g", "z_g", "x_e", "y_e", "z_e",
    "quat_g_w", "quat_g_x", "quat_g_y", "quat_g_z",
    "quat_e_w", "quat_e_x", "quat_e_y", "quat_e_z",
    "joint_1", "joint_2", "joint_3", "joint_4",
    "vel_1", "vel_2", "vel_3", "vel_4",
    "acc_1", "acc_2", "acc_3", "acc_4",
    "pos_des_1", "pos_des_2", "pos_des_3", "pos_des_4",
    "condition_number", "yoshikawa_index",
    "sv_1", "sv_2", "sv_3", "sv_4",
    "waypoint_idx", "progress",
]

STRESS_LOG_COLUMNS = [
    't_s',
    'cmd_x', 'cmd_y', 'cmd_z',
    'act_rot_y_deg',
    'joint_0_deg', 'joint_1_deg', 'joint_2_deg', 'joint_3_deg',
    'vel_0_degps', 'vel_1_degps', 'vel_2_degps', 'vel_3_degps',
    'loop_util_pct', 'headroom_ms',
    'sensor_ms', 'ik_fk_ms', 'pwm_ms',
    'imu_hz', 'adc_hz',
    'sequence',
]


def _percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * q))
    return ordered[idx]


def _cpu_affinity(core):
    return None if core is None else {int(core)}


def _quat_from_y_deg(angle_deg: float) -> np.ndarray:
    half_angle = 0.5 * math.radians(float(angle_deg))
    return np.asarray([
        math.cos(half_angle),
        0.0,
        math.sin(half_angle),
        0.0,
    ], dtype=np.float32)


def _csv_safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _discover_log_schema(hardware) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    imu_roles = list(getattr(hardware, '_imu_sensor_roles', []))
    adc_names = []
    try:
        adc_names = list(hardware.get_latest_adc_readings().keys())
    except Exception:
        adc_plan = getattr(hardware, '_adc_channel_plan', None) or []
        adc_names = [entry['name'] for entry in adc_plan if isinstance(entry, dict) and 'name' in entry]

    adc_name_map = {name: _csv_safe_name(name) for name in adc_names}

    sensor_columns = []
    for role in imu_roles:
        sensor_columns.extend([
            f'imu_{role}_quat_w',
            f'imu_{role}_quat_x',
            f'imu_{role}_quat_y',
            f'imu_{role}_quat_z',
            f'imu_{role}_gyro_x',
            f'imu_{role}_gyro_y',
            f'imu_{role}_gyro_z',
        ])
    for name in adc_names:
        sensor_columns.append(f"adc_{adc_name_map[name]}")

    return TRAJECTORY_CSV_COLUMNS + STRESS_LOG_COLUMNS + sensor_columns, imu_roles, adc_names, adc_name_map


def _read_sensor_snapshot(hardware, imu_roles: list[str], adc_names: list[str]) -> list[float]:
    role_to_quat = {}
    role_to_gyro = {}

    try:
        quats = hardware.read_all_imu_quaternions()
        if quats is not None:
            for role, quat in zip(imu_roles, quats):
                role_to_quat[role] = np.asarray(quat, dtype=np.float32)
    except Exception:
        pass

    try:
        base_imu = hardware.read_base_imu()
        if base_imu is not None and 'gyro' in base_imu:
            role_to_gyro['base'] = np.asarray(base_imu['gyro'], dtype=np.float32)
    except Exception:
        pass

    try:
        payload = hardware.try_read_imu_gyro() if hasattr(hardware, 'try_read_imu_gyro') else None
        gyro_packets = None if not payload else payload.get('gyro')
        joint_roles = list(getattr(hardware, '_imu_joint_roles', []))
        if gyro_packets is not None:
            for role, gyro in zip(joint_roles, gyro_packets):
                role_to_gyro[role] = np.asarray(gyro, dtype=np.float32)
    except Exception:
        pass

    try:
        adc_snapshot = hardware.get_latest_adc_snapshot().get('readings', {})
    except Exception:
        adc_snapshot = {}

    values: list[float] = []
    for role in imu_roles:
        quat = role_to_quat.get(role)
        if quat is None or len(quat) < 4:
            values.extend([float('nan')] * 4)
        else:
            values.extend(float(quat[i]) for i in range(4))

        gyro = role_to_gyro.get(role)
        if gyro is None or len(gyro) < 3:
            values.extend([float('nan')] * 3)
        else:
            values.extend(float(gyro[i]) for i in range(3))

    for name in adc_names:
        raw = adc_snapshot.get(name)
        values.append(float(raw) if raw is not None else float('nan'))

    return values


def _load_rate_config():
    cfg_path = ROOT / "configuration_files" / "control_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    rates = data.get("rates", {}) if isinstance(data, dict) else {}
    controller = data.get("controller", {}) if isinstance(data, dict) else {}

    control_hz = float(rates.get("control_hz", 100.0))
    output_limits = (
        float(controller.get("output_limits_min", -1.0)),
        float(controller.get("output_limits_max", 1.0)),
    )
    return control_hz, output_limits


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-stack stress test with explicit zero PWM commands")
    parser.add_argument("--rate-hz", type=float, default=None,
                        help="Override control rate; IMU is firmware-fixed at 200 Hz, ADC is fixed at 20 Hz")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--ready-timeout-s", type=float, default=30.0)
    parser.add_argument("--fifo-priority", type=int, default=75)
    parser.add_argument("--control-core", type=int, default=2,
                        help="CPU core for sender + control loop (default: 2)")
    parser.add_argument("--io-core", type=int, default=3,
                        help="CPU core for USB reader + IMU + ADC threads (default: 3)")
    parser.add_argument("--ik-dither-m", type=float, default=0.004,
                        help="Small IK target oscillation to exercise reachability checks; valves remain zero")
    parser.add_argument("--settle-s", type=float, default=12.0,
                        help="Seconds to run after warmup before recording stats; must exceed the controller's "
                             "rolling-window size (deadline_window_sec, typically 10 s) for clean ctrl-miss readings")
    parser.add_argument("--log", action="store_true",
                        help="Write per-cycle CSV trajectory log to logs_stress/ (heaviest stack mode)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for CSV log output (default: <repo>/logs_stress/)")
    parser.add_argument("--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.WARNING))

    hardware = None
    service = None
    sender_intervals_ms = deque(maxlen=4000)
    sender_miss_count = 0
    sender_miss_window = deque(maxlen=4000)
    sender_cycle_count = 0
    log_rows = [] if args.log else None
    log_columns = None
    log_imu_roles = []
    log_adc_names = []
    prev_joint_vel_rad = None
    prev_joint_vel_t = None

    try:
        config_control_hz, output_limits = _load_rate_config()
        target_hz = float(args.rate_hz) if args.rate_hz is not None else float(config_control_hz)

        hardware = HardwareInterface(
            config_file=str(ROOT / "configuration_files" / "servo_config_200.yaml"),
            log_level=args.log_level.upper(),
            pump_auto_mode=True,
            cleanup_disable_osc=False,
            perf_enabled=True,
            enable_adc=True,
            adc_sample_hz=20.0,
            rt_lock_memory=True,
            usb_rt_priority=args.fifo_priority,
            imu_rt_priority=args.fifo_priority,
            adc_rt_priority=args.fifo_priority,
            usb_cpu_core=args.io_core,
            imu_cpu_core=args.io_core,
            adc_cpu_core=args.io_core,
        )

        deadline = time.time() + max(1.0, args.ready_timeout_s)
        while not hardware.is_hardware_ready():
            if time.time() >= deadline:
                raise TimeoutError("Hardware did not become ready before timeout")
            time.sleep(0.1)

        original_send_named_pwm_commands = hardware.send_named_pwm_commands

        def send_zero_named_pwm_commands(commands):
            zero_commands = {name: 0.0 for name in dict(commands)}
            return original_send_named_pwm_commands(zero_commands)

        hardware.send_named_pwm_commands = send_zero_named_pwm_commands

        controller = ExcavatorController(
            hardware,
            config=ControllerConfig(output_limits=output_limits, control_frequency=target_hz),
            enable_perf_tracking=True,
            log_level=args.log_level.upper(),
            rt_priority=args.fifo_priority,
            rt_lock_memory=True,
            rt_cpu_core=args.control_core,
        )
        service = RobotService(controller, hardware)
        service.start()

        if args.log:
            log_columns, log_imu_roles, log_adc_names, _ = _discover_log_schema(hardware)

        time.sleep(max(0.0, args.warmup_s))
        service.reset_perf_stats()

        apply_rt_to_thread(
            priority=args.fifo_priority,
            policy=SCHED_FIFO,
            cpu_affinity=_cpu_affinity(args.control_core),
            quiet=True,
        )

        base_pos, base_rot = service.get_pose()
        base_x = float(base_pos[0])
        base_y = float(base_pos[1])
        base_z = float(base_pos[2])
        base_rot = float(base_rot)
        dither_m = max(0.0, float(args.ik_dither_m))

        period_s = 1.0 / max(1.0, target_hz)
        miss_threshold_s = period_s * 1.01
        settle_started = time.perf_counter()
        next_run = settle_started
        last_sender_start = None
        last_print = settle_started
        sequence = 0
        measuring = False
        started = settle_started  # overwritten when settle ends

        print(
            f"[stress] full stack start: control={target_hz:.1f}Hz imu=200Hz(fw) "
            f"adc=20Hz settle={args.settle_s:.1f}s duration={args.duration_s:.1f}s "
            f"fifo={args.fifo_priority} control_core={args.control_core} io_core={args.io_core} "
            f"lock_memory=True ik_dither={dither_m:.4f}m valves=forced_zero"
            + (" CSV_log=enabled" if args.log else "")
        )

        while True:
            loop_start = time.perf_counter()

            # Transition from settle to measurement once settle period elapses.
            if not measuring and (loop_start - settle_started) >= args.settle_s:
                measuring = True
                started = loop_start
                last_sender_start = None
                last_print = loop_start
                sender_cycle_count = 0
                sender_miss_count = 0
                sender_intervals_ms.clear()
                sender_miss_window.clear()
                service.reset_perf_stats()
                print(f"[stress] settle complete — measurement started (settle={args.settle_s:.1f}s)")

            if measuring and (loop_start - started) >= args.duration_s:
                break

            if measuring and last_sender_start is not None:
                interval_s = loop_start - last_sender_start
                sender_intervals_ms.append(interval_s * 1000.0)
                missed = interval_s > miss_threshold_s
                sender_miss_window.append(1 if missed else 0)
                if missed:
                    sender_miss_count += 1
            if measuring:
                last_sender_start = loop_start

            phase = 1.0 if sequence % 2 == 0 else -1.0
            ik_cmd = ControlCommand(
                sequence=sequence,
                timestamp_ms=int(time.time() * 1000) & 0xFFFFFFFF,
                mode=ControlMode.IK,
                pose=PoseTarget(base_x + phase * dither_m, base_y, base_z, base_rot),
                direct=DirectCommand(0.0, 0.0, 0.0, 0.0),
            )
            decoded = decode_command_message(encode_command_message(ik_cmd))
            service.submit_command(decoded)

            telemetry = service.get_state()
            encode_telemetry_message(telemetry)

            if measuring:
                sender_cycle_count += 1

            # Per-cycle logging (heaviest stack mode)
            if args.log and measuring:
                perf_stats = service.get_debug_state().get("perf_stats", {})
                hw_stats = perf_stats.get('hardware_stats', {})
                act_pos, act_rot_deg = service.get_pose()
                cmd_pos = np.asarray([base_x + phase * dither_m, base_y, base_z], dtype=np.float32)
                joint_deg = np.asarray(controller.get_joint_angles(), dtype=np.float32)
                if joint_deg.shape[0] < 4:
                    joint_deg = np.pad(joint_deg, (0, 4 - joint_deg.shape[0]))
                vel_degps = np.asarray(controller.get_joint_velocities_degps() or [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                if vel_degps.shape[0] < 4:
                    vel_degps = np.pad(vel_degps, (0, 4 - vel_degps.shape[0]))
                joint_pos_rad = np.radians(joint_deg[:4].astype(np.float32))
                joint_vel_rad = np.radians(vel_degps[:4].astype(np.float32))
                now_t = time.perf_counter()
                if prev_joint_vel_rad is not None and prev_joint_vel_t is not None:
                    dt_acc = max(now_t - prev_joint_vel_t, 1e-6)
                    joint_acc_rad = (joint_vel_rad - prev_joint_vel_rad) / dt_acc
                else:
                    joint_acc_rad = np.zeros(4, dtype=np.float32)
                prev_joint_vel_rad = joint_vel_rad.copy()
                prev_joint_vel_t = now_t
                planned_quat = _quat_from_y_deg(base_rot)
                actual_quat = _quat_from_y_deg(act_rot_deg)
                cond = controller.get_condition_number()
                t_elapsed = loop_start - started
                progress = min(max(t_elapsed / max(args.duration_s, 1e-6), 0.0), 1.0)
                sensor_values = _read_sensor_snapshot(hardware, log_imu_roles, log_adc_names)
                log_rows.append([
                    float(cmd_pos[0]), float(cmd_pos[1]), float(cmd_pos[2]),
                    float(act_pos[0]), float(act_pos[1]), float(act_pos[2]),
                    float(planned_quat[0]), float(planned_quat[1]), float(planned_quat[2]), float(planned_quat[3]),
                    float(actual_quat[0]), float(actual_quat[1]), float(actual_quat[2]), float(actual_quat[3]),
                    float(joint_pos_rad[0]), float(joint_pos_rad[1]), float(joint_pos_rad[2]), float(joint_pos_rad[3]),
                    float(joint_vel_rad[0]), float(joint_vel_rad[1]), float(joint_vel_rad[2]), float(joint_vel_rad[3]),
                    float(joint_acc_rad[0]), float(joint_acc_rad[1]), float(joint_acc_rad[2]), float(joint_acc_rad[3]),
                    float(joint_pos_rad[0]), float(joint_pos_rad[1]), float(joint_pos_rad[2]), float(joint_pos_rad[3]),
                    float(cond), float('nan'),
                    float('nan'), float('nan'), float('nan'), float('nan'),
                    int(sequence % 2),
                    float(progress),
                    round(t_elapsed, 6),
                    round(base_x + phase * dither_m, 6), round(base_y, 6), round(base_z, 6),
                    round(float(act_rot_deg), 4),
                    round(float(joint_deg[0]), 4), round(float(joint_deg[1]), 4),
                    round(float(joint_deg[2]), 4), round(float(joint_deg[3]), 4),
                    round(float(vel_degps[0]), 4), round(float(vel_degps[1]), 4),
                    round(float(vel_degps[2]), 4), round(float(vel_degps[3]), 4),
                    round(perf_stats.get('loop_util_pct', perf_stats.get('cpu_usage_pct', 0.0)), 3),
                    round(perf_stats.get('avg_headroom_ms', 0.0), 4),
                    round(perf_stats.get('avg_sensor_ms', 0.0), 4),
                    round(perf_stats.get('avg_ik_fk_ms', 0.0), 4),
                    round(perf_stats.get('avg_pwm_ms', 0.0), 4),
                    round(perf_stats.get('imu_hz', 0.0), 2),
                    round(perf_stats.get('adc_hz', 0.0), 2),
                    sequence,
                    *sensor_values,
                ])

            sequence += 1

            now = time.perf_counter()
            if measuring and now - last_print >= 1.0:
                if not args.log:
                    perf_stats = service.get_debug_state().get("perf_stats", {})
                    hw_stats = perf_stats.get('hardware_stats', {})
                else:
                    hw_stats = perf_stats.get('hardware_stats', {})
                sender_hz = sender_cycle_count / max(1e-6, now - started)
                sender_recent_miss_pct = (sum(sender_miss_window) / max(1, len(sender_miss_window))) * 100.0
                ctrl_recent_miss_count = perf_stats.get('deadline_miss_1pct_count_recent', 0)
                ctrl_recent_miss_pct = perf_stats.get('deadline_miss_1pct_pct_recent', 0.0)
                p95_ms = _percentile(list(sender_intervals_ms), 0.95)
                p99_ms = _percentile(list(sender_intervals_ms), 0.99)
                window_sec = perf_stats.get('deadline_window_sec', 0.0)
                imu_hz = perf_stats.get('imu_hz', 0.0)
                adc_hz = perf_stats.get('adc_hz', 0.0)
                miss_label = "(window warming)" if (now - started) < window_sec else ""
                print(
                    "[stress] "
                    f"sender={sender_hz:.1f}Hz miss1%={sender_miss_count} ({sender_recent_miss_pct:.1f}% recent) "
                    f"p95={p95_ms:.2f}ms p99={p99_ms:.2f}ms | "
                    f"ctrl={perf_stats.get('actual_hz', 0.0):.1f}Hz imu={imu_hz:.0f}Hz adc={adc_hz:.1f}Hz "
                    f"procCPU={perf_stats.get('process_cpu_pct', 0.0):.1f}% "
                    f"loopUtil={perf_stats.get('loop_util_pct', perf_stats.get('cpu_usage_pct', 0.0)):.1f}% "
                    f"miss1%={ctrl_recent_miss_count} "
                    f"({ctrl_recent_miss_pct:.1f}%/{window_sec:.1f}s){miss_label}"
                )
                imu_s = hw_stats.get('imu', {})
                adc_s = hw_stats.get('adc', {})
                print(
                    "        "
                    f"sensor={perf_stats.get('avg_sensor_ms', 0.0):.2f}/{perf_stats.get('max_sensor_ms', 0.0):.2f}ms "
                    f"ik_fk={perf_stats.get('avg_ik_fk_ms', 0.0):.2f}/{perf_stats.get('max_ik_fk_ms', 0.0):.2f}ms "
                    f"pwm={perf_stats.get('avg_pwm_ms', 0.0):.2f}/{perf_stats.get('max_pwm_ms', 0.0):.2f}ms "
                    f"headroom={perf_stats.get('avg_headroom_ms', 0.0):.2f}ms "
                    f"loop_p99={perf_stats.get('jitter_p99_ms', 0.0):.2f}ms | "
                    f"imu_arr={imu_s.get('avg_interval_ms', 0.0):.1f}±{imu_s.get('std_interval_ms', 0.0):.2f}ms "
                    f"(max={imu_s.get('max_interval_ms', 0.0):.1f}ms) "
                    f"adc_arr={adc_s.get('avg_interval_ms', 0.0):.1f}±{adc_s.get('std_interval_ms', 0.0):.1f}ms "
                    f"(max={adc_s.get('max_interval_ms', 0.0):.1f}ms)"
                )
                last_print = now

            next_run += period_s
            sleep_s = next_run - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_run = time.perf_counter()

        perf_stats = service.get_debug_state().get("perf_stats", {})
        hw_stats = perf_stats.get('hardware_stats', {})
        sender_avg_hz = sender_cycle_count / max(1e-6, time.perf_counter() - started)
        print(
            "[stress] done | "
            f"sender={sender_avg_hz:.1f}Hz cycles={sender_cycle_count} | "
            f"ctrl={perf_stats.get('actual_hz', 0.0):.1f}Hz "
            f"imu={perf_stats.get('imu_hz', 0.0):.0f}Hz adc={perf_stats.get('adc_hz', 0.0):.1f}Hz "
            f"procCPU={perf_stats.get('process_cpu_pct', 0.0):.1f}% "
            f"loopUtil={perf_stats.get('loop_util_pct', perf_stats.get('cpu_usage_pct', 0.0)):.1f}% "
            f"sender_miss1%={sender_miss_count} ({(sender_miss_count / max(1, sender_cycle_count - 1)) * 100.0:.1f}% cumulative) "
            f"ctrl_miss1%={perf_stats.get('deadline_miss_1pct_count_recent', 0)} ({perf_stats.get('deadline_miss_1pct_pct_recent', 0.0):.1f}% recent)"
        )
        imu_s = hw_stats.get('imu', {})
        adc_s = hw_stats.get('adc', {})
        print(
            "        "
            f"sensor={perf_stats.get('avg_sensor_ms', 0.0):.2f}/{perf_stats.get('max_sensor_ms', 0.0):.2f}ms "
            f"ik_fk={perf_stats.get('avg_ik_fk_ms', 0.0):.2f}/{perf_stats.get('max_ik_fk_ms', 0.0):.2f}ms "
            f"pwm={perf_stats.get('avg_pwm_ms', 0.0):.2f}/{perf_stats.get('max_pwm_ms', 0.0):.2f}ms "
            f"headroom={perf_stats.get('avg_headroom_ms', 0.0):.2f}ms "
            f"loop_p99={perf_stats.get('jitter_p99_ms', 0.0):.2f}ms | "
            f"imu_arr={imu_s.get('avg_interval_ms', 0.0):.1f}±{imu_s.get('std_interval_ms', 0.0):.2f}ms "
            f"(max={imu_s.get('max_interval_ms', 0.0):.1f}ms) "
            f"adc_arr={adc_s.get('avg_interval_ms', 0.0):.1f}±{adc_s.get('std_interval_ms', 0.0):.1f}ms "
            f"(max={adc_s.get('max_interval_ms', 0.0):.1f}ms)"
        )

        if args.log and log_rows:
            log_dir = Path(args.log_dir) if args.log_dir else ROOT / "logs_stress"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"stress_{ts}_{target_hz:.0f}hz_{int(args.duration_s)}s.csv"
            with log_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(log_columns or (TRAJECTORY_CSV_COLUMNS + STRESS_LOG_COLUMNS))
                writer.writerows(log_rows)
            print(f"[stress] log saved → {log_path} ({len(log_rows)} rows)")

        return 0

    finally:
        reset_to_normal(quiet=True)
        if service is not None:
            try:
                service.stop()
            except Exception:
                pass
        elif hardware is not None:
            try:
                hardware.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
