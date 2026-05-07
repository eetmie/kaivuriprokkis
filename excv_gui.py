#!/usr/bin/env python3
"""
Server-side receiver: accepts commands from client GUI, drives the excavator.

  python excv_gui.py                   # Normal operation
  python excv_gui.py --perf            # Performance monitoring (compact)
  python excv_gui.py --jac             # Jacobian debugging
  python excv_gui.py --log-level DEBUG # Verbose module logging
  python excv_gui.py --fifo-priority 75 --lock-memory --control-core 2 --io-core 3
"""

import time
import logging
import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

from modules.udp_socket import UDPSocket
from modules.excavator_controller import ExcavatorController
from modules.control_protocol import (
    COMMAND_PACKET_SIZE,
    TELEMETRY_PACKET_SIZE,
    decode_command_message,
    encode_telemetry_message,
)
from modules.robot_service import RobotService
from modules.rt_utils import apply_rt_to_thread, reset_to_normal, SCHED_FIFO


def _cpu_affinity(core):
    return None if core is None else {int(core)}


def _fmt_vec(v, n=3):
    return ", ".join(f"{float(x):+.3f}" for x in list(v)[:n])


def _print_normal(last_command, actual_pos, actual_rot, packet_rate, packets_received, debug_state):
    cond = debug_state.get('condition_number', 0) if debug_state else 0
    cond_str = f" | Cond: {cond:.1f}" if cond > 0 else ""
    base_str = ""
    if debug_state:
        base_quat = debug_state.get('base_imu_quat')
        if base_quat is not None and len(base_quat) >= 4:
            w, x, y, z = float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3])
            pitch_rad = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
            base_str = f" | Base: [{_fmt_vec(base_quat, n=4)}] pitch={np.degrees(pitch_rad):+.1f}°"
    print(f"[{packet_rate:.1f} Hz] "
          f"Target: [{last_command.pose.x:+.3f}, {last_command.pose.y:+.3f}, {last_command.pose.z:+.3f}]m "
          f"Rot: {last_command.pose.rot_y_deg:+.1f}° | "
          f"Actual: [{actual_pos[0]:+.3f}, {actual_pos[1]:+.3f}, {actual_pos[2]:+.3f}]m "
          f"Rot: {actual_rot:+.1f}° | "
          f"Packets: {packets_received}"
          f"{cond_str}"
          f"{base_str}")


def _print_perf(debug_state, network_times, packet_rate, perf_only):
    if not debug_state:
        return
    perf_stats = debug_state.get('perf_stats', {})
    if not perf_stats:
        return

    actual_hz = perf_stats.get('actual_hz', perf_stats.get('hz', 0.0))
    loop_util = perf_stats.get('loop_util_pct', perf_stats.get('cpu_usage_pct', 0.0))
    line = (f"Loop: {perf_stats.get('avg_loop_time_ms', 0):.2f}ms "
            f"(min={perf_stats.get('min_loop_time_ms', 0):.2f} max={perf_stats.get('max_loop_time_ms', 0):.2f}) | "
            f"Timing: std={perf_stats.get('std_loop_time_ms', 0):.2f} "
            f"p95={perf_stats.get('jitter_p95_ms', 0):.2f} "
            f"p99={perf_stats.get('jitter_p99_ms', 0):.2f} "
            f"maxΔ={perf_stats.get('max_step_ms', 0):.2f}ms | "
            f"Hz: {actual_hz:.1f} | "
            f"ProcCPU: {perf_stats.get('process_cpu_pct', 0):.1f}% | "
            f"LoopUtil: {loop_util:.1f}% | "
            f"Headroom: {perf_stats.get('avg_headroom_ms', 0):.2f}ms")

    stages = []
    sensor_avg = perf_stats.get('avg_sensor_ms', 0)
    if sensor_avg > 0:
        stages.append(f"Sensors: {sensor_avg:.2f}ms "
                      f"({perf_stats.get('min_sensor_ms', 0):.2f}-{perf_stats.get('max_sensor_ms', 0):.2f})")
    ik_avg = perf_stats.get('avg_ik_fk_ms', 0)
    if ik_avg > 0:
        stages.append(f"IK/FK: {ik_avg:.2f}ms "
                      f"({perf_stats.get('min_ik_fk_ms', 0):.2f}-{perf_stats.get('max_ik_fk_ms', 0):.2f})")
    pwm_avg = perf_stats.get('avg_pwm_ms', 0)
    if pwm_avg > 0:
        stages.append(f"PWM: {pwm_avg:.2f}ms "
                      f"({perf_stats.get('min_pwm_ms', 0):.2f}-{perf_stats.get('max_pwm_ms', 0):.2f})")
    if network_times and not perf_only:
        stages.append(f"Net: {np.mean(network_times):.2f}ms "
                      f"({np.min(network_times):.2f}-{np.max(network_times):.2f})")
    if stages:
        line += " | " + " | ".join(stages)

    cond = debug_state.get('condition_number', 0)
    if cond > 0:
        line += f" | Cond: {cond:.1f}"

    if not perf_only:
        line += f" | Pkt: {packet_rate:.1f}Hz"
        try:
            last_vel = perf_stats.get('last_joint_vel_degps', [])
            vel_cap = perf_stats.get('effective_vel_cap_degps', [])
            vel_lim_on = perf_stats.get('ik_vel_lim_enabled', False)
            if last_vel:
                names = ['slew', 'boom', 'arm', 'bucket']
                line += " | Vel(deg/s): " + ", ".join(f"{n}={v:+.1f}" for n, v in zip(names, last_vel))
                if vel_lim_on and vel_cap:
                    line += " | Cap(deg/s): " + ", ".join(f"{n}={c:.1f}" for n, c in zip(names, vel_cap))
        except Exception:
            pass

    print(line)


def _print_jacobian(telemetry, actual_pos, last_command, debug_state, state):
    if not debug_state:
        return
    fk_quats = debug_state.get('fk_quaternions')
    robot_config = debug_state.get('robot_config')
    if fk_quats is None or len(fk_quats) < 4 or robot_config is None:
        return
    try:
        from modules.differential_ik import compute_jacobian, extract_axis_rotation, project_to_rotation_axes

        J = compute_jacobian(fk_quats, robot_config)
        J_pos = J[0:3, :]
        J_ang = J[3:6, :]
        s = np.linalg.svd(J_pos, compute_uv=False)
        sv = s.astype(np.float32)
        s_nonzero = s[s > 1e-9]
        cond = float(np.max(s_nonzero) / np.min(s_nonzero)) if len(s_nonzero) > 0 else float('inf')

        axes = np.array(robot_config.rotation_axes, dtype=np.float32)
        projected = project_to_rotation_axes(fk_quats, axes)
        joint_angles = np.array([
            extract_axis_rotation(q, axis) for q, axis in zip(projected, robot_config.rotation_axes)
        ], dtype=np.float32)

        ee_pos = np.asarray(telemetry.joint_positions[4], dtype=np.float32)
        joint_deg = np.degrees(joint_angles).astype(np.float32)

        prev_ja = state.get('prev_joint_angles')
        prev_ee = state.get('prev_ee_pos')
        dq = dq_deg = ee_pred = ee_actual = None
        if prev_ja is not None:
            dq = (joint_angles - prev_ja).astype(np.float32)
            dq_deg = np.degrees(dq).astype(np.float32)
            ee_pred = (J_pos @ dq).astype(np.float32)
        if prev_ee is not None:
            ee_actual = (ee_pos - prev_ee).astype(np.float32)

        state['prev_joint_angles'] = joint_angles
        state['prev_ee_pos'] = ee_pos.astype(np.float32)

        print(f"  [JAC] s=[{_fmt_vec(sv, n=min(3, len(sv)))}] cond={cond:.2f}")
        if dq is not None:
            print(f"  [JAC] dq_deg=[{_fmt_vec(dq_deg, n=4)}]; ee_pred=[{_fmt_vec(ee_pred, n=3)}] m")
        if ee_actual is not None:
            print(f"  [JAC] ee_act=[{_fmt_vec(ee_actual, n=3)}] m")
        jp = J_pos.astype(np.float32)
        print(f"  [Jpos] r0=[{_fmt_vec(jp[0], n=4)}] | r1=[{_fmt_vec(jp[1], n=4)}] | r2=[{_fmt_vec(jp[2], n=4)}]")
        ja = J_ang.astype(np.float32)
        print(f"  [Jang] r3=[{_fmt_vec(ja[0], n=4)}] | r4=[{_fmt_vec(ja[1], n=4)}] | r5=[{_fmt_vec(ja[2], n=4)}]")
        if joint_deg is not None:
            dyaw = float(dq_deg[0]) if (dq_deg is not None and len(dq_deg) > 0) else 0.0
            print(f"  [JNT] deg=[{_fmt_vec(joint_deg, n=4)}]; "
                  f"[SLEW] yaw={float(joint_deg[0]):+.1f} deg dyaw={dyaw:+.2f} deg")
        if last_command:
            target = np.array([last_command.pose.x, last_command.pose.y, last_command.pose.z], dtype=np.float32)
            pos_err = target - np.asarray(actual_pos, dtype=np.float32)
            print(f"  [ERR] pos=[{_fmt_vec(pos_err, n=3)}] m")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Excavator server (receives commands, drives hardware)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--jac", action="store_true", help="Jacobian debug output")
    parser.add_argument("--perf", action="store_true", help="Performance monitoring output")
    parser.add_argument("--fifo-priority", "--rt-priority", dest="fifo_priority", type=int, default=75)
    parser.add_argument("--lock-memory", action="store_true", help="Call mlockall() for RT threads")
    parser.add_argument("--control-core", type=int, default=None, help="CPU core for main loop and control loop")
    parser.add_argument("--io-core", type=int, default=None, help="CPU core for USB reader, IMU, and ADC threads")
    args, _ = parser.parse_known_args()

    quiet = args.perf
    jac_state = {}  # mutable state for Jacobian delta tracking

    # Logger
    app_logger = logging.getLogger("excv_gui")
    app_logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
    app_logger.addHandler(handler)

    if not quiet:
        app_logger.info("=" * 60)
        app_logger.info("  EXCAVATOR SERVER — receiving commands from client GUI")
        app_logger.info("=" * 60)

    # ---- UDP setup ----
    server = UDPSocket(local_id=2, max_age_seconds=0.5)
    server.setup("192.168.0.132", 8080,
                 num_inputs=COMMAND_PACKET_SIZE, num_outputs=TELEMETRY_PACKET_SIZE,
                 is_server=True, data_format='b')

    if not quiet:
        app_logger.info("Waiting for GUI connection...")
    if not server.handshake(timeout=30.0):
        app_logger.error("Handshake failed!")
        return
    handshake_info = server.get_handshake_info()
    client_rate_hz = handshake_info.get("remote_nominal_rate_hz")
    if not quiet:
        app_logger.info("Connected to GUI!")
        if client_rate_hz is not None:
            app_logger.info(f"Client nominal rate: {client_rate_hz:.2f} Hz")

    # ---- Hardware ----
    from modules.hardware_interface import HardwareInterface

    hw_log_level = "WARNING" if quiet else args.log_level.upper()
    hardware = HardwareInterface(
        log_level=hw_log_level, pump_auto_mode=False, cleanup_disable_osc=False,
        enable_adc=False, start_adc_reader=False,
        rt_lock_memory=args.lock_memory,
        usb_rt_priority=args.fifo_priority,
        imu_rt_priority=args.fifo_priority,
        adc_rt_priority=args.fifo_priority,
        usb_cpu_core=args.io_core,
        imu_cpu_core=args.io_core,
        adc_cpu_core=args.io_core,
    )
    while not hardware.is_hardware_ready():
        time.sleep(0.1)

    # ---- Controller + Service ----
    ctrl_log_level = "WARNING" if quiet else ("DEBUG" if args.jac else args.log_level.upper())
    controller = ExcavatorController(
        hardware, config=None, enable_perf_tracking=args.perf, log_level=ctrl_log_level,
        rt_priority=args.fifo_priority,
        rt_lock_memory=args.lock_memory,
        rt_cpu_core=args.control_core,
    )
    service = RobotService(controller, hardware)
    service.start()

    if not quiet:
        app_logger.info("Controller started, warming up...")
    time.sleep(5.0)

    # RT priority
    if args.fifo_priority > 0 or args.lock_memory or args.control_core is not None:
        if not apply_rt_to_thread(
            priority=args.fifo_priority,
            policy=SCHED_FIFO,
            lock_memory=args.lock_memory,
            cpu_affinity=_cpu_affinity(args.control_core),
            quiet=True,
        ):
            if not quiet:
                app_logger.warning("Failed to apply requested RT settings (run as root)")

    if not quiet and (args.lock_memory or args.control_core is not None or args.io_core is not None):
        app_logger.info(
            "RT settings: fifo=%s lock_memory=%s control_core=%s io_core=%s",
            args.fifo_priority, args.lock_memory, args.control_core, args.io_core,
        )

    if args.perf:
        service.reset_perf_stats()

    if not quiet:
        app_logger.info("READY TO RECEIVE COMMANDS")

    # ---- Main loop ----
    server.start_receiving()
    last_command = None
    packets_received = 0
    last_packet_count = 0
    last_print_time = time.time()
    network_times = []
    print_interval = 2.0 if args.perf else 1.0
    needs_debug = args.perf or args.jac
    loop_rate_hz = max(1.0, float(client_rate_hz)) if client_rate_hz is not None and client_rate_hz > 0 else 20.0
    loop_period = 1.0 / loop_rate_hz
    next_run_time = time.perf_counter()

    if not quiet:
        app_logger.info(f"UDP loop rate: {loop_rate_hz:.2f} Hz")

    try:
        while True:
            net_start = time.perf_counter()

            data = server.get_latest()
            if data:
                try:
                    last_command = decode_command_message(data)
                    service.submit_command(last_command)
                    packets_received += 1
                except Exception as e:
                    if not quiet:
                        app_logger.error(f"Decode error: {e}")
                    continue

            try:
                telemetry = service.get_state()
                server.send(encode_telemetry_message(telemetry))
            except Exception:
                telemetry = None

            if args.perf and telemetry:
                net_ms = (time.perf_counter() - net_start) * 1000.0
                network_times.append(net_ms)
                if len(network_times) > 1000:
                    network_times.pop(0)

            now = time.time()
            if now - last_print_time >= print_interval:
                new_packets = packets_received - last_packet_count
                dt = now - last_print_time
                packet_rate = new_packets / dt if dt > 0 else 0.0

                debug_state = service.get_debug_state() if needs_debug else None
                if telemetry and last_command is not None:
                    actual_pos = np.array([
                        telemetry.measured_pose.x,
                        telemetry.measured_pose.y,
                        telemetry.measured_pose.z,
                    ])
                    actual_rot = telemetry.measured_pose.rot_y_deg
                    if args.perf:
                        _print_perf(debug_state, network_times, packet_rate, perf_only=True)
                    elif args.jac:
                        _print_normal(last_command, actual_pos, actual_rot, packet_rate, packets_received, debug_state)
                        _print_jacobian(telemetry, actual_pos, last_command, debug_state, jac_state)
                    else:
                        _print_normal(last_command, actual_pos, actual_rot, packet_rate, packets_received, debug_state)
                elif telemetry is None and not quiet:
                    stats = server.get_connection_stats()
                    if stats['is_connected']:
                        app_logger.info("Waiting for position data...")
                    else:
                        app_logger.warning("No data received (connection lost)")

                last_packet_count = packets_received
                last_print_time = now

            next_run_time += loop_period
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run_time = time.perf_counter()

    except KeyboardInterrupt:
        if not quiet:
            app_logger.info("Interrupted by user")
    except Exception as e:
        app_logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reset_to_normal(quiet=True)
        if not quiet:
            app_logger.info("Stopping controller...")
        service.stop()
        final_pos, final_rot_y = service.get_pose()
        if not quiet:
            app_logger.info(f"Final: [{final_pos[0]:+.3f}, {final_pos[1]:+.3f}, {final_pos[2]:+.3f}]m "
                            f"Rot: {final_rot_y:+.2f} | Packets: {packets_received}")


if __name__ == "__main__":
    main()
