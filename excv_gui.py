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
from modules.server_diagnostics import ServerDiagnostics, DiagMode
from modules.rt_utils import apply_rt_to_thread, reset_to_normal, SCHED_FIFO


def _cpu_affinity(core):
    return None if core is None else {int(core)}


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

    # Determine diagnostics mode
    if args.perf:
        diag_mode = DiagMode.PERF
    elif args.jac:
        diag_mode = DiagMode.JACOBIAN
    else:
        diag_mode = DiagMode.NORMAL
    quiet = (diag_mode == DiagMode.PERF)

    diagnostics = ServerDiagnostics(diag_mode)

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
            args.fifo_priority,
            args.lock_memory,
            args.control_core,
            args.io_core,
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
    needs_debug = (diag_mode in (DiagMode.PERF, DiagMode.JACOBIAN))
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

            # Periodic status print
            now = time.time()
            if now - last_print_time >= print_interval:
                new_packets = packets_received - last_packet_count
                dt = now - last_print_time
                packet_rate = new_packets / dt if dt > 0 else 0.0

                debug_state = service.get_debug_state() if needs_debug else None
                if telemetry:
                    diagnostics.print_status(
                        telemetry, last_command, debug_state,
                        network_times, packet_rate, packets_received,
                        perf_only=args.perf,
                    )
                elif not quiet:
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
