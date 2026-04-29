#!/usr/bin/env python3
"""Bench driving helper — same UDP/gamepad input as drive_logger.py, no recording.

Usage:
    sudo python simple_drive.py

Buttons (remote gamepad, same wire format as drive_logger.py):
    Button 0: toggle live joint-angle print (abs + rel)
    Button 2: toggle hydraulic pump

The controller runs in DIRECT mode — joystick axes go straight to valves,
IK/PID are bypassed. Useful for sanity-checking joint readouts (e.g., after
removing the slew joint limit) without touching the data-logger codepath.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.udp_socket import UDPSocket
from modules.hardware_interface import HardwareInterface, HardwareFaultError
from modules.excavator_controller import ExcavatorController


SAMPLING_FREQUENCY = 100              # main loop Hz
PRINT_DECIMATION = 10                 # print every Nth iteration when enabled (~10Hz)
JOINT_NAMES = ['rotate', 'lift_boom', 'tilt_boom', 'scoop']


def main():
    # ---- UDP (mirror drive_logger.py wire format exactly) ----
    server = UDPSocket(local_id=2)
    server.setup("192.168.0.132", 8080, num_inputs=10, num_outputs=0, is_server=True)

    # ---- Hardware ----
    print("Initializing hardware...")
    hardware = HardwareInterface(
        pump_auto_mode=False,
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
    print("Controller in DIRECT mode.")

    # ---- UDP handshake ----
    print("Waiting for remote controller...")
    if not server.handshake(timeout=30.0):
        print("UDP handshake failed.")
        controller.exit_direct_mode()
        controller.stop()
        hardware.shutdown()
        raise SystemExit(1)
    server.start_receiving()
    print("Connected. Drive with the gamepad. Button 0 toggles joint-angle print.\n")

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

                # Button 0: toggle joint-angle printing.
                if buttons[0] > button_threshold and button_prev[0] <= button_threshold:
                    print_angles = not print_angles
                    state = "ON" if print_angles else "OFF"
                    print(f"\n[Button 0] joint-angle print {state}")

                # Button 2: pump toggle (handy on the bench).
                if buttons[2] > button_threshold and button_prev[2] <= button_threshold:
                    if hardware.pwm_controller is not None:
                        new_state = not hardware.pwm_controller.pump_enabled
                        hardware.pwm_controller.set_pump_enabled(new_state)
                        print(f"\n[Button 2] pump {'ON' if new_state else 'OFF'}")

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
            controller.give_direct_commands(commands)

            # Live joint-angle readout — only when toggled on, decimated to ~10Hz.
            iter_count += 1
            if print_angles and (iter_count % PRINT_DECIMATION == 0):
                rel = controller.get_joint_angles()              # degrees, parent-relative
                abs_a = controller.get_absolute_link_angles()    # degrees, cab-frame cumulative
                line = (
                    f"[REL] slew={rel[0]:+7.2f}  boom={rel[1]:+7.2f}  "
                    f"arm={rel[2]:+7.2f}  bucket={rel[3]:+7.2f} | "
                    f"[ABS] slew={abs_a[0]:+7.2f}  boom={abs_a[1]:+7.2f}  "
                    f"arm={abs_a[2]:+7.2f}  bucket={abs_a[3]:+7.2f} deg"
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
