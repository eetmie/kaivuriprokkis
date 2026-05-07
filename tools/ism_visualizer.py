"""
ISM330 IMU Sensor — Live 3D ASCII Cube Visualizer

Reads quaternion data from the Pico over USB serial (binary protocol)
and renders a rotating cube in the terminal that follows the sensor's
orientation in real time.

The first stable reading is captured as the "zero" position
(like pressing tare on a scale), so all rotations shown
are relative to where the sensor was when you started.

Usage:
    1. Place sensor flat on table
    2. Run: python -m tools.ism_visualizer
       (or: python tools/ism_visualizer.py from repo root)
    3. Tilt the sensor and watch the cube follow!
    4. Ctrl+C to stop
"""

import argparse
import math
import os
import select
import sys
import time

import numpy as np

from tools.ascii_cube import render_frame, clear_screen
from modules.usb_serial_reader import USBSerialReader
from modules.quaternion_math import (
    quat_normalize,
    quat_conjugate,
    quat_multiply,
    quat_enforce_hemisphere,
    quat_remove_yaw,
    quat_slerp,
    euler_xyz_from_quat,
)


# ============================================================
# Main
# ============================================================

WARMUP_FRAMES = 100  # let the firmware AHRS settle (~0.5 s at 200 Hz)


class KeyPoller:
    def __enter__(self):
        self._restore = None
        if os.name != 'nt' and sys.stdin.isatty():
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)

            def restore():
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

            self._restore = restore
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._restore is not None:
            self._restore()

    def poll(self):
        if os.name == 'nt':
            import msvcrt

            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ('\x00', '\xe0') and msvcrt.kbhit():
                    msvcrt.getwch()
                    continue
                return ch.lower()
            return None

        if not sys.stdin.isatty():
            return None

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            return sys.stdin.read(1).lower()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="ISM330 IMU — Live 3D ASCII Cube Visualizer"
    )

    # Serial
    parser.add_argument('--port', default=None, help='Serial port (auto-detect if omitted)')
    parser.add_argument('--baud', type=int, default=115200, help='Serial baud rate')

    # Visualizer options
    parser.add_argument('--no-yaw', action='store_true',
                        help='Zero out yaw (Z-rotation) to diagnose drift')
    parser.add_argument('--smooth', type=float, default=0.0,
                        help='Output smoothing 0.0-0.95 (SLERP low-pass). 0=off, 0.3=light, 0.6=medium, 0.9=heavy')
    parser.add_argument('--imu', type=int, default=0,
                        help='Which IMU index to visualize (default: 0)')

    parser.add_argument('--debug', action='store_true', help='Print raw serial data')

    args = parser.parse_args()

    print('\033[2J')  # clear terminal
    print("=== ISM330 IMU — Live 3D Visualizer ===")
    if args.no_yaw:
        print("** --no-yaw active: yaw is zeroed out **")
    if args.smooth > 0.0:
        print(f"Smoothing: {args.smooth:.2f} (SLERP low-pass)")
    print(f"Visualizing IMU index {args.imu}")
    print(f"Warming up ({WARMUP_FRAMES} frames)...")

    reader = USBSerialReader(
        baud_rate=args.baud,
        port=args.port,
        debug=args.debug,
    )

    zero_q_inv = None
    prev_q = None
    smooth_q = None
    frame_count = 0
    status_line = "Press R to re-zero"

    try:
        with KeyPoller() as key_poller:
            while True:
                data = reader.read_imus()
                if data is None:
                    time.sleep(0.001)
                    continue

                # Pick the requested IMU
                if args.imu >= len(data):
                    time.sleep(0.001)
                    continue

                imu = data[args.imu]
                q = np.array([imu[0], imu[1], imu[2], imu[3]], dtype=np.float32)
                q = quat_normalize(q)

                # Keep quaternion hemisphere consistent
                if prev_q is not None:
                    q = quat_enforce_hemisphere(q, prev_q)
                prev_q = q

                frame_count += 1

                # Let firmware AHRS settle
                if frame_count < WARMUP_FRAMES:
                    if frame_count % 10 == 0:
                        sys.stdout.write(f"\r  Warmup: {frame_count}/{WARMUP_FRAMES}")
                        sys.stdout.flush()
                    continue

                key = key_poller.poll()
                if key == 'r':
                    zero_q_inv = quat_conjugate(q)
                    smooth_q = None
                    status_line = "Re-zeroed current pose"

                # Capture zero position on first post-warmup reading
                if zero_q_inv is None:
                    zero_q_inv = quat_conjugate(q)
                    print(f"\n  Zero captured: W={q[0]:+.4f} X={q[1]:+.4f} Y={q[2]:+.4f} Z={q[3]:+.4f}")
                    print("  Tilt the sensor and watch the cube follow!")
                    print("  Press R to re-zero. Ctrl+C to stop.\n")
                    time.sleep(0.3)
                    continue

                # Relative orientation must be q0^-1 * q, not q * q0^-1.
                q_corrected = quat_normalize(quat_multiply(zero_q_inv, q))

                # Optionally remove yaw to diagnose drift
                if args.no_yaw:
                    q_corrected = quat_remove_yaw(q_corrected)

                # Output smoothing (SLERP low-pass)
                if args.smooth > 0.0 and smooth_q is not None:
                    q_corrected = quat_slerp(q_corrected, smooth_q, args.smooth)
                smooth_q = q_corrected

                # Get Euler angles for the readout
                roll, pitch, yaw = euler_xyz_from_quat(q_corrected)
                roll_deg = math.degrees(float(roll))
                pitch_deg = math.degrees(float(pitch))
                yaw_deg = math.degrees(float(yaw))

                # Render the cube (ascii_cube accepts any indexable [w,x,y,z])
                frame = render_frame(q_corrected, roll_deg, pitch_deg, yaw_deg)
                clear_screen()
                print(frame)

                st = reader.status()
                sps_str = f"{st['sps']:.0f}"
                if st['target_sps']:
                    sps_str += f"/{st['target_sps']}"
                print(f"  SPS: {sps_str} Hz  |  IMU: {args.imu}  |  {status_line}")

    except KeyboardInterrupt:
        pass

    finally:
        print(f"\nProcessed {frame_count} frames.")
        reader.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
