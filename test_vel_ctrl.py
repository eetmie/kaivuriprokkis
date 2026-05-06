#!/usr/bin/env python3
"""Direct-drive / joint velocity-PI prototype test.

Uses the same UDP/gamepad mapping as simple_drive.py. Button 3 cycles between
raw valve control, closed-loop velocity control, and velocity-assisted raw
control for gyro-backed joints. Slew and tracks remain raw in all modes.

Usage:
    sudo python test_vel_ctrl.py
"""

from __future__ import annotations

import sys
import time
import math
import argparse
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.udp_socket import UDPSocket
from modules.hardware_interface import HardwareInterface, HardwareFaultError
from modules.excavator_controller import ExcavatorController
from modules.pid import PIDController

SAMPLING_FREQUENCY = 100
LOOP_PERIOD = 1.0 / SAMPLING_FREQUENCY

# Velocity PI: joystick command -> desired deg/s -> PI -> valve command.
# Slew (rotate) has no gyro feedback channel here, so it stays open-loop.
VEL_CTRL_JOINTS = {'lift_boom': 1, 'tilt_boom': 2, 'scoop': 3}
CONTROL_MODES = ('RAW', 'VEL', 'ASSIST')
ASSIST_MAX_TRIM = 0.80
VELOCITY_MAX_DEGPS = 80.0
VELOCITY_KI_MAX = 0.05

# Limits target velocity step changes before the PI sees them. This fights the
# large initial valve kick when slamming the joystick out of stiction/deadband.
VEL_TARGET_SLEW_DEGPS_PER_S = 1000.0

# Per-joint, per-joystick-direction velocity feed-forward gains.
# Format: joint_name: (positive_joystick_gain, negative_joystick_gain)
VELOCITY_FEED_FORWARD = {
    'lift_boom': (0.0, 0.0),
    'tilt_boom': (0.0, 0.0),
    'scoop': (0.0, 0.0),
}

# Per-joint, per-direction PID gains. D stays zero for now.
# Format: joint_name: {'pos': (kp, ki), 'neg': (kp, ki)}
VELOCITY_PID_GAINS = {
    'lift_boom': {'pos': (0.020, 0.01), 'neg': (0.020, 0.01)},
    'tilt_boom': {'pos': (0.015, 0.015), 'neg': (0.015, 0.01)},
    'scoop': {'pos': (0.012, 0.01), 'neg': (0.012, 0.015)},
}


class JointVelocityController:
    def __init__(
        self,
        max_degps: float,
        ff: dict[str, tuple[float, float]],
    ):
        self.max_degps = max_degps
        self.ff = ff
        self.target_dps: dict[str, float] = {name: 0.0 for name in VEL_CTRL_JOINTS}
        self.active_dir: dict[str, str | None] = {name: None for name in VEL_CTRL_JOINTS}
        self.pid: dict[str, dict[str, PIDController]] = {}
        for name in VEL_CTRL_JOINTS:
            gains = VELOCITY_PID_GAINS.get(name, {})
            self.pid[name] = {
                'pos': self._make_pid(gains.get('pos', (0.0, 0.0)), min_output=0.0, max_output=1.0),
                'neg': self._make_pid(gains.get('neg', (0.0, 0.0)), min_output=-1.0, max_output=0.0),
            }

    def _make_pid(
        self,
        gains: tuple[float, float],
        *,
        min_output: float,
        max_output: float,
    ) -> PIDController:
        kp, ki = gains
        return PIDController(
            kp=kp,
            ki=ki,
            kd=0.0,
            min_output=min_output,
            max_output=max_output,
            Imin=-VELOCITY_KI_MAX,
            Imax=VELOCITY_KI_MAX,
        )

    def apply(
        self,
        commands: dict[str, float],
        joint_vels_degps,
        dt: float,
        *,
        assist: bool = False,
    ) -> dict[str, float]:
        out = dict(commands)
        for name, vel_idx in VEL_CTRL_JOINTS.items():
            if vel_idx >= len(joint_vels_degps):
                continue
            joystick_cmd = float(commands.get(name, 0.0))
            target = joystick_cmd * self.max_degps
            if joystick_cmd == 0.0:
                self._reset_joint(name)
                out[name] = 0.0
                continue
            direction = 'pos' if joystick_cmd > 0.0 else 'neg'
            if self.active_dir[name] != direction:
                self._reset_joint(name)
                self.active_dir[name] = direction

            actual = float(joint_vels_degps[vel_idx])
            desired = self._slew_target(name, target, dt)
            feedback = self.pid[name][direction].compute(desired, actual, dt=dt)
            ff_pos, ff_neg = self.ff.get(name, (0.0, 0.0))
            feed_forward = (ff_pos if direction == 'pos' else ff_neg) * joystick_cmd
            if assist:
                feedback = float(np.clip(feedback, -ASSIST_MAX_TRIM, ASSIST_MAX_TRIM))
                valve_cmd = joystick_cmd + feedback
            else:
                valve_cmd = feed_forward + feedback
            valve_cmd = self._apply_direction_gate(valve_cmd, joystick_cmd)
            out[name] = float(np.clip(valve_cmd, -1.0, 1.0))
        return out

    def _reset_joint(self, name: str) -> None:
        for pid in self.pid[name].values():
            pid.reset()
        self.target_dps[name] = 0.0
        self.active_dir[name] = None

    def _apply_direction_gate(self, valve_cmd: float, joystick_cmd: float) -> float:
        joystick_sign = math.copysign(1.0, joystick_cmd)
        # Keep active velocity commands on the user's requested valve side.
        # If PI tries to cross zero, gate it to zero instead of briefly commanding
        # the opposite side and re-entering the long deadband/stiction region.
        return joystick_sign * max(0.0, valve_cmd * joystick_sign)

    def _slew_target(self, name: str, target: float, dt: float) -> float:
        current = self.target_dps[name]
        max_step = VEL_TARGET_SLEW_DEGPS_PER_S * max(float(dt), 0.0)
        delta = float(np.clip(target - current, -max_step, max_step))
        current += delta
        self.target_dps[name] = current
        return current

    def reset(self) -> None:
        for name in VEL_CTRL_JOINTS:
            self._reset_joint(name)


def parse_args():
    parser = argparse.ArgumentParser(description="Raw direct-drive / velocity-PI prototype test")
    return parser.parse_args()


def main():
    args = parse_args()

    server = UDPSocket(local_id=2)
    server.setup("192.168.0.132", 8080, num_inputs=10, num_outputs=0, is_server=True)

    print("Initializing hardware...")
    hardware = HardwareInterface(
        pump_auto_mode=False,
        toggle_channels=True,
        stale_timeout_s=0.5,
        enable_pwm=True,
        enable_imu=True,
        enable_adc=False,
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

    print("Starting controller...")
    controller = ExcavatorController(hardware, config=None, enable_perf_tracking=False)
    controller.start()
    time.sleep(2.0)
    controller.enter_direct_mode()
    controller.set_velocity_mode('gyro_only')
    print("Controller in DIRECT mode (velocity feedback: gyro_only).")

    print("Waiting for remote controller...")
    if not server.handshake(timeout=30.0):
        print("UDP handshake failed.")
        controller.exit_direct_mode()
        controller.stop()
        hardware.shutdown()
        raise SystemExit(1)
    server.start_receiving()
    print(
        "Connected. Same controls as simple_drive.py. "
        "Button 3 cycles RAW / VEL / ASSIST mode.\n"
    )

    vel_controller = JointVelocityController(
        max_degps=VELOCITY_MAX_DEGPS,
        ff=VELOCITY_FEED_FORWARD,
    )
    control_mode = 0

    next_run_time = time.perf_counter()
    prev_loop_time = time.perf_counter()
    right_rl = right_ud = left_rl = left_ud = 0.0
    right_paddle = left_paddle = 0.0
    button_prev = [0.0, 0.0, 0.0, 0.0]
    button_threshold = 0.5

    try:
        while True:
            now = time.perf_counter()
            actual_dt = now - prev_loop_time
            prev_loop_time = now

            float_data = server.get_latest_floats()
            if float_data:
                right_rl = float_data[9]   # scoop
                right_ud = float_data[8]   # lift
                left_rl = float_data[7]    # rotate/slew
                left_ud = float_data[6]    # tilt/arm
                right_paddle   = float_data[5]
                left_paddle    = float_data[4]
                buttons = [float_data[0], float_data[1], float_data[2], float_data[3]]

                if buttons[1] > button_threshold and button_prev[1] <= button_threshold:
                    ok = hardware.reload_config()
                    print(f"\n[Button 1] reload config {'OK' if ok else 'FAILED'}")
                if buttons[2] > button_threshold and button_prev[2] <= button_threshold:
                    if hardware.pwm_controller is not None:
                        new_state = not hardware.pwm_controller.pump_enabled
                        hardware.pwm_controller.set_pump_enabled(new_state)
                        print(f"\n[Button 2] pump {'ON' if new_state else 'OFF'}")
                if buttons[3] > button_threshold and button_prev[3] <= button_threshold:
                    control_mode = (control_mode + 1) % len(CONTROL_MODES)
                    vel_controller.reset()
                    print(f"\n[Button 3] control mode: {CONTROL_MODES[control_mode]}")
                button_prev = buttons

            commands = {
                'rotate':    left_rl,
                'lift_boom': right_ud,
                'tilt_boom': left_ud,
                'scoop':     right_rl,
                'trackR':    right_paddle,
                'trackL':    left_paddle,
            }

            joint_vels, vel_age = controller.get_joint_velocities_with_age()
            vel_stale = vel_age > 0.05
            mode_label = CONTROL_MODES[control_mode]
            if mode_label in ('VEL', 'ASSIST'):
                if joint_vels is not None and not vel_stale:
                    commands = vel_controller.apply(
                        commands,
                        joint_vels,
                        actual_dt,
                        assist=(mode_label == 'ASSIST'),
                    )
                else:
                    vel_controller.reset()
                    for name in VEL_CTRL_JOINTS:
                        commands[name] = 0.0

            controller.give_direct_commands(commands)

            stale_flag = " STALE" if vel_stale else "      "
            lift_cmd = commands['lift_boom']
            arm_cmd = commands['tilt_boom']
            bucket_cmd = commands['scoop']
            print(
                f"mode={mode_label}  vel_age={vel_age:.3f}s{stale_flag}  "
                f"cmd rotate={commands['rotate']:+.3f} lift={lift_cmd:+.3f} "
                f"arm={arm_cmd:+.3f} bucket={bucket_cmd:+.3f} "
                f"trackR={commands['trackR']:+.3f} trackL={commands['trackL']:+.3f}",
                end="\r", flush=True,
            )

            next_run_time += LOOP_PERIOD
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        print("Shutting down...")
        try:
            controller.give_direct_commands({})
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
