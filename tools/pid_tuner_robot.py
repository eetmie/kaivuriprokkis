#!/usr/bin/env python3
"""Robot-side single-joint PID tuner.

Run this on the robot instead of excv_gui.py while tuning. It reads current IMU
joint angles, runs one PID controller for the selected joint, and sends one
named PWM command to the hydraulic valve. All non-selected valves are zeroed by
the hardware interface.

Client -> robot, 10 float32 values:
  [joint_id, target_deg, kp, ki, kd, enabled, pump_enabled, reload, reset_pid, max_output]

Robot -> client, 12 float32 values:
  [joint_id, slew_deg, boom_deg, arm_deg, bucket_deg, target_deg, output,
   kp, ki, kd, enabled, pump_enabled]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.excavator_ik_utils import canonical_joint_angles_from_imus
from modules.differential_ik_cfg import load_excavator_robot_config
from modules.hardware_interface import HardwareInterface
from modules.pid import PIDController
from modules.udp_socket import UDPSocket


COMMAND_SIZE = 10
TELEMETRY_SIZE = 12
JOINT_NAMES = ["slew", "boom", "arm", "bucket"]
PWM_NAMES = ["rotate", "lift_boom", "tilt_boom", "scoop"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _angle_error(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


def _safe_joint_id(value: float) -> int:
    return max(0, min(3, int(round(float(value)))))


def _read_joint_angles(hardware: HardwareInterface, robot_config) -> np.ndarray:
    quats = hardware.read_all_imu_quaternions()
    if quats is None:
        raise RuntimeError("IMU quaternions unavailable")
    return canonical_joint_angles_from_imus(np.asarray(quats, dtype=np.float32), robot_config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Robot-side PID tuner for one excavator joint")
    parser.add_argument("--host", default="192.168.0.132", help="Local robot IP to bind")
    parser.add_argument("--port", type=int, default=8090, help="UDP port; separate from excv_gui.py")
    parser.add_argument("--rate", type=float, default=50.0, help="Control loop rate Hz")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--config-file", default="configuration_files/servo_config_200.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("pid_tuner_robot")

    server = UDPSocket(local_id=12, max_age_seconds=0.5, data_format="f", nominal_rate_hz=args.rate)
    server.setup(
        args.host,
        args.port,
        num_inputs=COMMAND_SIZE,
        num_outputs=TELEMETRY_SIZE,
        is_server=True,
        data_format="f",
    )
    log.info("Waiting for tuner client on %s:%d", args.host, args.port)
    if not server.handshake(timeout=60.0):
        log.error("Handshake failed")
        return 1
    server.start_receiving()

    hardware = HardwareInterface(
        config_file=args.config_file,
        pump_auto_mode=True,
        cleanup_disable_osc=False,
        enable_adc=False,
        start_adc_reader=False,
        log_level=args.log_level,
    )
    robot_config = load_excavator_robot_config()

    log.info("Waiting for hardware")
    while not hardware.is_hardware_ready():
        time.sleep(0.1)
    log.info("Hardware ready. Tuner is idle until client enables output.")

    pid = PIDController(kp=0.0, ki=0.0, kd=0.0, min_output=-1.0, max_output=1.0)
    joint_id = 0
    target_deg = 0.0
    enabled = False
    pump_enabled = False
    last_gains = (0.0, 0.0, 0.0)
    last_joint_id = joint_id
    last_reset_flag = 0.0
    last_reload_flag = 0.0
    output = 0.0
    last_joint_angles: np.ndarray | None = None

    period = 1.0 / max(1.0, float(args.rate))
    next_t = time.perf_counter()
    last_status = time.time()

    try:
        while True:
            command = server.get_latest()
            if command is not None and len(command) >= COMMAND_SIZE:
                requested_joint_id = _safe_joint_id(command[0])
                requested_target_deg = _clamp(command[1], -180.0, 180.0)
                kp = _clamp(command[2], 0.0, 50.0)
                ki = _clamp(command[3], 0.0, 20.0)
                kd = _clamp(command[4], 0.0, 20.0)
                enabled = command[5] > 0.5
                requested_pump = command[6] > 0.5
                reload_flag = float(command[7])
                reset_flag = float(command[8])
                max_output = _clamp(command[9], 0.05, 1.0)
                joint_changed = requested_joint_id != last_joint_id
                joint_id = requested_joint_id
                target_deg = requested_target_deg

                gains = (kp, ki, kd)
                if gains != last_gains or joint_changed:
                    pid = PIDController(kp=kp, ki=ki, kd=kd, min_output=-max_output, max_output=max_output)
                    last_gains = gains
                    last_joint_id = joint_id
                else:
                    pid.min_output = -max_output
                    pid.max_output = max_output

                if joint_changed and last_joint_angles is not None:
                    target_deg = float(math.degrees(last_joint_angles[joint_id]))
                    pid.reset()

                if reset_flag > 0.5 and last_reset_flag <= 0.5:
                    pid.reset()
                last_reset_flag = reset_flag

                if reload_flag > 0.5 and last_reload_flag <= 0.5:
                    hardware.reload_config()
                last_reload_flag = reload_flag

                if requested_pump != pump_enabled:
                    pump_enabled = requested_pump
                    hardware.set_pump_enabled(pump_enabled)
            else:
                enabled = False
                if pump_enabled:
                    pump_enabled = False
                    hardware.set_pump_enabled(False)

            try:
                joint_angles = _read_joint_angles(hardware, robot_config)
                last_joint_angles = joint_angles.copy()
                current_rad = float(joint_angles[joint_id])
                target_rad = math.radians(target_deg)
                error = _angle_error(target_rad, current_rad)

                if enabled:
                    output = pid.compute(0.0, -error, dt=period)
                    hardware.send_named_pwm_commands({PWM_NAMES[joint_id]: output}, unset_to_zero=True)
                else:
                    output = 0.0
                    hardware.reset(reset_pump=False)

                telemetry = [
                    float(joint_id),
                    float(math.degrees(joint_angles[0])),
                    float(math.degrees(joint_angles[1])),
                    float(math.degrees(joint_angles[2])),
                    float(math.degrees(joint_angles[3])),
                    float(target_deg),
                    float(output),
                    float(pid.kp),
                    float(pid.ki),
                    float(pid.kd),
                    1.0 if enabled else 0.0,
                    1.0 if pump_enabled else 0.0,
                ]
                server.send(telemetry)
            except Exception as exc:
                log.warning("Loop read/control failed: %s", exc)
                hardware.reset(reset_pump=False)

            now = time.time()
            if now - last_status >= 1.0:
                log.info(
                    "%s target=%+.1f deg output=%+.3f enabled=%s pump=%s",
                    JOINT_NAMES[joint_id], target_deg, output, enabled, pump_enabled,
                )
                last_status = now

            next_t += period
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        try:
            hardware.reset(reset_pump=True)
            hardware.shutdown()
        finally:
            server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
