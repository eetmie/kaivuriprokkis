#!/usr/bin/env python3
"""DOES NOT WORK ATM, VERY WIP!!!
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.differential_ik_cfg import load_excavator_robot_config
from modules.excavator_ik_utils import canonical_joint_angles_from_imus
from modules.hardware_interface import HardwareInterface
from modules.udp_socket import UDPSocket


# Tune these after a couple of real runs.
DEADBAND_BACKOFF = 0.80  # Use 80% of first-moving physical pulse offset.
MAX_PROBE_COMMAND = 0.60
COARSE_STEP_US = 25.0
FINE_STEP_US = 5.0
PROBE_DWELL_S = 0.70
PROBE_RATE_HZ = 50.0
SETTLE_S = 0.25
MOTION_DELTA_DEG = 0.45
MOTION_VEL_MIN_DELTA_DEG = 0.15
MOTION_VEL_DEG_S = 1.0
MAX_JOINT_DELTA_DEG = 20.0
CONFIRM_ATTEMPTS = 4
RETURN_COMMAND = 0.12
RETURN_OFFSET_GAIN = 1.15
RETURN_MAX_COMMAND = 0.45
RETURN_TOLERANCE_DEG = 2.0
RETURN_TIMEOUT_S = 8.0
UDP_HOST = "192.168.0.132"
UDP_PORT = 8080
UDP_POSITION_HZ = 100.0
UDP_PRINT_DECIMATION = 10
BUTTON_THRESHOLD = 0.5

BASE_CONFIG = ROOT_DIR / "configuration_files" / "servo_config_base.yaml"
OUTPUT_DIR = ROOT_DIR / "configuration_files"
CHANNEL_ORDER = ["scoop", "tilt_boom", "lift_boom", "rotate"]
JOINT_BY_CHANNEL = {
    "rotate": (0, "slew"),
    "lift_boom": (1, "boom"),
    "tilt_boom": (2, "arm"),
    "scoop": (3, "bucket"),
}


@dataclass(frozen=True)
class ProbeResult:
    moved: bool
    max_delta_deg: float
    max_vel_deg_s: float
    final_delta_deg: float


@dataclass(frozen=True)
class DirectionResult:
    key: str
    command_sign: int
    movement_sign: int
    first_move_offset_us: float
    safe_offset_us: float


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} did not contain a YAML mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def _valve_channels(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    channels = config.get("CHANNEL_CONFIGS", {})
    if not isinstance(channels, dict):
        raise RuntimeError("CHANNEL_CONFIGS missing or not a mapping")
    return {
        name: cfg
        for name, cfg in channels.items()
        if name != "pump" and isinstance(cfg, dict)
    }


def _candidate_from_base(base: dict[str, Any]) -> dict[str, Any]:
    channels = base.get("CHANNEL_CONFIGS", {})
    if not isinstance(channels, dict):
        raise RuntimeError("Base config CHANNEL_CONFIGS missing or not a mapping")

    out: dict[str, Any] = {
        "pwm_frequency": base.get("pwm_frequency", 200),
        "CHANNEL_CONFIGS": {},
    }
    out_channels = out["CHANNEL_CONFIGS"]

    for name, cfg in channels.items():
        if not isinstance(cfg, dict):
            continue
        if name == "pump":
            out_channels[name] = dict(cfg)
            continue

        out_channels[name] = {
            "output_channel": cfg["output_channel"],
            "affects_pump": bool(cfg.get("affects_pump", False)),
            "toggleable": bool(cfg.get("toggleable", False)),
            "center": float(cfg["center"]),
            "direction": int(cfg["direction"]),
            "pulse_min": int(cfg["pulse_min"]),
            "pulse_max": int(cfg["pulse_max"]),
            "deadzone": 0.0,
            "deadband_us_pos": 0.0,
            "deadband_us_neg": 0.0,
            "dither_enable": bool(cfg.get("dither_enable", False)),
            "dither_amp_us": float(cfg.get("dither_amp_us", 0.0)),
            "dither_hz": float(cfg.get("dither_hz", 0.0)),
            "dither_taper": bool(cfg.get("dither_taper", False)),
            "dither_taper_us": float(cfg.get("dither_taper_us", 0.0)),
            "ramp_enable": bool(cfg.get("ramp_enable", False)),
            "ramp_limit": float(cfg.get("ramp_limit", 0.0)),
            "ramp_skip_deadband": bool(cfg.get("ramp_skip_deadband", False)),
            "gamma": 1.0,
        }
    return out


def _read_joint_angles(hardware: HardwareInterface, robot_config) -> np.ndarray:
    quats = hardware.read_all_imu_quaternions()
    if quats is None:
        raise RuntimeError("IMU quaternions unavailable")
    return canonical_joint_angles_from_imus(np.asarray(quats, dtype=np.float32), robot_config)


def _direction_key(channel_cfg: dict[str, Any], command_sign: int) -> str:
    physical_sign = float(command_sign) * float(channel_cfg["direction"])
    return "deadband_us_pos" if physical_sign > 0.0 else "deadband_us_neg"


def _command_for_pulse_offset(channel_cfg: dict[str, Any], pulse_offset: float, command_sign: int) -> float:
    center = float(channel_cfg["center"])
    physical_sign = float(command_sign) * float(channel_cfg["direction"])
    if physical_sign > 0.0:
        span = max(1e-6, float(channel_cfg["pulse_max"]) - center)
    else:
        span = max(1e-6, center - float(channel_cfg["pulse_min"]))
    return max(0.0, min(1.0, float(pulse_offset) / span))


def _max_probe_offset(channel_cfg: dict[str, Any], command_sign: int) -> float:
    center = float(channel_cfg["center"])
    physical_sign = float(command_sign) * float(channel_cfg["direction"])
    if physical_sign > 0.0:
        span = max(0.0, float(channel_cfg["pulse_max"]) - center)
    else:
        span = max(0.0, center - float(channel_cfg["pulse_min"]))
    return span * MAX_PROBE_COMMAND


def _round_up_step(value: float, step: float) -> float:
    if value <= 0.0:
        return float(step)
    return float(np.ceil(float(value) / float(step)) * float(step))


def _round_down_step(value: float, step: float) -> float:
    return float(np.floor(max(0.0, float(value)) / float(step)) * float(step))


class ValveTuner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.log = logging.getLogger("valve_tuning_helper")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path = OUTPUT_DIR / f"valve_config_auto_{timestamp}.yaml"
        base_path = Path(args.base_config)
        if not base_path.is_absolute():
            base_path = ROOT_DIR / base_path
        self.base_config = _load_yaml(base_path)
        self.candidate = _candidate_from_base(self.base_config)
        self.probe_channels = _valve_channels(self.candidate)
        self.base_channels = _valve_channels(self.base_config)
        self.udp: UDPSocket | None = None
        _write_yaml(self.output_path, self.candidate)

        self.robot_config = load_excavator_robot_config()
        self.hardware = HardwareInterface(
            config_file=str(self.output_path),
            pump_auto_mode=False,
            cleanup_disable_osc=False,
            enable_adc=False,
            start_adc_reader=False,
            log_level=args.log_level,
        )
        if args.udp_positioning:
            self.udp = UDPSocket(local_id=2, max_age_seconds=0.5, nominal_rate_hz=UDP_POSITION_HZ)
            self.udp.setup(args.host, args.port, num_inputs=10, num_outputs=0, is_server=True)

    def wait_ready(self) -> None:
        self.log.info("Waiting for hardware")
        while not self.hardware.is_hardware_ready():
            time.sleep(0.1)
        self.hardware.set_pump_auto(False)
        self.hardware.set_pump_enabled(True)
        self.log.info("Hardware ready; fixed pump enabled")

    def shutdown(self) -> None:
        try:
            self.hardware.reset(reset_pump=True)
        finally:
            if self.udp is not None:
                self.udp.close()
            self.hardware.shutdown()

    def start_udp_positioning(self) -> None:
        if self.udp is None:
            return
        self.log.info("Waiting for UDP gamepad on %s:%d", self.args.host, self.args.port)
        if not self.udp.handshake(timeout=30.0):
            raise RuntimeError("UDP positioning handshake failed")
        self.udp.start_receiving()
        self.log.info("UDP gamepad connected; Button 3 accepts the current pose")

    def manual_position(self, channel: str, joint_index: int, joint_name: str) -> float:
        if self.udp is None:
            input(
                f"\nPlace {joint_name}/{channel} in a safe pose with +/-{MAX_JOINT_DELTA_DEG:.0f} deg room, "
                "then press Enter..."
            )
            return float(_read_joint_angles(self.hardware, self.robot_config)[joint_index])

        print(
            f"\nDrive {joint_name}/{channel} to a safe pose. "
            "Button 3 accepts pose, Button 0 toggles angle print."
        )
        print_angles = False
        button_prev = [0.0, 0.0, 0.0, 0.0]
        iter_count = 0
        loop_period = 1.0 / UDP_POSITION_HZ
        next_t = time.perf_counter()

        while True:
            float_data = self.udp.get_latest_floats()
            if float_data:
                buttons = [float_data[0], float_data[1], float_data[2], float_data[3]]
                if buttons[0] > BUTTON_THRESHOLD and button_prev[0] <= BUTTON_THRESHOLD:
                    print_angles = not print_angles
                    print(f"\n[Button 0] angle print {'ON' if print_angles else 'OFF'}")

                if buttons[2] > BUTTON_THRESHOLD and button_prev[2] <= BUTTON_THRESHOLD:
                    if self.hardware.pwm_controller is not None:
                        new_state = not self.hardware.pwm_controller.pump_enabled
                        self.hardware.pwm_controller.set_pump_enabled(new_state)
                        print(f"\n[Button 2] pump {'ON' if new_state else 'OFF'}")

                if buttons[3] > BUTTON_THRESHOLD and button_prev[3] <= BUTTON_THRESHOLD:
                    self.hardware.reset(reset_pump=False)
                    angle = float(_read_joint_angles(self.hardware, self.robot_config)[joint_index])
                    print(f"\n[Button 3] accepted {joint_name} start angle {np.degrees(angle):+.2f} deg")
                    return angle

                button_prev = buttons
                commands = {
                    "rotate": float_data[7],
                    "lift_boom": float_data[8],
                    "tilt_boom": float_data[6],
                    "scoop": float_data[9],
                }
                self.hardware.send_named_pwm_commands(commands, unset_to_zero=True)
            else:
                self.hardware.reset(reset_pump=False)

            iter_count += 1
            if print_angles and iter_count % UDP_PRINT_DECIMATION == 0:
                angles = np.degrees(_read_joint_angles(self.hardware, self.robot_config))
                print(
                    f"slew={angles[0]:+7.2f} boom={angles[1]:+7.2f} "
                    f"arm={angles[2]:+7.2f} bucket={angles[3]:+7.2f}",
                    end="\r",
                    flush=True,
                )

            next_t += loop_period
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()

    def probe_command(self, channel: str, command: float, joint_index: int) -> ProbeResult:
        self.hardware.reset(reset_pump=False)
        time.sleep(SETTLE_S)
        start_angle_rad = float(_read_joint_angles(self.hardware, self.robot_config)[joint_index])
        period = 1.0 / PROBE_RATE_HZ
        end_t = time.monotonic() + PROBE_DWELL_S
        prev_angle = start_angle_rad
        prev_t = time.monotonic()
        max_delta_deg = 0.0
        max_vel_deg_s = 0.0
        final_delta_deg = 0.0
        moved = False

        while time.monotonic() < end_t:
            self.hardware.send_named_pwm_commands({channel: command}, unset_to_zero=True)
            angles = _read_joint_angles(self.hardware, self.robot_config)
            now = time.monotonic()
            angle = float(angles[joint_index])
            signed_delta_deg = float(np.degrees(np.arctan2(np.sin(angle - start_angle_rad), np.cos(angle - start_angle_rad))))
            delta_deg = abs(signed_delta_deg)
            final_delta_deg = signed_delta_deg
            step_delta = np.arctan2(np.sin(angle - prev_angle), np.cos(angle - prev_angle))
            dt = max(1e-6, now - prev_t)
            vel_deg_s = abs(float(np.degrees(step_delta / dt)))
            max_delta_deg = max(max_delta_deg, float(delta_deg))
            max_vel_deg_s = max(max_vel_deg_s, vel_deg_s)
            if delta_deg >= MAX_JOINT_DELTA_DEG:
                self.hardware.reset(reset_pump=False)
                raise RuntimeError(
                    f"{channel}: exceeded {MAX_JOINT_DELTA_DEG:.1f} deg max delta while probing; reposition and retry"
                )
            if delta_deg >= MOTION_DELTA_DEG or (
                delta_deg >= MOTION_VEL_MIN_DELTA_DEG and vel_deg_s >= MOTION_VEL_DEG_S
            ):
                moved = True
            prev_angle = angle
            prev_t = now
            time.sleep(period)

        self.hardware.reset(reset_pump=False)
        time.sleep(SETTLE_S)
        return ProbeResult(
            moved=moved,
            max_delta_deg=max_delta_deg,
            max_vel_deg_s=max_vel_deg_s,
            final_delta_deg=final_delta_deg,
        )

    def probe_offset(self, channel: str, offset_us: float, command_sign: int, joint_index: int) -> ProbeResult:
        command_abs = _command_for_pulse_offset(self.probe_channels[channel], offset_us, command_sign)
        return self.probe_command(channel, float(command_sign) * command_abs, joint_index)

    def _probe_and_log_offset(self, channel: str, offset_us: float, command_sign: int, joint_index: int) -> ProbeResult:
        command_abs = _command_for_pulse_offset(self.probe_channels[channel], offset_us, command_sign)
        result = self.probe_offset(channel, offset_us, command_sign, joint_index)
        self.log.info(
            "%s offset=%+.1fus cmd=%+.3f moved=%s delta=%.2fdeg final=%+.2fdeg vel=%.2fdeg/s",
            channel,
            float(command_sign) * float(offset_us),
            float(command_sign) * command_abs,
            result.moved,
            result.max_delta_deg,
            result.final_delta_deg,
            result.max_vel_deg_s,
        )
        return result

    def find_first_moving_offset(self, channel: str, command_sign: int, joint_index: int) -> tuple[float, int] | None:
        base_cfg = self.base_channels[channel]
        key = _direction_key(self.probe_channels[channel], command_sign)
        baseline = float(base_cfg.get(key, FINE_STEP_US))
        max_offset = _round_down_step(_max_probe_offset(self.probe_channels[channel], command_sign), FINE_STEP_US)
        if max_offset < FINE_STEP_US:
            return None

        high = min(max_offset, _round_up_step(max(FINE_STEP_US, baseline), FINE_STEP_US))
        low = 0.0

        result = self._probe_and_log_offset(channel, high, command_sign, joint_index)
        high_result = result
        if not result.moved:
            low = high
            while high < max_offset:
                high = min(max_offset, _round_up_step(high + COARSE_STEP_US, FINE_STEP_US))
                result = self._probe_and_log_offset(channel, high, command_sign, joint_index)
                if result.moved:
                    high_result = result
                    break
                low = high
            if not result.moved:
                return None

        low_i = int(round(low / FINE_STEP_US))
        high_i = int(round(high / FINE_STEP_US))
        while high_i - low_i > 1:
            mid_i = (low_i + high_i) // 2
            offset = float(mid_i) * FINE_STEP_US
            result = self._probe_and_log_offset(channel, offset, command_sign, joint_index)
            if result.moved:
                high_i = mid_i
                high_result = result
            else:
                low_i = mid_i
        movement_sign = int(np.sign(high_result.final_delta_deg))
        if movement_sign == 0:
            movement_sign = int(np.sign(command_sign))
        return float(high_i) * FINE_STEP_US, movement_sign

    def confirm_no_motion(self, channel: str, offset_us: float, command_sign: int, joint_index: int) -> bool:
        command_abs = _command_for_pulse_offset(self.probe_channels[channel], offset_us, command_sign)
        result = self.probe_offset(channel, offset_us, command_sign, joint_index)
        self.log.info(
            "%s confirm offset=%+.1fus cmd=%+.3f moved=%s delta=%.2fdeg final=%+.2fdeg vel=%.2fdeg/s",
            channel,
            float(command_sign) * float(offset_us),
            float(command_sign) * command_abs,
            result.moved,
            result.max_delta_deg,
            result.final_delta_deg,
            result.max_vel_deg_s,
        )
        return not result.moved

    def tune_direction(self, channel: str, command_sign: int, joint_index: int) -> DirectionResult | None:
        channel_cfg = self.probe_channels[channel]
        key = _direction_key(channel_cfg, command_sign)
        first_moving = self.find_first_moving_offset(channel, command_sign, joint_index)
        if first_moving is None:
            self.log.warning("%s %s: no movement found up to %.1fus", channel, key, _max_probe_offset(channel_cfg, command_sign))
            return None
        first_moving_offset, movement_sign = first_moving

        candidate_offset = first_moving_offset * DEADBAND_BACKOFF
        confirmed = False
        for _ in range(CONFIRM_ATTEMPTS):
            candidate_offset = _round_down_step(candidate_offset, FINE_STEP_US)
            if candidate_offset <= 0.0:
                candidate_offset = FINE_STEP_US
            if self.confirm_no_motion(channel, candidate_offset, command_sign, joint_index):
                confirmed = True
                break
            candidate_offset *= DEADBAND_BACKOFF
        if not confirmed:
            self.log.warning("%s %s: confirmation still showed motion; using additionally backed-off value", channel, key)

        self.log.info(
            "%s %s safe=%.1fus first_move=%.1fus",
            channel,
            key,
            candidate_offset,
            first_moving_offset,
        )
        return DirectionResult(
            key=key,
            command_sign=int(command_sign),
            movement_sign=int(movement_sign),
            first_move_offset_us=first_moving_offset,
            safe_offset_us=candidate_offset,
        )

    def apply_channel_result(self, channel: str, results: list[DirectionResult]) -> None:
        by_key = {result.key: result for result in results}
        channel_cfg = self.candidate["CHANNEL_CONFIGS"][channel]
        if "deadband_us_pos" not in by_key or "deadband_us_neg" not in by_key:
            for result in results:
                channel_cfg[result.key] = round(float(result.safe_offset_us), 3)
            _write_yaml(self.output_path, self.candidate)
            self.log.warning("%s: incomplete directions; wrote asymmetric measured offsets only", channel)
            return

        old_center = float(self.probe_channels[channel]["center"])
        pos_edge = old_center + float(by_key["deadband_us_pos"].safe_offset_us)
        neg_edge = old_center - float(by_key["deadband_us_neg"].safe_offset_us)
        new_center = 0.5 * (pos_edge + neg_edge)
        symmetric_deadband = 0.5 * (pos_edge - neg_edge)

        channel_cfg["center"] = round(float(new_center), 3)
        channel_cfg["deadband_us_pos"] = round(float(symmetric_deadband), 3)
        channel_cfg["deadband_us_neg"] = round(float(symmetric_deadband), 3)
        _write_yaml(self.output_path, self.candidate)
        self.log.info(
            "%s centered: old_center=%.1f new_center=%.1f deadband=%.1fus pos_edge=%.1f neg_edge=%.1f",
            channel,
            old_center,
            new_center,
            symmetric_deadband,
            pos_edge,
            neg_edge,
        )

    def return_to_angle(
        self,
        channel: str,
        joint_index: int,
        target_angle_rad: float,
        results: list[DirectionResult],
    ) -> None:
        if not results:
            self.log.warning("%s: no movement direction data; skipping automatic return", channel)
            return

        end_t = time.monotonic() + RETURN_TIMEOUT_S
        period = 1.0 / PROBE_RATE_HZ
        while time.monotonic() < end_t:
            current = float(_read_joint_angles(self.hardware, self.robot_config)[joint_index])
            error_deg = float(np.degrees(np.arctan2(np.sin(target_angle_rad - current), np.cos(target_angle_rad - current))))
            if abs(error_deg) <= RETURN_TOLERANCE_DEG:
                self.hardware.reset(reset_pump=False)
                self.log.info("%s returned to %.2f deg tolerance", channel, RETURN_TOLERANCE_DEG)
                return

            desired_sign = int(np.sign(error_deg))
            result = next((item for item in results if item.movement_sign == desired_sign), None)
            if result is None:
                self.log.warning("%s: no command direction known for return error %+0.2f deg", channel, error_deg)
                break

            return_offset = result.first_move_offset_us * RETURN_OFFSET_GAIN
            command_abs = _command_for_pulse_offset(self.probe_channels[channel], return_offset, result.command_sign)
            command_abs = max(RETURN_COMMAND, min(RETURN_MAX_COMMAND, command_abs))
            self.hardware.send_named_pwm_commands({channel: float(result.command_sign) * command_abs}, unset_to_zero=True)
            time.sleep(period)

        self.hardware.reset(reset_pump=False)
        current = float(_read_joint_angles(self.hardware, self.robot_config)[joint_index])
        error_deg = float(np.degrees(np.arctan2(np.sin(target_angle_rad - current), np.cos(target_angle_rad - current))))
        self.log.warning("%s return stopped with remaining error %+0.2f deg", channel, error_deg)

    def run(self) -> Path:
        available = _valve_channels(self.candidate)
        ordered = [name for name in CHANNEL_ORDER if name in available]
        ordered.extend(name for name in available if name not in ordered)
        self.wait_ready()

        self.log.info("Candidate config: %s", self.output_path)
        self.start_udp_positioning()
        for channel in ordered:
            if channel not in JOINT_BY_CHANNEL:
                self.log.warning("Skipping %s: no joint mapping", channel)
                continue
            joint_index, joint_name = JOINT_BY_CHANNEL[channel]
            channel_start_angle = self.manual_position(channel, joint_index, joint_name)
            self.log.info("Tuning %s (%s)", channel, joint_name)
            results = []
            command_signs = (1, -1)
            for index, command_sign in enumerate(command_signs):
                result = self.tune_direction(channel, command_sign, joint_index)
                if result is not None:
                    results.append(result)
                self.hardware.reset(reset_pump=False)
                if index < len(command_signs) - 1:
                    self.manual_position(channel, joint_index, joint_name)
            self.apply_channel_result(channel, results)
            self.return_to_angle(channel, joint_index, channel_start_angle, results)
        return self.output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="IMU-assisted valve deadband tuning helper")
    parser.add_argument("--i-understand", action="store_true", help="Required safety acknowledgement")
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--udp-positioning", action="store_true", help="Use the gamepad/UDP wire format from simple_drive.py for manual positioning")
    parser.add_argument("--host", default=UDP_HOST, help="Local robot IP for UDP positioning")
    parser.add_argument("--port", type=int, default=UDP_PORT, help="UDP port for positioning")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    if not args.i_understand:
        parser.error("Refusing to move hardware without --i-understand")
    if not (0.0 < DEADBAND_BACKOFF < 1.0):
        parser.error("DEADBAND_BACKOFF must be in (0, 1)")

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(name)s: %(message)s")
    tuner = ValveTuner(args)
    try:
        path = tuner.run()
        logging.getLogger("valve_tuning_helper").info("Saved candidate config %s", path)
    finally:
        tuner.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
