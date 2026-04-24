import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import diff_ik_V2 as diff_ik
from .control_protocol import (
    ControlCommand,
    ControlMode,
    DirectCommand,
    PoseTarget,
    RobotTelemetry,
)


@dataclass
class RobotServiceState:
    mode: ControlMode = ControlMode.IK
    paused: bool = False
    target_pose: PoseTarget = field(default_factory=PoseTarget)
    direct_command: DirectCommand = field(default_factory=DirectCommand)
    last_command_sequence: int = 0


class RobotService:
    """Authoritative robot-side command and telemetry boundary."""

    def __init__(self, controller, hardware):
        self.controller = controller
        self.hardware = hardware
        self.robot_config = diff_ik.load_excavator_robot_config()
        self.state = RobotServiceState()
        self._telemetry_sequence = 0

        try:
            measured_pos, measured_rot = self.controller.get_pose()
            self.state.target_pose = PoseTarget(
                float(measured_pos[0]),
                float(measured_pos[1]),
                float(measured_pos[2]),
                float(measured_rot),
            )
        except Exception:
            pass

    def _set_pump_enabled(self, enabled: bool) -> None:
        try:
            if hasattr(self.hardware, "set_pump_enabled"):
                self.hardware.set_pump_enabled(enabled)
        except Exception:
            pass

    def submit_command(self, command: ControlCommand) -> None:
        self.state.last_command_sequence = int(command.sequence)

        if command.reload_config:
            self.hardware.reload_config()

        if command.pause and not self.state.paused:
            self.controller.pause()
            self._set_pump_enabled(False)
            self.state.paused = True

        if command.resume and self.state.paused:
            self.controller.resume()
            self._set_pump_enabled(True)
            self.state.paused = False

        if command.mode == ControlMode.DIRECT and self.state.mode != ControlMode.DIRECT:
            self.controller.enter_direct_mode()
            self.state.mode = ControlMode.DIRECT
        elif command.mode == ControlMode.IK and self.state.mode != ControlMode.IK:
            self.controller.exit_direct_mode()
            self.state.mode = ControlMode.IK
            try:
                measured_pos, measured_rot = self.controller.get_pose()
                self.state.target_pose = PoseTarget(
                    float(measured_pos[0]),
                    float(measured_pos[1]),
                    float(measured_pos[2]),
                    float(measured_rot),
                )
            except Exception:
                pass

        if self.state.mode == ControlMode.DIRECT:
            self.state.direct_command = DirectCommand(
                float(command.direct.slew),
                float(command.direct.boom),
                float(command.direct.arm),
                float(command.direct.bucket),
            )
            if not self.state.paused:
                self.controller.give_direct_commands({
                    "rotate": self.state.direct_command.slew,
                    "lift_boom": self.state.direct_command.boom,
                    "tilt_boom": self.state.direct_command.arm,
                    "scoop": self.state.direct_command.bucket,
                })
        else:
            self.state.target_pose = PoseTarget(
                float(command.pose.x),
                float(command.pose.y),
                float(command.pose.z),
                float(command.pose.rot_y_deg),
            )
            if not self.state.paused:
                self.controller.give_pose(
                    np.array([self.state.target_pose.x, self.state.target_pose.y, self.state.target_pose.z], dtype=np.float32),
                    float(self.state.target_pose.rot_y_deg),
                )

    # ---- Lifecycle wrappers ----

    def start(self):
        """Start the controller's background control loop."""
        self.controller.start()

    def stop(self):
        """Stop the controller and reset hardware."""
        self.controller.stop()
        self.hardware.reset(reset_pump=True)

    def reset_perf_stats(self):
        """Reset performance statistics on controller and hardware."""
        try:
            self.controller.reset_performance_stats()
        except Exception:
            pass
        try:
            if hasattr(self.hardware, "reset_perf_stats"):
                self.hardware.reset_perf_stats()
        except Exception:
            pass

    def get_debug_state(self) -> dict:
        """Return debug info for diagnostics without exposing controller internals."""
        state = {
            'raw_quaternions': self.controller.get_raw_quaternions(),
            'condition_number': self.controller.get_condition_number(),
            'perf_stats': self.controller.get_performance_stats() or {},
            'robot_config': self.robot_config,
        }
        # Include base IMU if available
        base_imu = self.hardware.read_base_imu()
        if base_imu is not None:
            state['base_imu_quat'] = base_imu.get('quat')
            state['base_imu_gyro'] = base_imu.get('gyro')
        return state

    def get_pose(self):
        """Return (position_array, rot_y_deg) from the controller."""
        return self.controller.get_pose()

    def get_state(self) -> RobotTelemetry:
        measured_pos, measured_rot = self.controller.get_pose()
        joint_angles = self.controller.get_joint_angles()
        raw_quats = self.controller.get_raw_quaternions()

        joint_positions = tuple((0.0, 0.0, 0.0) for _ in range(5))
        if raw_quats is not None and len(raw_quats) >= 4:
            jp = diff_ik.get_joint_positions(raw_quats, self.robot_config)
            ee_pos, _ = diff_ik.get_pose(raw_quats, self.robot_config)
            positions = [tuple(float(v) for v in pos) for pos in jp]
            positions.append(tuple(float(v) for v in ee_pos))
            joint_positions = tuple(positions[:5])

        perf_stats = {}
        try:
            perf_stats = self.controller.get_performance_stats() or {}
        except Exception:
            perf_stats = {}

        try:
            hardware_ready = bool(self.hardware.is_hardware_ready())
        except Exception:
            hardware_ready = False

        self._telemetry_sequence += 1
        return RobotTelemetry(
            sequence=self._telemetry_sequence,
            timestamp_ms=int(time.time() * 1000) & 0xFFFFFFFF,
            mode=self.state.mode,
            paused=self.state.paused,
            hardware_ready=hardware_ready,
            slew_fusion_enabled=bool(perf_stats.get("slew_fusion_enabled", False)),
            slew_fusion_active=bool(perf_stats.get("slew_fusion_active", False)),
            measured_pose=PoseTarget(
                float(measured_pos[0]),
                float(measured_pos[1]),
                float(measured_pos[2]),
                float(measured_rot),
            ),
            target_pose=PoseTarget(
                float(self.state.target_pose.x),
                float(self.state.target_pose.y),
                float(self.state.target_pose.z),
                float(self.state.target_pose.rot_y_deg),
            ),
            joint_angles_deg=tuple(float(v) for v in np.asarray(joint_angles, dtype=np.float32).tolist()),
            joint_positions=joint_positions,
            slew_fusion_gyro_z_degps=float(perf_stats.get("slew_fusion_gyro_z_degps", 0.0)),
        )
