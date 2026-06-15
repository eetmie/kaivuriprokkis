import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .direct_controller import DirectController
from .ik import get_state
from .control_protocol import (
    ControlCommand,
    ControlMode,
    DirectCommand,
    PoseTarget,
    RobotTelemetry,
)
from .reachability import ReachabilityResult


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
        self.direct = DirectController(hardware)
        self.model = getattr(controller, "model", None)
        if self.model is None:
            raise ValueError("RobotService requires controller.model (ExcavatorModel)")
        self._state_lock = threading.Lock()
        self.state = RobotServiceState()
        self._telemetry_sequence = 0

        try:
            measured_pos, measured_rot = self.controller.get_pose()
            with self._state_lock:
                self.state.target_pose = PoseTarget(
                    float(measured_pos[0]),
                    float(measured_pos[1]),
                    float(measured_pos[2]),
                    float(measured_rot),
                )
        except Exception:
            pass

    def _log_service_warning(self, message: str, *args) -> None:
        logger = getattr(self.controller, "logger", None)
        if logger is not None and hasattr(logger, "warning"):
            try:
                logger.warning(message, *args)
            except Exception:
                pass

    def _set_pump_enabled(self, enabled: bool) -> None:
        try:
            if hasattr(self.hardware, "set_pump_enabled"):
                self.hardware.set_pump_enabled(enabled)
        except Exception as exc:
            self._log_service_warning("Pump enable sync failed: %s", exc)
            pass

    def submit_command(self, command: ControlCommand) -> Optional[ReachabilityResult]:
        """Apply ``command`` to the controller.

        Returns the ``ReachabilityResult`` for the pose check when a pose
        was submitted (None for direct mode, paused commands, or when the
        reachability check is disabled).
        """
        with self._state_lock:
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
                self.controller.suspend_ik_output()
                self.state.mode = ControlMode.DIRECT
            elif command.mode == ControlMode.IK and self.state.mode != ControlMode.IK:
                self.direct.clear()
                self.controller.resume_ik_output()
                self.state.mode = ControlMode.IK
                try:
                    measured_pos, measured_rot = self.controller.get_pose()
                    self.state.target_pose = PoseTarget(
                        float(measured_pos[0]),
                        float(measured_pos[1]),
                        float(measured_pos[2]),
                        float(measured_rot),
                    )
                except Exception as exc:
                    self._log_service_warning("Failed to sync IK target pose after mode switch: %s", exc)
                    pass

            if self.state.mode == ControlMode.DIRECT:
                self.state.direct_command = DirectCommand(
                    float(command.direct.slew),
                    float(command.direct.boom),
                    float(command.direct.arm),
                    float(command.direct.bucket),
                )
                if not self.state.paused:
                    self.direct.give_commands({
                        "slew": self.state.direct_command.slew,
                        "boom": self.state.direct_command.boom,
                        "arm": self.state.direct_command.arm,
                        "bucket": self.state.direct_command.bucket,
                    })
                    self.direct.send_pending()
            else:
                requested_pose = PoseTarget(
                    float(command.pose.x),
                    float(command.pose.y),
                    float(command.pose.z),
                    float(command.pose.rot_y_deg),
                )
                if not self.state.paused:
                    result = self.controller.give_pose(
                        np.array([requested_pose.x, requested_pose.y, requested_pose.z], dtype=np.float32),
                        float(requested_pose.rot_y_deg),
                    )
                    if result is not None and not result.reachable:
                        return result
                    self.state.target_pose = requested_pose
                    return result
                self.state.target_pose = requested_pose
            return None

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
        except Exception as exc:
            self._log_service_warning("Controller perf reset failed: %s", exc)
            pass
        try:
            if hasattr(self.hardware, "reset_perf_stats"):
                self.hardware.reset_perf_stats()
        except Exception as exc:
            self._log_service_warning("Hardware perf reset failed: %s", exc)
            pass

    def get_debug_state(self) -> dict:
        """Return debug info for diagnostics without exposing controller internals."""
        return {
            'fk_quaternions': self.controller.get_fk_quaternions(),
            'condition_number': self.controller.get_condition_number(),
            'perf_stats': self.controller.get_performance_stats() or {},
            'model': self.model,
        }

    def get_pose(self):
        """Return (position_array, rot_y_deg) from the controller."""
        return self.controller.get_pose()

    def get_state(self) -> RobotTelemetry:
        measured_pos, measured_rot = self.controller.get_pose()
        joint_angles_deg = self.controller.get_joint_angles()

        joint_positions = tuple((0.0, 0.0, 0.0) for _ in range(5))
        joint_angles_arr = np.asarray(joint_angles_deg, dtype=np.float32)
        # Guard against duck-typed stubs in tests that pass an opaque model.
        model_num_joints = getattr(self.model, "num_joints", None)
        if model_num_joints is not None and joint_angles_arr.shape[0] == model_num_joints:
            q_rad = np.radians(joint_angles_arr)
            # FK only (no Jacobian) — telemetry just wants link origins + tip.
            kin = get_state(q_rad, self.model, include_jacobian=False)
            positions = [tuple(float(v) for v in pos) for pos in kin.joint_origins_world]
            positions.append(tuple(float(v) for v in kin.ee_position))
            joint_positions = tuple(positions[:5])

        try:
            hardware_ready = bool(self.hardware.is_hardware_ready())
        except Exception as exc:
            self._log_service_warning("Hardware readiness check failed: %s", exc)
            hardware_ready = False

        with self._state_lock:
            self._telemetry_sequence += 1
            telemetry_sequence = self._telemetry_sequence
            mode = self.state.mode
            paused = self.state.paused
            target_pose = PoseTarget(
                float(self.state.target_pose.x),
                float(self.state.target_pose.y),
                float(self.state.target_pose.z),
                float(self.state.target_pose.rot_y_deg),
            )

        return RobotTelemetry(
            sequence=telemetry_sequence,
            timestamp_ms=int(time.time() * 1000) & 0xFFFFFFFF,
            mode=mode,
            paused=paused,
            hardware_ready=hardware_ready,
            measured_pose=PoseTarget(
                float(measured_pos[0]),
                float(measured_pos[1]),
                float(measured_pos[2]),
                float(measured_rot),
            ),
            target_pose=target_pose,
            joint_angles_deg=tuple(float(v) for v in joint_angles_arr.tolist()),
            joint_positions=joint_positions,
        )
