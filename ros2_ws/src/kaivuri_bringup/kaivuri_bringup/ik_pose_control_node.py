import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from kaivuri_bringup.project_paths import add_project_import_path, resolve_project_root


JOINT_NAMES = [
    "revolute_carriage",
    "revolute_lift",
    "revolute_tilt",
    "revolute_tool",
    "revolute_gripper",
    "revolute_claw_1",
    "revolute_claw_2",
]

_N_ACTIVE = 4
_Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class IkPoseControlNode(Node):
    """ROS 2 bridge from end-effector targets to the existing IK controller.

    Inputs:
      - /kaivuri/target_pose: geometry_msgs/PoseStamped
        position is the desired tool tip position in the excavator frame.
        orientation is reduced to the controller's supported body-frame pitch.
      - /kaivuri/target_pose_y: std_msgs/Float32MultiArray
        data order [x, y, z, rot_y_deg].

    Outputs:
      - /joint_states
      - /kaivuri/tool_pose
    """

    def __init__(self) -> None:
        super().__init__("kaivuri_ik_pose_control_node")
        self.declare_parameter("project_root", os.environ.get("KAIVURI_PROJECT_ROOT", "/work"))
        self.declare_parameter("robot", "auto")
        self.declare_parameter("config_file", "")
        self.declare_parameter("control_config_file", "")
        self.declare_parameter("pwm_i2c_bus", -1)
        self.declare_parameter("pwm_i2c_addr", -1)
        self.declare_parameter("target_pose_topic", "/kaivuri/target_pose")
        self.declare_parameter("target_pose_y_topic", "/kaivuri/target_pose_y")
        self.declare_parameter("state_rate_hz", 30.0)
        self.declare_parameter("command_timeout_s", 1.0)
        self.declare_parameter("ready_timeout_s", 30.0)
        self.declare_parameter("pump_auto_mode", False)
        self.declare_parameter("toggle_channels", True)
        self.declare_parameter("stale_timeout_s", 0.5)
        self.declare_parameter("publish_tool_pose", True)
        self.declare_parameter("log_reachability_rejections", True)

        self._project_root = resolve_project_root(str(self.get_parameter("project_root").value))
        add_project_import_path(self._project_root)

        from modules.board import resolve_profile
        from modules.ik import (
            extract_axis_rotation,
            get_pose_from_joint_angles,
            load_excavator_robot_config,
        )
        from modules.excavator_controller import ExcavatorController
        from modules.hardware_interface import HardwareInterface

        self._extract_axis_rotation = extract_axis_rotation
        self._get_pose_from_joint_angles = get_pose_from_joint_angles

        robot_profile = resolve_profile(str(self.get_parameter("robot").value))
        config_file = self._param_or_profile("config_file", robot_profile["servo_config_file"])
        control_config_file = self._param_or_profile("control_config_file", robot_profile["control_config_file"])
        pwm_i2c_bus = self._int_param_or_profile("pwm_i2c_bus", int(robot_profile["pwm_i2c_bus"]), unset=-1)
        pwm_i2c_addr = self._int_param_or_profile("pwm_i2c_addr", int(robot_profile["pwm_i2c_addr"]), unset=-1)

        control_config_path = self._resolve_project_path(control_config_file)
        self._robot_config = load_excavator_robot_config(str(control_config_path))
        config_path = self._resolve_project_path(config_file)

        self.get_logger().info(
            "IK pose control profile="
            f"{robot_profile['profile_name']} board={robot_profile['board']} "
            f"servo_config={config_path} control_config={control_config_path} "
            f"I2C bus={pwm_i2c_bus} addr=0x{pwm_i2c_addr:02X}"
        )

        self._hardware = HardwareInterface(
            config_file=str(config_path),
            control_config_file=str(control_config_path),
            pump_auto_mode=bool(self.get_parameter("pump_auto_mode").value),
            toggle_channels=bool(self.get_parameter("toggle_channels").value),
            stale_timeout_s=float(self.get_parameter("stale_timeout_s").value),
            enable_pwm=True,
            enable_imu=True,
            enable_adc=False,
            start_imu_reader=True,
            start_adc_reader=False,
            cleanup_disable_osc=False,
            pwm_i2c_bus=pwm_i2c_bus,
            pwm_i2c_addr=pwm_i2c_addr,
        )
        self._wait_for_hardware(float(self.get_parameter("ready_timeout_s").value))

        self._controller = ExcavatorController(
            self._hardware,
            config=None,
            enable_perf_tracking=False,
            control_config_file=str(control_config_path),
        )
        self._controller.start()

        self._last_command_time: Optional[float] = None
        self._target_active = False

        target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        target_pose_y_topic = str(self.get_parameter("target_pose_y_topic").value)
        self.create_subscription(PoseStamped, target_pose_topic, self._on_target_pose, 10)
        self.create_subscription(Float32MultiArray, target_pose_y_topic, self._on_target_pose_y, 10)

        self._joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self._pose_pub = self.create_publisher(PoseStamped, "kaivuri/tool_pose", 10)

        state_rate_hz = max(1.0, float(self.get_parameter("state_rate_hz").value))
        self.create_timer(1.0 / state_rate_hz, self._state_tick)
        self.get_logger().info(
            f"IK pose control ready; subscribe {target_pose_topic} or {target_pose_y_topic}"
        )

    def _resolve_project_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self._project_root / path

    def _param_or_profile(self, name: str, profile_value: str) -> str:
        value = str(self.get_parameter(name).value).strip()
        return value or profile_value

    def _int_param_or_profile(self, name: str, profile_value: int, *, unset: int = None) -> int:
        value = int(self.get_parameter(name).value)
        if unset is not None and value == unset:
            return profile_value
        return value

    def _wait_for_hardware(self, timeout_s: float) -> None:
        deadline = time.time() + max(0.0, timeout_s)
        while not self._hardware.is_hardware_ready():
            if timeout_s > 0.0 and time.time() >= deadline:
                raise TimeoutError("Hardware did not become ready before timeout")
            time.sleep(0.1)

    def _on_target_pose(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        if float(np.linalg.norm(quat_wxyz)) < 1e-6:
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rot_y_deg = float(np.degrees(self._extract_axis_rotation(quat_wxyz, _Y_AXIS)))
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )
        self._send_target(position, rot_y_deg)

    def _on_target_pose_y(self, msg: Float32MultiArray) -> None:
        values = self._finite_values(list(msg.data))
        if len(values) < 3:
            self.get_logger().warn(
                "Ignoring /kaivuri/target_pose_y: expected at least [x, y, z]",
                throttle_duration_sec=1.0,
            )
            return
        rot_y_deg = values[3] if len(values) >= 4 else 0.0
        self._send_target(np.array(values[:3], dtype=np.float32), rot_y_deg)

    def _finite_values(self, values: List[float]) -> List[float]:
        out = []
        for value in values:
            try:
                f = float(value)
            except Exception:
                f = 0.0
            out.append(f if np.isfinite(f) else 0.0)
        return out

    def _send_target(self, position: np.ndarray, rot_y_deg: float) -> None:
        result = self._controller.give_pose(position, rot_y_deg)
        rejected = result is not None and not result.reachable
        if rejected:
            if bool(self.get_parameter("log_reachability_rejections").value):
                self.get_logger().warn(
                    f"Rejected unreachable target pos={np.round(position, 4)} "
                    f"rot_y={rot_y_deg:.2f} closest={np.round(result.closest_position, 4)} "
                    f"err={result.pos_error_m:.4f}m",
                    throttle_duration_sec=1.0,
                )
            return

        self._last_command_time = time.monotonic()
        self._target_active = True

    def _state_tick(self) -> None:
        self._clear_stale_target_if_needed()
        self._publish_state()

    def _clear_stale_target_if_needed(self) -> None:
        if not self._target_active or self._last_command_time is None:
            return
        timeout_s = max(0.0, float(self.get_parameter("command_timeout_s").value))
        if timeout_s > 0.0 and (time.monotonic() - self._last_command_time) > timeout_s:
            self._controller.clear_target()
            self._target_active = False
            self.get_logger().warn("IK target timed out; cleared active target")

    def _publish_state(self) -> None:
        try:
            joint_angles_deg = self._controller.get_joint_angles()
            joint_angles_rad = np.radians(np.asarray(joint_angles_deg, dtype=np.float32))
        except Exception as exc:
            self.get_logger().warn(f"Joint state unavailable: {exc}", throttle_duration_sec=2.0)
            return

        joint_vel_degps, vel_age = self._controller.get_joint_velocities_with_age()
        if joint_vel_degps is not None and vel_age < 0.2:
            joint_vel_radps = [float(np.radians(v)) for v in joint_vel_degps[:_N_ACTIVE]]
        else:
            joint_vel_radps = [0.0] * _N_ACTIVE

        now = self.get_clock().now().to_msg()
        joint_msg = JointState()
        joint_msg.header.stamp = now
        joint_msg.name = JOINT_NAMES
        joint_msg.position = [float(v) for v in joint_angles_rad[:_N_ACTIVE]] + [0.0, 0.0, 0.0]
        joint_msg.velocity = joint_vel_radps + [0.0, 0.0, 0.0]
        self._joint_pub.publish(joint_msg)

        if bool(self.get_parameter("publish_tool_pose").value):
            ee_pos, ee_quat = self._get_pose_from_joint_angles(joint_angles_rad[:_N_ACTIVE], self._robot_config)
            pose_msg = PoseStamped()
            pose_msg.header.stamp = now
            pose_msg.header.frame_id = "excavator"
            pose_msg.pose.position.x = float(ee_pos[0])
            pose_msg.pose.position.y = float(ee_pos[1])
            pose_msg.pose.position.z = float(ee_pos[2])
            pose_msg.pose.orientation.w = float(ee_quat[0])
            pose_msg.pose.orientation.x = float(ee_quat[1])
            pose_msg.pose.orientation.y = float(ee_quat[2])
            pose_msg.pose.orientation.z = float(ee_quat[3])
            self._pose_pub.publish(pose_msg)

    def destroy_node(self) -> bool:
        try:
            self._controller.clear_target()
            self._controller.stop()
        except Exception:
            pass
        try:
            self._hardware.shutdown()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IkPoseControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
