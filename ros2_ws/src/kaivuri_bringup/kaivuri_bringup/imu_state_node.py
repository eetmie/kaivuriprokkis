import os
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

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

_N_ACTIVE = 4  # joints with IMU coverage; gripper/claws remain at zero


class ImuStateNode(Node):
    def __init__(self) -> None:
        super().__init__("kaivuri_imu_state_node")
        self.declare_parameter("project_root", os.environ.get("KAIVURI_PROJECT_ROOT", "/work"))
        self.declare_parameter("robot", "auto")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("config_file", "")
        self.declare_parameter("control_config_file", "")
        self.declare_parameter("publish_tool_pose", True)
        self._project_root = resolve_project_root(str(self.get_parameter("project_root").value))
        add_project_import_path(self._project_root)

        from modules.ik import get_pose_from_joint_angles, load_excavator_robot_config
        from modules.excavator_controller import ExcavatorController
        from modules.hardware_interface import HardwareInterface
        from modules.board import resolve_profile

        self._get_pose_from_joint_angles = get_pose_from_joint_angles

        robot_profile = resolve_profile(str(self.get_parameter("robot").value))
        config_file = self._param_or_profile("config_file", robot_profile["servo_config_file"])
        control_config_file = self._param_or_profile("control_config_file", robot_profile["control_config_file"])
        control_config_path = self._resolve_project_path(control_config_file)
        self._robot_config = load_excavator_robot_config(str(control_config_path))

        config_path = self._resolve_project_path(config_file)
        self.get_logger().info(
            "IMU state profile="
            f"{robot_profile['profile_name']} board={robot_profile['board']} "
            f"servo_config={config_path} control_config={control_config_path}"
        )
        self._hardware = HardwareInterface(
            config_file=str(config_path),
            control_config_file=str(control_config_path),
            enable_pwm=False,
            enable_imu=True,
            enable_adc=False,
            start_imu_reader=True,
            start_adc_reader=False,
            cleanup_disable_osc=False,
        )

        self._controller = ExcavatorController(
            self._hardware,
            config=None,
            enable_perf_tracking=False,
            control_config_file=str(control_config_path),
        )
        self._controller.start()
        # State-only node: suspend the IK loop's PWM output. The controller
        # keeps reading sensors so joint angles / FK stay fresh.
        self._controller.suspend_ik_output()

        self._joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self._pose_pub = self.create_publisher(PoseStamped, "kaivuri/tool_pose", 10)
        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self.create_timer(1.0 / rate_hz, self._read_joint_angles)
        self.get_logger().info(
            f"Publishing IMU-derived joint_states from {self._project_root} "
            "(finite-difference velocities from Pico-fused joint angles)"
        )

    def _resolve_project_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self._project_root / path

    def _param_or_profile(self, name: str, profile_value: str) -> str:
        value = str(self.get_parameter(name).value).strip()
        return value or profile_value

    def _read_joint_angles(self) -> Optional[np.ndarray]:
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
            ee_pos, ee_quat = self._get_pose_from_joint_angles(joint_angles_rad, self._robot_config)
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

        return joint_angles_rad

    def destroy_node(self) -> bool:
        try:
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
    node = ImuStateNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
