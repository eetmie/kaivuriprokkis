import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = [
    "revolute_cabin",
    "revolute_lift",
    "revolute_tilt",
    "revolute_scoop",
]


class ImuStateNode(Node):
    def __init__(self) -> None:
        super().__init__("kaivuri_imu_state_node")
        self.declare_parameter("project_root", os.environ.get("KAIVURI_PROJECT_ROOT", "/work"))
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("config_file", "configuration_files/servo_config_200.yaml")
        self.declare_parameter("publish_tool_pose", True)

        self._project_root = Path(str(self.get_parameter("project_root").value)).resolve()
        self._add_project_import_path(self._project_root)

        from modules.differential_ik import get_pose_from_joint_angles
        from modules.differential_ik_cfg import load_excavator_robot_config
        from modules.excavator_ik_utils import canonical_joint_angles_from_imus
        from modules.hardware_interface import HardwareInterface

        self._canonical_joint_angles_from_imus = canonical_joint_angles_from_imus
        self._get_pose_from_joint_angles = get_pose_from_joint_angles
        self._robot_config = load_excavator_robot_config()

        config_file = str(self.get_parameter("config_file").value)
        config_path = self._resolve_project_path(config_file)
        self._hardware = HardwareInterface(
            config_file=str(config_path),
            enable_pwm=False,
            enable_imu=True,
            enable_adc=False,
            start_imu_reader=True,
            start_adc_reader=False,
            cleanup_disable_osc=False,
        )

        self._joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self._pose_pub = self.create_publisher(PoseStamped, "kaivuri/tool_pose", 10)
        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self.create_timer(1.0 / rate_hz, self._publish)
        self.get_logger().info(f"Publishing IMU-derived joint_states from {self._project_root}")

    def _add_project_import_path(self, project_root: Path) -> None:
        root = str(project_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _resolve_project_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self._project_root / path

    def _read_joint_angles(self) -> Optional[np.ndarray]:
        try:
            quaternions = self._hardware.read_all_imu_quaternions()
            if quaternions is None:
                return None
            sensor_quats = np.asarray(quaternions, dtype=np.float32)
            return self._canonical_joint_angles_from_imus(sensor_quats, self._robot_config)
        except Exception as exc:
            self.get_logger().warn(f"IMU state unavailable: {exc}", throttle_duration_sec=2.0)
            return None

    def _publish(self) -> None:
        joint_angles = self._read_joint_angles()
        if joint_angles is None:
            return

        now = self.get_clock().now().to_msg()
        joint_msg = JointState()
        joint_msg.header.stamp = now
        joint_msg.name = JOINT_NAMES
        joint_msg.position = [float(v) for v in joint_angles[:len(JOINT_NAMES)]]
        self._joint_pub.publish(joint_msg)

        if bool(self.get_parameter("publish_tool_pose").value):
            ee_pos, ee_quat = self._get_pose_from_joint_angles(joint_angles, self._robot_config)
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
