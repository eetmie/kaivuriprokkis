import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


JOINT_NAMES = [
    "revolute_cabin",
    "revolute_lift",
    "revolute_tilt",
    "revolute_scoop",
]

COMMAND_NAMES = [
    "rotate",
    "lift_boom",
    "tilt_boom",
    "scoop",
    "trackL",
    "trackR",
]


class RawDirectDriveNode(Node):
    """Raw normalized valve/track command bridge for ROS testing.

    Subscribes to std_msgs/Float32MultiArray on /kaivuri/direct_pwm.
    Data order is [rotate, lift_boom, tilt_boom, scoop, trackL, trackR].
    """

    def __init__(self) -> None:
        super().__init__("kaivuri_raw_direct_drive_node")
        self.declare_parameter("project_root", os.environ.get("KAIVURI_PROJECT_ROOT", "/work"))
        self.declare_parameter("config_file", "configuration_files/servo_config_200.yaml")
        self.declare_parameter("command_topic", "/kaivuri/direct_pwm")
        self.declare_parameter("command_rate_hz", 50.0)
        self.declare_parameter("state_rate_hz", 30.0)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("ready_timeout_s", 30.0)
        self.declare_parameter("pump_auto_mode", True)
        self.declare_parameter("toggle_channels", True)
        self.declare_parameter("stale_timeout_s", 0.5)
        self.declare_parameter("publish_tool_pose", True)

        self._project_root = Path(str(self.get_parameter("project_root").value)).resolve()
        self._add_project_import_path(self._project_root)

        from modules.differential_ik import get_pose_from_joint_angles
        from modules.differential_ik_cfg import load_excavator_robot_config
        from modules.excavator_controller import ExcavatorController
        from modules.hardware_interface import HardwareInterface

        self._get_pose_from_joint_angles = get_pose_from_joint_angles
        self._robot_config = load_excavator_robot_config()

        config_path = self._resolve_project_path(str(self.get_parameter("config_file").value))
        self._hardware = HardwareInterface(
            config_file=str(config_path),
            pump_auto_mode=bool(self.get_parameter("pump_auto_mode").value),
            toggle_channels=bool(self.get_parameter("toggle_channels").value),
            stale_timeout_s=float(self.get_parameter("stale_timeout_s").value),
            enable_pwm=True,
            enable_imu=True,
            enable_adc=False,
            start_imu_reader=True,
            start_adc_reader=False,
            cleanup_disable_osc=False,
        )
        self._wait_for_hardware(float(self.get_parameter("ready_timeout_s").value))

        self._controller = ExcavatorController(
            self._hardware,
            config=None,
            enable_perf_tracking=False,
        )
        self._controller.start()
        self._controller.enter_direct_mode()

        self._latest_commands: Dict[str, float] = {}
        self._last_command_time = 0.0
        self._stale_zeroed = True

        command_topic = str(self.get_parameter("command_topic").value)
        self.create_subscription(Float32MultiArray, command_topic, self._on_direct_pwm, 10)
        self._joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self._pose_pub = self.create_publisher(PoseStamped, "kaivuri/tool_pose", 10)

        command_rate_hz = max(1.0, float(self.get_parameter("command_rate_hz").value))
        state_rate_hz = max(1.0, float(self.get_parameter("state_rate_hz").value))
        self.create_timer(1.0 / command_rate_hz, self._command_tick)
        self.create_timer(1.0 / state_rate_hz, self._state_tick)
        self.get_logger().info(
            f"Raw direct drive ready on {command_topic}; order={COMMAND_NAMES}"
        )

    def _add_project_import_path(self, project_root: Path) -> None:
        root = str(project_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _resolve_project_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self._project_root / path

    def _wait_for_hardware(self, timeout_s: float) -> None:
        deadline = time.time() + max(0.0, timeout_s)
        while not self._hardware.is_hardware_ready():
            if timeout_s > 0.0 and time.time() >= deadline:
                raise TimeoutError("Hardware did not become ready before timeout")
            time.sleep(0.1)

    def _on_direct_pwm(self, msg: Float32MultiArray) -> None:
        values = self._clamped_values(list(msg.data))
        if len(values) < 4:
            self.get_logger().warn(
                "Ignoring /kaivuri/direct_pwm: expected at least 4 values",
                throttle_duration_sec=1.0,
            )
            return

        while len(values) < len(COMMAND_NAMES):
            values.append(0.0)
        self._latest_commands = {
            name: values[idx]
            for idx, name in enumerate(COMMAND_NAMES)
        }
        self._last_command_time = time.monotonic()
        self._stale_zeroed = False

    def _clamped_values(self, values: List[float]) -> List[float]:
        out = []
        for value in values:
            try:
                out.append(float(np.clip(float(value), -1.0, 1.0)))
            except Exception:
                out.append(0.0)
        return out

    def _command_tick(self) -> None:
        timeout_s = max(0.0, float(self.get_parameter("command_timeout_s").value))
        age_s = time.monotonic() - self._last_command_time
        if not self._latest_commands or age_s > timeout_s:
            if not self._stale_zeroed:
                self._controller.give_direct_commands({})
                self._stale_zeroed = True
            return
        self._controller.give_direct_commands(self._latest_commands)

    def _state_tick(self) -> None:
        try:
            joint_angles_deg = self._controller.get_joint_angles()
            joint_angles_rad = np.radians(np.asarray(joint_angles_deg, dtype=np.float32))
        except Exception as exc:
            self.get_logger().warn(f"Joint state unavailable: {exc}", throttle_duration_sec=2.0)
            return

        now = self.get_clock().now().to_msg()
        joint_msg = JointState()
        joint_msg.header.stamp = now
        joint_msg.name = JOINT_NAMES
        joint_msg.position = [float(v) for v in joint_angles_rad[:len(JOINT_NAMES)]]
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

    def destroy_node(self) -> bool:
        try:
            self._controller.give_direct_commands({})
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
    node = RawDirectDriveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
