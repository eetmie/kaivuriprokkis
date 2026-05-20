import math
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = [
    "revolute_carriage",
    "revolute_lift",
    "revolute_tilt",
    "revolute_tool",
    "revolute_gripper",
    "revolute_claw_1",
    "revolute_claw_2",
]


class DemoStateNode(Node):
    def __init__(self) -> None:
        super().__init__("kaivuri_demo_state_node")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("animate", True)
        self.declare_parameter("joint_names", JOINT_NAMES)
        self.declare_parameter("base_positions", [0.0, -0.2, 1.4, -0.2, 0.0, 0.0, 0.0])
        self.declare_parameter("amplitudes", [0.35, 0.2, 0.3, 0.25, 0.0, 0.0, 0.0])

        self._joint_names = list(self.get_parameter("joint_names").value)
        self._base_positions = self._float_list("base_positions", len(self._joint_names))
        self._amplitudes = self._float_list("amplitudes", len(self._joint_names))
        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))

        self._publisher = self.create_publisher(JointState, "joint_states", 10)
        self._start_time = self.get_clock().now()
        self.create_timer(1.0 / rate_hz, self._publish)
        self.get_logger().info("Publishing demo joint_states for Kaivuri")

    def _float_list(self, parameter_name: str, expected_len: int) -> List[float]:
        values = list(self.get_parameter(parameter_name).value)
        values = [float(v) for v in values[:expected_len]]
        while len(values) < expected_len:
            values.append(0.0)
        return values

    def _publish(self) -> None:
        now = self.get_clock().now()
        elapsed = (now - self._start_time).nanoseconds * 1e-9
        animate = bool(self.get_parameter("animate").value)

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = self._joint_names
        if animate:
            msg.position = [
                base + amp * math.sin(elapsed * 0.6 + idx * 0.9)
                for idx, (base, amp) in enumerate(zip(self._base_positions, self._amplitudes))
            ]
        else:
            msg.position = list(self._base_positions)
        msg.velocity = []
        msg.effort = []
        self._publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DemoStateNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
