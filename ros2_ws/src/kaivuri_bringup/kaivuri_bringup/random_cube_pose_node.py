import math
import random

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class RandomCubePoseNode(Node):
    """Publish random cube poses for testing the cube touch expert without Isaac."""

    def __init__(self) -> None:
        super().__init__("random_cube_pose_node")
        self.declare_parameter("cube_pose_topic", "/kaivuri/cube_pose")
        self.declare_parameter("period_s", 8.0)
        self.declare_parameter("use_polar_workspace", True)
        self.declare_parameter("radius_min", 0.350)
        self.declare_parameter("radius_max", 0.680)
        self.declare_parameter("angle_min_deg", -90.0)
        self.declare_parameter("angle_max_deg", 90.0)
        self.declare_parameter("x_min", 0.350)
        self.declare_parameter("x_max", 0.680)
        self.declare_parameter("y_min", -0.150)
        self.declare_parameter("y_max", 0.150)
        self.declare_parameter("z", 0.0)
        self.declare_parameter("frame_id", "excavator")

        topic = str(self.get_parameter("cube_pose_topic").value)
        period_s = max(0.1, float(self.get_parameter("period_s").value))
        self._pub = self.create_publisher(PoseStamped, topic, 10)
        self.create_timer(period_s, self._publish_cube)
        self._publish_cube()
        self.get_logger().info(f"Publishing random cube poses to {topic} every {period_s:.1f}s")

    def _publish_cube(self) -> None:
        x, y = self._sample_xy()
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter("frame_id").value)
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = float(self.get_parameter("z").value)
        msg.pose.orientation.w = 1.0
        self._pub.publish(msg)

    def _sample_xy(self) -> tuple[float, float]:
        if not bool(self.get_parameter("use_polar_workspace").value):
            return (
                random.uniform(
                    float(self.get_parameter("x_min").value),
                    float(self.get_parameter("x_max").value),
                ),
                random.uniform(
                    float(self.get_parameter("y_min").value),
                    float(self.get_parameter("y_max").value),
                ),
            )

        radius_min = max(0.0, float(self.get_parameter("radius_min").value))
        radius_max = max(radius_min, float(self.get_parameter("radius_max").value))
        angle_min = max(-90.0, float(self.get_parameter("angle_min_deg").value))
        angle_max = min(90.0, float(self.get_parameter("angle_max_deg").value))
        if angle_min > angle_max:
            angle_min, angle_max = -90.0, 90.0

        radius = math.sqrt(random.uniform(radius_min * radius_min, radius_max * radius_max))
        angle = math.radians(random.uniform(angle_min, angle_max))
        return radius * math.cos(angle), radius * math.sin(angle)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RandomCubePoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
