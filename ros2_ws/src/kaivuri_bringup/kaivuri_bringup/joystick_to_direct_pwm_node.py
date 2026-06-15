from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


COMMAND_NAMES = [
    "slew",
    "boom",
    "arm",
    "bucket",
    "trackL",
    "trackR",
]


class JoystickToDirectPwmNode(Node):
    """Map raw joystick axis values to normalized raw direct-drive commands."""

    def __init__(self) -> None:
        super().__init__("joystick_to_direct_pwm_node")
        self.declare_parameter("input_topic", "joystick_values")
        self.declare_parameter("output_topic", "/kaivuri/direct_pwm")
        self.declare_parameter("raw_max", 127.0)
        self.declare_parameter("axis_indices", [0, 1, 2, 3, 4, 5])
        self.declare_parameter("axis_signs", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._raw_max = max(1e-6, abs(float(self.get_parameter("raw_max").value)))
        self._deadband = max(0.0, min(1.0, abs(float(self.get_parameter("deadband").value))))
        self._axis_indices = self._fixed_int_list(
            list(self.get_parameter("axis_indices").value),
            default=[0, 1, 2, 3, 4, 5],
        )
        self._axis_signs = self._fixed_float_list(
            list(self.get_parameter("axis_signs").value),
            default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )

        self._publisher = self.create_publisher(Float32MultiArray, output_topic, 10)
        self.create_subscription(Float32MultiArray, input_topic, self._on_joystick_values, 10)

        self.get_logger().info(
            f"Mapping {input_topic} to {output_topic}; order={COMMAND_NAMES}, "
            f"axis_indices={self._axis_indices}, axis_signs={self._axis_signs}, raw_max={self._raw_max}"
        )

    def _fixed_int_list(self, values: List[object], *, default: List[int]) -> List[int]:
        result = []
        for value in values[:len(COMMAND_NAMES)]:
            try:
                result.append(int(value))
            except Exception:
                result.append(-1)
        while len(result) < len(COMMAND_NAMES):
            result.append(default[len(result)])
        return result

    def _fixed_float_list(self, values: List[object], *, default: List[float]) -> List[float]:
        result = []
        for value in values[:len(COMMAND_NAMES)]:
            try:
                result.append(float(value))
            except Exception:
                result.append(default[len(result)])
        while len(result) < len(COMMAND_NAMES):
            result.append(default[len(result)])
        return result

    def _on_joystick_values(self, msg: Float32MultiArray) -> None:
        raw_values = list(msg.data)
        direct_pwm = []
        for command_name, axis_index, axis_sign in zip(COMMAND_NAMES, self._axis_indices, self._axis_signs):
            if axis_index < 0:
                direct_pwm.append(0.0)
                continue
            if axis_index >= len(raw_values):
                self.get_logger().warn(
                    f"Joystick message missing axis {axis_index} for {command_name}; sending 0.0",
                    throttle_duration_sec=2.0,
                )
                direct_pwm.append(0.0)
                continue

            normalized = float(raw_values[axis_index]) / self._raw_max
            normalized = max(-1.0, min(1.0, normalized * axis_sign))

            direct_pwm.append(normalized)

        out = Float32MultiArray()
        out.data = direct_pwm
        self._publisher.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JoystickToDirectPwmNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
