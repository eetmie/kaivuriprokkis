import math
from enum import Enum
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


class Stage(str, Enum):
    IDLE = "idle"
    APPROACH = "approach"
    DESCEND = "descend"
    HOLD = "hold"
    RETRACT = "retract"
    DONE = "done"


class CubeTouchExpertNode(Node):
    """Generate expert end-effector targets from a cube pose.

    The node expects Isaac Sim, or another task generator, to publish a cube pose.
    It then publishes a smooth approach -> descend -> hold -> retract trajectory
    to /kaivuri/target_pose_y and emits episode events suitable for rosbag
    segmentation.
    """

    def __init__(self) -> None:
        super().__init__("cube_touch_expert_node")
        self.declare_parameter("cube_pose_topic", "/kaivuri/cube_pose")
        self.declare_parameter("tool_pose_topic", "/kaivuri/tool_pose")
        self.declare_parameter("target_pose_y_topic", "/kaivuri/target_pose_y")
        self.declare_parameter("episode_event_topic", "/kaivuri/episode_event")
        self.declare_parameter("task_instruction_topic", "/kaivuri/task_instruction")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("speed_mps", 0.08)
        self.declare_parameter("rot_speed_degps", 30.0)
        self.declare_parameter("position_tolerance_m", 0.02)
        self.declare_parameter("hold_s", 0.25)
        self.declare_parameter("timeout_s", 60)
        self.declare_parameter("approach_height_m", 0.05)
        self.declare_parameter("retract_height_m", 0.05)
        self.declare_parameter("touch_clearance_m", 0.005)
        self.declare_parameter("cube_size_m", 0.05)
        self.declare_parameter("cube_pose_is_top_center", True)
        self.declare_parameter("rot_y_deg", 0.0)
        self.declare_parameter("instruction", "touch the top of the red cube")
        self.declare_parameter("wait_for_target_subscriber", True)
        self.declare_parameter("tool_pose_timeout_s", 0.5)
        self.declare_parameter("startup_tool_pose_samples", 0)
        self.declare_parameter("startup_hold_s", 1.0)
        self.declare_parameter("cube_restart_distance_m", 0.01)

        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self._dt = 1.0 / rate_hz
        self._stage = Stage.IDLE
        self._episode_id = 0
        self._episode_started = self.get_clock().now()
        self._hold_started: Optional[object] = None
        self._tool_position: Optional[np.ndarray] = None
        self._last_tool_pose_time: Optional[object] = None
        self._current_target: Optional[np.ndarray] = None
        self._target_rot_y_deg = float(self.get_parameter("rot_y_deg").value)
        self._goals: dict[Stage, np.ndarray] = {}
        self._pending_start_from_tool_pose = False
        self._waiting_for_target_subscriber = False
        self._waiting_for_tool_pose = False
        self._tool_pose_sample_count = 0
        self._episode_tool_pose_start_count = 0
        self._startup_ready_time: Optional[object] = None
        self._last_episode_cube_center: Optional[np.ndarray] = None

        cube_pose_topic = str(self.get_parameter("cube_pose_topic").value)
        tool_pose_topic = str(self.get_parameter("tool_pose_topic").value)
        target_pose_y_topic = str(self.get_parameter("target_pose_y_topic").value)
        episode_event_topic = str(self.get_parameter("episode_event_topic").value)
        task_instruction_topic = str(self.get_parameter("task_instruction_topic").value)

        self.create_subscription(PoseStamped, cube_pose_topic, self._on_cube_pose, 10)
        self.create_subscription(PoseStamped, tool_pose_topic, self._on_tool_pose, 10)
        self._target_pub = self.create_publisher(Float32MultiArray, target_pose_y_topic, 10)
        self._event_pub = self.create_publisher(String, episode_event_topic, 10)
        self._instruction_pub = self.create_publisher(String, task_instruction_topic, 10)
        self.create_timer(self._dt, self._tick)

        self._publish_instruction()

    """Published by ik_control_node.py to command the end-effector pose. """
    def _on_tool_pose(self, msg: PoseStamped) -> None:
        self._tool_position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )
        self._last_tool_pose_time = self.get_clock().now()
        self._tool_pose_sample_count += 1
        if self._current_target is None:
            self._current_target = self._tool_position.copy()

    """Published by Isaac Sim to provide the current pose of the cube. """
    def _on_cube_pose(self, msg: PoseStamped) -> None:
        center = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )
        if not self._should_start_episode_for_cube(center):
            return

        if bool(self.get_parameter("cube_pose_is_top_center").value):
            top = center
        else:
            cube_size = float(self.get_parameter("cube_size_m").value)
            top = center + np.array([0.0, 0.0, cube_size * 0.5], dtype=np.float32)

        touch = top + np.array(
            [0.0, 0.0, float(self.get_parameter("touch_clearance_m").value)],
            dtype=np.float32,
        )
        approach = touch + np.array(
            [0.0, 0.0, float(self.get_parameter("approach_height_m").value)],
            dtype=np.float32,
        )
        retract = touch + np.array(
            [0.0, 0.0, float(self.get_parameter("retract_height_m").value)],
            dtype=np.float32,
        )

        self._goals = {
            Stage.APPROACH: approach,
            Stage.DESCEND: touch,
            Stage.HOLD: touch,
            Stage.RETRACT: retract,
        }
        self._last_episode_cube_center = center.copy()
        self._episode_id += 1
        self._episode_started = self.get_clock().now()
        self._episode_tool_pose_start_count = self._tool_pose_sample_count
        self._startup_ready_time = None
        
        self._current_target = None
        self._pending_start_from_tool_pose = True
        self._set_stage(Stage.APPROACH)
        self._publish_instruction()


    def _tick(self) -> None:

        if self._stage == Stage.IDLE or self._current_target is None:
            if self._stage != Stage.IDLE and self._pending_start_from_tool_pose:
                self._try_seed_current_target_from_tool_pose()
            return

        if self._should_wait_for_target_subscriber():
            return

        if not self._tool_pose_is_fresh():
            self._waiting_for_tool_pose = True
            self.get_logger().warn(
                "Waiting for fresh /kaivuri/tool_pose before advancing expert target",
                throttle_duration_sec=2.0,
            )
            return

        if not self._has_startup_tool_pose_samples():

            self._waiting_for_tool_pose = True
            self.get_logger().warn(
                "Waiting for startup /kaivuri/tool_pose samples before advancing expert target",
                throttle_duration_sec=2.0,
            )
            return

        if self._waiting_for_target_subscriber or self._waiting_for_tool_pose:

            now = self.get_clock().now()
            self._waiting_for_target_subscriber = False
            self._waiting_for_tool_pose = False
            self._episode_started = now
            self._startup_ready_time = now
            self._try_seed_current_target_from_tool_pose()
            if self._current_target is None:
                return

        if self._startup_ready_time is not None:

            startup_hold_s = max(0.0, float(self.get_parameter("startup_hold_s").value))
            if self._elapsed(self._startup_ready_time) < startup_hold_s:

                self._publish_target()
                return
            self._startup_ready_time = None

        if self._timed_out():
            self._publish_event("failure_timeout")
            self._publish_event("episode_end")
            self._set_stage(Stage.DONE)
            return

        if self._stage == Stage.HOLD:

            self._publish_target()
            if self._hold_started is None:
                self._hold_started = self.get_clock().now()
            if self._elapsed(self._hold_started) >= float(self.get_parameter("hold_s").value):
                self._publish_event("touch_success")
                self._set_stage(Stage.RETRACT)
            return

        if self._stage == Stage.DONE:
            self._publish_target()
            return

        goal = self._goals.get(self._stage)
        if goal is None:
            return

        self._current_target = self._step_toward(self._current_target, goal)
        self._publish_target()
        if self._target_reached(goal):
            if self._stage == Stage.APPROACH:
                self._set_stage(Stage.DESCEND)
            elif self._stage == Stage.DESCEND:
                self._set_stage(Stage.HOLD)
            elif self._stage == Stage.RETRACT:
                self._publish_event("episode_end")
                self._set_stage(Stage.DONE)

    def _step_toward(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        delta = goal - current
        distance = float(np.linalg.norm(delta))
        max_step = max(0.0, float(self.get_parameter("speed_mps").value)) * self._dt
        if distance <= max_step or distance < 1e-9:
            return goal.copy()
        return current + delta * (max_step / distance)

    def _try_seed_current_target_from_tool_pose(self) -> None:
        if not self._pending_start_from_tool_pose:
            return
        if not self._tool_pose_is_fresh():
            self._waiting_for_tool_pose = True
            self.get_logger().warn(
                "Waiting for fresh /kaivuri/tool_pose to seed expert start target",
                throttle_duration_sec=2.0,
            )
            return
        self._current_target = self._tool_position.copy()
        self._pending_start_from_tool_pose = False

    def _should_start_episode_for_cube(self, center: np.ndarray) -> bool:
        if self._stage not in (Stage.IDLE, Stage.DONE):
            return False
        if self._last_episode_cube_center is None:
            return True
        restart_distance = max(0.0, float(self.get_parameter("cube_restart_distance_m").value))

        moved = float(np.linalg.norm(center - self._last_episode_cube_center))

        return moved >= restart_distance

    def _should_wait_for_target_subscriber(self) -> bool:
        if not bool(self.get_parameter("wait_for_target_subscriber").value):
            return False
        if self._target_pub.get_subscription_count() > 0:
            return False
        self._waiting_for_target_subscriber = True
        self.get_logger().warn(
            "Waiting for a subscriber on /kaivuri/target_pose_y before advancing expert target",
            throttle_duration_sec=2.0,
        )
        return True

    def _tool_pose_is_fresh(self) -> bool:
        if self._tool_position is None or self._last_tool_pose_time is None:
            return False
        timeout_s = max(0.0, float(self.get_parameter("tool_pose_timeout_s").value))
        if timeout_s <= 0.0:
            return True
        return self._elapsed(self._last_tool_pose_time) <= timeout_s

    def _has_startup_tool_pose_samples(self) -> bool:
        required = max(0, int(self.get_parameter("startup_tool_pose_samples").value))
        observed = self._tool_pose_sample_count - self._episode_tool_pose_start_count
        return observed >= required

    def _target_reached(self, goal: np.ndarray) -> bool:
        reference = self._tool_position if self._tool_position is not None else self._current_target
        if reference is None:
            return False
        tolerance = max(0.0, float(self.get_parameter("position_tolerance_m").value))
        return float(np.linalg.norm(reference - goal)) <= tolerance

    def _set_stage(self, stage: Stage) -> None:
        self._stage = stage
        self._hold_started = None
        if stage == Stage.APPROACH:
            self._publish_event("episode_start")
        if stage != Stage.DONE:
            self._publish_event(stage.value)

    def _publish_target(self) -> None:
        if self._current_target is None:
            return
        msg = Float32MultiArray()
        msg.data = [
            float(self._current_target[0]),
            float(self._current_target[1]),
            float(self._current_target[2]),
            float(self._target_rot_y_deg),
        ]
        self._target_pub.publish(msg)

    def _publish_event(self, event: str) -> None:
        msg = String()
        msg.data = f"{self._episode_id}:{event}"
        self._event_pub.publish(msg)

    def _publish_instruction(self) -> None:
        msg = String()
        msg.data = str(self.get_parameter("instruction").value)
        self._instruction_pub.publish(msg)

    def _timed_out(self) -> bool:
        timeout_s = max(0.0, float(self.get_parameter("timeout_s").value))
        return timeout_s > 0.0 and self._elapsed(self._episode_started) > timeout_s

    def _elapsed(self, start_time) -> float:
        return (self.get_clock().now() - start_time).nanoseconds * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CubeTouchExpertNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
