from __future__ import annotations

import importlib
from typing import Any, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger


class LeRobotRos2RecorderNode(Node):
    """Record Kaivuri ROS 2 topics directly into a LeRobot dataset."""

    def __init__(self) -> None:
        super().__init__("lerobot_ros2_recorder_node")

        self.declare_parameter("repo_id", "kaivuri/ros2_recording")
        self.declare_parameter("root", "/mnt/c/users/sh25016/kaivuri_lerobot_live")
        self.declare_parameter("fps", 3.0)
        self.declare_parameter("episode_time_s", 60.0)
        self.declare_parameter("num_episodes", 1)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("save_partial_on_shutdown", True)
        self.declare_parameter("push_to_hub_on_shutdown", False)
        self.declare_parameter("robot_type", "kaivuri")
        self.declare_parameter("task", "touch the top of the red cube")
        self.declare_parameter("task_instruction_topic", "/kaivuri/task_instruction")
        self.declare_parameter("hut_image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("top_image_topic", "/camera/camera1/color/image_raw")
        self.declare_parameter("tool_pose_topic", "/kaivuri/tool_pose")
        self.declare_parameter("action_topic", "/kaivuri/target_pose_y")
        self.declare_parameter("max_sample_age_s", 1.0)

        self._dataset: Optional[Any] = None
        self._recording = False
        self._episode_frames = 0
        self._episodes_saved = 0
        self._latest_hut_image: Optional[np.ndarray] = None
        self._latest_top_image: Optional[np.ndarray] = None
        self._latest_tool_pose: Optional[np.ndarray] = None
        self._latest_action: Optional[np.ndarray] = None
        self._latest_hut_time: Optional[Any] = None
        self._latest_top_time: Optional[Any] = None
        self._latest_tool_pose_time: Optional[Any] = None
        self._latest_action_time: Optional[Any] = None
        self._task = str(self.get_parameter("task").value)

        self.create_subscription(Image, str(self.get_parameter("hut_image_topic").value), self._on_hut_image, 10)
        self.create_subscription(Image, str(self.get_parameter("top_image_topic").value), self._on_top_image, 10)
        self.create_subscription(PoseStamped, str(self.get_parameter("tool_pose_topic").value), self._on_tool_pose, 10)
        self.create_subscription(Float32MultiArray, str(self.get_parameter("action_topic").value), self._on_action, 10)
        self.create_subscription(
            String,
            str(self.get_parameter("task_instruction_topic").value),
            self._on_task_instruction,
            10,
        )

        self.create_service(Trigger, "~/start_episode", self._start_episode_service)
        self.create_service(Trigger, "~/stop_episode", self._stop_episode_service)

        fps = max(0.1, float(self.get_parameter("fps").value))
        self.create_timer(1.0 / fps, self._record_tick)

        if bool(self.get_parameter("auto_start").value):
            self._recording = True
            self.get_logger().info("LeRobot ROS 2 recorder armed; waiting for all required topics")
        else:
            self.get_logger().info("LeRobot ROS 2 recorder ready; call ~/start_episode to record")

    def _import_lerobot_dataset(self):
        try:
            module = importlib.import_module("lerobot.datasets.lerobot_dataset")
        except ModuleNotFoundError as exc:
            if exc.name != "lerobot" and not str(exc.name).startswith("lerobot."):
                raise
            try:
                module = importlib.import_module("opentau.datasets.lerobot_dataset")
            except ModuleNotFoundError:
                raise RuntimeError(
                    "lerobot or opentau is required for lerobot_ros2_recorder_node. Run this node with the "
                    "Python environment that contains LeRobot/OpenTau dataset dependencies."
                ) from exc
            self.get_logger().warn("Using opentau.datasets.lerobot_dataset because upstream lerobot was not found")
        return module.LeRobotDataset

    def _ensure_dataset(self) -> bool:
        if self._dataset is not None:
            return True
        if self._latest_hut_image is None or self._latest_top_image is None:
            return False

        LeRobotDataset = self._import_lerobot_dataset()
        features = {
            "observation.images.hut": {
                "dtype": "video",
                "shape": tuple(self._latest_hut_image.shape),
                "names": ["height", "width", "channel"],
            },
            "observation.images.top": {
                "dtype": "video",
                "shape": tuple(self._latest_top_image.shape),
                "names": ["height", "width", "channel"],
            },
            "observation.tool_pose": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            },
            "action": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["x", "y", "z", "rot_y_deg"],
            },
        }

        kwargs = {
            "repo_id": str(self.get_parameter("repo_id").value),
            "fps": int(round(float(self.get_parameter("fps").value))),
            "root": str(self.get_parameter("root").value),
            "robot_type": str(self.get_parameter("robot_type").value),
            "features": features,
            "use_videos": True,
        }
        try:
            self._dataset = LeRobotDataset.create(image_writer_threads=4, **kwargs)
        except TypeError:
            self._dataset = LeRobotDataset.create(**kwargs)

        self.get_logger().info(
            "Created LeRobot dataset "
            f"{self.get_parameter('repo_id').value!r} at {self.get_parameter('root').value!r}"
        )
        return True

    def _on_hut_image(self, msg: Image) -> None:
        self._latest_hut_image = self._decode_image(msg)
        self._latest_hut_time = self.get_clock().now()

    def _on_top_image(self, msg: Image) -> None:
        self._latest_top_image = self._decode_image(msg)
        self._latest_top_time = self.get_clock().now()

    def _on_tool_pose(self, msg: PoseStamped) -> None:
        pose = msg.pose
        self._latest_tool_pose = np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ],
            dtype=np.float32,
        )
        self._latest_tool_pose_time = self.get_clock().now()

    def _on_action(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 4:
            self.get_logger().warn("Ignoring action: expected [x, y, z, rot_y_deg]", throttle_duration_sec=2.0)
            return
        self._latest_action = np.array(msg.data[:4], dtype=np.float32)
        self._latest_action_time = self.get_clock().now()

    def _on_task_instruction(self, msg: String) -> None:
        if msg.data.strip():
            self._task = msg.data.strip()

    def _decode_image(self, msg: Image) -> np.ndarray:
        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding).lower()
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        if encoding in ("rgb8", "bgr8"):
            image = data.reshape(height, width, 3)
            if encoding == "bgr8":
                image = image[..., ::-1]
            return image.copy()

        if encoding in ("rgba8", "bgra8"):
            image = data.reshape(height, width, 4)
            if encoding == "bgra8":
                image = image[..., [2, 1, 0, 3]]
            return image[..., :3].copy()

        if encoding in ("mono8", "8uc1"):
            image = data.reshape(height, width)
            return np.repeat(image[..., None], 3, axis=2).copy()

        raise ValueError(f"unsupported image encoding {msg.encoding!r}")

    def _record_tick(self) -> None:
        if not self._recording:
            return

        num_episodes = int(self.get_parameter("num_episodes").value)
        if num_episodes > 0 and self._episodes_saved >= num_episodes:
            self._recording = False
            return

        missing = self._missing_or_stale_inputs()
        if missing:
            self.get_logger().warn(f"Waiting for fresh recorder inputs: {', '.join(missing)}", throttle_duration_sec=2.0)
            return

        if not self._ensure_dataset():
            return

        frame = {
            "observation.images.hut": self._latest_hut_image,
            "observation.images.top": self._latest_top_image,
            "observation.tool_pose": self._latest_tool_pose,
            "action": self._latest_action,
            "task": self._task,
        }
        self._dataset.add_frame(frame)
        self._episode_frames += 1

        max_frames = int(round(float(self.get_parameter("episode_time_s").value) * float(self.get_parameter("fps").value)))
        if max_frames > 0 and self._episode_frames >= max_frames:
            self._save_episode()

    def _missing_or_stale_inputs(self) -> list[str]:
        inputs = [
            ("hut_image", self._latest_hut_image, self._latest_hut_time),
            ("top_image", self._latest_top_image, self._latest_top_time),
            ("tool_pose", self._latest_tool_pose, self._latest_tool_pose_time),
            ("action", self._latest_action, self._latest_action_time),
        ]
        missing = []
        max_age_s = max(0.0, float(self.get_parameter("max_sample_age_s").value))
        now = self.get_clock().now()
        for name, value, stamp in inputs:
            if value is None or stamp is None:
                missing.append(name)
                continue
            age_s = (now - stamp).nanoseconds / 1e9
            if max_age_s > 0.0 and age_s > max_age_s:
                missing.append(name)
        return missing

    def _save_episode(self) -> None:
        if self._dataset is None or self._episode_frames == 0:
            return
        self._dataset.save_episode()
        self._episodes_saved += 1
        self.get_logger().info(f"Saved LeRobot episode {self._episodes_saved} with {self._episode_frames} frames")
        self._episode_frames = 0

        num_episodes = int(self.get_parameter("num_episodes").value)
        if num_episodes > 0 and self._episodes_saved >= num_episodes:
            self._recording = False
            self.get_logger().info("Configured episode count reached; recording stopped")

    def _start_episode_service(self, _request, response):
        self._recording = True
        response.success = True
        response.message = "recording started"
        return response

    def _stop_episode_service(self, _request, response):
        self._recording = False
        self._save_episode()
        response.success = True
        response.message = "recording stopped"
        return response

    def destroy_node(self) -> bool:
        if bool(self.get_parameter("save_partial_on_shutdown").value):
            self._save_episode()

        if self._dataset is not None and hasattr(self._dataset, "finalize"):
            self._dataset.finalize()

        if self._dataset is not None and bool(self.get_parameter("push_to_hub_on_shutdown").value):
            self._dataset.push_to_hub()

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeRobotRos2RecorderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
