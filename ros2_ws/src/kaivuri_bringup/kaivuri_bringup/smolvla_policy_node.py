from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, String


DEFAULT_JOINT_ORDER = [
    "revolute_carriage",
    "revolute_lift",
    "revolute_tilt",
    "revolute_tool",
    "revolute_gripper",
    "revolute_claw_1",
    "revolute_claw_2",
]


class SmolVLAPolicyNode(Node):
    """Run a fine-tuned SmolVLA policy and publish IK end-effector targets."""

    def __init__(self) -> None:
        super().__init__("smolvla_policy_node")

        self.declare_parameter(
            "checkpoint",
            "/mnt/c/users/sh25016/kaivuri_smolvla_test/checkpoints/000020/pretrained_model",
        )
        self.declare_parameter("dataset_root", "/mnt/c/users/sh25016/kaivuri_lerobot_dataset")
        self.declare_parameter("dataset_repo_id", "kaivuri_lerobot_dataset")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("instruction_topic", "/kaivuri/task_instruction")
        self.declare_parameter("target_pose_y_topic", "/kaivuri/target_pose_y")
        self.declare_parameter("rate_hz", 3.0)
        self.declare_parameter("device", "auto")
        self.declare_parameter("instruction", "touch the top of the red cube")
        self.declare_parameter("joint_order", DEFAULT_JOINT_ORDER)
        self.declare_parameter("max_action_jump_m", 0.05)
        self.declare_parameter("x_min", 0.35)
        self.declare_parameter("x_max", 0.75)
        self.declare_parameter("y_min", -0.25)
        self.declare_parameter("y_max", 0.25)
        self.declare_parameter("z_min", 0.02)
        self.declare_parameter("z_max", 0.35)
        self.declare_parameter("rot_y_min_deg", -90.0)
        self.declare_parameter("rot_y_max_deg", 90.0)

        self._image_tensor: Optional[torch.Tensor] = None
        self._state_tensor: Optional[torch.Tensor] = None
        self._last_action: Optional[np.ndarray] = None
        self._instruction = str(self.get_parameter("instruction").value)

        self._device = self._resolve_device(str(self.get_parameter("device").value))
        self._joint_order = [str(v) for v in list(self.get_parameter("joint_order").value)]

        self.get_logger().info(f"Loading SmolVLA checkpoint on {self._device}")
        self._load_policy()
        self.get_logger().info("SmolVLA policy ready")

        image_topic = str(self.get_parameter("image_topic").value)
        joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        instruction_topic = str(self.get_parameter("instruction_topic").value)
        target_pose_y_topic = str(self.get_parameter("target_pose_y_topic").value)

        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)
        self.create_subscription(String, instruction_topic, self._on_instruction, 10)
        self._target_pub = self.create_publisher(Float32MultiArray, target_pose_y_topic, 10)

        rate_hz = max(0.1, float(self.get_parameter("rate_hz").value))
        self.create_timer(1.0 / rate_hz, self._tick)

    def _resolve_device(self, configured: str) -> str:
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured

    def _load_policy(self) -> None:
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.datasets.factory import make_dataset
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.policies.pretrained import PreTrainedConfig

        checkpoint = str(self.get_parameter("checkpoint").value)
        cfg = TrainPipelineConfig.from_pretrained(checkpoint)
        cfg.dataset.root = str(self.get_parameter("dataset_root").value)
        cfg.dataset.repo_id = str(self.get_parameter("dataset_repo_id").value)
        cfg.dataset.video_backend = "torchcodec"
        cfg.batch_size = 1
        cfg.policy.device = self._device
        cfg.rename_map = {"observation.images.front": "observation.images.camera1"}

        dataset = make_dataset(cfg)
        policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
        policy_cfg.device = self._device

        self._policy = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=cfg.rename_map)
        self._policy.eval()

        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )

    def _on_instruction(self, msg: String) -> None:
        if msg.data.strip():
            self._instruction = msg.data.strip()

    def _on_joint_state(self, msg: JointState) -> None:
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        positions = []
        velocities = []
        for joint_name in self._joint_order:
            idx = name_to_index.get(joint_name)
            if idx is None:
                positions.append(0.0)
                velocities.append(0.0)
                continue
            positions.append(float(msg.position[idx]) if idx < len(msg.position) else 0.0)
            velocities.append(float(msg.velocity[idx]) if idx < len(msg.velocity) else 0.0)

        state = torch.tensor(positions + velocities, dtype=torch.float32, device=self._device)
        self._state_tensor = state

    def _on_image(self, msg: Image) -> None:
        try:
            image = self._decode_image(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode image: {exc}", throttle_duration_sec=2.0)
            return

        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        self._image_tensor = tensor.to(self._device, dtype=torch.float32) / 255.0

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

    def _tick(self) -> None:
        if self._image_tensor is None or self._state_tensor is None:
            self.get_logger().warn(
                "Waiting for image and joint state before running SmolVLA",
                throttle_duration_sec=2.0,
            )
            return

        batch = {
            "observation.images.front": self._image_tensor.unsqueeze(0),
            "observation.state": self._state_tensor.unsqueeze(0),
            "task": self._instruction,
        }

        try:
            with torch.no_grad():
                processed = self._preprocessor(batch)
                action = self._policy.select_action(processed)
                action = self._postprocessor(action)
        except Exception as exc:
            self.get_logger().error(f"SmolVLA inference failed: {exc}", throttle_duration_sec=2.0)
            return

        action_np = action.detach().cpu().numpy().reshape(-1)
        if action_np.size < 4:
            self.get_logger().error(f"Policy returned too few action values: {action_np}")
            return

        command = self._clamp_action(action_np[:4].astype(np.float32))
        self._last_action = command.copy()

        msg = Float32MultiArray()
        msg.data = [float(v) for v in command]
        self._target_pub.publish(msg)

    def _clamp_action(self, action: np.ndarray) -> np.ndarray:
        action[0] = np.clip(
            action[0],
            float(self.get_parameter("x_min").value),
            float(self.get_parameter("x_max").value),
        )
        action[1] = np.clip(
            action[1],
            float(self.get_parameter("y_min").value),
            float(self.get_parameter("y_max").value),
        )
        action[2] = np.clip(
            action[2],
            float(self.get_parameter("z_min").value),
            float(self.get_parameter("z_max").value),
        )
        action[3] = np.clip(
            action[3],
            float(self.get_parameter("rot_y_min_deg").value),
            float(self.get_parameter("rot_y_max_deg").value),
        )

        if self._last_action is not None:
            max_jump = max(0.0, float(self.get_parameter("max_action_jump_m").value))
            delta = action[:3] - self._last_action[:3]
            distance = float(np.linalg.norm(delta))
            if max_jump > 0.0 and distance > max_jump:
                action[:3] = self._last_action[:3] + delta * (max_jump / distance)

        return action


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SmolVLAPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
