from __future__ import annotations

import importlib
from typing import Any, Optional

import av
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float32MultiArray, String


av.logging.set_level(av.logging.ERROR)

DEFAULT_JOINT_ORDER = [
    "revolute_carriage",
    "revolute_lift",
    "revolute_tilt",
    "revolute_tool",
    "revolute_gripper",
    "revolute_claw_1",
    "revolute_claw_2",
]

OBS_IMAGE_1_KEY = "observation.images.OBS_IMAGE_1"  # top view
OBS_IMAGE_2_KEY = "observation.images.OBS_IMAGE_2"  # wrist/front view
TOOL_POSE_KEY = "observation.tool_pose"
STATE_KEY = "observation.state"


def _import_required(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise RuntimeError(
            f"{module_name} is required for smolvla_policy_node. "
            "Run it with the Python environment that has the SmolVLA/Lerobot "
            "dependencies installed, or install the missing package there."
        ) from exc


class SmolVLAPolicyNode(Node):
    """Run a fine-tuned SmolVLA policy and publish IK end-effector targets.

    By default the policy output is interpreted as an end-effector delta:
    [dx, dy, dz, d_rot_y_deg]. The IK controller subscribes to that delta
    topic and turns it into an absolute target from the live tool pose.
    """

    def __init__(self) -> None:
        super().__init__("smolvla_policy_node")
        self._torch = _import_required("torch")

        self.declare_parameter(
            "checkpoint",
            "/mnt/c/Users/sh25016/kaivuri_smolvla_run_002/checkpoints/010000/pretrained_model",
        )
        self.declare_parameter("dataset_root", "/mnt/c/users/sh25016/kaivuri_lerobot_live_014")
        self.declare_parameter("dataset_repo_id", "kaivuri/ros2_recording_2")
        self.declare_parameter("obs_image_1_topic", "/camera/camera1/color/rgb_compressed")
        self.declare_parameter("obs_image_2_topic", "/camera/camera/color/rgb_compressed")
        self.declare_parameter("tool_pose_topic", "/kaivuri/tool_pose")
        self.declare_parameter("observation_state_source", "tool_pose")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("instruction_topic", "/kaivuri/task_instruction")
        self.declare_parameter("target_pose_y_topic", "/kaivuri/target_pose_y")
        self.declare_parameter("target_pose_delta_y_topic", "/kaivuri/target_pose_delta_y")
        self.declare_parameter("action_mode", "delta")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("device", "auto")
        self.declare_parameter("instruction", "touch the top of the red cube")
        self.declare_parameter("joint_order", DEFAULT_JOINT_ORDER)
        self.declare_parameter("max_action_jump_m", 0.05)
        self.declare_parameter("max_delta_translation_m", 0.05)
        self.declare_parameter("max_delta_rot_y_deg", 5.0)
        self.declare_parameter("x_min", 0.35)
        self.declare_parameter("x_max", 0.75)
        self.declare_parameter("y_min", -0.25)
        self.declare_parameter("y_max", 0.25)
        self.declare_parameter("z_min", 0.02)
        self.declare_parameter("z_max", 0.35)
        self.declare_parameter("rot_y_min_deg", -80.0)
        self.declare_parameter("rot_y_max_deg", 80.0)

        self._obs_image_1_tensor: Optional[Any] = None
        self._obs_image_2_tensor: Optional[Any] = None
        self._obs_image_1_compressed_msg: Optional[CompressedImage] = None
        self._obs_image_2_compressed_msg: Optional[CompressedImage] = None
        self._obs_image_1_decoded_stamp: Optional[tuple[int, int]] = None
        self._obs_image_2_decoded_stamp: Optional[tuple[int, int]] = None
        self._obs_image_1_h264_decoder = av.CodecContext.create("h264", "r")
        self._obs_image_2_h264_decoder = av.CodecContext.create("h264", "r")
        self._state_tensor: Optional[Any] = None
        self._state_input_key = TOOL_POSE_KEY
        self._last_action: Optional[np.ndarray] = None
        self._instruction = str(self.get_parameter("instruction").value)

        self._device = self._resolve_device(str(self.get_parameter("device").value))
        self._joint_order = [str(v) for v in list(self.get_parameter("joint_order").value)]
        self._observation_state_source = str(
            self.get_parameter("observation_state_source").value
        ).strip().lower()
        if self._observation_state_source not in ("tool_pose", "joint_states"):
            self.get_logger().warn(
                "Unsupported observation_state_source="
                f"{self._observation_state_source!r}; falling back to 'tool_pose'"
            )
            self._observation_state_source = "tool_pose"
        self._action_mode = str(self.get_parameter("action_mode").value).strip().lower()
        if self._action_mode not in ("delta", "absolute"):
            self.get_logger().warn(
                f"Unsupported action_mode={self._action_mode!r}; falling back to 'delta'"
            )
            self._action_mode = "delta"

        self.get_logger().info(f"Loading SmolVLA checkpoint on {self._device}")
        self._load_policy()
        self.get_logger().info("SmolVLA policy ready")

        obs_image_1_topic = str(self.get_parameter("obs_image_1_topic").value).strip()
        obs_image_2_topic = str(self.get_parameter("obs_image_2_topic").value).strip()
        tool_pose_topic = str(self.get_parameter("tool_pose_topic").value)
        joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        instruction_topic = str(self.get_parameter("instruction_topic").value)
        target_pose_y_topic = str(self.get_parameter("target_pose_y_topic").value)
        target_pose_delta_y_topic = str(self.get_parameter("target_pose_delta_y_topic").value)
        target_topic = target_pose_delta_y_topic if self._action_mode == "delta" else target_pose_y_topic
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._obs_image_1_compressed = self._is_compressed_topic(obs_image_1_topic)
        self._obs_image_2_compressed = self._is_compressed_topic(obs_image_2_topic)
        if self._obs_image_1_compressed:
            self.create_subscription(CompressedImage, obs_image_1_topic, self._on_obs_image_1_compressed, image_qos)
        else:
            self.create_subscription(Image, obs_image_1_topic, self._on_obs_image_1, image_qos)
        if self._obs_image_2_compressed:
            self.create_subscription(CompressedImage, obs_image_2_topic, self._on_obs_image_2_compressed, image_qos)
        else:
            self.create_subscription(Image, obs_image_2_topic, self._on_obs_image_2, image_qos)
        if self._observation_state_source == "tool_pose":
            self.create_subscription(PoseStamped, tool_pose_topic, self._on_tool_pose, 10)
            state_topic = tool_pose_topic
        else:
            self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)
            state_topic = joint_states_topic
        self.create_subscription(String, instruction_topic, self._on_instruction, 10)
        self._target_pub = self.create_publisher(Float32MultiArray, target_topic, 10)
        self.get_logger().info(
            f"SmolVLA action_mode={self._action_mode}; publishing actions to {target_topic}"
        )
        self.get_logger().info(
            f"Camera normalization: {OBS_IMAGE_1_KEY} <- {obs_image_1_topic}; "
            f"{OBS_IMAGE_2_KEY} <- {obs_image_2_topic}"
        )
        self.get_logger().info(
            f"Observation state: {self._state_input_key} <- {state_topic} "
            f"({self._observation_state_source})"
        )

        rate_hz = max(0.1, float(self.get_parameter("rate_hz").value))
        self.create_timer(1.0 / rate_hz, self._tick)

    def _resolve_device(self, configured: str) -> str:
        if configured == "auto":
            return "cuda" if self._torch.cuda.is_available() else "cpu"
        return configured

    def _load_policy(self) -> None:
        try:
            from lerobot.configs.train import TrainPipelineConfig
            from lerobot.datasets.factory import make_dataset
            from lerobot.policies import make_policy, make_pre_post_processors
            from lerobot.policies.pretrained import PreTrainedConfig
        except ModuleNotFoundError as exc:
            if exc.name != "lerobot" and not str(exc.name).startswith("lerobot."):
                raise
            raise RuntimeError(
                "lerobot is required for smolvla_policy_node. Run it with the "
                "Python environment that contains the trained-policy dependencies."
            ) from exc

        checkpoint = str(self.get_parameter("checkpoint").value)
        cfg = TrainPipelineConfig.from_pretrained(checkpoint)
        cfg.dataset.root = str(self.get_parameter("dataset_root").value)
        cfg.dataset.repo_id = str(self.get_parameter("dataset_repo_id").value)
        cfg.dataset.video_backend = "torchcodec"
        cfg.batch_size = 1
        cfg.policy.device = self._device
        cfg.rename_map = dict(getattr(cfg, "rename_map", {}) or {})
        if self._observation_state_source == "tool_pose":
            cfg.rename_map.setdefault(TOOL_POSE_KEY, STATE_KEY)
        self._state_input_key = self._input_key_for_policy_state(cfg.rename_map)
        if self._observation_state_source == "joint_states":
            self._state_input_key = STATE_KEY

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

    def _input_key_for_policy_state(self, rename_map: dict[str, str]) -> str:
        for source_key, target_key in rename_map.items():
            if target_key == STATE_KEY:
                return source_key
        return STATE_KEY

    def _is_compressed_topic(self, topic: str) -> bool:
        topic = topic.rstrip("/")
        return topic.endswith("_compressed") or topic.endswith("/compressed")

    def _on_instruction(self, msg: String) -> None:
        if msg.data.strip():
            self._instruction = msg.data.strip()

    def _on_tool_pose(self, msg: PoseStamped) -> None:
        pose = msg.pose
        self._state_tensor = self._torch.tensor(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ],
            dtype=self._torch.float32,
            device=self._device,
        )

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

        state = self._torch.tensor(
            positions + velocities,
            dtype=self._torch.float32,
            device=self._device,
        )
        self._state_tensor = state

    def _on_obs_image_1(self, msg: Image) -> None:
        self._obs_image_1_tensor = self._image_msg_to_tensor(msg)
        self._obs_image_1_compressed_msg = None

    def _on_obs_image_2(self, msg: Image) -> None:
        self._obs_image_2_tensor = self._image_msg_to_tensor(msg)
        self._obs_image_2_compressed_msg = None

    def _on_obs_image_1_compressed(self, msg: CompressedImage) -> None:
        self._obs_image_1_compressed_msg = msg

    def _on_obs_image_2_compressed(self, msg: CompressedImage) -> None:
        self._obs_image_2_compressed_msg = msg

    def _image_msg_to_tensor(self, msg: Image):
        try:
            image = self._decode_image(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode image: {exc}", throttle_duration_sec=2.0)
            return None

        tensor = self._torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return tensor.to(self._device, dtype=self._torch.float32) / 255.0

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

    def _compressed_image_msg_to_tensor(
        self,
        msg: CompressedImage,
        codec: av.CodecContext,
    ):
        try:
            image = self._decode_compressed_image(msg, codec)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode compressed image: {exc}", throttle_duration_sec=2.0)
            return None

        tensor = self._torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return tensor.to(self._device, dtype=self._torch.float32) / 255.0

    def _decode_compressed_image(self, msg: CompressedImage, codec: av.CodecContext) -> np.ndarray:
        fmt = str(msg.format).lower()
        if "h264" not in fmt:
            raise ValueError(f"unsupported compressed image format {msg.format!r}")

        packet = av.Packet(bytes(msg.data))
        frames = codec.decode(packet)
        if not frames:
            raise ValueError("H264 packet did not produce a frame")
        return np.ascontiguousarray(frames[-1].to_ndarray(format="rgb24"))

    def _stamp_key(self, msg: CompressedImage) -> tuple[int, int]:
        return (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))

    def _latest_obs_image_1_tensor(self):
        if not self._obs_image_1_compressed:
            return self._obs_image_1_tensor
        if self._obs_image_1_compressed_msg is None:
            return None
        stamp = self._stamp_key(self._obs_image_1_compressed_msg)
        if self._obs_image_1_tensor is not None and self._obs_image_1_decoded_stamp == stamp:
            return self._obs_image_1_tensor
        self._obs_image_1_tensor = self._compressed_image_msg_to_tensor(
            self._obs_image_1_compressed_msg,
            self._obs_image_1_h264_decoder,
        )
        self._obs_image_1_decoded_stamp = stamp if self._obs_image_1_tensor is not None else None
        return self._obs_image_1_tensor

    def _latest_obs_image_2_tensor(self):
        if not self._obs_image_2_compressed:
            return self._obs_image_2_tensor
        if self._obs_image_2_compressed_msg is None:
            return None
        stamp = self._stamp_key(self._obs_image_2_compressed_msg)
        if self._obs_image_2_tensor is not None and self._obs_image_2_decoded_stamp == stamp:
            return self._obs_image_2_tensor
        self._obs_image_2_tensor = self._compressed_image_msg_to_tensor(
            self._obs_image_2_compressed_msg,
            self._obs_image_2_h264_decoder,
        )
        self._obs_image_2_decoded_stamp = stamp if self._obs_image_2_tensor is not None else None
        return self._obs_image_2_tensor

    def _tick(self) -> None:
        obs_image_1_tensor = self._latest_obs_image_1_tensor()
        obs_image_2_tensor = self._latest_obs_image_2_tensor()
        if obs_image_1_tensor is None or obs_image_2_tensor is None or self._state_tensor is None:
            self.get_logger().warn(
                "Waiting for OBS_IMAGE_1, OBS_IMAGE_2, and observation state before running SmolVLA",
                throttle_duration_sec=2.0,
            )
            return

        batch = {
            OBS_IMAGE_1_KEY: obs_image_1_tensor.unsqueeze(0),
            OBS_IMAGE_2_KEY: obs_image_2_tensor.unsqueeze(0),
            self._state_input_key: self._state_tensor.unsqueeze(0),
            "task": self._instruction,
        }

        try:
            with self._torch.no_grad():
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

        if self._action_mode == "delta":
            command = self._clamp_delta_action(action_np[:4].astype(np.float32))
        else:
            command = self._clamp_absolute_action(action_np[:4].astype(np.float32))
        self._last_action = command.copy()

        msg = Float32MultiArray()
        msg.data = [float(v) for v in command]
        self._target_pub.publish(msg)

    def _clamp_delta_action(self, action: np.ndarray) -> np.ndarray:
        max_translation = max(0.0, float(self.get_parameter("max_delta_translation_m").value))
        if max_translation > 0.0:
            distance = float(np.linalg.norm(action[:3]))
            if distance > max_translation:
                action[:3] = action[:3] * (max_translation / distance)

        max_rot = max(0.0, float(self.get_parameter("max_delta_rot_y_deg").value))
        if max_rot > 0.0:
            action[3] = np.clip(action[3], -max_rot, max_rot)

        return action

    def _clamp_absolute_action(self, action: np.ndarray) -> np.ndarray:
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
