import os
import math
import random
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from kaivuri_bringup.project_paths import add_project_import_path, resolve_project_root

_N_ACTIVE = 4
IK_ORIGIN_Z_IN_EXCAVATOR_FRAME = -0.05042
CUBE_TOP_Z_IN_EXCAVATOR_FRAME = 0.025000000372529023
DEFAULT_CUBE_TOP_Z_IN_IK_FRAME = CUBE_TOP_Z_IN_EXCAVATOR_FRAME - IK_ORIGIN_Z_IN_EXCAVATOR_FRAME
FRONT_ANGLE_MIN_DEG = -80.0
FRONT_ANGLE_MAX_DEG = 80.0


class SuccessCubeRelocatorNode(Node):
    """Move the cube to a new fixed-base reachable top-center pose after success."""

    def __init__(self) -> None:
        super().__init__("success_cube_relocator_node")
        self.declare_parameter("cube_pose_topic", "/kaivuri/cube_pose")
        self.declare_parameter("cube_command_topic", "/kaivuri/cube_pose_cmd")
        self.declare_parameter("episode_event_topic", "/kaivuri/episode_event")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("project_root", os.environ.get("KAIVURI_PROJECT_ROOT", "/work"))
        self.declare_parameter("control_config_file", "configuration_files/profiles/rpi/control_config.yaml")
        self.declare_parameter("frame_id", "excavator")
        self.declare_parameter("publish_initial_pose", False)
        self.declare_parameter("cube_pose_is_top_center", True)
        self.declare_parameter("cube_size_m", 0.05)
        self.declare_parameter("touch_clearance_m", 0.01)
        self.declare_parameter("rot_y_deg", 0.0)
        self.declare_parameter("min_move_distance_m", 0.08)
        self.declare_parameter("max_attempts", 200)
        self.declare_parameter("relocate_delay_s", 0.5)
        self.declare_parameter("use_ik_reachability", False)
        self.declare_parameter("relocate_invalid_cube_pose", True)
        self.declare_parameter("relocate_on_failure", True)

        # Conservative fixed-base workspace. These are cube top-center
        # positions in the IK frame, not track-drive targets. Polar sampling is
        # limited to the front workspace: -80 to +80 degrees around +X.
        self.declare_parameter("use_polar_workspace", True)
        self.declare_parameter("radius_min", 0.45)
        self.declare_parameter("radius_max", 0.68)
        self.declare_parameter("angle_min_deg", FRONT_ANGLE_MIN_DEG)
        self.declare_parameter("angle_max_deg", FRONT_ANGLE_MAX_DEG)
        self.declare_parameter("x_min", 0.45)
        self.declare_parameter("x_max", 0.68)
        self.declare_parameter("y_min", -0.15)
        self.declare_parameter("y_max", 0.15)
        self.declare_parameter("z", DEFAULT_CUBE_TOP_Z_IN_IK_FRAME)
        self.declare_parameter("use_measured_cube_z", True)

        self._project_root = resolve_project_root(str(self.get_parameter("project_root").value))
        add_project_import_path(self._project_root)
        self._reachability_enabled = False
        self._check_reachability = None
        self._model = None
        self._ik_cfg = None
        if bool(self.get_parameter("use_ik_reachability").value):
            self._try_enable_ik_reachability()

        self._latest_joint_angles: Optional[np.ndarray] = None
        self._last_cube_top_center_ik: Optional[np.ndarray] = None
        self._last_measured_cube_pose: Optional[np.ndarray] = None
        self._success_episodes: set[int] = set()
        self._relocated_episodes: set[int] = set()
        self._pending_success_episode: Optional[int] = None
        self._pending_relocation_reason: Optional[str] = None
        self._relocate_after_time = None

        cube_pose_topic = str(self.get_parameter("cube_pose_topic").value)
        cube_command_topic = str(self.get_parameter("cube_command_topic").value)
        event_topic = str(self.get_parameter("episode_event_topic").value)
        joint_states_topic = str(self.get_parameter("joint_states_topic").value)

        self._cube_pub = self.create_publisher(PoseStamped, cube_command_topic, 10)
        self.create_subscription(PoseStamped, cube_pose_topic, self._on_cube_pose, 10)
        self.create_subscription(String, event_topic, self._on_episode_event, 10)
        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)

        self._published_initial = False
        if bool(self.get_parameter("publish_initial_pose").value):
            self.create_timer(0.2, self._publish_initial_once)
        self.create_timer(0.1, self._relocation_tick)

        self.get_logger().info(
            f"Relocating cube top-center on touch_success within fixed-base IK workspace "
            f"r=[{self.get_parameter('radius_min').value}, {self.get_parameter('radius_max').value}] "
            f"angle=[{self.get_parameter('angle_min_deg').value}, {self.get_parameter('angle_max_deg').value}] deg"
        )

    def _try_enable_ik_reachability(self) -> None:
        try:
            from modules.ik import IKConfig, load_excavator_model
            from modules.reachability import check_reachability
        except Exception as exc:
            self.get_logger().warn(
                f"IK reachability disabled because modules.ik could not be imported: {exc}"
            )
            return

        try:
            import yaml
            control_config_path = self._project_root / str(
                self.get_parameter("control_config_file").value
            )
            self._model = load_excavator_model(str(control_config_path))
            self._ik_cfg = self._load_ik_config(IKConfig, yaml, control_config_path)
            self._check_reachability = check_reachability
            self._reachability_enabled = True
            self.get_logger().info("IK reachability filtering enabled")
        except Exception as exc:
            self.get_logger().warn(f"IK reachability disabled: {exc}")

    def _load_ik_config(self, ik_config_cls, yaml_module, control_config_path):
        with control_config_path.open("r", encoding="utf-8") as f:
            cfg = yaml_module.safe_load(f) or {}
        ik_cfg = cfg.get("ik", {})
        params = ik_cfg.get("params", {})
        return ik_config_cls(
            method=str(ik_cfg.get("method", "dls")),
            k_val=float(params.get("k_val", 1.0)),
            lambda_val=float(params.get("lambda_val", 1e-3)),
            min_singular_value=float(params.get("min_singular_value", 1e-5)),
            enable_velocity_limiting=True,
            enable_joint_limit_avoidance=True,
            enable_adaptive_damping=bool(ik_cfg.get("enable_adaptive_damping", True)),
            adaptive_damping_max_multiplier=float(ik_cfg.get("adaptive_damping_max_multiplier", 2.0)),
            condition_number_threshold=float(ik_cfg.get("condition_number_threshold", 40.0)),
        )

    def _publish_initial_once(self) -> None:
        if self._published_initial:
            return
        self._published_initial = True
        self._publish_new_cube_pose("initial")

    def _on_joint_state(self, msg: JointState) -> None:
        if len(msg.position) < _N_ACTIVE:
            return
        self._latest_joint_angles = np.asarray(
            list(msg.position[:_N_ACTIVE]),
            dtype=np.float32,
        )

    def _on_cube_pose(self, msg: PoseStamped) -> None:
        self._last_measured_cube_pose = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )
        if self._last_cube_top_center_ik is None:
            self._last_cube_top_center_ik = self._last_measured_cube_pose.copy()
        if (
            bool(self.get_parameter("relocate_invalid_cube_pose").value)
            and self._pending_relocation_reason is None
            and not self._inside_configured_workspace(self._last_measured_cube_pose)
        ):
            delay_s = max(0.0, float(self.get_parameter("relocate_delay_s").value))
            self._pending_relocation_reason = "invalid_cube_pose"
            self._pending_success_episode = None
            self._relocate_after_time = self.get_clock().now() + Duration(seconds=delay_s)
            self.get_logger().warn(
                f"Scheduled cube relocation for invalid pose outside workspace: {np.round(self._last_measured_cube_pose, 4)}",
                throttle_duration_sec=2.0,
            )

    def _on_episode_event(self, msg: String) -> None:
        try:
            episode_text, event = msg.data.split(":", 1)
            episode_id = int(episode_text)
        except ValueError:
            return

        if event == "touch_success":
            if episode_id in self._relocated_episodes:
                return
            self._success_episodes.add(episode_id)
            self._pending_success_episode = episode_id
            self._pending_relocation_reason = f"touch_success episode={episode_id}"
            delay_s = max(0.0, float(self.get_parameter("relocate_delay_s").value))
            self._relocate_after_time = self.get_clock().now() + Duration(seconds=delay_s)
            self.get_logger().info(f"Scheduled cube relocation after touch_success episode={episode_id}")
            return

        if event.startswith("failure") and bool(self.get_parameter("relocate_on_failure").value):
            if episode_id in self._relocated_episodes:
                return
            self._pending_success_episode = episode_id
            self._pending_relocation_reason = f"{event} episode={episode_id}"
            delay_s = max(0.0, float(self.get_parameter("relocate_delay_s").value))
            self._relocate_after_time = self.get_clock().now() + Duration(seconds=delay_s)
            self.get_logger().warn(f"Scheduled cube relocation after {event} episode={episode_id}")
            return

        if (
            event == "episode_end"
            and self._pending_success_episode == episode_id
            and episode_id in self._success_episodes
            and episode_id not in self._relocated_episodes
        ):
            delay_s = max(0.0, float(self.get_parameter("relocate_delay_s").value))
            self._relocate_after_time = self.get_clock().now() + Duration(seconds=delay_s)

    def _relocation_tick(self) -> None:
        if self._relocate_after_time is None or self._pending_relocation_reason is None:
            return
        if self.get_clock().now() < self._relocate_after_time:
            return

        episode_id = self._pending_success_episode
        reason = self._pending_relocation_reason
        if self._publish_new_cube_pose(reason):
            self._pending_success_episode = None
            self._pending_relocation_reason = None
            self._relocate_after_time = None
            if episode_id is not None:
                self._relocated_episodes.add(episode_id)

    def _publish_new_cube_pose(self, reason: str) -> bool:
        top_center_ik = self._sample_reachable_cube_top_center_ik()
        if top_center_ik is None:
            self.get_logger().error("Could not find a reachable cube pose inside configured workspace")
            return False

        self._last_cube_top_center_ik = top_center_ik.copy()
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter("frame_id").value)
        msg.pose.position.x = float(top_center_ik[0])
        msg.pose.position.y = float(top_center_ik[1])
        msg.pose.position.z = float(top_center_ik[2])
        msg.pose.orientation.w = 1.0
        self._cube_pub.publish(msg)
        self.get_logger().info(f"Published cube top-center IK pose after {reason}: {np.round(top_center_ik, 4)}")
        return True

    def _sample_reachable_cube_top_center_ik(self) -> Optional[np.ndarray]:
        max_attempts = max(1, int(self.get_parameter("max_attempts").value))
        min_move = max(0.0, float(self.get_parameter("min_move_distance_m").value))
        q_seed = self._joint_seed() if self._reachability_enabled else None

        for _ in range(max_attempts):
            x, y = self._sample_xy()
            top_center_ik = np.array(
                [
                    x,
                    y,
                    self._command_z(),
                ],
                dtype=np.float32,
            )
            if (
                self._last_cube_top_center_ik is not None
                and float(np.linalg.norm(top_center_ik[:2] - self._last_cube_top_center_ik[:2])) < min_move
            ):
                continue
            if self._touch_pose_is_reachable(top_center_ik, q_seed):
                return top_center_ik
        return None

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
        angle_min, angle_max = self._front_angle_bounds()

        radius = math.sqrt(random.uniform(radius_min * radius_min, radius_max * radius_max))
        angle = math.radians(random.uniform(angle_min, angle_max))
        return radius * math.cos(angle), radius * math.sin(angle)

    def _front_angle_bounds(self) -> tuple[float, float]:
        angle_min = max(FRONT_ANGLE_MIN_DEG, float(self.get_parameter("angle_min_deg").value))
        angle_max = min(FRONT_ANGLE_MAX_DEG, float(self.get_parameter("angle_max_deg").value))
        if angle_min > angle_max:
            return FRONT_ANGLE_MIN_DEG, FRONT_ANGLE_MAX_DEG
        return angle_min, angle_max

    def _command_z(self) -> float:
        if (
            bool(self.get_parameter("use_measured_cube_z").value)
            and self._last_measured_cube_pose is not None
        ):
            return float(self._last_measured_cube_pose[2])
        return float(self.get_parameter("z").value)

    def _joint_seed(self) -> np.ndarray:
        if self._latest_joint_angles is not None:
            return self._latest_joint_angles.copy()
        return np.radians(np.array([0.0, -31.5, 71.5, -43.3], dtype=np.float32))

    def _touch_pose_is_reachable(self, cube_top_center_ik: np.ndarray, joint_seed: Optional[np.ndarray]) -> bool:
        if not self._inside_configured_workspace(cube_top_center_ik):
            return False
        if not self._reachability_enabled:
            return True

        if bool(self.get_parameter("cube_pose_is_top_center").value):
            top = cube_top_center_ik
        else:
            top = cube_top_center_ik + np.array(
                [0.0, 0.0, float(self.get_parameter("cube_size_m").value) * 0.5],
                dtype=np.float32,
            )
        target = top + np.array(
            [0.0, 0.0, float(self.get_parameter("touch_clearance_m").value)],
            dtype=np.float32,
        )
        result = self._check_reachability(
            self._model,
            self._ik_cfg,
            joint_seed,
            target,
            float(self.get_parameter("rot_y_deg").value),
            pos_tol=0.02,
            max_iters=80,
            dt=0.01,
            position_only=True,
        )
        return bool(result.reachable)

    def _inside_configured_workspace(self, cube_top_center_ik: np.ndarray) -> bool:
        if bool(self.get_parameter("use_polar_workspace").value):
            radius = float(np.linalg.norm(cube_top_center_ik[:2]))
            angle_deg = math.degrees(math.atan2(float(cube_top_center_ik[1]), float(cube_top_center_ik[0])))
            angle_min, angle_max = self._front_angle_bounds()
            return (
                float(self.get_parameter("radius_min").value) <= radius <= float(self.get_parameter("radius_max").value)
                and angle_min <= angle_deg <= angle_max
            )
        return (
            float(self.get_parameter("x_min").value) <= float(cube_top_center_ik[0]) <= float(self.get_parameter("x_max").value)
            and float(self.get_parameter("y_min").value) <= float(cube_top_center_ik[1]) <= float(self.get_parameter("y_max").value)
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SuccessCubeRelocatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
