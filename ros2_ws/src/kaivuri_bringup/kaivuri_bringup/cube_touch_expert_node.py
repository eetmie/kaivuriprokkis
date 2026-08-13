import math
from enum import Enum
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from lerobot_interfaces.srv import EndEpisode, StartEpisode
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


_LOCAL_Z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
_Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float32)
DEFAULT_CUBE_POSE_Z_OFFSET_M = 0.0


class Stage(str, Enum):
    IDLE = "idle"
    WAIT_CUBE_HIDDEN = "wait_cube_hidden"
    HOME = "home"
    WAIT_CUBE_VISIBLE = "wait_cube_visible"
    APPROACH = "approach"
    DESCEND = "descend"
    HOLD = "hold"
    RETRACT = "retract"
    RETURN_HOME = "return_home"
    DONE = "done"


class CubeTouchExpertNode(Node):
    """Generate expert end-effector targets from a cube pose.

    The node expects Isaac Sim, or another task generator, to publish a cube pose.
    It samples the current tool pose at the policy/data rate, publishes that
    synced observation state, and publishes the next absolute end-effector target
    [x, y, z, rot_y_deg] to the IK controller.
    By default, recording ends while the tool is still holding the touch pose;
    retract happens outside the episode.
    """

    def __init__(self) -> None:
        super().__init__("cube_touch_expert_node")
        self.declare_parameter("cube_pose_topic", "/kaivuri/cube_pose")
        self.declare_parameter("tool_pose_topic", "/kaivuri/tool_pose")
        self.declare_parameter("expert_observation_state_topic", "/kaivuri/expert_observation_state")
        self.declare_parameter("target_pose_y_topic", "/kaivuri/target_pose_y")
        self.declare_parameter("max_rot_y_deg_per_step", 5.0)
        self.declare_parameter("episode_event_topic", "/kaivuri/episode_event")
        self.declare_parameter("task_instruction_topic", "/kaivuri/task_instruction")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("speed_mps", 0.04)
        self.declare_parameter("position_tolerance_m", 0.02)
        self.declare_parameter("approach_xy_tolerance_m", 0.005)
        self.declare_parameter("approach_z_tolerance_m", 0.05)
        self.declare_parameter("touch_xy_tolerance_m", 0.02)
        self.declare_parameter("touch_z_tolerance_m", 0.01)
        self.declare_parameter("retract_xy_tolerance_m", 0.02)
        self.declare_parameter("retract_z_tolerance_m", 0.05)
        self.declare_parameter("approach_settle_s", 1.5)
        self.declare_parameter("hold_s", 1.0)
        self.declare_parameter("record_retract", False)
        self.declare_parameter("timeout_s", 180.0)
        self.declare_parameter("approach_height_m", 0.05)
        self.declare_parameter("retract_height_m", 0.05)
        self.declare_parameter("cube_pose_z_offset_m", DEFAULT_CUBE_POSE_Z_OFFSET_M)
        self.declare_parameter("touch_clearance_m", 0.0)
        self.declare_parameter("cube_size_m", 0.05)
        self.declare_parameter("cube_pose_is_top_center", False)
        self.declare_parameter("rot_y_deg", 0.0)
        self.declare_parameter("instruction", "touch the top of the red cube")
        self.declare_parameter("wait_for_target_subscriber", True)
        self.declare_parameter("tool_pose_timeout_s", 0.5)
        self.declare_parameter("startup_tool_pose_samples", 0)
        self.declare_parameter("startup_hold_s", 1.0)
        self.declare_parameter("cube_restart_distance_m", 0.01)
        self.declare_parameter("post_episode_cube_ignore_s", 1.0)
        self.declare_parameter("home_before_episode", True)
        self.declare_parameter("default_tool_x", 0.45)
        self.declare_parameter("default_tool_y", 0.0)
        self.declare_parameter("default_tool_z", 0.20)
        self.declare_parameter("home_tolerance_m", 0.02)
        self.declare_parameter("home_settle_s", 0.5)
        self.declare_parameter("use_polar_workspace", True)
        self.declare_parameter("radius_min", 0.45)
        self.declare_parameter("radius_max", 0.68)
        self.declare_parameter("angle_min_deg", -90.0)
        self.declare_parameter("angle_max_deg", 90.0)
        self.declare_parameter("x_min", 0.45)
        self.declare_parameter("x_max", 0.68)
        self.declare_parameter("y_min", -0.15)
        self.declare_parameter("y_max", 0.15)
        self.declare_parameter("call_recorder_services", True)
        self.declare_parameter("recorder_start_service", "/start_episode")
        self.declare_parameter("recorder_stop_service", "/end_episode")
        self.declare_parameter("log_target_errors", False)

        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self._dt = 1.0 / rate_hz
        self._stage = Stage.IDLE
        self._episode_id = 0
        self._episode_started = self.get_clock().now()
        self._hold_started: Optional[object] = None
        self._tool_position: Optional[np.ndarray] = None
        self._tool_rot_y_deg: Optional[float] = None
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
        self._approach_reached_since: Optional[object] = None
        self._home_reached_since: Optional[object] = None
        self._last_episode_cube_center: Optional[np.ndarray] = None
        self._ignore_cube_pose_until: Optional[object] = None
        self._missing_recorder_service_warnings: set[str] = set()
        self._episode_recording_started = False
        self._episode_recording_stopped = False
        self._cube_ready = False
        self._cube_hidden = False

        cube_pose_topic = str(self.get_parameter("cube_pose_topic").value)
        tool_pose_topic = str(self.get_parameter("tool_pose_topic").value)
        expert_observation_state_topic = str(self.get_parameter("expert_observation_state_topic").value)
        target_pose_y_topic = str(self.get_parameter("target_pose_y_topic").value)
        episode_event_topic = str(self.get_parameter("episode_event_topic").value)
        task_instruction_topic = str(self.get_parameter("task_instruction_topic").value)

        self.create_subscription(PoseStamped, cube_pose_topic, self._on_cube_pose, 10)
        self.create_subscription(PoseStamped, tool_pose_topic, self._on_tool_pose, 10)
        self.create_subscription(String, episode_event_topic, self._on_episode_event, 10)
        self._expert_observation_state_pub = self.create_publisher(
            Float32MultiArray,
            expert_observation_state_topic,
            10,
        )
        self._target_pub = self.create_publisher(Float32MultiArray, target_pose_y_topic, 10)
        self._event_pub = self.create_publisher(String, episode_event_topic, 10)
        self._instruction_pub = self.create_publisher(String, task_instruction_topic, 10)
        self._recorder_start_client = self.create_client(
            StartEpisode,
            str(self.get_parameter("recorder_start_service").value),
        )
        self._recorder_stop_client = self.create_client(
            EndEpisode,
            str(self.get_parameter("recorder_stop_service").value),
        )
        self.create_timer(self._dt, self._tick)
        self.get_logger().info(
            "Cube touch expert publishes synced observation/action at "
            f"{rate_hz:.2f} Hz: state to {expert_observation_state_topic}; "
            f"absolute targets to {target_pose_y_topic}"
        )

        self._publish_instruction()

    def _on_episode_event(self, msg: String) -> None:
        try:
            _, event = msg.data.split(":", 1)
        except ValueError:
            return
        if event == "cube_ready":
            self._cube_ready = True
            self._cube_hidden = False
        elif event == "cube_hidden":
            self._cube_hidden = True
            self._cube_ready = False

    """Published by ik_control_node.py to command the end-effector pose. """
    def _on_tool_pose(self, msg: PoseStamped) -> None:
        self._tool_position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )
        q = msg.pose.orientation
        self._tool_rot_y_deg = self._axis_rotation_deg(
            np.array([q.w, q.x, q.y, q.z], dtype=np.float32),
            _Y_AXIS,
        )
        self._last_tool_pose_time = self.get_clock().now()
        self._tool_pose_sample_count += 1
        if self._current_target is None:
            self._current_target = self._tool_position.copy()

    """Published by Isaac Sim to provide the current pose of the cube. """
    def _on_cube_pose(self, msg: PoseStamped) -> None:
        if self._ignore_cube_pose_until is not None:
            if self.get_clock().now() < self._ignore_cube_pose_until:
                return
            self._ignore_cube_pose_until = None

        center = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )
        if self._is_hidden_cube_pose(center):
            return
        if not self._cube_ready:
            self.get_logger().warning(
                "Waiting for success_cube_relocator_node to confirm a surface cube pose before starting",
                throttle_duration_sec=2.0,
            )
            return
        if not self._is_reasonable_cube_pose(center):
            self.get_logger().warning(
                f"Ignoring cube pose that is not at a reasonable surface height: {np.round(center, 4)}",
                throttle_duration_sec=2.0,
            )
            return
        if not self._should_start_episode_for_cube(center):
            return
        if not self._inside_configured_workspace(center):
            self.get_logger().warning(
                f"Ignoring cube pose outside fixed-base workspace: {np.round(center, 4)}",
                throttle_duration_sec=2.0,
            )
            return
        if bool(self.get_parameter("cube_pose_is_top_center").value):
            top = center
        else:
            cube_size = float(self.get_parameter("cube_size_m").value)
            top = center + self._pose_up_axis(msg) * cube_size * 0.5

        up_axis = self._pose_up_axis(msg)
        top = top + up_axis * float(self.get_parameter("cube_pose_z_offset_m").value)
        touch = top + up_axis * float(self.get_parameter("touch_clearance_m").value)
        approach = touch + up_axis * float(self.get_parameter("approach_height_m").value)
        retract = touch + up_axis * float(self.get_parameter("retract_height_m").value)

        self._goals = {
            Stage.HOME: self._default_tool_pose(),
            Stage.APPROACH: approach,
            Stage.DESCEND: touch,
            Stage.HOLD: touch,
            Stage.RETRACT: retract,
            Stage.RETURN_HOME: self._default_tool_pose(),
        }
        self._last_episode_cube_center = center.copy()
        self._episode_id += 1
        self._episode_started = self.get_clock().now()
        self._episode_tool_pose_start_count = self._tool_pose_sample_count
        self._startup_ready_time = None
        self._episode_recording_started = False
        self._episode_recording_stopped = False
        self._cube_hidden = False

        self._current_target = None
        self._pending_start_from_tool_pose = True
        if bool(self.get_parameter("home_before_episode").value):
            self._set_stage(Stage.WAIT_CUBE_HIDDEN)
        else:
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
            self.get_logger().warning(
                "Waiting for fresh /kaivuri/tool_pose before advancing expert target",
                throttle_duration_sec=2.0,
            )
            return

        if not self._has_startup_tool_pose_samples():

            self._waiting_for_tool_pose = True
            self.get_logger().warning(
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
                self._publish_target(self._current_target)
                return
            self._startup_ready_time = None

        if self._stage in (Stage.WAIT_CUBE_HIDDEN, Stage.WAIT_CUBE_VISIBLE):
            self._publish_target()
            if self._stage == Stage.WAIT_CUBE_HIDDEN and self._cube_hidden:
                self._start_episode_recording()
                self._set_stage(Stage.HOME)
            elif self._stage == Stage.WAIT_CUBE_VISIBLE and self._cube_ready:
                self._set_stage(Stage.APPROACH)
            return

        if self._stage not in (Stage.IDLE, Stage.DONE) and self._timed_out():
            self._publish_event("failure_timeout")
            self._end_episode_recording()
            self._set_stage(Stage.DONE)
            return

        if self._stage == Stage.HOLD:
            hold_goal = self._goals.get(Stage.HOLD)
            if hold_goal is not None:
                self._current_target = hold_goal.copy()
                self._publish_target(hold_goal)
            else:
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

        current_position = self._tool_position.copy()
        if self._stage in (Stage.HOME, Stage.APPROACH):
            self._current_target = self._step_approach_toward(current_position, goal)
        else:
            self._current_target = self._step_toward(current_position, goal)
        
        self._publish_target()
        if self._target_reached(goal, self._stage):
            if self._stage == Stage.HOME:
                if not self._home_settled():
                    return
                self._episode_started = self.get_clock().now()
                self._set_stage(Stage.WAIT_CUBE_VISIBLE)
            elif self._stage == Stage.APPROACH:
                if not self._approach_settled():
                    return
                self._set_stage(Stage.DESCEND)
            elif self._stage == Stage.DESCEND:
                hold_goal = self._goals.get(Stage.HOLD)
                if hold_goal is not None:
                    self._current_target = hold_goal.copy()
                self._set_stage(Stage.HOLD)
            elif self._stage == Stage.RETRACT:
                self._set_stage(Stage.RETURN_HOME)
            elif self._stage == Stage.RETURN_HOME:
                if not self._home_settled():
                    return
                self._end_episode_recording()
                self._set_stage(Stage.DONE)
        elif self._stage in (Stage.HOME, Stage.APPROACH, Stage.RETURN_HOME):
            self._approach_reached_since = None
            self._home_reached_since = None

    def _step_toward(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        delta = goal - current
        distance = float(np.linalg.norm(delta))
        max_step = max(0.0, float(self.get_parameter("speed_mps").value)) * self._dt
        if distance <= max_step or distance < 1e-9:
            return goal.copy()
        return current + delta * (max_step / distance)

    def _step_approach_toward(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        # z_tolerance = max(0.0, float(self.get_parameter("approach_z_tolerance_m").value))
        safe_z = float(goal[2])
        # if float(current[2]) < safe_z - z_tolerance:
        #     lift_goal = current.copy()
        #     lift_goal[2] = safe_z
        #     return self._step_toward(current, lift_goal)

        horizontal_goal = goal.copy()
        horizontal_goal[2] = safe_z
        return self._step_toward(current, horizontal_goal)

    def _pose_up_axis(self, msg: PoseStamped) -> np.ndarray:
        q = msg.pose.orientation
        quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-6:
            return _LOCAL_Z.copy()
        w, x, y, z = quat / norm
        up = np.array(
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=np.float32,
        )
        up_norm = float(np.linalg.norm(up))
        if up_norm < 1e-6:
            return _LOCAL_Z.copy()
        return up / up_norm

    def _try_seed_current_target_from_tool_pose(self) -> None:
        if not self._pending_start_from_tool_pose:
            return
        if not self._tool_pose_is_fresh():
            self._waiting_for_tool_pose = True
            self.get_logger().warning(
                "Waiting for fresh /kaivuri/tool_pose to seed expert start target",
                throttle_duration_sec=2.0,
            )
            return
        self._current_target = self._tool_position.copy()
        self._pending_start_from_tool_pose = False

    def _default_tool_pose(self) -> np.ndarray:
        values = [
            self.get_parameter("default_tool_x").value,
            self.get_parameter("default_tool_y").value,
            self.get_parameter("default_tool_z").value,
        ]
        out = []
        for value in values:
            try:
                f = float(value)
            except Exception:
                f = 0.0
            out.append(f if np.isfinite(f) else 0.0)
        return np.asarray(out, dtype=np.float32)

    def _should_start_episode_for_cube(self, center: np.ndarray) -> bool:
        if self._stage not in (Stage.IDLE, Stage.DONE):
            return False
        if self._last_episode_cube_center is None:
            return True
        restart_distance = max(0.0, float(self.get_parameter("cube_restart_distance_m").value))

        moved = float(np.linalg.norm(center - self._last_episode_cube_center))
        return moved >= restart_distance

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

    @staticmethod
    def _is_hidden_cube_pose(position: np.ndarray) -> bool:
        return float(position[2]) < -1.0

    def _is_reasonable_cube_pose(self, position: np.ndarray) -> bool:
        return self._inside_configured_workspace(position) and -0.01 <= float(position[2]) <= 0.20

    def _front_angle_bounds(self) -> tuple[float, float]:
        angle_min = max(-90.0, float(self.get_parameter("angle_min_deg").value))
        angle_max = min(90.0, float(self.get_parameter("angle_max_deg").value))
        if angle_min > angle_max:
            return -90.0, 90.0
        return angle_min, angle_max

    def _should_wait_for_target_subscriber(self) -> bool:
        if not bool(self.get_parameter("wait_for_target_subscriber").value):
            return False
        if self._target_pub.get_subscription_count() > 0:
            return False
        self._waiting_for_target_subscriber = True
        self.get_logger().warning(
            f"Waiting for a subscriber on {self.get_parameter('target_pose_y_topic').value} "
            "before advancing expert target",
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

    def _target_reached(self, goal: np.ndarray, stage: Stage) -> bool:
        reference = self._tool_position if self._tool_position is not None else self._current_target
        if reference is None:
            return False

        if stage == Stage.HOME:
            tolerance = max(0.0, float(self.get_parameter("home_tolerance_m").value))
            return float(np.linalg.norm(reference - goal)) <= tolerance
        if stage == Stage.RETURN_HOME:
            tolerance = max(0.0, float(self.get_parameter("home_tolerance_m").value))
            return float(np.linalg.norm(reference - goal)) <= tolerance
        if self._current_target is None:
            
            return False
        xy_tolerance, z_tolerance = self._stage_tolerances(stage)

        measured_xy_error = float(np.linalg.norm(reference[:2] - goal[:2]))
        measured_z_error = abs(float(reference[2] - goal[2]))
        target_xy_error = float(np.linalg.norm(self._current_target[:2] - goal[:2]))
        target_z_error = abs(float(self._current_target[2] - goal[2]))
        if bool(self.get_parameter("log_target_errors").value):
            self.get_logger().info(
                f"{stage.value} tolerance xy={xy_tolerance}, z={z_tolerance}; "
                f"measured=({measured_xy_error:.4f}, {measured_z_error:.4f}); "
                f"target=({target_xy_error:.4f}, {target_z_error:.4f})",
                throttle_duration_sec=0.5,
            )
        return (
            measured_xy_error <= xy_tolerance
            and measured_z_error <= z_tolerance
            and target_xy_error <= xy_tolerance
            and target_z_error <= z_tolerance
        )

    def _stage_tolerances(self, stage: Stage) -> tuple[float, float]:
        if stage == Stage.APPROACH:
            return (
                max(0.0, float(self.get_parameter("approach_xy_tolerance_m").value)),
                max(0.0, float(self.get_parameter("approach_z_tolerance_m").value)),
            )
        if stage == Stage.DESCEND:
            return (
                max(0.0, float(self.get_parameter("touch_xy_tolerance_m").value)),
                max(0.0, float(self.get_parameter("touch_z_tolerance_m").value)),
            )
        if stage == Stage.RETRACT:
            return (
                max(0.0, float(self.get_parameter("retract_xy_tolerance_m").value)),
                max(0.0, float(self.get_parameter("retract_z_tolerance_m").value)),
            )
        tolerance = max(0.0, float(self.get_parameter("position_tolerance_m").value))
        return tolerance, tolerance


    def _approach_settled(self) -> bool:
        settle_s = max(0.0, float(self.get_parameter("approach_settle_s").value))
        if self._approach_reached_since is None:
            self._approach_reached_since = self.get_clock().now()
            return settle_s <= 0.0
        return self._elapsed(self._approach_reached_since) >= settle_s

    def _home_settled(self) -> bool:
        settle_s = max(0.0, float(self.get_parameter("home_settle_s").value))
        if self._home_reached_since is None:
            self._home_reached_since = self.get_clock().now()
            return settle_s <= 0.0
        return self._elapsed(self._home_reached_since) >= settle_s

    def _set_stage(self, stage: Stage) -> None:
        if stage != self._stage:
            self.get_logger().info(f"Episode {self._episode_id} stage: {self._stage.value} -> {stage.value}")
        self._stage = stage
        self._hold_started = None
        self._approach_reached_since = None
        self._home_reached_since = None
        if stage == Stage.WAIT_CUBE_HIDDEN:
            self._cube_hidden = False
            self._publish_event("hide_cube")
            self._publish_event(stage.value)
            return
        if stage == Stage.WAIT_CUBE_VISIBLE:
            self._cube_ready = False
            self._publish_event("show_cube")
            self._publish_event(stage.value)
            return
        if stage == Stage.HOME:
            self._start_episode_recording()
            self._publish_event(stage.value)
            return
        if stage == Stage.APPROACH:
            self._start_episode_recording()
        if stage in (Stage.RETRACT, Stage.RETURN_HOME):
            self._publish_event("hide_cube")
        if stage != Stage.DONE:
            self._publish_event(stage.value)

    def _start_episode_recording(self) -> None:
        if self._episode_recording_started:
            return
        self._episode_recording_started = True
        self._publish_event("episode_start")

    def _end_episode_recording(self) -> None:
        if self._episode_recording_stopped:
            return
        self._episode_recording_stopped = True
        self._publish_event("episode_end")
        self._ignore_cube_pose_until = self.get_clock().now() + Duration(
            seconds=max(0.0, float(self.get_parameter("post_episode_cube_ignore_s").value))
        )

    def _publish_target(self, target_position: Optional[np.ndarray] = None) -> None:
        if self._tool_position is None:
            return
        current_rot_y_deg = self._tool_rot_y_deg if self._tool_rot_y_deg is not None else self._target_rot_y_deg

        state_msg = Float32MultiArray()
        state_msg.data = [
            float(self._tool_position[0]),
            float(self._tool_position[1]),
            float(self._tool_position[2]),
            float(current_rot_y_deg),
        ]
        self._expert_observation_state_pub.publish(state_msg)

        if target_position is None:
            target_position = self._current_target
        if target_position is None:
            return

        target_rot_y_deg = self._step_rot_y_toward(current_rot_y_deg, self._target_rot_y_deg)
        target_msg = Float32MultiArray()
        target_msg.data = [
            float(target_position[0]),
            float(target_position[1]),
            float(target_position[2]),
            float(target_rot_y_deg),
        ]
        self._target_pub.publish(target_msg)

    def _step_rot_y_toward(self, current_rot_y_deg: float, target_rot_y_deg: float) -> float:
        delta_rot_y_deg = self._angle_delta_deg(target_rot_y_deg, current_rot_y_deg)
        max_rot = max(0.0, float(self.get_parameter("max_rot_y_deg_per_step").value))
        if max_rot > 0.0:
            delta_rot_y_deg = float(np.clip(delta_rot_y_deg, -max_rot, max_rot))
        return float(current_rot_y_deg + delta_rot_y_deg)

    def _publish_event(self, event: str) -> None:
        msg = String()
        msg.data = f"{self._episode_id}:{event}"
        self._event_pub.publish(msg)
        self.get_logger().info(f"Episode event: {msg.data}")
        if event == "episode_start":
            self._call_recorder_start()
        elif event == "episode_end":
            self._call_recorder_end()

    def _recorder_service_ready(self, client, label: str) -> bool:
        if not bool(self.get_parameter("call_recorder_services").value):
            return False
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=0.0):
            if label not in self._missing_recorder_service_warnings:
                self._missing_recorder_service_warnings.add(label)
                self.get_logger().warning(
                    f"Recorder service for {label} is not available; "
                    "launch lerobot_ros dataset_recorder or set call_recorder_services:=false"
                )
            return False
        self._missing_recorder_service_warnings.discard(label)
        return True

    def _call_recorder_start(self) -> None:
        label = "start_episode"
        if not self._recorder_service_ready(self._recorder_start_client, label):
            return
        request = StartEpisode.Request()
        request.task = str(self.get_parameter("instruction").value)
        future = self._recorder_start_client.call_async(request)
        future.add_done_callback(lambda done: self._log_recorder_start_result(done, label))

    def _call_recorder_end(self) -> None:
        label = "end_episode"
        if not self._recorder_service_ready(self._recorder_stop_client, label):
            return
        request = EndEpisode.Request()
        request.discard = False
        request.episde_id = int(self._episode_id)
        future = self._recorder_stop_client.call_async(request)
        future.add_done_callback(lambda done: self._log_recorder_end_result(done, label))

    def _log_recorder_start_result(self, future, label: str) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warning(f"Recorder service {label} failed: {exc}")
            return
        self.get_logger().info(f"Recorder service {label} started episode_id={response.episode_id}")

    def _log_recorder_end_result(self, future, label: str) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warning(f"Recorder service {label} failed: {exc}")
            return
        self.get_logger().info(f"Recorder service {label} ended episode with {response.frames} frames")

    def _publish_instruction(self) -> None:
        msg = String()
        msg.data = str(self.get_parameter("instruction").value)
        self._instruction_pub.publish(msg)

    def _timed_out(self) -> bool:
        timeout_s = max(0.0, float(self.get_parameter("timeout_s").value))
        return timeout_s > 0.0 and self._elapsed(self._episode_started) > timeout_s

    def _elapsed(self, start_time) -> float:
        return (self.get_clock().now() - start_time).nanoseconds * 1e-9

    @staticmethod
    def _axis_rotation_deg(quat_wxyz: np.ndarray, axis: np.ndarray) -> float:
        norm = float(np.linalg.norm(quat_wxyz))
        if norm < 1e-6:
            return 0.0

        quat = quat_wxyz / norm
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6:
            return 0.0
        unit_axis = axis / axis_norm

        twist_vec = unit_axis * float(np.dot(quat[1:], unit_axis))
        twist = np.array([quat[0], twist_vec[0], twist_vec[1], twist_vec[2]], dtype=np.float32)
        twist_norm = float(np.linalg.norm(twist))
        if twist_norm < 1e-6:
            return 0.0
        twist /= twist_norm

        signed = float(np.dot(twist[1:], unit_axis))
        return math.degrees(2.0 * math.atan2(signed, float(twist[0])))

    @staticmethod
    def _angle_delta_deg(target_deg: float, current_deg: float) -> float:
        return ((float(target_deg) - float(current_deg) + 180.0) % 360.0) - 180.0


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
