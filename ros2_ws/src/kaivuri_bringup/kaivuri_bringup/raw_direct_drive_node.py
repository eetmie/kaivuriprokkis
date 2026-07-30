import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from kaivuri_bringup.project_paths import add_project_import_path, resolve_project_root


COMMAND_NAMES = [
    "slew",
    "boom",
    "arm",
    "bucket",
    "trackL",
    "trackR",
]


class RawDirectDriveNode(Node):
    """Raw normalized valve/track command bridge for ROS testing.

    Subscribes to std_msgs/Float32MultiArray on /kaivuri/direct_pwm.
    Data order is [slew, boom, arm, bucket, trackL, trackR].
    """

    def __init__(self) -> None:
        super().__init__("kaivuri_raw_direct_drive_node")
        self.declare_parameter("project_root", os.environ.get("KAIVURI_PROJECT_ROOT", "/work"))
        self.declare_parameter("robot", "auto")
        self.declare_parameter("config_file", "")
        self.declare_parameter("control_config_file", "")
        self.declare_parameter("pwm_i2c_bus", -1)
        self.declare_parameter("pwm_i2c_addr", -1)
        self.declare_parameter("command_topic", "/kaivuri/direct_pwm")
        self.declare_parameter("command_rate_hz", 50.0)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("ready_timeout_s", 30.0)
        self.declare_parameter("pump_auto_mode", False)
        self.declare_parameter("toggle_channels", True)
        self.declare_parameter("stale_timeout_s", 0.5)

        self._project_root = resolve_project_root(str(self.get_parameter("project_root").value))
        add_project_import_path(self._project_root)

        from modules.hardware_interface import HardwareInterface
        from modules.board import resolve_profile

        robot_profile = resolve_profile(str(self.get_parameter("robot").value))
        config_file = self._param_or_profile("config_file", robot_profile["servo_config_file"])
        control_config_file = self._param_or_profile("control_config_file", robot_profile["control_config_file"])
        pwm_i2c_bus = self._int_param_or_profile("pwm_i2c_bus", int(robot_profile["pwm_i2c_bus"]), unset=-1)
        pwm_i2c_addr = self._int_param_or_profile("pwm_i2c_addr", int(robot_profile["pwm_i2c_addr"]), unset=-1)

        control_config_path = self._resolve_project_path(control_config_file)

        config_path = self._resolve_project_path(config_file)
        self.get_logger().info(
            "Raw direct drive profile="
            f"{robot_profile['profile_name']} board={robot_profile['board']} "
            f"servo_config={config_path} control_config={control_config_path} "
            f"I2C bus={pwm_i2c_bus} addr=0x{pwm_i2c_addr:02X}"
        )
        self._hardware = HardwareInterface(
            config_file=str(config_path),
            control_config_file=str(control_config_path),
            pump_auto_mode=bool(self.get_parameter("pump_auto_mode").value),
            toggle_channels=bool(self.get_parameter("toggle_channels").value),
            stale_timeout_s=float(self.get_parameter("stale_timeout_s").value),
            enable_pwm=True,
            enable_imu=False,
            enable_adc=False,
            start_imu_reader=False,
            start_adc_reader=False,
            cleanup_disable_osc=False,
            pwm_i2c_bus=pwm_i2c_bus,
            pwm_i2c_addr=pwm_i2c_addr,
        )
        self._wait_for_hardware(float(self.get_parameter("ready_timeout_s").value))

        self._latest_commands: Dict[str, float] = {}
        self._last_command_time = 0.0
        self._stale_zeroed = True

        command_topic = str(self.get_parameter("command_topic").value)
        self.create_subscription(Float32MultiArray, command_topic, self._on_direct_pwm, 10)

        command_rate_hz = max(1.0, float(self.get_parameter("command_rate_hz").value))
        self.create_timer(1.0 / command_rate_hz, self._command_tick)
        self.get_logger().info(
            f"Raw direct drive ready on {command_topic}; order={COMMAND_NAMES}"
        )

    def _resolve_project_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self._project_root / path

    def _param_or_profile(self, name: str, profile_value: str) -> str:
        value = str(self.get_parameter(name).value).strip()
        return value or profile_value

    def _int_param_or_profile(self, name: str, profile_value: int, *, unset: int = None) -> int:
        value = int(self.get_parameter(name).value)
        if unset is not None and value == unset:
            return profile_value
        return value

    def _wait_for_hardware(self, timeout_s: float) -> None:
        deadline = time.time() + max(0.0, timeout_s)
        while not self._hardware.is_hardware_ready():
            if timeout_s > 0.0 and time.time() >= deadline:
                raise TimeoutError("Hardware did not become ready before timeout")
            time.sleep(0.1)

    def _on_direct_pwm(self, msg: Float32MultiArray) -> None:
        values = self._clamped_values(list(msg.data))
        if len(values) < 4:
            self.get_logger().warning(
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
                self._hardware.reset(reset_pump=False)
                self._stale_zeroed = True
            return
        self._hardware.send_named_pwm_commands(self._latest_commands)
        self._stale_zeroed = False

    def destroy_node(self) -> bool:
        try:
            self._hardware.reset(reset_pump=False)
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
