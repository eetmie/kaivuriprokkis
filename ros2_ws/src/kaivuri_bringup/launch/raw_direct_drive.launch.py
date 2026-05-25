from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    robot = LaunchConfiguration("robot")
    config_file = LaunchConfiguration("config_file")
    control_config_file = LaunchConfiguration("control_config_file")
    pwm_i2c_bus = LaunchConfiguration("pwm_i2c_bus")
    pwm_i2c_addr = LaunchConfiguration("pwm_i2c_addr")
    command_timeout_s = LaunchConfiguration("command_timeout_s")

    return LaunchDescription([
        DeclareLaunchArgument(
            "project_root",
            default_value=EnvironmentVariable("KAIVURI_PROJECT_ROOT", default_value="/work"),
        ),
        DeclareLaunchArgument("robot", default_value="auto"),
        DeclareLaunchArgument("config_file", default_value=""),
        DeclareLaunchArgument("control_config_file", default_value=""),
        DeclareLaunchArgument("pwm_i2c_bus", default_value="-1"),
        DeclareLaunchArgument("pwm_i2c_addr", default_value="-1"),
        DeclareLaunchArgument("command_timeout_s", default_value="0.5"),
        Node(
            package="kaivuri_bringup",
            executable="raw_direct_drive_node",
            name="kaivuri_raw_direct_drive_node",
            output="screen",
            parameters=[{
                "project_root": project_root,
                "robot": robot,
                "config_file": config_file,
                "control_config_file": control_config_file,
                "pwm_i2c_bus": ParameterValue(pwm_i2c_bus, value_type=int),
                "pwm_i2c_addr": ParameterValue(pwm_i2c_addr, value_type=int),
                "command_timeout_s": ParameterValue(command_timeout_s, value_type=float),
            }],
        ),
    ])
