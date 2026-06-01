from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    udp_host = LaunchConfiguration("udp_host")
    udp_port = LaunchConfiguration("udp_port")
    robot = LaunchConfiguration("robot")
    joystick_topic = LaunchConfiguration("joystick_topic")
    direct_pwm_topic = LaunchConfiguration("direct_pwm_topic")
    raw_max = LaunchConfiguration("raw_max")
    command_timeout_s = LaunchConfiguration("command_timeout_s")

    return LaunchDescription([
        DeclareLaunchArgument(
            "project_root",
            default_value=EnvironmentVariable("KAIVURI_PROJECT_ROOT", default_value="/work"),
        ),
        DeclareLaunchArgument("udp_host", default_value="10.214.33.132"),
        DeclareLaunchArgument("udp_port", default_value="8080"),
        DeclareLaunchArgument("robot", default_value="auto"),
        DeclareLaunchArgument("joystick_topic", default_value="joystick_values"),
        DeclareLaunchArgument("direct_pwm_topic", default_value="/kaivuri/direct_pwm"),
        DeclareLaunchArgument("raw_max", default_value="127.0"),
        DeclareLaunchArgument("command_timeout_s", default_value="0.5"),
        Node(
            package="kaivuri_bringup",
            executable="udp_joystick_values_node",
            name="udp_joystick_values_node",
            output="screen",
            parameters=[{
                "project_root": project_root,
                "host": udp_host,
                "port": udp_port,
                "topic": joystick_topic,
            }],
        ),
        Node(
            package="kaivuri_bringup",
            executable="joystick_to_direct_pwm_node",
            name="joystick_to_direct_pwm_node",
            output="screen",
            parameters=[{
                "input_topic": joystick_topic,
                "output_topic": direct_pwm_topic,
                "raw_max": raw_max,
            }],
        ),
        Node(
            package="kaivuri_bringup",
            executable="raw_direct_drive_node",
            name="kaivuri_raw_direct_drive_node",
            output="screen",
            parameters=[{
                "project_root": project_root,
                "command_topic": direct_pwm_topic,
                "robot": robot,
                "command_timeout_s": ParameterValue(command_timeout_s, value_type=float),
            }],
        ),
    ])
