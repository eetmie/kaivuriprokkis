from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    command_timeout_s = LaunchConfiguration("command_timeout_s")

    return LaunchDescription([
        DeclareLaunchArgument(
            "project_root",
            default_value=EnvironmentVariable("KAIVURI_PROJECT_ROOT", default_value="/work"),
        ),
        DeclareLaunchArgument("command_timeout_s", default_value="0.5"),
        Node(
            package="kaivuri_bringup",
            executable="raw_direct_drive_node",
            name="kaivuri_raw_direct_drive_node",
            output="screen",
            parameters=[{
                "project_root": project_root,
                "command_timeout_s": command_timeout_s,
            }],
        ),
    ])
