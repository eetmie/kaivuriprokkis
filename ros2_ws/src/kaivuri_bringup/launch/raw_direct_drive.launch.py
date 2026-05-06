from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    description_dir = Path(get_package_share_directory("kaivuri_description"))
    urdf_path = description_dir / "urdf" / "kaivuri.urdf"
    robot_description = urdf_path.read_text(encoding="utf-8")

    use_sim_time = LaunchConfiguration("use_sim_time")
    project_root = LaunchConfiguration("project_root")
    command_timeout_s = LaunchConfiguration("command_timeout_s")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "project_root",
            default_value=EnvironmentVariable("KAIVURI_PROJECT_ROOT", default_value="/work"),
        ),
        DeclareLaunchArgument("command_timeout_s", default_value="0.5"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }],
        ),
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
