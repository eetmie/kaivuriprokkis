from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    description_dir = Path(get_package_share_directory("kaivuri_description"))
    urdf_path = description_dir / "urdf" / "kaivuri.urdf"
    robot_description = urdf_path.read_text(encoding="utf-8")

    use_sim_time = LaunchConfiguration("use_sim_time")
    animate = LaunchConfiguration("animate")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("animate", default_value="true"),
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
            executable="demo_state_node",
            name="kaivuri_demo_state_node",
            output="screen",
            parameters=[{"animate": animate}],
        ),
    ])
