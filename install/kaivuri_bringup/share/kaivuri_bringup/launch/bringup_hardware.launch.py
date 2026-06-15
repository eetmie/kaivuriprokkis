from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    robot = LaunchConfiguration("robot")
    rate_hz = LaunchConfiguration("rate_hz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "project_root",
            default_value=EnvironmentVariable("KAIVURI_PROJECT_ROOT", default_value="/work"),
        ),
        DeclareLaunchArgument("robot", default_value="auto"),
        DeclareLaunchArgument("rate_hz", default_value="50.0"),
        Node(
            package="kaivuri_bringup",
            executable="imu_state_node",
            name="kaivuri_imu_state_node", # Reads IMU's and publishes
            output="screen",
            parameters=[{
                "project_root": project_root,
                "robot": robot,
                "rate_hz": rate_hz,
            }],
        ),
    ])
