from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_random_cube", default_value="false"),
        DeclareLaunchArgument("cube_size_m", default_value="0.05"),
        DeclareLaunchArgument("rate_hz", default_value="50.0"),
        DeclareLaunchArgument("speed_mps", default_value="0.08"),
        DeclareLaunchArgument("position_tolerance_m", default_value="0.02"),
        DeclareLaunchArgument("target_tracking_tolerance_m", default_value="0.003"),
        Node(
            package="kaivuri_bringup",
            executable="cube_touch_expert_node",
            name="cube_touch_expert_node",
            output="screen",
            parameters=[{
                "cube_size_m": LaunchConfiguration("cube_size_m"),
                "rate_hz": LaunchConfiguration("rate_hz"),
                "speed_mps": LaunchConfiguration("speed_mps"),
                "position_tolerance_m": LaunchConfiguration("position_tolerance_m"),
                "target_tracking_tolerance_m": LaunchConfiguration("target_tracking_tolerance_m"),
            }],
        ),
        Node(
            package="kaivuri_bringup",
            executable="random_cube_pose_node",
            name="random_cube_pose_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_random_cube")),
            parameters=[{
                "period_s": 8.0,
            }],
        ),
    ])
