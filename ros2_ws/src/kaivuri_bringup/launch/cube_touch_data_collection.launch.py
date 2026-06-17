from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_success_cube_relocator", default_value="true"),
        DeclareLaunchArgument("use_random_cube", default_value="false"),
        DeclareLaunchArgument("cube_size_m", default_value="0.05"),
        DeclareLaunchArgument("cube_command_topic", default_value="/kaivuri/cube_pose_cmd"),
        DeclareLaunchArgument("rate_hz", default_value="50.0"),
        DeclareLaunchArgument("speed_mps", default_value="0.08"),
        DeclareLaunchArgument("position_tolerance_m", default_value="0.02"),
        DeclareLaunchArgument("cube_x_min", default_value="0.35"),
        DeclareLaunchArgument("cube_x_max", default_value="0.68"),
        DeclareLaunchArgument("cube_y_min", default_value="-0.15"),
        DeclareLaunchArgument("cube_y_max", default_value="0.15"),
        DeclareLaunchArgument("cube_relocate_delay_s", default_value="1.2"),
        DeclareLaunchArgument("post_episode_cube_ignore_s", default_value="1.0"),
        DeclareLaunchArgument("use_ik_reachability", default_value="false"),
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
                "post_episode_cube_ignore_s": LaunchConfiguration("post_episode_cube_ignore_s"),
            }],
        ),
        Node(
            package="kaivuri_bringup",
            executable="success_cube_relocator_node",
            name="success_cube_relocator_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_success_cube_relocator")),
            parameters=[{
                "cube_command_topic": LaunchConfiguration("cube_command_topic"),
                "cube_size_m": LaunchConfiguration("cube_size_m"),
                "x_min": LaunchConfiguration("cube_x_min"),
                "x_max": LaunchConfiguration("cube_x_max"),
                "y_min": LaunchConfiguration("cube_y_min"),
                "y_max": LaunchConfiguration("cube_y_max"),
                "relocate_delay_s": LaunchConfiguration("cube_relocate_delay_s"),
                "use_ik_reachability": LaunchConfiguration("use_ik_reachability"),
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
