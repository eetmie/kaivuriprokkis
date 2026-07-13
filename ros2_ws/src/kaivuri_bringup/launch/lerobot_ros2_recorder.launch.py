from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("repo_id", default_value="kaivuri/ros2_recording"),
        DeclareLaunchArgument("root", default_value="/mnt/c/users/sh25016/kaivuri_lerobot_live"),
        DeclareLaunchArgument("fps", default_value="3.0"),
        DeclareLaunchArgument("episode_time_s", default_value="60.0"),
        DeclareLaunchArgument("num_episodes", default_value="1"),
        DeclareLaunchArgument("auto_start", default_value="false"),
        DeclareLaunchArgument("robot_type", default_value="kaivuri"),
        DeclareLaunchArgument("task", default_value="touch the top of the red cube"),
        DeclareLaunchArgument("hut_image_topic", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument("top_image_topic", default_value="/camera/camera1/color/image_raw"),
        DeclareLaunchArgument("tool_pose_topic", default_value="/kaivuri/tool_pose"),
        DeclareLaunchArgument("action_topic", default_value="/kaivuri/target_pose_y"),
        DeclareLaunchArgument("task_instruction_topic", default_value="/kaivuri/task_instruction"),
        DeclareLaunchArgument("max_sample_age_s", default_value="0.0"),
        DeclareLaunchArgument(
            "python_executable",
            default_value=EnvironmentVariable(
                "KAIVURI_PYTHON_EXECUTABLE",
                default_value="/mnt/c/Users/sh25016/Documents/kaivuriprokkis/.venv/bin/python",
            ),
        ),
        Node(
            package="kaivuri_bringup",
            executable="lerobot_ros2_recorder_node",
            name="lerobot_ros2_recorder_node",
            output="screen",
            prefix=[LaunchConfiguration("python_executable"), " "],
            parameters=[{
                "repo_id": LaunchConfiguration("repo_id"),
                "root": LaunchConfiguration("root"),
                "fps": LaunchConfiguration("fps"),
                "episode_time_s": LaunchConfiguration("episode_time_s"),
                "num_episodes": LaunchConfiguration("num_episodes"),
                "auto_start": LaunchConfiguration("auto_start"),
                "robot_type": LaunchConfiguration("robot_type"),
                "task": LaunchConfiguration("task"),
                "hut_image_topic": LaunchConfiguration("hut_image_topic"),
                "top_image_topic": LaunchConfiguration("top_image_topic"),
                "tool_pose_topic": LaunchConfiguration("tool_pose_topic"),
                "action_topic": LaunchConfiguration("action_topic"),
                "task_instruction_topic": LaunchConfiguration("task_instruction_topic"),
                "max_sample_age_s": LaunchConfiguration("max_sample_age_s"),
            }],
        ),
    ])
