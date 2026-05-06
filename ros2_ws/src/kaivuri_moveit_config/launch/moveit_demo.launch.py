from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def generate_launch_description():
    description_dir = Path(get_package_share_directory("kaivuri_description"))
    moveit_dir = Path(get_package_share_directory("kaivuri_moveit_config"))

    robot_description = {
        "robot_description": (description_dir / "urdf" / "kaivuri.urdf").read_text(encoding="utf-8")
    }
    robot_description_semantic = {
        "robot_description_semantic": (moveit_dir / "config" / "kaivuri.srdf").read_text(encoding="utf-8")
    }
    robot_description_kinematics = {
        "robot_description_kinematics": _load_yaml(moveit_dir / "config" / "kinematics.yaml")
    }
    joint_limits = {
        "robot_description_planning": _load_yaml(moveit_dir / "config" / "joint_limits.yaml")
    }

    ompl = _load_yaml(moveit_dir / "config" / "ompl_planning.yaml")
    planning_pipelines = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl,
    }
    trajectory_execution = {
        "allow_trajectory_execution": False,
        "moveit_manage_controllers": False,
    }
    planning_scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    launch_rviz = LaunchConfiguration("launch_rviz")

    common_parameters = [
        robot_description,
        robot_description_semantic,
        robot_description_kinematics,
        joint_limits,
        planning_pipelines,
        trajectory_execution,
        planning_scene_monitor,
        _load_yaml(moveit_dir / "config" / "moveit_controllers.yaml"),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=common_parameters,
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(launch_rviz),
            parameters=common_parameters,
        ),
    ])
