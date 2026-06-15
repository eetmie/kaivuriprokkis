from pathlib import Path
import xml.etree.ElementTree as ET

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def _load_urdf(model_dir: str) -> str:
    model_dir_path = Path(model_dir)
    urdf_path = model_dir_path / "test4v6.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if not filename or "://" in filename or filename.startswith("package:"):
            continue
        mesh.set("filename", (model_dir_path / filename).resolve().as_uri())

    return ET.tostring(root, encoding="unicode")


def _launch_setup(context, *args, **kwargs):
    robot_description = _load_urdf(LaunchConfiguration("model_dir").perform(context))
    use_sim_time = LaunchConfiguration("use_sim_time")
    project_root = LaunchConfiguration("project_root")
    robot = LaunchConfiguration("robot")
    state_rate_hz = LaunchConfiguration("state_rate_hz")
    command_timeout_s = LaunchConfiguration("command_timeout_s")
    visualization_only = LaunchConfiguration("visualization_only")
    python_executable = LaunchConfiguration("python_executable")

    return [
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
            executable="ik_pose_control_node",
            name="kaivuri_ik_pose_control_node",
            output="screen",
            prefix=[python_executable, " "],
            parameters=[{
                "project_root": project_root,
                "robot": robot,
                "state_rate_hz": state_rate_hz,
                "command_timeout_s": command_timeout_s,
                "visualization_only": visualization_only,
            }],
        ),
    ]


def generate_launch_description():
    project_root_default = EnvironmentVariable(
        "KAIVURI_PROJECT_ROOT",
        default_value="/mnt/c/Users/sh25016/Documents/kaivuriprokkis",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_dir",
            default_value=EnvironmentVariable(
                "KAIVURI_MODEL_DIR",
                default_value=PathJoinSubstitution([project_root_default, "models", "test4"]),
            ),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "project_root",
            default_value=project_root_default,
        ),
        DeclareLaunchArgument("robot", default_value="auto"),
        DeclareLaunchArgument("state_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("command_timeout_s", default_value="1.0"),
        DeclareLaunchArgument("visualization_only", default_value="false"),
        DeclareLaunchArgument(
            "python_executable",
            default_value=EnvironmentVariable(
                "KAIVURI_PYTHON_EXECUTABLE",
                default_value="/mnt/c/Users/sh25016/Documents/kaivuriprokkis/.venv/bin/python",
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
