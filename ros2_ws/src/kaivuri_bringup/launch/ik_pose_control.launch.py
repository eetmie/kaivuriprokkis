import os
from pathlib import Path
import xml.etree.ElementTree as ET

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def _default_project_root() -> str:
    candidates = []

    project_root = os.environ.get("KAIVURI_PROJECT_ROOT")
    if project_root:
        candidates.append(Path(project_root))

    for base in [Path.cwd(), Path(__file__).resolve()]:
        candidates.extend([base, *base.parents])

    for candidate in candidates:
        if (candidate / "modules").is_dir() and (candidate / "configuration_files").is_dir():
            return str(candidate)

    return str(Path.cwd())


def _default_model_dir() -> str:
    project_root = Path(_default_project_root())
    candidates = [project_root / "models" / "test4"]

    for base in [Path.cwd(), Path(__file__).resolve()]:
        candidates.extend(parent / "models" / "test4" for parent in [base, *base.parents])

    for candidate in candidates:
        if (candidate / "test4v6.urdf").exists():
            return str(candidate)

    return str(candidates[0])


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
    target_pose_y_topic = LaunchConfiguration("target_pose_y_topic")
    ik_command_type = LaunchConfiguration("ik_command_type")
    ik_condition_number_threshold = LaunchConfiguration("ik_condition_number_threshold")
    python_executable_value = LaunchConfiguration("python_executable").perform(context).strip()
    executable = "ik_pose_control_node"
    prefix = None
    if python_executable_value:
        if os.name == "nt":
            executable = "ik_pose_control_node-script.py"
        prefix = [python_executable_value, " "]

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
            executable=executable,
            name="kaivuri_ik_pose_control_node",
            output="screen",
            prefix=prefix,
            parameters=[{
                "project_root": project_root,
                "robot": robot,
                "use_sim_time": use_sim_time,
                "state_rate_hz": state_rate_hz,
                "command_timeout_s": command_timeout_s,
                "visualization_only": visualization_only,
                "target_pose_y_topic": target_pose_y_topic,
                "ik_command_type": ik_command_type,
                "ik_condition_number_threshold": ik_condition_number_threshold,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "model_dir",
            default_value=EnvironmentVariable("KAIVURI_MODEL_DIR", default_value=_default_model_dir()),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "project_root",
            default_value=EnvironmentVariable("KAIVURI_PROJECT_ROOT", default_value=_default_project_root()),
        ),
        DeclareLaunchArgument("robot", default_value="auto"),
        DeclareLaunchArgument("state_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("command_timeout_s", default_value="0.0"),
        DeclareLaunchArgument("target_pose_y_topic", default_value="/kaivuri/target_pose_y"),
        DeclareLaunchArgument("ik_command_type", default_value="pose"),
        DeclareLaunchArgument("ik_condition_number_threshold", default_value="0.0"),
        DeclareLaunchArgument("visualization_only", default_value="false"),
        DeclareLaunchArgument(
            "python_executable",
            default_value=EnvironmentVariable(
                "KAIVURI_PYTHON_EXECUTABLE",
                default_value="",
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
