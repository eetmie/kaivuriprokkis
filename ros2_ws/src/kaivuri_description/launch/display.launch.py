import os
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_model_dir():
    candidates = []

    project_root = os.environ.get("KAIVURI_PROJECT_ROOT")
    if project_root:
        candidates.append(Path(project_root) / "models" / "test4")

    for base in [Path.cwd(), Path(__file__).resolve()]:
        candidates.extend(parent / "models" / "test4" for parent in [base, *base.parents])

    for candidate in candidates:
        if (candidate / "test4v6.urdf").exists():
            return str(candidate)

    return str(candidates[0])


def _load_urdf(model_dir):
    model_dir = Path(model_dir)
    urdf_path = model_dir / "test4v6.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if not filename or "://" in filename or filename.startswith("package:"):
            continue

        mesh_path = model_dir / filename
        mesh.set("filename", mesh_path.resolve().as_uri())

    return ET.tostring(root, encoding="unicode")


def _launch_setup(context, *args, **kwargs):
    description_dir = Path(get_package_share_directory("kaivuri_description"))
    rviz_config_path = description_dir / "rviz" / "kaivuri.rviz"
    robot_description = _load_urdf(LaunchConfiguration("model_dir").perform(context))

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_gui = LaunchConfiguration("use_gui")
    rviz = LaunchConfiguration("rviz")

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
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
            condition=IfCondition(use_gui),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", str(rviz_config_path)],
            condition=IfCondition(rviz),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model_dir", default_value=_default_model_dir()),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        OpaqueFunction(function=_launch_setup),
    ])
