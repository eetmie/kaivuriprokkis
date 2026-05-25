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
    project_root = os.environ.get("KAIVURI_PROJECT_ROOT")
    if project_root:
        return str(Path(project_root) / "models" / "test4")

    cwd = Path.cwd()
    candidates = [
        cwd / "models" / "test4",
        cwd.parent / "models" / "test4",
        Path("/home/ai-masi/kaivuriprokkis/models/test4"),
    ]
    for candidate in candidates:
        if (candidate / "test.urdf").exists():
            return str(candidate)
    return str(candidates[-1])


def _load_visual_urdf(model_dir):
    model_dir = Path(model_dir)
    urdf_path = model_dir / "test.urdf"
    mesh_dir = model_dir / "meshes"

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)

        for visual in list(link.findall("visual")):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue

            filename = mesh.attrib.get("filename", "")
            mesh_path = mesh_dir / Path(filename).name
            if not mesh_path.exists():
                link.remove(visual)
                continue

            mesh.set("filename", mesh_path.resolve().as_uri())

    return ET.tostring(root, encoding="unicode")


def _launch_setup(context, *args, **kwargs):
    description_dir = Path(get_package_share_directory("kaivuri_description"))
    rviz_config = description_dir / "rviz" / "test4.rviz"

    robot_description = _load_visual_urdf(
        LaunchConfiguration("model_dir").perform(context)
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="test4_robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="test4_joint_state_publisher_gui",
            condition=IfCondition(LaunchConfiguration("use_gui")),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="test4_rviz2",
            arguments=["-d", str(rviz_config)],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model_dir", default_value=_default_model_dir()),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        OpaqueFunction(function=_launch_setup),
    ])
