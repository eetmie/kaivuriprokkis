from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def generate_launch_description():
    realsense_launch = Path(get_package_share_directory("realsense2_camera")) / "launch" / "rs_launch.py"

    camera_name = LaunchConfiguration("camera_name")
    camera_namespace = LaunchConfiguration("camera_namespace")
    enable_color = LaunchConfiguration("enable_color")
    enable_depth = LaunchConfiguration("enable_depth")
    align_depth = LaunchConfiguration("align_depth")
    enable_sync = LaunchConfiguration("enable_sync")
    enable_rgbd = LaunchConfiguration("enable_rgbd")
    enable_gyro = LaunchConfiguration("enable_gyro")
    enable_accel = LaunchConfiguration("enable_accel")
    rgb_camera_profile = LaunchConfiguration("rgb_camera_profile")
    depth_module_profile = LaunchConfiguration("depth_module_profile")
    unite_imu_method = LaunchConfiguration("unite_imu_method")
    pointcloud_enable = LaunchConfiguration("pointcloud_enable")
    rsusb_lib = LaunchConfiguration("rsusb_lib")

    return LaunchDescription([
        DeclareLaunchArgument("camera_name", default_value="camera"),
        DeclareLaunchArgument("camera_namespace", default_value="camera"),
        DeclareLaunchArgument("enable_color", default_value="true"),
        DeclareLaunchArgument("enable_depth", default_value="true"),
        DeclareLaunchArgument("align_depth", default_value="true"),
        DeclareLaunchArgument("enable_sync", default_value="true"),
        DeclareLaunchArgument("enable_rgbd", default_value="false"),
        DeclareLaunchArgument("enable_gyro", default_value="true"),
        DeclareLaunchArgument("enable_accel", default_value="true"),
        DeclareLaunchArgument("rgb_camera_profile", default_value="640x480x30"),
        DeclareLaunchArgument("depth_module_profile", default_value="640x480x30"),
        DeclareLaunchArgument("unite_imu_method", default_value="2"),
        DeclareLaunchArgument("pointcloud_enable", default_value="false"),
        DeclareLaunchArgument(
            "rsusb_lib",
            default_value=EnvironmentVariable(
                "REALSENSE_RSUSB_LIB",
                default_value="/opt/librealsense-rsusb/lib",
            ),
        ),
        SetEnvironmentVariable(
            "LD_LIBRARY_PATH",
            [rsusb_lib, ":", EnvironmentVariable("LD_LIBRARY_PATH", default_value="")],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(realsense_launch)),
            launch_arguments={
                "camera_name": camera_name,
                "camera_namespace": camera_namespace,
                "enable_color": enable_color,
                "enable_depth": enable_depth,
                "align_depth.enable": align_depth,
                "enable_sync": enable_sync,
                "enable_rgbd": enable_rgbd,
                "enable_gyro": enable_gyro,
                "enable_accel": enable_accel,
                "rgb_camera.profile": rgb_camera_profile,
                "depth_module.profile": depth_module_profile,
                "unite_imu_method": unite_imu_method,
                "pointcloud.enable": pointcloud_enable,
            }.items(),
        ),
    ])
