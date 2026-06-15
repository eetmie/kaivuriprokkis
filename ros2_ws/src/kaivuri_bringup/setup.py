from setuptools import find_packages, setup

package_name = "kaivuri_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [
            "launch/bringup_hardware.launch.py",
            "launch/cube_touch_data_collection.launch.py",
            "launch/ik_pose_control.launch.py",
            "launch/joystick_direct_drive.launch.py",
            "launch/realsense_d435i.launch.py",
        ]),
    ],
    install_requires=["setuptools", "inputs"],
    zip_safe=True,
    maintainer="joel",
    maintainer_email="joel@example.com",
    description="Bringup and state publisher nodes for the Kaivuri excavator.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "cube_touch_expert_node = kaivuri_bringup.cube_touch_expert_node:main",
            "demo_state_node = kaivuri_bringup.demo_state_node:main",
            "imu_state_node = kaivuri_bringup.imu_state_node:main",
            "ik_pose_control_node = kaivuri_bringup.ik_pose_control_node:main",
            "joystick_to_direct_pwm_node = kaivuri_bringup.joystick_to_direct_pwm_node:main",
            "random_cube_pose_node = kaivuri_bringup.random_cube_pose_node:main",
            "raw_direct_drive_node = kaivuri_bringup.raw_direct_drive_node:main",
            "target_pose_y_test_publisher = kaivuri_bringup.target_pose_y_test_publisher:main",
            "udp_joystick_values_node = kaivuri_bringup.udp_joystick_values_node:main",
            "xbox_target_pose_y_node = kaivuri_bringup.xbox_target_pose_y_node:main",
        ],
    },
)
