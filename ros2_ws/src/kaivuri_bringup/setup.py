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
            "launch/joystick_direct_drive.launch.py",
            "launch/realsense_d435i.launch.py",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="joel",
    maintainer_email="joel@example.com",
    description="Bringup and state publisher nodes for the Kaivuri excavator.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "demo_state_node = kaivuri_bringup.demo_state_node:main",
            "imu_state_node = kaivuri_bringup.imu_state_node:main",
            "joystick_to_direct_pwm_node = kaivuri_bringup.joystick_to_direct_pwm_node:main",
            "raw_direct_drive_node = kaivuri_bringup.raw_direct_drive_node:main",
            "udp_joystick_values_node = kaivuri_bringup.udp_joystick_values_node:main",
        ],
    },
)
