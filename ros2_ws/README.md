# Kaivuri ROS 2 Workspace

Starter ROS 2 Jazzy workspace for the Kaivuri excavator.

The first goal is visualization, MoveIt planning experiments, and rosbag-friendly
state topics. This does not drive hydraulics.

Important: the current URDF is a placeholder model. Its geometry and joint angle
conventions do not yet match the real machine, so RViz/MoveIt output is useful
for ROS plumbing tests, not for validating real excavator kinematics or limits.

## URDF Calibration Direction

Do not treat the current CAD-exported URDF as the long-term kinematic source of
truth. The better path is to create a clean `kaivuri.urdf.xacro` model where
joint origins, axes, signs, zero offsets, limits, and link dimensions are explicit
parameters. Use CAD/Blender/FreeCAD mainly for visual mesh cleanup; use xacro and
measured geometry for the kinematic chain.

Suggested calibration workflow:

1. Keep the existing mesh model as a visual placeholder.
2. Build a simple xacro chain for `revolute_cabin`, `revolute_lift`,
   `revolute_tilt`, and `revolute_scoop`.
3. Add per-joint sign and zero-offset parameters so IMU-derived canonical angles
   can be mapped into ROS joint conventions without changing the IMU math.
4. Publish known `/joint_states`, inspect TF/RViz, and compare against the real
   machine.
5. Compare ROS TF tool pose against the custom FK `/kaivuri/tool_pose`; tune the
   xacro parameters until the two agree.
6. Regenerate MoveIt/SRDF configuration only after the kinematic URDF is sane.

Useful references:

- ROS 2 URDF tutorials: https://docs.ros.org/en/jazzy/Tutorials/Intermediate/URDF/URDF-Main.html
- `robot_state_publisher`: https://docs.ros.org/en/ros2_packages/jazzy/api/robot_state_publisher/
- xacro: https://docs.ros.org/en/ros2_packages/rolling/api/xacro/
- MoveIt Setup Assistant: https://docs.ros.org/en/ros2_packages/jazzy/api/moveit_setup_assistant/

## Packages

- `kaivuri_description`: URDF and mesh assets packaged for ROS.
- `kaivuri_bringup`: demo, IMU-derived state, and raw direct-drive nodes.
- `kaivuri_moveit_config`: minimal MoveIt config for the four active joints.

Active joint order:

```text
revolute_cabin
revolute_lift
revolute_tilt
revolute_scoop
```

## Build

From the repo root, enter the Jazzy container and build:

```bash
cd /home/joel/kaivuriprokkis
ros2-jazzy
cd /work/ros2_ws
colcon build
source install/setup.bash
```

One-shot build from the host:

```bash
cd /home/joel/kaivuriprokkis/ros2_ws
ros2-jazzy colcon build
```

Avoid `--symlink-install` when building in Docker and running from a different
host path. It can leave installed URDF/mesh assets pointing at the container
path instead of real files. Quick check:

```bash
test -e install/kaivuri_description/share/kaivuri_description/urdf/kaivuri.urdf
test -e install/kaivuri_description/share/kaivuri_description/meshes/upper_carriage_Mesh.obj
```

## Demo State Bringup

Use this without robot hardware:

```bash
source /work/ros2_ws/install/setup.bash
ros2 launch kaivuri_bringup bringup_demo.launch.py
```

This starts:

- `robot_state_publisher`
- `kaivuri_demo_state_node`
- `/joint_states`
- TF from the URDF

Disable animation:

```bash
ros2 launch kaivuri_bringup bringup_demo.launch.py animate:=false
```

To view the model in RViz, source the same install in another terminal and run:

```bash
ros2 launch kaivuri_description display.launch.py
```

Use fixed frame `excavator`. Do not run `joint_state_publisher_gui` at the same
time as the demo node, because both publish `/joint_states`.

## Hardware State Bringup

Use this on the Pi when IMUs are connected:

```bash
cd /home/joel/kaivuriprokkis
ros2-jazzy bash -lc 'cd /work/ros2_ws && source install/setup.bash && ros2 launch kaivuri_bringup bringup_hardware.launch.py'
```

The hardware node is state-only. It starts the IMU reader with PWM and ADC
disabled, converts corrected IMU quaternions to the four URDF joint angles, and
publishes:

- `/joint_states`
- `/kaivuri/tool_pose`

## Raw Direct Drive

Use this only when the machine is safe to move. This is raw normalized driving:
no IK, no path planning, no linkage compensation, and no MoveIt execution.

```bash
cd /home/joel/kaivuriprokkis
ros2-jazzy bash -lc 'cd /work/ros2_ws && source install/setup.bash && ros2 launch kaivuri_bringup raw_direct_drive.launch.py'
```

Command topic:

```text
/kaivuri/direct_pwm
std_msgs/msg/Float32MultiArray
```

Data order:

```text
[rotate, lift_boom, tilt_boom, scoop, trackL, trackR]
```

Values are clamped to `[-1.0, 1.0]`. If only the first four values are sent,
tracks default to zero. The node stops sending direct commands when messages are
older than `command_timeout_s` seconds, default `0.5`.

Example small boom command:

```bash
ros2 topic pub --once /kaivuri/direct_pwm std_msgs/msg/Float32MultiArray \
  "{data: [0.0, 0.15, 0.0, 0.0, 0.0, 0.0]}"
```

While driving, the same node publishes:

- `/joint_states`
- `/kaivuri/tool_pose`

If running outside the repo-mounted Docker layout, set:

```bash
export KAIVURI_PROJECT_ROOT=/home/joel/kaivuriprokkis
```

## Intel RealSense D435i

Use this when the D435i is connected to the Jetson through USB 3:

```bash
cd /home/joel/kaivuriprokkis
ros2-jazzy bash -lc 'cd /work/ros2_ws && source install/setup.bash && ros2 launch kaivuri_bringup realsense_d435i.launch.py'
```

The launch file starts Intel's `realsense2_camera` driver with color, depth,
aligned depth, gyro, accel, sync, and fused IMU enabled. Default profiles are
`640x480x30` for RGB and depth. It publishes the standard RealSense ROS topics,
including:

- `/camera/camera/color/image_raw`
- `/camera/camera/depth/image_rect_raw`
- `/camera/camera/aligned_depth_to_color/image_raw`
- `/camera/camera/imu`
- `/camera/camera/color/camera_info`
- `/camera/camera/depth/camera_info`

Optional point cloud:

```bash
ros2 launch kaivuri_bringup realsense_d435i.launch.py pointcloud_enable:=true
```

Useful checks:

```bash
ros2 topic list | grep camera
ros2 topic hz /camera/camera/imu
```

## RViz and MoveIt on a Laptop

The current Pi image is headless and does not include RViz/MoveIt. Run RViz and
MoveIt on a Jazzy laptop or a richer Jazzy container with:

```bash
sudo apt install ros-jazzy-rviz2 ros-jazzy-moveit
```

Build this workspace on the laptop too, or copy it over, then:

```bash
source ~/kaivuriprokkis/ros2_ws/install/setup.bash
ros2 launch kaivuri_moveit_config moveit_demo.launch.py
```

To connect laptop RViz/MoveIt to the Pi over LAN:

```bash
export ROS_DOMAIN_ID=23
```

Use the same `ROS_DOMAIN_ID` on both machines. ROS 2 discovery usually works on
local networks with multicast enabled. If topics do not appear on the laptop,
check firewall/multicast first:

```bash
ros2 topic list
ros2 node list
```

## Rosbag

Useful first recording set:

```bash
ros2 bag record /joint_states /tf /tf_static /kaivuri/tool_pose
```

Bag replay is safe with this initial stack because there is no ROS actuation node.
