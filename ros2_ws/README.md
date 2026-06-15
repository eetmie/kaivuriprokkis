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

For UDP joystick control, use the separate mapper node. The UDP receiver keeps
publishing the MotionPlatform packet values on `/joystick_values`
(`[axis0..axis7, button_mask]`), and `joystick_to_direct_pwm_node` converts
the selected axis values from int8 range to normalized `[-1.0, 1.0]` commands
on `/kaivuri/direct_pwm`.

```bash
ros2 launch kaivuri_bringup joystick_direct_drive.launch.py
```

From the operator machine, MotionPlatform sends to the robot UDP server:

```bash
python main.py --ip <robot-ip>:8080
```

Default mapper order:

```text
joystick_values[0] -> rotate
joystick_values[1] -> lift_boom
joystick_values[2] -> tilt_boom
joystick_values[3] -> scoop
joystick_values[4] -> trackL
joystick_values[5] -> trackR
```

While driving, the same node publishes:

- `/joint_states`
- `/kaivuri/tool_pose`

## IK End-Effector Control

Use this only when the machine is safe to move. This starts the existing Python
`ExcavatorController` in IK mode and exposes ROS topics for end-effector
targets. The controller still owns the IK, reachability check, PID loop, and PWM
output.

```bash
cd /home/joel/kaivuriprokkis
ros2-jazzy bash -lc 'cd /work/ros2_ws && source install/setup.bash && ros2 launch kaivuri_bringup ik_pose_control.launch.py'
```

Target topics:

```text
/kaivuri/target_pose
geometry_msgs/msg/PoseStamped
```

Position is `[x, y, z]` in the `excavator` frame. The orientation quaternion is
reduced to the body-frame tool pitch that the current 4-joint excavator can
control.

```text
/kaivuri/target_pose_y
std_msgs/msg/Float32MultiArray
```

Data order:

```text
[x, y, z, rot_y_deg]
```

Example small target:

```bash
ros2 topic pub --once /kaivuri/target_pose_y std_msgs/msg/Float32MultiArray \
  "{data: [0.45, 0.00, 0.02, 0.0]}"
```

The node clears the active IK target if no new target arrives within
`command_timeout_s`, default `1.0`. While running, it publishes:

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

## Red Cube Expert Data Collection

Use this as the first Isaac Sim data-collection loop for VLA/behavior-cloning
experiments. Isaac Sim should spawn a red cube and publish its pose:

```text
/kaivuri/cube_pose
geometry_msgs/msg/PoseStamped
```

`cube_touch_expert_node` turns that cube pose into a smooth expert trajectory:

```text
approach above cube -> descend to top -> hold -> retract
```

It publishes:

```text
/kaivuri/target_pose_y        std_msgs/msg/Float32MultiArray  [x, y, z, rot_y_deg]
/kaivuri/episode_event        std_msgs/msg/String             "<episode_id>:<event>"
/kaivuri/task_instruction     std_msgs/msg/String
```

Start the IK node first:

```bash
source /work/ros2_ws/install/setup.bash
ros2 run kaivuri_bringup ik_pose_control_node --ros-args \
  -p visualization_only:=true \
  -p project_root:=/work
```

Then start the cube-touch expert:

```bash
ros2 run kaivuri_bringup cube_touch_expert_node
```

For a ROS-only smoke test without Isaac, publish random cube poses:

```bash
ros2 run kaivuri_bringup random_cube_pose_node
```

Or launch both helper nodes together:

```bash
ros2 launch kaivuri_bringup cube_touch_data_collection.launch.py use_random_cube:=true
```

Recommended rosbag command for the first dataset:

```bash
ros2 bag record \
  /camera/camera/color/image_raw \
  /camera/camera/color/camera_info \
  /joint_states \
  /kaivuri/tool_pose \
  /kaivuri/target_pose_y \
  /kaivuri/cube_pose \
  /kaivuri/episode_event \
  /kaivuri/task_instruction
```
