# ROS 2 bridge

`kaivuri_bridge.py` puts the native stack on the topics the Isaac sim publishes (sim-scratchpad `ros2/`), so one
viewer, and later one client, works on either. It follows the excavator-vla topic contract v1 for the machine-facing
topics. Read-only by default (STATUS polls only); `live --commands` also accepts `/excavator/command`.

ROS 2 Jazzy runs bare metal on the board (`ros-jazzy-ros-base`, from the official ROS apt source, 2026-11-10).
The bridge is one Python file; no colcon build.

```bash
source /opt/ros/jazzy/setup.bash
python3 -B native_stack/ros2/kaivuri_bridge.py live [--calibration SESSION/sand-calibration.json]
python3 -B native_stack/ros2/kaivuri_bridge.py replay SESSION --episode 1   # a recorded take, looped
```

`live` reads the controller socket and the `kaivuri-camera` frame store at their default paths (`--socket`,
`--frames`). Viewer, on any machine with the sim-scratchpad checkout and `rviz2`:

```bash
ros2 launch scripts/sim-scratchpad/ros2/view_excavator.launch.py sim:=false   # sim:=true for Isaac
```

| Topic | Type | Rate | Notes |
| --- | --- | --- | --- |
| `/joint_states` | `sensor_msgs/JointState` | 50 Hz | URDF joints `revolute_carriage, revolute_lift, revolute_tilt, revolute_tool` (rad), for `robot_state_publisher` |
| `/excavator/joint_angles` | `std_msgs/Float32MultiArray` | 50 Hz | contract `[slew, lift, tilt, scoop]` in [-1, 1]; not published while unhealthy |
| `/excavator/feedback_healthy` | `std_msgs/Bool` | 50 Hz | no controller fault, IMUs healthy, FK valid, angles inside the profile bounds |
| `/excavator/adapter_info` | `std_msgs/String` | 1 Hz | `interface_id`, `calibration_id` derived from `kaivuri.profile.json` as the shared stack does |
| `/excavator/cameras/{cabin,arm}/image_raw` | `sensor_msgs/Image` mono8 | 5 Hz (`--camera-hz`) | IR; frames `cabin_camera`, `arm_camera` |
| `/excavator/cameras/{cabin,arm}/depth/{image_raw,camera_info}` | `Image` 16UC1 mm, `CameraInfo` | 5 Hz (`--depth-hz`) | cameras started with `--depth`; finer camera depth units are converted to mm |
| `/excavator/episode` | `std_msgs/String` JSON, latched | on the pad | A starts an episode of `--task` or ends it as a success, B ends it as a failure |
| `/tf_static` | | | with `--calibration`: the depth frames of both cameras on `upper_carriage` / `tilt_boom` |

## Commands, height map and a policy (2026-11-16)

`live --commands` sends `/excavator/command` `[slew, lift, tilt, scoop]` in [-1, 1] as native DRIVE rows (times
`native_command_sign`) on the bridge's one connection, which then owns the drive lease. The controller is armed
only from an all-zero command; a command older than 150 ms, an invalid one, an unhealthy IMU or a lost gamepad
disarms. Pump permission never comes from ROS: X on the gamepad toggles it. Checked on the board with live PWM
and the hydraulics unpowered: signs as in the profile, stop 140 ms after the last command.

The whole observation side of the sim runs on the board: with ROS sourced, build sim-scratchpad `ros2/setup_ws.sh`
(colcon from a venv is enough; 2 minutes), then

```bash
kaivuri-control ... & kaivuri-camera --camera 233522073919:2 --camera 141722076647:1 --rotate-180 233522073919 \
    --depth 141722076647 &
python3 -B native_stack/ros2/kaivuri_bridge.py live --commands --camera-hz 60 --calibration CAL --task "..." &
ros2 launch ~/sim-scratchpad/ros2/view_excavator.launch.py sim:=false rviz:=false teleop:=false calibration:=CAL
```

gives depth, the self-filtered cloud, FastDEM's scan and `/excavator/heightmap/image` at 5.0 Hz each (FastDEM
9 % and the bridge 9 % of one core). The sand reads about -0.23 m in `upper_carriage` here, against -0.15 m in
the sim (mid-gray): the height image is darker than the sim's until one of the two is moved.

A policy runs through ros-excavator `policy_node.py`. With an X-VLA TensorRT bundle it uses the python
branch's engine runtime, no LeRobot or Torch:

```bash
source /opt/ros/jazzy/setup.bash && cd ~/ros-excavator
~/kaivuriprokkis-python/.venv/bin/python policy_node.py profiles/kaivuri_xvla_ir.json \
    ~/bundles/xvla-masi_dual_camera-c30x10-full-17000-fp16 --execute 15
```

Measured with that bundle, both IR cameras at 30 Hz and live PWM (hydraulics unpowered): chunks of 30 in
342 ms (11 steps), commands at 29.9 Hz, the controller armed throughout an episode and neutral after it,
3.1 GB of RAM in use. `--execute` must leave more rows than the latency: 15 for a 30-row chunk.

URDF positions are the native angles plus fixed offsets (0, 0, −0.02813, +0.00355 rad), same signs: the URDF stick
line is 1.6° below its link axis and its bucket lip 0.2° off the native tip. Checked against the native FK
(boom 0.468, arm 0.240, tip [−0.009, 0, −0.1335]) and `sand_height.py`'s camera mounts at four poses: pivots and
camera frames agree exactly, the bucket lip to 1.3 mm (the sim bucket is 0.6 mm shorter).

Measured on pikkumasi2tb, dry controller with real IMUs and both IR cameras: 65 MB PSS, 5 % of one core,
50.8 Hz state and 5.0 Hz per camera. ROS discovery works on one LAN as is; across subnets (Tailscale) set
`ROS_STATIC_PEERS` to the other machine's address on both ends. Two raw IR feeds at 5 Hz are 3 MB/s, which a
Tailscale link could not carry (state fell to 2.6 Hz); with `--camera-hz 0.5` state held 49 Hz.

## Not done yet

- Command signs in the profile are not commissioned: check each axis with powered motion at a small magnitude
  first. Slew also needs `kaivuri-control --enable-slew`.
- Recorded depth in `replay`; compressed images for WiFi viewers; the arm camera in the map (its mount is a
  placeholder in sim-scratchpad `sensors.py`).
- The recorded angles pass the profile bounds by ~1-2° at the end stops (sand-depth-001: boom -51°, arm 145.5°,
  bucket -121.3° / +21.7°), so `feedback_healthy` is false 18-28 % of a dig take. Widen the bounds (it changes the
  normalization) or keep them.
- A TensorRT bundle of the sim-trained (cabin + height map) model: the tested bundle is the dual-IR X-VLA.
