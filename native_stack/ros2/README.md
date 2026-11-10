# ROS 2 bridge

`kaivuri_bridge.py` puts the native stack on the topics the Isaac sim publishes (sim-scratchpad `ros2/`), so one
viewer, and later one client, works on either. It follows the excavator-vla topic contract v1 for the machine-facing
topics. **Read-only for now:** it only polls STATUS and never sends DRIVE.

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
| `/tf_static` | | | with `--calibration`: the depth frames of both cameras on `upper_carriage` / `tilt_boom` |

URDF positions are the native angles plus fixed offsets (0, 0, −0.02813, +0.00355 rad), same signs: the URDF stick
line is 1.6° below its link axis and its bucket lip 0.2° off the native tip. Checked against the native FK
(boom 0.468, arm 0.240, tip [−0.009, 0, −0.1335]) and `sand_height.py`'s camera mounts at four poses: pivots and
camera frames agree exactly, the bucket lip to 1.3 mm (the sim bucket is 0.6 mm shorter).

Measured on pikkumasi2tb, dry controller with real IMUs and both IR cameras: 65 MB PSS, 5 % of one core,
50.8 Hz state and 5.0 Hz per camera. ROS discovery works on one LAN as is; across subnets (Tailscale) set
`ROS_STATIC_PEERS` to the other machine's address on both ends. Two raw IR feeds at 5 Hz are 3 MB/s, which a
Tailscale link could not carry (state fell to 2.6 Hz); with `--camera-hz 0.5` state held 49 Hz.

## Not done yet

- **Commands:** `/excavator/command` → native DRIVE, ported from `~/kaivuri-ros-fit` (11-04 dry fit: 150 ms command
  expiry, arming only from neutral, one socket owns the drive lease). Command signs in the profile are not
  commissioned: check each axis with powered motion at a small magnitude first.
- **Pump permission** stays with the robot's gamepad, as a toggle (X, like `vla_live.py`); ROS never grants it.
  Future: a client-side toggle (e.g. the contract's `OperatorState`) as a second way to switch it, with the pad
  still able to turn it off.
- Depth point cloud and `/excavator/heightmap` from `sand_height_live.py`; `camera_info`; compressed images for
  WiFi viewers.
- The recorded angles pass the profile bounds by ~1-2° at the end stops (sand-depth-001: boom -51°, arm 145.5°,
  bucket -121.3° / +21.7°), so `feedback_healthy` is false 18-28 % of a dig take. Widen the bounds (it changes the
  normalization) or keep them.
- Inference: the sim runs stock LeRobot; the robot runs the TRT bundles from jetson-orin-nano-vla, hooked back to
  LeRobot later. One LeRobot robot class over these topics could then serve both.
