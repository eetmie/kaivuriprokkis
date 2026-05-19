# ROS 2 Jazzy over Tailscale

This guide sets up the Kaivuri ROS 2 IMU state publisher on `pikkukaivuri-dev`
and a subscriber on this machine over Tailscale.

Plain ROS 2 discovery usually does not work across Tailscale because ROS 2 DDS
discovery expects LAN multicast, and Tailscale does not forward multicast. Use a
Fast DDS discovery server so the ROS 2 nodes discover each other over unicast.

## Subscriber Machine

Run the discovery server on this subscriber side.

Get the subscriber machine's Tailscale IPv4 address:

```bash
tailscale ip -4
```

Start the discovery server, replacing `100.x.y.z` with this machine's Tailscale
IP:

```bash
source /opt/ros/jazzy/setup.bash
fastdds discovery -i 0 -l 100.x.y.z -p 11811
```

Keep that terminal running.

In another terminal on this same subscriber machine:

```bash
export ROS_DOMAIN_ID=23
export ROS_DISCOVERY_SERVER=100.x.y.z:11811
export ROS_LOCALHOST_ONLY=0

source /opt/ros/jazzy/setup.bash
source ~/kaivuriprokkis/ros2_ws/install/setup.bash

ros2 topic list
ros2 topic echo /joint_states
```

## Publisher Machine: pikkukaivuri-dev

On `pikkukaivuri-dev`, run the ROS 2 Jazzy environment with host networking so
DDS can use the host Tailscale interface.

```bash
cd /home/joel/kaivuriprokkis
ros2-jazzy
cd /work/ros2_ws
```

Inside that ROS 2 environment, point ROS 2 at the subscriber machine's discovery
server:

```bash
export ROS_DOMAIN_ID=23
export ROS_DISCOVERY_SERVER=100.x.y.z:11811
export ROS_LOCALHOST_ONLY=0

source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

Launch the hardware IMU state publisher:

```bash
ros2 launch kaivuri_bringup bringup_hardware.launch.py
```

Expected published topics include:

```text
/joint_states
/kaivuri/tool_pose
/tf
/tf_static
```

## Checks

Run these on both machines:

```bash
echo "$ROS_DOMAIN_ID"
echo "$ROS_DISCOVERY_SERVER"
tailscale ping 100.x.y.z
```

Run these on the subscriber side:

```bash
ros2 node list
ros2 topic list
ros2 topic echo /joint_states
```

Both ends must use the same `ROS_DOMAIN_ID`, and both ends must point
`ROS_DISCOVERY_SERVER` at the subscriber machine's Tailscale IP and discovery
server port.
