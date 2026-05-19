#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ROS_IMAGE="${ROS_IMAGE:-ros:jazzy-ros-base}"

source "${SCRIPT_DIR}/tailscale_ros2_env.bash"

echo "Waiting for /joint_states from pikkukaivuri-dev..."

if [ -f /opt/ros/jazzy/setup.bash ]; then
  source /opt/ros/jazzy/setup.bash
  source "${SCRIPT_DIR}/install/setup.bash"
  exec ros2 topic echo /joint_states
fi

exec docker run --rm -it \
  --network host \
  -v "${PROJECT_ROOT}:/work" \
  -w /work/ros2_ws \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -e ROS_DISCOVERY_SERVER="${ROS_DISCOVERY_SERVER}" \
  -e ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
  "${ROS_IMAGE}" \
  bash -lc 'source /opt/ros/jazzy/setup.bash && source install/setup.bash && exec ros2 topic echo /joint_states'
