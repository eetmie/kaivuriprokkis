#!/usr/bin/env bash
set -euo pipefail

TAILSCALE_ROS2_DISCOVERY_HOST="${TAILSCALE_ROS2_DISCOVERY_HOST:-100.96.100.117}"
TAILSCALE_ROS2_DISCOVERY_PORT="${TAILSCALE_ROS2_DISCOVERY_PORT:-11811}"
ROS_IMAGE="${ROS_IMAGE:-ros:jazzy-ros-base}"

if [ -f /opt/ros/jazzy/setup.bash ]; then
  source /opt/ros/jazzy/setup.bash
  exec fastdds discovery \
    -i 0 \
    -l "${TAILSCALE_ROS2_DISCOVERY_HOST}" \
    -p "${TAILSCALE_ROS2_DISCOVERY_PORT}"
fi

exec docker run --rm -it \
  --network host \
  "${ROS_IMAGE}" \
  bash -lc "source /opt/ros/jazzy/setup.bash && exec fastdds discovery -i 0 -l '${TAILSCALE_ROS2_DISCOVERY_HOST}' -p '${TAILSCALE_ROS2_DISCOVERY_PORT}'"
