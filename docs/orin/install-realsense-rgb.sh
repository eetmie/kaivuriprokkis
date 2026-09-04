#!/usr/bin/env bash
# RGB-only Intel RealSense D435i on JetPack 7.2 — the LIGHT path.
#
# We only need the D435i's *color* stream for SmolVLA. That means NO kernel work:
# no PREEMPT_RT kernel, no UVC metadata patches, no HID-sensor modules. We build
# librealsense with the userspace USB backend (FORCE_RSUSB_BACKEND=ON), which talks
# to the camera over libusb and runs on the stock JetPack 7.2 kernel untouched.
# (Depth metadata + the onboard IMU are what needed the kernel patches — not RGB.)
#
# Produces: librealsense + the `pyrealsense2` Python binding (cp312) that camera.py
# imports. Run once:  ./install-realsense-rgb.sh
set -euo pipefail

RS_TAG="${RS_TAG:-v2.57.7}"           # librealsense release tag (override via env)
SRC_DIR="${SRC_DIR:-$HOME/src/librealsense}"
JOBS="$(nproc)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing build dependencies (apt) ..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git cmake build-essential pkg-config \
  libusb-1.0-0-dev libudev-dev libssl-dev \
  libgl1-mesa-dev libglu1-mesa-dev \
  python3-dev

echo "==> Installing the libusb udev rules (RGB/RSUSB, no kernel modules) ..."
sudo cp "${HERE}/udev/99-realsense-libusb.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "==> Fetching librealsense ${RS_TAG} ..."
if [[ ! -d "${SRC_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${RS_TAG}" https://github.com/IntelRealSense/librealsense "${SRC_DIR}"
else
  git -C "${SRC_DIR}" fetch --depth 1 origin "${RS_TAG}"
  git -C "${SRC_DIR}" checkout "${RS_TAG}"
fi

echo "==> Configuring (RSUSB userspace backend, Python bindings, no examples) ..."
cmake -S "${SRC_DIR}" -B "${SRC_DIR}/build" \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_WITH_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local

echo "==> Building with ${JOBS} jobs (this takes a while) ..."
cmake --build "${SRC_DIR}/build" -j "${JOBS}"

echo "==> Installing ..."
sudo cmake --install "${SRC_DIR}/build"
sudo ldconfig

echo
echo "==> Done. Verify (unplug/replug the D435i first):"
echo "    rs-enumerate-devices | head"
echo "    python3 -c 'import pyrealsense2 as rs; print(\"pyrealsense2 ok\")'"
