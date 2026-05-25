#!/bin/bash
# MASI Robot Jetson Setup Script
#
# Minimal Jetson bring-up. Jetson uses only the main header I2C bus for the
# PCA9685 on this robot; that bus is Linux I2C bus 7. IMU data comes over USB
# serial. This intentionally does not configure Raspberry Pi boot overlays,
# virtual i2c-gpio buses, or the OLED systemd service from setup.sh.
#
# TODO(jetson): test whether the main I2C bus 7 can be driven at 1 MHz for the
#               PCA9685. The 1 MHz main bus setup works well on Raspberry Pi,
#               but Jetson uses different device-tree/kernel configuration, so
#               the RPi dtparam=i2c_arm_baudrate path does not apply here.
#
# It installs basic OS packages, creates/updates a lightweight project .venv,
# and grants the invoking user access to serial/I2C/GPIO device groups where
# those groups exist. Analysis packages and Raspberry Pi OLED/display packages
# are intentionally not installed here. After group changes, log out and back
# in or reboot.

set -e

echo "========================================"
echo "  MASI Robot Jetson Setup"
echo "========================================"

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo ./setup_jetson.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USERNAME="${SUDO_USER:-$USER}"
USER_HOME=$(getent passwd "$USERNAME" | cut -d: -f6)
VENV_DIR="${SCRIPT_DIR}/.venv"

echo ""
echo "[1/4] Installing system packages..."
apt-get update -qq
apt-get install -y python3-pip python3-venv i2c-tools git > /dev/null 2>&1
echo "  OK: System packages installed"
echo "  NOTE: Jetson PCA9685 is expected on I2C bus 7; run 'i2cdetect -y 7' to check."
echo "  NOTE: 1MHz I2C on Jetson bus 7 is not configured by this script yet."

echo ""
echo "[2/4] Creating/updating project virtualenv..."
if [ ! -d "$VENV_DIR" ]; then
    sudo -u "$USERNAME" python3 -m venv "$VENV_DIR"
fi

sudo -u "$USERNAME" "$VENV_DIR/bin/python" -m pip install --upgrade pip
sudo -u "$USERNAME" "$VENV_DIR/bin/python" -m pip install \
    numpy \
    numba \
    pyyaml \
    pyserial \
    smbus2
echo "  OK: Python packages installed into ${VENV_DIR}"
echo "  NOTE: Skipped Raspberry Pi OLED/display packages and analysis-only scipy/pandas."

echo ""
echo "[3/4] Granting device group access..."
for group in dialout i2c gpio plugdev; do
    if getent group "$group" > /dev/null; then
        usermod -aG "$group" "$USERNAME"
        echo "  OK: ${USERNAME} added to ${group}"
    else
        echo "  SKIP: group ${group} does not exist on this image"
    fi
done

echo ""
echo "[4/4] Adding shell helper..."
BASHRC="${USER_HOME}/.bashrc"
if ! grep -q "kaivuri-venv" "$BASHRC" 2>/dev/null; then
    cat >> "$BASHRC" << EOF

# MASI robot virtualenv
alias kaivuri-venv='cd ${SCRIPT_DIR} && source .venv/bin/activate'
EOF
    echo "  OK: kaivuri-venv alias added"
else
    echo "  OK: kaivuri-venv alias already present"
fi

echo ""
echo "========================================"
echo "  Jetson Setup Complete"
echo "========================================"
echo ""
echo "Run robot code with:"
echo "  cd ${SCRIPT_DIR}"
echo "  source .venv/bin/activate"
echo "  python run_hw_v2.py --robot jetson ..."
echo ""
echo "Do not use plain 'sudo python ...' for normal USB serial access; it bypasses"
echo "the project virtualenv. USB serial access should work after reboot if the"
echo "device is owned by dialout or another group added above."
echo ""
echo ">>> REBOOT OR RELOGIN REQUIRED FOR GROUP CHANGES <<<"
