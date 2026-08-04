#!/bin/bash
# MASI Robot Jetson Setup Script
#
# Minimal Jetson bring-up. Jetson uses only the main header I2C bus for the
# PCA9685 on this robot; that bus is Linux I2C bus 7. IMU data comes over USB
# serial. This intentionally does not configure Raspberry Pi boot overlays,
# virtual i2c-gpio buses, or the OLED systemd service from setup.sh.
#
# I2C bus 7 speed:
#   The Jetson stock default for node /bus@0/i2c@c250000 is 400 kHz, which the
#   PCA9685 handles fine, so by default this script leaves the bus alone.
#   1 MHz is opt-in (SETUP_I2C_1MHZ=1). The RPi dtparam=i2c_arm_baudrate path
#   does NOT apply to Jetson; when opted in, the 1 MHz speed is set with a
#   device-tree overlay that bumps clock-frequency on that node and is applied
#   through Jetson-IO (step [4/5]), taking effect after reboot. Before going to
#   1 MHz, check pull-ups / rise time on a scope. See the I2C7 guide section 4.
#
# RT (PREEMPT_RT) kernel:
#   Not handled here. NVIDIA's prebuilt RT kernel packages were published for
#   JetPack 6.2.1 / Jetson Linux r36.4.x only; there is no equivalent for
#   JetPack 7 / r39.x, so the old auto-install step could never do anything but
#   print a skip notice. If you need PREEMPT_RT on a supported release, install
#   it deliberately rather than as a side effect of this script.
#   Note that rt-tests (cyclictest) is still installed below -- it is useful for
#   latency baselining on the stock PREEMPT kernel.
#
# It installs basic OS packages, creates/updates a lightweight project .venv,
# and grants the invoking user access to serial/I2C/GPIO device groups where
# those groups exist. Analysis packages and Raspberry Pi OLED/display packages
# are intentionally not installed here. After group/overlay changes, reboot.
#
# Optional toggles (env vars):
#   SETUP_I2C_1MHZ=1   enable the I2C-7 1 MHz overlay (default: off => stock 400 kHz)

set -e

# --- Robot-specific constants (per the I2C7 guide) ------------------------
I2C_BUS=7
I2C_NODE_PATH="/bus@0/i2c@c250000"   # what i2c-7 must map to on Orin Nano

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
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y \
    python3-pip python3-venv git \
    i2c-tools device-tree-compiler rt-tests > /dev/null 2>&1
echo "  OK: System packages installed"
echo "  NOTE: Jetson PCA9685 is expected on I2C bus ${I2C_BUS}; run 'i2cdetect -y ${I2C_BUS}' to check."

echo ""
echo "[2/5] Creating/updating project virtualenv..."
if [ ! -d "$VENV_DIR" ]; then
    sudo -u "$USERNAME" python3 -m venv "$VENV_DIR"
fi

sudo -u "$USERNAME" "$VENV_DIR/bin/python" -m pip install --upgrade pip
sudo -u "$USERNAME" "$VENV_DIR/bin/python" -m pip install \
    numpy \
    scipy \
    numba \
    pandas \
    pyyaml \
    pyserial \
    smbus2 \
    inputs
echo "  OK: Python packages installed into ${VENV_DIR}"
echo "  NOTE: Skipped Raspberry Pi OLED/display packages."
# inputs is pure Python and tiny; it backs modules/gamepad.py (XboxController).
# It reads /dev/input/event* directly, so the user also needs the 'input' group
# granted in step [3/5] -- without it, get_gamepad() raises PermissionError.
# pandas is NOT analysis-only on this robot: simple_drive.py --record buffers
# samples in memory and writes the CSV via pandas in DataLogger.save(). Without
# it, stopping a recording raises ModuleNotFoundError and the segment is lost.

echo ""
echo "[3/5] Granting device group access..."
for group in dialout i2c gpio plugdev input; do
    if getent group "$group" > /dev/null; then
        usermod -aG "$group" "$USERNAME"
        echo "  OK: ${USERNAME} added to ${group}"
    else
        echo "  SKIP: group ${group} does not exist on this image"
    fi
done

echo ""
echo "[4/5] I2C bus ${I2C_BUS} speed (1 MHz overlay is opt-in)..."
if [ "${SETUP_I2C_1MHZ:-0}" != "1" ]; then
    echo "  SKIP: 1 MHz overlay not requested; bus stays at stock 400 kHz."
    echo "        Set SETUP_I2C_1MHZ=1 to build and apply the 1 MHz overlay."
else
    echo "  Requested 1 MHz; building and applying device-tree overlay..."
    # Verify what Linux calls i2c-7 actually maps to the expected DT node.
    I2C7_OF_NODE=""
    if [ -e "/sys/bus/i2c/devices/i2c-${I2C_BUS}/of_node" ]; then
        I2C7_OF_NODE=$(readlink -f "/sys/bus/i2c/devices/i2c-${I2C_BUS}/of_node" || true)
    fi
    echo "  i2c-${I2C_BUS} of_node: ${I2C7_OF_NODE:-<not found>}"
    case "$I2C7_OF_NODE" in
        *i2c@c250000) : ;;  # expected
        "") echo "  WARN: i2c-${I2C_BUS} not present right now; writing overlay anyway." ;;
        *)  echo "  WARN: i2c-${I2C_BUS} maps to an unexpected node (expected ...${I2C_NODE_PATH#*/bus@0})." ;;
    esac

    if ! command -v dtc >/dev/null 2>&1; then
        echo "  ERROR: dtc (device-tree-compiler) not found; cannot build overlay. Skipping."
    elif [ ! -x /opt/nvidia/jetson-io/config-by-hardware.py ]; then
        echo "  WARN: Jetson-IO not found at /opt/nvidia/jetson-io/; this may not be a Jetson L4T image."
        echo "        Skipping overlay apply."
    else
        # Build the overlay 'compatible' list: board-reported strings first,
        # then the known Orin Nano/NX p3768+p3767 SKUs as fallback (deduped).
        COMPAT_RAW=$(
            {
                if [ -e /sys/firmware/devicetree/base/compatible ]; then
                    tr '\0' '\n' < /sys/firmware/devicetree/base/compatible
                fi
                printf '%s\n' \
                    "nvidia,p3768-0000+p3767-0005" \
                    "nvidia,p3768-0000+p3767-0004" \
                    "nvidia,p3768-0000+p3767-0003" \
                    "nvidia,p3768-0000+p3767-0001" \
                    "nvidia,p3768-0000+p3767-0000"
            } | awk 'NF && !seen[$0]++'
        )
        # Format as a DTS list: indented quoted strings, ';'-terminated.
        COMPAT_DTS=$(printf '%s\n' "$COMPAT_RAW" \
            | sed 's/^/                 "/; s/$/",/' \
            | sed '$ s/",$/";/')

        cat > /tmp/i2c7-1mhz.dts <<EOF_DTS
/dts-v1/;
/plugin/;

/ {
    overlay-name = "HDR40 I2C7 1MHz";
    jetson-header-name = "Jetson 40pin Header";

    compatible =
${COMPAT_DTS}

    fragment@0 {
        target-path = "${I2C_NODE_PATH}";
        __overlay__ {
            status = "okay";
            clock-frequency = <1000000>;
        };
    };
};
EOF_DTS

        if dtc -@ -I dts -O dtb -o /tmp/i2c7-1mhz.dtbo /tmp/i2c7-1mhz.dts 2>/tmp/i2c7-dtc.log; then
            cp /tmp/i2c7-1mhz.dtbo /boot/
            echo "  OK: overlay compiled and copied to /boot/i2c7-1mhz.dtbo"

            # Applying the overlay rewrites boot config; back it up first.
            if [ -f /boot/extlinux/extlinux.conf ]; then
                cp /boot/extlinux/extlinux.conf \
                   "/boot/extlinux/extlinux.conf.bak-$(date +%F-%H%M%S)"
                echo "  OK: backed up /boot/extlinux/extlinux.conf"
            fi

            set +e
            /opt/nvidia/jetson-io/config-by-hardware.py -l 2>/dev/null \
                | grep -q "HDR40 I2C7 1MHz"
            SEEN=$?
            if [ $SEEN -ne 0 ]; then
                echo "  WARN: Jetson-IO did not list 'HDR40 I2C7 1MHz' (compatible may not match this board)."
            fi
            /opt/nvidia/jetson-io/config-by-hardware.py -n "HDR40 I2C7 1MHz"
            APPLY=$?
            set -e
            if [ $APPLY -eq 0 ]; then
                echo "  OK: overlay applied via Jetson-IO (takes effect after reboot)."
            else
                echo "  ERROR: Jetson-IO apply failed (status $APPLY). Boot config backup is in /boot/extlinux/."
            fi
        else
            echo "  ERROR: dtc failed to compile the overlay. See /tmp/i2c7-dtc.log. Skipping."
        fi
    fi
fi

echo ""
echo "[5/5] Adding shell helper..."
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
echo "Activate the project virtualenv with:"
echo "  kaivuri-venv"
echo ""
echo "After rebooting, verify the changes:"
echo "  # I2C bus ${I2C_BUS} clock-frequency (400000 stock, 1000000 if 1 MHz overlay was applied):"
echo "  python3 -c \"import struct; print(struct.unpack('>I', open('/proc/device-tree${I2C_NODE_PATH}/clock-frequency','rb').read(4))[0])\""
echo "  # PCA9685 should answer at 0x40 (0x70 is its All-Call address):"
echo "  i2cdetect -y ${I2C_BUS}"
echo "  # IMU Pico should enumerate as a USB serial device:"
echo "  ls -l /dev/serial/by-id/"
echo "  # Gamepad (optional) should enumerate and be readable via the 'input' group:"
echo "  .venv/bin/python -c \"import inputs; print([g.name for g in inputs.devices.gamepads])\""
echo ""
echo ">>> REBOOT REQUIRED for group changes and the I2C 1 MHz overlay <<<"
