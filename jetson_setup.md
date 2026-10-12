# MASI Jetson manual bring-up

Validated on `pikkumasi2tb` on 2026-10-12: Orin Nano, Ubuntu 24.04.4,
L4T R39.2.1, kernel `6.8.12-1021-tegra`, Python 3.12.3.
Repository: `/home/joel/kaivuriprokkis`.

`setup_jetson.sh` installs a Python environment, device groups, realtime limits,
an optional CPU governor service and an opt-in I2C overlay. It does **not** install
an RT kernel. Its defaults leave I2C speed unchanged. This runbook records the
manual configuration used alongside the previously validated VLA installation.

## Packages and Python environment

Install missing native build dependencies without upgrading existing packages:

```bash
sudo apt-get update
sudo apt-get install --no-upgrade --no-install-recommends \
  python3-pip python3-venv git i2c-tools device-tree-compiler rt-tests \
  build-essential cmake pkg-config libyaml-cpp-dev libusb-1.0-0-dev libssl-dev
```

The validated host uses CMake `3.31.9-0kitware1ubuntu24.04.1`; retain it rather
than implicitly upgrading to a different major version. CUDA/TensorRT development
files were already supplied by the VLA host setup.

For the existing Python robot stack, create an independent environment:

```bash
cd ~/kaivuriprokkis
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy scipy numba pandas pyyaml pyserial smbus2 inputs
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > /tmp/kaivuri-python-requirements.txt
```

The board's installed versions are saved in
`native_stack/results/manual-bringup/python-requirements.txt`. This environment
is separate from `/home/joel/vla-clean-install-20261012/source/.venv-*`.
The C++ inference process does not use it. The shell helper added to `~/.bashrc` is:

```bash
alias kaivuri-venv='cd /home/joel/kaivuriprokkis && source .venv/bin/activate'
```

## Device access and realtime limits

Add `joel` to the groups present on this image:

```bash
sudo usermod -aG dialout,i2c,gpio,plugdev,input joel
```

`/etc/security/limits.d/90-kaivuri-realtime.conf`:

```text
joel - rtprio 90
joel - memlock unlimited
```

A new SSH login confirms the groups and `ulimit -r` = 90. Reconnecting is enough
for these settings; no reboot was performed.

`/etc/udev/rules.d/99-kaivuri-realsense.rules` grants access to the two D435i devices:

```text
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="8086", ATTR{idProduct}=="0b3a", MODE="0660", GROUP="plugdev"
```

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --action=change --subsystem-match=usb \
  --attr-match=idVendor=8086 --attr-match=idProduct=0b3a
sudo udevadm settle --timeout=10
```

The native RealSense build uses RSUSB, so this bench needs no UVC kernel patches
or Python camera bindings. USB autosuspend remains at its original settings.

## Clocks, kernel and I2C

The validated VLA setup already enabled `jetson-perf.service`. Its verifier
confirms MAXN_SUPER, all six CPUs fixed at 1728 MHz, GPU at 1020 MHz and EMC at
3199 MHz:

```bash
sudo /usr/local/sbin/jetson-perf.sh --verify
```

The CPU governor reports `schedutil`, but its frequency floor equals its ceiling;
no additional `kaivuri-cpu-performance.service` was installed.
The stock kernel and boot configuration were retained. I2C bus 7 remains 400 kHz.
There is no 1 MHz overlay or RT kernel installation in this session.

The PCA9685 is expected at `0x40` on bus 7. This controller does not support SMBus
Quick Write, so plain `i2cdetect -y 7` skips some addresses, including `0x40`.
A targeted register read is a better presence check:

```bash
cd ~/kaivuriprokkis
.venv/bin/python - <<'PY'
from smbus2 import SMBus
with SMBus(7) as bus:
    print('PCA9685 MODE1:', hex(bus.read_byte_data(0x40, 0)))
PY
```

Use the native hardware runbook for neutral writes, bounded exercises and
instrumented waveform checks. Register readback verifies programmed counts;
it does not establish physical pulse accuracy or valve/pump behavior.

## Native control, cameras and inference

Build instructions and the five-minute dual-IR OpenPI pi0.5 bench are in
[`native_stack/README.md`](native_stack/README.md). Hardware checks are in
[`native_stack/notes/HARDWARE_RUNBOOK.md`](native_stack/notes/HARDWARE_RUNBOOK.md).

Both D435i cameras have firmware `5.17.3.10` and negotiate 5 Gbit/s on the
10 Gbit/s hub:

| SDK serial | Linux USB path | SDK port | Link |
| --- | --- | --- | --- |
| 233522073919 | 2-1.3 | 2-1.3-3 | 5 Gbit/s |
| 141722076647 | 2-1.4 | 2-1.4-4 | 5 Gbit/s |

Pico startup calibration normally takes about 13 seconds and can retry for
roughly 35 seconds. Wait for `imu_healthy` and `firmware_blocked=false` before
arming; an initial calibration-wait report is not a permanent failure.
