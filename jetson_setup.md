# MASI Robot — Jetson Setup & Test Runbook

Board: NVIDIA Jetson Orin Nano Super 8GB
Script: `setup_jetson.sh`
RT requirement: **JetPack 6.2.1 / Jetson Linux r36.4.x** (last release with prebuilt RT kernel packages)

This is the bring-up procedure for the Jetson. Run the stages in order and verify each before moving on.

## What the script does

1. Installs OS packages: `python3-pip`, `python3-venv`, `git`, `i2c-tools`, `device-tree-compiler`, `rt-tests`.
2. Creates/updates the project `.venv` and installs `numpy`, `scipy`, `numba`, `pandas`, `pyyaml`, `pyserial`, `smbus2`. `pandas` is required at runtime — `simple_drive.py --record` writes its CSV through it.
3. Adds the invoking user to `dialout`, `i2c`, `gpio`, `plugdev` (where present).
4. **I2C bus 7 → 1 MHz**: builds and applies a device-tree overlay (`clock-frequency = <1000000>` on `/bus@0/i2c@c250000`) via Jetson-IO. Static single speed — correct for a single-device bus.
5. **RT kernel**: installs NVIDIA prebuilt RT packages from the r36.4 repo, but only after force-checking the board is on r36.4. Skips otherwise.
6. Adds the `kaivuri-venv` shell alias.

Boot config (`/boot/extlinux/extlinux.conf`) is backed up before steps 4 and 5. Nothing reboots automatically.

## Toggles (env vars)

| Var | Default | Effect |
|---|---|---|
| `SETUP_I2C_1MHZ` | `1` | Set `0` to skip the I2C overlay |
| `SETUP_RT_KERNEL` | `1` | Set `0` to skip the RT kernel install |
| `FORCE_RT` | `0` | Set `1` to install RT even if not on r36.4 (**not recommended**) |

## Pre-flight

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release            # expect: # R36 (release), REVISION: 4.x  (== JetPack 6.2.1)
uname -a
tr -d '\0' < /proc/device-tree/aliases/i2c7; echo
readlink -f /sys/bus/i2c/devices/i2c-7/of_node   # expect: .../i2c@c250000
```

If `/etc/nv_tegra_release` is not R36 REVISION 4.x, the RT step will skip by design.

## Staged run

Run each stage, reboot, verify, then continue. Staging keeps a boot failure attributable to one change.

```bash
# Stage 1 — base provisioning only (robot still at 400 kHz)
sudo SETUP_I2C_1MHZ=0 SETUP_RT_KERNEL=0 ./setup_jetson.sh
# log out / back in (or reboot) for group membership, then sanity-check the robot

# Stage 2 — add 1 MHz I2C overlay
sudo SETUP_RT_KERNEL=0 ./setup_jetson.sh
sudo reboot
# verify clock + scope the bus (below)

# Stage 3 — RT kernel last
sudo SETUP_I2C_1MHZ=0 ./setup_jetson.sh
sudo reboot
# verify RT + cyclictest (below)
```

## Verify

I2C bus 7 clock (expect `1000000`):

```bash
python3 -c "import struct; print(struct.unpack('>I', open('/proc/device-tree/bus@0/i2c@c250000/clock-frequency','rb').read(4))[0])"
i2cdetect -l | grep -E 'i2c-7|c250000'
sudo i2cdetect -y 7        # PCA9685 should answer at 0x40
```

Confirm the 1 MHz waveform on a scope/logic analyzer at the device pins — a correct device-tree value does not guarantee a clean bus (pull-ups, cable length, Fast-Mode-Plus support).

RT kernel:

```bash
uname -a
grep -E 'PREEMPT_RT' /boot/config-$(uname -r) | head
dpkg -l | grep nvidia-l4t-rt-kernel
sudo cyclictest -p 80 -t$(nproc) -n -i 1000 -l 100000   # run under real workload for a true number
```

Make RT the default boot entry (or revert):

```bash
sudo sed -i 's/^DEFAULT .*/DEFAULT real-time/' /boot/extlinux/extlinux.conf   # use RT
sudo sed -i 's/^DEFAULT .*/DEFAULT primary/'   /boot/extlinux/extlinux.conf   # back to stock
sudo reboot
```

## Run the robot

```bash
cd <project-dir>
source .venv/bin/activate          # or: kaivuri-venv
python run_hw_v2.py --robot jetson ...
```

Avoid plain `sudo python ...` for serial — it bypasses the venv. Serial works after the group reboot.

## Rollback

```bash
ls -lt /boot/extlinux/extlinux.conf.bak-*
sudo cp /boot/extlinux/extlinux.conf.bak-YYYY-MM-DD-HHMMSS /boot/extlinux/extlinux.conf
sudo reboot
```

Remove RT packages:

```bash
sudo apt remove -y nvidia-l4t-rt-kernel nvidia-l4t-rt-kernel-headers \
  nvidia-l4t-rt-kernel-oot-modules nvidia-l4t-display-rt-kernel
sudo reboot
```

If the board will not boot: restore `/boot/extlinux/extlinux.conf` from the backup via serial console or by mounting the boot partition from another Linux machine.

## Watch-fors during testing

- Stage 2: the script greps Jetson-IO for the `HDR40 I2C7 1MHz` entry and prints a WARN if the board's compatible string did not match. If you see that WARN, the overlay was staged but may not apply.
- Stage 3: the RT gate reads `/etc/nv_tegra_release`. Wrong release → it skips and tells you, rather than installing the r36.4 repo onto the wrong base.
- Running stages 2 and 3 separately produces one `extlinux.conf` backup each — expected.
