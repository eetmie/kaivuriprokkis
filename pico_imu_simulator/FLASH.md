# Flashing pico_imu_simulator to Seeed XIAO RP2040

## Requirements

- `arduino-cli` with `rp2040:rp2040` core installed
- Board: **Seeed XIAO RP2040** (`rp2040:rp2040:seeed_xiao_rp2040`)

## Normal upload (auto-reset)

```bash
arduino-cli compile --fqbn rp2040:rp2040:seeed_xiao_rp2040 pico_imu_simulator/
arduino-cli upload --fqbn rp2040:rp2040:seeed_xiao_rp2040 --port /dev/ttyACM0 pico_imu_simulator/
```

The Arduino core supports the 1200-baud bootloader reset — `arduino-cli upload` triggers it
automatically. If it works, the device reboots directly into the new firmware.

## Manual UF2 fallback (if auto-reset fails)

If the device doesn't re-enumerate as a serial port after upload, the RPI-RP2 drive may not
have auto-mounted. Check with `lsblk -o NAME,LABEL,MOUNTPOINT` — it will appear as `/dev/sda1`
with label `RPI-RP2`.

```bash
# Put XIAO into bootloader: hold BOOT, tap RESET, release BOOT
# Or let a failed arduino-cli upload trigger it, then:

sudo mount /dev/sda1 /mnt

arduino-cli compile --fqbn rp2040:rp2040:seeed_xiao_rp2040 \
    --output-dir /tmp/pico_sim pico_imu_simulator/

sudo cp /tmp/pico_sim/pico_imu_simulator.ino.uf2 /mnt/
# Device reboots automatically after copy
```

## Verify

```bash
python3 modules/usb_serial_reader.py --port /dev/ttyACM0
```

Expected startup sequence:
1. Config sent (`SR=200|GYRO_DPS=250|...`)
2. Calibration phase (~3 s, `CAL_WAIT` heartbeats)
3. Calibration report logged (4 sensors, accepted)
4. Descriptor frame: `4 sensor(s) @ 200 Hz`
5. Streaming pitch readouts at 200 Hz

## Tested

2026-05-07 — tested on Raspberry Pi 5, XIAO RP2040 on `/dev/ttyACM0`.
Full handshake, calibration, descriptor, and streaming verified against `modules/usb_serial_reader.py`.
