# RealSense logging — stream choice and USB headroom

**Date:** 2026-08-10 · **Board:** Orin Nano Super, L4T R39.2.0 (JetPack 7), kernel 6.8.12-tegra,
Ubuntu 24.04, Python 3.12 · **SDK:** librealsense 2.57.7, RSUSB backend · **Camera:** D435i, FW 5.17.3.10

## Decision

Log **infrared cam1 only** at **640x480**. No RGB, no depth, no IMU streams from the camera.

**20 Hz is not selectable.** The D435i offers 6 / 15 / 30 / 60 / 90 fps for `infrared` idx1 `y8`
at 640x480 — 20 is not in the list. Use **15 fps** (nearest below 20). Both 15 and 30 were
verified clean, so 30 stays available if temporal resolution matters more than disk.

| Config | Data rate | Per minute |
|---|---|---|
| IR1 y8 640x480 @15 | 4.6 MB/s | 0.28 GB |
| IR1 y8 640x480 @30 | 9.2 MB/s | 0.55 GB |
| *(previous plan: +RGB bgr8 @30)* | *36.8 MB/s* | *2.2 GB* |

## Measured headroom

5-minute concurrent test — RealSense (color+IR1 @640x480x30) + 4 Pico IMUs @200 Hz + Xbox pad,
with a 100 Hz control loop:

- Camera: 8993 frames per stream, **0 dropped**, arrival jitter std 0.155 ms
- Control loop: 30000 ticks in 300.0000 s = 99.99997 Hz, period std 0.062 ms, 0 overruns >20 ms
- IMU: 199.94–199.97 sps, **0 checksum/header failures, 0 reconnects**
- No xHCI errors or bandwidth warnings in dmesg; system load ~11% of 6 cores, 48 °C

USB bandwidth is not a constraint and was never close to one. The camera is on **Bus 002**
(USB 3, 5 Gbps root hub) at ~6% utilisation; the Pico and pad are on **Bus 001** (USB 2, 480M)
at <0.1%. Separate host controllers — the camera cannot starve the other two.

IR1-only re-verified standalone: 15 fps → 14.999 measured, std 0.103 ms, 0 dropped.
30 fps → 29.993 measured, std 0.089 ms, 0 dropped.

## Gotchas

1. **Turn the emitter off.** IR1 comes off the stereo module, so the laser projector is on by
   default (`emitter_enabled = 1.0`) even with depth disabled — otherwise every IR frame carries
   the projected dot pattern:
   ```python
   prof.get_device().first_depth_sensor().set_option(rs.option.emitter_enabled, 0)
   ```

2. **Leave `global_time` enabled (the default).** The host monotonic clock and the D435i hardware
   clock differ by **82 ppm** — measured ~4.9 ms/min, ~0.3 s/hour. `global_time` regresses the
   hardware clock onto the host clock and absorbs this. Recording raw `hardware_clock` stamps
   instead leaves a 30-minute session ~150 ms out of sync with the actuator/IMU logs.

3. **`pyrealsense2` is not in `.venv`.** It is a system install at
   `/usr/local/lib/python3.12/dist-packages`. Venv code needs either
   `PYTHONPATH=/usr/local/lib/python3.12/dist-packages` or a `.pth` file dropped into
   `.venv/lib/python3.12/site-packages/`.

## Not yet tested

Ingest only. No hydraulics/PWM I2C traffic, no VLA inference, and **no bag writing** — rosbag2
serialisation and mcap encoding of the image stream is the remaining unknown. Disk is fine
(NVMe, 57 GB free); the open question is CPU.
