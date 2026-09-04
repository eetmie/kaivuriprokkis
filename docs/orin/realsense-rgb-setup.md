# realsense-rgb — D435i RGB on JetPack 7.2 (no kernel patches)

The SmolVLA runtime uses **only the D435i color stream**. That removes all the hard
parts of the old JetPack 6 setup: no PREEMPT_RT kernel, no UVC metadata patches, no
HID-sensor kernel modules. We build **librealsense with the userspace USB backend**
(`FORCE_RSUSB_BACKEND=ON`), which drives the camera over libusb on the **stock
JetPack 7.2 kernel** — nothing in `/boot` or `/lib/modules` is touched.

## Install

```bash
./install-realsense-rgb.sh        # builds + installs librealsense + pyrealsense2 (cp312)
```

Override the release tag if needed: `RS_TAG=v2.57.7 ./install-realsense-rgb.sh`.

## Verify

```bash
rs-enumerate-devices | head
python3 -c 'import pyrealsense2 as rs; print("pyrealsense2 ok")'
# end-to-end RGB through the runtime (no model needed):
cd ../smolvla-runtime && python run_pipeline.py --backend mock --source realsense --duration-s 5
```

`pyrealsense2` installs under `/usr/local/lib/.../dist-packages`; the runtime's
`camera.py` already adds the Jetson system-install paths to `sys.path`, so a venv
created with `--system-site-packages` sees it.

## Out of scope (deferred)

Depth, the onboard IMU (accel/gyro), hardware frame metadata, and a real-time
(PREEMPT_RT) kernel for a 100 Hz on-device control loop. Those are exactly what
required the kernel rebuild + patches before — re-introduce them only if a future
task needs depth or the camera IMU. RGB-only needs none of it.
