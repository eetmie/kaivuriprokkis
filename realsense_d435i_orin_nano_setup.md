# RealSense D435i Setup on Jetson Orin Nano (JetPack 6.2.1) — RGB-Only

**Purpose:** Install the RealSense SDK (librealsense) on a Jetson Orin Nano so an Intel RealSense D435i works for **RGB streaming only**. The IMU and depth/infrared streams are intentionally not used.

**Target environment**
- Board: Jetson Orin Nano
- OS image: JetPack 6.2.1 (L4T r36.x, Ubuntu 22.04, Python 3.10)
- Camera: Intel RealSense D435i (USB)
- SDK target: librealsense **v2.57.7**

---

## Key decisions (do not skip)

1. **Build from source — do NOT use `pip install pyrealsense2`.** There are no aarch64/Jetson wheels on PyPI; pip will fail with `No matching distribution found`. The Python bindings are produced by the source build instead.
2. **Use the RSUSB (userspace) backend** via `-DFORCE_RSUSB_BACKEND=true`. JetPack 6 removed kernel HID support (`hiddraw`), which breaks the kernel/HID backend and can prevent the camera from enumerating at all. The RSUSB backend bypasses the kernel and needs no patching.
3. **v2.57.7** is the SDK version that supports JetPack 6.2. Anything older than 2.57.3 does not officially support JetPack 6.2.

---

## Step 1 — Install build dependencies

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential cmake git pkg-config \
  libssl-dev libusb-1.0-0-dev libudev-dev \
  v4l-utils libv4l-dev \
  libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
  libgtk-3-dev \
  python3 python3-dev python3-pip
```

> On a stock JetPack 6.2.1 image most of these are already present; only `libssl-dev`, `libusb-1.0-0-dev`, `libglfw3-dev`, `libglu1-mesa-dev`, and `libgtk-3-dev` actually needed installing in the verified run. The GL/GTK packages are only for the GUI examples (`realsense-viewer`); drop them for a headless build. `libusb-1.0-0-dev` is the one hard requirement for the RSUSB backend.
>
> The RSUSB backend uses libusb, **not** V4L — `v4l-utils`/`libv4l-dev` were not required to build or run in the verified run (the build reports `using RS2_USE_LIBUVC_BACKEND`). They're harmless to keep installed.

## Step 2 — Clone the repository and check out the target version

```bash
cd ~
git clone https://github.com/realsenseai/librealsense.git
cd librealsense
git checkout v2.57.7
```

> Note: The RealSense repos moved from the `IntelRealSense` org to the `realsenseai` org. The old URLs currently redirect but the canonical location is `github.com/realsenseai/librealsense`.

## Step 3 — Install udev rules

```bash
./scripts/setup_udev_rules.sh
```

## Step 4 — Configure the build

```bash
mkdir -p build && cd build
cmake .. \
  -DFORCE_RSUSB_BACKEND=true \
  -DBUILD_PYTHON_BINDINGS=true \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DBUILD_EXAMPLES=true \
  -DCMAKE_BUILD_TYPE=Release
```

## Step 5 — Compile and install

```bash
make -j$(nproc)
sudo make install
sudo ldconfig
```

> **If the build is killed (out of memory):** the Orin Nano can run out of RAM during compilation. Retry with fewer jobs (`make -j4` or `make -j2`), or add temporary swap before building:
> ```bash
> sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
> sudo mkswap /swapfile && sudo swapon /swapfile
> ```

## Step 6 — Python bindings (no action needed)

On this build `sudo make install` places the bindings in the **system** dist-packages directory, which is already on the default `sys.path`:

```
/usr/lib/python3/dist-packages/pyrealsense2/
```

So `import pyrealsense2` works out of the box — **no `PYTHONPATH` edit is required.** (Earlier notes pointing at `/usr/local/lib/python3.10/dist-packages/pyrealsense2` were wrong; that path is not created by this build.)

Confirm the install location if needed:

```bash
python3 -c "import pyrealsense2; print(pyrealsense2.__file__)"
```

> Verified on L4T R36.4.4 (JetPack 6.2.x), Ubuntu 22.04.5, Python 3.10.12, librealsense v2.57.7.

---

## Verification

Plug the camera into a **USB 3.x** port first.

1. **USB enumeration and negotiated speed:**
   ```bash
   lsusb | grep -i intel        # camera should appear
   lsusb -t                     # confirm it negotiated 5000M (USB 3.x), not 480M (USB 2.0)
   ```

2. **SDK sees the device:**
   ```bash
   rs-enumerate-devices
   ```
   Should list the D435i. If it says "no device found" while `lsusb` shows the camera, see Troubleshooting.

3. **Python bindings import:**
   ```bash
   python3 -c "import pyrealsense2 as rs; print('ok, classes:', hasattr(rs, 'pipeline'))"
   ```
   > Note: this build's `pyrealsense2` does **not** expose `rs.__version__` — checking for it raises `AttributeError` even though the module is fine. Test for a real class (e.g. `rs.pipeline`) instead, and use `rs-enumerate-devices` to read the SDK/firmware version.

4. **Grab a single RGB frame (sanity check):**
   ```bash
   python3 rgb_test.py
   ```
   Using the script in the next section.

---

## RGB-only streaming configuration

The camera only transmits a stream over USB if that stream is started. To minimize USB bandwidth, enable **only** the color stream — depth, infrared, and IMU then never start.

`rgb_test.py`:
```python
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()

# Enable ONLY the color stream. Depth / IR / IMU are never started.
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
try:
    for _ in range(30):                 # warm up / grab a few frames
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())
        print("Got RGB frame:", img.shape)
finally:
    pipeline.stop()
```

---

## Bandwidth tuning reference

If USB bandwidth is still constrained, adjust the color stream in this order of impact:

| Lever | Effect | How |
|---|---|---|
| Resolution | Largest single saving; bandwidth scales with pixel count | `1280x720` or `640x480` instead of `1920x1080` |
| Frame rate | Linear; 15 fps ≈ half of 30 fps | change the last arg in `enable_stream` |
| Pixel format | `yuyv` is 2 bytes/px (~33% less than `bgr8`); MJPEG is compressed (lightest, needs decode) | `rs.format.yuyv` |

Not enabling depth/infrared/IMU is the dominant saving and is already handled by the RGB-only config above.

---

## Troubleshooting

- **`pip install pyrealsense2` fails** → expected on Jetson/ARM. Use the source build (Steps 1–6). Do not retry pip.
- **CMake error about V4L / `v4l2` / missing video4linux headers** → install the V4L packages: `sudo apt-get install -y v4l-utils libv4l-dev`, then re-run cmake. (Already included in Step 1.)
- **`rs-enumerate-devices` shows "no device found" but `lsusb` lists the camera** → almost always the JetPack 6 kernel/HID backend issue. Confirm the build used `-DFORCE_RSUSB_BACKEND=true`; rebuild if not.
- **Camera negotiated USB 2.0 (`480M` in `lsusb -t`)** → use a USB-3 port and a known-good USB-3 cable. On USB 2.0 the higher resolution/fps combos are bandwidth-capped.
- **Build process killed** → out of RAM. Lower `make -j` count and/or add swap (see Step 5).
- **`import pyrealsense2` fails** (`ModuleNotFoundError`) → `sudo make install` / `sudo ldconfig` did not complete, or you built against a different Python than `python3`. Confirm the `.so` exists under `/usr/lib/python3/dist-packages/pyrealsense2/` and that `python3 --version` matches the `cpythonXY` tag on that file. No `PYTHONPATH` edit should be needed (see Step 6).
- **`AttributeError: module 'pyrealsense2' has no attribute '__version__'`** → not a failure; this build simply doesn't expose `__version__`. The import worked. Use `rs-enumerate-devices` for the version.

---

## Notes / caveats

- The 2.57.x line is officially tagged beta, but it is the line that carries JetPack 6.2 support. For RGB + depth use it is stable; the only flagged weakness is IMU stream quality, which is irrelevant here since the IMU is unused.
- Recommended D435i firmware is 5.17.0.10 or later. If needed, update with `rs-fw-update` after the SDK is installed.
