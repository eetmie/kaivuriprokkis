# Native control + IR camera bench stack

Built on Spark on 2026-10-11 for the next hardware session. Everything new lives in this folder. The existing Python stack and Pico firmware remain the reference and are unchanged.

This is a working C++17 bench implementation: independent safety/valve service, Pico IMU parser, forward kinematics, and an IR-only RealSense service. Python tools use the standard library. There is no inference runtime, ROS, IK/PID, Jacobian/SVD, recording pipeline, or GPU context in these services.

| Component | Current behavior |
| --- | --- |
| `kaivuri-control` | 100 Hz loop; disarmed boot; one command owner; deadman/lease/IMU/deadline checks; direct Linux I2C PWM; FK status |
| `kaivuri-camera` | One or two explicitly selected cameras; one IR1 or IR2 Y8 stream per camera; native timestamp metadata; latest frame in mapped memory |
| `tools/client.py` | Status, disarm, emergency stop, bounded bench exercise; reusable IPC client for later TRT integration |
| `tools/frame_reader.py` | Coherent Y8 snapshot through the native C ABI; rejects stale frames |
| `tools/measure.py` | Per-process RSS, PSS, swap, CPU; no model or hardware imports |

The default control mode reads real IMUs but **does not open I2C**. `--simulate-imu` gives a completely hardware-free control run. `--live-pwm` enables physical output and refuses simulated IMUs. Slew and tracks also require their own daemon flags. The pump requires an explicit command flag and an armed state.

## Build

Run from this folder. Needed: C++17 compiler, CMake, Git, pkg-config, Python 3, yaml-cpp development package, and libusb development package. The SDK build may also need the distro's OpenSSL development package. On Jetson, rebuild from source locally; Spark binaries and library paths are not a deployment package.

```bash
cd ~/kaivuriprokkis/native_stack

# Core-only build: no camera SDK needed.
cmake -S . -B build-control -DCMAKE_BUILD_TYPE=Release -DBUILD_CAMERA=OFF
cmake --build build-control --parallel 2
ctest --test-dir build-control --output-on-failure

# Full build: pinned beta SDK in .deps/, no system installation or kernel patches.
bash scripts/build_realsense.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/.deps/realsense-2.58.4"
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The SDK is [RealSense beta v2.58.4](https://github.com/realsenseai/librealsense/releases/tag/v2.58.4), commit `34d6c778e1134d8505adcd56bb57acbe7598a459`. This build uses RSUSB, disables DDS/CUDA/Python bindings/examples/tools, and installs only under `.deps/realsense-2.58.4`. The script omits upstream's generated `ldconfig` step. Existing camera access permissions are still required.

## Hardware-free use

In one terminal:

```bash
./build/kaivuri-control --simulate-imu
```

In another:

```bash
python3 -B tools/client.py status
python3 -B tools/client.py exercise --values 0,0.1,-0.1,0.1,0,0 --seconds 2
python3 -B tools/client.py disarm
```

Synthetic camera producer:

```bash
./build/kaivuri-camera --synthetic --camera test1:1 --camera test2:2
```

Reader, in another terminal:

```bash
python3 -B tools/frame_reader.py \
  "/tmp/kaivuri-native-$(id -u)/cameras/test1-ir1.frame" --seconds 5
```

Simulated IMUs emit identity raw quaternions. Existing mounting offsets still apply, so simulated reported joint angles need not all be zero.

## Results and next steps

Spark dry control measured **4.48 MiB RSS**; the process producing two synthetic IR streams measured **15.17 MiB RSS**, both stable across the sampled 20 seconds. Real camera/serial/I2C loads and Jetson memory are still unmeasured. See [validation](notes/VALIDATION.md).

Start tomorrow with [the hardware runbook](notes/HARDWARE_RUNBOOK.md). The critical limitation is unchanged by the language choice: a PCA9685 keeps its last outputs after a controller crash or `SIGKILL`. Physical pump disable or a hardware watchdog is needed to cover that failure. Normal process shutdown and detected faults write calibrated neutral plus pump minimum.

Further details: [architecture and research](notes/ARCHITECTURE.md), [TRT integration contract](notes/PROTOCOL.md), [remaining work](notes/TODO.md).
