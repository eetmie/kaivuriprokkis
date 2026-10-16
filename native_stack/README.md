# Native control + IR camera bench stack

Built on Spark on 2026-10-11 for the next hardware session. Everything new lives in this folder. The existing Python stack and Pico firmware remain the reference and are unchanged.

This is a working C++17 bench implementation: independent safety/valve service, Pico IMU parser, forward kinematics, and an IR-only RealSense service. The control and camera services have no inference runtime, ROS, IK/PID, Jacobian/SVD, video encoder, or GPU context. An optional separate C++ TensorRT process runs the validated OpenPI pi0.5 LIBERO policy for bench measurements. Python monitoring tools use the standard library.

| Component | Current behavior |
| --- | --- |
| `kaivuri-control` | Configurable loop (100 Hz default); disarmed boot; one command owner; deadman/lease/IMU/deadline checks; direct Linux I2C PWM; complete per-tick FK/IMU snapshot |
| `kaivuri-camera` | One or two explicitly selected cameras; one IR1 or IR2 Y8 stream per camera; native timestamp metadata; latest frame in mapped memory |
| `kaivuri-inference` | Optional OpenPI pi0.5 LIBERO TensorRT + AOT CUDA graph; five-fixture parity gate; two fresh IR inputs; normalized actions recorded without actuator IPC |
| `kaivuri-record` | C++ gamepad teleoperation, bounded dual-IR Y8 episode spool and full FK/IMU/action snapshots; offline LeRobot v3 export |
| `tools/client.py` | Status, disarm, emergency stop, bounded bench exercise; reusable IPC client for later TRT integration |
| `tools/frame_reader.py` | Coherent Y8 snapshot through the native C ABI; rejects stale frames |
| `tools/measure.py` | Per-process RSS, PSS, swap, CPU; no model or hardware imports |

The default control mode reads real IMUs but **does not open I2C**. `--simulate-imu` gives a completely hardware-free control run. `--live-pwm` enables physical output and refuses simulated IMUs. Slew and tracks also require their own daemon flags. The pump requires an explicit command flag and an armed state.

FK status includes tool XYZ/quaternion, upper-carriage XZ/XYZ/quaternion and bucket pitch, separate slew yaw, every joint pivot XYZ and link quaternion in both frames, local joint rotation quaternions, and firmware/corrected IMU quaternions plus gyro/acceleration. One snapshot is computed per healthy tick, without heap allocation, and reused by status readers. `fk_valid` and IMU timing distinguish valid observations from stale data. See [field names and frame conventions](notes/PROTOCOL.md#kinematic-observation-fields).

For a full-stack stress run, add `--control-hz 200 --status-hz 200 --live-pwm` to the bench command below. It keeps control disarmed and logs complete FK/IMU replies to `fk-status.jsonl` while writing neutral PWM, capturing both IR streams and running pi0.5. High-rate observations stream to disk rather than accumulating in RAM. The controller CSV measures FK work separately from whole-tick work; the summary reports achieved status rate and distinct IMU/tick counts. This does not include LeRobot video encoding.

Optional `--rate-hz 2` on the inference executable or bench tool limits inference starts to one per 500 ms. Warm-up remains unrestricted. It sleeps on the host before reading the latest camera frames, then runs inference and records the result immediately; it does not retain an image across the pacing wait. Overruns start one fresh job when ready and schedule the following job from that actual start, so there is no catch-up burst or observation backlog. Shutdown interrupts the wait. CSV reports `rate_wait_ms` and `rate_lateness_ms`, and the inference summary includes the configured rate and work-overrun count. Pacing frees compute time between jobs; resident model weights/buffers remain allocated.

Prediction horizon, execution FPS and inference refresh rate are separate settings. OpenPI's default prediction horizon is 50, but this validated LIBERO bundle exports **10 action rows** with 10 denoising steps. The measured latency belongs to that 10-row export. At a future 30 Hz excavator action rate, 10 rows cover only 333 ms, which is insufficient for continuous execution at the measured latency. A 30–50-row excavator export is a starting point to evaluate: 50 rows span 1.67 s, 2 Hz replanning advances about 15 rows per update, and the current inference delay consumes about 13 additional 30 Hz steps. Longer exports may change inference latency; verify the resulting timing and plan coverage together. Switching chunks requires explicit delay handling and overlapping execution, rather than replaying an old chunk from its first row. Changing prediction horizon requires a new export and quality/performance validation. The inference bench currently publishes no actuator commands and does not implement RTC.

## Episode recording

The native recorder pairs camera `233522073919:2` (rotated 180°) with `141722076647:1`, runs teleoperation independently of its bounded disk writer, and exports committed takes to LeRobot 0.5.1/v3 after capture. See [recording instructions, controls and data fields](notes/RECORDING.md). Capture requires no LeRobot, Torch or Python loop.

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

## OpenPI pi0.5 C++ bench

Tested on `pikkumasi2tb`, Ubuntu 24.04, L4T R39.2.1, CUDA 13.2 and TensorRT 10.16.2.10. The runtime ports the device chain from `jetson-orin-nano-vla/bench/vendor/pi05_trt.py`: refitted layer templates, one shared scratch allocation, device-resident K/V and Euler updates, and the existing serialized AOT attention plugin. It requires CUDA/TensorRT development files and OpenSSL. No Python interpreter, Torch, Triton compiler, or ONNX Runtime participates in the inference loop.

Use the **existing validated bundle and optimized cache built on this board**. The cache's template/engine checksums and each refitted weight checksum are checked. Before opening live input, the executable tests all five full normalized action chunks against the export's FP16 reference (cosine >= 0.99999 and maximum absolute error <= 0.005) and requires exact image preprocessing parity.

```bash
cd ~/kaivuriprokkis/native_stack
VLA_ROOT=/home/joel/vla-clean-install-20261012

# Build-time extraction only; verifies the portable bundle's MANIFEST.sha256.
"$VLA_ROOT/source/.venv-trt/bin/python" tools/prepare_pi05.py \
  --bundle "$VLA_ROOT/bundles/pi05-libero" --out results/pi05-parity-fixtures

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_INFERENCE=ON \
  -DCMAKE_PREFIX_PATH="$PWD/.deps/realsense-2.58.4"
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure

# New output directory per run. Control stays dry and disarmed, with real IMUs.
python3 -B tools/bench_pi05.py --build build \
  --bundle "$VLA_ROOT/bundles/pi05-libero" \
  --cache "$VLA_ROOT/runs/pi05-libero/optimized-cache" \
  --fixtures results/pi05-parity-fixtures \
  --camera 233522073919:1 --camera 141722076647:1 \
  --duration-s 300 --out-dir results/pi05-dual-ir-new
```

Skip fixture preparation when that output directory already exists. `kaivuri-inference --help` describes standalone validation and capture inputs. `kaivuri-camera --list` reports firmware, SDK USB version and physical port; both tested cameras negotiate 5 Gbit/s on paths `2-1.3` and `2-1.4` under the 10 Gbit/s hub. Device enumeration is retained for both streams to avoid reopening an already claimed RSUSB interface.

The bench reports per-stage inference time, start interval variation, oldest camera arrival to completed action chunk, camera arrival intervals and callback counter gaps, RSS/PSS/swap, whole-system memory, CPU usage, and the control loop's wake-up lateness. Optional `kaivuri-control --timing-csv FILE --duration-s SEC` records in preallocated memory and writes only after neutral shutdown. Camera CSV logging can skip rows under load; callback gap counts distinguish that from actual missing frames.

This is an **unfinetuned LIBERO performance demo**. IR gray is replicated into three channels with OpenPI's antialiased resize and centered padding. Prompts use the exported token table; normalized seven-dimensional LIBERO outputs are not excavator valve commands. Fine-tuning must establish the IR view order/orientation, proprioception, action meaning, normalization and training FPS; then export that checkpoint and implement its runtime boundary. The current LIBERO exporter pins a specific public checkpoint, so a custom checkpoint needs an export path adapted to its trained contract. Camera arrival time is measured; exposure-to-host clock mapping and synchronized exposure are not established by this bench.

After the register-level hardware checks, `bench_pi05.py --live-pwm` also completed a 60-second inference run with real 100 Hz neutral I2C writes. It stays disarmed and refuses simulated IMUs. Inference p50/p99 was 425.50/426.98 ms; control wake-up lateness p99/max was 0.070/0.242 ms, and per-tick control work p99 was 1.108 ms. Both camera callback counters had zero gaps/resets. One camera skipped two shared-frame publications (latest-frame writer busy) across capture including model load; the other skipped none. Evidence is in `results/pi05-dual-ir-live-pwm-60s-20261012/`.

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

FK/output extension session spanning 2026-10-12/13: local and Jetson CTest pass, and all 12,192 original-code parity cases pass across three profiles (maximum FK component difference `1.12e-7`). The **10-minute 200 Hz live-neutral full-stack run** validated 125,452 complete FK/IMU snapshots against independent Python FK, including all pivots and quaternions in both frames. No whole-tick work or wake-up lateness exceeded the 5 ms budget, including startup. During steady inference, FK preparation p50/p99 was 2.37/5.86 microseconds and control work p99/max was 1.25/2.47 ms. Control wake-up lateness p99/max was 0.071/0.300 ms. Full-status logging achieved 199.99 Hz; Pico decoding and control ticks both averaged 200 Hz. All sampled IMUs/FK stayed healthy. Both cameras remained near 30 fps with zero capture-counter gaps/resets and five combined latest-frame writer-busy skips. Minimum available system RAM was 1.056 GiB, median 1.074 GiB; control, camera and inference had no process swap. Inference p50/p99 was 427.32/429.97 ms over 1,390 runs. Artifacts: `results/pi05-fk200-full-stack-600s-20261012/`. The monitoring script hit `/proc` process-exit `ProcessLookupError` after inference completed normally; summaries were recovered from complete logs, the original monitor session retained, and that race fixed.

The following **five-minute paced comparison** completed 600 runs at 2 Hz: median start interval 500.059 ms, maximum 500.113 ms, zero inference work overruns. Median wait between jobs was 68.59 ms (about 13.7% of the 500 ms cycle). The wait precedes capture: oldest camera arrival age at input was 18.34 ms median and 33.24 ms p99; arrival-to-result median was 449.41 ms. Inference p50/p99 was 427.03/428.73 ms. 200 Hz control/FK and status logging continued; every one of 65,557 logged FK snapshots independently validated, and no 5 ms work/lateness budget was exceeded. Cameras had zero capture-counter gaps/resets and one shared-frame publication skip. Minimum available RAM was 1.070 GiB and model allocations remained resident. Both runs used disarmed neutral I2C writes; PCA9685 readback confirmed calibrated neutral and pump minimum after shutdown. Artifacts: `results/pi05-fk200-paced2hz-300s-20261012/`. Large full-status JSONL streams remain on the board; timing traces, summaries and calibration snapshots are also copied locally. These runs test the **10-action-row, 10-denoising-step LIBERO export**; they establish neither finetuned policy quality nor longer-horizon inference speed, and include no LeRobot video encoder. Cameras remain USB-only/free-running, with real timestamps and pair skew recorded.

Jetson session on 2026-10-12: all three CTest checks and all five C++ OpenPI policy fixtures pass. A 300-second run with two real D435i IR1 streams and healthy real Pico IMUs completed 698 inferences. Inference p50/p99 was 425.45/427.00 ms (0.672 ms standard deviation); oldest camera arrival to result p50/p99 was 451.22/463.32 ms. The dry 100 Hz controller's wake-up lateness p99/max was 0.071/0.816 ms. Both cameras sustained 29.99 fps with zero callback counter gaps, resets or writer drops. Whole-system RAM in use was about 6.27 GiB with at least 1.08 GiB available; these services had no process swap. RSS was 5.63 MiB for the instrumented controller and 24.30 MiB for capture. Inference RSS alone omits most CUDA allocations, so whole-system RAM is the appropriate fit measurement. Raw evidence and summary are in `results/pi05-dual-ir-300s-20261012/` on the board and local checkout.

With servos and pump confirmed unpowered, PCA9685 neutral, bounded boom output, pump permission, second-producer rejection, timeout decay, emergency stop and owner-disconnect checks passed using real IMUs and register readback. Shutdown left calibrated neutral and pump minimum. This establishes the programmed register behavior; physical waveform, hydraulic motion and independent watchdog acceptance remain pending.

Earlier Spark dry control measured **4.48 MiB RSS**; the process producing two synthetic IR streams measured **15.17 MiB RSS**, both stable across the sampled 20 seconds. See [the original validation](notes/VALIDATION.md).

Start tomorrow with [the hardware runbook](notes/HARDWARE_RUNBOOK.md). The critical limitation is unchanged by the language choice: a PCA9685 keeps its last outputs after a controller crash or `SIGKILL`. Physical pump disable or a hardware watchdog is needed to cover that failure. Normal process shutdown and detected faults write calibrated neutral plus pump minimum.

Further details: [architecture and research](notes/ARCHITECTURE.md), [TRT integration contract](notes/PROTOCOL.md), [remaining work](notes/TODO.md).
