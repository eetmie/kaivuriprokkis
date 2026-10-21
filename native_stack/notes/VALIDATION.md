# Spark validation

Host: `thinkstationpgx-1a35`, aarch64. Date: 2026-10-11. GCC 11.5.0, C++17, CMake Release build. No Jetson or hardware tests in this phase.

## Checks completed

- Core checks: YAML loading, calibrated neutral/full-scale outputs, dither bounds, existing count conversion, analytical FK, right-side mounting correction, relative pitch, fragmented packets, checksum recovery, timestamp rollover/duplicates, nonfinite IMUs/actions, stale packets, firmware calibration gating, parser recovery after 100,000 pseudorandom bytes, arming, lease decay, fault latching, disabled axes, native chunk interpolation/exhaustion, observation expiry.
- Process tests: persistent command ownership, rejected second producer, lease expiry, explicit reset/rearm, emergency stop from another authorized connection, replay rejection, owner disconnect, missed control deadline after SIGSTOP/SIGCONT, actual Linux PTY input with real firmware-format packets, IMU timeout, two synthetic IR sources, coherent pixels, stale-camera rejection.
- Frame store: concurrent producer/consumer over 20,000 publication attempts, coherent metadata/pixels, uninitialized store, duplicate writer rejection, undersized consumer buffer, metadata-only read.
- Optional original-code parity: **12,192 cases** across `jetson`, `jetson_bucket`, `jetson_bucket_dz0` (the bucket profiles were dropped on 2026-10-21; the test now runs `jetson` only: 4,064 cases, max PWM 1.14e-11 us, FK 1.18e-7). Uses original PWM functions and original FK function bodies without hardware/JIT. Covers active dither, gamma/taper variants, ramp/deadband-skip variants, and 200 FK poses per profile. Maximum PWM difference **1.16e-11 us**; maximum FK position/quaternion component difference **1.12e-7** (existing reference uses float32). PWM count comparisons permit a one-count boundary difference from floating-point operation order.
- Address/undefined-behavior sanitizer run of core, frame store, and control IPC/PTY tests: passed. Sanitizer core-only configuration skips the SDK camera process test; Release configuration runs it with synthetic cameras.
- The pinned beta SDK compiled and installed into a private prefix with RSUSB, CUDA/DDS/Python bindings off. The revised build/install script also passed without invoking system `ldconfig`.

Commands, from repository root:

```bash
ctest --test-dir native_stack/build --output-on-failure
python3 -B native_stack/tests/reference_parity.py \
  native_stack/build/reference-probe "$PWD"

cmake -S native_stack -B native_stack/build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_CAMERA=OFF -DENABLE_SANITIZERS=ON
cmake --build native_stack/build-sanitize --parallel 2
ctest --test-dir native_stack/build-sanitize --output-on-failure
```

The parity script needs NumPy and PyYAML on the reference-test host. These are not native-service dependencies. Python is invoked with `-B` to avoid writing reference-package bytecode caches.

## Sampled memory baseline

20.10 s, 100 `/proc/<pid>/smaps_rollup` samples, command producer at 40 Hz, two synthetic 640×480 Y8 streams at 30 FPS, two active readers. Controller armed with dry PWM and simulated identity IMUs; pump permission off.

| Process | Start/end/peak RSS (MiB) | Peak PSS (MiB) | Swap (MiB) | CPU (% of one core) |
| --- | --- | --- | --- | --- |
| Control | 4.484 / 4.484 / 4.484 | 1.348 | 0 | 0.199 |
| Two synthetic IR sources | 15.168 / 15.168 / 15.168 | 11.993 | 0 | 0.547 |

Final control state: armed, 2038 ticks, maximum observed lateness 0.000819 s, no reported errors. The Python producer/consumer benchmark process is separate and not included in those rows. Shared mapped memory appears in multiple processes' RSS; use PSS to avoid double counting. The camera synthetic mode links the SDK but does not construct its USB context or real device buffers, so this is not a measured physical RealSense footprint. The control benchmark does not include physical I2C/serial work. A 20 s stable sample is not a long-term leak test.

Reproduce:

```bash
python3 -B native_stack/tests/spark_benchmark.py --seconds 20
```

Raw report is in ignored `results/spark-synthetic.json`. Hardware acceptance is deliberately pending: actual waveform, serial calibration, camera metadata/profile support, global time behavior, sync, USB disconnect handling, and Jetson memory/jitter under model load.

## Native episode recorder — 2026-10-14

Built on `pikkumasi2tb` in `/home/joel/kaivuriprokkis-recorder`, independently of
its existing checkout. CTest: core, frame-store, service-integration and new
recorder-integration all pass. The new suite checks applied-command attribution
and disabled channels; save/discard; SIGINT finalization; limited-file-size disk
failure; camera loss; retained incomplete takes; disarming; and export rejection
of truncated pixels or inaccurate FPS. An interrupted IPC poll originally failed
SIGINT save; retrying EINTR fixed it and the signal test now passes.

Real IMUs plus physical `233522073919:IR1` and `141722076647:IR2`, 30 seconds,
**dry PWM**, 200 Hz control and 100 Hz full-status capture: 897 frame pairs at
29.981 Hz, 3,002 telemetry rows. Maximum master arrival interval 33.450 ms; both
cameras reported zero capture-counter gaps/resets. Controller wake-up lateness
p99/max: 0.063/1.364 ms; controller work p99/max: 0.105/0.173 ms, including startup
and shutdown. These are dry timings, without inference or video encoding. Session
and per-tick CSV: `results/recorder-real-dual-ir-timing-2/` on the board. The previous
sampler-arrival timing gate could skip pairs; gating on master camera arrivals
fixed it. Export validation rejects the earlier affected take rather than
silently accepting an incorrect timebase.

A separate 10-second real session yielded 298 pairs at 29.990 Hz. Exported with
LeRobot 0.5.1 into two H.264 videos plus v3 Parquet. The actual LeRobot reader
successfully reopened it; first/middle/last views, IR indices, exact host
nanoseconds, applied actions, full/upper FK and corrected IMU fields round-trip.
All 100 Hz JSONL bytes are preserved in the exported sidecar. The optional
`tests/recorder_export.py SESSION` performs this test in an export environment.
Local artifacts: `results/recorder-validation/lerobot-real-10s/`,
`roundtrip-result.txt`, and `validation-summary.json`.

The IMU USB connection needed replugging before stable real capture resumed.
No physical gamepad was present, and no live PWM/pump/motion was enabled. Physical
pad direction checks, live teleoperation, and long-session/inference-load tests
remain explicit follow-up acceptance work; they are not inferred from this dry
capture test.
