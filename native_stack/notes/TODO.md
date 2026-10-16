# Work items

## Completed on Spark

- [x] Separate C++ safety/IMU/valve/FK process; retain existing profiles and calibration math.
- [x] Safe boot, centered arming, explicit pump permission, bounded IPC, owner/sequence checks, timeout decay, latched failures.
- [x] Real serial packet parser with checksum/numeric/calibration/time validation.
- [x] Direct Linux I2C implementation and neutral shutdown path; physical waveform validation remains pending.
- [x] Pinned RealSense v2.58.4 beta build isolated in this folder.
- [x] IR1/IR2 Y8 capture, explicit serial selection, raw internal timestamp metadata, two-device support.
- [x] Bounded coherent shared frame store and stdlib consumer/client tools.
- [x] Core/process/frame-store tests, original-code parity, sanitizer checks, Spark memory baseline.
- [x] Hardware session commands and result sheet.

## Tomorrow: hardware acceptance

- [x] Build on Jetson and record versions/permissions/profile.
- [ ] Validate Pico startup calibration, mounting signs, angles and known-pose FK.
- [ ] Measure neutral/pump/active pulse widths and compare with legacy output.
- [ ] Confirm physical stop behavior for deadman/lease/IMU/producer failures, with an independent stop available.
- [x] Verify IR-only streaming and sensor/readout timestamps through RSUSB on the installed firmware.
- [x] Verify IR1 and IR2, auto-exposure metadata, frame gaps, and stale-frame handling.
- [ ] Tune manual exposure/gain and settle the physical camera orientation for training.
- [x] Add second camera; verify both 5 Gbit/s links and dual capture under inference load.
- [ ] Characterize device-clock drift and test hardware synchronization if needed.
- [x] Measure RSS/PSS/swap/jitter with real cameras and IMUs under OpenPI pi0.5 inference load.
- [x] Expose per-tick full/upper tool poses, every pivot/link pose, local joint quaternions and raw/corrected IMU observations.
- [x] Validate 200 Hz real IMU/control/FK with neutral live I2C, dual IR, pi0.5 and 200 Hz full-status logging for 10 minutes; independently replay every logged snapshot.
- [x] Add fresh-frame inference pacing and verify 2 Hz under the same load for five minutes.
- [ ] Longer camera run through timestamp rollover; longer native memory-stability run.

## After the bench session

- [ ] Independent MCU/hardware watchdog for retained PCA9685 outputs and physical pump disable. This is the highest-priority remaining stability/safety gap.
- [ ] Optional deployment/supervision files only after hardware paths/timings are established; none are installed now.
- [ ] Add a clock mapper for camera exposure timestamps, host monotonic time, and Pico device time; characterize latency/drift before fusion.
- [x] Port the validated OpenPI pi0.5 GPU chain to a separate C++ TensorRT runtime, with full fixture parity and live IR consumption through the frame C ABI.
- [ ] Integrate the fine-tuned runtime with the command interface; preserve actual training FPS, normalization, state/action meaning and observation timestamps. The LIBERO bench records actions only.
- [ ] Rebuild operator deadman/manual/task-switch supervision in that producer, disarming on source/task changes.
- [ ] Set lease/observation limits from measured worst inference latency and required stop time. Do not silently refresh old observation timestamps.
- [ ] Evaluate beta GPU-frame APIs only when profiling shows selected-frame host copies matter; current capture does not enable CUDA zero-copy.
- [ ] Compare native total system memory with the legacy stack under the same model/device load before stating an actual Jetson memory saving.
- [ ] Fine-tuned 30 Hz action contract: benchmark a 30–50-step prediction export and delay-aware overlapping execution. Current LIBERO performance bundle has only 10 action rows.
- [x] Add a native C++ dual-IR episode spool with bounded asynchronous disk writes, full FK/IMU and applied/requested actions, calibration provenance, failure/discard lifecycle and offline LeRobot 0.5.1/v3 export. See RECORDING.md.
- [ ] Validate physical gamepad mapping, live teleoperation/stop behavior and long-session memory/disk stability; video encoding is offline and has not been tested concurrently with inference.
