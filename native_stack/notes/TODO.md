# Work items

## MLP data collection

Implementation order and acceptance criteria are in the
[native MLP logging plan](MLP_LOGGING_PLAN.md). One 100 Hz process writes the
PCA9685 directly, without arming or other software gates, and is kept separate
from the VLA recorder.

- [x] Per-packet `ImuParser` emission, shared `Pwm`/gamepad modules and an IMU byte-replay input.
- [x] `kaivuri-mlp-collect`: manual driving, pump toggle, `drive_log`/`imu_raw`/`excitation` output.
- [x] Port sine/chirp/stroke excitation with cross-language fixtures.
- [x] Broaden default frequency coverage and deploy generator version 4 to the isolated 2026-10-20 Jetson tree; local 89 Python/7 native checks, Jetson 7 native checks and a fresh real-Pico dry run pass.
- [x] `pump_on` handling in `training/drive_logs.py`; validate q/v/u on a replayed strip.
- [x] 30-minute real-IMU/live-neutral-PWM Jetson run with servo/pump power off; zero missed periods/drops, flat 12.52 MiB RSS and neutral register readback. See [MLP validation](MLP_VALIDATION.md).
- [ ] Operator: physical gamepad mapping and a short powered collection before full dataset collection.

The randomized IK/PID core now has its own initial collector described in
[separate PID collection](PID_COLLECTION.md), alongside the offline preview in
[IK-assisted collection](IK_COLLECTION_PLAN.md). Its control flow remains separate
from the manual MLP collector. Old MLP driving, circle tests and model training
remain outside this logging milestone.

## pi0.5 chunk consistency (RTC)

- [ ] Inference-time soft inpainting between chunks (latency-aligned anchor, blend op in the denoise loop). Scope, reference mechanism and patch list: [RTC scope](RTC_SCOPE.md); phased work plan (A now, C-ready): [RTC plan](RTC_PLAN.md).

## IK-assisted collection

- [x] Audit coverage reports, coordinate/calibration contracts, IK/PID references and native integration points; write the design and example settings.
- [ ] Export machine-readable full joint-space coverage with native gyro/quality handling and an offline route preview.
- [x] Implement native task-pose/Jacobian/IK and explicit-dt PID with Python parity.
- [x] Build the separate PID collector, post-combination joint guard, recorded manual takeover, PID telemetry/protocol attachments and common atomic commit support.
- [x] Extend training import compatibility for PID sidecars/source labels and exclude dry rows.
- [ ] Operator: powered command-direction, stopping-margin and tracking acceptance for the initial PID collector.
- [x] Add a hardware-free randomized pitch/velocity preview: 0.1–3 s commands, independently varied PID gains and 25/50/100 Hz updates, separate 100 Hz excitation clock, measured coverage and replay provenance. All 10 local native tests pass.
- [x] Allow inward PID recovery from outside the working range (fixed gains, 100 Hz, 0.25 output cap, no overlay); print incremental measured working-grid coverage every 30 seconds and at exit.
- [ ] Add single-waypoint assistance, manual takeover and command decomposition to the recorder; validate by replay before powered acceptance.
- [ ] Add coverage-guided continuation over checked, connected transitions.
- [ ] Validate fast-excitation preservation, regional coverage, physical envelope and loaded Jetson timing.

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
- [x] Validate Pico startup calibration, mounting signs, angles and known-pose FK. As of 2026-10-16, IMU readout and FK are expected to be correct (user).
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
- [x] Integrate the fine-tuned runtime with the command interface (`--drive`, PI05_FINETUNE.md); Spark dry end-to-end only.
- [ ] Jetson: native template cache for the fine-tune, `--validate-only`, then a dry `--drive` run against real IMUs/cameras; measure the FP32-residual 50-row latency against the 500 ms budget.
- [ ] Port the Python pipeline's operator control to the producer: A toggles manual/model, D-pad switches task; disarm on source/task changes. No separate hold-to-run deadman (user, 2026-10-17: the Python SmolVLA/X-VLA scheme is sufficient). Today `--drive` sets the deadman flag itself.
- [ ] Set lease/observation limits from measured worst inference latency and required stop time. Do not silently refresh old observation timestamps. (Producer side done: renewals keep the observation time; limits still need Jetson numbers.)
- [ ] Evaluate beta GPU-frame APIs only when profiling shows selected-frame host copies matter; current capture does not enable CUDA zero-copy.
- [ ] Compare native total system memory with the legacy stack under the same model/device load before stating an actual Jetson memory saving.
- [ ] Fine-tuned 30 Hz action contract: benchmark a 30–50-step prediction export and delay-aware overlapping execution. Current LIBERO performance bundle has only 10 action rows.
- [x] Add a native C++ dual-IR episode spool with bounded asynchronous disk writes, full FK/IMU and applied/requested actions, calibration provenance, failure/discard lifecycle and offline LeRobot 0.5.1/v3 export. See RECORDING.md.
- [ ] Validate physical gamepad mapping, live teleoperation/stop behavior and long-session memory/disk stability; video encoding is offline and has not been tested concurrently with inference.
