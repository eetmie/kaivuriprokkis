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

- [ ] Build on Jetson and record versions/permissions/profile.
- [ ] Validate Pico startup calibration, mounting signs, angles and known-pose FK.
- [ ] Measure neutral/pump/active pulse widths and compare with legacy output.
- [ ] Confirm physical stop behavior for deadman/lease/IMU/producer failures, with an independent stop available.
- [ ] Verify IR-only streaming and sensor/readout timestamps through RSUSB on the installed firmware.
- [ ] Verify IR1 versus IR2, exposure/gain, frame gaps, and stale-frame handling.
- [ ] Add separate camera; measure USB/power behavior and clocks; test hardware sync only if needed.
- [ ] Measure RSS/PSS/swap/jitter with real devices and later with model memory pressure.
- [ ] Longer camera run through timestamp rollover; longer native memory-stability run.

## After the bench session

- [ ] Independent MCU/hardware watchdog for retained PCA9685 outputs and physical pump disable. This is the highest-priority remaining stability/safety gap.
- [ ] Optional deployment/supervision files only after hardware paths/timings are established; none are installed now.
- [ ] Add a clock mapper for camera exposure timestamps, host monotonic time, and Pico device time; characterize latency/drift before fusion.
- [ ] Integrate the user-built TRT runtime through the existing binary/C interfaces; preserve actual training FPS, normalization and observation timestamps.
- [ ] Rebuild operator deadman/manual/task-switch supervision in that producer, disarming on source/task changes.
- [ ] Set lease/observation limits from measured worst inference latency and required stop time. Do not silently refresh old observation timestamps.
- [ ] Evaluate beta GPU-frame APIs only when profiling shows selected-frame host copies matter; current capture does not enable CUDA zero-copy.
- [ ] Compare native total system memory with the legacy stack under the same model/device load before stating an actual Jetson memory saving.
