# Native MLP logging validation — 2026-10-19

Scope: the standalone MLP recorder, shared hardware/gamepad/parser modules,
waveforms, asynchronous CSV writer and actuator training importer. IK/PID,
old-model driving, circles and model training are deferred. Commands and data
semantics are in [MLP_RECORDING.md](MLP_RECORDING.md).

## Software and data checks

| Check | Result |
| --- | --- |
| Full native build, including RealSense/VLA recorder | 8/8 CTest checks pass |
| Jetson Release build without camera/inference dependencies | 7/7 CTest checks pass |
| Address/undefined-behavior sanitizers | Parser, waveform/overflow, replay and lifecycle checks pass (4/4) |
| Existing PWM/count/FK reference parity, three current profiles | 12,192 cases pass; maximum pulse error 1.16e-11 µs, FK component error 1.12e-7 |
| Actuator importer | 10 pytest checks pass |
| Frozen Python waveform fixtures | Sine, logarithmic/constant-frequency chirp and stroke match within 1e-10 |
| Known-motion native serial replay → real training preparation | 579 samples at 100 Hz; max q error 1.01e-7 rad, v error 4.00e-9 rad/s, u error 1.49e-8 |

Replay covers fragmented and multi-packet reads, 32-bit device timestamp wrap,
descriptor/reset splitting, invalid quaternion/checksum, deliberate packet
gaps, and both type 1 (quaternion/gyro) and type 4 (+acceleration) packets.
Every accepted packet in the captured span appears exactly once in raw CSV.
Rows reference packets available before their PWM write; missed ticks are
represented explicitly rather than replaced by a nominal clock.

Lifecycle checks use the real shared `Pwm` implementation with a test-only
I2C replacement that saves register bytes. Repeated start/save, SIGINT,
timeout, pad disconnect and full-disk behavior pass, with exit register bursts
equal to startup neutral/pump minimum. Overflow leaves partial files and an
incomplete manifest. A second hardware owner is rejected. Xbox hat/button
D-pad handling and malformed CLI values are covered.

The importer checks the final commit marker and trio hashes; rejects pump-off,
stale, failed-write, invalid-input/state and causally unavailable IMU rows;
preserves waveform/source metadata; and refuses collection leakage between
train/dev/test splits. The fixed historical training calibration is retained,
with runtime calibration recorded separately.

Final local evidence: `native_stack/results/mlp-20261019/roundtrip-final/`,
`roundtrip-final.txt`, and `sanitizers.txt`. Results are ignored by Git.

One repeated VLA synthetic-camera check encountered a 65.8 ms publication gap
and correctly rejected that take. The unchanged rate requirement passed on
the next full run; no test tolerance was relaxed.

Separate 60-second dry replays on Spark exercised the actual collector with
excitation enabled, 200 Hz byte input, a lift target and background CSV output:

| Waveform | Rows | Work p50 / p99 / max (ms) | Missed periods |
| --- | ---: | --- | ---: |
| Sine | 6,000 | 0.0238 / 0.1903 / 0.8663 | 0 |
| Chirp | 6,000 | 0.0190 / 0.0422 / 0.6796 | 0 |
| Stroke | 6,000 | 0.0207 / 0.1165 / 0.6397 | 0 |

These exclude I2C and run on a different host from the hardware test; they are
not a direct latency comparison with Jetson. Evidence:
`native_stack/results/mlp-20261019/waveform-timing/summary.json`.

## Jetson timing and resources

The final build is isolated at
`/home/joel/kaivuriprokkis-mlp-20261019-final`; existing deployments are untouched.
Release C++17, real Pico on
`/dev/serial/by-id/usb-RPi_IMU_Reader_4250305031363910-if00`, live PCA9685 on
I2C 7 / address 0x40, neutral valves and pump minimum throughout. The user
confirmed servo and pump power were physically off. No camera or inference
process ran during this MLP-only test.

Jetson toolchain: GCC 13.3.0, CMake 3.31.9, yaml-cpp 0.8.0 and OpenSSL 3.0.13.
All 25 source/header/profile hashes in the deployed session match the local
workspace. Tested executable SHA-256:
`dce356a99c261ff92113871b606c4be7cf40d09b692520d839e525ea2faf3397`.

The final 30-minute run completed with 180,000 drive rows and 360,018 accepted
raw packets. There were **zero missed periods and zero dropped records**.
Queue high-water was 15 of 4096 records. No raw sequence gaps or device packet
intervals above 7.5 ms were detected within either epoch; maximum device
interval was 7.243 ms. There were zero checksum errors and no additional invalid
frames or epoch changes after the startup fragment described below.

| Measurement (ms) | p50 | p99 | p99.9 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Tick wake-up lateness | 0.0546 | 0.0601 | 0.0665 | 0.3030 |
| Tick work through completed PWM write | 1.0031 | 1.0263 | 1.0503 | 2.1322 |
| Interval between completed PWM writes | 9.9999 | 10.0242 | 10.0421 | 10.9907 |

Warm RSS remained exactly **12,820 KiB (12.52 MiB)** at every one-second sample,
with zero growth. CPU averaged **1.45% of one core**. Capture ran for 1800
seconds; shutdown/hash/fsync and offline summary brought total bench elapsed
time to 1811.29 seconds.

After shutdown, an independent register read under the shared hardware lock
matched every configured neutral/pump-minimum count:
`[614, 614, 670, 677, 644, 657, 655, 614, 409]` on channels 0–8. MODE1 was
0x20 (awake, auto-increment), MODE2 0x04 and prescale 60. No collector remains
running.

Full artifacts remain under `bench/real-imu-live-30min/` in the isolated Jetson
tree. Summaries, the resource trace, collector output, both provenance
manifests and `pca9685_neutral_readback.json` are copied locally to
`native_stack/results/mlp-20261019/final-bench/`.

On USB reopen, six old queued packets were followed by a 45.97-second device
clock gap. The parser rejected the discontinuity frame and split the one-row
startup fragment from steady recording. Its cumulative counters therefore
include one startup invalid frame / epoch change. The pump stayed off and the
fragment cannot form a training chunk; steady counters are assessed separately.

A preliminary 1,220.81-second capture, stopped cleanly to deploy the final
schema/metadata fixes, had zero missed periods, dropped records, checksum or
invalid frames, or detected raw gaps. Wake lateness p99/max was 0.059/0.615 ms;
work p99/max was 1.028/1.779 ms. Warm RSS stayed exactly 12,692 KiB; CPU averaged
1.42% of one core. That result is preliminary, not the final 30-minute
acceptance measurement. Original evidence remains in
`/home/joel/kaivuriprokkis-mlp-20261019-1315/bench/real-imu-live-30min/`.

## Remaining physical checks

The bench uses neutral commands with hydraulic/servo power off. Physical
gamepad teleoperation and a short powered collection remain for the operator;
no pad was attached during this session. PWM pulse-edge timing and hydraulic
motion were not measured. Host write-completion timing does not establish the
PCA carrier phase or physical actuator response. Keep bench/replay outputs
outside training datasets.

## Wider frequency defaults — local and Jetson validation, 2026-10-20

Native generator version 4 and Python sine/chirp version 3 broaden the default
collection frequencies: chirp is 0.05–12 Hz and sine carriers are log-uniform
0.2–10 Hz with a 12 Hz instantaneous carrier limit. Amplitudes, stroke behavior,
manual controls and PWM shaping are unchanged.

The local Release build passes all 7 CTest checks. Python excitation, input and
profile checks pass 89/89, including actual slow/fast signal spectra. Frozen
cross-language fixtures include an 8 Hz sine carrier and a 0.05–12 Hz chirp,
alongside the earlier slow fixtures. The native recorder to real actuator
importer replay retains 579 samples at 100 Hz, with maximum errors of
1.01e-7 rad in position, 4.00e-9 rad/s in velocity and 1.48e-8 in valve command.

The updated source is deployed and built in
`/home/joel/kaivuriprokkis-mlp-20261020`. All 7 CTest checks pass on Jetson too.
Executable SHA-256:
`4e44b045e749be449361874f7da7b863257bafece0f741d67787621e31adffe0`.
A fresh 15-second real-Pico dry run captured 1500 control rows and 2998 raw
packets, with zero missed ticks, dropped records, checksum errors, invalid
frames or raw sequence gaps. Wake lateness p99/max was 0.0611/0.1133 ms; dry
tick work p99/max was 0.0226/0.0530 ms. These exclude I2C and are not a replacement
for the earlier live-neutral hardware timing measurement.

These checks use dry PWM or intercepted I2C. The changed defaults have not been
run on powered hydraulics. The isolated 2026-10-19 final tree retains its earlier
defaults and original 30-minute hardware-bench executable. Local artifacts are
under `native_stack/results/excitation-defaults-20261020/`; the fresh Jetson
capture is under `bench/current-pico-dry/` in the new tree. No collector is left
running. IK-assisted collection is a separate design, not enabled by this update.
