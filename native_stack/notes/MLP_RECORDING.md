# Native actuator data collection

`kaivuri-mlp-collect` is the MLP actuator recorder. One C++ process reads the
gamepad and Pico, generates excitation, shapes valve pulses and writes the
PCA9685 at 100 Hz. A bounded background writer saves every accepted IMU packet
(normally 200 Hz) and one command/state row per control tick. It does not need
`kaivuri-control`, cameras, Python, Torch or a model during collection.

The VLA recorder remains `kaivuri-record`; its camera spool, controls and offline
LeRobot export are described in [RECORDING.md](RECORDING.md). Both paths use
the same native gamepad reader, valve calibration math and PWM register writer.
MLP excitation and CSV writing have their own modules.

## Build and launch

From the repository root, with CMake, a C++17 compiler, yaml-cpp and OpenSSL
development libraries installed:

```bash
cmake -S native_stack -B native_stack/build/mlp \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_CAMERA=OFF -DBUILD_INFERENCE=OFF
cmake --build native_stack/build/mlp -j4
ctest --test-dir native_stack/build/mlp --output-on-failure
```

The updated Jetson copy is built and tested at
`/home/joel/kaivuriprokkis-mlp-20261020/native_stack/build/kaivuri-mlp-collect`.
It uses the wider frequency defaults below. Its source/profile tree is the
parent repository directory. The commands below use the fresh-build layout
above; substitute `native_stack/build/` when using that existing Jetson tree.
The 2026-10-19 final tree retains the earlier defaults and original 30-minute
neutral hardware-bench executable.

On `pikkumasi2tb`, use the stable Pico path: its current tty is `ttyACM1`, which
can change after reconnecting. Set `--gamepad` to the controller's evdev event
device (inspect `/proc/bus/input/devices`). No gamepad was connected during the
neutral hardware bench.

```bash
native_stack/build/mlp/kaivuri-mlp-collect \
  --profile configuration_files/profiles/jetson \
  --serial /dev/serial/by-id/usb-RPi_IMU_Reader_4250305031363910-if00 \
  --gamepad /dev/input/eventN \
  --out /path/to/actuator-data/2026-10-20 \
  --suffix chirp_lift --excitation chirp --excitation-target lift
```

Replace `eventN` with the actual device. PWM writes are **live by default**.
X directly toggles the pump. There is no arming, deadman, lease or software
pump permission. The physical servo/pump power switch is the cutoff. `--dry`
skips I2C for desk tests; keep those outputs in a separate directory.

The collector takes the same per-user `control.lock` as `kaivuri-control`,
independent of socket paths. Run the two hardware owners under the same user.
Opening another collector/controller fails before it writes PWM. Normal exit,
Ctrl+C, timeout and fatal control errors write calibrated neutral and pump minimum.
`SIGKILL`/power loss cannot run an exit handler; PCA9685 outputs persist.

## Controls

| Input | Behavior |
| --- | --- |
| Sticks | Manual valve commands at all times; the shared 25% deadzone and axis signs apply |
| A | Start a strip; press again to save it |
| B | Toggle excitation; enabling starts a new block through a one-second taper |
| X | Toggle pump between calibrated minimum and run pulses |
| Y | Reload profile/servo configuration between strips, after the previous save completes |
| D-pad up/down | Cycle all, lift, tilt, scoop and the three joint pairs |
| Ctrl+C | Neutralize, save the active strip and exit |

Slew and tracks require `--enable-slew` / `--enable-tracks`. Slew excitation is a
separate, last D-pad target; stroke never excites slew. Tracks have no automatic
excitation. Xbox D-pad buttons and hat axes are both accepted.

Saving a strip disables excitation; manual driving and the pump setting
continue. The default 600-second strip timeout additionally writes neutral and
turns the pump off, then waits for another A press. `--record-seconds` changes
that timeout. A pad disconnect centers manual commands, disables excitation
and marks subsequent rows invalid; it does not change the X pump setting.
Restart the process to reconnect a pad.

## Excitation and timing

`--excitation sine|chirp|stroke` selects the waveform, initially off. Sine draws
independent log-uniform 0.2–10 Hz carriers per joint/block, with FM limited to
12 Hz instantaneous carrier frequency. This retains slow motion while adding
the fast valve corrections missing from the earlier data. Sine amplitude
still draws 0.35–1.0 per joint, with the existing envelope and filtered noise.
Chirp defaults to 0.05–12 Hz, 60 seconds up / 60 down / 10 neutral, amplitude 0.35.
Stroke defaults to an eight-second reference sweep within the profile's joint
limits minus a ten-degree margin, with a 0.5–12 Hz chirp (20 seconds each way,
amplitude 0.4) on alternate strokes. Stale feedback makes the stroke
contribution zero; feedback outside hard limits disables it. Manual sticks
still add to the signal before clipping to [-1, 1].

`--excitation-amplitude`, `--chirp-start-hz`, `--chirp-end-hz`,
`--chirp-seconds`, `--stroke-seconds` and `--stroke-margin-deg` override those
settings. `--excitation-seed N` repeats a strip's parameter sequence; omitted
seeds are regenerated for each strip. `--chirp-*` does not change sine's carrier
range. Use several sine seeds/blocks, or chirp, to cover the full frequency band.
`--chirp-end-hz 0.9` restores the earlier slow chirp explicitly.
Native generator version 4 records the wider default frequency distribution;
version 3 recordings used the earlier defaults. It retains SplitMix64 /
Box–Muller, with separate parameter/noise state. Its seed is not a
claim of NumPy sample identity. The companion JSON stores every block's drawn
parameters, initial noise state/directions, limits and algorithm version.
Noise advances on a fixed 100 Hz grid independent of polling jitter. Frozen
Python fixtures check sine, chirp (including constant frequency) and stroke.

The control loop sleeps on an absolute monotonic 10 ms grid. It drains bounded
serial input, reads the gamepad, computes the signal and calibrated pulses,
writes one PWM burst, then copies fixed-size records into a preallocated SPSC
queue. CSV formatting, hashing, file flushing and fsync run in the writer.
Missed periods create counted gaps instead of catch-up bursts. The queue holds
4096 records by default (`--queue-records`, a power of two in 16..65536).
Overflow or disk failure leaves an incomplete strip and manual control
continues. Press A again after the writer has recovered to start a new strip.

`post_write_ns` is the actual completed-write host time, and `timestamp` is
that value in monotonic seconds. It is not elapsed wall time or the scheduled
tick. The 100 Hz PCA oscillator is not phase-locked to host writes; register
completion does not measure the physical pulse edge (up to one carrier period
later). A dry write measures computation/queue behavior without I2C latency.

## Files and training

Each strip creates `drive_log_*.csv`, `imu_raw_*.csv`, `excitation_*.json` and
`session_*.json`. During recording they have `.partial` suffixes. Successful
save flushes and syncs the data files, then renames the session manifest last:
that manifest commits the complete trio and its SHA-256 hashes. Incomplete
files are retained for inspection and the importer rejects them.

Drive columns retain historical hydraulic channel names: rotate/lift/tilt/scoop.
`combined_cmd_*` is the clipped normalized command passed to valve shaping.
`manual_cmd_*`, `sine_cmd_*`, clipping flags and effective excitation retain
their historical meanings. `pwm_us_0..15` records shaped pulses;
`pwm_count_0..15` records the exact quantized register counts. `pump_on`,
`write_ok`, `input_valid`, `imu_valid`, freshness, scheduled/start/post-write
times, cumulative missed periods and parser counters describe each row.
PWM fields apply to the configured valve/pump channels; unused channel columns
are zero placeholders and those registers are not written.

Positions are radians; drive-log gyro/acceleration columns use rad/s and m/s².
Historical `joint_vel_*` and `vel_age_s` are NaN because this tool has no
finite-difference velocity cache. `joint_gyro_vel_*` contains same-packet
relative gyro rates. Raw CSV gyro/acceleration units are dps and g; raw
quaternions retain mounting offsets and are reordered into base/boom/arm/bucket
roles. Type 1 packets have quaternion + gyro and NaN acceleration; type 4 has
all three. Ranges are recorded when a type 6 descriptor was actually received,
otherwise NaN (the firmware only retries its descriptor during startup).

Raw packets carry raw/unwrapped device clocks, sequence, parser epoch and host
read time. Host time means availability at `read()`, not USB arrival. A parser
epoch change splits the strip and disables excitation. The session freezes
source/header/config bytes and hashes, executable hash, effective options,
sensor mapping, runtime mounting and the fixed training sensor contract.
`collection_id` groups strips from one configuration load, including reset
splits, for experiment separation.

The training importer keeps the existing fixed `SENSOR_CONTRACT` mounting
angles. Runtime mounting is provenance; it is not subtracted again. Thus raw
data joins existing training and model bundle coordinates even when the runtime
profile's arm zero differs. Non-Y mounting / unsupported chains are rejected.

From the `Isaac-hydraulic-actuator` checkout:

```bash
python3 -m training.drive_logs prepare --split train --out runs/new-actuator \
  --logs /path/to/actuator-data/drive_log_EXACT_STRIP.csv
```

Only explicitly passed files are imported. Pump-off, stale, failed-write and
invalid-input/state rows split the usable timeline. The importer checks the
commit marker/hashes and same-packet causal availability, holds only prior
commands, preserves source/excitation labels and refuses to assign one
collection to multiple train/dev/test splits. Gyro velocity labels are the
default; `--target_v smooth21` remains an explicit comparison.

## Reproduce acceptance

```bash
# Known-motion raw serial replay through native recording and the real importer:
python3 -B native_stack/tests/mlp_training_roundtrip.py \
  --binary native_stack/build/mlp/kaivuri-mlp-collect \
  --profile configuration_files/profiles/jetson \
  --training-root /path/to/Isaac-hydraulic-actuator \
  --out native_stack/results/mlp-roundtrip-NEW

# Servo and pump power must be physically OFF for this neutral live-write bench:
python3 -B native_stack/tools/mlp_bench.py \
  --binary native_stack/build/mlp/kaivuri-mlp-collect \
  --profile configuration_files/profiles/jetson \
  --serial /dev/serial/by-id/usb-RPi_IMU_Reader_4250305031363910-if00 \
  --out native_stack/results/mlp-power-off-NEW --seconds 1800 --live
```

The bench keeps manual commands zero, excitation off and pump at minimum. It
writes `bench_summary.json`, a one-second resource trace and collector output.
The data is deliberately unsuitable for training. See
[MLP_VALIDATION.md](MLP_VALIDATION.md) for measured results and remaining physical
checks. PID/IK control has its own executable and sidecars, described in
[separate PID collection](PID_COLLECTION.md); it is not added to this manual
collector's control flow. The offline prototype and full-reach design remain in
[IK-assisted collection](IK_COLLECTION_PLAN.md). Old model driving, circles and
model training remain deferred.
