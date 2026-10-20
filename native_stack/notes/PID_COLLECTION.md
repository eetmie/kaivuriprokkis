# Separate randomized PID collector

20 October 2026. Initial implementation: `kaivuri-pid-collect` owns its own
operator lifecycle, randomized pitch/velocity controller and controller
telemetry. `kaivuri-mlp-collect` remains the manual-plus-excitation application.
Neither application runs through the other or needs a second hardware owner.

Both use the shared gamepad, IMU parser/raw capture, valve calibration/PWM and
committed actuator-recording modules. The PID implementation and its sidecar
writer are separate libraries; the manual executable does not link them.

## Joint ranges and command guarding

The current `jetson_bucket` profile has these relative joint limits:

| Joint | Profile limits | Initial IK target range, with 10° margin |
| --- | --- | --- |
| Boom | −50…30° | −40…20° |
| Arm | 28…144° | 38…134° |
| Bucket | −120…20° | −110…10° |

Goal draws and every accepted IK joint reference stay inside the target range.
IK must satisfy both position and cutting-lip pitch. An unreachable reference
produces zero output and a labelled retry; it is not silently accepted by its
position alone.

The independent joint guard runs **after PID and excitation are combined**.
It starts tapering outward commands five degrees before the target boundary,
and blocks the outward direction at that boundary. Measured velocity toward
the boundary consumes additional headroom through a 0.1 s lookahead. Inward
manual or PID recovery remains available beyond the target margin, including
a measured pose outside the profile limits. Each affected joint can only be
commanded toward the working range. Stale or nonfinite feedback cannot pass
the guard. Suppression is fed back to PID anti-windup.

Starting PID from outside the target range first runs joint-space recovery:
fixed profile gains at 100 Hz, normalized output capped at 0.25 and targets
one degree inside the target range. Random commands and excitation stay off.
Once all measured joints are at least half a degree inside the target range,
randomized IK collection starts from the measured pose. B does not enable an
overlay during recovery; press it after recovery for an excited block. The
recovery segment is labelled in both PID sidecars and has no requested timeout.

Zero commands bypass retained PWM ramp state and dither. A direction reversal
cannot retain an old outward ramp. While a joint is limited, tapering cannot
be enlarged by ramping or dither. Actual pulses are recorded.

These are target and valve-command limits, **not a guarantee that a physical
joint cannot coast past a boundary**. Stopping distance, command signs and
tracking need powered acceptance. Joint limits also do not check the bucket
against the machine, floor or external objects. Collision-checked full-reach
routing is not implemented; the protocol explicitly records
`collision_checked: false`.

## Build and desk/replay launch

From the repository root:

```sh
cmake -S native_stack -B native_stack/build/mlp \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_CAMERA=OFF -DBUILD_INFERENCE=OFF
cmake --build native_stack/build/mlp -j4
ctest --test-dir native_stack/build/mlp --output-on-failure

native_stack/build/mlp/kaivuri-pid-collect \
  --profile configuration_files/profiles/jetson_bucket \
  --settings native_stack/config/pid_collection.yaml \
  --serial /dev/serial/by-id/usb-RPi_IMU_Reader_4250305031363910-if00 \
  --gamepad /dev/input/eventN --out /path/to/pid-desk-test --dry --seed 42
```

`--dry` suppresses all I2C writes. As in the existing manual collector, omitting
it enables live PWM; the pump starts off. No powered motion is part of the
initial software validation. There is no added arming/deadman/lease layer.
Both programs take the same per-user `control.lock`, so only one can own PWM.

Settings are actually loaded from [pid_collection.yaml](../config/pid_collection.yaml).
The older `ik_collection.example.yaml` remains a design reference, not a launch
configuration. `--help` lists the PID collector's independent options.

## Operator lifecycle

| Input | PID collector behavior |
| --- | --- |
| A | Start/save a recording strip. Saving neutralizes commands and turns the pump off. |
| X | Toggle pump; pump-off cancels PID and its overlay. |
| Start | Toggle PID tracking. Starting requires an active recording, pump on, fresh measured pose and centered sticks. Outside the target range, inward recovery runs first. |
| B | Toggle the sine/chirp overlay while randomized PID is active. It has its own clock and stays off during recovery. |
| Sticks | Immediately cancel PID and the overlay; recorded manual repositioning uses the same joint guard. Centering sticks does not restart PID. |
| Ctrl+C / termination | Neutralize, stop the pump and commit the completed capture. |

Only boom, arm and bucket are driven by this application. Slew and tracks stay
neutral. Manual repositioning also requires recording and pump-on in this PID
application. The separate manual collector retains its own original controls.

Lost input, stale feedback or an IMU epoch change cancels PID/overlay. An epoch
change rotates the recording; pump-off and explicit Start synchronize a new
controller start. A strip timeout saves and stops motion. Disk, sidecar or queue
failure neutralizes outputs, leaves the take incomplete and exits; restart the
collector after resolving the recording problem.

For byte replay, `--scripted` replaces the gamepad. Commands on stdin are
`start`, `save`, `pump`, `assist`, `excitation`, `manual BOOM ARM BUCKET` and `quit`.
`--record-on-start` opens a strip; it does not start PID or the pump.

## Randomized commands and clocks

Initial settings retain the existing prototype: 0.1–3 s pitch-plus-X/Z-velocity
segments, 5–20 mm/s, bounded reference lead, and a mix of coverage travel, free
directions, reversals, pitch-only motion and holds. PID update rates are drawn
independently from 25/50/100 Hz, and P/I/D gains from 0.5–1.5 times each profile
gain. Zero base D remains zero. Gain changes preserve the integral contribution.

PWM and command recording stay at 100 Hz; every accepted IMU packet is saved,
normally at 200 Hz. PID scheduling uses the 100 Hz output grid; integration and
derivative math use actual elapsed time. Requested rates, update flags and actual
dt are recorded separately. A gap resets control instead of catching up an old
reference. Normalized command-growth signs are separate from the profile's
electrical PWM directions and are applied to both PID and guard math.

The default optional overlay is a 0.05–12 Hz chirp, 60 s each way, with 10 s rest
and amplitude 0.35. `--excitation sine` selects the existing version-4 sine
generator. Command segment changes never restart the overlay clock. The
preview's automatic 20-second clean/excited schedule is not yet an operator
mode in this first collector; use B to collect clean and excited strips.

Coverage currently starts from a fresh in-memory full joint grid. Goal selection
prefers low measured moving dwell; scheduled goals are not counted as visits.
The runner prints `Joint coverage: X% (visited/total working-range cells...)`
every 30 seconds and at exit. This is a rough **joint-pose coverage** indicator,
not a Cartesian area estimate: with the default 10° bins, the percentage uses
the 720 cells overlapping the three working ranges. Only fresh measured poses
during PID activity count; out-of-profile poses and requested targets do not.
Visits persist across pauses and strips within a run. A small bit map and an
incremental counter keep status reporting constant-time, without rescanning the
grid. Historical map import and checked graph routing remain separate work.

## Recording and training compatibility

Each take contains the standard training trio and two PID sidecars:

```text
drive_log_<take>.csv
imu_raw_<take>.csv
excitation_<take>.json
pid_control_<take>.csv
pid_protocol_<take>.json
session_<take>.json
```

The common writer owns commit ordering. All files start as `.partial`; it
flushes, hashes and syncs every output, then renames the session manifest last.
Its `outputs` retains the base trio; `attachments` names/hashes the two PID
files. A PID take cannot be accepted without both sidecars. The PID telemetry
queue is allocated only by this application, with no per-tick disk formatting
or allocation in its producer.

Base CSVs retain the final `combined_cmd_*` training input and raw sensor
contract. PID takes add controller-source labels, controller validity, assist
and constraint contributions. Their exact decomposition is:

```text
combined_cmd = manual_cmd + assist_cmd + effective_excitation_cmd + constraint_cmd
```

`constraint_cmd` accounts for post-combination joint/lifecycle suppression.
`effective_excitation_cmd` accounts for ordinary amplitude clipping before that
constraint correction. Requested overlay values remain in `sine_cmd_*`.
PID sidecars record measured task pose, task/joint references, actual applied
gains, requested rates, update dt, guard scales/predictions and IK/failure reasons.
The protocol freezes every requested segment, goal, rate and gain; source,
profile and settings contents/hashes are retained in the session manifest.

The actuator importer accepts both collectors in one split, verifies PID
attachments, distinguishes PID/manual/idle rows and excludes dry data. Keep
`--target_v gyro` for unsmoothed targets. It still splits gaps and preserves
session-level train/dev/test separation. Bench/replay directories are test
evidence and must not be passed as collected training data.

## Remaining acceptance

- Validate command directions, dynamic stopping margin and PID tuning with the operator.
- Add checked collision/floor/exclusion geometry and routes before full-reach automation.
- Import compatible historical joint coverage and add automatic clean/excited block scheduling.
- Run long powered capture, inspect actual command/feedback spectra, and assess tracking and dataset quality.
