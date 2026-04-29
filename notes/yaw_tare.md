# Slew yaw tare

## Problem

The Pico AHRS has no magnetometer. At boot, whatever direction each IMU is
facing becomes its internal yaw=0 reference. Mounting offsets
(`imu.mounting_offsets_quat`) are static link-frame corrections only — they
don't tare yaw at runtime. `canonical_joint_angles_from_imus` then averages
the four IMU Z-twists every cycle, but that's denoising, not zeroing.

Result: "machine forward = yaw 0" holds only if the Pico boots with the cab
already facing forward. Power-cycle 30° off → 30° silently becomes the zero
for the rest of the session.

## Three options

### 1. Boot procedure (current)
Park forward, then power the Pico / start `excv_gui.py`. No code, fits the
prototype scope. Drift over a session is unbounded but small in practice.

### 2. Software tare (Pi5 prototype)
At a defined moment ("operator presses tare"), snapshot the current fused
slew yaw and subtract it from all subsequent readings. Two equivalent
implementations:

- **Compose into the live mounting offset.** Build
  `q_tare = quat_from_axis_angle(Z, current_slew_yaw)`. Right-multiply each
  per-IMU `mounting_offset` by `q_tare⁻¹` so the next
  `_correct_imu_quaternion` call already produces a tared quat. Downstream
  code (canonical extraction, IK, FK, telemetry) sees yaw=0 with no further
  changes. Single source of truth.
- **Subtract at the canonical-extraction boundary.** Hold a `_yaw_tare_rad`
  on the controller, subtract from `joint_angles[0]` and rotate the
  composed `joint_quats` accordingly. Less invasive but adds a state field
  that other code paths must remember to honour.

Recommend the first form. Hooks naturally near `enter_direct_mode()` /
first `give_pose()` or as a dedicated service command. ~10 lines plus a
test.

Caveat: a software tare doesn't fix gyro drift over a long session — it
only re-aligns the datum at the moment you call it. For the Pi5 prototype
this is fine because sessions are short and you tare at the start of
each IK test.

### 3. Hardware zero-position sensor (real machine)
The production excavator has a slew zero-position sensor (limit switch or
mechanical index) that fires when the cab crosses a known absolute heading.
This gives the system a **real** absolute reference, not just a session-
relative one — equivalent to a magnetometer/GPS for the slew axis only,
and considerably more reliable for an industrial machine.

Integration sketch:

- New hardware reader in `HardwareInterface` that latches the sensor edge
  (rising/falling, configurable) along with the slew quaternion at the
  moment of the edge.
- On edge: compute the offset between the latched IMU yaw and the known
  absolute heading of the sensor (e.g. "fires when cab faces +X world").
  That offset becomes the persistent tare — same compose-into-mounting-
  offset mechanism as option 2.
- Optional: re-tare on every edge crossing, so accumulated gyro drift gets
  corrected each time the operator slews past zero. This turns the IMU
  into a high-rate interpolator between absolute-truth events.
- Optional: persist the tare across reboots (config file / NVRAM) so the
  machine wakes up with a usable yaw before the first sensor crossing.

This is the right long-term path for the real machine. The software tare
(option 2) is the right short-term path for the Pi5 prototype and remains
useful on the real machine as a manual override / fallback if the sensor
is faulty.

## Suggested code shape

Keep the tare as a single piece of state on `ExcavatorController` (or
`HardwareInterface`, if you want it to apply to telemetry that bypasses the
controller too). Public surface:

```python
controller.tare_slew_yaw()                # snapshot now → zero
controller.set_slew_yaw_offset(rad)       # explicit value (sensor path)
controller.get_slew_yaw_offset() -> rad   # for telemetry / persistence
```

Wire the zero-position sensor's edge handler to call
`set_slew_yaw_offset(known_heading_rad - latched_yaw_rad)`.

Tests:

- Tare immediately after boot with a synthetic non-zero start yaw → all
  downstream consumers (`get_joint_angles`, `get_absolute_link_angles`,
  IK target seed) see slew=0.
- Slew the synthetic state by N degrees after taring → readings reflect
  N degrees, not (boot_offset + N).
- Edge-trigger path: simulate two zero-crossings with N° of accumulated
  gyro drift between them → second crossing re-tares back to truth.
