# IMU chain and FK notes

Measured 2026-10-20 on pikkumasi2tb. Sources: the two-take sand-depth session (102 s of 100 Hz
samples; one flat take, one digging take), the cabin RealSense depth, and
`tools/imu_audit.py` / `tools/sand_height.py`. Values flagged *low data* need the free-air and
side-camera sessions below before anything is calibrated from them.

## How the chain is built

- The `base` IMU sits on the **cabin** (upper carriage), not the chassis. The slew angle is its
  own twist about the slew axis (`relative_axis_twist`, no parent), so the slew is gyro-only and
  drifts slowly.
- Boom, arm and bucket angles are differences of gravity pitch between consecutive IMUs
  (`gravity_pitch_delta`). The arm chain is therefore relative to the cabin and correct in the
  upper frame whatever the cabin does.
- Up to 2026-10-20 the base frame was the slew rotation only. Any cabin tilt (machine tipping,
  slew-bearing rock, ringing) went straight into base-frame positions.

## Level frame (controller, `robot.level_frame`)

- `cabin_tilt: true` (default): the cabin IMU's orientation splits into twist (slew) and swing
  (tilt). The swing turns the slewed chain about the base origin. `fk_xyz_m`, `fk_joint_xyz_m`,
  `fk_link_quat_wxyz` and `fk_quat_wxyz` are then gravity level. The upper-frame fields are
  unchanged.
- New status fields: `cabin_roll_rad` and `cabin_pitch_rad` (gravity angles in carriage axes),
  `cabin_tilt_quat_wxyz`, and `link_roll_rad`.
- `link_roll: false` (default): when true, each link also turns about its own +X by its measured
  roll relative to the parent IMU. Gravity cannot see that roll within about 11° of vertical, so
  it reads 0 there. A mounting roll looks exactly like play, so set the mounting offsets first.
- Heading still follows the gyro slew. Track-edge lift and machine travel are not modelled; that
  was deliberately left out for this machine.

### Mounting offsets after new IMU mounts

1. Park with every joint still and the machine on a flat surface.
2. Run `imu_audit.py` on a session with still periods. *Relative roll* is each link's roll offset
   against the cabin. *Hinge axis* is each IMU's turn against its hinge (needs single-joint
   motion).
3. Put those into `imu.mounting_offsets_quat`.
4. Then `link_roll: true` can be tried. Only roll that changes with load or speed is play.

## Findings

| | Result | Confidence |
| --- | --- | --- |
| Hinge axis vs IMU axis | cabin 0.20° (vs slew axis), boom 0.39°, **arm 1.12°** (95% 1.03–1.26), bucket 0.36° | boom and bucket *low data* |
| Gyro rate off the hinge axis | 1–5% median, ≤10% p90 | no sign of large hinge play at these speeds |
| Link roll vs cabin | boom −0.17°, arm −0.04°, bucket −0.45°, scatter ±0.5° | *low data* (5 s still) |
| Angle rate vs gyro (gain / lag) | slew 0.999 / 0 ms, arm 0.996 / −10 ms, bucket 0.994 / 0 ms, boom 0.96 / +10 ms | boom *low data* |
| Cabin static attitude | roll −1.35° (IMU + floor, unknown split); pitch −2.2…+0.2° over the takes | |
| Cabin ringing | pitch about 7.5 Hz, roll about 9–14 Hz; 0.7 / 0.3 dps rms, p99 2.8 / 1.5 dps | consistent across takes |
| Tipping under load | bucket on the container bottom tipped the cabin about 2° and lifted the cabin camera about 16 mm for about 3 s; recovered within about 1 s | one event |
| Rock after slews | about ±0.5° that settles over about 6 s (the operator sees the bounce) | |
| Slew gyro | gain 1.00; drift under 1° per take against the sandbox seen by the cabin camera | one take |
| Arm FK | wrist-camera heights 5–10 mm off at some poses: zero offsets or link lengths, not play | side camera will tell |

Full chain with yaw/roll per joint: not worth it yet. Off-axis motion is small; link roll is
already an option; per-joint yaw cannot be seen by gravity and drifts in the AHRS, so it needs
the side camera or a magnetometer. The bigger error terms were cabin tilt (now in the FK) and
joint zero/link-length calibration (side camera).

## For the hydraulics (MLP actuator) model

- **Structural modes.** The cabin rings at about 7.5 Hz (pitch) and 9–14 Hz (roll). Fast arm
  moves excite them, and they ring on for a while. 100 Hz IMU and 200 Hz control resolve them
  (Nyquist 50 Hz). Do not low-pass state inputs below about 20 Hz if the model should learn the
  dynamics. Smooth targets, not measurements.
- **Inputs.** Feed the cabin attitude rates (`imu_gyro_dps` row 0, and `cabin_roll_rad` /
  `cabin_pitch_rad`) as inputs or disturbance states. Otherwise the model attributes the
  carriage's reaction motion to the valves. Arm accelerations push the cabin, so the joints are
  coupled through the slew bearing.
- **Rates.** Use the gyro hinge rate (child minus parent about the hinge axis) rather than
  differentiated angles. The angle path tracks it with 0 ± 10 ms lag and gain within 0.5%, and
  the gyro is cleaner.
- **Frames.** Joint angles are relative and unaffected by the level frame. Use level-frame
  positions only for ground-referenced targets (dig depth).
- **Load.** Digging into the bottom tips the machine about 2°. Under contact the joint
  pressure/force picture changes, so mark or exclude contact phases in free-air actuator
  datasets.

## Data still needed

- Free-air back-and-forth of one joint at a time: 3–4 cycles, slow and fast, with still pauses.
  Plus a slew back-and-forth. Gives the hinge axes (boom especially), gain/lag and ringing
  decay.
- Side camera (the cabin D435i on a tripod, recorded as cam2) with AprilTag 36h11 tags, 50–60 mm:
  two per link near the pivots, plus the cabin, the lower carriage and the box. Gives joint zero
  offsets, link lengths, deflection under load, the real cabin rock, and IMU latency.
- Re-run `tools/imu_audit.py SESSION ...` on everything together. It flags what is still
  *low data*.
