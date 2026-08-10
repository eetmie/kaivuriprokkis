# IMU tuneup — AHRS settings

Notes for reflashing the Pico IMU firmware. Written 2026-08-07 after tracing
reported spiking/oscillation in the IMU data back through the fusion path.

Everything here is **firmware-side**. The AHRS settings are hardcoded on the
Pico and are not negotiated from the host — `control_config.yaml` only documents
them, it does not set them.

---

## TL;DR

| Setting | File | Now | Change to |
|---|---|---|---|
| `ahrsGain` | `src/imu_reader/settings/settings.c:8` | `4.5f` | **`1.0f`** |
| gyro range | `src/imu_reader/settings/settings.c:7` | `250.0f` | `500.0f` (optional) |
| accel range | `src/imu_reader/ism330dlc/ism330dlc_config.h:6-7` | `±2 g` | `±4 g` (second step) |
| `ahrsAccelRejection` | `src/imu_reader/settings/settings.c:9` | `20.0f` | leave for now |

**Change the gain first, alone, and re-measure.** It is the confirmed cause.
The other two are insurance against a cause I could not measure from the logs.

---

## Is 1.0 a "small" gain?

No — it is not a servo gain, and bigger is not better. It is the **inverse of a
time constant**.

From `Fusion/Fusion/FusionAhrs.c:145,174`:

```c
halfAccelerometerFeedback = Feedback(normalise(accelerometer), halfGravity);
adjustedHalfGyroscope     = halfGyroscope + halfAccelerometerFeedback * rampedGain;
```

`Feedback()` returns approximately `0.5·θ_err` for small tilt errors, and the
quaternion integrates at twice the "half" rate, so the correction applied is

```
dθ_err/dt = −gain · θ_err        →        τ = 1 / gain   seconds
```

| gain | τ | accel-correction corner |
|---|---|---|
| 0.5 (Fusion default) | 2.0 s | 0.08 Hz |
| **1.0** | **1.0 s** | **0.16 Hz** |
| 2.0 | 0.5 s | 0.32 Hz |
| 4.5 (current) | 0.22 s | 0.72 Hz |

So gain = 1.0 means *"correct accumulated tilt error with a one-second time
constant."* The gyro is the accurate sensor on short timescales; the
accelerometer exists only to stop slow gyro-bias drift, and ISM330DLC bias
drifts over tens of seconds to minutes. A 1-second correction is already much
faster than the error it is there to fix.

Raising the gain does not make the estimate track better. It makes the
estimator **trust the accelerometer more**, and on a hydraulic excavator the
accelerometer is measuring cylinder acceleration, valve slam and structural
ringing on top of gravity. At 4.5 the corner sits at 0.72 Hz — which is inside
the 1.0–2.0 Hz excitation band that `simple_drive.py` deliberately injects
(`SineExcitationGenerator`, `CARRIER_FREQ_HZ = 1.0`, `MAX_INSTANT_FREQ_HZ = 2.0`).
The estimator is partly tracking the machine's own shaking.

---

## Evidence from the recorded data

Checked all 11 logs in `data_collection/hydraulic_data/` — 599,159 rows.

### Gyro is NOT clipping — this was a red herring

Fusion resets the estimator whenever any gyro axis exceeds `0.98 × range`
(`FusionAhrs.c:88,119-124`), which at the current 250 dps setting is 245 dps.
That reset is violent — it keeps the quaternion but restarts the gain ramp.

It never fires. Peak measured rates across every log:

| sensor | peak | headroom to 245 dps trip |
|---|---|---|
| boom | 34 dps | 86% |
| arm | 67 dps | 73% |
| bucket | 174 dps | 29% |

Zero samples over the trip, zero samples at hardware full scale. **The gyro
range is not causing the spikes.** Bucket at 174 dps is close enough that a
harder curl could reach it, so 500 dps is cheap insurance, but it is not the fix.

### The fusion is inventing motion — this is the actual problem

`joint_pos_*` is derived from the AHRS quaternion; `imu_g*` is the raw gyro. The
joint angles are relative (`gravity_pitch_delta`, parent→child), so a joint's
true rate cannot exceed the **sum** of the parent and child measured gyro
magnitudes. That is a hard physical bound.

Fraction of samples that violate it:

| joint | violating samples | worst implied rate |
|---|---|---|
| boom | 121,107 / 599,148 (**20.2%**) | 5.5 rad/s = 313 dps |
| arm | 88,089 / 599,148 (**14.7%**) | 8.9 rad/s = 507 dps |
| bucket | 85,269 / 599,148 (**14.2%**) | 13.8 rad/s = **790 dps** |

790 dps of implied rotation on a link whose gyro never read above 174 dps. The
gyroscope is the ground truth for rotation rate, and the fused angle is moving
4–7× faster than any gyro says it moved. Nothing but the accelerometer feedback
term can produce that.

This is not occasional spiking — it is 14–20% of every recording. Any model
trained on `joint_pos_*` from these logs is partly fitting fusion artifacts.

---

## Changes

### 1. Gain — do this one first

`src/imu_reader/settings/settings.c`:

```c
.ahrsGain = 1.0f,     // was 4.5f
```

Start at `1.0f`. If drift is still acceptable after a few minutes of driving,
try `0.5f` (the Fusion default). If the horizon visibly walks away over a
10-minute strip, go back up toward `2.0f`. Do not exceed `2.0f` on this machine.

### 2. Gyro range — optional headroom

`src/imu_reader/settings/settings.c`:

```c
.gyroRangeDps = 500.0f,   // was 250.0f
```

This is used in three consistent places, so one edit is enough: the CTRL2_G
register mask (`ism330dlc.c:85-93`, written at `:111-114`), the raw→dps scaling
(`ism330dlc.c:62,112`), and the Fusion clip threshold (`InitFusion.c:65`).
Costs one bit of resolution; buys 2× headroom before the reset cliff.

### 3. Accelerometer range — second step, if spikes persist

`src/imu_reader/ism330dlc/ism330dlc_config.h`:

```c
#define XL_G_RANGE_MASK 0x08   // ±4 g   (was 0x00 = ±2 g)
#define XL_G_RANGE      4      // was 2
```

**Watch the encoding** — `FS_XL[1:0]` (CTRL1_XL bits 3:2) is *not* in
ascending order on this part family: `00` = ±2 g, `01` = ±16 g, `10` = ±4 g,
`11` = ±8 g. So ±4 g is `0x08` and ±8 g is `0x0C`; `0x04` would give you ±16 g,
not ±4 g. There are no `FS_XL` constants in `ism330dlc_registers.h` to lean on,
so the raw mask is the only thing being written.

Also note the inconsistency: the gyro range is runtime (from the settings
struct) but the accel range is a compile-time constant in this header, so the
two are edited in different places. Worth unifying later.

I could not confirm accel clipping from the logs, because the accelerometer
never leaves the Pico — `write_sensor_output()` (`main.c:161`) streams only the
quaternion and the gyro. ±2 g is the smallest range the part offers and is
marginal for a machine that jolts, so it is a plausible secondary source. Only
change it after the gain change has been measured on its own.

---

## Worth adding while you are in the firmware

Fusion already computes exactly the diagnostics needed here, and none of them
are streamed:

- `FusionAhrsGetFlags()` → `angularRateRecovery`, `accelerationRecovery`,
  `accelerometerIgnored` (`FusionAhrs.c:440-450`)
- `FusionAhrsGetInternalStates()` → `accelerationError` in degrees
  (`FusionAhrs.c:428`)

Streaming even one byte of flags per sensor would make the difference between
"the data looks wrong" and "the accelerometer was rejected for 40 ms here."
Streaming raw accel magnitude would also settle the ±2 g question directly.

---

## Verifying after the flash

Host side already changed: `simple_drive.py` now logs `imu_device_ts_us`, the
Pico's own clock for the IMU frame each row sampled.

1. **Re-run the physical-bound check.** The 14–20% violation rate is the number
   to watch. It should collapse toward zero.

   ```
   python3 /tmp/.../bound.py     # see git history / rewrite from this doc
   ```

2. **Check decimation.** The Pico streams 200 Hz, the loop samples 100 Hz, so
   `diff(imu_device_ts_us)` should be a constant 10000 us. Repeated values mean
   the same frame was read twice; varying deltas mean loop jitter is aliasing
   content above 50 Hz into the recorded band. This is a separate problem from
   the gain and will not be fixed by reflashing.

   The firmware truncates this clock to 32 bits (`utils/output.c:87`), so it
   wraps every ~71.6 min and the host does not unwrap it
   (`usb_serial_reader.py:730`). Take deltas mod 2**32; a single strip never
   spans a wrap, but a multi-strip session will.

3. **Check drift.** Park the machine, leave it stationary for 10 minutes, and
   confirm `joint_pos_*` does not walk. This is what the gain is trading against
   — if it holds still at gain 1.0, the gain is not too low.
