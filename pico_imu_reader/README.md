# pico_imu_reader

Firmware for **Seeed XIAO RP2040** that reads up to four ISM330DHCX IMUs over two I2C buses and streams fused orientation data as binary frames over USB CDC. Used as the IMU front-end for the kaivuriprokkis excavator project.

---

## Hardware assembly

### Board

**Seeed XIAO RP2040** — pin assignments below are fixed and match the physical wiring. Do not change them without rewiring.

### I2C bus wiring

| Bus  | SDA | SCL | IMU addresses |
|------|-----|-----|---------------|
| I2C0 | P28 | P29 | 0x6A, 0x6B    |
| I2C1 | P6  | P7  | 0x6A, 0x6B    |

Up to two IMUs per bus — set each sensor's `SDO` pin low for 0x6A or high for 0x6B. The firmware probes all four slots at runtime; only detected IMUs are streamed.

Power the IMU modules from the **3V3 and GND pins on one side of the board**; I2C0 (P28/P29) and I2C1 (P6/P7) are on the opposite side.

### IMU

**ISM330DHCX** (detected via `WHO_AM_I == 0x6B`). Accelerometer range: ±4 g. Gyro range: ±500 dps. Internal ODR: 416 Hz; AHRS update rate: 416 Hz; output rate to host: 200 Hz.

In-sensor filtering is on: accelerometer LPF2 at ODR/10 (41.6 Hz) and gyro LPF1
at FTYPE 5. Both are enabled through `CTRL1_XL` bit 1 and `CTRL4_C` bit 1
respectively; setting a cutoff without those enables leaves the filter inert.
`ism330dhcx_init()` reads the four filter bits back and refuses to bring a
sensor up if they did not take.

**The gyro's full scale is not `range/32768`.** ST specifies sensitivity
directly: 8.75 mdps/LSB at the ±250 dps setting, 17.5 at ±500. The output code
therefore saturates 14.7% above the range's nominal name — 573 dps at the
±500 dps setting. Conversions read that sensitivity from
`ISM330_GYRO_FULL_SCALES`, and Fusion is given the saturation value, not the
nominal range, so its angular-rate recovery trips at real full scale. The
accelerometer is different: `range/32768` is its exact sensitivity.

---

## Building

Requires the **Raspberry Pi Pico VS Code extension** or a manual CMake toolchain with `arm-none-eabi-gcc`.

```bash
# Clone with submodule
git submodule update --init --recursive

# Configure (VS Code extension does this automatically)
cmake -B build -DPICO_BOARD=seeed_xiao_rp2040

# Build all targets
cmake --build build
```

Three firmware targets are produced:

| Target         | Description                                                  |
|----------------|--------------------------------------------------------------|
| `main`         | Full stack: AHRS fusion, startup calibration, USB CDC stream |
| `simple_stream`| Minimal stream without AHRS — raw IMU values only            |
| `i2c_scan_demo`| Bus scan utility; prints detected devices over USB serial    |

Flash the desired `.uf2` from `build/` by holding BOOT and connecting USB, then copying the file to the mass-storage drive.

---

## Startup calibration (`main` firmware)

On power-up the board settles for 3 seconds, then runs up to **three 10-second
stationary windows**, stopping at the first one it accepts:

- Streams `CAL_WAIT` throughout, with the status LED blinking amber at full
  brightness.
- Collects gyro and accelerometer statistics per IMU and sends a calibration
  report after **every** attempt, accepted or not, so a rejected window and its
  reason reach the host.
- Acceptance gates (hardcoded in `main.c`), all of which must pass on every
  detected IMU:
  - At least 800 valid samples in the window
  - Gyro stddev < 0.5 dps per axis
  - Accel stddev < 0.05 g per axis
  - Mean accel norm 0.75 – 1.25 g
  - Gravity direction moves < 0.5° between the first and last third of the
    window. Standard deviation cannot see a steady slow rotation; this can, for
    any rotation that is not purely about gravity. Yaw about gravity stays
    invisible to any accelerometer-only test.
- If a window is accepted: its gyro mean becomes the bias offset and streaming
  starts. A clean boot reaches this in about 13 s.
- If all three are rejected: **no bias is applied and nothing is streamed.** The
  board sends `ERR_CAL` and blinks red, fast and at full brightness, until it is
  reset.

Leave the machine **completely stationary** from power-on until the LED turns
green.

> Earlier firmware applied the measured bias even when its own gates failed and
> streamed regardless, so a boot with the machine moving produced a silently
> wrong bias on every sensor. Set `STARTUP_CALIBRATION_REQUIRE_ACCEPTED` to 0 in
> `main.c` to restore that behaviour for a bench comparison against old
> recordings — not for production.

---

## Binary protocol (USB CDC)

All frames start with the sync bytes `0xAA 0x55` followed by a frame-type byte.

| Frame              | type   | Description                                                            |
|--------------------|--------|------------------------------------------------------------------------|
| Control/error      | `0x02` | `CAL_WAIT`, `ERR_CAL`, `ERR_IMU`, `ERR_I2C` status codes                |
| Data               | `0x04` | count, timestamp, 10 floats per sensor                                  |
| Calibration report | `0x05` | Per-IMU stats, one per calibration attempt                              |
| Descriptor         | `0x06` | Sent at stream start — sensor count, rate, bus/addr and full scales     |

Each data frame carries 10 floats per sensor: `qw qx qy qz gx gy gz ax ay az`
— the fused quaternion plus the exact bias-corrected gyro [dps] and
accelerometer [g] pair that produced it, which is what makes an offline replay
at different settings possible.

The frame timestamp is the Pico's microsecond clock **truncated to 32 bits**, so
it wraps every 2^32 µs (about 71.6 minutes). A host that differences it must
unwrap.

Calibration reports carry a per-sensor `failure_flags` field. `0x0010` is
`CAL_FAIL_GRAVITY_DRIFT`; older host readers ignore unknown bits, so a report
with only that bit set reads as a bare failure until the host learns the name.

Pitch and roll are derived from the gravity vector, not from `2*atan2(y, w)`.
The host-side reader must match this convention.

---

## Verifying hardware with `i2c_scan_demo`

Flash `i2c_scan_demo.uf2`, then open a serial terminal (115200 baud). The firmware scans both buses and prints detected addresses. Use this to confirm all IMUs are physically present before flashing the main firmware.

---

## Settings

AHRS settings are hardcoded in `src/imu_reader/settings/settings.c`:

```c
.sampleRate          = 200,     // USB output rate [Hz]
.ahrsRateHz          = 200,     // overwritten by main.c with SENSOR_INTERNAL_ODR_HZ
.gyroRangeDps        = 500.0f,
.accelRangeG         = 4.0f,
.gyroSaturationDps   = 0.0f,    // resolved by initialize_sensors()
.ahrsGain            = 0.5f,
.ahrsAccelRejection  = 10.0f,
.ahrsRecoveryPeriodS = 2.0f,
```

Edit and rebuild to change them. There is no runtime configuration interface.

`ahrsGain` is the inverse of a time constant, not a servo gain: 0.5 corrects
accumulated tilt error with a 2-second time constant, putting the accelerometer
correction corner at 0.08 Hz. Raising it makes the estimator trust the
accelerometer more, and on a hydraulic excavator the accelerometer is measuring
cylinder acceleration, valve slam and structural ringing on top of gravity. The
previous 4.5 put that corner at 0.72 Hz, inside the 1–2 Hz band the machine is
deliberately excited in. Use 1.0 if stationary drift matters more than lag; do
not exceed 2.0 on this machine.

`ahrsRecoveryPeriodS` and the range are both consumed in AHRS updates, not USB
frames. `sampleRate` is the transport rate only; anything counted per AHRS
update reads `ahrsRateHz`.

The full scales are requests. `initialize_sensors()` snaps each to the nearest
range the part supports, writes the result back, and reports the resolved values
in the descriptor frame. Both were raised from the earlier ±250 dps / ±2 g:
across 1.3 M recorded frames the bucket peaked at 224 dps corrected and reached
1.9998 g, leaving effectively no margin on either. The quantisation cost is far
below the sensors' own noise in band.
