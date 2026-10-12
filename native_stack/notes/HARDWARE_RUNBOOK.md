# Hardware session: 2026-10-12

The 2026-10-12 Jetson bench now covers native builds, five OpenPI pi0.5 policy fixtures, dual IR1/IR2 capture, a five-minute inference/real-IMU load run, and unpowered PCA9685 register-level control checks. See the README and `results/` for evidence. Physical waveform, calibrated known-pose FK, powered hydraulic behavior and watchdog acceptance remain pending. Start with ordinary scheduling and the current conservative safety timings.

## 1. Establish one hardware owner

Stop the existing actuator/recording/inference launch before starting native control. The native lock only coordinates native processes under the same UID. Keep pump power/hydraulics disabled for initial checks. Identify the serial device with `/dev/serial/by-id`, confirm I2C bus 7/address 0x40 match this machine, and select the intended calibration profile. The current native v1 hardware defaults are bus 7 and address 0x40.

Check device/group permissions. Run the client and controller as the same user. Do not start a second controller to issue a stop: use the first controller's socket.

## 2. Real IMUs, dry PWM

From `native_stack/`:

```bash
./build/kaivuri-control \
  --profile ../configuration_files/profiles/jetson \
  --serial /dev/serial/by-id/YOUR_PICO_DEVICE
```

In another terminal:

```bash
python3 -B tools/client.py status
python3 -B tools/client.py exercise --values 0,0,0,0,0,0 --seconds 2
```

Use `jetson_bucket` or `jetson_bucket_dz0` instead if that is the intended profile. Wait for Pico calibration; firmware CAL_WAIT/rejected calibration prevents arming. Check `imu_healthy`, `firmware_blocked`, increasing device time/sequence, joint sign and mounting-offset behavior, and plausible FK against a known physical pose. Compare with existing logs or a separate reference run; do not run both serial/control owners simultaneously.

Unplug the Pico during a dry exercise: expect fault and neutral preview within roughly 100 ms plus a tick. Stop feeding while keeping the producer connection open: expect pump-off preview at 500 ms and valve decay completed by about 750 ms. Closing an armed producer faults immediately. Reconnect/restart and explicitly rearm; there is no automatic serial reconnection in v1.

## 3. Physical PWM, hydraulics still disabled

Stop the dry controller first. Connect an oscilloscope/logic analyzer to check the actual waveform.

```bash
./build/kaivuri-control \
  --profile ../configuration_files/profiles/jetson \
  --serial /dev/serial/by-id/YOUR_PICO_DEVICE \
  --live-pwm
```

The controller boots disarmed and writes neutral. Expected logical targets for the standard Jetson profile:

| Channel | Function | Neutral target (us) |
| --- | --- | --- |
| 0 / 1 | trackL / trackR | 1500 / 1500 |
| 2 | slew | 1637.5 |
| 3 | boom | 1655 |
| 4 | arm | 1575 |
| 5 | bucket | 1605 |
| 6 / 7 | extras | 1600 / 1500 |
| 8 | pump stopped | 1000 |

Confirm nominal 100 Hz and compare measured pulse widths with the existing writer. Prescale/oscillator quantization means measured widths are not exactly the logical targets. Confirm pump minimum actually stops the installed driver and center pulses do not move the valves. Check SIGTERM cleanup, client disconnection, stale IMUs, and explicit emergency stop:

```bash
python3 -B tools/client.py estop
python3 -B tools/client.py disarm
```

If the daemon is stopped, a one-shot neutral write is available (requires I2C access, no serial needed):

```bash
./build/kaivuri-control --live-pwm --neutralize
```

The PCA9685 retains its last waveform if the daemon is killed. Do not use a powered hydraulic test of SIGKILL/OOM to validate this software: it cannot neutralize after death. Test that failure only with hydraulics disabled or an independent verified pump cutoff/watchdog.

## 4. Bounded valve exercise

After the waveform/neutral checks and physical stop mechanism are verified, exercise one joint at a time using the intended profile and explicit pump permission when appropriate:

```bash
python3 -B tools/client.py exercise \
  --values 0,0.05,0,0,0,0 --seconds 0.5 --pump
```

The small normalized value still crosses the calibrated deadband; it does not necessarily mean a tiny PWM change. Check direction before increasing range. Dither is retained on active valves. Slew and tracks remain blocked unless their daemon options were explicitly supplied. The exercise tool disarms afterward and caps commands at 0.15 by default.

## 5. One real IR camera and timestamps

List exact serials and firmware first:

```bash
./build/kaivuri-camera --list
mkdir -p results
./build/kaivuri-camera --camera SERIAL_A:1 \
  --duration-s 30 --metadata-csv results/ir1-timestamps.csv
```

Read/save one image while capture runs:

```bash
python3 -B tools/frame_reader.py \
  "/tmp/kaivuri-native-$(id -u)/cameras/SERIAL_A-ir1.frame" \
  --seconds 5 --save-pgm results/ir1.pgm
```

Repeat with `SERIAL_A:2` to compare IR2. Check timestamp validity bits, counter continuity, SDK domain, exposure, readable image, and motion blur. Capture defaults to global time; `--hardware-clock` preserves the SDK hardware-domain timestamp instead. Raw device sensor/readout metadata is retained in both modes. If required metadata fails, record the error; `--allow-missing-metadata` is only for diagnosis and does not satisfy the timestamp requirement.

For fixed exposure, test `--exposure-us 1000 --gain 16` if the device accepts that range and lighting permits; there is no automatic assumption that those values suit this scene. The emitter is disabled, so external IR illumination may be needed in low light.

Unplug the camera and verify the consumer rejects the retained image after 200 ms; the service reports missing frames and exits after its stale-stream timeout. On restart, reopen the consumer. An inference producer must disarm/stop renewing plans when its observation source is unusable.

## 6. Two devices and loaded memory

```bash
./build/kaivuri-camera --camera SERIAL_A:1 --camera SERIAL_B:2 \
  --duration-s 60 --metadata-csv results/two-camera-timestamps.csv
```

Start standalone. Use independent USB ports with adequate power and record frame gaps. Optional hardware synchronization is `SERIAL_A:1:1` (master) plus `SERIAL_B:2:2` (slave), with the correct cable and identical rates. IR-only synchronization still needs a timing-event test; do not infer alignment from equal frame counters or subtract raw device epochs.

Measure the actual service PIDs in another terminal:

```bash
python3 -B tools/measure.py CONTROL_PID CAMERA_PID --seconds 60 \
  --output results/jetson-native-services.json
```

Record JetPack/kernel, firmware, profile, RSUSB metadata support, RSS/PSS/swap, CPU, `worst_lateness_s`, and whole-system memory (`tegrastats`/`free -h`). Repeat with the future model process loaded. Only then consider CPU pinning/RT priority/memory locking and changing the 500 ms producer lease. Watch a real camera for more than 75 minutes to cover a hardware timestamp rollover; Spark synthetic tests do not establish SDK rollover behavior on the camera.

## Session result sheet

| Check | Result / measurement |
| --- | --- |
| Profile, JetPack/kernel, firmware, serial IDs | |
| IMU calibration/angles/FK known pose | |
| Measured neutral/pump waveform | |
| Command direction/deadband/dither | |
| Producer / IMU / emergency-stop latency | |
| IR1 / IR2 image and timestamp validity | |
| Separate-camera drift / sync timing event | |
| Memory/jitter, idle and model-loaded | |
| Remaining hardware watchdog action | |
