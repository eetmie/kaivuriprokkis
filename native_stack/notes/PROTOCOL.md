# Interface for the later TRT runtime

The control service is an independent process. Use one persistent Unix `SOCK_SEQPACKET` connection for command ownership and observation/status reads. Closing the owning connection faults the controller. Status-only clients do not own it. All tools and daemon should run under the same UID; granting hardware access through groups/udev is preferable to starting the daemon with a different account.

Default socket: `/tmp/kaivuri-native-<uid>/control.sock`. Version 1 is little-endian. Header size is 52 bytes; maximum packet size is 3124 bytes. Python's exact header format is `<4sHHIQQIfQQ`.

| Offset | Type | Field |
| --- | --- | --- |
| 0 | 4 bytes | `KVR1` |
| 4 | uint16 | Version = 1 |
| 6 | uint16 | Operation: drive=1, disarm=2, emergency stop=3, status=4 |
| 8 | uint32 | Flags: deadman=1, arm=2, pump permission=4 |
| 12 | uint64 | Strictly increasing sequence number per connection |
| 20 | uint64 | Packet send time: local monotonic nanoseconds |
| 28 | uint32 | Number of action rows; 1–128 for drive, 0 otherwise |
| 32 | float32 | Action FPS, positive and at most 500; use the model's training FPS |
| 36 | uint64 | Plan execution start: local monotonic nanoseconds; 0 means receive time |
| 44 | uint64 | Observation time: local monotonic nanoseconds; mandatory for multi-row plans |
| 52 | float32 rows | Six values per row: slew, boom, arm, bucket, trackL, trackR |

Values must be finite and in `[-1,1]`. The packet cannot be over 100 ms old or over 10 ms in the future. The plan cannot extend beyond the configured maximum horizon, currently 5 s. Provided observation time is checked for point commands too. Non-drive operations contain no action rows. Each status request returns a single JSON packet; drive/disarm/estop have no acknowledgement. A full status socket drops the reply instead of blocking the controller.

Status includes safety state/reason, actual current tick pulses, pump authorization, IMU health/age/sequence/device time/arrival time, host monotonic time, angles in radians, FK position in meters and quaternion `[w,x,y,z]`, counters, command sequence, and maximum observed loop lateness. FK fields are zeros/identity if IMUs are unhealthy; always check the health flag. A status response can precede the PWM write for an operation received in the same tick; verify physical stop with instrumentation, not just a JSON response.

### Kinematic observation fields

One complete kinematic snapshot is computed per healthy control tick and reused by all status readers on that tick. It uses one Pico sample for joint extraction, the full FK chain and IMU output. `--control-hz 200` overrides the profile's default 100 Hz loop; it changes neither the PCA9685 carrier frequency nor the safety timeouts. `control_hz` reports the selected rate, and `fk_work_us` measures mounting correction, joint extraction, FK and sensor-array preparation together (excluding JSON, serial reads and PWM writes). Timing CSV includes that phase for every tick, even without a status reader.

Status JSON is bounded to 8192 bytes; clients must receive a whole `SOCK_SEQPACKET` message and reject truncation. All four-row arrays below follow `joint_names = [slew, boom, arm, bucket]`. IMU rows correspond to the base, boom, arm and bucket sensors, respectively; `imu_sensor_indices` gives their physical Pico indices. Quaternions use `[w,x,y,z]`, distances meters and derived angles radians. No Euler angles are required.

| Field | Meaning |
| --- | --- |
| `fk_xyz_m`, `fk_quat_wxyz` | Existing tool-tip position and bucket/tool-frame orientation in the nominal fixed base frame |
| `fk_upper_xyz_m`, `fk_upper_xz_m`, `fk_upper_quat_wxyz` | Tool-tip pose in the rotating upper-carriage frame, and its explicit `[X,Z]` view |
| `slew_yaw_rad` | Estimated slew joint angle; IMU-derived and subject to heading drift |
| `bucket_pitch_rad` | Signed bucket/tool +X-axis pitch about upper +Y, wrapped to `[-pi,pi)`; positive +Y rotation tips +X downward toward -Z |
| `fk_joint_xyz_m`, `fk_joint_xyz_upper_m` | Each joint pivot position before that joint's rotation, in base and upper frames |
| `fk_joint_quat_wxyz` | Each joint's local axis-angle rotation relative to its pre-rotation joint frame; excludes fixed translations |
| `fk_link_quat_wxyz`, `fk_link_quat_upper_wxyz` | Orientation after each joint's rotation, in base and upper frames; final row equals the corresponding tool quaternion |
| `imu_quat_wxyz` | Firmware AHRS sensor quaternions before host mounting correction, reordered into named rows |
| `imu_corrected_quat_wxyz` | Those AHRS quaternions after right-multiplying the configured inverse mounting correction and normalizing |
| `imu_gyro_dps`, `imu_accel_g` | Corresponding firmware measurements in sensor axes, preserving degrees/second and g; synthetic mode reports zeros |
| `fk_valid` | True only with healthy IMUs; every new pose/sensor field above is `null` when false (legacy FK fields retain zeros/identity) |

The base origin is the geometry model's ground reference, with the slew pivot at the first joint's `parent_to_joint_xyz` (currently `[0,0,0.07]` m). The upper origin is that pivot, with axes following the carriage: +X forward in the boom plane, +Y left, +Z along the slew axis. Upper FK is calculated directly from the arm chain without the estimated slew angle. With the current coplanar Y-axis joints, upper Y is zero, and upper XZ, bucket pitch, all upper pivot positions and upper link orientations remain unchanged during pure slew. The base-frame positions rotate along circular arcs; changing arm extension changes their radius.

The tool quaternion describes the configured bucket frame, not the direction of the pivot-to-tip position vector. A cutting-edge orientation needs its own calibrated fixed angular offset. This four-joint FK constrains the arm to its model plane; it does not measure chassis roll/pitch, terrain-relative height, machine travel or independent tool roll. Sensor AHRS quaternions retain their sensor/filter reference frames and individual heading drift; mounting correction alone does not establish a common world heading. Save the active geometry, IMU mapping, mounting offsets and code revision with datasets so all derived quantities can be reproduced.

For arming, send one centered row with deadman and arm flags repeatedly for at least 250 ms while IMUs are healthy. Pump permission may be included, but the pump stays at minimum until armed. To recover from a fault, explicitly disarm, fix its cause, and perform the centered arming hold again. Task/manual-policy switching should disarm first.

## Integration skeleton

```python
import time
from client import Client, DRIVE, DISARM, DEADMAN, PUMP

with Client() as control:
    control.send(DISARM)
    control.arm(pump=False)
    state = control.status()
    assert state["imu_healthy"]
    # Runtime supplies already normalized valve controls, not EE pose or joint targets.
    controls = [(0, 0, 0, 0, 0, 0), (0, 0.05, 0, 0, 0, 0)]
    oldest_observation_ns = state["imu_host_monotonic_ns"]
    control.send(DRIVE, controls, flags=DEADMAN, fps=30,
                 t0_ns=time.monotonic_ns(), observation_ns=oldest_observation_ns)
    control.send(DISARM)
```

Add `native_stack/tools` to the runtime import path, or implement the same binary contract in C++. A four-action model must append two zero track values. Translate model action names using the bundle's physical boundary; do not assume every six-column output means valves. Pump permission must come from the operator/supervisor, not a learned output.

**Inference scheduling matters:** every accepted drive packet grants only a 500 ms lease. A long chunk does not extend that lease automatically. If inference can exceed this, run a responsive supervisory publisher that renews the existing plan before expiry while preserving its original `t0_ns` and `observation_ns`. Renewal must use increasing packet sequence/send times. It must not reset the observation age or restart the chunk to conceal stalled inference. Even a renewed plan stops on exhaustion or at 1.5 s observation age. Decide timing changes from tomorrow's measured latency and required stopping behavior. There is no unlimited keepalive operation.

Sample state and camera freshness at the runtime's actual observation rate. Set observation time to the oldest constituent monotonic arrival, not inference-completion time. Camera device timestamps remain additional evidence for exposure timing; computing an exposure-aligned host observation timestamp requires the explicit clock mapper described in the architecture notes.

## Camera consumer

`libkaivuri_frames.so` exposes `kaivuri_frame_open`, `kaivuri_frame_read`, and `kaivuri_frame_close`. Metadata ABI v1 is the 96-byte `FrameMetadata` struct in `include/kaivuri/frame_store.hpp`. A successful read copies one coherent latest frame. Return 0 means no frame/busy; -1 means invalid arguments or insufficient destination capacity. A null pixel pointer reads metadata only. The caller owns the pixel buffer.

`tools/frame_reader.py` wraps this with ctypes and checks arrival age (default 200 ms). Its returned memoryview is valid until that reader's next read. Convert/resize gray to the model's expected input only when selected. Reopen the reader after producer restart. Files remain after shutdown, but an old timestamp makes the frame unusable.

Metadata validity bits: bit 0 sensor timestamp, bit 1 readout/frame timestamp, bit 2 frame counter, bit 3 exposure, bit 4 gain. Timestamp domain: 0 hardware clock, 1 OS system time, 2 global time. Host/Pico/camera clocks must not be mixed without calibration.
