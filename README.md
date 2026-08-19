# kaivuriprokkis

Hardware/IRL control stack for the MASI excavator: a Raspberry Pi or Jetson
Orin Nano drives the hydraulic valves through a PCA9685, reads joint angles
from per-link IMUs, and runs teleoperation, data collection, and on-device VLA
inference against the real machine.

Start with `simple_drive.py` (open-loop gamepad driving + hydraulic data
collection). Closed-loop and compensated driving live in `control_prototype/`,
and the LeRobot/SmolVLA workflow in `lerobot_vla/`.

```bash
.venv/bin/python simple_drive.py --robot jetson
```

## Control Architecture

Three layers, and the boundary between them matters:

| Layer | What it does |
|---|---|
| `modules/pwm/` | PCA9685 valve output: deadband, gamma, ramp, dither, pulse clamps, watchdogs |
| `modules/excavator_controller.py` | Background control thread at `control_hz` (100 Hz): IMU→joint angles, IK/PID, and valve output |
| producers | gamepad, UDP client, VLA policy — whatever rate they manage |

**Producers must not write the PWM layer directly.** `PWMController` has no
clock of its own: dither phase, ramp integration, the stale-command watchdog
and the input-rate gate all advance only when `update_named()` is called, so a
slow or bursty caller aliases the dither and silently trips the rate gate.
Hand setpoints to the control thread instead:

```python
controller.enter_direct_command_mode()
controller.give_direct_commands({"boom": 0.4})        # single setpoint
controller.give_direct_chunk(chunk, fps=30.0)         # policy action chunk
```

The thread resamples onto its fixed rate and writes every tick; setpoint age is
tracked separately and decays the command to zero if the producer stalls. See
`modules/setpoint_schedule.py` and `modules/pwm/README.md`.

The controller has four mutually exclusive output modes: IK (`give_pose`),
velocity (`enter_velocity_command_mode`), direct
(`enter_direct_command_mode`), and suspended (`suspend_ik_output`, which hands
the bus to a caller-owned `DirectController`).

## Platform Setup And Privileges

`setup.sh` is Raspberry Pi-specific. It edits Raspberry Pi boot config, creates
virtual I2C buses for the ADC/OLED wiring, and installs an OLED service. Do not
use it on Jetson.

The Raspberry Pi main I2C bus runs well at 1 MHz in this project.

Runtime profile defaults live in `configuration_files/profiles/<name>/profile.yaml`.
The profile selects a `board` (`rpi` or `jetson`) for compute-platform defaults
such as I2C bus numbers, then points at profile-local `servo_config.yaml` and
`control_config.yaml` files for robot-specific valve, geometry, IMU, IK, PID,
and controller settings. `--robot auto` only auto-detects the board profile;
robot-specific profiles must be selected explicitly.

USB serial should not require running the robot process with `sudo`. If
`modules/usb_serial_reader.py` cannot import `serial`, check that you are
running `.venv/bin/python`; plain `sudo python ...` usually switches to
system/root Python and bypasses `.venv`. If opening `/dev/ttyACM0` or
`/dev/ttyUSB0` needs sudo, add the user to the device group shown by
`ls -l /dev/ttyACM0`, usually `dialout`, then relogin or reboot.

## Jetson Orin Nano Super WIP Notes

Jetson support targets the **Jetson Orin Nano Super (8 GB)** and is still work
in progress, separate from the main Raspberry Pi usage above. The original
Maxwell-era Jetson Nano is not supported — TensorRT FP16 and the VLA runner
need Ampere or newer.

```bash
sudo ./setup_jetson.sh
```

Jetson setup is intentionally minimal: it installs OS packages, updates a
lightweight project `.venv`, and adds the user to device groups such as
`dialout`, `i2c`, and `gpio` when those groups exist. This robot's Jetson
profile uses only the main header I2C bus for the PCA9685, mapped as Linux bus
7, plus USB serial for IMU reading. The Raspberry Pi virtual I2C channels, OLED
service, and OLED/display Python packages are not part of the Jetson setup.
`pandas` is installed: it is a runtime dependency, not analysis-only, because
`simple_drive.py --record` writes drive logs through it.

Jetson bus 7 (default i2c bus) has not yet been tested at 1 MHz on the Orin
Nano Super, and it needs a Jetson-specific device-tree or kernel configuration
path rather than Raspberry Pi `dtparam=i2c_arm_baudrate`. Before changing bus
speed, check the current PCA9685 bus with:

```bash
i2cdetect -y 7
```

Real-time scheduling is not available on this Jetson — `rt_utils` requests are
expected to fail there, and loop jitter is a CPU-governor question instead.

## Teleoperation

Local gamepad, straight to the valves:

```bash
.venv/bin/python simple_drive.py --robot jetson
.venv/bin/python simple_drive.py --enable-slew --enable-tracks
```

Buttons: **A** start/stop a recording strip · **B** sine excitation · **X**
pump · **Y** reload servo config · D-pad U/D sine amplitude.

Remote operator over UDP — pass `--ip` on the robot to take commands from a
client instead of the local pad:

```bash
# robot
.venv/bin/python simple_drive.py --ip 0.0.0.0:8080
# or the full IK/pose server
.venv/bin/python excv_gui.py --port 8080

# operator machine — robot host/port are set in the GUI (default 192.168.0.132:8080)
python -m clients.client_gui
```

Both scripts request SCHED_FIFO for the control loop, best-effort: the request
needs privileges and is skipped silently when unavailable, so `sudo` helps on
the Pi and does nothing useful on the Jetson.

## EE PID Tuner

Tools for tuning the joint PIDs against an EE x-plane sweep. The host runs the
full production control stack (`HardwareInterface` + `ExcavatorController`);
a thin Tk client shows the ideal X line vs. the measured EE travel and the
tracking-error stats per stroke.

```bash
# robot side — starts both per-joint (port 8090) and EE-plane (port 8091) handlers
sudo .venv/bin/python -m tools.pid_tuner.robot

# operator laptop — tabbed GUI with Per-Joint and EE Tracking tabs
python -m tools.pid_tuner.client 192.168.0.132
```

What it does:

* Drives the EE back and forth along a straight X line at fixed Y, Z, rot=0.
  EE speed is configurable in the `20..70 mm/s` band; Z height is whatever you
  set in the GUI.
* Optionally swaps the chosen joint's PID for an in-script `DirectionalPID`
  wrapper that holds **separate gain triples for out (+X) and in (-X)
  strokes**. During a tuning run the wrapper is pinned to the current stroke
  direction, so the +X strokes always use `pid_pos` (`kp_fwd / ki_fwd / kd_fwd`)
  and the -X strokes always use `pid_neg` (`kp_rev / ki_rev / kd_rev`) —
  useful for asymmetric hydraulics (extend vs. retract).
* Tracking error is also **scored per direction**: every run reports OUT
  (+X) and IN (-X) RMSE / max / mean / cost as separate numbers, and the
  auto-tuner judges fwd-gain probes only against the fwd subscore (and
  vice versa). The GUI shows both lines so you can tell which direction is
  the worse offender at a glance.
* Auto-tune is coordinate descent ("twiddle") with adaptive step size, scoring
  each full back-and-forth run with
  `cost = w_rmse*RMSE + w_max*max|err| + w_overshoot*tail` (computed twice —
  once per direction). Each iteration cycles deterministically through a
  3x3 grid of `(z, speed)` built from `z_min..z_max` and `v_min..v_max` so the
  gains are robust across the whole envelope.
* No changes to the on-disk control stack — the tuner only mutates the live
  `controller.joint_pids` list and restores originals on exit.

Recommended starting points for the excavator (x-plane smoothness is the main
target):

* Joint selector is locked to `boom` (lift), `arm` (tilt) and `bucket` (scoop)
  — the three joints that actually shape EE x motion. Slew is excluded; it
  doesn't contribute to +X / -X tracking when the cab faces forward.
* Auto-tune budget can be generous (e.g. `auto-tune runs = 60..120`); each
  iteration is one full back-and-forth so the wall time scales with
  `strokes_per_run * stroke_duration`. Sweeps can run for tens of minutes
  without issue.
* For finer convergence, run the auto-tuner twice: first with a wider
  `(z, v)` envelope to get robust gains, then with a narrow envelope around
  the operating point you care about most.

The tuner runs paused until the client sends `START`, and PIDs zero outputs
when paused. See the docstrings in `tools/pid_tuner/robot.py` and
`tools/pid_tuner/client.py` for the full CLI, wire format, and algorithm
notes.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Use the project venv — the system Python has no `numba`, so `modules/ik` fails
to import and every controller test errors at collection.

## Layout

| Path | Role |
|---|---|
| `simple_drive.py` | Open-loop gamepad driving + hydraulic data collection. Start here. |
| `excv_gui.py` | UDP robot-side server for the remote client GUI. |
| `modules/` | Hardware interface, controller, IK, PWM, IMU, realtime helpers. |
| `lerobot_vla/` | LeRobot dataset collection + on-device SmolVLA inference. |
| `control_prototype/` | Unvalidated control work: reachability limiter, compensation, closed loop. |
| `clients/` | Operator-side GUIs and input handling. |
| `configuration_files/` | Board/robot profiles: valve, geometry, IMU, IK, PID settings. |
| `tools/` | Bring-up, calibration, analysis, and the PID tuner package. |
| `data_collection/` | Recorded drive logs and LeRobot datasets. |
| `pico_imu_reader/` | XIAO RP2040 firmware: 4× ISM330DHCX → fused orientation over USB CDC. |

The remaining folders are prototypes, hardware bring-up, or experiments.

## More Detail

| Doc | Covers |
|---|---|
| `modules/pwm/README.md` | Valve output path, safety layers, the call-rate contract |
| `modules/ik/README.md` | IK solver internals and the numba hot path |
| `lerobot_vla/README.md` | Dataset recording, remote dataset viz, split-engine inference |
| `control_prototype/README.md` | Smooth reachability limiter design and rubber-band model |
| `jetson_setup.md`, `imu_tuneup.md` | Board bring-up and IMU tuning notes |
