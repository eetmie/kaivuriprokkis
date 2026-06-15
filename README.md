# kaivuriprokkis

Hardware/IRL runner for MASI excavator path planning. The main entry point is
`run_hw_v2.py`.

The goal is to run the same planner logic as the Isaac Sim side, execute it on
the real machine, and log data in a format that can be compared with sim runs.
The shared planner package lives in `pathing/` and matches the sim package.

## Main Run

Typical hardware run:

```bash
source .venv/bin/activate
sudo .venv/bin/python run_hw_v2.py --algorithm a_star -r --task in-and-out --log --once
```

Useful examples:

```bash
sudo .venv/bin/python run_hw_v2.py --algorithm a_star --task in-and-out --log --once
sudo .venv/bin/python run_hw_v2.py --algorithm rrt -r --task rotation --log --debug
sudo .venv/bin/python run_hw_v2.py --algorithm prm -r -p --task empty --log --once
sudo .venv/bin/python run_hw_v2.py --test --once
```

`--test` sends direct A/B pose commands without the planner. It is useful for
checking basic control behavior, but normal pathing work should use planner
mode.

On Raspberry Pi, run `run_hw_v2.py` with `sudo` as shown above. This lets the
realtime helpers in `rt_utils` apply the requested scheduling settings while
still using the project virtual environment.

## Platform Setup And Privileges

`setup.sh` is Raspberry Pi-specific. It edits Raspberry Pi boot config, creates
virtual I2C buses for the ADC/OLED wiring, and installs an OLED service. Do not
use it on Jetson Nano.

The Raspberry Pi main I2C bus runs well at 1 MHz in this project.

## Jetson Orin Nano Super WIP Notes

Jetson support targets the **Jetson Orin Nano Super (8 GB)** and is still work
in progress, separate from the main Raspberry Pi usage above. The original
Maxwell-era Jetson Nano is not supported — Reflex/TensorRT FP16 and the VLA
runner (`run_vla_v0.py`) need Ampere or newer.

For the Orin Nano Super, use:

```bash
sudo ./setup_jetson.sh
```

Jetson setup is intentionally minimal: it installs OS packages, updates a
lightweight project `.venv`, and adds the user to device groups such as
`dialout`, `i2c`, and `gpio` when those groups exist. This robot's Jetson
profile uses only the main header I2C bus for the PCA9685, mapped as Linux bus
7, plus USB serial for IMU reading. The Raspberry Pi virtual I2C channels, OLED
service, OLED/display Python packages, and analysis-only `scipy`/`pandas`
packages are not part of the Jetson setup.

Jetson bus 7 (default i2c bus) has not yet been tested at 1 MHz on the Orin
Nano Super, and it needs a Jetson-specific device-tree or kernel configuration
path rather than Raspberry Pi `dtparam=i2c_arm_baudrate`. Before changing bus speed, check the current
PCA9685 bus with:

```bash
i2cdetect -y 7
```

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

## Main Flags

| Argument | Description |
|---|---|
| `--algorithm` | Base planner: `a_star`, `rrt`, `rrt_star`, `prm`. Default: `a_star`. |
| `-r`, `--radial` | Use the radial-first wrapper around the base planner. Best fit for large excavator slew moves. |
| `-p`, `--planar` | Use the planar variant of the base planner. |
| `--radial-mode` | `reconstructed` by default. Use `raw` to skip radial reconstruction. |
| `--task` | Task preset: `in-and-out`, `rotation`, `empty`. |
| `--obstacles-json PATH` | Load obstacles exported by sim with `run_sim_v2.py --dump-obstacles`. |
| `--log` | Write trajectory CSVs and metrics to `logs_hw/`. |
| `--once` | Run one sweep through the task goals and stop. |
| `--test` | Direct A/B pose test. No planner. |
| `--debug` | Verbose hardware, controller, and planner logging. |
| `--debug-planning` | Verbose planner logs only. |
| `--rt-priority` | Control-loop realtime priority. Use `0` to disable. |
| `--imu-priority` | IMU thread realtime priority. Use `0` for normal scheduling. |

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

## Sim Replay Flow

To compare sim and IRL against the same obstacle layout:

1. Export obstacles from sim:

   ```bash
   .\isaaclab.bat -p scripts/masi/pathing/run_sim_v2.py --algorithm a_star -r --dump-obstacles obstacles.json
   ```

2. Copy or reuse that JSON on the hardware side.

3. Run hardware with:

   ```bash
   sudo .venv/bin/python run_hw_v2.py --algorithm a_star -r --obstacles-json obstacles.json --log --once
   ```

## Layout

| Path | Role |
|---|---|
| `run_hw_v2.py` | Main hardware pathing runner. Start here. |
| `pathing/` | Shared planner package, identical to the sim pathing package. |
| `configuration_files/` | Task, environment, and execution settings. |
| `modules/` | Hardware interface, controller, math, and realtime helpers. |
| `tools/pid_tuner/` | PID tuner package: `robot.py` (robot server) + `client.py` (PC GUI) + `common.py` (shared protocol). |
| `logs_hw/` | Hardware logs when `--log` is enabled. Created at runtime. |

The other folders and scripts are mostly for prototypes, hardware bring-up,
data collection, or testing ideas. They can be useful, but they are not the
main pathing workflow.

## More Detail

`pathing/README.md` documents the shared planner internals, including radial
and planar variants plus the red/yellow path outputs used in sim visualization.
