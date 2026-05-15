# kaivuriprokkis

Hardware/IRL runner for MASI excavator path planning. The main entry point is
`run_hw_v2.py`.

The goal is to run the same planner logic as the Isaac Sim side, execute it on
the real machine, and log data in a format that can be compared with sim runs.
The shared planner package lives in `pathing/` and matches the sim package.

## Main Run

Typical hardware run:

```bash
sudo python run_hw_v2.py --algorithm a_star -r --task in-and-out --log --once
```

Useful examples:

```bash
sudo python run_hw_v2.py --algorithm a_star --task in-and-out --log --once
sudo python run_hw_v2.py --algorithm rrt -r --task rotation --log --debug
sudo python run_hw_v2.py --algorithm prm -r -p --task empty --log --once
sudo python run_hw_v2.py --test --once
```

`--test` sends direct A/B pose commands without the planner. It is useful for
checking basic control behavior, but normal pathing work should use planner
mode.

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

## Sim Replay Flow

To compare sim and IRL against the same obstacle layout:

1. Export obstacles from sim:

   ```bash
   .\isaaclab.bat -p scripts/masi/pathing/run_sim_v2.py --algorithm a_star -r --dump-obstacles obstacles.json
   ```

2. Copy or reuse that JSON on the hardware side.

3. Run hardware with:

   ```bash
   sudo python run_hw_v2.py --algorithm a_star -r --obstacles-json obstacles.json --log --once
   ```

## Layout

| Path | Role |
|---|---|
| `run_hw_v2.py` | Main hardware pathing runner. Start here. |
| `pathing/` | Shared planner package, identical to the sim pathing package. |
| `configuration_files/` | Task, environment, and execution settings. |
| `modules/` | Hardware interface, controller, math, and realtime helpers. |
| `logs_hw/` | Hardware logs when `--log` is enabled. Created at runtime. |

The other folders and scripts are mostly for prototypes, hardware bring-up,
data collection, or testing ideas. They can be useful, but they are not the
main pathing workflow.

## More Detail

`pathing/README.md` documents the shared planner internals, including radial
and planar variants plus the red/yellow path outputs used in sim visualization.
