# Recorded pose coverage

`plot_coverage.py` is an offline plotter for actuator-data collection. It reads
saved logs, geometry and joint limits and writes six PNGs plus `summary.json`.
It opens no devices and does not import control loops, Torch, Isaac Sim or USD.
Dependencies: NumPy, matplotlib and PyYAML. The robot's system `python3` currently
has all three; no control environment changes are needed.

From the kaivuriprokkis repository root:

```bash
python3 data_collection/plot_coverage.py \
  --logs 'data_collection/hydraulic_data/drive_log_20260911_*.csv' \
  --profile configuration_files/profiles/jetson_bucket/control_config.yaml \
  --label september11_v2 \
  --out data_collection/coverage_plots/september11_v2
```

For a motion-only report, use a separate output directory:

```bash
python3 data_collection/plot_coverage.py \
  --logs 'data_collection/hydraulic_data/drive_log_20260911_*.csv' \
  --profile configuration_files/profiles/jetson_bucket/control_config.yaml \
  --motion-only --min-speed-deg-s 1 \
  --out data_collection/coverage_plots/september11_v2_moving
```

`--logs` accepts multiple files, quoted globs or directories. Directories select
only their immediate `drive_log_*.csv` and `drive_log_*.csv.gz` children: old
calibration groups and validation sessions are not selected recursively. Within
the selected directory a validation-named file is included like any other log;
select explicit files or a narrower glob to exclude it. Repeated paths and exact
file copies count once. Missing requested files/columns cause an error.
Output defaults to a timestamped `data_collection/coverage_plots/` directory;
an existing nonempty output directory is never overwritten. The same script can
run on the training computer with copied logs and a copied profile.

## Reading the plots

- `workspace.png`: bucket-tip X/Z position, pooling all bucket orientations.
- `workspace_by_pitch.png`: the same positions sliced by absolute bucket pitch.
  All 24 default slices are shown, including empty ones.
- `joint_pairs.png`: boom/arm, boom/bucket and arm/bucket angle combinations,
  pooling the third joint. Dashed lines mark the configured joint limits.
- `joint_boom_arm_by_bucket.png`, `joint_boom_bucket_by_arm.png`, and
  `joint_arm_bucket_by_boom.png`: joint pairs sliced by the third joint, revealing
  missing configurations hidden by the pooled views.

Colour means recorded **seconds per bin**, on one logarithmic scale across all
six figures. White means no accepted recorded duration. In workspace plots,
gray means no sampled geometry configuration reached that bin; white means a
configuration was sampled there but no data was recorded there. Sparse geometry
sampling can miss narrow regions. Geometry uses joint limits only, without
collision or terrain checks. Treat it as a geometric envelope, not an automated
motion plan. Recorded points outside that envelope are still shown.

Default bins: 20 mm X/Z, 10 degrees per joint, 15 degrees per pitch/third-joint
slice. Change `--spatial-bin-mm`, `--joint-bin-deg`, or `--pitch-bin-deg` to inspect
different scales. `--workspace-step-deg` defaults to 3 degrees; reducing it
densifies the workspace reference. `--dpi` controls image resolution.

## Frames and calibration

Forward kinematics follows the robot's URDF-style chain using the profile's full
link vectors and bucket-tip offset, including nonzero Z components. Slew is held
at zero for a carriage-relative arm-plane map. Base rocking, global yaw and track
movement are excluded: these plots describe the driven arm's configuration,
not a world-space map. Angles in CSVs are radians; plot angles are degrees.
Bucket pitch is positive about Y, wrapped to [-180, 180) degrees.

The recommended `jetson_bucket` profile includes the cutting-tip geometry,
`coordinate_origin: slew_bearing` and a cutting-lip `pitch_offset_rad`. A legacy
profile without that offset instead reports bucket joint-frame pitch. Never
compare absolute tip coordinates generated using different geometry origins.
An exported bundle's `robot_geometry.json` is also accepted as `--profile`.

Geometry selection does **not** recalibrate logged angles. Keep IMU calibration
groups separate and use `--label` to identify them. If an independently known
conversion is needed, `--angle-offset-deg BOOM ARM BUCKET` explicitly adds those
offsets before FK and plotting; the report records them. No correction is guessed
from a filename or from today's mounting offsets. The example September reports
use angles as recorded and the bucket profile's geometry (zero added offsets).

## Accepted duration

Required columns: `timestamp`, `joint_pos_boom/arm/bucket`, and
`combined_cmd_lift/tilt/scoop`. A usable row has finite values and, when available:

- `cmd_stale == 0`;
- finite, nonnegative `cmd_age_s` and `state_age_s` <= `--max-age-ms` (default 30).

The duration from row i to i+1 counts only when both rows are usable, timestamps
increase by at most `--max-gap-ms` (default 30), and `sample_idx` advances by one
when present. The last row has no known duration. Invalid rows and gaps split
coverage; they are never removed and bridged. In-limit and out-of-limit recorded
poses are both included, with out-of-limit counts in the JSON report.

`--motion-only` also requires finite `joint_vel_boom/arm/bucket`, at least one
absolute rate >= `--min-speed-deg-s`, and fresh `vel_age_s` when present. These are
the logged velocity estimates; adjust the threshold if stationary sensor noise
appears as motion. Neutral holds remain useful data in the overall report.

Older logs without quality flags can still be plotted. `summary.json` lists
unavailable checks, accepted seconds, rejected row reasons, discontinuities,
angle ranges, out-of-limit counts and input/profile hashes. Rejection reasons
can overlap. Companion raw IMU strips are not required for these pose plots.

This is **recorded pose coverage**, not MLP training-window coverage or proof of
adequate excitation. Training additionally requires synchronized sensor data,
complete histories and valid targets. Direction, valve amplitude, reversals and
multi-joint flow sharing can be explored separately after filling pose gaps.

## Hardware-free verification

```bash
.venv/bin/python -m pytest tests/test_plot_coverage.py -q
```

Tests compare FK with the robot's existing implementation at random and limit
poses and check actual-duration accounting, gaps, missing samples, invalid/stale
rows, motion filtering, legacy quality fields and compressed input. Generating
plots from real logs exercises rendering without opening hardware.
