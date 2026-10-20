# reference

Frozen copies of the Python code the native stack was ported from. Nothing here runs on the robot.
They exist so these checks keep working:

| Check | Loads |
| --- | --- |
| `native_stack/tests/reference_parity.py` (PWM pulses + FK) | `modules/pwm/`, `modules/perf_tracker.py`, `modules/ik/kinematics.py` |
| `ik-reference-parity` (CTest) | `modules/pid.py`, `data_collection/plot_coverage.py` |
| `native_stack/tests/freeze_mlp_excitation.py` (fixture regeneration) | `simple_drive.py` |

`modules/drive_log.py` (log schema), `modules/gamepad.py` (button map) and
`data_collection/EXCITATION.md` are the specs the MLP collector follows.

Do not edit these to make a test pass; the full originals live on branch `python`.
