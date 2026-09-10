# Excitation recordings

Run from the repository with its Python environment active. Board selection
works as before (`--robot rpi`, `--robot jetson`, or auto-detection).

For a repeatable bucket chirp:

```bash
python simple_drive.py --excitation chirp --excitation-target scoop --excitation-seed 42 --suffix bucket_chirp
```

This starts with excitation **off**. The chirp amplitude defaults to 0.35 of
the normalized valve command range. It sweeps logarithmically from 0.05 to
0.9 Hz over 60 seconds, back down over 60 seconds, then supplies zero
excitation for 10 seconds. The cycle repeats. These are collection settings,
not a measurement of the machine's bandwidth.

1. Position the arm with manual control and select the target.
2. Press **A** to record, then **B** to enable the selected waveform.
3. Use neutral sticks to isolate the selected joint. Manual input remains
   available and is added to excitation, so use it to manage available travel.
4. **B** turns excitation off immediately. **A**, or the ten-minute recording
   timeout, stops recording and disables excitation. Press B again when ready
   for another excitation block.

**X** toggles the pump, **Y** reloads calibration, and local **D-pad up/down**
cycles the target. Selecting a new target stops excitation on the old one and
starts the new one through a one-second entry taper. Chirps also taper into
their neutral interval. Manual commands are not tapered by this generator.
Input loss neutralizes commands and disarms excitation; reconnecting alone
does not re-enable it.

The chirp's neutral interval only zeros **excitation**. Release the sticks to
record a neutral hold. For single-axis data, keep other manual channels at zero.
Excitation is open-loop and does not impose angle or workspace limits.

## Options

| Option | Default / behavior |
| --- | --- |
| `--excitation sine` | Existing randomized modulated sine with filtered noise |
| `--excitation chirp` | Clean up/down logarithmic chirp, no added noise |
| `--excitation-target` | `all`; also `lift`, `tilt`, `scoop`, the three pairs, or `slew` |
| `--excitation-seed 42` | Reuse a session seed each recording; omitted draws a new seed |
| `--excitation-amplitude 0.35` | Fixed excitation amplitude/cap in `(0, 1]`; default sine draws 0.35–1.0 per joint, chirp uses 0.35 |
| `--chirp-start-hz 0.05` | Positive lower frequency |
| `--chirp-end-hz 0.9` | Upper frequency, at least the lower one and at most 0.9 Hz |
| `--chirp-seconds 60` | Time per direction; minimum 2 seconds |
| `--suffix bucket_chirp` | Label for all companion files |

Slew requires `--enable-slew` and is a solo excitation target. Repeat it with
the arm tucked and extended to check inertia dependence. Tracks retain their
manual controls and are never excited automatically.

For the randomized sine with fixed amplitude and repeat seed:

```bash
python simple_drive.py --excitation sine --excitation-target lift --excitation-amplitude 0.35 --excitation-seed 42 --suffix boom_sine
```

Use `--excitation-target tilt` for arm and `scoop` for bucket. Different seeds
and poses broaden coverage; repeated protocols measure repeatability.
Keep some complete recordings without excitation. Include individual-axis
steps, reversals and 5–10 second neutral holds, plus a stationary 30–60 second
recording after startup calibration.

## Files and reproducibility

Copy the complete set sharing one timestamp and suffix:

- `drive_log_*.csv`: existing command/state columns, plus excitation metadata
  and command saturation diagnostics.
- `imu_raw_*.csv`: raw IMU stream, when IMUs are enabled.
- `excitation_*.json`: generator version, seed, parameters and first recorded
  sample for each block. Saved once per block rather than repeated on every row.

Historical `sine_cmd_*`, `sine_enabled`, `sine_target` and `sine_seed` columns
remain present for both waveforms. Read `excitation_mode` to distinguish them.
The new fields include `excitation_version`, `excitation_block`,
`excitation_elapsed_s`, `excitation_noise_tick` and `excitation_stage`
(`off`, `run`, `up`, `down`, `rest`). The JSON records the target for each
block. Reseeding, enabling or changing targets starts a new parameter block.

Sine version 2 uses separate parameter/noise streams and a fixed 100 Hz noise
grid. The waveform at the same elapsed time is independent of polling jitter.
The JSON includes NumPy/RNG versions and all drawn parameters. A seed plus the
same sequence of block changes reproduces the session, but manually pressing
buttons at different times does not produce identical wall-clock commands.
Use the recorded block and elapsed-time fields for alignment.

The manual-plus-excitation sum is clipped to [-1, 1].
`command_clipped_*` flags saturation and `effective_excitation_cmd_*` records
`combined_cmd_* - manual_cmd_*`. Save-time output reports clipping counts.
Use `combined_cmd_*` as the authoritative requested valve inputs for model
training. They are upstream of the existing PWM calibration/ramp/dither;
the logger does not measure physical valve spool position.

## Verification

Hardware-free tests:

```bash
python -m pytest tests/test_drive_excitation.py tests/test_simple_drive_input.py tests/test_simple_drive_profiles.py -q
```

These cover phase/frequency integration, bounded commands, neutral intervals,
target routing, reproducibility, operator-input loss, recording stop and
matching CSV/JSON outputs. They do not operate the excavator.
`data_collection/test_sine_valve_range.py` is a separate **hardware experiment**,
not a software test.
