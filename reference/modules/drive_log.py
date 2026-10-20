"""The drive-log schema shared by every recorder on this robot.

simple_drive.py's operator recordings and learned_control/run_circle.py's
controller runs write the same pair of files, so one training pipeline reads
both:

    drive_log_*.csv  hydraulic commands + joint state at the 100 Hz control rate
    imu_raw_*.csv    every IMU frame at the full 200 Hz stream rate

Units are the Isaac convention, not the controller convention:
  timestamp  seconds since recording start (monotonic)
  joint_pos  radians          (controller API returns degrees)
  joint_vel  rad/s            (controller API returns deg/s)
  imu_g*     rad/s            (hardware API returns deg/s)
  imu_a*     m/s^2            (firmware reports g)
  *_cmd_*    normalized [-1, 1]
  *_age_s    seconds          how stale that reading was when the row was written
  *_ts_us    microseconds     Pico device clock, -1 when never reported

Command channels use hydraulic names (rotate/lift/tilt/scoop), not joint
names (slew/boom/arm/bucket), because that is what the training and
benchmark scripts read. COMMAND_CHANNELS is the mapping.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

# Samples per packed block. Long Python lists of floats make every full garbage
# collection walk each logged value -- 30+ ms once a recording holds minutes of
# data, long enough to stall a 100 Hz loop. numpy arrays are not walked, so the
# buffers are packed into them block by block, a few columns per sample so no
# single tick pays for a whole block (~3 ms on the Orin).
PACK_ROWS = 250
PACK_COLUMNS_PER_ROW = 16
PACK_IMU_FRAMES = 100

JOINT_NAMES = ['slew', 'boom', 'arm', 'bucket']
G_TO_MS2          = 9.80665   # firmware reports accel in g; the dataset is SI

# ── log schema ───────────────────────────────────────────────────────────────
#
# The CSV names command channels hydraulically (rotate/lift/tilt/scoop) while the
# controller speaks joint names (slew/boom/arm/bucket). This is the one place
# that mapping is written down: the recorder builds every command column from it
# rather than repeating a positional index per column.
COMMAND_CHANNELS = (
    ('slew',   'rotate'),
    ('boom',   'lift'),
    ('arm',    'tilt'),
    ('bucket', 'scoop'),
)

# Joints whose finite-difference velocity is logged. Slew has none: its angle is
# an absolute world yaw with no zeroing anywhere in the stack, so it is not
# comparable across sessions and the model does not use it.
VELOCITY_JOINTS = ('boom', 'arm', 'bucket')

# IMU roles whose gyro/accel vectors go into the drive log, in stream order. The
# base/cabin sensor is not a joint, so it has no pos/vel counterpart -- it is
# recorded because it is the only sensor that observes slew and machine tilt.
IMU_VECTOR_ROLES = ('boom', 'arm', 'bucket', 'base')


def clean_suffix(raw: str | None) -> str:
    """Normalize a --suffix into a filename tail, with its leading underscore.

    Sanitized rather than trusted: the value lands in a path, and a stray slash
    or space would either scatter strips into unintended directories or produce
    names the training scripts have to be quoted around. Anything outside
    [A-Za-z0-9._-] collapses to a single dash.

    Returns "" for empty or all-punctuation input, which restores the plain
    ``drive_log_<ts>.csv`` name rather than leaving a dangling underscore.
    """
    if not raw:
        return ""
    kept = re.sub(r'[^A-Za-z0-9._-]+', '-', raw.strip()).strip('-_.')
    return f"_{kept}" if kept else ""


class DriveLog:
    """Column buffers for one drive_log/imu_raw pair, and their writers.

    The schema lives in one place, :meth:`build_row`. Samples accumulate as one
    list per named column, so adding a channel is one line there rather than a
    matching edit in three. A recorder may add its own columns to a row before
    :meth:`append`; readers ignore columns they do not know.
    """

    def __init__(self):
        # The open block, one list per column; a closed block still being
        # packed; and the packed blocks. Order per column: packed, closing, open.
        self._cols: dict[str, list] = {}
        self._closing: dict[str, list] = {}
        self._packed: dict[str, list[np.ndarray]] = {}
        self._closed_rows = 0
        self._clear_imu_raw()

    def append(self, row: dict) -> None:
        """Fan one sample into the per-column lists.

        Rows are built by a single expression so they always carry the same
        keys, but this checks rather than trusts: a column that skipped one
        sample would shift every later value against the timeline, and the CSV
        would still look well formed.
        """
        if not self._cols:
            self._cols = {name: [] for name in row}
            self._packed = {name: [] for name in row}
        elif row.keys() != self._cols.keys():
            drift = sorted(set(row) ^ set(self._cols))
            raise RuntimeError(f"log row changed shape mid-recording: {drift}")
        for name, value in row.items():
            self._cols[name].append(value)
        for name in list(self._closing)[:PACK_COLUMNS_PER_ROW]:
            self._packed[name].append(np.asarray(self._closing.pop(name)))
        if len(self._cols['timestamp']) >= PACK_ROWS:
            # Normally long done; finish it so only one block is ever closing.
            for name, values in self._closing.items():
                self._packed[name].append(np.asarray(values))
            self._closing = self._cols
            self._cols = {name: [] for name in self._closing}
            self._closed_rows += PACK_ROWS

    def n_samples(self) -> int:
        return self._closed_rows + len(self._cols.get('timestamp', ()))

    def columns(self) -> dict[str, np.ndarray]:
        """Every logged column as one array, in build_row's order."""
        out = {}
        for name, values in self._cols.items():
            parts = [*self._packed[name]]
            if name in self._closing:
                parts.append(np.asarray(self._closing[name]))
            out[name] = np.concatenate([*parts, np.asarray(values)]) if parts else np.asarray(values)
        return out

    @staticmethod
    def _imu_vectors(gyro: dict | None) -> dict:
        """Per-role gyro [rad/s] and accel [m/s^2] columns, NaN where absent.

        The firmware reports dps and g; the dataset is SI throughout. Roles the
        stream did not supply come back NaN rather than zero -- zero is a real
        reading, and a missing sensor must not look like a stationary one.
        """
        nan3 = (np.nan, np.nan, np.nan)
        gyros = list(gyro['gyro']) if gyro else []
        accels = list(gyro.get('accel') or []) if gyro else []

        out: dict[str, float] = {}
        for i, role in enumerate(IMU_VECTOR_ROLES):
            if role == 'base':
                # The base sensor is carried outside the per-joint arrays.
                g = gyro.get('base_gyro') if gyro else None
                a = gyro.get('base_accel') if gyro else None
            else:
                g = gyros[i] if i < len(gyros) else None
                a = accels[i] if i < len(accels) else None
            gx, gy, gz = np.radians(g) if g is not None else nan3
            ax, ay, az = np.multiply(a, G_TO_MS2) if a is not None else nan3
            out[f'imu_gx_{role}'] = float(gx)
            out[f'imu_gy_{role}'] = float(gy)
            out[f'imu_gz_{role}'] = float(gz)
            out[f'imu_ax_{role}'] = float(ax)
            out[f'imu_ay_{role}'] = float(ay)
            out[f'imu_az_{role}'] = float(az)
        return out

    def build_row(self, t: float, manual: dict, sine: dict, combined: dict,
                   pos_deg, state_ts, state_imu_us,
                   vels, vel_age: float, gyro: dict | None,
                   cmd_age_s: float, cmd_stale: bool, sine_enabled: bool,
                   sine_target: str, sine_seed: int, excitation_meta=None) -> dict:
        """One CSV row. THE schema -- every column this file writes is named here."""
        now = time.perf_counter()
        pos = np.radians(pos_deg)
        fresh_vel = vels is not None and vel_age < 0.05

        row: dict = {'timestamp': t, 'sample_idx': self.n_samples()}

        for joint, channel in COMMAND_CHANNELS:
            row[f'manual_cmd_{channel}'] = float(manual.get(joint, 0.0))
        for joint, channel in COMMAND_CHANNELS:
            row[f'sine_cmd_{channel}'] = float(sine.get(joint, 0.0))
        for joint, channel in COMMAND_CHANNELS:
            row[f'combined_cmd_{channel}'] = float(combined.get(joint, 0.0))
            requested = manual.get(joint, 0.0) + sine.get(joint, 0.0)
            row[f'command_clipped_{channel}'] = int(abs(requested) > 1.0 + 1e-9)
            row[f'effective_excitation_cmd_{channel}'] = (
                float(combined.get(joint, 0.0)) - float(manual.get(joint, 0.0)))

        for i, joint in enumerate(JOINT_NAMES):
            row[f'joint_pos_{joint}'] = float(pos[i]) if i < len(pos) else np.nan
        for joint in VELOCITY_JOINTS:
            i = JOINT_NAMES.index(joint)
            row[f'joint_vel_{joint}'] = float(np.radians(vels[i])) if fresh_vel else np.nan

        row.update(self._imu_vectors(gyro))

        row['cmd_stale'] = int(bool(cmd_stale))
        row['cmd_age_s'] = float(cmd_age_s) if np.isfinite(cmd_age_s) else np.nan
        row['sine_enabled'] = int(bool(sine_enabled))

        # ── capture clocks ───────────────────────────────────────────────────
        # How stale each reading was when this row was written. The control
        # thread and the IMU stream both publish into latest-value caches that
        # this loop samples at its own rate, so without these a repeated sample
        # and a fresh one are indistinguishable after the fact.
        row['state_age_s'] = np.nan if state_ts is None else max(0.0, now - float(state_ts))
        row['vel_age_s'] = float(vel_age) if np.isfinite(vel_age) else np.nan
        # Pico clocks, joining these rows to the companion imu_raw_*.csv. Two of
        # them because they come from two reads: the pose was computed from one
        # IMU frame and the gyro/accel columns above are another, and the loop
        # can straddle a stream update between the two. -1 means the source has
        # not reported yet; int64 has no NaN and 0 is a real Pico timestamp.
        row['state_imu_ts_us'] = -1 if state_imu_us is None else int(state_imu_us)
        gyro_ts = gyro.get('device_timestamp_us') if gyro else None
        row['imu_device_ts_us'] = -1 if gyro_ts is None else int(gyro_ts)

        # The sine target is D-pad switchable mid-recording, so it is per-sample
        # rather than per-file. Historical sine_* names also carry chirp;
        # excitation_mode distinguishes them without breaking old readers.
        row['sine_target'] = str(sine_target)
        row['sine_seed'] = int(sine_seed)
        meta = excitation_meta or {}
        row['excitation_mode'] = meta.get('mode', 'sine')
        row['excitation_version'] = meta.get('version', 1)
        row['excitation_block'] = meta.get('block_id', -1)
        row['excitation_elapsed_s'] = meta.get('elapsed_s', np.nan)
        row['excitation_noise_tick'] = meta.get('noise_tick', -1)
        row['excitation_stage'] = meta.get('stage', 'run' if sine_enabled else 'off')
        return row

    def write_drive_log(self, out: Path):
        """Write the hydraulic strip; column order is build_row's insertion order."""
        import pandas as pd

        df = pd.DataFrame(self.columns())
        df.to_csv(out, index=False)
        print(f"[SAVE] {len(df)} samples ({df['timestamp'].iloc[-1]/60:.2f} min) → {out}")
        self._report_staleness(df)
        for _, channel in COMMAND_CHANNELS:
            count = int(df[f'command_clipped_{channel}'].sum())
            if count:
                print(f"[CLIP] {channel}: {count}/{len(df)} commands saturated")
        return df

    @staticmethod
    def _report_staleness(df) -> None:
        """Say how fresh the readings behind this strip actually were.

        Worth a line at save time rather than a question later: the ages are in
        the file either way, but nobody goes looking at them unless something
        already looks wrong, and by then the run is over.
        """
        for column, label in (('state_age_s', 'pose'), ('vel_age_s', 'velocity')):
            ages = df[column].to_numpy(dtype=float)
            finite = ages[np.isfinite(ages)]
            if finite.size == 0:
                print(f"[AGE ] {label}: never reported")
                continue
            missing = ages.size - finite.size
            note = f", {missing} rows without a reading" if missing else ""
            print(f"[AGE ] {label}: median {np.median(finite)*1e3:.1f} ms, "
                  f"max {finite.max()*1e3:.1f} ms{note}")

    def _clear_imu_raw(self):
        self._imu_ts:    list = []
        self._imu_vals:  list = []
        self._imu_packed: list[tuple[np.ndarray, np.ndarray]] = []

    def log_imu_raw(self, frames, n_sensors: int):
        """Buffer raw IMU frames drained from the reader.

        One row per frame at the stream's own rate, not the control rate — the
        control loop runs at 100 Hz while the Pico streams 200 Hz, and halving
        the sample rate of an AHRS input changes the very integration behaviour
        a gain sweep is trying to measure.
        """
        for ts_us, packets in frames:
            row = []
            for i in range(n_sensors):
                pkt = packets[i] if i < len(packets) else []
                # Old firmware sends 7 floats; pad so the row width is fixed.
                row.extend(pkt[:10] + [np.nan] * (10 - len(pkt[:10])))
            self._imu_ts.append(int(ts_us))
            self._imu_vals.append(row)
        if len(self._imu_ts) >= PACK_IMU_FRAMES:
            self._imu_packed.append((np.asarray(self._imu_ts, dtype=np.int64),
                                     np.asarray(self._imu_vals, dtype=np.float64)))
            self._imu_ts, self._imu_vals = [], []

    def n_imu_raw_samples(self) -> int:
        return sum(len(ts) for ts, _ in self._imu_packed) + len(self._imu_ts)

    def write_imu_raw(self, out: Path, roles: list[str], stream_info: dict) -> Path | None:
        """Write the raw IMU strip: quaternion + the gyro/accel that produced it.

        Kept out of the hydraulic CSV rather than bolted onto it — the two run at
        different rates, and the hydraulic schema is what the training and
        benchmark scripts read. Join on device_ts_us against the hydraulic log's
        imu_device_ts_us column.

        Units are the firmware's, not the Isaac convention used by the hydraulic
        log: gyro in dps and accel in g, which is what Fusion's AHRS takes, so
        an offline replay can feed these columns in without converting.
        """
        if not self.n_imu_raw_samples():
            return None

        import pandas as pd

        width = 10 * len(roles)
        vals = np.concatenate([v for _, v in self._imu_packed]
                              + [np.asarray(self._imu_vals, dtype=np.float64).reshape(-1, width)])
        cols = {'device_ts_us': np.concatenate([t for t, _ in self._imu_packed]
                                               + [np.asarray(self._imu_ts, dtype=np.int64)])}
        for i, role in enumerate(roles):
            base = i * 10
            for j, name in enumerate(('qw', 'qx', 'qy', 'qz',
                                      'gx_dps', 'gy_dps', 'gz_dps',
                                      'ax_g', 'ay_g', 'az_g')):
                cols[f'imu_{role}_{name}'] = vals[:, base + j]
        df = pd.DataFrame(cols)

        ranges = stream_info.get('ranges') or {}
        # Full scales are constant for a run; carrying them per row keeps the
        # file self-describing, so headroom can be judged without knowing which
        # firmware was flashed.
        df['gyro_range_dps'] = ranges.get('gyro_dps', np.nan)
        df['accel_range_g']  = ranges.get('accel_g', np.nan)

        df.to_csv(out, index=False)
        span_s = (df['device_ts_us'].iloc[-1] - df['device_ts_us'].iloc[0]) / 1e6
        rate = len(df) / span_s if span_s > 0 else float('nan')
        # A non-zero count means the reader's buffer overflowed between drains,
        # so the strip has gaps — check device_ts_us deltas before trusting it.
        dropped = int(stream_info.get('capture_dropped', 0) or 0)
        drop_note = f", {dropped} dropped since capture start" if dropped else ""
        print(f"[SAVE] {len(df)} IMU frames ({rate:.0f} Hz{drop_note}) → {out}")
        return out
