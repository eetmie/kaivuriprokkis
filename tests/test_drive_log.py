"""modules/drive_log.py: packed column blocks write the same files as one long list."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import drive_log  # noqa: E402


def record(monkeypatch, tmp_path, rows, frames):
    """Log a mixed-type row stream and raw frames; write both strips."""
    monkeypatch.setattr(drive_log, "PACK_ROWS", rows)
    monkeypatch.setattr(drive_log, "PACK_IMU_FRAMES", frames)
    log = drive_log.DriveLog()
    for k in range(23):
        row = log.build_row(
            k * 0.01, {}, {}, {"boom": 0.1 * (k % 3)}, np.arange(4) + k, None, 1000 + k,
            None if k == 5 else [1.0, 2.0, 3.0, 4.0], 0.001, None, 0.0, k == 7, True, "all", -1,
            {"mode": "circle_mlp", "stage": "wait" if k < 10 else "circle", "elapsed_s": np.nan if k < 10 else k},
        )  # fmt: skip
        row["pass_index"] = k // 10
        log.append(row)
        log.log_imu_raw([(2 * k, [[1.0, 0.5 * k]] * 2), (2 * k + 1, [[1.0] + [0.0] * 9] * 2)], 2)
    tag = f"{rows}_{frames}"
    log.write_drive_log(tmp_path / f"drive_{tag}.csv")
    log.write_imu_raw(tmp_path / f"raw_{tag}.csv", ["boom", "base"], {})
    assert log.n_samples() == 23 and log.n_imu_raw_samples() == 46
    return (tmp_path / f"drive_{tag}.csv").read_text(), (tmp_path / f"raw_{tag}.csv").read_text()


def test_packed_blocks_write_the_same_files(monkeypatch, tmp_path):
    whole = record(monkeypatch, tmp_path, 10**6, 10**6)
    packed = record(monkeypatch, tmp_path, 4, 7)
    assert packed == whole
    table = pd.read_csv(tmp_path / "drive_4_7.csv")
    assert (table["sample_idx"] == np.arange(23)).all()
    assert table["cmd_stale"].sum() == 1 and table["joint_vel_boom"].isna().sum() == 1
