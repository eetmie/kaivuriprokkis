#!/usr/bin/env python3
"""Exercise the offline executable, segment provenance and delivered PID clocks."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import yaml

JOINTS = ("boom", "arm", "bucket")


def main():
    binary, profile = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="kaivuri-ik-preview-") as tmp:
        outputs = [Path(tmp) / name for name in ("first", "repeat")]
        tables = []
        for out in outputs:
            subprocess.run([binary, "--out", str(out), "--profile", profile, "--seconds", "150"], check=True)
            with (out / "preview.csv").open() as stream:
                tables.append(list(csv.DictReader(stream)))
        first, repeat = tables
        assert len(first) == len(repeat) == 15000
        # Wall time is observational; every controller/excitation value must replay identically.
        for a, b in zip(first, repeat):
            assert {k: v for k, v in a.items() if k != "step_work_us"} == {k: v for k, v in b.items() if k != "step_work_us"}
        protocol = yaml.safe_load((outputs[0] / "protocol.yaml").read_text())
        summary = yaml.safe_load((outputs[0] / "summary.yaml").read_text())
        assert protocol["synthetic_only"] and summary["synthetic_only"]
        assert not protocol["collision_checked"] and not summary["collision_checked"]
        assert protocol["output_hz"] == 100
        segments = {s["id"]: s for s in protocol["segments"]}
        rates_seen = set()
        last_updates = [None] * 3
        last_gains = [None] * 3
        for tick, row in enumerate(first):
            now = float(row["timestamp"])
            assert abs(now - tick / 100) < 1e-12
            s = segments[int(row["segment_id"])]
            assert .1 <= s["duration_requested_s"] <= 3
            assert abs(s["duration_requested_s"] * 100 - round(s["duration_requested_s"] * 100)) < 1e-10
            if row["valid"] == "0":
                assert row["reason"] == "ik_unreachable", row["reason"]
                assert all(float(row[f"command_{j}"]) == 0 for j in JOINTS)
                last_updates = [None] * 3
                continue
            assert float(row["ik_position_error_m"]) < .0005
            assert float(row["ik_pitch_error_rad"]) < .005
            measured = [float(row[f"task_{j}"]) for j in ("x_m", "z_m", "pitch_rad")]
            reference = [float(row[f"task_ref_{j}"]) for j in ("x_m", "z_m", "pitch_rad")]
            assert math.hypot(reference[0] - measured[0], reference[1] - measured[1]) <= .03000000001
            for index, j in enumerate(JOINTS):
                command, assist, excitation = (float(row[f"{key}_{j}"]) for key in ("command", "assist", "excitation_requested"))
                assert -1 <= command <= 1 and -.5 <= assist <= .5
                assert abs(command - max(-1, min(1, assist + excitation))) < 1e-12
                assert abs(float(row[f"excitation_effective_{j}"]) - (command - assist)) < 1e-12
                rate = float(row[f"pid_rate_requested_hz_{j}"])
                rates_seen.add(rate)
                current_gains = [float(row[f"{term}_applied_{j}"]) for term in ("kp", "ki", "kd")]
                if row[f"pid_updated_{j}"] == "1":
                    dt = float(row[f"pid_dt_s_{j}"])
                    if last_updates[index] is not None:
                        assert abs(dt - (now - last_updates[index])) < 1e-11
                        assert dt >= 1 / rate - 1e-12
                    else:
                        assert abs(dt - .01) < 1e-12
                    requested = s["pid_gains_requested"][index]
                    assert current_gains == [requested[term] for term in ("kp", "ki", "kd")]
                    last_updates[index] = now
                    last_gains[index] = current_gains
                else:
                    assert math.isnan(float(row[f"pid_dt_s_{j}"]))
                    assert current_gains == last_gains[index]
        assert rates_seen == {25, 50, 100}
        blocks = protocol["excitation_blocks"]
        assert {b["mode"] for b in blocks} == {"clean", "sine", "chirp"}
        assert all(b["end_ns_requested"] - b["begin_ns"] == 20_000_000_000 for b in blocks)
        # Short command segments must not restart the long excitation sweeps.
        for block in (b for b in blocks if b["mode"] == "chirp"):
            block_rows = [r for r in first if int(r["excitation_block"]) == block["index"]]
            if len(block_rows) < 2000:
                continue
            assert {r["excitation_stage"] for r in block_rows} == {"up", "down", "rest"}
            assert len({r["segment_id"] for r in block_rows}) > 3
            samples = np.asarray([[float(r[f"excitation_requested_{j}"]) for j in JOINTS] for r in block_rows[:1000]])
            power = abs(np.fft.rfft(samples, axis=0)) ** 2
            hz = np.fft.rfftfreq(len(samples), .01)
            assert power[hz >= 3].sum() / power.sum() > .01
        # Rebuild dwell directly from measured samples, not from chosen coverage goals.
        config = yaml.safe_load((Path(profile) / "control_config.yaml").read_text())
        limits = np.deg2rad(config["ik"]["joint_limits_relative"][1:4])
        bin_size = protocol["coverage_bin_rad"]
        bins = np.ceil((limits[:, 1] - limits[:, 0]) / bin_size).astype(int)
        expected = {}
        for r in first[:-1]:
            q = np.asarray([float(r[f"q_rad_{j}"]) for j in JOINTS])
            indices = np.clip(np.floor((q - limits[:, 0]) / bin_size).astype(int), 0, bins - 1)
            cell = int(np.ravel_multi_index(tuple(indices), tuple(bins)))
            expected[cell] = expected.get(cell, 0) + .01
        with (outputs[0] / "coverage.csv").open() as stream:
            actual = {int(r["cell"]): float(r["dwell_s"]) for r in csv.DictReader(stream)}
        assert actual.keys() == expected.keys()
        assert all(abs(actual[k] - expected[k]) < 1e-9 for k in actual)
        assert abs(sum(actual.values()) - 149.99) < 1e-8
        for bad in (("--pid-rates", "60"), ("--gain-min", "nan"), ("--max-seconds", "0.01")):
            result = subprocess.run([binary, "--out", str(Path(tmp) / "invalid"), "--profile", profile, *bad], capture_output=True)
            assert result.returncode != 0
            assert not (Path(tmp) / "invalid").exists()
        result = subprocess.run([binary, "--out", str(outputs[0]), "--profile", profile], capture_output=True)
        assert result.returncode != 0 and b"already exists" in result.stderr
        print(f"Preview replay, PID clocks, measured coverage and fast chirp checks passed: {len(first)} rows")


if __name__ == "__main__":
    main()
