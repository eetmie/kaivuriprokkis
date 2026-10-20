#!/usr/bin/env python3
"""Summarize a synthetic IK/PID preview without importing any hardware modules."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

JOINTS = ("boom", "arm", "bucket")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    protocol = yaml.safe_load((root / "protocol.yaml").read_text())
    assert protocol["synthetic_only"], "Expected an explicitly synthetic preview"
    with (root / "preview.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    def values(key):
        return np.asarray([float(r[key]) for r in rows])
    def joint_values(prefix):
        return np.column_stack([values(f"{prefix}_{j}") for j in JOINTS])

    t = values("timestamp")
    q = np.rad2deg(joint_values("q_rad"))
    command = joint_values("command")
    rates = joint_values("pid_rate_requested_hz")
    segments = protocol["segments"]
    duration = np.asarray([s["duration_requested_s"] for s in segments])
    kp = np.asarray([[g["kp"] for g in s["pid_gains_requested"]] for s in segments])
    ki = np.asarray([[g["ki"] for g in s["pid_gains_requested"]] for s in segments])
    base_p = np.asarray([g["kp"] for g in protocol["base_gains"]])
    base_i = np.asarray([g["ki"] for g in protocol["base_gains"]])
    # Average per-block spectra with the same window/normalization for both modes.
    spectra = {"clean": [], "excited": []}
    for block in protocol["excitation_blocks"]:
        indices = np.flatnonzero(values("excitation_block") == block["index"])
        if len(indices) != 2000:
            continue
        signal = command[indices] - command[indices].mean(axis=0)
        signal *= np.hanning(2000)[:, None]
        power = (abs(np.fft.rfft(signal, axis=0)) ** 2).sum(axis=1)
        spectra["excited" if block["enabled"] else "clean"].append(power)
    hz = np.fft.rfftfreq(2000, .01)
    mean_spectra = {mode: np.mean(samples, axis=0) for mode, samples in spectra.items() if samples}
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    ax = axes[0, 0]
    point = ax.scatter(values("task_x_m")[::5] * 1000, values("task_z_m")[::5] * 1000,
                       c=t[::5], s=2, rasterized=True)
    ax.set(xlabel="Tip X [mm]", ylabel="Tip Z [mm]", title="Measured synthetic tip path", aspect="equal")
    figure.colorbar(point, ax=ax, label="Time [s]")
    ax = axes[0, 1]
    point = ax.scatter(q[::5, 0], q[::5, 1], c=q[::5, 2], s=2, rasterized=True)
    ax.set(xlabel="Boom [deg]", ylabel="Arm [deg]", title="Full joint configurations")
    figure.colorbar(point, ax=ax, label="Bucket [deg]")
    ax = axes[0, 2]
    for mode, power in mean_spectra.items():
        ax.semilogy(hz, np.maximum(power, 1e-14), label=mode)
    ax.set(xlim=(0, 15), xlabel="Frequency [Hz]", ylabel="Mean windowed command power", title="Final valve command spectra")
    ax.axvline(3, color="grey", lw=.6, ls=":"); ax.axvline(12, color="grey", lw=.6, ls=":"); ax.legend()
    ax = axes[1, 0]
    bins = np.geomspace(.1, 3.001, 18)
    coverage = np.asarray([s["kind"] == "coverage" for s in segments])
    ax.hist([duration[~coverage], duration[coverage]], bins=bins, stacked=True, label=["other commands", "coverage travel"])
    ax.set(xscale="log", xlabel="Requested segment duration [s]", ylabel="Segments", title="Short changes and longer travel")
    ax.set_xticks([.1, .3, 1, 3], labels=["0.1", "0.3", "1", "3"]); ax.legend(fontsize=8)
    ax = axes[1, 1]
    choices = protocol["pid_rate_choices_hz"]
    for j, name in enumerate(JOINTS):
        dwell = [np.count_nonzero(rates[:, j] == rate) / 100 for rate in choices]
        ax.bar(np.arange(len(choices)) + (j - 1) * .25, dwell, width=.24, label=name)
    ax.set_xticks(range(len(choices)), labels=choices)
    ax.set(xlabel="Requested PID rate [Hz]", ylabel="Time [s]", title="Independent controller rates"); ax.legend(fontsize=8)
    ax = axes[1, 2]
    ax.hist((kp / base_p).ravel(), bins=np.linspace(.5, 1.5, 21), alpha=.6, label="P gain")
    ax.hist((ki / base_i).ravel(), bins=np.linspace(.5, 1.5, 21), histtype="step", lw=2, label="I gain")
    ax.set(xlabel="Multiplier of selected profile gain", ylabel="Joint/segment draws", title="Independent gain variation"); ax.legend()
    for ax in axes.flat: ax.grid(alpha=.15)
    figure.suptitle(f"Randomized IK/PID collection — synthetic {len(rows) / 100:g} s preview, seed {protocol['seed']}", fontsize=15)
    figure.savefig(root / "preview.png", dpi=170)
    plt.close(figure)
    bands = [(0, 1), (1, 3), (3, 6), (6, 9), (9, 12.000001)]
    report = {
        "synthetic_only": True,
        "collision_checked": False,
        "requested_segment_seconds_observed": [float(duration.min()), float(duration.max())],
        "gain_scale_p_observed": [float((kp / base_p).min()), float((kp / base_p).max())],
        "gain_scale_i_observed": [float((ki / base_i).min()), float((ki / base_i).max())],
        "pid_rate_choices_observed_hz": sorted(set(rates.ravel())),
        "measured_joint_range_deg": {name: [float(q[:, j].min()), float(q[:, j].max())] for j, name in enumerate(JOINTS)},
        "segment_kinds": {kind: sum(s["kind"] == kind for s in segments) for kind in ("coverage", "free", "reverse", "pitch", "hold")},
        "command_power_fraction_by_band": {
            mode: {f"{lo:g}-{min(hi, 12):g}Hz": float(power[(hz >= lo) & (hz < hi)].sum() / power.sum()) for lo, hi in bands}
            for mode, power in mean_spectra.items()
        },
        "step_work_us_percentiles": {str(p): float(np.percentile(values("step_work_us"), p)) for p in (50, 95, 99, 100)},
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
