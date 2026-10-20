#!/usr/bin/env python3
"""Compare the new native task adapter/PID against the actual existing Python math."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    probe, *profiles = sys.argv[1:]
    pid_module = module("reference_pid", ROOT / "modules/pid.py")
    coverage = module("reference_coverage", ROOT / "data_collection/plot_coverage.py")
    maxima = {"fk_m": 0.0, "pitch_rad": 0.0, "pid_output": 0.0, "pid_integral": 0.0, "pid_derivative": 0.0}
    counts = {"fk": 0, "pid": 0}
    for profile in profiles:
        geometry = coverage.load_profile(Path(profile) / "control_config.yaml")
        controllers = {}
        raw = subprocess.check_output([probe, profile], text=True)
        for line in raw.splitlines():
            row = json.loads(line)
            counts[row["type"]] += 1
            if row["type"] == "fk":
                xyz, pitch = coverage.forward_kinematics(row["q"], geometry)
                # Plotter includes offset0; native upper frame starts at the slew pivot.
                xyz = xyz[0] - geometry["offsets"][0]
                maxima["fk_m"] = max(maxima["fk_m"], abs(xyz[0] - row["pose"][0]), abs(xyz[2] - row["pose"][1]))
                delta = np.deg2rad(pitch[0]) - row["pose"][2]
                maxima["pitch_rad"] = max(maxima["pitch_rad"], abs((delta + np.pi) % (2 * np.pi) - np.pi))
            else:
                scenario = row["scenario"]
                if scenario not in controllers:
                    kp, ki, kd, tau = row["gains"]
                    controllers[scenario] = pid_module.PIDController(kp, ki, kd, -.5, .5, tau)
                p = controllers[scenario]
                value = p.compute(row["target"], row["measured"], row["dt"])
                for key, a, b in (("pid_output", value, row["output"]),
                                  ("pid_integral", p.integral_sum, row["integral"]),
                                  ("pid_derivative", p.d_filt, row["derivative"])):
                    maxima[key] = max(maxima[key], abs(a - b))
    assert all(value < 2e-12 for value in maxima.values()), maxima
    print(json.dumps({"samples": counts, "max_abs_error": maxima}, indent=2))


if __name__ == "__main__":
    main()
