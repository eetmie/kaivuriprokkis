#!/usr/bin/env python3
"""Measure a neutral MLP collection run; --live requires physical power off."""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def percentiles(values):
    values = sorted(values)
    if not values:
        return {}

    def at(fraction):
        index = (len(values) - 1) * fraction
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {"p50": at(.5), "p99": at(.99), "p99_9": at(.999), "max": values[-1]}


def summarize(root):
    lateness, work, intervals, device_intervals = [], [], [], []
    rows = packets = missed = dropped = high_water = sequence_gaps = 0
    checksums = invalid = 0
    sessions = sorted(root.glob("session_*.json"))
    for path in sessions:
        manifest = json.loads(path.read_text())
        if not manifest["complete"]:
            raise RuntimeError(f"Incomplete session: {path}")
        rows += manifest["rows"]
        packets += manifest["raw_packets"]
        missed = max(missed, manifest["missed_periods"])
        checksums = max(checksums, manifest["checksum_errors"])
        invalid = max(invalid, manifest["invalid_imu_frames"])
        dropped += manifest["dropped"]
        high_water = max(high_water, manifest["queue_high_water"])
        previous = None
        drive = next(name for name in manifest["outputs"] if name.startswith("drive_log_"))
        with (root / drive).open() as stream:
            for row in csv.DictReader(stream):
                lateness.append(float(row["lateness_ms"]))
                work.append(float(row["work_ms"]))
                stamp = int(row["post_write_ns"])
                if previous is not None:
                    intervals.append((stamp - previous) / 1e6)
                previous = stamp
        previous = previous_sequence = None
        raw = next(name for name in manifest["outputs"] if name.startswith("imu_raw_"))
        with (root / raw).open() as stream:
            for row in csv.DictReader(stream):
                stamp = int(row["device_ts_us"])
                sequence = int(row["sequence"])
                if previous is not None:
                    device_intervals.append((stamp - previous) / 1000)
                    sequence_gaps += max(0, sequence - previous_sequence - 1)
                previous, previous_sequence = stamp, sequence
    return {
        "sessions": len(sessions), "rows": rows, "raw_packets": packets,
        "missed_periods": missed, "dropped": dropped, "queue_high_water": high_water,
        "checksum_errors": checksums, "invalid_imu_frames": invalid,
        "raw_sequence_gaps": sequence_gaps,
        "device_intervals_over_7_5ms": sum(value > 7.5 for value in device_intervals),
        "lateness_ms": percentiles(lateness), "work_ms": percentiles(work),
        "post_write_interval_ms": percentiles(intervals),
        "device_interval_ms": percentiles(device_intervals),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--live", action="store_true", help="Use only with servo/pump power physically OFF")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if list(args.out.iterdir()):
        parser.error("--out must be empty so benchmark runs stay separate")
    command = [str(args.binary.resolve()), "--profile", str(args.profile.resolve()),
               "--serial", args.serial, "--out", str(args.out.resolve()),
               "--scripted", "--record-on-start", "--duration-s", str(args.seconds),
               "--record-seconds", str(args.seconds + 60), "--suffix", "bench_power_off"]
    if not args.live:
        command.append("--dry")
    resources = []
    started = time.monotonic()
    with (args.out / "collector.txt").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text()
                    rss = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
                    stat = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                    cpu = (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")
                    resources.append({"elapsed_s": time.monotonic() - started, "rss_kib": rss, "cpu_s": cpu})
                except (FileNotFoundError, ProcessLookupError, StopIteration):
                    pass
                time.sleep(1)
        except BaseException:
            process.terminate()
            process.wait(timeout=30)
            raise
    (args.out / "resources.json").write_text(json.dumps(resources) + "\n")
    if process.returncode:
        raise RuntimeError(f"Collector exited {process.returncode}; see {args.out / 'collector.txt'}")
    report = summarize(args.out)
    report.update(command=command, elapsed_s=time.monotonic() - started, live_pwm=args.live,
                  neutral_only=True)
    if resources:
        warm = [row for row in resources if row["elapsed_s"] >= 30] or resources
        first, last = warm[:60], warm[-60:]
        first_rss = statistics.median(row["rss_kib"] for row in first)
        last_rss = statistics.median(row["rss_kib"] for row in last)
        report["memory"] = {"first_warm_median_kib": first_rss, "last_median_kib": last_rss,
                            "growth_kib": last_rss - first_rss,
                            "min_kib": min(row["rss_kib"] for row in warm),
                            "max_kib": max(row["rss_kib"] for row in warm)}
        report["cpu_one_core_percent"] = 100 * resources[-1]["cpu_s"] / resources[-1]["elapsed_s"]
    (args.out / "bench_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
