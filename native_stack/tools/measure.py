"""Measure existing Linux processes; reports MiB and CPU percent of one core."""
import argparse
import json
import os
from pathlib import Path
import time


def snapshot(pid):
    root = Path("/proc") / str(pid)
    result = {}
    for line in (root / "smaps_rollup").read_text().splitlines():
        parts = line.split()
        if parts[0] in ("Rss:", "Pss:", "Swap:"):
            result[parts[0].strip(":").lower() + "_mib"] = int(parts[1]) / 1024
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    result["cpu_seconds"] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    result["pid"] = pid
    return result


def measure(pids, seconds=10, interval=.2):
    begin = time.monotonic()
    initial = {pid: snapshot(pid) for pid in pids}
    peaks = {pid: dict(data) for pid, data in initial.items()}
    latest = initial
    samples = 0
    while time.monotonic() - begin < seconds:
        time.sleep(interval)
        latest = {pid: snapshot(pid) for pid in pids}
        samples += 1
        for pid, data in latest.items():
            for name in ("rss_mib", "pss_mib", "swap_mib"):
                peaks[pid][name] = max(peaks[pid][name], data[name])
    elapsed = time.monotonic() - begin
    return {"elapsed_s": elapsed, "samples": samples, "processes": [
        {"pid": pid, "rss_start_mib": initial[pid]["rss_mib"], "rss_end_mib": latest[pid]["rss_mib"],
         "rss_peak_mib": peaks[pid]["rss_mib"], "pss_peak_mib": peaks[pid]["pss_mib"],
         "swap_peak_mib": peaks[pid]["swap_mib"],
         "cpu_percent_one_core": 100 * (latest[pid]["cpu_seconds"] - initial[pid]["cpu_seconds"]) / elapsed}
        for pid in pids]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pids", type=int, nargs="+")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("positive measurement duration required")
    report = json.dumps(measure(args.pids, args.seconds), indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
