"""Bench C++ pi0.5 with two real IR cameras and disarmed native control.

Only owns the processes it launches. No DRIVE/ARM packets (kaivuri-inference runs
without --drive). Optional live PWM writes neutral only; otherwise the controller
uses dry PWM. A pi05_openpi_state bundle reads its joint-angle state from control.
Memory samples include load and warm-up; steady samples are marked separately.
"""
import argparse
import csv
import datetime
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import threading
import time

from client import Client


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    def percentile(p):
        pos = p * (len(values) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    mean = sum(values) / len(values)
    return {"count": len(values), "min": values[0], "p50": percentile(.5),
            "p95": percentile(.95), "p99": percentile(.99), "max": values[-1],
            "mean": mean,
            "stddev": math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))}


def process_snapshot(pid):
    root = Path("/proc") / str(pid)
    result = {}
    for line in (root / "smaps_rollup").read_text().splitlines():
        parts = line.split()
        if parts[0] in ("Rss:", "Pss:", "Swap:"):
            result[parts[0][:-1].lower() + "_mib"] = int(parts[1]) / 1024
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    result["cpu_seconds"] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    result["threads"] = int(fields[17])
    return result


def system_snapshot():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        info[parts[0][:-1]] = int(parts[1])
    return {"used_mib": (info["MemTotal"] - info["MemAvailable"]) / 1024,
            "available_mib": info["MemAvailable"] / 1024,
            "swap_used_mib": (info["SwapTotal"] - info["SwapFree"]) / 1024}


def summarize_camera(path, window):
    by_serial = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            arrival = int(row["host_monotonic_ns"])
            if window and not window[0] <= arrival <= window[1]:
                continue
            by_serial.setdefault(row["serial"], []).append(row)
    report = {}
    for serial, rows in by_serial.items():
        arrivals = [int(r["host_monotonic_ns"]) for r in rows]
        intervals = [(b - a) / 1e6 for a, b in zip(arrivals, arrivals[1:])]
        counters = [int(r["frame_counter"]) for r in rows]
        seq = [int(r["sequence"]) for r in rows]
        report[serial] = {
            "frames_logged": len(rows), "ir_index": int(rows[0]["ir_index"]),
            "observed_fps": (len(rows) - 1) * 1e9 / (arrivals[-1] - arrivals[0]) if len(rows) > 1 else 0,
            "arrival_interval_ms": distribution(intervals),
            "device_counter_gaps": sum(max(0, b - a - 1) for a, b in zip(counters, counters[1:])),
            "unexplained_device_counter_gaps": sum(max(0, counters[i + 1] - counters[i] - (seq[i + 1] - seq[i])) for i in range(len(rows) - 1)),
            "device_counter_resets": sum(b <= a for a, b in zip(counters, counters[1:])),
            "logger_sequence_gaps": sum(max(0, b - a - 1) for a, b in zip(seq, seq[1:])),
            "metadata_bits": sorted({int(r["valid_bits"]) for r in rows}),
            "exposure_us": distribution([int(r["exposure_us"]) for r in rows]),
            "timestamp_domains": sorted({int(r["domain"]) for r in rows}),
        }
    return report


def summarize(output, samples):
    inference = json.loads((output / "inference.json").read_text())
    with (output / "inference.csv").open() as handle:
        timings = list(csv.DictReader(handle))
    window = (int(timings[0]["start_ns"]), int(timings[-1]["finish_ns"]))
    metrics = {key: distribution([float(row[key]) for row in timings]) for key in (
        "preprocess_ms", "vision_ms", "language_ms", "denoise_ms", "inference_ms",
        "iteration_ms", "observation_age_start_ms", "observation_to_result_ms", "camera_arrival_skew_ms",
        "state_age_ms", "prompt_ms") if key in timings[0]}
    starts = [int(row["start_ns"]) for row in timings]
    metrics["inference_start_interval_ms"] = distribution([(b - a) / 1e6 for a, b in zip(starts, starts[1:])])
    for key in ("rate_wait_ms", "rate_lateness_ms"):
        if key in timings[0]:
            metrics[key] = distribution([float(row[key]) for row in timings])
    with (output / "control-timing.csv").open() as handle:
        control = [row for row in csv.DictReader(handle) if window[0] <= int(row["start_ns"]) <= window[1]]
    metrics["control_lateness_ms"] = distribution([float(r["lateness_ms"]) for r in control])
    metrics["control_work_ms"] = distribution([float(r["work_ms"]) for r in control])
    if control and "fk_work_us" in control[0]:
        metrics["fk_work_us"] = distribution([float(r["fk_work_us"]) for r in control])
    control_starts = [int(r["start_ns"]) for r in control]
    metrics["control_start_interval_ms"] = distribution([(b - a) / 1e6 for a, b in zip(control_starts, control_starts[1:])])
    steady = [s for s in samples if window[0] <= s["monotonic_ns"] <= window[1]]
    memory = {}
    for name in ("control", "camera", "inference"):
        records = [(s["monotonic_ns"], s["processes"][name]) for s in steady if name in s["processes"]]
        if records:
            elapsed = (records[-1][0] - records[0][0]) / 1e9
            values = [r[1] for r in records]
            memory[name] = {"rss_mib": distribution([r["rss_mib"] for r in values]),
                            "pss_mib": distribution([r["pss_mib"] for r in values]),
                            "swap_mib": distribution([r["swap_mib"] for r in values]),
                            "rss_start_mib": values[0]["rss_mib"], "rss_end_mib": values[-1]["rss_mib"],
                            "cpu_percent_one_core": 100 * (values[-1]["cpu_seconds"] - values[0]["cpu_seconds"]) / elapsed if elapsed else 0}
    statuses = [s["control_status"] for s in steady if "control_status" in s]
    camera_report = summarize_camera(output / "camera-timestamps.csv", window)
    for match in re.finditer(r"serial=(\d+) received=(\d+) busy_drops=(\d+) counter_gaps=(\d+) counter_resets=(\d+)", (output / "camera.log").read_text()):
        if match[1] in camera_report:
            camera_report[match[1]]["capture_totals_including_load"] = {
                "received": int(match[2]), "writer_busy_drops": int(match[3]),
                "device_counter_gaps": int(match[4]), "device_counter_resets": int(match[5])}
    report = {"inference": inference, "metrics": metrics, "processes": memory,
              "system_used_mib": distribution([s["system"]["used_mib"] for s in steady]),
              "system_available_mib": distribution([s["system"]["available_mib"] for s in steady]),
              "system_swap_used_mib": distribution([s["system"]["swap_used_mib"] for s in steady]),
              "cameras": camera_report,
              "control": {"states": sorted({s["state"] for s in statuses}),
                          "imu_healthy_samples": sum(s["imu_healthy"] for s in statuses),
                          "samples": len(statuses), "last_status": statuses[-1] if statuses else None},
              "observation_time_basis": "host monotonic camera arrival; exposure clocks are not mapped",
              "camera_alignment": "standalone unsynchronized cameras; arrival skew is not exposure skew"}
    status_log = output / "fk-status.jsonl"
    if status_log.exists():
        count = invalid = unhealthy = 0
        first = last = None
        roundtrips = []
        ticks, imu_sequences = set(), set()
        with status_log.open() as handle:
            for line in handle:
                row = json.loads(line)
                if not window[0] <= row["received_ns"] <= window[1]:
                    continue
                status = row["status"]
                count += 1
                first = row["received_ns"] if first is None else first
                last = row["received_ns"]
                invalid += not status["fk_valid"]
                unhealthy += not status["imu_healthy"]
                ticks.add(status["ticks"])
                imu_sequences.add(status["imu_sequence"])
                roundtrips.append((row["received_ns"] - row["sent_ns"]) / 1e6)
        report["high_rate_status"] = {"samples": count, "invalid_fk": invalid,
            "unhealthy_imu": unhealthy, "distinct_ticks": len(ticks),
            "distinct_imu_samples": len(imu_sequences),
            "received_hz": (count - 1) * 1e9 / (last - first) if count > 1 else 0,
            "roundtrip_ms": distribution(roundtrips)}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run(args):
    stack = Path(__file__).resolve().parents[1]
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build = args.build.resolve()
    children, handles = {}, []
    def start(name, command):
        log = (output / (name + ".log")).open("w")
        handles.append(log)
        child = subprocess.Popen([str(v) for v in command], cwd=stack, stdout=log, stderr=subprocess.STDOUT)
        children[name] = child
        return child
    camera_dir = output / "frames"
    socket_path = f"/tmp/kaivuri-native-{os.geteuid()}/pi05-bench.sock"
    control = [build / "kaivuri-control", "--socket", socket_path,
               "--control-hz", str(args.control_hz),
               "--duration-s", str(args.duration_s + 600),
               "--timing-csv", output / "control-timing.csv"]
    control += ["--simulate-imu"] if args.simulate_imu else ["--serial", args.serial]
    if args.live_pwm:
        control += ["--live-pwm"]
    if args.rt_priority:
        control += ["--rt-priority", str(args.rt_priority)]
    camera = [build / "kaivuri-camera", "--output-dir", camera_dir,
              "--metadata-csv", output / "camera-timestamps.csv"]
    for spec in args.camera:
        camera += ["--camera", spec]
    for serial in args.rotate_180:
        camera += ["--rotate-180", serial]
    command = [build / "kaivuri-inference", "--bundle", args.bundle.resolve(),
               "--control-socket", socket_path,
               "--cache", args.cache.resolve(), "--fixtures", args.fixtures.resolve(),
               "--task", args.task, "--duration-s", str(args.duration_s),
               "--warmup", str(args.warmup), "--out", output / "inference.json"]
    if args.rate_hz:
        command += ["--rate-hz", str(args.rate_hz)]
    if args.parity_cosine is not None:
        command += ["--parity-cosine", str(args.parity_cosine), "--parity-max-abs", str(args.parity_max_abs)]
    frame_files = []
    for spec in args.camera:
        serial, index = spec.split(":")
        path = camera_dir / f"{serial}-ir{index}.frame"
        frame_files.append(path)
        command += ["--camera-frame", path]
    started = time.monotonic()
    client = None
    samples = []
    successful = False
    initial_memory = system_snapshot()
    status_stop = threading.Event()
    status_errors = []
    status_thread = None

    def record_status():
        try:
            # Stream complete observations to disk; do not retain high-rate JSON in RAM.
            with Client(socket_path) as reader, (output / "fk-status.jsonl").open("w") as log:
                next_read = time.monotonic()
                while not status_stop.is_set():
                    sent = time.monotonic_ns()
                    observation = reader.status()
                    received = time.monotonic_ns()
                    if observation["state"] != "disarmed" or observation["pump_on"]:
                        raise RuntimeError("status stress requires disarmed control and pump off")
                    if observation["fk_valid"]:
                        for key in ("fk_joint_xyz_m", "fk_joint_xyz_upper_m", "fk_link_quat_wxyz",
                                    "fk_link_quat_upper_wxyz", "imu_quat_wxyz", "imu_corrected_quat_wxyz"):
                            values = observation[key]
                            if len(values) != 4 or not all(math.isfinite(v) for row in values for v in row):
                                raise RuntimeError(f"invalid kinematics in {key}")
                        if observation["fk_link_quat_wxyz"][-1] != observation["fk_quat_wxyz"]:
                            raise RuntimeError("incoherent link/tool quaternion")
                    log.write(json.dumps({"sent_ns": sent, "received_ns": received,
                                          "status": observation}) + "\n")
                    next_read += 1 / args.status_hz
                    if next_read < time.monotonic() - 1 / args.status_hz:
                        next_read = time.monotonic()
                    status_stop.wait(max(0, next_read - time.monotonic()))
        except BaseException as error:
            status_errors.append(str(error))

    try:
        start("control", control)
        start("camera", camera)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for name, child in children.items():
                if child.poll() is not None:
                    raise RuntimeError(f"{name} exited during startup; see {name}.log")
            if all(p.exists() for p in frame_files) and Path(socket_path).exists():
                client = Client(socket_path)
                break
            time.sleep(.1)
        if client is None:
            raise TimeoutError("camera/control startup timeout")
        status = client.status()
        if status["live_pwm"] != args.live_pwm or status["state"] != "disarmed":
            raise RuntimeError("bench requires disarmed control in the requested PWM mode")
        if args.status_hz:
            status_thread = threading.Thread(target=record_status, name="fk-status-recorder")
            status_thread.start()
        inference = start("inference", command)
        if shutil.which("tegrastats"):
            start("tegrastats", ["tegrastats", "--interval", "500"])
        telemetry = (output / "telemetry.jsonl").open("w")
        handles.append(telemetry)
        while inference.poll() is None:
            if status_errors:
                raise RuntimeError("status recorder failed: " + status_errors[0])
            if time.monotonic() - started > args.duration_s + 600:
                raise TimeoutError("inference load/bench timeout")
            record = {"monotonic_ns": time.monotonic_ns(), "system": system_snapshot(), "processes": {}}
            for name in ("control", "camera", "inference"):
                if children[name].poll() is not None:
                    if name != "inference":
                        raise RuntimeError(f"{name} exited during bench; see {name}.log")
                    continue
                try:
                    record["processes"][name] = process_snapshot(children[name].pid)
                except (FileNotFoundError, ProcessLookupError):
                    if children[name].poll() is None:
                        raise
            record["control_status"] = client.status()
            if record["control_status"]["live_pwm"] != args.live_pwm or record["control_status"]["state"] != "disarmed":
                raise RuntimeError("control entered unexpected state or PWM mode")
            samples.append(record)
            telemetry.write(json.dumps(record) + "\n")
            telemetry.flush()
            time.sleep(.5)
        if inference.returncode:
            raise RuntimeError(f"inference failed ({inference.returncode}); see inference.log")
        if status_errors:
            raise RuntimeError("status recorder failed: " + status_errors[0])
        successful = True
    finally:
        status_stop.set()
        if status_thread:
            status_thread.join(timeout=3)
        if client:
            client.close()
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for child in children.values():
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for handle in handles:
            handle.close()
        (output / "session.json").write_text(json.dumps({
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command": [str(v) for v in command], "control_command": [str(v) for v in control],
            "camera_command": [str(v) for v in camera],
            "initial_system_memory": initial_memory,
            "process_exit_codes": {n: c.returncode for n, c in children.items()},
            "completed": successful}, indent=2) + "\n")
    report = summarize(output, samples)
    print(json.dumps({"output": str(output), "samples": report["inference"]["samples"],
                      "inference_ms": report["metrics"]["inference_ms"],
                      "control_lateness_ms": report["metrics"]["control_lateness_ms"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--camera", action="append", required=True, help="SERIAL:IR_INDEX, twice")
    parser.add_argument("--rotate-180", action="append", default=[], metavar="SERIAL",
                        help="kaivuri-camera rotation; must match the bundle's training cameras")
    parser.add_argument("--serial", default="/dev/ttyACM0")
    parser.add_argument("--simulate-imu", action="store_true")
    parser.add_argument("--live-pwm", action="store_true", help="disarmed neutral writes to PCA9685; real IMUs required")
    parser.add_argument("--rt-priority", type=int, default=0)
    parser.add_argument("--control-hz", type=float, default=100)
    parser.add_argument("--status-hz", type=float, default=0, help="stream full FK/IMU status to disk at this target rate")
    parser.add_argument("--rate-hz", type=float, default=0, help="optional inference limit; waits before reading fresh camera frames")
    parser.add_argument("--task", default="pick up the red block")
    parser.add_argument("--duration-s", type=float, default=300)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--parity-cosine", type=float, help="fixture gate override (pi05_openpi_state); needs --parity-max-abs")
    parser.add_argument("--parity-max-abs", type=float)
    args = parser.parse_args()
    if len(args.camera) != 2 or any(len(c.split(":")) != 2 for c in args.camera):
        parser.error("select two cameras using SERIAL:IR_INDEX")
    if not 0 < args.duration_s <= 3600 or args.warmup < 0 or not 0 <= args.rt_priority <= 80:
        parser.error("bounded duration, nonnegative warmup and priority 0..80 required")
    if args.live_pwm and args.simulate_imu:
        parser.error("live PWM cannot use simulated IMUs")
    if not 50 <= args.control_hz <= 500 or not 0 <= args.status_hz <= 500:
        parser.error("control rate must be 50..500 Hz and status rate 0..500 Hz")
    if not 0 <= args.rate_hz <= 500 or 0 < args.rate_hz < .001:
        parser.error("inference rate must be 0 (unrestricted) or 0.001..500 Hz")
    run(args)
