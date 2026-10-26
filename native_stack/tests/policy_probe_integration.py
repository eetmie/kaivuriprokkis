"""Exercise capture against real native IPC/shared memory with simulated inputs only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

build = Path(sys.argv[1]).resolve()
source = Path(__file__).resolve().parents[1]
probe = source / "tools/policy_probe.py"

with tempfile.TemporaryDirectory(prefix="probe-test-") as tmp:
    root = Path(tmp)
    task = root / "task.txt"
    task.write_text("scoop the sand")
    out = root / "history"
    command = [sys.executable, "-B", str(probe), "capture", "--synthetic",
               "--build", str(build), "--library", str(build / "libkaivuri_frames.so"),
               "--out", str(out), "--task-file", str(task), "--duration-s", "4",
               "--history-ms", "500", "--history-ms", "1000", "--rate-hz", "4"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not list(out.glob("*.ready")) and process.poll() is None:
            assert time.monotonic() < deadline, "first temporal observation timed out"
            time.sleep(.02)
        task.write_text("put it into the container instead")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        summary = json.loads((out / "session.json").read_text())
        assert summary["status"] == "complete" and not summary["actuator_output"]
        assert summary["snapshots"] >= 5, summary
        prompts = set()
        for sample in sorted(out.glob("*.ready")):
            record = json.loads((sample / "observation.json").read_text())
            prompts.add(record["task"])
            assert record["controller_status"]["state"] == "disarmed"
            assert not record["controller_status"]["live_pwm"]
            assert not record["actuator_output"]
            assert [s["requested_offset_ns"] for s in record["slots"]] == [1_000_000_000, 500_000_000, 0]
            current = record["slots"][-1]["views"]
            assert record["observation_ns"] == min(
                record["state"]["imu_host_monotonic_ns"],
                *(v["metadata"]["host_monotonic_ns"] for v in current))
            for slot in record["slots"]:
                for i, view in enumerate(slot["views"]):
                    assert hashlib.sha256((sample / view["file"]).read_bytes()).hexdigest() == view["sha256"]
                    delta = view["actual_offset_ns"] - slot["requested_offset_ns"]
                    assert 0 <= delta <= 60_000_000, delta
                    if slot["requested_offset_ns"]:
                        assert view["metadata"]["sequence"] < current[i]["metadata"]["sequence"]
        assert prompts == {"scoop the sand", "put it into the container instead"}, prompts
        assert not list(out.glob("*.partial"))
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=8)

    # Current-only mode is supported without inventing a second frame.
    out2 = root / "current"
    run = subprocess.run([sys.executable, "-B", str(probe), "capture", "--synthetic",
                          "--build", str(build), "--library", str(build / "libkaivuri_frames.so"),
                          "--out", str(out2), "--task", "another arbitrary instruction",
                          "--duration-s", ".5", "--history-ms", "0"], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, (run.stdout, run.stderr)
    r = json.loads((next(out2.glob("*.ready")) / "observation.json").read_text())
    assert len(r["slots"]) == 1
print("native observation/history integration passed")
