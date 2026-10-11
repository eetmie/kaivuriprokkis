"""Hardware-free memory measurement on the current host, with two synthetic IR streams."""
import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source / "tools"))
from client import Client, DRIVE, DISARM, DEADMAN
from frame_reader import FrameReader
from measure import measure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-dir", type=Path, default=source / "build")
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--output", type=Path, default=source / "results/spark-synthetic.json")
args = parser.parse_args()
assert args.seconds > 0
stop = threading.Event()
errors = []
with tempfile.TemporaryDirectory(prefix="kaivuri-benchmark-") as temporary:
    temp = Path(temporary)
    control = subprocess.Popen([str(args.build_dir / "kaivuri-control"), "--simulate-imu", "--socket", str(temp / "control.sock"),
                                "--profile", str(source.parent / "configuration_files/profiles/jetson"), "--safety", str(source / "config/safety.yaml")], stderr=subprocess.PIPE)
    camera = subprocess.Popen([str(args.build_dir / "kaivuri-camera"), "--synthetic", "--camera", "bench1:1", "--camera", "bench2:2", "--output-dir", temporary], stderr=subprocess.PIPE)
    worker = None
    try:
        deadline = time.monotonic() + 3
        while not ((temp / "control.sock").exists() and (temp / "bench2-ir2.frame").exists()):
            if time.monotonic() > deadline or control.poll() is not None or camera.poll() is not None:
                raise RuntimeError("benchmark processes did not start")
            time.sleep(.01)
        with Client(temp / "control.sock") as client, FrameReader(temp / "bench1-ir1.frame", args.build_dir / "libkaivuri_frames.so") as reader1, FrameReader(temp / "bench2-ir2.frame", args.build_dir / "libkaivuri_frames.so") as reader2:
            client.arm()

            def publish():
                try:
                    while not stop.wait(.025):
                        client.send(DRIVE, [(0, .1, -.1, .1, 0, 0)], flags=DEADMAN)
                        reader1.read()
                        reader2.read()
                except BaseException as error:
                    errors.append(str(error))

            worker = threading.Thread(target=publish)
            worker.start()
            report = measure([control.pid, camera.pid], args.seconds)
            stop.set()
            worker.join()
            status = client.status()
            report.update(host=platform.node(), machine=platform.machine(), mode="dry PWM + simulated IMUs + two synthetic 640x480 Y8 IR streams", control_status=status, errors=errors)
            assert not errors and status["state"] == "armed", report
            report["processes"][0]["role"] = "control"
            report["processes"][1]["role"] = "synthetic_camera"
            client.send(DISARM)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    finally:
        stop.set()
        if worker:
            worker.join()
        for process in (control, camera):
            process.terminate()
            _, stderr = process.communicate(timeout=3)
            if process.returncode:
                print(stderr.decode(), file=sys.stderr)
