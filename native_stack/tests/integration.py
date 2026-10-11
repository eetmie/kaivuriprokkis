"""Linux process/IPC tests. Uses dry PWM and simulated or PTY IMUs, never hardware."""
import json
import os
from pathlib import Path
import pty
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time

binary, source = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(source / "tools"))
from client import Client, DRIVE, DISARM, ESTOP, DEADMAN, PUMP
from frame_reader import FrameReader


def wait_until(function, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = function()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError("condition timed out")


def launch(temp, *options):
    path = Path(temp) / "control.sock"
    process = subprocess.Popen([str(binary), "--profile", str(source.parent / "configuration_files/profiles/jetson"),
                                "--safety", str(source / "config/safety.yaml"), "--socket", str(path), *options],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_until(lambda: path.exists() or process.poll() is not None)
        assert process.poll() is None, process.communicate()
    except BaseException:
        process.terminate()
        process.communicate(timeout=3)
        raise
    return process, path


def finish(process):
    process.terminate()
    stdout, stderr = process.communicate(timeout=3)
    assert process.returncode == 0, (process.returncode, stdout, stderr)


def control_test(temp):
    process, path = launch(temp, "--simulate-imu")
    try:
        with Client(path) as owner, Client(path) as other:
            assert owner.status()["state"] == "disarmed"
            owner.arm(pump=True)
            owner.send(DRIVE, [(1, .1, 0, 0, 1, 1)], flags=DEADMAN | PUMP)
            time.sleep(.03)
            status = owner.status()
            assert status["state"] == "armed" and status["pump_on"]
            assert status["pulses_us"][0] == 1500 and status["pulses_us"][2] == 1637.5
            other.send(DRIVE, [(0, -1, 0, 0, 0, 0)], flags=DEADMAN)
            time.sleep(.02)
            assert owner.status()["pump_on"], "second producer must not overwrite owner"
            time.sleep(.55)
            status = owner.status()
            assert status["state"] == "decaying" and not status["pump_on"]
            time.sleep(.3)
            assert owner.status()["state"] == "fault"
            owner.send(DISARM)
            owner.arm()
            other.send(ESTOP)
            status = wait_until(lambda: (s if (s := owner.status())["state"] == "fault" else None))
            assert status["state"] == "fault" and status["reason"] == "emergency_stop"
            owner.send(DISARM)
            owner.arm()
            # Replay exact sequence as an invalid packet from the owning connection.
            owner.sequence -= 1
            owner.send(DRIVE, [(0,) * 6], flags=DEADMAN)
            assert owner.status()["reason"] == "invalid_or_replayed_packet"
            owner.send(DISARM)
            owner.arm()
        with Client(path) as status_client:
            status = wait_until(lambda: (s if (s := status_client.status())["state"] == "fault" else None))
            assert status["reason"] == "producer_disconnected"
            assert not status["pump_on"]
        # A paused control process detects its own missed deadline when resumed.
        os.kill(process.pid, signal.SIGSTOP)
        time.sleep(.08)
        os.kill(process.pid, signal.SIGCONT)
        with Client(path) as client:
            assert client.status()["reason"] == "control_deadline_missed"
    finally:
        finish(process)


def serial_test(temp):
    master, slave = pty.openpty()
    stop = threading.Event()

    def stream():
        timestamp = 1000
        while not stop.wait(.01):
            payload = bytearray(b"\xaa\x55\x04\x04") + struct.pack("<I", timestamp)
            for _ in range(4):
                payload += struct.pack("<10f", 1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
            payload += struct.pack("<H", sum(payload[2:]) & 65535)
            os.write(master, payload)
            timestamp += 10000

    thread = threading.Thread(target=stream)
    thread.start()
    process = None
    try:
        process, path = launch(temp, "--serial", os.ttyname(slave))
        with Client(path) as client:
            wait_until(lambda: client.status()["imu_healthy"])
            client.arm()
            stop.set()
            thread.join()
            time.sleep(.15)
            status = client.status()
            assert status["state"] == "fault" and status["reason"] == "imu_unhealthy"
            assert not status["pump_on"] and status["imu_sequence"] > 20
    finally:
        stop.set()
        thread.join()
        if process:
            finish(process)
        os.close(master)
        os.close(slave)


def camera_test(temp):
    camera = binary.parent / "kaivuri-camera"
    if not camera.exists():
        return
    process = subprocess.Popen([str(camera), "--synthetic", "--camera", "test1:1", "--camera", "test2:2",
                                "--output-dir", temp, "--metadata-csv", str(Path(temp) / "metadata.csv")],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_until(lambda: (Path(temp) / "test2-ir2.frame").exists())
        with FrameReader(Path(temp) / "test1-ir1.frame", binary.parent / "libkaivuri_frames.so") as reader:
            snapshot = wait_until(reader.read)
            metadata, pixels = snapshot
            assert metadata.valid_metadata == 31 and metadata.width == 640 and metadata.height == 480
            assert all(p == metadata.sequence % 256 for p in pixels), "coherent shared pixel snapshot"
            finish(process)
            process = None
            time.sleep(.25)
            assert reader.read() is None, "camera death must make the retained frame stale"
        assert len((Path(temp) / "metadata.csv").read_text().splitlines()) >= 3
    finally:
        if process:
            finish(process)


with tempfile.TemporaryDirectory(prefix="kaivuri-tests-") as temp:
    control_test(temp)
    serial_test(temp)
    camera_test(temp)
print("IPC ownership, watchdog, deadline, replay, PTY IMU, and camera freshness tests passed")
