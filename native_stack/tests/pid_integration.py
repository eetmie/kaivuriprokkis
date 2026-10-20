#!/usr/bin/env python3
"""Separate PID collector: real byte replay and test-only intercepted I2C."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pty
import resource
import signal
import struct
import subprocess
import tempfile
import threading
import time

import yaml

from mlp_integration import descriptor
from mlp_lifecycle import check_neutral, wait_until

JOINTS = ("boom", "arm", "bucket")
CHANNELS = ("lift", "tilt", "scoop")


class Replay:
    def __init__(self, profile):
        self.profile = yaml.safe_load((profile / "control_config.yaml").read_text())
        self.q = [math.radians(v) for v in (-10, 86, -50)]
        self.enabled = True
        self.reset = False

    def packet(self, timestamp):
        imu = self.profile["imu"]
        absolute = [0.0]
        for q in self.q:
            absolute.append(absolute[-1] + q)
        sensors = [None] * 4
        for pitch, role in zip(absolute, ("base", *JOINTS)):
            mounting = imu["mounting_offsets_quat"][role]
            raw_pitch = pitch + 2 * math.atan2(mounting[2], mounting[0])
            sensors[imu["imu_mapping"][role]] = (math.cos(raw_pitch / 2), 0, math.sin(raw_pitch / 2), 0)
        body = bytearray(struct.pack("<BBI", 4, 4, timestamp & 0xFFFFFFFF))
        for quat in sensors:
            body.extend(struct.pack("<10f", *quat, 0, 0, 0, 0, 0, 1))
        return b"\xaa\x55" + body + struct.pack("<H", sum(body) & 0xFFFF)


@contextlib.contextmanager
def collector(args, root, extra=(), pad=None, full_disk=False):
    root.mkdir()
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    replay = Replay(args.profile)
    preload = ":".join(filter(None, (os.environ.get("KAIVURI_TEST_ASAN_RUNTIME"), str(args.device))))
    env = dict(os.environ, LD_PRELOAD=preload, KAIVURI_TEST_PWM_TRACE=str(root / "pwm.bin"))
    command = [str(args.binary), "--profile", str(args.profile), "--serial", os.ttyname(slave),
               "--out", str(root), "--seed", "42", "--duration-s", "15", *extra]
    command += ["--gamepad", str(pad)] if pad else ["--scripted"]
    def file_limit():
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (32768, 32768))
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=env, preexec_fn=file_limit if full_disk else None)
    stop = threading.Event()
    def feed():
        index, base, deadline = 0, 2**32 - 500000, time.monotonic()
        while not stop.is_set():
            if replay.reset:
                os.write(master, descriptor()); base = 1000 - index * 5000; replay.reset = False
            if replay.enabled:
                payload = bytearray(replay.packet(base + index * 5000) + replay.packet(base + (index + 1) * 5000))
                while payload and not stop.is_set():
                    try:
                        del payload[:os.write(master, payload)]
                    except BlockingIOError:
                        stop.wait(.001)
                    except OSError:
                        return
            index += 2; deadline += .01
            stop.wait(max(0, deadline - time.monotonic()))
    thread = threading.Thread(target=feed)
    try:
        line = process.stdout.readline()
        assert "PID collector" in line and "PWM LIVE" in line, (line, process.stderr.read())
        thread.start()
        time.sleep(.10)
        yield process, replay
    finally:
        stop.set()
        if thread.is_alive(): thread.join(timeout=2)
        if process.poll() is None: process.send_signal(signal.SIGINT)
        out, err = process.communicate(timeout=10)
        os.close(master); os.close(slave)
        assert process.returncode == (1 if full_disk else 0), (out, err)
        if not full_disk:
            assert "[PID] Joint coverage:" in out and "working-range cells" in out, out
    check_neutral(root / "pwm.bin")


def send(process, text):
    process.stdin.write(text + "\n"); process.stdin.flush()


def read_csv(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def verify(root):
    all_rows, all_pid, protocols = [], [], []
    sessions = sorted(root.glob("session_*.json"))
    assert sessions
    for path in sessions:
        m = json.loads(path.read_text())
        assert m["format"] == "kaivuri-native-mlp-v1" and m["collector"] == "kaivuri-pid-collect"
        assert m["complete"] and m["dropped"] == 0 and not m["dry_pwm"]
        assert len(m["outputs"]) == 3 and len(m["attachments"]) == 2
        for name, digest in {**m["outputs"], **m["attachments"]}.items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        drive = read_csv(root / next(n for n in m["outputs"] if n.startswith("drive_log_")))
        pid = read_csv(root / next(n for n in m["attachments"] if n.startswith("pid_control_")))
        raw = read_csv(root / next(n for n in m["outputs"] if n.startswith("imu_raw_")))
        protocol = json.loads((root / next(n for n in m["attachments"] if n.startswith("pid_protocol_"))).read_text())
        assert protocol["seed"] == 42 and not protocol["collision_checked"]
        protocols.append(protocol)
        assert len(drive) == len(pid) == m["rows"] and len(raw) == m["raw_packets"]
        assert len({r["sequence"] for r in raw}) == len(raw)
        assert all(int(a["device_ts_us"]) < int(b["device_ts_us"]) for a, b in zip(raw, raw[1:]))
        by_timestamp = {r["device_ts_us"]: r for r in raw}
        for d, p in zip(drive, pid):
            assert d["post_write_ns"] == p["post_write_ns"] and d["control_source"] == p["control_source"]
            for channel in CHANNELS:
                total = sum(float(d[f"{prefix}_{channel}"]) for prefix in
                            ("manual_cmd", "assist_cmd", "effective_excitation_cmd", "constraint_cmd"))
                assert abs(total - float(d[f"combined_cmd_{channel}"])) < 1e-12
                assert -1 <= float(d[f"combined_cmd_{channel}"]) <= 1
            if d["imu_valid"] == "1":
                packet = by_timestamp[d["state_imu_ts_us"]]
                assert int(packet["host_read_ns"]) <= int(d["post_write_ns"])
            if d["controller_valid"] == "0" or d["input_valid"] == "0" or p["guard_valid"] == "0":
                assert d["cmd_stale"] == "1"
                assert all(float(d[f"written_cmd_{j}"]) == 0 for j in JOINTS), {
                    k: d[k] for k in ("imu_valid", "input_valid", "controller_valid", "state_age_s", "written_cmd_boom", "written_cmd_arm", "written_cmd_bucket")
                }
            if d["imu_valid"] == "0":
                assert d["cmd_stale"] == "1"
                # Freshness can cross 30 ms during a write. Preserve the command
                # actually written and reject that row for training; later stale
                # ticks must be neutral, rather than falsifying the write record.
                if float(d["state_age_s"]) >= .04:
                    assert all(float(d[f"written_cmd_{j}"]) == 0 for j in JOINTS)
            if p["active"] == "1" and d["controller_valid"] == "1":
                for joint, (lo, hi) in zip(JOINTS, ((-40, 20), (38, 134), (-110, 10))):
                    ref = math.degrees(float(p[f"joint_ref_rad_{joint}"]))
                    assert lo - 1e-6 <= ref <= hi + 1e-6
                assert float(p["ik_position_error_m"]) < .0005 and float(p["ik_pitch_error_rad"]) < .005
                if p["recovery"] == "1":
                    assert p["segment_kind"] == "recovery" and d["sine_enabled"] == "0"
                    for joint, channel in zip(JOINTS, CHANNELS):
                        assert float(p[f"pid_rate_requested_hz_{joint}"]) == 100
                        assert abs(float(p[f"assist_requested_{joint}"])) <= .25
                        assert float(d[f"sine_cmd_{channel}"]) == 0
        all_rows.extend(drive); all_pid.extend(pid)
    return all_rows, all_pid, protocols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("binary", "profile", "device", "manual_binary"):
        parser.add_argument(field, type=lambda v: Path(v).resolve())
    parser.add_argument("--artifacts", type=Path, help="Keep a four-second fake-I2C training-compatibility fixture")
    args = parser.parse_args()
    if args.artifacts:
        with collector(args, args.artifacts, ["--record-on-start", "--suffix", "test_only_intercepted_i2c", "--chirp-seconds", "2"]) as (p, replay):
            send(p, "pump\nassist\nexcitation"); time.sleep(4)
        rows, _, _ = verify(args.artifacts)
        print(f"Saved {len(rows)} synthetic fake-I2C rows for importer validation at {args.artifacts}")
        return
    with tempfile.TemporaryDirectory(prefix="pid-collector-") as tmp:
        root = Path(tmp)
        run = root / "lifecycle"
        with collector(args, run, ["--record-on-start", "--chirp-seconds", "2"]) as (p, replay):
            second = subprocess.run([str(args.manual_binary), "--dry", "--scripted", "--serial", "/dev/null",
                                     "--out", str(root / "excluded")], capture_output=True, text=True)
            assert second.returncode and "another native controller" in second.stderr
            send(p, "pump\nassist\nexcitation"); time.sleep(.7)
            replay.enabled = False; time.sleep(.12); replay.enabled = True; time.sleep(.12)
            send(p, "assist\nexcitation"); time.sleep(.35)
            send(p, "manual .4 0 0"); time.sleep(.12)
            send(p, "manual 0 0 0"); time.sleep(.12)
            replay.q = [math.radians(v) for v in (19.5, 86, -50)]
            send(p, "manual .6 0 0"); time.sleep(.12)
            replay.q = [math.radians(v) for v in (21, 86, -50)]; time.sleep(.12)
            send(p, "manual -.4 0 0"); time.sleep(.12)
            replay.q = [math.radians(v) for v in (31, 86, -50)]; time.sleep(.12)
            send(p, "manual .4 0 0"); time.sleep(.12)
            replay.q = [math.radians(v) for v in (-10, 86, -50)]
            send(p, "manual 0 0 0"); time.sleep(.06)
            replay.reset = True; time.sleep(.15)
            send(p, "save")
            wait_until(lambda: len(list(run.glob("session_*.json"))) >= 2)
            time.sleep(.06)
            send(p, "start\npump\nassist"); time.sleep(.3)
        rows, pid, protocols = verify(run)
        assert len(rows) > 200 and sum(r["active"] == "1" for r in pid) > 60
        assert len(protocols) >= 3 and any(s["segments"] for s in protocols)
        assert any(r["imu_valid"] == "0" and r["pump_on"] == "1" for r in rows)
        near = [r for r in rows if math.radians(19.4) < float(r["joint_pos_boom"]) < math.radians(19.6)]
        assert near and all(0 <= float(r["combined_cmd_lift"]) <= .060001 for r in near)
        beyond = [r for r in rows if math.radians(20.9) < float(r["joint_pos_boom"]) < math.radians(21.1)]
        assert any(float(r["manual_cmd_lift"]) > 0 and float(r["combined_cmd_lift"]) == 0 for r in beyond)
        assert any(float(r["combined_cmd_lift"]) < -.39 for r in beyond)
        hard = [r for r in rows if float(r["joint_pos_boom"]) > math.radians(30.9)]
        assert hard and all(r["controller_valid"] == "1" and r["pump_on"] == "1" for r in hard if r["imu_valid"] == "1")
        assert any(float(r["combined_cmd_lift"]) < -.39 for r in hard)
        assert any(float(r["manual_cmd_lift"]) > 0 and float(r["combined_cmd_lift"]) == 0 for r in hard)
        assert any(r["control_source"] == "manual" and r["sine_enabled"] == "0" for r in rows)
        assert any(float(r["sine_cmd_lift"]) != 0 for r in rows)

        with collector(args, root / "recovery", ["--record-on-start"]) as (p, replay):
            replay.q = [math.radians(v) for v in (-50.15, 92.78, -119.2)]
            time.sleep(.1)
            send(p, "pump\nassist\nexcitation"); time.sleep(.3)
            send(p, "excitation"); time.sleep(.1)
            replay.q = [math.radians(v) for v in (-39.8, 92.78, -109.8)]; time.sleep(.15)
            replay.q = [math.radians(v) for v in (-39, 92.78, -109)]; time.sleep(.3)
        recovered, recovery_pid, recovery_protocol = verify(root / "recovery")
        recovering = [(d, c) for d, c in zip(recovered, recovery_pid) if c["recovery"] == "1"]
        assert len(recovering) >= 40
        parked = [(d, c) for d, c in recovering if float(d["joint_pos_boom"]) < math.radians(-50)]
        assert parked and all(d["controller_valid"] == "1" and d["pump_on"] == "1" for d, _ in parked)
        assert all(float(d["combined_cmd_lift"]) > 0 and float(d["combined_cmd_scoop"]) > 0 for d, _ in parked)
        assert all(c["coverage_cell"] == str(2**64 - 1) for _, c in parked)
        assert any(c["active"] == "1" and c["recovery"] == "0" and d["controller_valid"] == "1"
                   for d, c in zip(recovered[-25:], recovery_pid[-25:]))
        segments = recovery_protocol[0]["segments"]
        assert segments[0]["kind"] == "recovery" and segments[0]["duration_requested_s"] == 0
        assert any(s["kind"] != "recovery" for s in segments)

        fifo = root / "pad.fifo"; os.mkfifo(fifo)
        pad_fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        with collector(args, root / "disconnect", pad=fifo) as (p, replay):
            events = [(1, 304, 1), (1, 307, 1), (1, 315, 1), (1, 305, 1)]
            os.write(pad_fd, b"".join(struct.pack("llHHi", 0, 0, *event) for event in events))
            time.sleep(.3); os.close(pad_fd); time.sleep(.15)
        disconnected, _, _ = verify(root / "disconnect")
        lost = [r for r in disconnected if r["input_valid"] == "0"]
        assert len(lost) >= 8 and all(r["pump_on"] == "0" and r["sine_enabled"] == "0" for r in lost)

        with collector(args, root / "timeout", ["--record-on-start", "--record-seconds", ".2"]):
            time.sleep(.25)
        timed, _, _ = verify(root / "timeout")
        assert 15 <= len(timed) <= 24
        with collector(args, root / "disk", ["--record-on-start"], full_disk=True):
            time.sleep(.2)
        assert list((root / "disk").glob("*.partial")) and not list((root / "disk").glob("session_*.json"))
        for option, value in (("--seed", "-1"), ("--duration-s", "nan"), ("--excitation", "stroke"),
                              ("--queue-records", "17"), ("--excitation-target", "slew")):
            bad = subprocess.run([str(args.binary), "--dry", "--scripted", "--out", str(root / "invalid"), option, value], capture_output=True)
            assert bad.returncode and not (root / "invalid").exists()
    print("PID collector: bounded references/combined commands, takeover, stale/reset input, commits, exclusive ownership and neutral PWM passed")


if __name__ == "__main__": main()
