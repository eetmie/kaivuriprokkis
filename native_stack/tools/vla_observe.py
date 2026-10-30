"""Observation side of vla_live.py, shared by every model: the controller and camera services and
one coherent snapshot of both IR cameras and the controller status."""
from __future__ import annotations

import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

CAM1, CAM2 = "233522073919", "141722076647"        # digging_native_640: cam1 = ir2, rotated 180


def start_services(a, runtime: Path, procs: list):
    """Start the controller and cameras. Each child joins `procs` as soon as it exists, so the
    caller can stop whatever started when a later step fails."""
    build = Path(a.build)
    control = [build / "kaivuri-control", "--profile", a.profile, "--safety", a.safety,
               "--socket", runtime / "control.sock", "--control-hz", str(a.control_hz),
               "--serial", a.serial]
    if a.live:
        control.append("--live-pwm")
    if a.enable_slew:
        control.append("--enable-slew")
    cams = [build / "kaivuri-camera", "--camera", f"{CAM1}:2", "--camera", f"{CAM2}:1",
            "--output-dir", runtime / "cameras", "--fps", "30", "--width", "640", "--height", "480",
            "--rotate-180", CAM1]
    log = open(runtime / "services.log", "w")
    for cmd, ready in ((control, [runtime / "control.sock"]),
                       (cams, [runtime / "cameras" / f"{CAM1}-ir2.frame", runtime / "cameras" / f"{CAM2}-ir1.frame"])):
        # Own session: a terminal Ctrl+C reaches only this script, which stops them in order.
        p = subprocess.Popen([str(x) for x in cmd], start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        end = time.time() + 30
        while not all(r.exists() for r in ready):
            if p.poll() is not None or time.time() > end:
                raise RuntimeError(f"{cmd[0].name} did not start (see {runtime / 'services.log'})")
            time.sleep(0.1)
    time.sleep(1.5)       # USB streams start one after the other
    return runtime / "control.sock", [runtime / "cameras" / f"{CAM1}-ir2.frame",
                                      runtime / "cameras" / f"{CAM2}-ir1.frame"]


def stop_services(procs: list) -> None:
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)


# ── observation ─────────────────────────────────────────────────────────────


class HistoryFilling(RuntimeError):
    """A frame-history model has no earlier frame near its lag yet; this one was stored, no plan."""


class Incoherent(RuntimeError):
    """This snapshot cannot condition a plan; the loop observes again. `kind` groups the log."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}{' (' + detail + ')' if detail else ''}")
        self.kind = kind


class Observer:
    """One coherent snapshot of both cameras and the controller status.

    Each source is read, then the whole set is validated at ONE final monotonic time: every frame
    fresh and not from the future, the expected geometry and IR stream, no sequence going
    backwards (a restarted camera), camera-to-camera skew and state-to-image separation bounded,
    the IMU healthy, fresh and not from the future. The observation time is the oldest source.
    """

    def __init__(self, readers, files, status_conn, a):
        self.readers, self.status_conn, self.a = readers, status_conn, a
        self.ir = [int(m.group(1)) if (m := re.search(r"-ir(\d)\.frame$", str(f))) else None for f in files]
        self.last_seq = [None] * len(readers)

    def __call__(self):
        a = self.a
        frames, metas = [], []
        for r in self.readers:
            snap, end = None, time.monotonic() + 0.5
            while snap is None:
                snap = r.read()
                if snap is None:
                    if time.monotonic() > end:
                        raise Incoherent("no fresh camera frame")
                    time.sleep(0.002)
            m, px = snap
            frames.append(np.frombuffer(px, np.uint8).reshape(m.height, m.width).copy())
            metas.append((m.sequence, m.host_monotonic_ns, m.width, m.height, m.ir_index))
        st = self.status_conn.status()
        now = time.monotonic_ns()
        cams = []
        for k, (seq, t, w, h, ir) in enumerate(metas):
            age = (now - t) / 1e6
            if age < 0:
                raise Incoherent("frame from the future", f"cam{k + 1} {age:.1f} ms")
            if age > a.max_frame_age_ms:
                raise Incoherent("stale frame", f"cam{k + 1} {age:.0f} ms at the snapshot")
            if (w, h) != (640, 480) or (self.ir[k] is not None and ir != self.ir[k]):
                raise Incoherent("wrong camera stream", f"cam{k + 1} is {w}x{h} ir{ir}, expected 640x480 ir{self.ir[k]}")
            if self.last_seq[k] is not None and seq < self.last_seq[k]:
                was, self.last_seq = self.last_seq[k], [None] * len(self.readers)
                raise Incoherent("camera restarted", f"cam{k + 1} sequence {was} -> {seq}")
            cams.append(t)
        skew = (max(cams) - min(cams)) / 1e6
        if skew > a.max_camera_skew_ms:
            raise Incoherent("camera skew", f"{skew:.1f} ms")
        if not (st["imu_healthy"] and st["fk_valid"]):
            raise Incoherent("IMU unhealthy / FK invalid")
        imu = st["imu_host_monotonic_ns"]
        imu_age = (now - imu) / 1e6
        if imu_age < -1.0:
            raise Incoherent("IMU from the future", f"{imu_age:.1f} ms")
        if imu_age > a.max_imu_age_ms:
            raise Incoherent("stale IMU", f"{imu_age:.0f} ms")
        sep = max(abs(imu - t) for t in cams) / 1e6
        if sep > a.max_state_skew_ms:
            raise Incoherent("state-to-image separation", f"{sep:.1f} ms")
        self.last_seq = [m[0] for m in metas]
        sources = {"snapshot_ns": now, "imu_ns": imu, "cameras": [
            {"seq": int(m[0]), "ns": int(m[1])} for m in metas]}
        return frames, st, min([*cams, imu]), skew, sources


def frame_ns(sources: dict) -> int:
    """An observation's image time for the frame history: its oldest camera frame."""
    return min(c["ns"] for c in sources["cameras"])
