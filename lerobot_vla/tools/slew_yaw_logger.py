#!/usr/bin/env python3
"""Slew-yaw validation logger: every raw IMU, the D435i gyro and the IR camera.

Records what is needed to check the slew yaw against two independent references,
without touching record_episodes.py or the LeRobot dataset format:

    imu_raw.csv       every Pico IMU frame at the full 200 Hz stream rate, each
                      sensor separately and UNCORRECTED (no mounting offset):
                      device_ts_us, imu_<role>_{qw,qx,qy,qz,gx_dps,gy_dps,gz_dps,
                      ax_g,ay_g,az_g}. Same schema as simple_drive.py's
                      imu_raw_*.csv, so the per-IMU yaws can be
                      recomputed offline.
    d435i_motion.csv  the camera's own gyro (rad/s) and accel (m/s^2), ~400 Hz,
                      factory motion correction on. The camera is rigid on the
                      cab, so its gyro projected onto gravity is a yaw-rate
                      reference with no parallax and no AHRS in the way.
    ir.mp4            D435i IR left imager (emitter off), for the masked
                      visual-yaw check (lerobot_vla/tools/visual_yaw_probe.py).
    frames.csv        one row per IR frame: host clock, the controller's joint
                      angles (slew = the base IMU yaw the stack actually uses),
                      the Pico clock of the IMU frame behind them, the sent
                      action, and reference-position marks.
    meta.json         IR intrinsics, gyro->IR extrinsics, IMU roles by stream
                      index, mounting offsets, IMU chain, stream ranges, marks.

Clocks: imu_raw.csv is on the Pico clock (device_ts_us); frames.csv pairs that
clock with the host perf_counter (state_ts, imu_device_us), which is the
mapping. d435i_motion.csv carries both the RealSense global timestamp and the
host perf_counter at callback.

Gamepad (same stick mapping as record_episodes.py):
    A  start take / stop + SAVE take
    B  stop + DISCARD take
    X  toggle hydraulic pump
    Y  mark "parked at a reference position" (logged with time and slew)

Suggested take (one take, ~5-10 min):
    1. Pump on, cab parked. Press A. Keep everything still ~10 s
       (D435i gyro bias + gravity direction).
    2. Cab still parked: work boom/arm/scoop in front of the camera ~30 s
       (builds the boom mask for the visual check).
    3. Slew to marked positions (e.g. 0, +45, +90, 0, -45, -90, 0), park at each
       for ~3 s and press Y. Repeat a few times, fast and slow.
    4. Back to the start mark, still ~10 s, press A.

Usage:
    .venv-lerobot/bin/python -m lerobot_vla.tools.slew_yaw_logger
    .venv-lerobot/bin/python -m lerobot_vla.tools.slew_yaw_logger --exposure-ir 8000
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lerobot_vla.camera import CameraConfig, D435iCamera
from lerobot_vla.excavator_robot import JOINT_NAMES, MasiExcavator
from lerobot_vla.gamepad import BTN_A, BTN_B, BTN_X, BTN_Y, LocalGamepadInput
from lerobot_vla.record_episodes import manual_action_from_axes

DEFAULT_OUT = _ROOT / "data_collection" / "slew_yaw_logs"
IMU_FIELDS = ("qw", "qx", "qy", "qz", "gx_dps", "gy_dps", "gz_dps", "ax_g", "ay_g", "az_g")


# ── D435i motion module ──────────────────────────────────────────────────────

class D435iMotion:
    """Gyro + accel off the D435i motion sensor, opened beside the IR pipeline.

    The motion module is its own RealSense sensor, so it is opened directly with
    a callback instead of being added to D435iCamera's pipeline: the image path
    record_episodes relies on stays exactly as it is. It must come from the
    running pipeline's device: a second rs.context() cannot enumerate a device
    the pipeline already holds.
    """

    def __init__(self, camera: D435iCamera):
        import pyrealsense2 as rs
        self.rs = rs
        active = camera.active_profile
        dev = active.get_device()
        self.sensor = next(s for s in dev.query_sensors()
                           if any(p.stream_type() == rs.stream.gyro for p in s.get_stream_profiles()))
        profiles = self.sensor.get_stream_profiles()
        self.gyro = max((p for p in profiles if p.stream_type() == rs.stream.gyro), key=lambda p: p.fps())
        self.accel = max((p for p in profiles if p.stream_type() == rs.stream.accel), key=lambda p: p.fps())
        ir = active.get_stream(rs.stream.infrared, camera.cfg.ir_index)
        i = ir.as_video_stream_profile().get_intrinsics()
        ext = self.gyro.get_extrinsics_to(ir)
        corr = rs.option.enable_motion_correction
        self.meta = {
            "serial": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
            "ir_intrinsics": {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy,
                              "ppx": i.ppx, "ppy": i.ppy, "model": str(i.model),
                              "coeffs": list(i.coeffs)},
            # Column-major 3x3, librealsense convention.
            "gyro_to_ir1_rotation": list(ext.rotation),
            "gyro_to_ir1_translation_m": list(ext.translation),
            "gyro_hz": self.gyro.fps(), "accel_hz": self.accel.fps(),
            "motion_correction": (self.sensor.get_option(corr)
                                  if self.sensor.supports(corr) else None),
        }
        self.rows: list[tuple] = []
        self._recording = False

    def _cb(self, f):
        if not self._recording:
            return
        m = f.as_motion_frame().get_motion_data()
        kind = 0 if f.get_profile().stream_type() == self.rs.stream.gyro else 1
        self.rows.append((time.perf_counter(), f.get_timestamp(), kind, m.x, m.y, m.z))

    def start(self):
        self.sensor.open([self.gyro, self.accel])
        self.sensor.start(self._cb)

    def begin(self):
        self.rows = []
        self._recording = True

    def end(self) -> list[tuple]:
        self._recording = False
        rows, self.rows = self.rows, []
        return rows

    def stop(self):
        try:
            self.sensor.stop()
            self.sensor.close()
        except Exception:
            pass


# ── IR video writer ──────────────────────────────────────────────────────────

class IrWriter:
    """H.264 gray video on a background thread, so encoding never stalls the loop."""

    def __init__(self, path: Path, fps: int, width: int, height: int):
        import av
        self.av = av
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("h264", rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        # Near-lossless: the visual check measures sub-pixel shifts.
        self.stream.options = {"crf": "15", "preset": "veryfast"}
        self.q: queue.Queue = queue.Queue(maxsize=300)
        self.thread = threading.Thread(target=self._run, name="ir-writer", daemon=True)
        self.thread.start()

    def _run(self):
        while (gray := self.q.get()) is not None:
            frame = self.av.VideoFrame.from_ndarray(gray, format="gray")
            for pkt in self.stream.encode(frame):
                self.container.mux(pkt)

    def put(self, gray: np.ndarray):
        self.q.put(gray)

    def close(self):
        self.q.put(None)
        self.thread.join()
        for pkt in self.stream.encode():
            self.container.mux(pkt)
        self.container.close()


# ── take ─────────────────────────────────────────────────────────────────────

class Take:
    def __init__(self, out_dir: Path, robot: MasiExcavator, motion: D435iMotion, fps: int):
        self.dir = out_dir
        self.dir.mkdir(parents=True)
        self.robot, self.motion = robot, motion
        self.t0 = time.perf_counter()
        self.wall0 = datetime.now().isoformat(timespec="seconds")
        self.video = IrWriter(self.dir / "ir.mp4", fps, 640, 480)
        self.frames: list[tuple] = []
        self.imu: list[tuple] = []
        self.marks: list[dict] = []
        robot.hardware.start_imu_raw_capture()   # drops anything buffered earlier
        motion.begin()

    def drain_imu(self):
        self.imu.extend(self.robot.hardware.drain_imu_raw_capture())

    def add_frame(self, gray, cam_ts, angles, state_ts, imu_us, action, mark):
        self.video.put(gray)
        self.frames.append((cam_ts - self.t0,
                            (state_ts - self.t0) if state_ts else np.nan,
                            -1 if imu_us is None else int(imu_us),
                            *angles, *action, int(mark)))

    def mark(self, slew_deg: float):
        m = {"t": time.perf_counter() - self.t0, "slew_deg": float(slew_deg),
             "frame": len(self.frames)}
        self.marks.append(m)
        print(f"\n[MARK] #{len(self.marks)} at t={m['t']:.1f}s slew={slew_deg:+.2f} deg")

    def save(self, meta: dict):
        self.drain_imu()
        motion_rows = self.motion.end()
        self.robot.hardware.stop_imu_raw_capture()
        self.video.close()

        info = self.robot.hardware.imu_stream_info()
        roles = info["roles_by_index"]
        vals = np.full((len(self.imu), 10 * len(roles)), np.nan)
        for r, (_, packets) in enumerate(self.imu):
            for i, pkt in enumerate(packets[:len(roles)]):
                vals[r, 10 * i:10 * i + min(10, len(pkt))] = pkt[:10]
        df = pd.DataFrame({"device_ts_us": [int(ts) for ts, _ in self.imu]})
        for i, role in enumerate(roles):
            for j, name in enumerate(IMU_FIELDS):
                df[f"imu_{role}_{name}"] = vals[:, 10 * i + j]
        df.to_csv(self.dir / "imu_raw.csv", index=False)

        m = pd.DataFrame(motion_rows, columns=["host_t", "rs_ts_ms", "kind", "x", "y", "z"])
        m["host_t"] -= self.t0
        m["kind"] = m["kind"].map({0: "gyro", 1: "accel"})
        m.to_csv(self.dir / "d435i_motion.csv", index=False)

        cols = (["cam_t", "state_t", "imu_device_us"] + [f"{j}_deg" for j in JOINT_NAMES]
                + [f"action_{j}" for j in JOINT_NAMES] + ["mark"])
        pd.DataFrame(self.frames, columns=cols).to_csv(self.dir / "frames.csv", index=False)

        meta = dict(meta, start=self.wall0, marks=self.marks,
                    imu_stream={k: info[k] for k in ("roles_by_index", "ranges", "target_sps",
                                                     "descriptors", "capture_dropped")})
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

        span = (df["device_ts_us"].iloc[-1] - df["device_ts_us"].iloc[0]) / 1e6 if len(df) > 1 else 0
        n_gyro = int((m["kind"] == "gyro").sum())
        dur = self.frames[-1][0] if self.frames else 0
        print(f"[SAVE] {self.dir}\n"
              f"       {len(self.frames)} IR frames ({len(self.frames) / max(dur, 1e-6):.1f} Hz), "
              f"{len(df)} IMU frames ({len(df) / max(span, 1e-6):.0f} Hz, "
              f"{info['capture_dropped']} dropped), {n_gyro} D435i gyro samples, "
              f"{len(self.marks)} marks")

    def discard(self):
        self.motion.end()
        self.robot.hardware.stop_imu_raw_capture()
        self.video.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def control_meta(robot: MasiExcavator) -> dict:
    """The IMU chain and mounting offsets this run used, straight from the config."""
    import yaml
    cfg = yaml.safe_load(Path(robot.hardware.control_config_file).read_text()) or {}
    imu = cfg.get("imu", {})
    return {"control_config_file": str(robot.hardware.control_config_file),
            "imu_mapping": imu.get("imu_mapping"),
            "imu_chain": imu.get("chain"),
            "mounting_offsets_quat": imu.get("mounting_offsets_quat")}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output root (default {DEFAULT_OUT})")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--robot", default="auto", help="Board profile (auto = detect)")
    p.add_argument("--max-take-s", type=float, default=900.0, help="Auto-stop + save a take after this long")
    p.add_argument("--exposure-ir", type=float, default=None,
                   help="Lock IR exposure, microseconds (default: auto)")
    p.add_argument("--gain-ir", type=float, default=None, help="IR gain 16..248 (only with --exposure-ir)")
    args = p.parse_args()

    cam_cfg = CameraConfig(width=640, height=480, fps=args.fps,
                           exposure_us=args.exposure_ir, gain=args.gain_ir)
    robot = MasiExcavator(profile=args.robot, camera_config=cam_cfg,
                          use_control_thread=True,
                          setpoint_hold_s=max(0.1, 4.0 / args.fps),
                          setpoint_decay_s=0.2,
                          state_joints=list(JOINT_NAMES))
    motion = None
    try:
        robot.connect()
        motion = D435iMotion(robot.camera)
        motion.start()
    except BaseException:
        if motion is not None:
            motion.stop()
        try:
            robot.disconnect()
        except Exception:
            pass
        raise
    meta = {"fps": args.fps, "exposure_ir_us": args.exposure_ir, "gain_ir": args.gain_ir,
            "d435i": motion.meta, **control_meta(robot)}
    print(f"[d435i] gyro {motion.meta['gyro_hz']} Hz, accel {motion.meta['accel_hz']} Hz, "
          f"IR fx={motion.meta['ir_intrinsics']['fx']:.2f}")

    pad = LocalGamepadInput()
    if not pad.open():
        motion.stop()
        robot.disconnect()
        return 1
    print("\nA=start/save take  B=discard take  X=pump  Y=mark reference position  Ctrl+C=quit\n")

    period = 1.0 / args.fps
    take: Take | None = None
    last_cam_ts = 0.0
    mask_prev = 0
    last_status = time.time()
    try:
        while True:
            cam_ts = robot.wait_for_next_frame(last_cam_ts, timeout_s=4 * period)
            fresh = cam_ts > 0.0
            if fresh:
                last_cam_ts = cam_ts

            mark = False
            axes, mask = pad.poll()
            if axes is not None:
                def btn(b): return bool(mask & (1 << b))
                def prev(b): return bool(mask_prev & (1 << b))
                if btn(BTN_A) and not prev(BTN_A):
                    if take is None:
                        take = Take(args.out / datetime.now().strftime("%Y%m%d_%H%M%S"),
                                    robot, motion, args.fps)
                        print(f"\n[TAKE] recording -> {take.dir}")
                    else:
                        robot.stop_motion()
                        take.save(meta)
                        take = None
                if btn(BTN_B) and not prev(BTN_B) and take is not None:
                    robot.stop_motion()
                    take.discard()
                    take = None
                    print("\n[TAKE] discarded")
                if btn(BTN_X) and not prev(BTN_X):
                    print(f"\n[pump] {'ON' if robot.toggle_pump() else 'OFF'}")
                if btn(BTN_Y) and not prev(BTN_Y) and take is not None:
                    mark = True
                mask_prev = mask
                action = manual_action_from_axes(axes)
            else:
                action = np.zeros(len(JOINT_NAMES), dtype=np.float32)
            if not pad.is_live():
                action = np.zeros(len(JOINT_NAMES), dtype=np.float32)
            sent = robot.send_action(action)

            if take is not None:
                take.drain_imu()
                angles, state_ts, imu_us = robot.get_joint_angles()
                if mark:
                    take.mark(angles[0])
                img, img_ts = robot.camera.get_latest()
                if fresh and img is not None:
                    take.add_frame(img[:, :, 0], img_ts, angles, state_ts, imu_us, sent, mark)
                if time.perf_counter() - take.t0 >= args.max_take_s:
                    robot.stop_motion()
                    take.save(meta)
                    take = None

            now = time.time()
            if now - last_status >= 5.0:
                last_status = now
                ja, _, _ = robot.get_joint_angles()
                state = (f"REC {len(take.frames)} frames, {len(take.marks)} marks"
                         if take is not None else "idle")
                print(f"[STATUS] {state} | pump={'ON' if robot.pump_enabled else 'OFF'} | "
                      f"slew={ja[0]:+.2f} lift={ja[1]:+.1f} tilt={ja[2]:+.1f} scoop={ja[3]:+.1f} deg"
                      + ("" if pad.is_live() else " | *** GAMEPAD LOST ***"))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if take is not None:
            robot.stop_motion()
            take.save(meta)
        pad.close()
        motion.stop()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
