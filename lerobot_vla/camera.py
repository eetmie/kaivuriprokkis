"""Two serial-pinned RealSense IR readers; depth, RGB and emitters stay off.

Each device has its own acquisition thread and timestamp. The devices are not
hardware synchronized. Frames are copied before releasing the SDK frame buffer.
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

import numpy as np

CAM1_SERIAL = "233522073919"
CAM2_SERIAL = "141722076647"


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    serial: str = CAM1_SERIAL
    ir_index: int = 2
    rotate180: bool = True
    warmup_frames: int = 15
    exposure_us: float | None = None
    gain: float | None = None
    enable_cam2: bool = True
    cam2_serial: str = CAM2_SERIAL
    cam2_ir_index: int = 1
    cam2_rotate180: bool = False
    cam2_exposure_us: float | None = None
    cam2_gain: float | None = None

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("Camera dimensions and fps must be positive")
        if not self.serial or (self.enable_cam2 and not self.cam2_serial):
            raise ValueError("Camera serials must be explicit")
        if self.enable_cam2 and self.serial == self.cam2_serial:
            raise ValueError("cam1 and cam2 must select different RealSense devices")
        if self.ir_index not in (1, 2) or self.cam2_ir_index not in (1, 2):
            raise ValueError("IR stream index must be 1 or 2")


def add_camera_arguments(parser):
    parser.add_argument("--cam1-serial", default=CAM1_SERIAL)
    parser.add_argument("--cam2-serial", default=CAM2_SERIAL)
    parser.add_argument("--cam1-ir-index", type=int, choices=(1, 2), default=2)
    parser.add_argument("--cam2-ir-index", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cam1-rotate180", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cam2-rotate180", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exposure-ir", type=float, default=None, help="cam1 exposure in us; default auto")
    parser.add_argument("--gain-ir", type=float, default=None)
    parser.add_argument("--exposure-ir2", type=float, default=None, help="cam2 exposure in us; default auto")
    parser.add_argument("--gain-ir2", type=float, default=None)


def config_from_args(args, *, enable_cam2=True, fps=30):
    return CameraConfig(fps=fps, serial=args.cam1_serial, ir_index=args.cam1_ir_index,
                        rotate180=args.cam1_rotate180, exposure_us=args.exposure_ir,
                        gain=args.gain_ir, enable_cam2=enable_cam2,
                        cam2_serial=args.cam2_serial, cam2_ir_index=args.cam2_ir_index,
                        cam2_rotate180=args.cam2_rotate180,
                        cam2_exposure_us=args.exposure_ir2, cam2_gain=args.gain_ir2)


def exposure_units_per_us(rs, sensor) -> float:
    """Scale factor from microseconds to this sensor's own exposure unit.

    The D435i's two sensors do not agree: the stereo module reports exposure in
    microseconds (range 1..165000) while the RGB camera uses 100 us ticks
    (range 1..10000 — the same ~1 s ceiling expressed a hundred times smaller).
    Derived from the reported range rather than hardcoded per sensor name, so
    every caller can keep speaking microseconds.
    """
    rng = sensor.get_option_range(rs.option.exposure)
    return 1.0 if rng.max > 20000 else 0.01


def _apply_exposure(rs, sensor, exposure_us: float | None, gain: float | None,
                    label: str) -> None:
    """Lock exposure/gain on one sensor, or leave it on auto.

    The D435i's two sensors do not report exposure in the same unit: the stereo
    module uses microseconds (range 1..165000) while the RGB camera uses 100 us
    ticks (range 1..10000 — the same ~1 s ceiling, a hundredth of the numbers).
    Passing microseconds straight to the color sensor would therefore overexpose
    by 100x. Derive the unit from the reported range rather than hardcoding it,
    and read back what actually landed so the operator can check it.
    """
    if exposure_us is None:
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1)
        return

    rng = sensor.get_option_range(rs.option.exposure)
    units_per_us = exposure_units_per_us(rs, sensor)
    value = min(max(exposure_us * units_per_us, rng.min), rng.max)

    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.exposure, float(value))
    if gain is not None:
        grng = sensor.get_option_range(rs.option.gain)
        sensor.set_option(rs.option.gain, float(min(max(gain, grng.min), grng.max)))

    # Reading the values back is a diagnostic, not part of configuring the
    # camera. The RSUSB backend intermittently stalls a control transfer
    # (RuntimeError: get_data_usb failed, RS2_USB_STATUS_PIPE) when several are
    # issued back to back, and losing a recording session to a log line would be
    # absurd — the set_option calls above already either applied or raised.
    try:
        applied_us = sensor.get_option(rs.option.exposure) / units_per_us
        applied_gain = sensor.get_option(rs.option.gain)
    except Exception as exc:
        print(f"[camera] {label}: exposure set to {exposure_us:.0f} us "
              f"(readback unavailable: {exc})")
        return
    print(f"[camera] {label}: exposure {applied_us:.0f} us "
          f"(requested {exposure_us:.0f}), gain {applied_gain:.0f}")
    if abs(applied_us - exposure_us) > max(1.0, 0.01 * exposure_us):
        print(f"[camera] {label}: *** exposure clamped to the sensor range "
              f"({rng.min / units_per_us:.0f}..{rng.max / units_per_us:.0f} us) ***")


class _IRReader:
    def __init__(self, cfg, serial, index, rotate, exposure, gain, label):
        self.cfg, self.serial, self.index = cfg, serial, index
        self.rotate, self.exposure, self.gain, self.label = rotate, exposure, gain, label
        self._latest, self._latest_ts = None, 0.0
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = self._pipeline = None
        self.error = None

    def start(self):
        import pyrealsense2 as rs
        self._stop.clear()
        self.error = None
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.infrared, self.index, self.cfg.width,
                             self.cfg.height, rs.format.y8, self.cfg.fps)
        profile = self._pipeline.start(config)
        sensor = profile.get_device().first_depth_sensor()
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 0)
            if sensor.get_option(rs.option.emitter_enabled) != 0:
                raise RuntimeError(f"{self.label}: could not disable IR emitter")
        _apply_exposure(rs, sensor, self.exposure, self.gain, self.label)
        for _ in range(self.cfg.warmup_frames):
            self._pipeline.wait_for_frames(timeout_ms=2000)
        self._thread = threading.Thread(target=self._run, name=self.label, daemon=True)
        self._thread.start()
        print(f"[camera] {self.label}: {self.serial}:IR{self.index}, "
              f"{self.cfg.width}x{self.cfg.height}@{self.cfg.fps}, "
              f"rotate180={self.rotate}, exposure={'auto' if self.exposure is None else self.exposure}")

    def _run(self):
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                ts = time.perf_counter()
                frame = frames.get_infrared_frame(self.index)
                if not frame:
                    continue
                gray = np.asanyarray(frame.get_data())
                if self.rotate:
                    gray = gray[::-1, ::-1]
                image = np.repeat(gray[:, :, None], 3, axis=2)
                with self._condition:
                    self._latest, self._latest_ts = image, ts
                    self._condition.notify_all()
            except Exception as exc:
                if not self._stop.is_set():
                    self.error = str(exc)
                break

    def latest(self):
        with self._condition:
            return self._latest, self._latest_ts

    def wait(self, after_ts, timeout_s):
        deadline = time.perf_counter() + timeout_s
        with self._condition:
            while self._latest_ts <= after_ts:
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or self._stop.is_set() or self.error:
                    return 0.0
                self._condition.wait(remaining)
            return self._latest_ts

    def stop(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._pipeline:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
        self._thread = self._pipeline = None


class D435iCamera:
    """Latest IR observations from one or two explicitly identified devices."""
    def __init__(self, cfg=None):
        self.cfg = cfg or CameraConfig()
        c = self.cfg
        self._cam1 = _IRReader(c, c.serial, c.ir_index, c.rotate180,
                               c.exposure_us, c.gain, "cam1 wrist IR")
        self._cam2 = (_IRReader(c, c.cam2_serial, c.cam2_ir_index, c.cam2_rotate180,
                                c.cam2_exposure_us, c.cam2_gain, "cam2 cabin IR")
                      if c.enable_cam2 else None)

    @property
    def active_profile(self):
        return self._cam1._pipeline.get_active_profile()

    @property
    def second_enabled(self):
        return self._cam2 is not None

    def start(self):
        try:
            self._cam1.start()
            if self._cam2:
                self._cam2.start()
            self.wait_for_frame()
        except BaseException:
            self.stop()
            raise

    def get_latest(self):
        return self._cam1.latest()

    def get_latest_cam2(self):
        return self._cam2.latest() if self._cam2 else (None, 0.0)

    def get_frames(self):
        img, ts = self.get_latest()
        img2, ts2 = self.get_latest_cam2()
        return img, ts, img2, ts2

    def wait_for_next(self, after_ts, timeout_s):
        return self._cam1.wait(after_ts, timeout_s)

    def wait_for_frame(self, timeout_s=5.0):
        deadline = time.perf_counter() + timeout_s
        for reader in (self._cam1, self._cam2):
            if reader and not reader.wait(0.0, max(0, deadline - time.perf_counter())):
                raise TimeoutError(f"No {reader.label} frame: {reader.error or 'timeout'}")
        return self.get_latest()[0]

    def stop(self):
        self._cam1.stop()
        if self._cam2:
            self._cam2.stop()


def check_frames(obs, camera_keys, max_age_s=0.15, max_skew_s=0.05):
    """Reject absent, stale or widely separated host-arrival timestamps.

    These timestamps measure host receipt, not synchronized sensor exposure.
    """
    now = time.perf_counter()
    stamps = []
    for key in camera_keys:
        ts = obs.get("img_ts" if key.endswith("cam1") else "cam2_ts", 0.0)
        if obs.get(key) is None or not ts or now - ts > max_age_s:
            raise RuntimeError(f"{key}: missing or stale IR frame")
        stamps.append(ts)
    if max(stamps) - min(stamps) > max_skew_s:
        raise RuntimeError("IR camera pair exceeds allowed arrival-time skew")


def fresh_observation(getter, camera_keys, timeout_s=0.2):
    """Allow acquisition threads to publish after GPU work, then check freshness.

    The control thread continues independently. A stalled stream still fails
    after a bounded wait; stale images are never passed to the policy.
    """
    deadline = time.perf_counter() + timeout_s
    while True:
        obs = getter()
        try:
            check_frames(obs, camera_keys)
            return obs
        except RuntimeError:
            if time.perf_counter() >= deadline:
                raise
            time.sleep(0.002)
