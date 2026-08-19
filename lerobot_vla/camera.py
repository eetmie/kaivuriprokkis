"""RealSense D435i reader for the VLA pipeline: IR cam1, optionally RGB cam2.

Streams the LEFT imager (infrared stream index 1) as Y8 with the laser
emitter DISABLED — without the emitter the IR image is a clean grayscale
photo instead of a dot-pattern projection. Depth stays off entirely, so the
USB load is a single 640x480x30 8-bit stream.

With ``enable_color`` the RGB imager is added to the *same* pipeline, so both
frames come out of one ``wait_for_frames()`` and are hardware-synchronized.
Concurrent IR1+color at 640x480x30 was measured clean on this board (0 dropped
frames over 5 minutes, USB at ~6% of the bus — see realsense_logging_bandwidth.md);
the cost is host CPU for a second video encode, not bandwidth.

Frames are published as HxWx3 uint8 — IR as gray replicated to 3 channels, color
as rgb8 — because SmolVLA and the LeRobot dataset format both expect 3-channel
RGB images.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    ir_index: int = 1          # 1 = left imager (the one physically positioned well)
    warmup_frames: int = 15    # let the (auto-)exposure settle before first use
    # Manual exposure lock. None = auto-exposure (varies with scene — bad for
    # VLA consistency; lock it once a good value is found, see tune_exposure.py).
    # At 30 fps the exposure ceiling is ~33000 us (one frame period).
    exposure_us: float | None = None
    gain: float | None = None  # sensor gain (D435i stereo range 16..248), None = default

    # Second camera: the RGB imager, published as cam2 alongside the IR cam1.
    enable_color: bool = False
    color_exposure_us: float | None = None
    color_gain: float | None = None   # D435i RGB gain range is 0..128, not 16..248


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


class D435iCamera:
    """Threaded latest-frame reader for the D435i, emitter off.

    Always publishes the left IR imager; publishes the color imager too when
    ``cfg.enable_color`` is set. Both are read from one pipeline so the pair
    shares a capture instant.
    """

    def __init__(self, cfg: CameraConfig | None = None):
        self.cfg = cfg or CameraConfig()
        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._latest_color: Optional[np.ndarray] = None
        self._latest_color_ts: float = 0.0
        self._frame_count = 0
        self._lock = threading.Lock()
        # Signals a newly stored frame to wait_for_next(). Wraps _lock, so the
        # existing `with self._lock:` blocks keep working unchanged.
        self._new_frame = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pipeline = None
        self._rs = None

    @property
    def color_enabled(self) -> bool:
        return bool(self.cfg.enable_color)

    def start(self) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.infrared, self.cfg.ir_index,
                             self.cfg.width, self.cfg.height,
                             rs.format.y8, self.cfg.fps)
        if self.cfg.enable_color:
            # rgb8 rather than bgr8: the dataset and SmolVLA both want RGB, so
            # let librealsense do the conversion instead of flipping channels
            # on every frame in Python.
            config.enable_stream(rs.stream.color,
                                 self.cfg.width, self.cfg.height,
                                 rs.format.rgb8, self.cfg.fps)
        profile = self._pipeline.start(config)

        # The emitter is a depth-sensor option even when only IR is streamed.
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        if depth_sensor.get_option(rs.option.emitter_enabled) != 0:
            raise RuntimeError("Could not disable the IR laser emitter")

        # Exposure: the IR imagers hang off the stereo (depth) sensor; the color
        # imager is a separate sensor with its own exposure, gain and units.
        _apply_exposure(rs, depth_sensor, self.cfg.exposure_us, self.cfg.gain, "IR cam1")
        if self.cfg.enable_color:
            _apply_exposure(rs, profile.get_device().first_color_sensor(),
                            self.cfg.color_exposure_us, self.cfg.color_gain, "RGB cam2")

        for _ in range(self.cfg.warmup_frames):
            self._pipeline.wait_for_frames(timeout_ms=2000)

        self._thread = threading.Thread(target=self._run, name="d435i-cam", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue
            now = time.perf_counter()
            ir = frames.get_infrared_frame(self.cfg.ir_index)
            color = frames.get_color_frame() if self.cfg.enable_color else None

            ir_img = None
            if ir:
                gray = np.asanyarray(ir.get_data())              # HxW uint8
                ir_img = np.repeat(gray[:, :, None], 3, axis=2)  # HxWx3
            color_img = np.asanyarray(color.get_data()) if color else None

            if ir_img is None and color_img is None:
                continue
            with self._new_frame:
                if ir_img is not None:
                    self._latest = ir_img
                    self._latest_ts = now
                    self._frame_count += 1
                if color_img is not None:
                    self._latest_color = color_img
                    self._latest_color_ts = now
                self._new_frame.notify_all()

    def get_latest(self) -> tuple[Optional[np.ndarray], float]:
        """Latest IR HxWx3 uint8 frame and its capture time (perf_counter)."""
        with self._lock:
            return self._latest, self._latest_ts

    def wait_for_next(self, after_ts: float, timeout_s: float) -> float:
        """Block until an IR frame newer than ``after_ts`` lands. Returns its
        capture time, or 0.0 if none arrived within ``timeout_s``.

        This is what lets a caller run on the camera's clock instead of its own.
        A free-running loop at a nominally identical rate is still a second
        clock: the D435i delivers at ~29.976 Hz, so a 30.000 Hz sleep loop beats
        against it and the frame age walks the whole 0..33 ms range, reusing a
        frame at one end and skipping one at the other. Waiting for the frame
        removes the second clock rather than correcting for it.

        The timeout exists so a stalled camera cannot wedge the caller: it has a
        gamepad to poll and setpoints to feed, and those must keep running (or
        visibly stop) rather than block behind a dead USB pipe.
        """
        deadline = time.perf_counter() + timeout_s
        with self._new_frame:
            while self._latest_ts <= after_ts:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0 or not self._new_frame.wait(remaining):
                    if self._latest_ts <= after_ts:
                        return 0.0
            return self._latest_ts

    def get_latest_color(self) -> tuple[Optional[np.ndarray], float]:
        """Latest RGB HxWx3 uint8 frame and its capture time, or (None, 0.0)."""
        with self._lock:
            return self._latest_color, self._latest_color_ts

    def _wait(self, getter, what: str, timeout_s: float) -> np.ndarray:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            img, _ = getter()
            if img is not None:
                return img
            time.sleep(0.02)
        raise TimeoutError(f"No {what} frame within timeout")

    def wait_for_frame(self, timeout_s: float = 5.0) -> np.ndarray:
        """Block until an IR frame is available (and a color one, if enabled)."""
        img = self._wait(self.get_latest, "IR", timeout_s)
        if self.cfg.enable_color:
            self._wait(self.get_latest_color, "color", timeout_s)
        return img

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
