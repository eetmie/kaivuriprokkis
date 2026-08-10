"""RealSense D435i infrared cam1 reader for the VLA pipeline.

Streams the LEFT imager (infrared stream index 1) as Y8 with the laser
emitter DISABLED — without the emitter the IR image is a clean grayscale
photo instead of a dot-pattern projection. Depth stays off entirely, so the
USB load is a single 640x480x30 8-bit stream.

Frames are published as HxWx3 uint8 (gray replicated to 3 channels) because
SmolVLA and the LeRobot dataset format both expect 3-channel images.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IRCameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    ir_index: int = 1          # 1 = left imager (the one physically positioned well)
    warmup_frames: int = 15    # let the (auto-)exposure settle before first use
    # Manual exposure lock. None = auto-exposure (varies with scene — bad for
    # VLA consistency; lock it once a good value is found, see tune_exposure.py).
    # At 30 fps the exposure ceiling is ~33000 us (one frame period).
    exposure_us: float | None = None
    gain: float | None = None  # sensor gain (D435i range 16..248), None = default


class D435iIRCamera:
    """Threaded latest-frame reader for the D435i left IR imager, emitter off."""

    def __init__(self, cfg: IRCameraConfig | None = None):
        self.cfg = cfg or IRCameraConfig()
        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._frame_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pipeline = None
        self._rs = None

    def start(self) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.infrared, self.cfg.ir_index,
                             self.cfg.width, self.cfg.height,
                             rs.format.y8, self.cfg.fps)
        profile = self._pipeline.start(config)

        # The emitter is a depth-sensor option even when only IR is streamed.
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        if depth_sensor.get_option(rs.option.emitter_enabled) != 0:
            raise RuntimeError("Could not disable the IR laser emitter")

        # Exposure: the IR imagers hang off the stereo (depth) sensor.
        if self.cfg.exposure_us is not None:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            depth_sensor.set_option(rs.option.exposure, float(self.cfg.exposure_us))
            if self.cfg.gain is not None:
                depth_sensor.set_option(rs.option.gain, float(self.cfg.gain))
        elif depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

        for _ in range(self.cfg.warmup_frames):
            self._pipeline.wait_for_frames(timeout_ms=2000)

        self._thread = threading.Thread(target=self._run, name="d435i-ir", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue
            ir = frames.get_infrared_frame(self.cfg.ir_index)
            if not ir:
                continue
            gray = np.asanyarray(ir.get_data())              # HxW uint8
            img = np.repeat(gray[:, :, None], 3, axis=2)     # HxWx3
            with self._lock:
                self._latest = img
                self._latest_ts = time.perf_counter()
                self._frame_count += 1

    def get_latest(self) -> tuple[Optional[np.ndarray], float]:
        """Latest HxWx3 uint8 frame and its capture time (perf_counter)."""
        with self._lock:
            return self._latest, self._latest_ts

    def wait_for_frame(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            img, _ = self.get_latest()
            if img is not None:
                return img
            time.sleep(0.02)
        raise TimeoutError("No IR frame within timeout")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
