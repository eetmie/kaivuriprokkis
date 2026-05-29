#!/usr/bin/env python3
"""SmolVLA baseline runner for the MASI excavator (Jetson Orin Nano Super).

Goal of this v0 is to close the loop end-to-end with no ROS2 and no ZeroMQ:

    RealSense D435i (RGB only)  ->  SmolVLA (Reflex inference)  ->  ExcavatorController

The policy step is stubbed for now (zero-action) so this file can be run
without the Reflex install to verify the camera + controller plumbing. Plug
the real Reflex client into ``VLAPolicy.infer`` once the engine is built.

Action mapping
--------------
SmolVLA outputs an absolute end-effector pose target (xyz + rotation). On this
robot that maps to ``ExcavatorController.give_pose(position, rotation_y_deg)``,
the same surface ``run_hw_v2.py`` uses. If you later want joint-space control
instead, switch to ``enter_direct_mode`` + ``give_direct_commands`` with the
four normalized valve channels: rotate, lift_boom, tilt_boom, scoop.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.board import resolve_profile as _resolve_board_profile
from modules.excavator_controller import ExcavatorController
from modules.hardware_interface import HardwareFaultError, HardwareInterface

logger = logging.getLogger("vla_v0")


# --------------------------------------------------------------------------- #
# Camera: RealSense D435i, RGB-only
# --------------------------------------------------------------------------- #


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    # Disable depth + IR streams to keep the USB and Orin Nano load low. Depth
    # alone is ~15 MB/s at 848x480x30 plus on-camera processing; we don't need
    # it for SmolVLA RGB conditioning.
    enable_depth: bool = False


class D435iRGB:
    """Minimal pyrealsense2 wrapper that publishes the latest RGB frame.

    The D435i happily runs RGB-only: only the color sensor is enabled in the
    pipeline config, so the depth ASIC, IR projector, and IR streams stay
    powered down and the USB 3 bandwidth stays under ~30 MB/s at 640x480x30.
    """

    def __init__(self, cfg: CameraConfig):
        import pyrealsense2 as rs

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, cfg.width, cfg.height,
                             rs.format.bgr8, cfg.fps)
        if cfg.enable_depth:
            config.enable_stream(rs.stream.depth, cfg.width, cfg.height,
                                 rs.format.z16, cfg.fps)
        self._config = config
        self._cfg = cfg

        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        profile = self._pipeline.start(self._config)
        # Lock exposure once warmed up: VLA policies are sensitive to auto-
        # exposure jitter, and the D435i lets us turn it off cleanly.
        color_sensor = profile.get_device().query_sensors()[1]
        try:
            color_sensor.set_option(self._rs.option.enable_auto_exposure, 0)
        except Exception as exc:
            logger.warning("Could not disable auto-exposure: %s", exc)

        self._thread = threading.Thread(target=self._run, name="d435i-rgb",
                                        daemon=True)
        self._thread.start()
        logger.info("D435i RGB stream started (%dx%d @ %d Hz)",
                    self._cfg.width, self._cfg.height, self._cfg.fps)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except Exception as exc:
                logger.warning("RealSense wait_for_frames failed: %s", exc)
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            with self._lock:
                self._latest = img
                self._latest_ts = time.perf_counter()

    def get_latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            return self._latest, self._latest_ts

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._pipeline.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Policy: SmolVLA via Reflex (stubbed until the engine is built)
# --------------------------------------------------------------------------- #


@dataclass
class PolicyOutput:
    ee_xyz_m: np.ndarray          # shape (3,), target end-effector position [m]
    rot_y_deg: float              # target rotation about Y in degrees
    confidence: float = 1.0


class VLAPolicy:
    """SmolVLA inference adapter.

    Replace ``infer`` with a Reflex client call once the ONNX engine is built
    on the Jetson. The signature is kept narrow on purpose: state in, pose
    out — no global state, easy to swap.
    """

    def __init__(self, instruction: str):
        self.instruction = instruction
        self._loaded = False

    def load(self) -> None:
        # TODO: instantiate Reflex Python client here, point it at the built
        # TensorRT engine, do a warmup forward pass. Until then the policy is
        # a no-op so we can validate the camera + controller loop standalone.
        self._loaded = True
        logger.info("VLAPolicy loaded (stub). Instruction: %r", self.instruction)

    def infer(self, image_bgr: np.ndarray, current_ee_xyz: np.ndarray,
              current_rot_y_deg: float) -> PolicyOutput:
        # Stub: hold position. Replace with Reflex inference.
        return PolicyOutput(
            ee_xyz_m=np.asarray(current_ee_xyz, dtype=np.float32),
            rot_y_deg=float(current_rot_y_deg),
            confidence=0.0,
        )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def _build_hardware(profile, debug: bool) -> tuple[HardwareInterface,
                                                    ExcavatorController]:
    hardware = HardwareInterface(
        config_file=profile["servo_config_file"],
        control_config_file=profile["control_config_file"],
        perf_enabled=debug,
        cleanup_disable_osc=False,
        enable_adc=False,
        enable_imu=bool(profile["enable_imu"]),
        start_adc_reader=False,
        imu_rt_priority=0,
        pwm_i2c_bus=profile["pwm_i2c_bus"],
        pwm_i2c_addr=profile["pwm_i2c_addr"],
    )
    controller = ExcavatorController(
        hardware_interface=hardware,
        enable_perf_tracking=True,
        log_level="DEBUG" if debug else "INFO",
        rt_priority=0,
        control_config_file=profile["control_config_file"],
    )
    return hardware, controller


def _wait_for_hw(hardware: HardwareInterface, timeout_s: float = 90.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if hardware.is_hardware_ready():
                return True
        except HardwareFaultError as exc:
            logger.critical("Hardware fault during startup: %s", exc)
            return False
        time.sleep(0.2)
    return False


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    profile = _resolve_board_profile(args.robot)
    hardware, controller = _build_hardware(profile, args.debug)

    camera = D435iRGB(CameraConfig(width=args.width, height=args.height,
                                   fps=args.fps))
    policy = VLAPolicy(instruction=args.instruction)

    stop_event = threading.Event()

    def _on_signal(signum, _frame):
        logger.info("Signal %d received, stopping...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        camera.start()
        policy.load()

        logger.info("Starting controller...")
        controller.start()
        time.sleep(5.0)

        if not _wait_for_hw(hardware):
            logger.error("Hardware not ready, aborting.")
            return 1

        period = 1.0 / args.policy_hz
        next_tick = time.perf_counter()
        last_log = next_tick
        step_count = 0

        while not stop_event.is_set():
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(period * 0.25, next_tick - now))
                continue
            next_tick += period

            image, img_ts = camera.get_latest()
            if image is None:
                continue

            # TODO: replace with real measured EE pose from the controller.
            current_ee = np.zeros(3, dtype=np.float32)
            current_rot = 0.0

            t_infer = time.perf_counter()
            out = policy.infer(image, current_ee, current_rot)
            infer_ms = (time.perf_counter() - t_infer) * 1000.0

            if not args.dry_run:
                controller.give_pose(out.ee_xyz_m.tolist(), out.rot_y_deg)

            step_count += 1
            if now - last_log >= 1.0:
                logger.info(
                    "step=%d hz=%.1f infer=%.1fms img_age=%.1fms",
                    step_count, step_count / (now - last_log + 1e-9),
                    infer_ms, (now - img_ts) * 1000.0,
                )
                step_count = 0
                last_log = now

    finally:
        logger.info("Shutting down...")
        try:
            controller.stop(timeout_s=2.0)
        except Exception:
            pass
        try:
            hardware.shutdown()
        except Exception:
            pass
        camera.stop()

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SmolVLA baseline runner (no ROS2).")
    p.add_argument("--robot", default="auto",
                   help="Board profile name (auto = detect).")
    p.add_argument("--instruction", default="dig a hole in front of you",
                   help="Natural-language instruction passed to SmolVLA.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--policy-hz", type=float, default=10.0,
                   help="VLA inference rate. SmolVLA on Orin Nano Super is "
                        "expected to land around 10-20 Hz with TensorRT FP16.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run camera + policy but do not send commands to "
                        "hardware. Useful for first bring-up.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(_parse_args()))
