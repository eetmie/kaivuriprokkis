"""MASI excavator as a LeRobot-style robot.

Thin wrapper around the kaivuriprokkis control layer (HardwareInterface +
ExcavatorController for joint state, DirectController for open-loop valve
commands — the same surface simple_drive.py uses) plus the D435i IR camera.

Observation:
    observation.state          float32[4]  joint angles [slew, lift, tilt, scoop] (deg)
    observation.images.cam1    uint8 HxWx3 infrared left imager, emitter off

Action:
    action                     float32[4]  normalized valve commands [-1, 1]
                                           [slew, lift, tilt, scoop]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_KAIVURI_ROOT = Path(__file__).resolve().parents[1]
if str(_KAIVURI_ROOT) not in sys.path:
    sys.path.insert(0, str(_KAIVURI_ROOT))

from lerobot_vla.ir_camera import D435iIRCamera, IRCameraConfig

# Dataset/logical order used everywhere in this package. Maps to the control
# layer's channel names [slew, boom, arm, bucket] one-to-one.
JOINT_NAMES = ["slew", "lift", "tilt", "scoop"]
_CONTROL_CHANNELS = ["slew", "boom", "arm", "bucket"]

CAMERA_KEY = "observation.images.cam1"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


class MasiExcavator:
    """Excavator robot: joint-angle observation + normalized valve actions."""

    robot_type = "masi_excavator"

    def __init__(self, profile: str = "auto",
                 camera_config: IRCameraConfig | None = None,
                 enable_slew: bool = True):
        self.profile_name = profile
        self.camera = D435iIRCamera(camera_config)
        self.enable_slew = enable_slew
        self.hardware = None
        self.controller = None
        self.direct = None
        self._connected = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def connect(self, start_camera: bool = True) -> None:
        from modules.board import resolve_profile
        from modules.bringup import wait_for_hardware_ready
        from modules.direct_controller import DirectController
        from modules.excavator_controller import ExcavatorController
        from modules.hardware_interface import HardwareInterface

        profile = resolve_profile(self.profile_name)
        if not profile["enable_imu"]:
            raise RuntimeError(
                "Profile has IMUs disabled — joint angles would be zeros, which "
                "is useless as observation.state. Use an IMU-enabled profile.")

        print("[robot] Initializing hardware...")
        self.hardware = HardwareInterface(
            config_file=profile["servo_config_file"],
            control_config_file=profile["control_config_file"],
            pump_auto_mode=False,
            toggle_channels=True,
            stale_timeout_s=0.5,
            enable_pwm=True,
            enable_imu=True,
            enable_adc=False,
            start_imu_reader=True,
            start_adc_reader=False,
            cleanup_disable_osc=False,
            pwm_i2c_bus=profile["pwm_i2c_bus"],
            pwm_i2c_addr=profile["pwm_i2c_addr"],
        )
        wait_for_hardware_ready(self.hardware)

        print("[robot] Starting controller...")
        self.controller = ExcavatorController(
            self.hardware, config=None, enable_perf_tracking=False,
            control_config_file=profile["control_config_file"],
        )
        self.controller.start()
        time.sleep(2.0)  # numba JIT warmup, same as simple_drive.py

        self.direct = DirectController(self.hardware)
        self.controller.suspend_ik_output()

        if start_camera:
            print("[robot] Starting IR camera (emitter off)...")
            self.camera.start()
            self.camera.wait_for_frame()

        self._connected = True
        print(f"[robot] Connected (profile: {profile['profile_name']})")

    def disconnect(self) -> None:
        self._connected = False
        if self.direct is not None:
            try:
                self.direct.clear()
                self.direct.send_pending()
            except Exception:
                pass
        if self.controller is not None:
            try:
                self.controller.emergency_stop(reset_pump=True)
            except Exception:
                pass
            try:
                self.controller.stop()
            except Exception:
                pass
        if self.hardware is not None:
            try:
                self.hardware.shutdown()
            except Exception:
                pass
        self.camera.stop()
        print("[robot] Disconnected.")

    # ── observation / action ─────────────────────────────────────────────────

    def get_joint_angles(self) -> np.ndarray:
        """Joint angles in degrees, [slew, lift, tilt, scoop]."""
        return np.asarray(self.controller.get_joint_angles(), dtype=np.float32)

    def get_observation(self) -> dict:
        img, img_ts = self.camera.get_latest()
        return {
            STATE_KEY: self.get_joint_angles(),
            CAMERA_KEY: img,
            "img_ts": img_ts,
        }

    def send_action(self, action: np.ndarray) -> np.ndarray:
        """Send 4 normalized valve commands [slew, lift, tilt, scoop] in [-1, 1].

        Returns the clipped action that was actually sent (this is what should
        be recorded in the dataset).
        """
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if not self.enable_slew:
            a[0] = 0.0
        cmds = {ch: float(v) for ch, v in zip(_CONTROL_CHANNELS, a)}
        self.direct.give_commands(cmds)
        self.direct.send_pending()
        return a

    def stop_motion(self) -> None:
        """Zero all valve outputs immediately."""
        self.direct.clear()
        self.direct.send_pending()

    # ── auxiliary controls (same surface simple_drive.py exposes) ────────────

    def set_pump(self, enabled: bool) -> None:
        self.hardware.set_pump_enabled(enabled)

    @property
    def pump_enabled(self) -> bool:
        pwm = self.hardware.pwm_controller
        return bool(pwm and pwm.pump_enabled)

    def toggle_pump(self) -> bool:
        new_state = not self.pump_enabled
        self.set_pump(new_state)
        return new_state

    def reload_config(self) -> bool:
        self.stop_motion()
        time.sleep(0.1)
        return bool(self.hardware.reload_config())
