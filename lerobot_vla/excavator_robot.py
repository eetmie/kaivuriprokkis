"""MASI excavator as a LeRobot-style robot.

Thin wrapper around the kaivuriprokkis control layer (HardwareInterface +
ExcavatorController for joint state and valve output) plus the D435i IR camera.

Observation:
    observation.state          float32[4]  joint angles [slew, lift, tilt, scoop] (deg)
    observation.images.cam1    uint8 HxWx3 infrared left imager, emitter off

Action:
    action                     float32[4]  normalized valve commands [-1, 1]
                                           [slew, lift, tilt, scoop]

Valve output runs on ExcavatorController's 100 Hz control thread via its
direct-command mode: ``send_action`` only *stores* a setpoint and returns
immediately, and the control thread resamples it onto the fixed loop rate.
Callers here (30 Hz recording, ~1 Hz VLA chunks) are far too slow to drive the
PWM layer directly — its dither, watchdog and input-rate gate all advance per
call, so writing at 30 Hz aliases the dither and trips the rate gate. See
modules/setpoint_schedule.py.

Pass ``use_control_thread=False`` to fall back to the old behaviour (a
DirectController writing I2C on the caller's thread) for A/B comparison on the
bench. That path has the rate problems described above; it is not for
recording or inference.
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
                 enable_slew: bool = True,
                 use_control_thread: bool = True,
                 setpoint_hold_s: float = 0.25,
                 setpoint_decay_s: float = 0.25,
                 setpoint_blend_s: float = 0.0):
        self.profile_name = profile
        self.camera = D435iIRCamera(camera_config)
        self.enable_slew = enable_slew
        self.use_control_thread = use_control_thread
        self.setpoint_hold_s = setpoint_hold_s
        self.setpoint_decay_s = setpoint_decay_s
        self.setpoint_blend_s = setpoint_blend_s
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
        # The rate gate and stale watchdog are sized for the *writer*, which is
        # the 100 Hz control thread — not for this process's 30 Hz observation
        # loop. Both are set explicitly rather than inherited: the old code took
        # HardwareInterface's default threshold of 20 Hz, which a 30 Hz caller
        # silently dipped under, after which update_named() discarded every
        # command until the rate recovered.
        rate_threshold = 50 if self.use_control_thread else 0
        self.hardware = HardwareInterface(
            config_file=profile["servo_config_file"],
            control_config_file=profile["control_config_file"],
            pump_auto_mode=False,
            toggle_channels=True,
            stale_timeout_s=0.5,
            input_rate_threshold=rate_threshold,
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

        if self.use_control_thread:
            self.controller.enter_direct_command_mode(
                hold_timeout_s=self.setpoint_hold_s,
                decay_s=self.setpoint_decay_s,
                blend_s=self.setpoint_blend_s,
                joint_names=_CONTROL_CHANNELS,
            )
            hz = self.controller.config.control_frequency
            print(f"[robot] Valve output on control thread at {hz:.0f} Hz "
                  f"(setpoint hold {self.setpoint_hold_s:.2f}s, "
                  f"decay {self.setpoint_decay_s:.2f}s)")
        else:
            self.direct = DirectController(self.hardware)
            self.controller.suspend_ik_output()
            print("[robot] *** Legacy direct-write mode: valves are driven at "
                  "the caller's rate ***")

        if start_camera:
            print("[robot] Starting IR camera (emitter off)...")
            self.camera.start()
            self.camera.wait_for_frame()

        self._connected = True
        print(f"[robot] Connected (profile: {profile['profile_name']})")

    def disconnect(self) -> None:
        self._connected = False
        if self.controller is not None and self.use_control_thread:
            try:
                self.controller.exit_direct_command_mode()
            except Exception:
                pass
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
        """Set 4 normalized valve commands [slew, lift, tilt, scoop] in [-1, 1].

        Non-blocking on the control-thread path: this stores the setpoint and
        returns; the 100 Hz control thread does the I2C write. Returns the
        clipped action (this is what should be recorded in the dataset — it is
        the setpoint the control thread was actually handed).
        """
        a = self._clip(action)
        cmds = {ch: float(v) for ch, v in zip(_CONTROL_CHANNELS, a)}
        if self.use_control_thread:
            self.controller.give_direct_commands(cmds)
        else:
            self.direct.give_commands(cmds)
            self.direct.send_pending()
        return a

    def send_action_chunk(self, chunk: np.ndarray, fps: float) -> np.ndarray:
        """Hand a policy action chunk over as a trajectory to play at ``fps``.

        The control thread indexes into the chunk by elapsed time and
        interpolates between steps, so an N-step plan authored at ``fps`` drives
        the valves at the full loop rate. Returns the clipped chunk.

        Requires the control-thread path; with ``use_control_thread=False``
        there is no scheduler to play it.
        """
        if not self.use_control_thread:
            raise RuntimeError(
                "send_action_chunk requires use_control_thread=True "
                "(legacy direct-write mode has no setpoint scheduler)")
        c = np.clip(np.asarray(chunk, dtype=np.float32), -1.0, 1.0)
        if c.ndim != 2 or c.shape[1] != len(_CONTROL_CHANNELS):
            raise ValueError(
                f"chunk must be (N, {len(_CONTROL_CHANNELS)}), got {c.shape}")
        if not self.enable_slew:
            c[:, 0] = 0.0
        self.controller.give_direct_chunk(c, fps, joint_names=_CONTROL_CHANNELS)
        return c

    def _clip(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if not self.enable_slew:
            a[0] = 0.0
        return a

    def get_setpoint_status(self) -> dict:
        """Held-setpoint telemetry: age, decay, chunk position."""
        if not self.use_control_thread:
            return {'active': False, 'age_s': 0.0, 'decay': 1.0,
                    'exhausted': False, 'chunk_pos': None, 'commands': {}}
        return self.controller.get_direct_status()

    def stop_motion(self) -> None:
        """Zero all valve outputs immediately."""
        if self.use_control_thread:
            # Zeroed setpoint, not a cleared schedule: the control thread keeps
            # writing zeros every tick, which holds the valves centered *and*
            # keeps the PWM watchdog fed.
            self.controller.give_direct_commands({ch: 0.0 for ch in _CONTROL_CHANNELS})
        else:
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
