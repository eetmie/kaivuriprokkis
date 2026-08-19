"""MASI excavator as a LeRobot-style robot.

Thin wrapper around the kaivuriprokkis control layer (HardwareInterface +
ExcavatorController for joint state and valve output) plus the D435i IR camera.

Observation:
    observation.state          float32[N]  joint angles, degrees. Defaults to
                                           [lift, tilt, scoop] — slew is left out,
                                           see DEFAULT_STATE_JOINTS below
    observation.images.cam1    uint8 HxWx3 infrared left imager, emitter off
    observation.images.cam2    uint8 HxWx3 color imager — only when the camera
                                           config sets enable_color

    img_ts / rgb_ts            float       perf_counter when the camera thread
                                           stored that frame
    state_ts / imu_device_us   float, int  perf_counter when the IMU frame behind
                                           observation.state was read, and that
                                           frame's own Pico clock in microseconds

The four *_ts entries are capture clocks rather than observation channels. Both
the camera and the 100 Hz control thread publish into latest-value caches, so a
caller sampling get_observation() at its own rate cannot otherwise tell a fresh
reading from one it already has. record_episodes.py writes them to the dataset
as the clock.* columns.

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

from lerobot_vla.camera import D435iCamera, CameraConfig

# Dataset/logical order used everywhere in this package. Maps to the control
# layer's channel names [slew, boom, arm, bucket] one-to-one.
JOINT_NAMES = ["slew", "lift", "tilt", "scoop"]
_CONTROL_CHANNELS = ["slew", "boom", "arm", "bucket"]

CAMERA_KEY = "observation.images.cam1"          # D435i infrared left imager
CAMERA_KEY_RGB = "observation.images.cam2"      # D435i color imager (optional)
STATE_KEY = "observation.state"
ACTION_KEY = "action"

# Which joints observation.state carries. Slew is excluded by default: its angle
# is `average_z_yaw` over the IMUs (control_config.yaml), an absolute world yaw
# with NO zeroing anywhere in the stack and no magnetometer to anchor it, so its
# origin is whatever the AHRS converged to at power-on. Within one session it is
# stable (measured 2026-08-19: dig/dump centroids drift -1.1/-1.5 deg over 31
# episodes), but ACROSS sessions the same physical pose can read any value in
# +-180 deg. A policy trained on one session's origin is then fed angles it never
# saw. The cameras observe slew directly, so the channel is not lost — only its
# unreliable absolute encoding is. Actions stay 4-dim; slew is still commanded.
DEFAULT_STATE_JOINTS = ["lift", "tilt", "scoop"]


class MasiExcavator:
    """Excavator robot: joint-angle observation + normalized valve actions."""

    robot_type = "masi_excavator"

    def __init__(self, profile: str = "auto",
                 camera_config: CameraConfig | None = None,
                 enable_slew: bool = True,
                 use_control_thread: bool = True,
                 setpoint_hold_s: float = 0.25,
                 setpoint_decay_s: float = 0.25,
                 setpoint_blend_s: float = 0.0,
                 state_joints: list[str] | None = None):
        self.state_joints = list(state_joints if state_joints is not None
                                 else DEFAULT_STATE_JOINTS)
        unknown = [j for j in self.state_joints if j not in JOINT_NAMES]
        if unknown:
            raise ValueError(f"state_joints {unknown} not in {JOINT_NAMES}")
        self._state_idx = [JOINT_NAMES.index(j) for j in self.state_joints]
        self.profile_name = profile
        self.camera = D435iCamera(camera_config)
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
            print("[robot] Starting IR camera (emitter off)"
                  + (" + RGB camera..." if self.camera.color_enabled else "..."))
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

    def get_joint_angles(self) -> tuple[np.ndarray, float | None, int | None]:
        """All four joint angles in degrees, [slew, lift, tilt, scoop], and their clocks.

        Always the full vector — status lines and diagnostics want slew even when
        the policy does not see it. Use ``get_state`` for the observation. Both
        return the clocks of the IMU frame the angles came from; discard them
        with `angles, _, _ = robot.get_joint_angles()` when only the pose matters.
        """
        angles, state_ts, imu_us = self.controller.get_joint_angles()
        return np.asarray(angles, dtype=np.float32), state_ts, imu_us

    def get_state(self) -> tuple[np.ndarray, float | None, int | None]:
        """observation.state, and the clocks of the IMU frame behind it.

        Returns (joints in ``state_joints`` in degrees, perf_counter seconds,
        Pico microseconds). The angles and their timestamps come out of one lock
        acquisition in the controller, so an age computed from them is exact
        rather than off by up to one control period. Both clocks are None until
        the 100 Hz thread has produced a state; callers that only want the pose
        discard them -- `state, _, _ = robot.get_state()`.
        """
        angles, state_ts, imu_us = self.get_joint_angles()
        return angles[self._state_idx], state_ts, imu_us

    @property
    def has_color(self) -> bool:
        """Whether cam2 (the RGB imager) is part of the observation."""
        return self.camera.color_enabled

    def wait_for_next_frame(self, after_ts: float, timeout_s: float) -> float:
        """Block until the camera publishes a frame newer than ``after_ts``.

        Returns the new frame's capture time, or 0.0 on timeout. Lets a caller
        pace itself on the camera rather than on its own sleep, which is what
        keeps a recording loop from beating against the sensor -- see
        D435iCamera.wait_for_next.
        """
        return self.camera.wait_for_next(after_ts, timeout_s)

    def get_observation(self) -> dict:
        img, img_ts = self.camera.get_latest()
        state, state_ts, imu_us = self.get_state()
        # The *_ts entries are capture clocks, not observation channels: the
        # camera runs its own thread behind a latest-frame cache and the joint
        # angles come off the 100 Hz control thread, so a caller sampling this
        # dict at its own rate cannot otherwise tell fresh data from a repeat.
        obs = {
            STATE_KEY: state,
            CAMERA_KEY: img,
            "img_ts": img_ts,
            "state_ts": state_ts,
            "imu_device_us": imu_us,
        }
        if self.has_color:
            # Same pipeline as cam1, so the pair shares a capture instant; the
            # separate timestamp is still reported for drop diagnosis.
            rgb, rgb_ts = self.camera.get_latest_color()
            obs[CAMERA_KEY_RGB] = rgb
            obs["rgb_ts"] = rgb_ts
        return obs

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
