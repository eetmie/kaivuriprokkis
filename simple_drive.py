#!/usr/bin/env python3
"""Excavator open-loop driving and hydraulic data collection.

Drives the valves straight from the stick — no compensation, no closed loop —
and records 10-minute strips for blackbox model training. Closed-loop and
compensated driving live in control_prototype/drive_compensated.py.

Usage:
    python simple_drive.py                        # local gamepad
    python simple_drive.py --robot jetson
    python simple_drive.py --ip 0.0.0.0:8080      # remote UDP client instead
    python simple_drive.py --enable-slew --enable-tracks

Input source:
    Default is a gamepad wired straight into this machine. Passing --ip
    switches to a remote client over UDP (clients/client_gui.py).

Button Controls:
    Button A (bit 0): Start / Stop data logging (saves strip on stop)
    Button B (bit 1): Toggle sine excitation on / off (default: OFF)
    Button X (bit 2): Toggle hydraulic pump
    Button Y (bit 3): Reload servo config from disk
    D-pad Up/Down (bits 4/5): Step sine amplitude (local pad only)

Amplitude is the only sine knob: every other parameter is randomized per joint
on each enable, which explores the input space better than stepping presets by
hand did. The seed for a run is printed and written to every logged sample.

Recording runs in 10-minute strips: a strip is auto-saved and a new one
started while logging stays on, so a long session lands as a series of files
rather than one unbounded CSV.
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.board import PROFILES as ROBOT_PROFILES, resolve_profile as _resolve_board_profile
from modules.bringup import wait_for_hardware_ready
from modules.direct_controller import DirectController
from modules.udp_socket import UDPSocket


# ── constants ────────────────────────────────────────────────────────────────

SAMPLING_FREQUENCY         = 100    # Hz
COMMAND_STALE_TIMEOUT_S    = 0.5
STATUS_PRINT_INTERVAL_S    = 5.0
PRINT_DECIMATION           = 10     # IMU print decimation (→ ~10 Hz)
STRIP_MINUTES              = 10.0   # auto-save cadence while logging

LOG_OUTPUT_DIR = Path(__file__).parent / "data_collection" / "hydraulic_data"

JOINT_NAMES       = ['slew', 'boom', 'arm', 'bucket']
AMPLITUDE_PRESETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
CONTROL_JOINT_NAMES = ['slew', 'lift', 'arm', 'bucket']
IMU_ROLE_ORDER      = ['base', 'boom', 'arm', 'bucket']


# ── IMU helpers ───────────────────────────────────────────────────────────────

def _euler_pry_deg(quat) -> tuple[float, float, float]:
    q = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q /= norm
    w, x, y, z = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sp    = 2*(w*y - z*x)
    pitch = np.copysign(np.pi/2, sp) if abs(sp) >= 1 else np.arcsin(sp)
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return tuple(float(v) for v in np.degrees([pitch, roll, yaw]))


def _get_control_joint_names(controller) -> list[str]:
    names = list(CONTROL_JOINT_NAMES)
    chain = getattr(getattr(controller, 'robot_config', None), 'imu_chain', None) or []
    for item in chain:
        if not isinstance(item, dict) or 'output_index' not in item:
            continue
        i = int(item['output_index'])
        if 0 <= i < len(names) and item.get('joint'):
            names[i] = str(item['joint'])
    return names


def _get_imu_role_order(controller) -> list[str]:
    rc = getattr(controller, 'robot_config', None)
    roles = getattr(rc, 'imu_sensor_roles', None) if rc is not None else None
    return list(roles) if roles else list(IMU_ROLE_ORDER)


def _format_imu_line(payload, joint_angles=None, joint_names=None, imu_role_order=None) -> str:
    if not payload:
        return "[IMU] waiting"
    quats    = payload.get('corrected_quats') or []
    role_map = payload.get('role_by_index') or {}
    idx_map  = {role: i for i, role in role_map.items()}
    descs    = payload.get('descriptors') or []

    ordered = []
    for role in (imu_role_order or IMU_ROLE_ORDER):
        i = idx_map.get(role)
        if i is not None and i < len(quats) and i not in ordered:
            ordered.append(i)
    ordered += [i for i in range(len(quats)) if i not in ordered]

    parts = []
    for i in ordered:
        role  = role_map.get(i, '-')
        label = descs[i].get('label', '') if i < len(descs) else ''
        p, r, y = _euler_pry_deg(quats[i])
        parts.append(f"imu{i}({role}{' '+label if label else ''}) P/R/Y={p:+7.2f}/{r:+7.2f}/{y:+7.2f}")

    if joint_angles is not None:
        names = joint_names or CONTROL_JOINT_NAMES
        jt = " ".join(f"{names[i] if i < len(names) else f'j{i}'}={float(a):+7.2f}"
                      for i, a in enumerate(joint_angles))
    else:
        jt = "joints waiting"
    return "[IMU] " + " | ".join(parts) + f" deg || [joints] {jt} deg"


# ── sine excitation ───────────────────────────────────────────────────────────

class SineExcitationGenerator:
    """Randomized modulated sine excitation, after Egli & Hutter (IROS 2020 / RA-L 2022).

    Per joint:

        s(t) = A·amp · [ sin(2π·f_env·e + φ₁)
                         · sin(2π·f_car·(e + depth·sin(2π·f_rate·e)) + φ₂)
                         + noise·n(t) ]

    The published formulation fixes f_env, f_car, depth and f_rate and varies
    only φ₁/φ₂ per joint, which makes every channel the same signal at a
    different phase — and makes every recording session replay the identical
    trajectory. For training data that is the wrong kind of repeatable, so
    every parameter is drawn per joint from the ranges below, and re-drawn
    each time the excitation is switched on. Channels decorrelate, and
    successive strips explore different regions.

    The randomization is bounded, not free: the carrier is frequency-modulated,
    so its peak instantaneous frequency is f_car·(1 + depth·2π·f_rate), and
    depth is clamped to hold that under MAX_INSTANT_FREQ_HZ. Above ~2 Hz the
    excitation is past the M545 velocity-control cutoff and the valves stop
    tracking it, which would put noise rather than signal in the dataset.

    On top of the deterministic term each joint carries band-limited noise, so
    the excitation is not a pure sum of tones and the recording sees frequency
    content between the carrier harmonics. The noise is a low-pass-filtered
    Gaussian process, not white: white noise at the 100 Hz sample rate is far
    above the valve bandwidth, so it would be filtered out mechanically while
    still chattering the solenoids.

    The only operator setting left is overall amplitude — everything else is
    drawn per joint, which covers more of the input space than hand-stepping
    ever did.

    Pass ``seed`` to reproduce a session; otherwise one is drawn and recorded
    in ``self.seed``.
    """

    ENV_FREQ_HZ         = 0.02      # envelope (slow amplitude sweep)
    CARRIER_FREQ_HZ     = 1.0       # carrier centre frequency
    FM_DEPTH            = 0.99      # carrier frequency-modulation depth
    FM_RATE_HZ          = 0.1       # carrier frequency-modulation rate
    MAX_INSTANT_FREQ_HZ = 2.0       # hard ceiling on peak carrier frequency
    NOISE_CUTOFF_HZ     = 1.2       # noise low-pass corner
    NOISE_FRACTION      = 0.15      # noise std as a fraction of joint amplitude
    NOISE_CLIP_SIGMA    = 3.0       # bound on the unit-variance noise state

    # Multiplicative jitter applied to each nominal value above.
    _JITTER = {
        'f_env':   (0.80, 1.25),
        'f_car':   (0.85, 1.15),
        'f_rate':  (0.70, 1.10),
        'depth':   (0.70, 1.20),
        'f_noise': (0.70, 1.30),
        'noise':   (0.70, 1.30),
        'amp':     (0.75, 1.00),    # per-joint amplitude trim, never above 1
    }

    def __init__(self, enabled: bool = False, seed: int | None = None):
        if seed is None:
            # Draw an explicit seed rather than passing None through, so the
            # session can be reproduced from what gets printed/stored.
            seed = int(np.random.SeedSequence().entropy % (2**32))
        self.seed  = int(seed)
        self._rng  = np.random.default_rng(self.seed)

        self.enabled       = enabled
        self._amp_idx      = AMPLITUDE_PRESETS.index(0.3)
        self.amplitude_scale = AMPLITUDE_PRESETS[self._amp_idx]
        self.start_time    = None
        self._params: dict[str, dict[str, float]] = {}
        self._noise:  dict[str, float] = {}
        self._last_t: float | None = None
        self.randomize()

    def _draw_params(self) -> dict[str, float]:
        j = self._JITTER
        f_env   = self.ENV_FREQ_HZ     * self._rng.uniform(*j['f_env'])
        f_car   = self.CARRIER_FREQ_HZ * self._rng.uniform(*j['f_car'])
        f_rate  = self.FM_RATE_HZ      * self._rng.uniform(*j['f_rate'])
        depth   = self.FM_DEPTH        * self._rng.uniform(*j['depth'])
        f_noise = self.NOISE_CUTOFF_HZ * self._rng.uniform(*j['f_noise'])

        # Peak instantaneous carrier frequency is f_car·(1 + depth·2π·f_rate).
        # Clamp depth so the drawn combination cannot exceed the ceiling.
        headroom = (self.MAX_INSTANT_FREQ_HZ / f_car) - 1.0
        depth = min(depth, max(0.0, headroom / (2.0 * np.pi * f_rate)))

        return {
            'f_env':   f_env,
            'f_car':   f_car,
            'f_rate':  f_rate,
            'depth':   depth,
            'f_noise': min(f_noise, self.MAX_INSTANT_FREQ_HZ),
            'noise':   self.NOISE_FRACTION * self._rng.uniform(*j['noise']),
            'amp':     float(self._rng.uniform(*j['amp'])),
            'phi1':    float(self._rng.uniform(0.0, 2.0 * np.pi)),
            'phi2':    float(self._rng.uniform(0.0, 2.0 * np.pi)),
        }

    def randomize(self):
        """Draw a fresh independent parameter set for every joint."""
        self._params = {n: self._draw_params() for n in JOINT_NAMES}
        self._noise  = {n: 0.0 for n in JOINT_NAMES}
        self._last_t = None

    def peak_freq_hz(self, joint: str) -> float:
        """Peak instantaneous carrier frequency for a joint, for verification."""
        p = self._params[joint]
        return p['f_car'] * (1.0 + p['depth'] * 2.0 * np.pi * p['f_rate'])

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.start_time = time.perf_counter()
            self.randomize()

    def step_amplitude(self, direction: int):
        """Move one amplitude preset up or down, clamped at the ends."""
        self._amp_idx = int(np.clip(self._amp_idx + direction, 0, len(AMPLITUDE_PRESETS) - 1))
        self.amplitude_scale = AMPLITUDE_PRESETS[self._amp_idx]

    def _advance_noise(self, t: float):
        """Step each joint's noise state to time t.

        Exact-discretization OU: with alpha = exp(-dt/tau) and a
        sqrt(1-alpha²) innovation, the state is unit-variance regardless of
        dt, so loop jitter changes the noise timing but not its level.
        """
        if self._last_t is None:
            self._last_t = t
            return
        dt = float(np.clip(t - self._last_t, 1e-4, 0.1))
        self._last_t = t
        for name, p in self._params.items():
            tau   = 1.0 / (2.0 * np.pi * p['f_noise'])
            alpha = float(np.exp(-dt / tau))
            x = alpha * self._noise[name] + np.sqrt(1.0 - alpha * alpha) * self._rng.standard_normal()
            self._noise[name] = float(np.clip(x, -self.NOISE_CLIP_SIGMA, self.NOISE_CLIP_SIGMA))

    def get_signal(self, joint: str, t: float) -> float:
        """Deterministic term plus the current noise sample.

        Does not advance the noise state — get_all() does that once per tick,
        so every joint sees a consistent timebase.
        """
        if not self.enabled:
            return 0.0
        if self.start_time is None:
            self.start_time = t
        e = t - self.start_time
        p = self._params[joint]
        env = np.sin(2.0 * np.pi * p['f_env'] * e + p['phi1'])
        car = np.sin(2.0 * np.pi * p['f_car']
                     * (e + p['depth'] * np.sin(2.0 * np.pi * p['f_rate'] * e))
                     + p['phi2'])
        amp = self.amplitude_scale * p['amp']
        return float(np.clip(amp * (env * car + p['noise'] * self._noise[joint]), -1.0, 1.0))

    def get_all(self, t: float) -> dict:
        if self.enabled:
            self._advance_noise(t)
        return {n: self.get_signal(n, t) for n in JOINT_NAMES}


# ── data logger ───────────────────────────────────────────────────────────────

class DataLogger:
    """100 Hz hydraulic actuator data recorder for blackbox model training.

    CSV schema matches data_collection/benchmark_actuator_models.py and the
    IsaacLab training pipeline (Isaac-hydraulic-actuator/train.py).

    Units are the Isaac convention, not the controller convention:
      timestamp  seconds since segment start (monotonic)
      joint_pos  radians          (controller API returns degrees)
      joint_vel  rad/s            (controller API returns deg/s)
      imu_g*     rad/s            (hardware API returns deg/s)
      *_cmd_*    normalized [-1, 1]

    Command channels use hydraulic names (rotate/lift/tilt/scoop), not joint
    names (slew/boom/arm/bucket), because that is what the training and
    benchmark scripts read.
    """

    def __init__(self, output_dir: Path):
        self.output_dir  = output_dir
        self.is_logging  = False
        self.segment_id  = 0
        self.session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._clear()

    def _clear(self):
        self._t0_wall = None
        self._t0_mono = None
        self._ts:   list = []
        self._idx:  list = []
        self._seg:  list = []
        self._man:  list = []
        self._sin:  list = []
        self._com:  list = []
        self._pos:  list = []
        self._vb:   list = []
        self._va:   list = []
        self._vbkt: list = []
        self._gb:   list = []
        self._ga:   list = []
        self._gk:   list = []
        self._stale: list = []
        self._age:   list = []
        self._sine_flag: list = []
        self._sine_amp:  list = []
        self._sine_seed: list = []

    def start(self):
        self.segment_id = 0
        self._clear()
        self._t0_wall = time.time()
        self._t0_mono = time.perf_counter()
        self.is_logging = True
        print(f"\n{'='*60}\n  DATA COLLECTION STARTED\n{'='*60}\n")

    def _start_segment(self):
        self.segment_id += 1
        self._clear()
        self._t0_wall = time.time()
        self._t0_mono = time.perf_counter()
        self.is_logging = True
        print(f"[RESUME] Segment #{self.segment_id} started")

    def log_sample(self, manual: dict, sine: dict, combined: dict,
                   controller, hardware, cmd_age_s: float, cmd_stale: bool,
                   sine_enabled: bool,
                   sine_amp: float = np.nan, sine_seed: int = -1):
        if not self.is_logging:
            return

        t = time.perf_counter() - self._t0_mono

        # Controller/hardware APIs speak degrees; the dataset is radians.
        pos = np.radians(controller.get_joint_angles())
        vels, vel_age = controller.get_joint_velocities_with_age()
        if vels is not None and vel_age < 0.05:
            vb, va, vbkt = (float(np.radians(vels[1])),
                            float(np.radians(vels[2])),
                            float(np.radians(vels[3])))
        else:
            vb = va = vbkt = np.nan

        gyro = hardware.try_read_imu_gyro()
        if gyro is not None:
            g  = gyro['gyro']
            gb = list(np.radians(g[0])) if len(g) > 0 else [np.nan]*3
            ga = list(np.radians(g[1])) if len(g) > 1 else [np.nan]*3
            gk = list(np.radians(g[2])) if len(g) > 2 else [np.nan]*3
        else:
            gb = ga = gk = [np.nan, np.nan, np.nan]

        i = len(self._ts)
        self._ts.append(t);         self._idx.append(i);           self._seg.append(self.segment_id)
        self._man.append([manual.get(n, 0.0)   for n in JOINT_NAMES])
        self._sin.append([sine.get(n, 0.0)     for n in JOINT_NAMES])
        self._com.append([combined.get(n, 0.0) for n in JOINT_NAMES])
        self._pos.append(list(pos))
        self._vb.append(vb);  self._va.append(va);  self._vbkt.append(vbkt)
        self._gb.append(gb);  self._ga.append(ga);  self._gk.append(gk)
        self._stale.append(int(bool(cmd_stale)))
        self._age.append(float(cmd_age_s) if np.isfinite(cmd_age_s) else np.nan)
        self._sine_flag.append(int(bool(sine_enabled)))
        self._sine_amp.append(float(sine_amp))
        self._sine_seed.append(int(sine_seed))

    def n_samples(self) -> int:
        return len(self._ts)

    def elapsed_min(self) -> float:
        return (time.time() - self._t0_wall) / 60.0 if self._t0_wall else 0.0

    def save(self) -> Path | None:
        if not self._ts:
            print("No data to save.")
            return None

        import pandas as pd

        man = np.array(self._man);  sin = np.array(self._sin);  com = np.array(self._com)
        pos = np.array(self._pos)
        gb  = np.array(self._gb);   ga  = np.array(self._ga);   gk  = np.array(self._gk)

        df = pd.DataFrame({
            'timestamp': self._ts, 'sample_idx': self._idx, 'segment_id': self._seg,
            # JOINT_NAMES order (slew, boom, arm, bucket) → hydraulic channel names
            'manual_cmd_rotate': man[:,0], 'manual_cmd_lift':  man[:,1],
            'manual_cmd_tilt':   man[:,2], 'manual_cmd_scoop': man[:,3],
            'sine_cmd_rotate':   sin[:,0], 'sine_cmd_lift':    sin[:,1],
            'sine_cmd_tilt':     sin[:,2], 'sine_cmd_scoop':   sin[:,3],
            'combined_cmd_rotate': com[:,0], 'combined_cmd_lift':  com[:,1],
            'combined_cmd_tilt':   com[:,2], 'combined_cmd_scoop': com[:,3],
            'joint_pos_slew':   pos[:,0], 'joint_pos_boom': pos[:,1],
            'joint_pos_arm':    pos[:,2], 'joint_pos_bucket': pos[:,3],
            'joint_vel_boom': self._vb, 'joint_vel_arm': self._va, 'joint_vel_bucket': self._vbkt,
            'imu_gx_boom': gb[:,0], 'imu_gy_boom': gb[:,1], 'imu_gz_boom': gb[:,2],
            'imu_gx_arm':  ga[:,0], 'imu_gy_arm':  ga[:,1], 'imu_gz_arm':  ga[:,2],
            'imu_gx_bucket': gk[:,0], 'imu_gy_bucket': gk[:,1], 'imu_gz_bucket': gk[:,2],
            'cmd_stale': self._stale, 'cmd_age_s': self._age, 'sine_enabled': self._sine_flag,
            # Amplitude is D-pad adjustable and the seed is re-drawn on every
            # sine enable, so both are per-sample rather than per-file. The seed
            # is what lets a row's exact excitation parameters be reconstructed.
            'sine_amplitude': self._sine_amp, 'sine_seed': self._sine_seed,
        })

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"drive_log_{ts}_seg{self.segment_id:03d}.csv"
        df.to_csv(out, index=False)
        print(f"[SAVE] {len(df)} samples ({df['timestamp'].iloc[-1]/60:.2f} min) → {out}")
        return out

    def save_with_pause(self, direct) -> Path | None:
        if not self._ts:
            print("No data to save.")
            return None
        was = self.is_logging
        self.is_logging = False
        direct.clear()
        direct.send_pending()
        time.sleep(0.3)
        path = self.save()
        if was:
            self._start_segment()
        return path


# ── stub controller ───────────────────────────────────────────────────────────

class PWMOnlyController:
    """Stand-in when IMUs are disabled — no IK, joint state returns zeros."""

    def __init__(self, hardware):
        self.hardware = hardware

    def start(self):                      pass
    def stop(self):                       pass
    def suspend_ik_output(self):          pass
    def resume_ik_output(self):           pass
    def get_joint_angles(self):           return np.zeros(4, dtype=np.float32)
    def get_joint_velocities_with_age(self): return None, float('inf')
    def emergency_stop(self, reset_pump=True): self.hardware.reset(reset_pump=reset_pump)


# ── input sources ─────────────────────────────────────────────────────────────

# Button bits. 0-3 are the mask the UDP client already sends; 4-7 are the D-pad
# and are only ever set by the local pad, so a UDP run just reads them as 0.
BTN_A, BTN_B, BTN_X, BTN_Y = 0, 1, 2, 3
BTN_DPAD_UP, BTN_DPAD_DOWN, BTN_DPAD_LEFT, BTN_DPAD_RIGHT = 4, 5, 6, 7

GAMEPAD_DEADZONE_PCT = 15.0
GAMEPAD_PADDING_PCT  = 0.0


class UDPInput:
    """Axes + button mask from a remote client over the network."""

    name = "udp"

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._sock = None

    def open(self) -> bool:
        print("Waiting for remote controller...")
        self._sock = UDPSocket(local_id=2)
        self._sock.setup(self._host, self._port, inputs='<8bH', outputs='', is_server=True)
        if not self._sock.handshake(timeout=30.0):
            print("UDP handshake failed.")
            return False
        self._sock.start_receiving()
        print("Connected.")
        return True

    def poll(self):
        """-> (axes, mask). axes is None when no new packet has arrived."""
        raw = self._sock.get_latest() or []
        if not raw:
            return None, 0
        fl = UDPSocket.ints_to_floats(raw[:8])
        axes = {
            'right_rl': fl[0], 'right_ud': fl[1],
            'left_rl':  fl[3], 'left_ud':  fl[4],
            'right_paddle': fl[6], 'left_paddle': fl[7],
        }
        return axes, int(raw[8])

    def is_live(self) -> bool:
        return True     # a packet arriving *is* the liveness signal

    def close(self):
        if self._sock is not None:
            try:
                self._sock.stop_receiving()
                self._sock.close()
            except Exception:
                pass


class LocalGamepadInput:
    """Axes + button mask from a pad wired straight into this machine.

    Axis signs follow clients.input_handler.InputHandler.GAMEPAD_DIRECT, which
    is what the UDP client applies before encoding, so both sources drive the
    machine the same direction.

    Tracks are on the triggers, which only read 0..1 — a local run drives them
    forward only, unlike the paddles on the remote client.
    """

    name = "local"

    def __init__(self, deadzone: float = GAMEPAD_DEADZONE_PCT,
                 padding: float = GAMEPAD_PADDING_PCT,
                 connect_timeout_s: float = 10.0):
        self._deadzone = deadzone
        self._padding  = padding
        self._timeout  = connect_timeout_s
        self._pad      = None

    def open(self) -> bool:
        try:
            from modules.gamepad import XboxController
        except Exception as e:
            print(f"Gamepad import failed: {e}")
            return False
        try:
            self._pad = XboxController(max_reconnect=None,
                                       deadzone=self._deadzone, padding=self._padding)
        except Exception as e:
            print(f"Gamepad open failed: {e}")
            return False

        print("Waiting for gamepad...")
        deadline = time.perf_counter() + self._timeout
        while not self._pad.is_connected():
            if time.perf_counter() >= deadline:
                print(f"No gamepad within {self._timeout:.0f}s. Is it plugged in, "
                      f"and is this user in the 'input' group?")
                return False
            time.sleep(0.1)
        print("Gamepad connected.")
        return True

    def poll(self):
        s = self._pad.read()
        axes = {
            'right_rl': -float(s['RightJoystickX']),   # bucket
            'right_ud':  float(s['RightJoystickY']),   # boom
            'left_rl':  -float(s['LeftJoystickX']),    # slew
            'left_ud':  -float(s['LeftJoystickY']),    # arm
            'right_paddle': float(s['RightTrigger']),
            'left_paddle':  float(s['LeftTrigger']),
        }
        mask = 0
        for bit, key in ((BTN_A, 'A'), (BTN_B, 'B'), (BTN_X, 'X'), (BTN_Y, 'Y'),
                         (BTN_DPAD_UP,   'UpDPad'),   (BTN_DPAD_DOWN,  'DownDPad'),
                         (BTN_DPAD_LEFT, 'LeftDPad'), (BTN_DPAD_RIGHT, 'RightDPad')):
            if int(s.get(key, 0)):
                mask |= 1 << bit
        return axes, mask

    def is_live(self) -> bool:
        # read() already zeroes every axis while disconnected, so the commands
        # stay safe; reporting not-live marks those samples stale in the log.
        return self._pad.is_connected()

    def close(self):
        if self._pad is not None:
            try:
                self._pad.stop_monitoring()
            except Exception:
                pass


def make_input_source(args) -> "UDPInput | LocalGamepadInput":
    """--ip selects the remote client; without it the local pad is used."""
    if not args.ip:
        return LocalGamepadInput()
    host, port = (args.ip.rsplit(":", 1)[0], int(args.ip.rsplit(":", 1)[1])) \
        if ":" in args.ip else (args.ip, 8080)
    return UDPInput(host, port)


# ── args / profile ────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Excavator open-loop driving + hydraulic data collection. "
                    "For compensated / closed-loop driving see "
                    "control_prototype/drive_compensated.py.")
    p.add_argument("--robot", choices=[*sorted(ROBOT_PROFILES), "auto"], default="auto",
                   help="Board profile (default: auto-detect)")
    p.add_argument("--ip", default=None, metavar="HOST[:PORT]",
                   help="Listen for a remote UDP client instead of using the local gamepad")
    p.add_argument("--enable-slew",   action="store_true",
                   help="Allow slew in manual commands and sine (default: off)")
    p.add_argument("--enable-tracks", action="store_true",
                   help="Allow track drive from the triggers/paddles (default: off)")
    return p.parse_args()


def _resolve_profile(args) -> dict:
    return _resolve_board_profile(args.robot)


resolve_robot_profile = _resolve_profile


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = _parse_args()
    profile = _resolve_profile(args)
    imu_on  = bool(profile['enable_imu'])

    out_dir = LOG_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── hardware ──────────────────────────────────────────────────────────────
    from modules.hardware_interface import HardwareFaultError, HardwareInterface

    print("Initializing hardware...")
    hardware = HardwareInterface(
        config_file=profile['servo_config_file'],
        control_config_file=profile['control_config_file'],
        pump_auto_mode=False,          # fixed pump; button X toggles it on/off
        toggle_channels=True,
        stale_timeout_s=0.5,
        enable_pwm=True,
        enable_imu=imu_on,
        enable_adc=False,
        start_imu_reader=imu_on,
        start_adc_reader=False,
        cleanup_disable_osc=False,
        pwm_i2c_bus=profile['pwm_i2c_bus'],
        pwm_i2c_addr=profile['pwm_i2c_addr'],
    )

    try:
        wait_for_hardware_ready(hardware)
    except HardwareFaultError as e:
        print(f"\n*** HARDWARE FAULT ({e.subsystem}): {e.reason} ***")
        hardware.shutdown(); raise SystemExit(1)
    except TimeoutError as e:
        print(f"\n*** {e} ***")
        hardware.shutdown(); raise SystemExit(1)

    # ── controller ────────────────────────────────────────────────────────────
    print("Starting controller...")
    if imu_on:
        from modules.excavator_controller import ExcavatorController
        controller = ExcavatorController(
            hardware, config=None, enable_perf_tracking=False,
            control_config_file=profile['control_config_file'],
        )
    else:
        controller = PWMOnlyController(hardware)

    controller.start()
    if imu_on:
        time.sleep(2.0)      # numba JIT warmup
    direct = DirectController(hardware)
    controller.suspend_ik_output()

    ctrl_joint_names = _get_control_joint_names(controller)
    imu_role_order   = _get_imu_role_order(controller)

    pwm = hardware.pwm_controller

    print(f"Profile: {profile['profile_name']} | pump: fixed"
          f" | slew: {'on' if args.enable_slew else 'off'}"
          f" | tracks: {'on' if args.enable_tracks else 'off'}")

    # ── RT scheduling (Linux only) ─────────────────────────────────────────────
    try:
        from modules.rt_utils import apply_rt_to_thread, SCHED_FIFO
        apply_rt_to_thread(priority=75, policy=SCHED_FIFO, lock_memory=False)
    except Exception:
        pass

    # ── helpers ───────────────────────────────────────────────────────────────
    sine_gen = SineExcitationGenerator()
    logger   = DataLogger(out_dir)

    # ── input source ──────────────────────────────────────────────────────────
    source = make_input_source(args)
    if not source.open():
        source.close()
        direct.clear(); controller.resume_ik_output(); controller.stop()
        hardware.shutdown(); raise SystemExit(1)
    print("A=log  B=sine  X=pump  Y=reload-config"
          + ("  Dpad U/D=sine-amp\n" if source.name == "local" else "\n"))

    # ── loop state ────────────────────────────────────────────────────────────
    loop_period     = 1.0 / SAMPLING_FREQUENCY
    next_run_time   = time.perf_counter()

    right_rl = right_ud = left_rl = left_ud = right_paddle = left_paddle = 0.0
    last_cmd_mono = None
    mask_prev     = 0

    last_status_time    = time.time()
    last_strip_time     = time.time()
    print_imu           = False
    iter_count          = 0

    try:
        while True:
            now = time.time()

            # ── 1. receive ────────────────────────────────────────────────────
            axes, mask = source.poll()
            if axes is not None:
                right_rl     = axes['right_rl']
                right_ud     = axes['right_ud']
                left_rl      = axes['left_rl']
                left_ud      = axes['left_ud']
                right_paddle = axes['right_paddle']
                left_paddle  = axes['left_paddle']
                if source.is_live():
                    last_cmd_mono = time.monotonic()

                def btn(b):  return bool(mask & (1 << b))
                def prev(b): return bool(mask_prev & (1 << b))

                # A: toggle logging
                if btn(BTN_A) and not prev(BTN_A):
                    if not logger.is_logging:
                        logger.start()
                        last_strip_time = now
                    else:
                        logger.save_with_pause(direct)
                        logger.is_logging = False

                # B: toggle sine
                if btn(BTN_B) and not prev(BTN_B):
                    sine_gen.toggle()
                    if sine_gen.enabled:
                        # Params are re-drawn on every enable, so note the seed:
                        # it is what makes a strip's excitation reproducible.
                        print(f"\n[Button B] Sine ON (amp={sine_gen.amplitude_scale:.2f} "
                              f"seed={sine_gen.seed})")
                    else:
                        print("\n[Button B] Sine OFF")

                # X: pump toggle
                if btn(BTN_X) and not prev(BTN_X):
                    if pwm is not None:
                        new_state = not pwm.pump_enabled
                        hardware.set_pump_enabled(new_state)
                        print(f"\n[Button X] Pump {'ON' if new_state else 'OFF'}")

                # Y: reload servo config from disk. Outputs are neutralised
                # first — a valve calibration swap while a channel is commanded
                # open would step that valve as the new mapping takes effect.
                if btn(BTN_Y) and not prev(BTN_Y):
                    direct.clear()
                    direct.send_pending()
                    time.sleep(0.1)
                    ok = hardware.reload_config()
                    print(f"\n[Button Y] Config reload {'OK' if ok else 'FAILED'}")

                # D-pad up/down: step sine amplitude
                for bit, step in ((BTN_DPAD_UP, +1), (BTN_DPAD_DOWN, -1)):
                    if btn(bit) and not prev(bit):
                        sine_gen.step_amplitude(step)
                        print(f"\n[D-pad] Sine amplitude → {sine_gen.amplitude_scale:.2f}")

                mask_prev = mask

            # ── 2. build commands ─────────────────────────────────────────────
            is_logging = logger.is_logging

            manual = {
                'slew':   left_rl if args.enable_slew else 0.0,
                'boom':   right_ud,
                'arm':    left_ud,
                'bucket': right_rl,
            }

            t = time.perf_counter()
            sine = sine_gen.get_all(t)
            if not args.enable_slew:
                sine['slew'] = 0.0

            combined = {n: float(np.clip(manual[n] + sine[n], -1.0, 1.0)) for n in JOINT_NAMES}
            if args.enable_tracks:
                combined['trackR'] = right_paddle
                combined['trackL'] = left_paddle

            # ── 3. send ───────────────────────────────────────────────────────
            direct.give_commands(combined)
            direct.send_pending()

            # ── 4. log ────────────────────────────────────────────────────────
            joint_angles = controller.get_joint_angles()

            if is_logging:
                cmd_age_s = np.nan
                cmd_stale = True
                if last_cmd_mono is not None:
                    cmd_age_s = max(0.0, time.monotonic() - last_cmd_mono)
                    cmd_stale = cmd_age_s > COMMAND_STALE_TIMEOUT_S
                logger.log_sample(manual, sine, combined, controller, hardware,
                                  cmd_age_s, cmd_stale, sine_gen.enabled,
                                  sine_gen.amplitude_scale, sine_gen.seed)

            # ── 5. strip rollover ─────────────────────────────────────────────
            if is_logging and (now - last_strip_time) >= STRIP_MINUTES * 60:
                last_strip_time = now
                logger.save_with_pause(direct)

            # ── 6. IMU print (decimated) ──────────────────────────────────────
            iter_count += 1
            if print_imu and imu_on and (iter_count % PRINT_DECIMATION == 0):
                print(_format_imu_line(
                    hardware.read_imu_debug_quaternions(),
                    joint_angles, ctrl_joint_names, imu_role_order,
                ), end="\r", flush=True)

            # ── 7. status ─────────────────────────────────────────────────────
            if now - last_status_time >= STATUS_PRINT_INTERVAL_S:
                last_status_time = now
                pump_on  = bool(pwm and pwm.pump_enabled)
                sine_str = (f"ON amp={sine_gen.amplitude_scale:.2f} seed={sine_gen.seed}"
                            if sine_gen.enabled else "OFF")
                print(
                    f"[STATUS] pump={'ON' if pump_on else 'OFF'} | sine={sine_str} | "
                    + (f"log=ON {logger.elapsed_min():.1f}min {logger.n_samples()} samples"
                       if is_logging else "log=OFF")
                    + ("" if source.is_live() else f" | *** {source.name} INPUT LOST ***")
                )
                print(f"[JOINTS] slew={joint_angles[0]:+.1f} boom={joint_angles[1]:+.1f} "
                      f"arm={joint_angles[2]:+.1f} bucket={joint_angles[3]:+.1f} deg")

            # ── 8. timing ─────────────────────────────────────────────────────
            next_run_time += loop_period
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C).")
    finally:
        print("Shutting down...")
        source.close()
        if logger.n_samples() > 0:
            logger.is_logging = False
            direct.clear()
            direct.send_pending()
            time.sleep(0.2)
            logger.save()
        try:
            controller.emergency_stop(reset_pump=True)
        except Exception:
            pass
        try:
            controller.stop()
        except Exception:
            pass
        try:
            hardware.shutdown()
        except Exception:
            pass
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
