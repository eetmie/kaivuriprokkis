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
    Button A (bit 0): Start / Stop a recording (saves on stop)
    Button B (bit 1): Toggle sine excitation on / off (default: OFF)
    Button X (bit 2): Toggle hydraulic pump
    Button Y (bit 3): Reload servo config from disk
    D-pad Up/Down (bits 4/5): Cycle sine target channel (local pad only)

The sine has no operator-set waveform knobs — amplitude, frequencies, phases
and noise are all drawn per joint, with a fresh seed per recording, so each
file is an independent sample of the input space. The seed is printed and
written to every logged row, which is what makes a run reconstructible.

The one manual sine control is where it goes: the D-pad cycles through
all / lift / tilt / scoop and the three two-channel pairs. Single channels
isolate one actuator; the pairs capture cross-coupling.

A recording auto-stops after RECORD_MINUTES and does not restart itself —
press A again for the next one. The gap is deliberate: it lets the hydraulics
cool, so a long session is a series of comparable sets rather than a slow
thermal drift.

Each recording writes two files:

    drive_log_*.csv  hydraulic commands + joint state at the 100 Hz control rate
    imu_raw_*.csv    every IMU frame at the full 200 Hz stream rate

The raw strip holds each sensor's fused quaternion next to the gyro and accel
that produced it, in the firmware's own units (dps and g, which is what Fusion
takes). That makes it replayable: re-run the AHRS offline at a different gain
and compare against the firmware quaternion, or check peak magnitudes against
the gyro_range_dps / accel_range_g columns to see whether a range is clipping.
Join the two files on the device timestamp (imu_device_ts_us / device_ts_us).
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
RECORD_MINUTES             = 10.0   # logging auto-stops after this long

LOG_OUTPUT_DIR = Path(__file__).parent / "data_collection" / "hydraulic_data"

JOINT_NAMES       = ['slew', 'boom', 'arm', 'bucket']
G_TO_MS2          = 9.80665   # firmware reports accel in g; the dataset is SI
CONTROL_JOINT_NAMES = ['slew', 'lift', 'arm', 'bucket']

# Where the sine excitation is routed, stepped with the D-pad. Named in
# hydraulic terms (lift/tilt/scoop) to match the logged command channels, not
# the joint names. Driving one channel at a time isolates that actuator's
# response; the pairs record the cross-coupling a single-channel strip cannot
# show, since a real dig loads several cylinders at once.
SINE_TARGET_MODES = [
    ('all',        ('slew', 'boom', 'arm', 'bucket')),
    ('lift',       ('boom',)),
    ('tilt',       ('arm',)),
    ('scoop',      ('bucket',)),
    ('lift+tilt',  ('boom', 'arm')),
    ('lift+scoop', ('boom', 'bucket')),
    ('tilt+scoop', ('arm', 'bucket')),
]
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
    depth is clamped to hold that under MAX_INSTANT_FREQ_HZ.

    Rates are set for this machine, not the published M545 figures. The original
    1 Hz carrier (2 Hz ceiling) drove the valves faster than the hydraulics can
    follow, and a model trained on that learns to jitter: it sees command energy
    that never became motion, so the only way to fit it is high-frequency
    chatter. The carrier now sits near 0.22 Hz with a 0.5 Hz ceiling — roughly a
    4 s stroke — which the cylinders actually track. If the recorded motion still
    looks smoother than the command, lower CARRIER_FREQ_HZ further; these four
    constants are the whole knob.

    On top of the deterministic term each joint carries band-limited noise, so
    the excitation is not a pure sum of tones and the recording sees frequency
    content between the carrier harmonics. The noise is a low-pass-filtered
    Gaussian process, not white: white noise at the 100 Hz sample rate is far
    above the valve bandwidth, so it would be filtered out mechanically while
    still chattering the solenoids.

    That filter is two-pole, not one. A single OU stage rolls off at only
    -20 dB/decade, which still leaves the sample-to-sample jump large — the
    state moved ~0.17 per tick on a unit-variance signal, so most of the
    command's *rate* was noise even after the carrier was slowed. Cascading two
    stages gives -40 dB/decade and drops that by roughly an order of magnitude,
    which is the difference between noise the hydraulics integrate away and
    noise that just buzzes the valves.

    Nothing about the waveform is operator-set any more — amplitude is drawn per
    joint alongside every other parameter. The one manual control left is which
    channels the excitation is routed to, which changes what is being measured
    rather than how it is shaped.

    Pass ``seed`` to reproduce a session; otherwise one is drawn and recorded
    in ``self.seed``.
    """

    ENV_FREQ_HZ         = 0.015     # envelope (slow amplitude sweep)
    CARRIER_FREQ_HZ     = 0.22      # carrier centre frequency (~4.5 s stroke)
    FM_DEPTH            = 0.99      # carrier frequency-modulation depth
    FM_RATE_HZ          = 0.04      # carrier frequency-modulation rate
    MAX_INSTANT_FREQ_HZ = 0.5       # hard ceiling on peak carrier frequency
    NOISE_CUTOFF_HZ     = 0.25      # noise low-pass corner
    NOISE_FRACTION      = 0.10      # noise std as a fraction of joint amplitude
    NOISE_CLIP_SIGMA    = 3.0       # bound on the unit-variance noise state

    # Absolute per-joint amplitude, drawn fresh with everything else. The floor
    # stays above the valve deadband so a joint is never commanded into a range
    # where nothing moves; the ceiling leaves room for a manual input on top
    # before the sum clips.
    AMPLITUDE_RANGE     = (0.15, 0.85)

    # Multiplicative jitter applied to each nominal value above.
    _JITTER = {
        'f_env':   (0.80, 1.25),
        'f_car':   (0.85, 1.15),
        'f_rate':  (0.70, 1.10),
        'depth':   (0.70, 1.20),
        'f_noise': (0.70, 1.30),
        'noise':   (0.70, 1.30),
    }

    def __init__(self, enabled: bool = False, seed: int | None = None):
        if seed is None:
            # Draw an explicit seed rather than passing None through, so the
            # session can be reproduced from what gets printed/stored.
            seed = int(np.random.SeedSequence().entropy % (2**32))
        self.seed  = int(seed)
        self._rng  = np.random.default_rng(self.seed)

        self.enabled       = enabled
        self.target_idx    = 0
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
            'amp':     float(self._rng.uniform(*self.AMPLITUDE_RANGE)),
            'phi1':    float(self._rng.uniform(0.0, 2.0 * np.pi)),
            'phi2':    float(self._rng.uniform(0.0, 2.0 * np.pi)),
        }

    def randomize(self):
        """Draw a fresh independent parameter set for every joint."""
        self._params = {n: self._draw_params() for n in JOINT_NAMES}
        self._noise  = {n: 0.0 for n in JOINT_NAMES}
        self._noise1 = {n: 0.0 for n in JOINT_NAMES}   # first filter stage
        self._last_t = None

    def peak_freq_hz(self, joint: str) -> float:
        """Peak instantaneous carrier frequency for a joint, for verification."""
        p = self._params[joint]
        return p['f_car'] * (1.0 + p['depth'] * 2.0 * np.pi * p['f_rate'])

    def reseed(self, seed: int | None = None):
        """Start a fresh random stream and redraw every parameter.

        randomize() alone keeps drawing from the same generator, so a session
        would walk deterministically down one sequence. Reseeding per recording
        makes each file an independent sample of the parameter space rather than
        the next step of a single long draw.
        """
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**32))
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.randomize()

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.start_time = time.perf_counter()
            self.randomize()

    @property
    def target_name(self) -> str:
        return SINE_TARGET_MODES[self.target_idx][0]

    @property
    def target_joints(self) -> tuple:
        return SINE_TARGET_MODES[self.target_idx][1]

    def step_target(self, direction: int):
        """Cycle which channels the excitation drives, wrapping at both ends."""
        self.target_idx = (self.target_idx + direction) % len(SINE_TARGET_MODES)

    def _advance_noise(self, t: float):
        """Step each joint's two-pole noise filter to time t.

        Stage one is an exact-discretization OU: with alpha = exp(-dt/tau) and a
        sqrt(1-alpha²) innovation, the state is unit-variance regardless of dt,
        so loop jitter changes the noise timing but not its level.

        Stage two feeds that through the same pole again. Driving an AR(1) with
        an AR(1) of the same coefficient has stationary variance
        (1+alpha²)/(1-alpha²)² per unit of input, so the gain below is its
        inverse square root — that keeps the output unit-variance too, and
        NOISE_FRACTION keeps meaning the same thing at any dt or corner.
        """
        if self._last_t is None:
            self._last_t = t
            return
        dt = float(np.clip(t - self._last_t, 1e-4, 0.1))
        self._last_t = t
        for name, p in self._params.items():
            tau   = 1.0 / (2.0 * np.pi * p['f_noise'])
            alpha = float(np.exp(-dt / tau))
            x1 = alpha * self._noise1[name] + np.sqrt(1.0 - alpha * alpha) * self._rng.standard_normal()
            self._noise1[name] = float(np.clip(x1, -self.NOISE_CLIP_SIGMA, self.NOISE_CLIP_SIGMA))

            gain = (1.0 - alpha * alpha) / np.sqrt(1.0 + alpha * alpha)
            x2 = alpha * self._noise[name] + gain * self._noise1[name]
            self._noise[name] = float(np.clip(x2, -self.NOISE_CLIP_SIGMA, self.NOISE_CLIP_SIGMA))

    def get_signal(self, joint: str, t: float) -> float:
        """Deterministic term plus the current noise sample.

        Does not advance the noise state — get_all() does that once per tick,
        so every joint sees a consistent timebase.
        """
        if not self.enabled or joint not in self.target_joints:
            return 0.0
        if self.start_time is None:
            self.start_time = t
        e = t - self.start_time
        p = self._params[joint]
        env = np.sin(2.0 * np.pi * p['f_env'] * e + p['phi1'])
        car = np.sin(2.0 * np.pi * p['f_car']
                     * (e + p['depth'] * np.sin(2.0 * np.pi * p['f_rate'] * e))
                     + p['phi2'])
        return float(np.clip(p['amp'] * (env * car + p['noise'] * self._noise[joint]), -1.0, 1.0))

    def get_all(self, t: float) -> dict:
        if self.enabled:
            self._advance_noise(t)
        return {n: self.get_signal(n, t) for n in JOINT_NAMES}


# ── data logger ───────────────────────────────────────────────────────────────

class DataLogger:
    """100 Hz hydraulic actuator data recorder for blackbox model training.

    CSV schema matches data_collection/benchmark_actuator_models.py and the
    IsaacLab training pipeline (Isaac-hydraulic-actuator/train.py).

    One recording is one file. Logging stops on its own at RECORD_MINUTES rather
    than rolling into a numbered continuation, so there is no segment concept
    here — the operator decides when the machine has cooled enough to start the
    next one.

    Units are the Isaac convention, not the controller convention:
      timestamp  seconds since recording start (monotonic)
      joint_pos  radians          (controller API returns degrees)
      joint_vel  rad/s            (controller API returns deg/s)
      imu_g*     rad/s            (hardware API returns deg/s)
      *_cmd_*    normalized [-1, 1]

    Command channels use hydraulic names (rotate/lift/tilt/scoop), not joint
    names (slew/boom/arm/bucket), because that is what the training and
    benchmark scripts read.
    """

    def __init__(self, output_dir: Path, imu_roles=None, stream_info_fn=None):
        self.output_dir  = output_dir
        self.is_logging  = False
        self.session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Set when IMUs are active: the raw strip is written alongside the
        # hydraulic one and needs the sensor role order plus the firmware's
        # reported full scales to be interpretable.
        self.imu_roles      = list(imu_roles) if imu_roles else []
        self.stream_info_fn = stream_info_fn
        self._clear()

    def _clear(self):
        self._t0_wall = None
        self._t0_mono = None
        self._ts:   list = []
        self._idx:  list = []
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
        self._ab:   list = []
        self._aa:   list = []
        self._ak:   list = []
        # Base/cabin IMU. Unused by the current training set, but it is the only
        # sensor that observes slew and machine tilt, so recording it now avoids
        # a re-collection when that becomes interesting.
        self._gbase: list = []
        self._abase: list = []
        self._stale: list = []
        self._age:   list = []
        self._sine_flag:   list = []
        self._sine_target: list = []
        self._sine_seed:   list = []
        self._dev_ts:      list = []
        self._clear_imu_raw()

    def start(self):
        self._clear()
        self._t0_wall = time.time()
        self._t0_mono = time.perf_counter()
        self.is_logging = True
        print(f"\n{'='*60}\n  DATA COLLECTION STARTED\n{'='*60}\n")

    def log_sample(self, manual: dict, sine: dict, combined: dict,
                   controller, hardware, cmd_age_s: float, cmd_stale: bool,
                   sine_enabled: bool,
                   sine_target: str = "", sine_seed: int = -1):
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

        nan3 = [np.nan, np.nan, np.nan]
        gyro = hardware.try_read_imu_gyro()
        if gyro is not None:
            g  = gyro['gyro']
            gb = list(np.radians(g[0])) if len(g) > 0 else list(nan3)
            ga = list(np.radians(g[1])) if len(g) > 1 else list(nan3)
            gk = list(np.radians(g[2])) if len(g) > 2 else list(nan3)
            # Firmware reports accel in g; the dataset is SI throughout, matching
            # the dps -> rad/s conversion above.
            a = gyro.get('accel')
            ab = list(np.multiply(a[0], G_TO_MS2)) if a and len(a) > 0 else list(nan3)
            aa = list(np.multiply(a[1], G_TO_MS2)) if a and len(a) > 1 else list(nan3)
            ak = list(np.multiply(a[2], G_TO_MS2)) if a and len(a) > 2 else list(nan3)
            bg = gyro.get('base_gyro')
            ba = gyro.get('base_accel')
            gbase = list(np.radians(bg)) if bg is not None else list(nan3)
            abase = list(np.multiply(ba, G_TO_MS2)) if ba is not None else list(nan3)
            dev_ts = gyro.get('device_timestamp_us')
        else:
            gb = ga = gk = list(nan3)
            ab = aa = ak = list(nan3)
            gbase = abase = list(nan3)
            dev_ts = None

        i = len(self._ts)
        self._ts.append(t);         self._idx.append(i)
        self._man.append([manual.get(n, 0.0)   for n in JOINT_NAMES])
        self._sin.append([sine.get(n, 0.0)     for n in JOINT_NAMES])
        self._com.append([combined.get(n, 0.0) for n in JOINT_NAMES])
        self._pos.append(list(pos))
        self._vb.append(vb);  self._va.append(va);  self._vbkt.append(vbkt)
        self._gb.append(gb);  self._ga.append(ga);  self._gk.append(gk)
        self._ab.append(ab);  self._aa.append(aa);  self._ak.append(ak)
        self._gbase.append(gbase);  self._abase.append(abase)
        self._stale.append(int(bool(cmd_stale)))
        self._age.append(float(cmd_age_s) if np.isfinite(cmd_age_s) else np.nan)
        self._sine_flag.append(int(bool(sine_enabled)))
        self._sine_target.append(str(sine_target))
        self._sine_seed.append(int(sine_seed))
        self._dev_ts.append(np.nan if dev_ts is None else float(dev_ts))

    def n_samples(self) -> int:
        return len(self._ts)

    def elapsed_min(self) -> float:
        return (time.time() - self._t0_wall) / 60.0 if self._t0_wall else 0.0

    def save(self) -> Path | None:
        if not self._ts:
            print("No data to save.")
            self._save_imu_raw_strip()
            return None

        import pandas as pd

        man = np.array(self._man);  sin = np.array(self._sin);  com = np.array(self._com)
        pos = np.array(self._pos)
        gb  = np.array(self._gb);   ga  = np.array(self._ga);   gk  = np.array(self._gk)
        ab  = np.array(self._ab);   aa  = np.array(self._aa);   ak  = np.array(self._ak)
        gbs = np.array(self._gbase); abs_ = np.array(self._abase)

        df = pd.DataFrame({
            'timestamp': self._ts, 'sample_idx': self._idx,
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
            # Accel in m/s^2, mounting-offset corrected like the gyro above.
            'imu_ax_boom': ab[:,0], 'imu_ay_boom': ab[:,1], 'imu_az_boom': ab[:,2],
            'imu_ax_arm':  aa[:,0], 'imu_ay_arm':  aa[:,1], 'imu_az_arm':  aa[:,2],
            'imu_ax_bucket': ak[:,0], 'imu_ay_bucket': ak[:,1], 'imu_az_bucket': ak[:,2],
            # Base/cabin IMU: not a joint, so it has no pos/vel counterpart.
            'imu_gx_base': gbs[:,0], 'imu_gy_base': gbs[:,1], 'imu_gz_base': gbs[:,2],
            'imu_ax_base': abs_[:,0], 'imu_ay_base': abs_[:,1], 'imu_az_base': abs_[:,2],
            'cmd_stale': self._stale, 'cmd_age_s': self._age, 'sine_enabled': self._sine_flag,
            # Pico clock for the IMU sample this row saw. Joins these rows to the
            # companion imu_raw_*.csv, which is timestamped on the same clock.
            'imu_device_ts_us': self._dev_ts,
            # The target is D-pad switchable mid-recording, so it is per-sample
            # rather than per-file — rows must be groupable by which channels
            # were actually driven. Amplitude is no longer a column: it is drawn
            # per joint, and the seed reconstructs it along with every other
            # waveform parameter.
            'sine_target': self._sine_target, 'sine_seed': self._sine_seed,
        })

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"drive_log_{ts}.csv"
        df.to_csv(out, index=False)
        print(f"[SAVE] {len(df)} samples ({df['timestamp'].iloc[-1]/60:.2f} min) → {out}")
        self._save_imu_raw_strip()
        return out

    def _save_imu_raw_strip(self) -> None:
        """Write the companion raw IMU strip, if IMU capture is active."""
        if not self.imu_roles or not self._imu_ts:
            return
        info = self.stream_info_fn() if self.stream_info_fn is not None else {}
        self.save_imu_raw(self.imu_roles, info or {})

    def _clear_imu_raw(self):
        self._imu_ts:    list = []
        self._imu_vals:  list = []

    def log_imu_raw(self, frames, n_sensors: int):
        """Buffer raw IMU frames drained from the reader.

        One row per frame at the stream's own rate, not the control rate — the
        control loop runs at 100 Hz while the Pico streams 200 Hz, and halving
        the sample rate of an AHRS input changes the very integration behaviour
        a gain sweep is trying to measure.
        """
        if not self.is_logging:
            return
        for ts_us, packets in frames:
            row = []
            for i in range(n_sensors):
                pkt = packets[i] if i < len(packets) else []
                # Old firmware sends 7 floats; pad so the row width is fixed.
                row.extend(pkt[:10] + [np.nan] * (10 - len(pkt[:10])))
            self._imu_ts.append(int(ts_us))
            self._imu_vals.append(row)

    def n_imu_raw_samples(self) -> int:
        return len(self._imu_ts)

    def save_imu_raw(self, roles: list[str], stream_info: dict) -> Path | None:
        """Write the raw IMU strip: quaternion + the gyro/accel that produced it.

        Kept out of the hydraulic CSV rather than bolted onto it — the two run at
        different rates, and the hydraulic schema is what the training and
        benchmark scripts read. Join on device_ts_us against the hydraulic log's
        imu_device_ts_us column.

        Units are the firmware's, not the Isaac convention used by the hydraulic
        log: gyro in dps and accel in g, which is what Fusion's AHRS takes, so
        an offline replay can feed these columns in without converting.
        """
        if not self._imu_ts:
            return None

        import pandas as pd

        vals = np.array(self._imu_vals, dtype=np.float64)
        cols = {'device_ts_us': self._imu_ts}
        for i, role in enumerate(roles):
            base = i * 10
            for j, name in enumerate(('qw', 'qx', 'qy', 'qz',
                                      'gx_dps', 'gy_dps', 'gz_dps',
                                      'ax_g', 'ay_g', 'az_g')):
                cols[f'imu_{role}_{name}'] = vals[:, base + j]
        df = pd.DataFrame(cols)

        ranges = stream_info.get('ranges') or {}
        # Full scales are constant for a run; carrying them per row keeps the
        # file self-describing, so headroom can be judged without knowing which
        # firmware was flashed.
        df['gyro_range_dps'] = ranges.get('gyro_dps', np.nan)
        df['accel_range_g']  = ranges.get('accel_g', np.nan)

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"imu_raw_{ts}.csv"
        df.to_csv(out, index=False)
        span_s = (df['device_ts_us'].iloc[-1] - df['device_ts_us'].iloc[0]) / 1e6
        rate = len(df) / span_s if span_s > 0 else float('nan')
        # A non-zero count means the reader's buffer overflowed between drains,
        # so the strip has gaps — check device_ts_us deltas before trusting it.
        dropped = int(stream_info.get('capture_dropped', 0) or 0)
        drop_note = f", {dropped} dropped since capture start" if dropped else ""
        print(f"[SAVE] {len(df)} IMU frames ({rate:.0f} Hz{drop_note}) → {out}")
        return out

    def stop_and_save(self, direct) -> Path | None:
        """End the recording: neutralise the valves, then write both strips.

        Outputs are zeroed before the write because saving blocks for a moment,
        and a joint left commanded would keep moving through it.
        """
        self.is_logging = False
        if not self._ts and not self._imu_ts:
            print("No data to save.")
            return None
        direct.clear()
        direct.send_pending()
        time.sleep(0.3)
        return self.save()


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

    # Raw IMU strips are only meaningful when IMUs are actually streaming.
    imu_stream = hardware.imu_stream_info() if imu_on else {}
    imu_capture_roles = imu_stream.get('roles_by_index', []) if imu_on else []
    logger = DataLogger(out_dir,
                        imu_roles=imu_capture_roles,
                        stream_info_fn=hardware.imu_stream_info if imu_on else None)

    if imu_on:
        rng = imu_stream.get('ranges')
        if not imu_stream.get('has_accel'):
            print("[IMU] Firmware sends no accelerometer — raw strips will hold "
                  "quaternion + gyro only. Reflash pico_imu_reader for accel.")
        if rng:
            print(f"[IMU] Full scale: gyro +/-{rng['gyro_dps']:.0f} dps, "
                  f"accel +/-{rng['accel_g']:.0f} g")

    # ── input source ──────────────────────────────────────────────────────────
    source = make_input_source(args)
    if not source.open():
        source.close()
        direct.clear(); controller.resume_ik_output(); controller.stop()
        hardware.shutdown(); raise SystemExit(1)
    print("A=log  B=sine  X=pump  Y=reload-config"
          + ("  Dpad U/D=sine-target\n" if source.name == "local" else "\n"))

    # ── loop state ────────────────────────────────────────────────────────────
    loop_period     = 1.0 / SAMPLING_FREQUENCY
    next_run_time   = time.perf_counter()

    right_rl = right_ud = left_rl = left_ud = right_paddle = left_paddle = 0.0
    last_cmd_mono = None
    mask_prev     = 0

    last_status_time    = time.time()
    record_start_time   = None
    print_imu           = False
    iter_count          = 0

    n_imu_sensors = len(imu_capture_roles)

    def drain_imu_raw():
        """Move buffered raw IMU frames into the logger.

        Must run before a save: the reader's buffer is bounded and saving clears
        the logger's, so anything left undrained would be dropped outright.
        """
        if not (imu_on and logger.is_logging):
            return
        frames = hardware.drain_imu_raw_capture()
        if frames:
            logger.log_imu_raw(frames, n_imu_sensors)

    def start_recording():
        # A fresh seed per recording, so each file is an independent draw from
        # the parameter space rather than the next step of one long sequence.
        sine_gen.reseed()
        if imu_on:
            hardware.start_imu_raw_capture()   # drops anything buffered earlier
        logger.start()
        print(f"[REC] target={sine_gen.target_name} seed={sine_gen.seed} "
              f"| auto-stops after {RECORD_MINUTES:.0f} min")

    def stop_recording(reason: str):
        drain_imu_raw()
        logger.stop_and_save(direct)
        if imu_on:
            hardware.stop_imu_raw_capture()
        print(f"[REC] stopped ({reason}). Press A to record the next set.")

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

                # A: start / stop a recording
                if btn(BTN_A) and not prev(BTN_A):
                    if not logger.is_logging:
                        start_recording()
                        record_start_time = now
                    else:
                        stop_recording("button A")
                        record_start_time = None

                # B: toggle sine
                if btn(BTN_B) and not prev(BTN_B):
                    sine_gen.toggle()
                    if sine_gen.enabled:
                        # Params are re-drawn on every enable, so note the seed:
                        # it is what makes a strip's excitation reproducible.
                        print(f"\n[Button B] Sine ON (target={sine_gen.target_name} "
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

                # D-pad up/down: cycle which channels the sine drives
                for bit, step in ((BTN_DPAD_UP, +1), (BTN_DPAD_DOWN, -1)):
                    if btn(bit) and not prev(bit):
                        sine_gen.step_target(step)
                        print(f"\n[D-pad] Sine target → {sine_gen.target_name}")

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
                                  sine_gen.target_name, sine_gen.seed)
                # Every IMU frame since the last tick, not just the newest one.
                drain_imu_raw()

            # ── 5. auto-stop ──────────────────────────────────────────────────
            # Ends the recording outright rather than rolling into a
            # continuation file: the pause between sets is what keeps the
            # hydraulics from heat-soaking, and data taken hot is not what the
            # model should be learning from.
            if is_logging and record_start_time is not None \
                    and (now - record_start_time) >= RECORD_MINUTES * 60:
                stop_recording(f"{RECORD_MINUTES:.0f} min reached")
                record_start_time = None

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
                sine_str = (f"ON target={sine_gen.target_name} seed={sine_gen.seed}"
                            if sine_gen.enabled else f"OFF (target={sine_gen.target_name})")
                print(
                    f"[STATUS] pump={'ON' if pump_on else 'OFF'} | sine={sine_str} | "
                    + (f"log=ON {logger.elapsed_min():.1f}/{RECORD_MINUTES:.0f}min "
                       f"{logger.n_samples()} samples"
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
        drain_imu_raw()
        if logger.n_samples() > 0 or logger.n_imu_raw_samples() > 0:
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
