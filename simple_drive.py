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
    python simple_drive.py --suffix slew          # label this run's strips

Input source:
    Default is a gamepad wired straight into this machine. Passing --ip
    switches to a remote client over UDP (clients/client_gui.py).

Button Controls:
    Button A (bit 0): Start / Stop a recording (saves on stop)
    Button B (bit 1): Toggle sine excitation on / off (default: OFF)
    Button X (bit 2): Toggle hydraulic pump
    Button Y (bit 3): Reload servo config from disk
    D-pad Up/Down (bits 4/5): Cycle sine target channel (local pad only)
    Bumpers (local pad only, --enable-tracks): hold the bumper on a side to
        reverse that track; the trigger on the same side still sets speed.
        Released = forward, held = reverse.

The sine has no operator-set waveform knobs — amplitude, frequencies, phases
and noise are all drawn per joint, with a fresh seed per recording, so each
file is an independent sample of the input space. The seed is printed and
written to every logged row, which is what makes a run reconstructible.

The one manual sine control is where it goes: the D-pad cycles through
all / lift / tilt / scoop and the three two-channel pairs. Single channels
isolate one actuator; the pairs capture cross-coupling. Slew is not in any of
them, 'all' included -- under --enable-slew it is appended as its own solo
mode, because it is a separate drive from the boom cylinders and mixing the
two records coupling between systems that share no model.

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

Both files carry the Pico's own clock, so they join on it. The drive log has two
such columns, because a row is built from two separate reads: state_imu_ts_us is
the IMU frame the logged *pose* was computed from, imu_device_ts_us the frame
its gyro/accel columns came from. They usually agree; when the loop straddles a
stream update they will not, and the row says so rather than implying a
coherence it does not have. Alongside them, state_age_s and vel_age_s record how
stale each reading was when the row was written -- the control thread publishes
into a latest-value cache that this loop samples at its own rate, so without
them a repeated sample and a fresh one look identical afterwards.
"""

from __future__ import annotations

import re
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

# ── log schema ───────────────────────────────────────────────────────────────
#
# The CSV names command channels hydraulically (rotate/lift/tilt/scoop) while the
# controller speaks joint names (slew/boom/arm/bucket). This is the one place
# that mapping is written down: the recorder builds every command column from it
# rather than repeating a positional index per column.
COMMAND_CHANNELS = (
    ('slew',   'rotate'),
    ('boom',   'lift'),
    ('arm',    'tilt'),
    ('bucket', 'scoop'),
)

# Joints whose finite-difference velocity is logged. Slew has none: its angle is
# an absolute world yaw with no zeroing anywhere in the stack, so it is not
# comparable across sessions and the model does not use it.
VELOCITY_JOINTS = ('boom', 'arm', 'bucket')

# IMU roles whose gyro/accel vectors go into the drive log, in stream order. The
# base/cabin sensor is not a joint, so it has no pos/vel counterpart -- it is
# recorded because it is the only sensor that observes slew and machine tilt.
IMU_VECTOR_ROLES = ('boom', 'arm', 'bucket', 'base')

# Where the sine excitation is routed, stepped with the D-pad. Named in
# hydraulic terms (lift/tilt/scoop) to match the logged command channels, not
# the joint names. Driving one channel at a time isolates that actuator's
# response; the pairs record the cross-coupling a single-channel strip cannot
# show, since a real dig loads several cylinders at once.
#
# Slew appears in none of these, 'all' included. It is a separate drive from the
# three boom cylinders — different actuator, and its only observation is an
# absolute world yaw that is not comparable across sessions (see
# VELOCITY_JOINTS) — so a strip that swings the cabin while the arm works
# records cross-coupling between systems that do not share a model. Slew is
# excited on its own or not at all, via SINE_SLEW_MODE below.
SINE_TARGET_MODES = [
    ('all',        ('boom', 'arm', 'bucket')),
    ('lift',       ('boom',)),
    ('tilt',       ('arm',)),
    ('scoop',      ('bucket',)),
    ('lift+tilt',  ('boom', 'arm')),
    ('lift+scoop', ('boom', 'bucket')),
    ('tilt+scoop', ('arm', 'bucket')),
]

# Appended to the D-pad cycle only under --enable-slew. Off by default, so
# without the flag the mode is not merely inert but absent: the operator cannot
# step onto a target that will not move.
SINE_SLEW_MODE = ('slew', ('slew',))


def sine_target_modes(enable_slew: bool = False) -> list:
    """The D-pad cycle for this run.

    Solo and last, so the hydraulic cycle keeps the index order every existing
    strip was recorded under and slew is something you step past the end into.
    """
    return [*SINE_TARGET_MODES, SINE_SLEW_MODE] if enable_slew else list(SINE_TARGET_MODES)
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
    chatter. The carrier sits at 0.35 Hz — roughly a 3 s stroke — under a 0.9 Hz
    ceiling, still well inside what the cylinders track. If the recorded motion
    looks smoother than the command, lower CARRIER_FREQ_HZ; these four constants
    are the whole knob.

    The f_car jitter is deliberately wide (0.6–1.6×, a 1.8–4.5 s stroke) rather
    than the ±15% it used to be. Narrow jitter made every strip the same tempo
    at a different phase, which is variety the model cannot learn anything from;
    the spread is what makes successive recordings independent samples of the
    frequency axis rather than repeats. The FM depth clamp still bounds the top
    end, so widening the draw cannot push a joint past the ceiling.

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

    ``enable_slew`` fixes the D-pad cycle for the life of the generator: without
    it the slew mode is absent rather than inert, so a machine started without
    the flag has no target the operator can step onto that will not move.

    Pass ``seed`` to reproduce a session; otherwise one is drawn and recorded
    in ``self.seed``.
    """

    ENV_FREQ_HZ         = 0.03      # envelope (slow amplitude sweep, ~33 s)
    CARRIER_FREQ_HZ     = 0.35      # carrier centre frequency (~2.9 s stroke)
    FM_DEPTH            = 0.99      # carrier frequency-modulation depth
    FM_RATE_HZ          = 0.04      # carrier frequency-modulation rate
    MAX_INSTANT_FREQ_HZ = 0.9       # hard ceiling on peak carrier frequency
    NOISE_CUTOFF_HZ     = 0.25      # noise low-pass corner
    NOISE_FRACTION      = 0.10      # noise std as a fraction of joint amplitude
    NOISE_CLIP_SIGMA    = 3.0       # bound on the unit-variance noise state

    # Absolute per-joint amplitude, drawn fresh with everything else. The floor
    # stays well above the valve deadband so a joint is never commanded into a
    # range where nothing moves. The ceiling is the full valve range: the
    # envelope multiplies the carrier, so a joint only reaches its drawn
    # amplitude at an envelope peak — mean |command| lands near 0.41·amp — and
    # a ceiling held back to leave manual headroom just costs stroke everywhere
    # for a sum the main loop clips to [-1, 1] anyway.
    AMPLITUDE_RANGE     = (0.35, 1.0)

    # Multiplicative jitter applied to each nominal value above.
    _JITTER = {
        'f_env':   (0.60, 1.50),
        'f_car':   (0.60, 1.60),
        'f_rate':  (0.70, 1.10),
        'depth':   (0.70, 1.20),
        'f_noise': (0.70, 1.30),
        'noise':   (0.70, 1.30),
    }

    def __init__(self, enabled: bool = False, seed: int | None = None,
                 enable_slew: bool = False):
        if seed is None:
            # Draw an explicit seed rather than passing None through, so the
            # session can be reproduced from what gets printed/stored.
            seed = int(np.random.SeedSequence().entropy % (2**32))
        self.seed  = int(seed)
        self._rng  = np.random.default_rng(self.seed)

        # Fixed for the life of the generator: the D-pad cycle is a run-level
        # decision, so slew cannot appear mid-recording on a machine that was
        # started without it.
        self.modes         = sine_target_modes(enable_slew)
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
        return self.modes[self.target_idx][0]

    @property
    def target_joints(self) -> tuple:
        return self.modes[self.target_idx][1]

    def step_target(self, direction: int):
        """Cycle which channels the excitation drives, wrapping at both ends."""
        self.target_idx = (self.target_idx + direction) % len(self.modes)

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

def _clean_suffix(raw: str | None) -> str:
    """Normalize a --suffix into a filename tail, with its leading underscore.

    Sanitized rather than trusted: the value lands in a path, and a stray slash
    or space would either scatter strips into unintended directories or produce
    names the training scripts have to be quoted around. Anything outside
    [A-Za-z0-9._-] collapses to a single dash.

    Returns "" for empty or all-punctuation input, which restores the plain
    ``drive_log_<ts>.csv`` name rather than leaving a dangling underscore.
    """
    if not raw:
        return ""
    kept = re.sub(r'[^A-Za-z0-9._-]+', '-', raw.strip()).strip('-_.')
    return f"_{kept}" if kept else ""


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
      imu_a*     m/s^2            (firmware reports g)
      *_cmd_*    normalized [-1, 1]
      *_age_s    seconds          how stale that reading was when the row was written
      *_ts_us    microseconds     Pico device clock, -1 when never reported

    Command channels use hydraulic names (rotate/lift/tilt/scoop), not joint
    names (slew/boom/arm/bucket), because that is what the training and
    benchmark scripts read. COMMAND_CHANNELS is the mapping.

    The schema lives in one place, :meth:`_build_row`. Samples accumulate as one
    list per named column, so adding a channel is one line there rather than a
    matching edit in three.
    """

    def __init__(self, output_dir: Path, imu_roles=None, stream_info_fn=None,
                 suffix: str = ""):
        self.output_dir  = output_dir
        self.is_logging  = False
        self.session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Operator label appended to every strip this run writes, so a special
        # recording is identifiable from the filename alone. Carries its own
        # leading underscore, or is "" when unset.
        self.suffix      = _clean_suffix(suffix)
        # Set when IMUs are active: the raw strip is written alongside the
        # hydraulic one and needs the sensor role order plus the firmware's
        # reported full scales to be interpretable.
        self.imu_roles      = list(imu_roles) if imu_roles else []
        self.stream_info_fn = stream_info_fn
        self._clear()

    def _clear(self):
        self._t0_wall = None
        self._t0_mono = None
        # One list per CSV column, keyed by the column's own name. The schema is
        # therefore written exactly once -- in _build_row -- instead of being
        # spread over a set of parallel lists here, an append site, and a block
        # of positional slicing in save().
        self._cols: dict[str, list] = {}
        self._clear_imu_raw()

    def _append(self, row: dict) -> None:
        """Fan one sample into the per-column lists.

        Rows are built by a single expression so they always carry the same
        keys, but this checks rather than trusts: a column that skipped one
        sample would shift every later value against the timeline, and the CSV
        would still look well formed.
        """
        if not self._cols:
            self._cols = {name: [] for name in row}
        elif row.keys() != self._cols.keys():
            drift = sorted(set(row) ^ set(self._cols))
            raise RuntimeError(f"log row changed shape mid-recording: {drift}")
        for name, value in row.items():
            self._cols[name].append(value)

    def start(self):
        self._clear()
        self._t0_wall = time.time()
        self._t0_mono = time.perf_counter()
        self.is_logging = True
        print(f"\n{'='*60}\n  DATA COLLECTION STARTED\n{'='*60}\n")

    @staticmethod
    def _imu_vectors(gyro: dict | None) -> dict:
        """Per-role gyro [rad/s] and accel [m/s^2] columns, NaN where absent.

        The firmware reports dps and g; the dataset is SI throughout. Roles the
        stream did not supply come back NaN rather than zero -- zero is a real
        reading, and a missing sensor must not look like a stationary one.
        """
        nan3 = (np.nan, np.nan, np.nan)
        gyros = list(gyro['gyro']) if gyro else []
        accels = list(gyro.get('accel') or []) if gyro else []

        out: dict[str, float] = {}
        for i, role in enumerate(IMU_VECTOR_ROLES):
            if role == 'base':
                # The base sensor is carried outside the per-joint arrays.
                g = gyro.get('base_gyro') if gyro else None
                a = gyro.get('base_accel') if gyro else None
            else:
                g = gyros[i] if i < len(gyros) else None
                a = accels[i] if i < len(accels) else None
            gx, gy, gz = np.radians(g) if g is not None else nan3
            ax, ay, az = np.multiply(a, G_TO_MS2) if a is not None else nan3
            out[f'imu_gx_{role}'] = float(gx)
            out[f'imu_gy_{role}'] = float(gy)
            out[f'imu_gz_{role}'] = float(gz)
            out[f'imu_ax_{role}'] = float(ax)
            out[f'imu_ay_{role}'] = float(ay)
            out[f'imu_az_{role}'] = float(az)
        return out

    def _build_row(self, t: float, manual: dict, sine: dict, combined: dict,
                   pos_deg, state_ts, state_imu_us,
                   vels, vel_age: float, gyro: dict | None,
                   cmd_age_s: float, cmd_stale: bool, sine_enabled: bool,
                   sine_target: str, sine_seed: int) -> dict:
        """One CSV row. THE schema -- every column this file writes is named here."""
        now = time.perf_counter()
        pos = np.radians(pos_deg)
        fresh_vel = vels is not None and vel_age < 0.05

        row: dict = {'timestamp': t, 'sample_idx': len(self._cols.get('timestamp', ()))}

        for joint, channel in COMMAND_CHANNELS:
            row[f'manual_cmd_{channel}'] = float(manual.get(joint, 0.0))
        for joint, channel in COMMAND_CHANNELS:
            row[f'sine_cmd_{channel}'] = float(sine.get(joint, 0.0))
        for joint, channel in COMMAND_CHANNELS:
            row[f'combined_cmd_{channel}'] = float(combined.get(joint, 0.0))

        for i, joint in enumerate(JOINT_NAMES):
            row[f'joint_pos_{joint}'] = float(pos[i]) if i < len(pos) else np.nan
        for joint in VELOCITY_JOINTS:
            i = JOINT_NAMES.index(joint)
            row[f'joint_vel_{joint}'] = float(np.radians(vels[i])) if fresh_vel else np.nan

        row.update(self._imu_vectors(gyro))

        row['cmd_stale'] = int(bool(cmd_stale))
        row['cmd_age_s'] = float(cmd_age_s) if np.isfinite(cmd_age_s) else np.nan
        row['sine_enabled'] = int(bool(sine_enabled))

        # ── capture clocks ───────────────────────────────────────────────────
        # How stale each reading was when this row was written. The control
        # thread and the IMU stream both publish into latest-value caches that
        # this loop samples at its own rate, so without these a repeated sample
        # and a fresh one are indistinguishable after the fact.
        row['state_age_s'] = np.nan if state_ts is None else max(0.0, now - float(state_ts))
        row['vel_age_s'] = float(vel_age) if np.isfinite(vel_age) else np.nan
        # Pico clocks, joining these rows to the companion imu_raw_*.csv. Two of
        # them because they come from two reads: the pose was computed from one
        # IMU frame and the gyro/accel columns above are another, and the loop
        # can straddle a stream update between the two. -1 means the source has
        # not reported yet; int64 has no NaN and 0 is a real Pico timestamp.
        row['state_imu_ts_us'] = -1 if state_imu_us is None else int(state_imu_us)
        gyro_ts = gyro.get('device_timestamp_us') if gyro else None
        row['imu_device_ts_us'] = -1 if gyro_ts is None else int(gyro_ts)

        # The sine target is D-pad switchable mid-recording, so it is per-sample
        # rather than per-file -- rows must be groupable by which channels were
        # actually driven. Amplitude is not a column: it is drawn per joint, and
        # the seed reconstructs it along with every other waveform parameter.
        row['sine_target'] = str(sine_target)
        row['sine_seed'] = int(sine_seed)
        return row

    def log_sample(self, manual: dict, sine: dict, combined: dict,
                   joint_state, hardware, cmd_age_s: float, cmd_stale: bool,
                   sine_enabled: bool,
                   sine_target: str = "", sine_seed: int = -1,
                   controller=None):
        """Record one sample.

        Args:
            joint_state: The ``(angles_deg, state_ts, imu_us)`` triple the caller
                already read this tick. Passed in rather than re-read so the
                logged pose is the same one the rest of the loop saw, and so its
                clocks describe *that* pose -- they come out of one lock
                acquisition inside the controller for exactly that reason.
            hardware: Source of the best-effort gyro/accel read.
            controller: Only for the velocity read, which carries its own age.
        """
        if not self.is_logging:
            return

        t = time.perf_counter() - self._t0_mono
        angles_deg, state_ts, state_imu_us = joint_state
        vels, vel_age = (controller.get_joint_velocities_with_age()
                         if controller is not None else (None, float('inf')))
        self._append(self._build_row(
            t, manual, sine, combined,
            angles_deg, state_ts, state_imu_us,
            vels, vel_age, hardware.try_read_imu_gyro(),
            cmd_age_s, cmd_stale, sine_enabled, sine_target, sine_seed,
        ))

    def n_samples(self) -> int:
        return len(self._cols.get('timestamp', ()))

    def elapsed_min(self) -> float:
        return (time.time() - self._t0_wall) / 60.0 if self._t0_wall else 0.0

    def save(self) -> Path | None:
        """Write the hydraulic strip. Column order is _build_row's insertion order."""
        if not self._cols:
            print("No data to save.")
            self._save_imu_raw_strip()
            return None

        import pandas as pd

        df = pd.DataFrame(self._cols)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"drive_log_{ts}{self.suffix}.csv"
        df.to_csv(out, index=False)
        print(f"[SAVE] {len(df)} samples ({df['timestamp'].iloc[-1]/60:.2f} min) → {out}")
        self._report_staleness(df)
        self._save_imu_raw_strip()
        return out

    @staticmethod
    def _report_staleness(df) -> None:
        """Say how fresh the readings behind this strip actually were.

        Worth a line at save time rather than a question later: the ages are in
        the file either way, but nobody goes looking at them unless something
        already looks wrong, and by then the run is over.
        """
        for column, label in (('state_age_s', 'pose'), ('vel_age_s', 'velocity')):
            ages = df[column].to_numpy(dtype=float)
            finite = ages[np.isfinite(ages)]
            if finite.size == 0:
                print(f"[AGE ] {label}: never reported")
                continue
            missing = ages.size - finite.size
            note = f", {missing} rows without a reading" if missing else ""
            print(f"[AGE ] {label}: median {np.median(finite)*1e3:.1f} ms, "
                  f"max {finite.max()*1e3:.1f} ms{note}")

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
        out = self.output_dir / f"imu_raw_{ts}{self.suffix}.csv"
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
        if not self._cols and not self._imu_ts:
            print("No data to save.")
            return None
        direct.clear()
        direct.send_pending()
        time.sleep(0.3)
        out = self.save()
        # Buffers are already on disk -- drop them so a later Ctrl+C exit
        # (which re-saves whatever is still buffered) doesn't write the same
        # recording out again under a new timestamp.
        self._clear()
        return out


# ── stub controller ───────────────────────────────────────────────────────────

class PWMOnlyController:
    """Stand-in when IMUs are disabled — no IK, joint state returns zeros."""

    def __init__(self, hardware):
        self.hardware = hardware

    def start(self):                      pass
    def stop(self):                       pass
    def suspend_ik_output(self):          pass
    def resume_ik_output(self):           pass
    def get_joint_angles(self):           return np.zeros(4, dtype=np.float32), None, None
    def get_joint_velocities_with_age(self): return None, float('inf')
    def emergency_stop(self, reset_pump=True): self.hardware.reset(reset_pump=reset_pump)


# ── input sources ─────────────────────────────────────────────────────────────

# Button bits. 0-3 are the mask the UDP client already sends; 4-7 are the D-pad
# and are only ever set by the local pad, so a UDP run just reads them as 0.
BTN_A, BTN_B, BTN_X, BTN_Y = 0, 1, 2, 3
BTN_DPAD_UP, BTN_DPAD_DOWN, BTN_DPAD_LEFT, BTN_DPAD_RIGHT = 4, 5, 6, 7

# Shared by teleop and by record_episodes, deliberately: the recorder is the
# training-data twin of this script, so a stick position has to mean the same
# valve command in both. A recorder with its own deadzone would train the policy
# on a machine that behaves differently from the one being driven.
# 25% matches modules.gamepad.XboxController's own default (this constant used to
# override it down to 15).
GAMEPAD_DEADZONE_PCT = 25.0
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

    Tracks are on the triggers, which only read 0..1. Direction is set by
    holding the bumper on the same side: released drives that track forward,
    held reverses it. This is a local-pad-only convention — the remote client
    already sends signed paddle values.
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
        right_sign = -1.0 if s.get('RightBumper') else 1.0
        left_sign  = -1.0 if s.get('LeftBumper') else 1.0
        axes = {
            'right_rl': -float(s['RightJoystickX']),   # bucket
            'right_ud':  float(s['RightJoystickY']),   # boom
            'left_rl':  -float(s['LeftJoystickX']),    # slew
            'left_ud':  -float(s['LeftJoystickY']),    # arm
            'right_paddle': right_sign * float(s['RightTrigger']),
            'left_paddle':  left_sign  * float(s['LeftTrigger']),
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
    p.add_argument("--suffix", default="", metavar="LABEL",
                   help="Append a label to every strip this run writes, e.g. "
                        "--suffix slew gives drive_log_<ts>_slew.csv")
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
    sine_gen = SineExcitationGenerator(enable_slew=args.enable_slew)

    # Raw IMU strips are only meaningful when IMUs are actually streaming.
    imu_stream = hardware.imu_stream_info() if imu_on else {}
    imu_capture_roles = imu_stream.get('roles_by_index', []) if imu_on else []
    logger = DataLogger(out_dir,
                        imu_roles=imu_capture_roles,
                        stream_info_fn=hardware.imu_stream_info if imu_on else None,
                        suffix=args.suffix)

    if args.suffix:
        # Echo the sanitized form, not what was typed: a label that got
        # rewritten (or dropped entirely) should be visible now rather than
        # discovered when the file turns up under an unexpected name.
        if logger.suffix:
            print(f"[LOG] Strips this run: drive_log_<ts>{logger.suffix}.csv")
        else:
            print(f"[LOG] --suffix {args.suffix!r} had no usable characters — "
                  "writing unlabelled strips.")

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
          + ("  Dpad U/D=sine-target  Bumper=reverse-track\n"
             if source.name == "local" else "\n"))

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
            # Redundant against the mode list, which has no slew target without
            # the flag -- kept because this is the line that actually reaches a
            # valve, and a wiring mistake upstream should not swing the cabin.
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
            # One read per tick, shared by the log and the prints below. The
            # controller hands back the pose's own sensor clocks alongside it,
            # from a single lock acquisition -- reading them separately could
            # straddle a control-thread update and describe a different pose.
            joint_state = controller.get_joint_angles()
            joint_angles = joint_state[0]

            if is_logging:
                cmd_age_s = np.nan
                cmd_stale = True
                if last_cmd_mono is not None:
                    cmd_age_s = max(0.0, time.monotonic() - last_cmd_mono)
                    cmd_stale = cmd_age_s > COMMAND_STALE_TIMEOUT_S
                logger.log_sample(manual, sine, combined, joint_state, hardware,
                                  cmd_age_s, cmd_stale, sine_gen.enabled,
                                  sine_gen.target_name, sine_gen.seed,
                                  controller=controller)
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
