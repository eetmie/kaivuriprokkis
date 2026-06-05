#!/usr/bin/env python3
"""Excavator driving and hydraulic data collection.

Merged from simple_drive.py (bench driving) and data_collection/drive_logger.py
(sine-excitation recording). Uses the modern board-profile bringup system.

Usage:
    python simple_drive.py
    python simple_drive.py --record
    python simple_drive.py --robot jetson --record --enable-slew
    python simple_drive.py --comp vel-pid
    python simple_drive.py --auto-pump

Button Controls:
    Button A (bit 0): Start / Stop data logging (saves segment on stop)
    Button B (bit 1): Toggle sine excitation on / off
    Button X (bit 2): Toggle hydraulic pump
    Button Y (bit 3): Cycle sine amplitude

Compensation modes (--comp flag, not button-toggled):
    none         — raw joystick commands, no correction
    raw          — per-joint valve scaling from linkage-rate table
    universal    — pump-gain modulation from universal shape table
    vel-pid      — closed-loop velocity PI (requires IMU)
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
from tools.linkage_rate_compensation import (
    DEFAULT_TABLE,
    DEFAULT_UNIVERSAL_TABLE,
    LinkageRateCompensator,
    UniversalShapeCompensator,
)


# ── constants ────────────────────────────────────────────────────────────────

SAMPLING_FREQUENCY         = 100    # Hz
COMMAND_STALE_TIMEOUT_S    = 0.5
STATUS_PRINT_INTERVAL_S    = 5.0
PRINT_DECIMATION           = 10     # IMU print decimation (→ ~10 Hz)

LOG_OUTPUT_DIR = Path(__file__).parent / "data_collection" / "hydraulic_data"

JOINT_NAMES       = ['slew', 'boom', 'arm', 'bucket']
AMPLITUDE_PRESETS = [0.1, 0.2, 0.3, 0.4, 0.5]
CONTROL_JOINT_NAMES = ['slew', 'lift', 'arm', 'bucket']
IMU_ROLE_ORDER      = ['base', 'boom', 'arm', 'bucket']


# ── velocity PID ─────────────────────────────────────────────────────────────

_VEL_CTRL_JOINTS = {'boom': 1, 'arm': 2, 'bucket': 3}


class JointVelocityController:
    """Joystick → desired deg/s → PI → valve command for the three arm joints."""

    def __init__(self, kp: float, ki: float, ki_max: float,
                 deadband_degps: float, max_degps: float):
        self.kp       = kp
        self.ki       = ki
        self.ki_max   = ki_max
        self.deadband = deadband_degps
        self.max_degps = max_degps
        self._integral: dict[str, float] = {n: 0.0 for n in _VEL_CTRL_JOINTS}

    def apply(self, commands: dict, joint_vels, dt: float) -> dict:
        out = dict(commands)
        for name, vel_idx in _VEL_CTRL_JOINTS.items():
            if name not in out or vel_idx >= len(joint_vels):
                continue
            desired = float(out[name]) * self.max_degps
            if abs(desired) < self.deadband:
                self._integral[name] = 0.0
                out[name] = 0.0
                continue
            err = desired - float(joint_vels[vel_idx])
            self._integral[name] = float(
                np.clip(self._integral[name] + err * dt, -self.ki_max, self.ki_max)
            )
            out[name] = float(np.clip(self.kp * err + self.ki * self._integral[name], -1.0, 1.0))
        return out

    def reset(self):
        for k in self._integral:
            self._integral[k] = 0.0


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
    """Modulated sine excitation per Egli & Hutter (IROS 2020 / RA-L 2022).

    s(t) = A·sin(2π·0.02·speed·t + φ₁) · sin(2π·(t + 0.99·sin(2π·0.1·t) + φ₂))

    Different phase offsets per joint give uncorrelated excitation across axes.
    Frequencies stay below ~2 Hz (above the M545 velocity control cutoff).
    """

    _PHASES = {
        'slew':   (0.0,   0.0),
        'boom':   (1.571, 0.785),
        'arm':    (3.142, 1.571),
        'bucket': (4.712, 2.356),
    }

    def __init__(self):
        self.enabled       = True
        self.speed_scale   = 1.0
        self._amp_idx      = 2
        self.amplitude_scale = AMPLITUDE_PRESETS[self._amp_idx]
        self.start_time    = None

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.start_time = time.perf_counter()

    def cycle_amplitude(self):
        self._amp_idx = (self._amp_idx + 1) % len(AMPLITUDE_PRESETS)
        self.amplitude_scale = AMPLITUDE_PRESETS[self._amp_idx]

    def get_signal(self, joint: str, t: float) -> float:
        if not self.enabled:
            return 0.0
        if self.start_time is None:
            self.start_time = t
        e  = t - self.start_time
        s  = self.speed_scale
        φ1, φ2 = self._PHASES[joint]
        env = np.sin(2.0 * np.pi * 0.02 * s * e + φ1)
        car = np.sin(2.0 * np.pi * (e + 0.99 * np.sin(2.0 * np.pi * 0.1 * e) + φ2))
        return self.amplitude_scale * env * car

    def get_all(self, t: float) -> dict:
        return {n: self.get_signal(n, t) for n in JOINT_NAMES}


# ── data logger ───────────────────────────────────────────────────────────────

class DataLogger:
    """100 Hz hydraulic actuator data recorder for blackbox model training.

    CSV schema matches data_collection/benchmark_actuator_models.py and the
    IsaacLab training pipeline (sanitize_drive_logs → train_blackbox).
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
                   sine_enabled: bool):
        if not self.is_logging:
            return

        t = time.perf_counter() - self._t0_mono

        pos = controller.get_joint_angles()
        vels, vel_age = controller.get_joint_velocities_with_age()
        if vels is not None and vel_age < 0.05:
            vb, va, vbkt = float(vels[1]), float(vels[2]), float(vels[3])
        else:
            vb = va = vbkt = np.nan

        gyro = hardware.try_read_imu_gyro()
        if gyro is not None:
            g  = gyro['gyro']
            gb = list(g[0]) if len(g) > 0 else [np.nan]*3
            ga = list(g[1]) if len(g) > 1 else [np.nan]*3
            gk = list(g[2]) if len(g) > 2 else [np.nan]*3
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
            'manual_cmd_slew': man[:,0], 'manual_cmd_boom': man[:,1],
            'manual_cmd_arm':  man[:,2], 'manual_cmd_bucket': man[:,3],
            'sine_cmd_slew':   sin[:,0], 'sine_cmd_boom':   sin[:,1],
            'sine_cmd_arm':    sin[:,2], 'sine_cmd_bucket': sin[:,3],
            'combined_cmd_slew': com[:,0], 'combined_cmd_boom': com[:,1],
            'combined_cmd_arm':  com[:,2], 'combined_cmd_bucket': com[:,3],
            'joint_pos_slew':   pos[:,0], 'joint_pos_boom': pos[:,1],
            'joint_pos_arm':    pos[:,2], 'joint_pos_bucket': pos[:,3],
            'joint_vel_boom': self._vb, 'joint_vel_arm': self._va, 'joint_vel_bucket': self._vbkt,
            'imu_gx_boom': gb[:,0], 'imu_gy_boom': gb[:,1], 'imu_gz_boom': gb[:,2],
            'imu_gx_arm':  ga[:,0], 'imu_gy_arm':  ga[:,1], 'imu_gz_arm':  ga[:,2],
            'imu_gx_bucket': gk[:,0], 'imu_gy_bucket': gk[:,1], 'imu_gz_bucket': gk[:,2],
            'cmd_stale': self._stale, 'cmd_age_s': self._age, 'sine_enabled': self._sine_flag,
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


# ── args / profile ────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Excavator driving + hydraulic data collection")
    p.add_argument("--robot", choices=[*sorted(ROBOT_PROFILES), "auto"], default="auto")
    p.add_argument("--ip", default="0.0.0.0:8080", metavar="HOST[:PORT]")
    p.add_argument("--config-file",         default=None)
    p.add_argument("--control-config-file", default=None)
    p.add_argument("--pwm-i2c-bus",  type=int,              default=None)
    p.add_argument("--pwm-i2c-addr", type=lambda v: int(v, 0), default=None)
    p.add_argument("--disable-imu",  action="store_true")
    p.add_argument("--disable",      action="store_true", help="Disable toggleable PWM channels")
    p.add_argument("--auto-pump",    dest="auto_pump", action="store_true")
    # data collection
    p.add_argument("--record",       action="store_true", help="Enable data recording")
    p.add_argument("--out",          default=str(LOG_OUTPUT_DIR), help="Output dir for drive logs")
    p.add_argument("--enable-slew",  action="store_true", help="Allow slew in commands and sine")
    p.add_argument("--auto-save-min", type=float, default=10.0, help="Auto-save interval in min (0=off)")
    # compensation (static, not button-toggled)
    p.add_argument("--comp", choices=["none", "raw", "universal", "vel-pid"], default="none",
                   help="Compensation mode (default: none — raw commands)")
    p.add_argument("--linkage-rate-table",           default=str(DEFAULT_TABLE))
    p.add_argument("--linkage-rate-universal-table", default=str(DEFAULT_UNIVERSAL_TABLE))
    p.add_argument("--linkage-rate-min-factor", type=float, default=0.35)
    p.add_argument("--linkage-rate-max-factor", type=float, default=2.25)
    p.add_argument("--vel-kp",       type=float, default=0.04)
    p.add_argument("--vel-ki",       type=float, default=0.008)
    p.add_argument("--vel-ki-max",   type=float, default=0.4)
    p.add_argument("--vel-deadband", type=float, default=2.0,  help="deg/s")
    p.add_argument("--vel-max-degps",type=float, default=30.0, help="deg/s at full stick")
    return p.parse_args()


def _resolve_profile(args) -> dict:
    profile = _resolve_board_profile(args.robot)
    if args.config_file:
        profile['servo_config_file'] = args.config_file
        profile['config_file'] = args.config_file
    if args.control_config_file:
        profile['control_config_file'] = args.control_config_file
    if args.pwm_i2c_bus  is not None: profile['pwm_i2c_bus']  = args.pwm_i2c_bus
    if args.pwm_i2c_addr is not None: profile['pwm_i2c_addr'] = args.pwm_i2c_addr
    if args.disable_imu:              profile['enable_imu']   = False
    return profile


resolve_robot_profile = _resolve_profile


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = _parse_args()
    profile = _resolve_profile(args)
    imu_on  = bool(profile['enable_imu'])

    out_dir = Path(args.out)
    if args.record:
        out_dir.mkdir(parents=True, exist_ok=True)

    ip_str = args.ip
    _host, _port = (ip_str.rsplit(":", 1)[0], int(ip_str.rsplit(":", 1)[1])) \
        if ":" in ip_str else (ip_str, 8080)

    # ── hardware ──────────────────────────────────────────────────────────────
    from modules.hardware_interface import HardwareFaultError, HardwareInterface

    print("Initializing hardware...")
    hardware = HardwareInterface(
        config_file=profile['servo_config_file'],
        control_config_file=profile['control_config_file'],
        pump_auto_mode=args.auto_pump,
        toggle_channels=not args.disable,
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

    # ── compensation setup ────────────────────────────────────────────────────
    linkage_comp   = None
    universal_comp = None
    vel_ctrl       = None

    if args.comp in ("raw", "universal"):
        try:
            linkage_comp = LinkageRateCompensator(
                args.linkage_rate_table,
                min_factor=args.linkage_rate_min_factor,
                max_factor=args.linkage_rate_max_factor,
            )
        except Exception as e:
            print(f"[WARN] Linkage-rate table unavailable: {e}")

    if args.comp == "universal":
        try:
            universal_comp = UniversalShapeCompensator(
                args.linkage_rate_universal_table,
                min_factor=args.linkage_rate_min_factor,
                max_factor=args.linkage_rate_max_factor,
            )
        except Exception as e:
            print(f"[WARN] Universal shape table unavailable: {e}")

    if args.comp == "vel-pid":
        if not imu_on:
            print("[WARN] --comp vel-pid requires IMU; falling back to none")
            args.comp = "none"
        else:
            vel_ctrl = JointVelocityController(
                kp=args.vel_kp, ki=args.vel_ki, ki_max=args.vel_ki_max,
                deadband_degps=args.vel_deadband, max_degps=args.vel_max_degps,
            )

    pwm = hardware.pwm_controller
    pump_gain_available = pwm is not None and getattr(pwm, 'pump_config', None) and pwm.pump_auto_mode
    pump_gain_base_us   = pwm.get_pump_activity_gain_us() if pump_gain_available else 0.0

    print(f"Comp: {args.comp} | pump: {'auto' if args.auto_pump else 'static'}"
          f" | slew: {'on' if args.enable_slew else 'off'}"
          f" | record: {'on' if args.record else 'off'}")

    # ── RT scheduling (Linux only) ─────────────────────────────────────────────
    try:
        from modules.rt_utils import apply_rt_to_thread, SCHED_FIFO
        apply_rt_to_thread(priority=75, policy=SCHED_FIFO, lock_memory=False)
    except Exception:
        pass

    # ── helpers ───────────────────────────────────────────────────────────────
    sine_gen = SineExcitationGenerator()
    logger   = DataLogger(out_dir) if args.record else None

    # ── UDP handshake ─────────────────────────────────────────────────────────
    print("Waiting for remote controller...")
    server = UDPSocket(local_id=2)
    server.setup(_host, _port, inputs='<8bH', outputs='', is_server=True)
    if not server.handshake(timeout=30.0):
        print("UDP handshake failed.")
        direct.clear(); controller.resume_ik_output(); controller.stop()
        hardware.shutdown(); raise SystemExit(1)
    server.start_receiving()
    print("Connected. A=log  B=sine  X=pump  Y=sine-amp\n")

    # ── loop state ────────────────────────────────────────────────────────────
    loop_period     = 1.0 / SAMPLING_FREQUENCY
    next_run_time   = time.perf_counter()
    prev_loop_time  = time.perf_counter()

    right_rl = right_ud = left_rl = left_ud = right_paddle = left_paddle = 0.0
    last_cmd_mono = None
    mask_prev     = 0

    last_status_time    = time.time()
    last_auto_save_time = time.time()
    print_imu           = False
    iter_count          = 0

    try:
        while True:
            now      = time.time()
            loop_now = time.perf_counter()
            actual_dt = loop_now - prev_loop_time
            prev_loop_time = loop_now

            # ── 1. receive ────────────────────────────────────────────────────
            raw = server.get_latest() or []
            if raw:
                fl   = UDPSocket.ints_to_floats(raw[:8])
                mask = raw[8]

                right_rl     = fl[0]
                right_ud     = fl[1]
                left_rl      = fl[3]
                left_ud      = fl[4]
                right_paddle = fl[6]
                left_paddle  = fl[7]
                last_cmd_mono = time.monotonic()

                def btn(b):  return bool(mask & (1 << b))
                def prev(b): return bool(mask_prev & (1 << b))

                # A: toggle logging
                if btn(0) and not prev(0):
                    if logger is None:
                        print("\n[Button A] --record not set; logging disabled")
                    elif not logger.is_logging:
                        logger.start()
                    else:
                        logger.save_with_pause(direct)
                        logger.is_logging = False

                # B: toggle sine
                if btn(1) and not prev(1):
                    sine_gen.toggle()
                    state = "ON" if sine_gen.enabled else "OFF"
                    print(f"\n[Button B] Sine {state} (amp={sine_gen.amplitude_scale:.2f})")

                # X: pump toggle
                if btn(2) and not prev(2):
                    if pwm is not None:
                        new_state = not pwm.pump_enabled
                        hardware.set_pump_enabled(new_state)
                        print(f"\n[Button X] Pump {'ON' if new_state else 'OFF'}")

                # Y: cycle sine amplitude
                if btn(3) and not prev(3):
                    sine_gen.cycle_amplitude()
                    print(f"\n[Button Y] Sine amplitude → {sine_gen.amplitude_scale:.2f}")

                mask_prev = mask

            # ── 2. build commands ─────────────────────────────────────────────
            is_logging = logger is not None and logger.is_logging

            manual = {
                'slew':   left_rl if args.enable_slew else 0.0,
                'boom':   right_ud,
                'arm':    left_ud,
                'bucket': right_rl,
            }

            t = time.perf_counter()
            sine = sine_gen.get_all(t) if is_logging else {n: 0.0 for n in JOINT_NAMES}
            if not args.enable_slew:
                sine['slew'] = 0.0

            combined = {n: float(np.clip(manual[n] + sine[n], -1.0, 1.0)) for n in JOINT_NAMES}
            combined['trackR'] = right_paddle
            combined['trackL'] = left_paddle

            # ── 3. compensation ───────────────────────────────────────────────
            joint_angles = controller.get_joint_angles()

            if args.comp == "raw" and linkage_comp is not None:
                combined = linkage_comp.apply(combined, joint_angles)
            elif args.comp == "vel-pid" and vel_ctrl is not None:
                vels, vel_age = controller.get_joint_velocities_with_age()
                if vel_age < 0.05:
                    combined = vel_ctrl.apply(combined, vels, actual_dt)
                else:
                    vel_ctrl.reset()

            # ── 4. send ───────────────────────────────────────────────────────
            direct.give_commands(combined)
            direct.send_pending()

            # ── 5. pump gain (universal shape) ────────────────────────────────
            if pump_gain_available:
                factor = (universal_comp.pump_correction_factor(combined, joint_angles)
                          if args.comp == "universal" and universal_comp is not None else 1.0)
                pwm.set_pump_activity_gain_us(pump_gain_base_us * factor)

            # ── 6. log ────────────────────────────────────────────────────────
            if is_logging:
                cmd_age_s = np.nan
                cmd_stale = True
                if last_cmd_mono is not None:
                    cmd_age_s = max(0.0, time.monotonic() - last_cmd_mono)
                    cmd_stale = cmd_age_s > COMMAND_STALE_TIMEOUT_S
                logger.log_sample(manual, sine, combined, controller, hardware,
                                  cmd_age_s, cmd_stale, sine_gen.enabled)

            # ── 7. auto-save ──────────────────────────────────────────────────
            if (is_logging and args.auto_save_min > 0
                    and (now - last_auto_save_time) >= args.auto_save_min * 60):
                last_auto_save_time = now
                logger.save_with_pause(direct)

            # ── 8. IMU print (decimated) ──────────────────────────────────────
            iter_count += 1
            if print_imu and imu_on and (iter_count % PRINT_DECIMATION == 0):
                print(_format_imu_line(
                    hardware.read_imu_debug_quaternions(),
                    joint_angles, ctrl_joint_names, imu_role_order,
                ), end="\r", flush=True)

            # ── 9. status ─────────────────────────────────────────────────────
            if now - last_status_time >= STATUS_PRINT_INTERVAL_S:
                last_status_time = now
                pump_on  = bool(pwm and pwm.pump_enabled)
                sine_str = f"ON amp={sine_gen.amplitude_scale:.2f}" if sine_gen.enabled else "OFF"
                print(
                    f"[STATUS] pump={'ON' if pump_on else 'OFF'} | sine={sine_str} | "
                    + (f"log=ON {logger.elapsed_min():.1f}min {logger.n_samples()} samples"
                       if is_logging else "log=OFF")
                )
                print(f"[JOINTS] slew={joint_angles[0]:+.1f} boom={joint_angles[1]:+.1f} "
                      f"arm={joint_angles[2]:+.1f} bucket={joint_angles[3]:+.1f} deg")

            # ── 10. timing ────────────────────────────────────────────────────
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
        if logger is not None and logger.n_samples() > 0:
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
