#!/usr/bin/env python3
#only rpi support atm!
"""
Data logger for hydraulic actuator model training.

Records valve commands, joint positions/velocities at 100Hz.
Overlays a modulated sine excitation signal on top of manual joystick
commands (Egli & Hutter approach) for richer training data. Sine is
always active during logging; Button B toggles it mid-session.

Uses the kaivuriprokkis ExcavatorController in direct mode (bypass IK/PID).

Usage:
    python data_collection/drive_logger.py
    python data_collection/drive_logger.py --perf
    python data_collection/drive_logger.py --auto-pump
    python data_collection/drive_logger.py --enable_slew

Button Controls (remote gamepad):
    Button A (bit 0): Start / Stop data logging (saves on stop)
    Button B (bit 1): Toggle sine excitation on / off
    Button X (bit 2): Toggle hydraulic pump
    Button Y (bit 3): Cycle sine amplitude
"""

import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path so module imports resolve
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.udp_socket import UDPSocket
from modules.hardware_interface import HardwareInterface, HardwareFaultError
from modules.direct_controller import DirectController
from modules.excavator_controller import ExcavatorController
from modules.perf_tracker import LoopPerfTracker
from modules.rt_utils import apply_rt_to_thread, SCHED_FIFO

# ============================================================
# SETTINGS
# ============================================================
SAMPLING_FREQUENCY = 100          # Hz (main loop and logging rate)
AUTO_SAVE_INTERVAL_MINUTES = 10   # Auto-save every N minutes (0 = disabled)
TARGET_DURATION_MINUTES = 60      # Target session length
STATUS_PRINT_INTERVAL = 5.0       # seconds
COMMAND_STALE_TIMEOUT_S = 0.5
OUTPUT_DIR = Path(__file__).parent / "hydraulic_data"

# Joint names matching profiles/rpi/servo_config.yaml
JOINT_NAMES = ['rotate', 'lift_boom', 'tilt_boom', 'scoop']

# Sine amplitude presets (cycled by button 3)
AMPLITUDE_PRESETS = [0.1, 0.2, 0.3, 0.4, 0.5]


# ============================================================
# SINE EXCITATION GENERATOR
# ============================================================
class SineExcitationGenerator:
    """Modulated sine excitation signal for hydraulic system identification.

    Based on Egli & Hutter (IROS 2020 / RA-L 2022):
        s(t) = A * sin(2*pi * 0.02 * speed * t + phi1)
                 * sin(2*pi * (t + 0.99 * sin(2*pi * 0.1 * t) + phi2))

    Different phase offsets per joint ensure uncorrelated excitation.
    Frequencies stay below ~2 Hz (above the M545 velocity control cutoff).
    """

    def __init__(self):
        self.enabled = True
        self.amplitude_scale = 0.3
        self.speed_scale = 1.0
        self._amplitude_idx = 2  # index into AMPLITUDE_PRESETS (0.3)
        self.start_time = None   # lazily set on first get_signal() call

        # Per-joint phase offsets (phi1, phi2) — different phases per joint
        self.joint_phases = {
            'rotate':    (0.0,   0.0),
            'lift_boom': (1.571, 0.785),   # pi/2, pi/4
            'tilt_boom': (3.142, 1.571),   # pi,   pi/2
            'scoop':     (4.712, 2.356),   # 3pi/2, 3pi/4
        }

    def toggle(self):
        """Toggle excitation on/off. Resets phase clock on re-enable."""
        self.enabled = not self.enabled
        if self.enabled:
            self.start_time = time.perf_counter()

    def cycle_amplitude(self):
        """Cycle through preset amplitude levels."""
        self._amplitude_idx = (self._amplitude_idx + 1) % len(AMPLITUDE_PRESETS)
        self.amplitude_scale = AMPLITUDE_PRESETS[self._amplitude_idx]

    def get_signal(self, joint_name, t):
        """Compute excitation signal for a joint at time t.

        Returns 0.0 when disabled. start_time is set lazily on first call.
        """
        if not self.enabled:
            return 0.0
        if self.start_time is None:
            self.start_time = t
        elapsed = t - self.start_time
        phi1, phi2 = self.joint_phases[joint_name]
        s = self.speed_scale
        envelope = np.sin(2.0 * np.pi * 0.02 * s * elapsed + phi1)
        carrier = np.sin(2.0 * np.pi * (elapsed + 0.99 * np.sin(2.0 * np.pi * 0.1 * elapsed) + phi2))
        return self.amplitude_scale * envelope * carrier

    def get_all_signals(self, t):
        """Compute excitation signals for all joints. Returns dict."""
        return {name: self.get_signal(name, t) for name in JOINT_NAMES}


# ============================================================
# DATA LOGGER
# ============================================================
class DataLogger:
    """Collects hydraulic actuator data for blackbox model training.

    Logs at 100Hz:
    - Manual valve commands (from joystick)
    - Sine excitation commands (from generator)
    - Combined commands (what actually goes to the valve)
    - Joint positions and velocities (from ExcavatorController)
    - Quality flags (command staleness, sine state)

    Saves as CSV for post-processing with sanitize_drive_logs.py.
    """

    def __init__(self):
        self.is_logging = False
        self.segment_id = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._reset_buffers()

    def _reset_buffers(self):
        """Clear all data buffers."""
        self.start_time = None
        self.start_time_mono = None

        # Core data (all lists, same length N)
        self.timestamps = []
        self.sample_indices = []
        self.segment_ids = []

        # Commands: manual, sine, combined — each (N, 4) for [rotate, lift, tilt, scoop]
        self.manual_cmds = []
        self.sine_cmds = []
        self.combined_cmds = []

        # Joint positions (N, 4): [slew, boom, arm, bucket] in degrees
        self.joint_positions = []
        # Joint velocities from IMU gyro Y-axis projection (deg/s)
        self.joint_vel_boom = []
        self.joint_vel_arm = []
        self.joint_vel_bucket = []
        # Raw IMU gyro vectors [gx, gy, gz] in deg/s per joint
        self.imu_gyro_boom = []
        self.imu_gyro_arm = []
        self.imu_gyro_bucket = []

        # Quality flags
        self.cmd_stale_flags = []
        self.cmd_age_seconds = []
        self.sine_enabled_flags = []

    def start(self):
        """Start a new logging session."""
        self.segment_id = 0
        self._reset_buffers()
        self.start_time = time.time()
        self.start_time_mono = time.perf_counter()
        self.is_logging = True
        print(f"\n{'='*60}")
        print(f"  DATA COLLECTION STARTED")
        print(f"  Target duration: {TARGET_DURATION_MINUTES} minutes")
        print(f"  Logging rate: {SAMPLING_FREQUENCY} Hz")
        print(f"{'='*60}\n")

    def start_new_segment(self):
        """Start a new segment after auto-save (keeps session_id)."""
        self.segment_id += 1
        self._reset_buffers()
        self.start_time = time.time()
        self.start_time_mono = time.perf_counter()
        self.is_logging = True
        print(f"[RESUME] Started new logging segment #{self.segment_id}")

    def log_sample(self, manual, sine, combined, controller, hardware,
                   cmd_age_s, cmd_stale, sine_enabled):
        """Log one data sample at 100Hz.

        Args:
            manual: dict {rotate, lift_boom, tilt_boom, scoop} raw joystick values
            sine: dict {rotate, lift_boom, tilt_boom, scoop} sine component
            combined: dict {rotate, lift_boom, tilt_boom, scoop} clamped sum
            controller: ExcavatorController instance
            hardware: HardwareInterface instance
            cmd_age_s: age of last UDP command in seconds
            cmd_stale: bool, True if command is stale
            sine_enabled: bool, True if sine excitation is active
        """
        if not self.is_logging:
            return

        current_time = time.perf_counter() - self.start_time_mono

        # --- Joint state ---
        joint_angles = controller.get_joint_angles()  # [slew, boom, arm, bucket] degrees

        # --- Joint velocities from controller (bias-corrected, proper axis projection) ---
        joint_vels, vel_age = controller.get_joint_velocities_with_age()
        if joint_vels is not None and vel_age < 0.05:
            vel_boom   = joint_vels[1]
            vel_arm    = joint_vels[2]
            vel_bucket = joint_vels[3]
        else:
            vel_boom = vel_arm = vel_bucket = np.nan

        # --- Raw IMU gyro vectors (retained for offline analysis) ---
        gyro_payload = hardware.try_read_imu_gyro()
        if gyro_payload is not None:
            gyros = gyro_payload['gyro']  # [boom, arm, bucket] per _imu_joint_indices order
            boom_g = list(gyros[0]) if len(gyros) > 0 else [np.nan, np.nan, np.nan]
            arm_g  = list(gyros[1]) if len(gyros) > 1 else [np.nan, np.nan, np.nan]
            buck_g = list(gyros[2]) if len(gyros) > 2 else [np.nan, np.nan, np.nan]
        else:
            boom_g = [np.nan, np.nan, np.nan]
            arm_g  = [np.nan, np.nan, np.nan]
            buck_g = [np.nan, np.nan, np.nan]

        # --- Commands in fixed order [rotate, lift, tilt, scoop] ---
        manual_row = [manual.get(n, 0.0) for n in JOINT_NAMES]
        sine_row = [sine.get(n, 0.0) for n in JOINT_NAMES]
        combined_row = [combined.get(n, 0.0) for n in JOINT_NAMES]

        # --- Store sample ---
        sample_idx = len(self.timestamps)
        self.timestamps.append(current_time)
        self.sample_indices.append(sample_idx)
        self.segment_ids.append(self.segment_id)
        self.manual_cmds.append(manual_row)
        self.sine_cmds.append(sine_row)
        self.combined_cmds.append(combined_row)
        self.joint_positions.append(list(joint_angles))
        self.joint_vel_boom.append(float(vel_boom))
        self.joint_vel_arm.append(float(vel_arm))
        self.joint_vel_bucket.append(float(vel_bucket))
        self.imu_gyro_boom.append(boom_g)
        self.imu_gyro_arm.append(arm_g)
        self.imu_gyro_bucket.append(buck_g)
        self.cmd_stale_flags.append(int(bool(cmd_stale)))
        self.cmd_age_seconds.append(float(cmd_age_s) if np.isfinite(cmd_age_s) else np.nan)
        self.sine_enabled_flags.append(int(bool(sine_enabled)))

    def get_elapsed_time(self):
        """Elapsed time in minutes since logging start."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 60.0

    def get_sample_count(self):
        """Number of samples collected in current segment."""
        return len(self.timestamps)

    def save(self):
        """Save collected data to CSV."""
        if not self.timestamps:
            print("No data to save!")
            return None

        import pandas as pd

        print(f"\n{'='*60}")
        print(f"  SAVING DATA...")
        print(f"{'='*60}")

        save_start = time.perf_counter()

        manual = np.array(self.manual_cmds)         # (N, 4)
        sine = np.array(self.sine_cmds)              # (N, 4)
        combined = np.array(self.combined_cmds)      # (N, 4)
        joint_pos = np.array(self.joint_positions)   # (N, 4)
        imu_boom = np.array(self.imu_gyro_boom)      # (N, 3)
        imu_arm  = np.array(self.imu_gyro_arm)       # (N, 3)
        imu_buck = np.array(self.imu_gyro_bucket)    # (N, 3)

        data = {
            'timestamp': np.array(self.timestamps),
            'sample_idx': np.array(self.sample_indices),
            'segment_id': np.array(self.segment_ids),
            # Manual commands
            'manual_cmd_rotate': manual[:, 0],
            'manual_cmd_lift': manual[:, 1],
            'manual_cmd_tilt': manual[:, 2],
            'manual_cmd_scoop': manual[:, 3],
            # Sine commands
            'sine_cmd_rotate': sine[:, 0],
            'sine_cmd_lift': sine[:, 1],
            'sine_cmd_tilt': sine[:, 2],
            'sine_cmd_scoop': sine[:, 3],
            # Combined (what actually went to the valve)
            'combined_cmd_rotate': combined[:, 0],
            'combined_cmd_lift': combined[:, 1],
            'combined_cmd_tilt': combined[:, 2],
            'combined_cmd_scoop': combined[:, 3],
            # Joint positions (degrees)
            'joint_pos_slew': joint_pos[:, 0],
            'joint_pos_boom': joint_pos[:, 1],
            'joint_pos_arm': joint_pos[:, 2],
            'joint_pos_bucket': joint_pos[:, 3],
            # Joint velocities — controller-fused, bias-corrected (deg/s)
            'joint_vel_boom': np.array(self.joint_vel_boom),
            'joint_vel_arm': np.array(self.joint_vel_arm),
            'joint_vel_bucket': np.array(self.joint_vel_bucket),
            # Raw IMU gyro vectors (deg/s)
            'imu_gx_boom': imu_boom[:, 0], 'imu_gy_boom': imu_boom[:, 1], 'imu_gz_boom': imu_boom[:, 2],
            'imu_gx_arm':  imu_arm[:, 0],  'imu_gy_arm':  imu_arm[:, 1],  'imu_gz_arm':  imu_arm[:, 2],
            'imu_gx_bucket': imu_buck[:, 0], 'imu_gy_bucket': imu_buck[:, 1], 'imu_gz_bucket': imu_buck[:, 2],
            # Quality flags
            'cmd_stale': np.array(self.cmd_stale_flags),
            'cmd_age_s': np.array(self.cmd_age_seconds),
            'sine_enabled': np.array(self.sine_enabled_flags),
        }

        df = pd.DataFrame(data)

        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUT_DIR / f"drive_log_{timestamp_str}_seg{self.segment_id:03d}.csv"
        df.to_csv(filename, index=False)

        save_time = time.perf_counter() - save_start
        print(f"\nSaved {len(df)} samples in {save_time:.1f}s")
        self._print_statistics(df)
        print(f"\n[OK] Data saved to: {filename}")
        print(f"{'='*60}\n")
        return filename

    def save_with_pause(self, direct):
        """Save data while machine is stopped (safe for Pi).

        Zeroes valve outputs via the :class:`DirectController` (the controller's
        sensor loop keeps running), saves synchronously, then starts a new
        segment.
        """
        if not self.timestamps:
            print("No data to save!")
            return None

        was_logging = self.is_logging
        self.is_logging = False

        # Zero all outputs (controller bg thread keeps reading sensors)
        print("\n[PAUSE] Stopping machine for save...")
        direct.clear()
        direct.send_pending()
        time.sleep(0.3)

        filename = self.save()

        if was_logging:
            print("[RESUME] Restarting data logging into a new file...")
            self.start_new_segment()
            print("[RESUME] Ready to continue driving!\n")

        return filename

    def _print_statistics(self, df):
        """Print summary statistics after save."""
        duration_sec = df['timestamp'].iloc[-1]
        duration_min = duration_sec / 60.0
        n = len(df)

        print(f"\n  === DATA STATISTICS ===")
        print(f"  Duration: {duration_min:.2f} min ({duration_sec:.1f}s)")
        print(f"  Samples: {n}")
        print(f"  Actual rate: {n / duration_sec:.1f} Hz")
        print(f"  Stale commands: {100.0 * df['cmd_stale'].mean():.1f}%")
        sine_pct = 100.0 * df['sine_enabled'].mean()
        print(f"  Sine enabled: {sine_pct:.1f}% of samples")

        # Combined command ranges
        print(f"\n  Combined Commands:")
        for label in ['lift', 'tilt', 'scoop', 'rotate']:
            col = f'combined_cmd_{label}'
            d = df[col].values
            active = (np.abs(d) > 0.05).sum() / n * 100
            print(f"    {label:>6s}: [{d.min():+.2f}, {d.max():+.2f}]  active {active:.0f}%")

        # Joint position ranges
        print(f"\n  Joint Positions (degrees):")
        for label in ['slew', 'boom', 'arm', 'bucket']:
            col = f'joint_pos_{label}'
            d = df[col].values
            print(f"    {label:>6s}: [{d.min():+.1f}, {d.max():+.1f}]")

        print(f"\n  Joint Velocities (deg/s):")
        for label in ['boom', 'arm', 'bucket']:
            col = f'joint_vel_{label}'
            d = df[col].dropna().values
            if len(d) > 0:
                print(f"    {label:>6s}: max |v| = {np.abs(d).max():.1f}")



# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Excavator data logger with sine excitation")
    parser.add_argument("--perf", action="store_true", help="Show performance metrics")
    parser.add_argument(
        "--auto-pump",
        action="store_true",
        dest="auto_pump",
        help="Use valve-activity-based automatic pump speed",
    )
    parser.add_argument(
        "--enable_slew",
        action="store_true",
        help="Allow slew/rotate commands. By default slew is held at zero.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- UDP setup ----
    server = UDPSocket(local_id=2)
    server.setup("192.168.0.132", 8080, inputs='<8bH', outputs='', is_server=True)

    # ---- Hardware ----
    print("Initializing hardware...")
    hardware = HardwareInterface(
        config_file="configuration_files/profiles/rpi/servo_config.yaml",
        control_config_file="configuration_files/profiles/rpi/control_config.yaml",
        pump_auto_mode=args.auto_pump,
        toggle_channels=True,
        stale_timeout_s=0.5,
        enable_pwm=True,
        enable_imu=True,
        enable_adc=False,
        cleanup_disable_osc=False,
    )

    print("Waiting for hardware to be ready...")
    try:
        while not hardware.is_hardware_ready():
            time.sleep(0.1)
        print("Hardware ready!")
    except HardwareFaultError as e:
        print(f"\n*** HARDWARE FAULT: {e.subsystem} ***")
        print(f"Reason: {e.reason}")
        hardware.shutdown()
        raise SystemExit(1)

    # ---- Controller (direct mode) ----
    print("Starting controller...")
    controller = ExcavatorController(
        hardware, config=None,
        enable_perf_tracking=args.perf,
        control_config_file="configuration_files/profiles/rpi/control_config.yaml",
    )
    controller.start()
    time.sleep(2.0)  # warmup (numba JIT)
    direct = DirectController(hardware)
    controller.suspend_ik_output()
    controller.set_velocity_mode('gyro_only')
    print("Controller in DIRECT mode (IK/PID bypassed)")
    if args.enable_slew:
        print("Slew axis enabled")
    else:
        print("Slew axis disabled (use --enable_slew to allow rotate commands)")
    print(f"Pump mode: {'AUTO' if args.auto_pump else 'STATIC'}")
    print("Sine excitation ON (Button B toggles, Button Y changes amplitude)")

    # ---- Perf tracking ----
    main_loop_perf = LoopPerfTracker(enabled=args.perf)
    if args.perf:
        try:
            hardware.set_perf_enabled(True)
            hardware.reset_perf_stats()
        except Exception:
            pass

    # ---- Helpers ----
    sine_gen = SineExcitationGenerator()
    logger = DataLogger()
    hardware.set_pump_speed_us(None)

    # ---- UDP handshake ----
    print("Waiting for remote controller...")
    if not server.handshake(timeout=30.0):
        print("UDP handshake failed!")
        direct.clear()
        controller.resume_ik_output()
        controller.stop()
        hardware.shutdown()
        raise SystemExit(1)
    server.start_receiving()
    print("Connected! Starting main loop...\n")

    # ---- Main loop state ----
    loop_period = 1.0 / SAMPLING_FREQUENCY  # 0.01s for 100Hz
    next_run_time = time.perf_counter()

    # Joystick state (persists between UDP packets)
    right_rl = 0.0   # scoop
    right_ud = 0.0   # lift
    left_rl = 0.0    # rotate
    left_ud = 0.0    # tilt
    right_paddle = 0.0
    left_paddle = 0.0
    last_command_mono = None

    # Button edge detection (bitmask)
    mask_prev = 0

    last_status_time = time.time()
    last_auto_save_time = time.time()

    apply_rt_to_thread(priority=75, policy=SCHED_FIFO, lock_memory=False)

    try:
        while True:
            main_loop_perf.tick_start()
            current_time = time.time()

            # --- 1. Receive joystick ---
            raw = server.get_latest() or []

            if raw:
                float_axes = UDPSocket.ints_to_floats(raw[:8])
                mask = raw[8]

                right_rl     = float_axes[0]   # scoop
                right_ud     = float_axes[1]   # lift
                left_rl      = float_axes[3]   # rotate (slew)
                left_ud      = float_axes[4]   # tilt
                right_paddle = float_axes[6]   # right track
                left_paddle  = float_axes[7]   # left track
                last_command_mono = time.monotonic()

                # Button A (bit 0): Start/Stop logging
                if (mask & 1) and not (mask_prev & 1):
                    if not logger.is_logging:
                        print("\n[Button A] Starting data collection...")
                        logger.start()
                    else:
                        print("\n[Button A] Stopping data collection and saving...")
                        logger.save_with_pause(direct)
                        logger.is_logging = False

                # Button B (bit 1): Toggle sine excitation
                if (mask & 2) and not (mask_prev & 2):
                    sine_gen.toggle()
                    state = "ON" if sine_gen.enabled else "OFF"
                    print(f"\n[Button B] Sine excitation {state} (amp={sine_gen.amplitude_scale:.2f})")

                # Button X (bit 2): Toggle pump
                if (mask & 4) and not (mask_prev & 4):
                    if hardware.pwm_controller is not None:
                        new_state = not hardware.pwm_controller.pump_enabled
                        hardware.set_pump_enabled(new_state)
                        print(f"\n[Button X] Pump {'ON' if new_state else 'OFF'}")

                # Button Y (bit 3): Cycle sine amplitude
                if (mask & 8) and not (mask_prev & 8):
                    sine_gen.cycle_amplitude()
                    print(f"\n[Button Y] Sine amplitude -> {sine_gen.amplitude_scale:.2f}")

                mask_prev = mask

            # --- 2. Build manual commands ---
            manual_cmds = {
                'rotate': left_rl if args.enable_slew else 0.0,
                'lift_boom': right_ud,
                'tilt_boom': left_ud,
                'scoop': right_rl,
            }

            # --- 3. Compute sine overlay ---
            t = time.perf_counter()
            sine_cmds = sine_gen.get_all_signals(t) if logger.is_logging else {n: 0.0 for n in JOINT_NAMES}
            if not args.enable_slew:
                sine_cmds['rotate'] = 0.0

            # --- 4. Combine and clamp ---
            combined_cmds = {}
            for name in JOINT_NAMES:
                combined_cmds[name] = float(np.clip(
                    manual_cmds[name] + sine_cmds[name], -1.0, 1.0
                ))

            # Add tracks (no sine overlay, just pass through)
            combined_cmds['trackR'] = right_paddle
            combined_cmds['trackL'] = left_paddle

            # --- 5. Send to direct controller ---
            direct.give_commands(combined_cmds)
            direct.send_pending()

            # --- 6. Log ---
            if logger.is_logging:
                cmd_age_s = np.nan
                cmd_stale = True
                if last_command_mono is not None:
                    cmd_age_s = max(0.0, time.monotonic() - last_command_mono)
                    cmd_stale = bool(cmd_age_s > COMMAND_STALE_TIMEOUT_S)

                logger.log_sample(
                    manual_cmds, sine_cmds, combined_cmds,
                    controller, hardware,
                    cmd_age_s, cmd_stale, sine_gen.enabled,
                )

            # --- 7. Auto-save ---
            if logger.is_logging and AUTO_SAVE_INTERVAL_MINUTES > 0:
                if (current_time - last_auto_save_time) >= (AUTO_SAVE_INTERVAL_MINUTES * 60):
                    last_auto_save_time = current_time
                    print(f"\n[AUTO-SAVE] Periodic save ({AUTO_SAVE_INTERVAL_MINUTES} min)")
                    logger.save_with_pause(direct)

            # --- 8. Status print ---
            if current_time - last_status_time >= STATUS_PRINT_INTERVAL:
                last_status_time = current_time
                elapsed_min = logger.get_elapsed_time()
                samples = logger.get_sample_count()
                log_state = "ON" if logger.is_logging else "OFF"

                if logger.is_logging and AUTO_SAVE_INTERVAL_MINUTES > 0:
                    remaining = max(0.0, (AUTO_SAVE_INTERVAL_MINUTES * 60) - (current_time - last_auto_save_time))
                    save_str = f"{remaining:.0f}s"
                else:
                    save_str = "n/a"

                sine_state = f"ON amp={sine_gen.amplitude_scale:.2f}" if sine_gen.enabled else "OFF"
                pump_enabled = bool(hardware.pwm_controller and hardware.pwm_controller.pump_enabled)

                print(f"[STATUS] Logging={log_state} | {elapsed_min:.1f}min | {samples} samples | Save in {save_str}")
                print(f"[SINE]   {sine_state}")
                print(f"[PUMP]   {'ON' if pump_enabled else 'OFF'}")

                # Joint angles
                angles = controller.get_joint_angles()
                print(f"[JOINTS] Slew={angles[0]:+.1f} Boom={angles[1]:+.1f} "
                      f"Arm={angles[2]:+.1f} Bucket={angles[3]:+.1f} deg")

                # Perf metrics
                if args.perf:
                    try:
                        perf = hardware.get_perf_stats()
                        loop_stats = main_loop_perf.get_stats()
                        if perf:
                            imu = perf.get('imu', {})
                            print(f"[PERF]  IMU={imu.get('hz', 0):.0f}Hz | "
                                  f"Loop={loop_stats.get('hz', 0):.0f}Hz "
                                  f"proc={loop_stats.get('proc_avg_ms', 0):.1f}ms")
                    except Exception:
                        pass

            # --- 9. Timing ---
            main_loop_perf.tick_end(target_period_s=loop_period)
            next_run_time += loop_period
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    finally:
        print("\nShutting down...")
        if logger.get_sample_count() > 0:
            print("Final save...")
            logger.is_logging = False
            direct.clear()
            direct.send_pending()
            time.sleep(0.2)
            logger.save()
        controller.resume_ik_output()
        controller.stop()
        hardware.reset(reset_pump=True)
        try:
            hardware.shutdown()
        except Exception:
            pass
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
