#!/usr/bin/env python3
"""
Data logger for hydraulic actuator model training.

Records valve commands, joint positions/velocities, and pressures at 100Hz.
Optionally overlays a modulated sine excitation signal on top of manual
joystick commands (Egli & Hutter approach) for richer training data.

Uses the kaivuriprokkis ExcavatorController in direct mode (bypass IK/PID).

Usage:
    python data_collection/drive_logger.py
    python data_collection/drive_logger.py --perf

Button Controls (remote gamepad):
    Button 0: Start / Stop data logging (saves on stop)
    Button 1: Toggle sine excitation on / off
    Button 2: Toggle hydraulic pump
    Button 3: Cycle sine amplitude (0.1 -> 0.2 -> 0.3 -> 0.4 -> 0.5 -> 0.1)
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
from modules.excavator_controller import ExcavatorController
from modules.perf_tracker import LoopPerfTracker

# ============================================================
# SETTINGS
# ============================================================
SAMPLING_FREQUENCY = 100          # Hz (main loop and logging rate)
PRESSURE_SAMPLING_FREQUENCY = 20  # Hz (ADC hardware limit)
AUTO_SAVE_INTERVAL_MINUTES = 10   # Auto-save every N minutes (0 = disabled)
TARGET_DURATION_MINUTES = 60      # Target session length
STATUS_PRINT_INTERVAL = 5.0       # seconds
COMMAND_STALE_TIMEOUT_S = 0.5
PRESSURE_MAX_AGE_S = 0.2
OUTPUT_DIR = Path(__file__).parent / "hydraulic_data"

# Joint names matching servo_config_200.yaml
JOINT_NAMES = ['rotate', 'lift_boom', 'tilt_boom', 'scoop']

# Sine amplitude presets (cycled by button 3)
AMPLITUDE_PRESETS = [0.1, 0.2, 0.3, 0.4, 0.5]

# NOTE: Pump is currently static (toggle on/off only via button 2).
# The Egli papers note that engine RPM variations affect actuator behavior.
# If initial data collection demos work well, a future iteration could add:
#   - Variable pump speed as a logged input channel
#   - Pump speed as a training input (like RPM/oil temp in the papers)
#   - Automatic pump speed modulation during excitation
# For now, keep pump constant during recording sessions for simpler modeling.


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
        self.enabled = False
        self.amplitude_scale = 0.3
        self.speed_scale = 1.0
        self._amplitude_idx = 2  # index into AMPLITUDE_PRESETS (0.3)
        self.start_time = None

        # Per-joint phase offsets (phi1, phi2) — different phases per joint
        self.joint_phases = {
            'rotate':    (0.0,   0.0),
            'lift_boom': (1.571, 0.785),   # pi/2, pi/4
            'tilt_boom': (3.142, 1.571),   # pi,   pi/2
            'scoop':     (4.712, 2.356),   # 3pi/2, 3pi/4
        }

    def toggle(self):
        """Toggle excitation on/off. Resets phase on enable for smooth start."""
        self.enabled = not self.enabled
        if self.enabled:
            self.start_time = time.perf_counter()

    def cycle_amplitude(self):
        """Cycle through preset amplitude levels."""
        self._amplitude_idx = (self._amplitude_idx + 1) % len(AMPLITUDE_PRESETS)
        self.amplitude_scale = AMPLITUDE_PRESETS[self._amplitude_idx]

    def get_signal(self, joint_name, t):
        """Compute excitation signal for a joint at time t.

        Returns 0.0 when disabled. Signal starts from zero phase on enable.
        """
        if not self.enabled or self.start_time is None:
            return 0.0
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
    - Cylinder pressures (from ADC)
    - Quality flags (staleness, sine state)

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

        # Joint state — (N, 4) for [slew, boom, arm, bucket]
        self.joint_positions = []   # degrees
        self.joint_velocities = []  # deg/s

        # Pressures — (N, 3) for [lift, tilt, scoop]
        self.pressures_extend = []
        self.pressures_retract = []
        self.pump_pressures = []
        self.pressure_timestamps = []

        # Quality flags
        self.cmd_stale_flags = []
        self.cmd_age_seconds = []
        self.pressure_stale_flags = []
        self.sine_enabled_flags = []

        # Pressure sample-and-hold state
        self._last_pressure_extend = None
        self._last_pressure_retract = None
        self._last_pump_pressure = np.nan
        self._last_pressure_wall_ts = None
        self._last_pressure_sample_time = None
        self._pressure_sample_interval = 1.0 / PRESSURE_SAMPLING_FREQUENCY

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
        print(f"  Pressure rate: {PRESSURE_SAMPLING_FREQUENCY} Hz")
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

        # --- Joint state from controller ---
        joint_angles = controller.get_joint_angles()  # [slew, boom, arm, bucket] degrees
        joint_vels = controller.get_joint_velocities_degps()  # [slew, boom, arm, bucket] deg/s or None
        if joint_vels is None:
            joint_vels = [np.nan, np.nan, np.nan, np.nan]

        # --- Commands in fixed order [rotate, lift, tilt, scoop] ---
        manual_row = [manual.get(n, 0.0) for n in JOINT_NAMES]
        sine_row = [sine.get(n, 0.0) for n in JOINT_NAMES]
        combined_row = [combined.get(n, 0.0) for n in JOINT_NAMES]

        # --- Pressure data (20Hz sample-and-hold to 100Hz) ---
        should_sample = False
        if self._last_pressure_sample_time is None:
            should_sample = True
        elif (current_time - self._last_pressure_sample_time) >= self._pressure_sample_interval:
            should_sample = True

        if should_sample:
            try:
                adc_snapshot = hardware.get_latest_adc_snapshot()
                readings = adc_snapshot.get("readings", {})
                self._last_pressure_wall_ts = adc_snapshot.get("timestamp", None)

                self._last_pressure_extend = [
                    readings.get("LiftBoom extend ps", np.nan),
                    readings.get("TiltBoom extend ps", np.nan),
                    readings.get("Scoop extend ps", np.nan),
                ]
                self._last_pressure_retract = [
                    readings.get("LiftBoom retract ps", np.nan),
                    readings.get("TiltBoom retract ps", np.nan),
                    readings.get("Scoop retract ps", np.nan),
                ]
                self._last_pump_pressure = readings.get("Pump ps", np.nan)
                self._last_pressure_sample_time = current_time
            except Exception:
                if self._last_pressure_extend is None:
                    self._last_pressure_extend = [np.nan, np.nan, np.nan]
                if self._last_pressure_retract is None:
                    self._last_pressure_retract = [np.nan, np.nan, np.nan]

        # Held pressure values
        p_extend = list(self._last_pressure_extend) if self._last_pressure_extend else [np.nan] * 3
        p_retract = list(self._last_pressure_retract) if self._last_pressure_retract else [np.nan] * 3
        pump_ps = self._last_pump_pressure if self._last_pump_pressure is not None else np.nan

        # Pressure staleness
        if self._last_pressure_wall_ts is not None:
            pressure_ts = max(0.0, self._last_pressure_wall_ts - self.start_time)
            pressure_age = max(0.0, time.time() - self._last_pressure_wall_ts)
        else:
            pressure_ts = np.nan
            pressure_age = np.nan

        pressure_stale = bool(np.isfinite(pressure_age) and pressure_age > PRESSURE_MAX_AGE_S)
        if pressure_stale:
            p_extend = [np.nan, np.nan, np.nan]
            p_retract = [np.nan, np.nan, np.nan]
            pump_ps = np.nan

        # --- Store sample ---
        sample_idx = len(self.timestamps)
        self.timestamps.append(current_time)
        self.sample_indices.append(sample_idx)
        self.segment_ids.append(self.segment_id)
        self.manual_cmds.append(manual_row)
        self.sine_cmds.append(sine_row)
        self.combined_cmds.append(combined_row)
        self.joint_positions.append(list(joint_angles))
        self.joint_velocities.append(list(joint_vels))
        self.pressures_extend.append(p_extend)
        self.pressures_retract.append(p_retract)
        self.pump_pressures.append(pump_ps)
        self.pressure_timestamps.append(pressure_ts)
        self.cmd_stale_flags.append(int(bool(cmd_stale)))
        self.cmd_age_seconds.append(float(cmd_age_s) if np.isfinite(cmd_age_s) else np.nan)
        self.pressure_stale_flags.append(int(bool(pressure_stale)))
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

        manual = np.array(self.manual_cmds)       # (N, 4)
        sine = np.array(self.sine_cmds)            # (N, 4)
        combined = np.array(self.combined_cmds)    # (N, 4)
        joint_pos = np.array(self.joint_positions)  # (N, 4)
        joint_vel = np.array(self.joint_velocities) # (N, 4)
        p_ext = np.array(self.pressures_extend)     # (N, 3)
        p_ret = np.array(self.pressures_retract)    # (N, 3)

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
            # Joint velocities (deg/s)
            'joint_vel_slew': joint_vel[:, 0],
            'joint_vel_boom': joint_vel[:, 1],
            'joint_vel_arm': joint_vel[:, 2],
            'joint_vel_bucket': joint_vel[:, 3],
            # Pressures (V)
            'pext_lift': p_ext[:, 0],
            'pext_tilt': p_ext[:, 1],
            'pext_scoop': p_ext[:, 2],
            'pret_lift': p_ret[:, 0],
            'pret_tilt': p_ret[:, 1],
            'pret_scoop': p_ret[:, 2],
            'pump_ps': np.array(self.pump_pressures),
            'pressure_ts': np.array(self.pressure_timestamps),
            # Quality flags
            'cmd_stale': np.array(self.cmd_stale_flags),
            'cmd_age_s': np.array(self.cmd_age_seconds),
            'pressure_stale': np.array(self.pressure_stale_flags),
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

    def save_with_pause(self, controller):
        """Save data while machine is stopped (safe for Pi).

        Zeroes valve outputs via controller (keeps sensor loop alive),
        saves synchronously, then starts a new segment.
        """
        if not self.timestamps:
            print("No data to save!")
            return None

        was_logging = self.is_logging
        self.is_logging = False

        # Zero all outputs (controller bg thread keeps reading sensors)
        print("\n[PAUSE] Stopping machine for save...")
        controller.give_direct_commands({})
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
        print(f"  Stale pressures: {100.0 * df['pressure_stale'].mean():.1f}%")
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

        # Joint velocity ranges
        print(f"\n  Joint Velocities (deg/s):")
        for label in ['slew', 'boom', 'arm', 'bucket']:
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
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- UDP setup (raw float mode, same as old logger) ----
    server = UDPSocket(local_id=2)
    server.setup("192.168.0.132", 8080, num_inputs=10, num_outputs=0, is_server=True)

    # ---- Hardware ----
    print("Initializing hardware...")
    hardware = HardwareInterface(
        pump_variable=False,
        toggle_channels=True,
        stale_timeout_s=0.5,
        adc_channels=[
            "LiftBoom retract ps",
            "LiftBoom extend ps",
            "TiltBoom retract ps",
            "TiltBoom extend ps",
            "Scoop extend ps",
            "Scoop retract ps",
            "Pump ps",
        ],
        adc_sample_hz=PRESSURE_SAMPLING_FREQUENCY,
        enable_pwm=True,
        enable_imu=True,
        enable_adc=True,
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
    )
    controller.start()
    time.sleep(2.0)  # warmup (numba JIT)
    controller.enter_direct_mode()
    print("Controller in DIRECT mode (IK/PID bypassed)")

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

    # ---- UDP handshake ----
    print("Waiting for remote controller...")
    if not server.handshake(timeout=30.0):
        print("UDP handshake failed!")
        controller.exit_direct_mode()
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

    # Button edge detection
    button_prev = [0.0, 0.0, 0.0, 0.0]
    button_threshold = 0.5

    last_status_time = time.time()
    last_auto_save_time = time.time()

    try:
        while True:
            main_loop_perf.tick_start()
            current_time = time.time()

            # --- 1. Receive joystick ---
            float_data = server.get_latest_floats()

            if float_data:
                right_rl = float_data[9]      # scoop
                right_ud = float_data[8]      # lift
                left_rl = float_data[7]       # rotate
                left_ud = float_data[6]       # tilt
                right_paddle = float_data[5]  # right track
                left_paddle = float_data[4]   # left track
                last_command_mono = time.monotonic()

                buttons = [float_data[0], float_data[1], float_data[2], float_data[3]]

                # Button 0: Start/Stop logging
                if buttons[0] > button_threshold and button_prev[0] <= button_threshold:
                    if not logger.is_logging:
                        print("\n[Button 0] Starting data collection...")
                        logger.start()
                    else:
                        print("\n[Button 0] Stopping data collection and saving...")
                        logger.save_with_pause(controller)
                        logger.is_logging = False

                # Button 1: Toggle sine excitation
                if buttons[1] > button_threshold and button_prev[1] <= button_threshold:
                    sine_gen.toggle()
                    state = "ON" if sine_gen.enabled else "OFF"
                    print(f"\n[Button 1] Sine excitation {state} (amp={sine_gen.amplitude_scale:.2f})")

                # Button 2: Toggle pump
                if buttons[2] > button_threshold and button_prev[2] <= button_threshold:
                    print("\n[Button 2] Toggling pump...")
                    if hardware.pwm_controller:
                        hardware.pwm_controller.set_pump(not hardware.pwm_controller.pump_enabled)

                # Button 3: Cycle sine amplitude
                if buttons[3] > button_threshold and button_prev[3] <= button_threshold:
                    sine_gen.cycle_amplitude()
                    print(f"\n[Button 3] Sine amplitude -> {sine_gen.amplitude_scale:.2f}")

                button_prev = buttons

            # --- 2. Build manual commands ---
            manual_cmds = {
                'rotate': left_rl,
                'lift_boom': right_ud,
                'tilt_boom': left_ud,
                'scoop': right_rl,
            }

            # --- 3. Compute sine overlay ---
            t = time.perf_counter()
            sine_cmds = sine_gen.get_all_signals(t)

            # --- 4. Combine and clamp ---
            combined_cmds = {}
            for name in JOINT_NAMES:
                combined_cmds[name] = float(np.clip(
                    manual_cmds[name] + sine_cmds[name], -1.0, 1.0
                ))

            # Add tracks (no sine overlay, just pass through)
            combined_cmds['trackR'] = right_paddle
            combined_cmds['trackL'] = left_paddle

            # --- 5. Send to controller ---
            controller.give_direct_commands(combined_cmds)

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
                    logger.save_with_pause(controller)

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

                print(f"[STATUS] Logging={log_state} | {elapsed_min:.1f}min | {samples} samples | Save in {save_str}")
                print(f"[SINE]   {sine_state}")

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
                            adc = perf.get('adc', {})
                            print(f"[PERF]  IMU={imu.get('hz', 0):.0f}Hz | "
                                  f"ADC={adc.get('hz', 0):.0f}Hz | "
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
            controller.give_direct_commands({})
            time.sleep(0.2)
            logger.save()
        controller.exit_direct_mode()
        controller.stop()
        hardware.reset(reset_pump=True)
        try:
            hardware.shutdown()
        except Exception:
            pass
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
