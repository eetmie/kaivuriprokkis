#!/usr/bin/env python3
"""Sine excitation valve response test.

Hardware test that commands ONE valve at a time with a sine sweep and
observes the IMU joint angle response. This answers:

1. Does the sine signal actually produce valve movement?
2. What amplitude is needed per joint to get visible motion?
3. Does the ramp limiter clip the sine at higher amplitudes?
4. What is the joint angle excursion at each amplitude preset?

For each joint, the test:
- Sends a simple low-frequency sine (0.2Hz) at each amplitude preset
- Records the IMU-measured joint angle over ~10 seconds
- Reports whether movement was detected and the angle range

Also includes an offline analysis section that reads servo_config_200.yaml
to analyze deadband/ramp interaction without hardware.

Usage:
    python data_collection/test_sine_valve_range.py                    # offline analysis only
    python data_collection/test_sine_valve_range.py --hardware         # run on real hardware
    python data_collection/test_sine_valve_range.py --hardware --joint lift_boom
    python data_collection/test_sine_valve_range.py --plot             # generate matplotlib plots
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drive_logger import SineExcitationGenerator, JOINT_NAMES, AMPLITUDE_PRESETS

RESULTS_DIR = Path(__file__).parent / "test_results"
CONFIG_PATH = _ROOT / "configuration_files" / "servo_config_200.yaml"

# Map drive_logger joint names to IMU joint indices
# controller.get_joint_angles() returns [slew, boom, arm, bucket]
JOINT_TO_IMU_IDX = {
    'rotate': 0,     # slew
    'lift_boom': 1,  # boom
    'tilt_boom': 2,  # arm
    'scoop': 3,      # bucket
}


# ============================================================
# OFFLINE ANALYSIS (no hardware needed)
# ============================================================

def load_valve_config():
    """Load servo config and extract per-joint valve characteristics."""
    with open(CONFIG_PATH, 'r') as f:
        raw = yaml.safe_load(f)

    configs = {}
    for name, cfg in raw.get('CHANNEL_CONFIGS', {}).items():
        if name == 'pump':
            continue
        center = cfg.get('center', (cfg['pulse_min'] + cfg['pulse_max']) / 2)
        db_pos = cfg.get('deadband_us_pos', 0.0)
        db_neg = cfg.get('deadband_us_neg', 0.0)

        configs[name] = {
            'center': center,
            'pulse_min': cfg['pulse_min'],
            'pulse_max': cfg['pulse_max'],
            'deadband_us_pos': db_pos,
            'deadband_us_neg': db_neg,
            'usable_range_pos_us': cfg['pulse_max'] - (center + db_pos),
            'usable_range_neg_us': (center - db_neg) - cfg['pulse_min'],
            'total_range_us': cfg['pulse_max'] - cfg['pulse_min'],
            'direction': cfg.get('direction', 1),
            'gamma': cfg.get('gamma', 1.0),
            'ramp_limit_us_s': cfg.get('ramp_limit', 0.0),
            'dither_enable': cfg.get('dither_enable', False),
            'dither_amp_us': cfg.get('dither_amp_us', 0.0),
        }
    return configs


def compute_base_pulse(cfg, value):
    """Replicate PWM controller's deadband compression logic."""
    s = float(value) * float(cfg['direction'])
    center = float(cfg['center'])
    if s == 0.0:
        return center
    elif s > 0.0:
        base = center + float(cfg['deadband_us_pos'])
        working = float(cfg['pulse_max']) - base
        return base + abs(float(value)) * working
    else:
        base = center - float(cfg['deadband_us_neg'])
        working = base - float(cfg['pulse_min'])
        return base - abs(float(value)) * working


def run_offline_analysis(valve_configs):
    """Analyze sine vs valve characteristics without hardware."""
    lines = []
    lines.append("=" * 70)
    lines.append("  OFFLINE: VALVE DEADBAND vs SINE EXCITATION ANALYSIS")
    lines.append("=" * 70)

    # 1. Valve physical characteristics
    lines.append("")
    lines.append("1. VALVE PHYSICAL CHARACTERISTICS")
    lines.append("-" * 50)
    lines.append(f"  {'Joint':<12} {'Center':>8} {'DB+':>6} {'DB-':>6} "
                 f"{'Usable+':>9} {'Usable-':>9} {'Gamma':>6} {'Ramp':>8}")

    for name in JOINT_NAMES:
        cfg = valve_configs[name]
        lines.append(
            f"  {name:<12} {cfg['center']:8.1f} {cfg['deadband_us_pos']:6.1f} "
            f"{cfg['deadband_us_neg']:6.1f} {cfg['usable_range_pos_us']:9.1f} "
            f"{cfg['usable_range_neg_us']:9.1f} "
            f"{cfg['gamma']:6.1f} {cfg['ramp_limit_us_s']:8.1f}")

    lines.append("")
    lines.append("  NOTE: Deadband compression means ANY non-zero command jumps over")
    lines.append("  the dead zone instantly. The sine signal is never 'eaten' by deadband.")
    lines.append("")

    # 2. Ramp limit vs sine slew rate
    lines.append("2. RAMP LIMIT vs SINE SLEW RATE (amp=0.3, speed=1.0)")
    lines.append("-" * 50)

    sine_gen = SineExcitationGenerator()
    sine_gen.amplitude_scale = 0.3
    sine_gen.enabled = True
    sine_gen.start_time = 0.0

    dt = 0.01  # 100Hz
    t_values = np.arange(0, 10, dt)

    for name in JOINT_NAMES:
        cfg = valve_configs[name]
        ramp_limit = cfg['ramp_limit_us_s']

        signals = np.array([sine_gen.get_signal(name, t) for t in t_values])
        pulses = np.array([compute_base_pulse(cfg, s) for s in signals])

        slew_rates = np.abs(np.diff(pulses)) / dt
        max_slew = slew_rates.max() if len(slew_rates) > 0 else 0
        pct_exceeds = (np.sum(slew_rates > ramp_limit) / max(1, len(slew_rates))) * 100

        status = "OK" if pct_exceeds < 1.0 else f"CLIPPED {pct_exceeds:.0f}% of time"
        lines.append(
            f"  {name:<12} ramp={ramp_limit:.0f} us/s | "
            f"sine_slew: max={max_slew:.0f} us/s | {status}")

    lines.append("")
    lines.append("  CLIPPED means the ramp limiter smooths/lags the sine output.")
    lines.append("  The blackbox model learns this lag from the data — not a problem.")
    lines.append("")

    # 3. Combined signal clipping
    lines.append("3. COMBINED SIGNAL CLIPPING (lift_boom, amp=0.3)")
    lines.append("-" * 50)
    lines.append(f"  {'Manual':>7} | {'Clip%':>6} | {'Effective sine RMS':>22} | {'Range':>16}")

    t_values = np.arange(0, 30, 0.01)
    for manual in [0.0, 0.3, 0.5, 0.7, 0.9]:
        signals = np.array([sine_gen.get_signal('lift_boom', t) for t in t_values])
        combined = np.clip(manual + signals, -1.0, 1.0)
        unclamped = manual + signals
        clip_pct = (np.sum(np.abs(unclamped) > 1.0) / len(unclamped)) * 100
        eff_rms = np.sqrt(np.mean((combined - manual)**2))
        orig_rms = np.sqrt(np.mean(signals**2))
        retention = (eff_rms / orig_rms * 100) if orig_rms > 0 else 0

        lines.append(
            f"  {manual:+7.2f} | {clip_pct:5.1f}% | "
            f"rms={eff_rms:.3f} ({retention:.0f}% retained) | "
            f"[{combined.min():+.3f}, {combined.max():+.3f}]")

    lines.append("")
    lines.append("  Above ~0.7 manual, sine gets partially clipped. This is fine —")
    lines.append("  Egli papers say ~50% of data should be unperturbed anyway.")
    lines.append("")

    return lines


# ============================================================
# HARDWARE TEST: single-valve sine + IMU observation
# ============================================================

def run_hardware_test(joints_to_test, amplitudes_to_test, duration_per_test=10.0):
    """Command one valve at a time with sine, observe IMU response.

    For each (joint, amplitude) pair:
    1. Send simple sine (0.2Hz) to that one valve
    2. Record IMU joint angle at 100Hz
    3. Report angle range and whether movement detected
    """
    from modules.hardware_interface import HardwareInterface, HardwareFaultError
    from modules.excavator_controller import ExcavatorController

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  HARDWARE: SINGLE-VALVE SINE RESPONSE TEST")
    lines.append("=" * 70)
    lines.append(f"  Sine frequency: 0.2 Hz (5s period)")
    lines.append(f"  Duration per test: {duration_per_test}s")
    lines.append(f"  Joints: {', '.join(joints_to_test)}")
    lines.append(f"  Amplitudes: {amplitudes_to_test}")
    lines.append("")

    hardware = None
    controller = None

    try:
        print("[hw] Initializing hardware...")
        hardware = HardwareInterface(
            pump_auto_mode=False,
            toggle_channels=True,
            stale_timeout_s=0.5,
            adc_channels=[
                "LiftBoom retract ps", "LiftBoom extend ps",
                "TiltBoom retract ps", "TiltBoom extend ps",
                "Scoop extend ps", "Scoop retract ps", "Pump ps",
            ],
            adc_sample_hz=20,
            enable_pwm=True,
            enable_imu=True,
            enable_adc=True,
            cleanup_disable_osc=False,
            log_level="WARNING",
        )

        deadline = time.time() + 15.0
        while not hardware.is_hardware_ready():
            if time.time() >= deadline:
                raise TimeoutError("Hardware not ready within 15s")
            time.sleep(0.1)
        print("[hw] Hardware ready")

        print("[hw] Starting controller (direct mode)...")
        controller = ExcavatorController(
            hardware, config=None,
            enable_perf_tracking=False,
            log_level="WARNING",
        )
        controller.start()
        time.sleep(2.0)
        controller.enter_direct_mode()
        print("[hw] Ready\n")

        # Let IMU settle — record baseline angles
        time.sleep(1.0)
        baseline = np.array(controller.get_joint_angles())
        lines.append(f"  Baseline angles: slew={baseline[0]:+.1f} boom={baseline[1]:+.1f} "
                     f"arm={baseline[2]:+.1f} bucket={baseline[3]:+.1f} deg")
        lines.append("")

        # Results table header
        lines.append(f"  {'Joint':<12} {'Amp':>5} | {'Angle min':>10} {'Angle max':>10} "
                     f"{'Range':>7} {'Movement':>10}")
        lines.append("  " + "-" * 62)

        sine_freq = 0.2  # Hz, simple sine

        for joint_name in joints_to_test:
            imu_idx = JOINT_TO_IMU_IDX[joint_name]

            for amp in amplitudes_to_test:
                # Zero all valves first, let arm settle
                controller.give_direct_commands({n: 0.0 for n in JOINT_NAMES})
                time.sleep(1.0)

                # Record angles during sine excitation
                angles_recorded = []
                test_start = time.perf_counter()
                loop_period = 0.01  # 100Hz
                next_run = test_start

                print(f"  Testing {joint_name} amp={amp:.2f} ...", end="", flush=True)

                while (time.perf_counter() - test_start) < duration_per_test:
                    t = time.perf_counter() - test_start
                    sine_value = amp * math.sin(2.0 * math.pi * sine_freq * t)

                    # Command ONLY this joint
                    cmds = {n: 0.0 for n in JOINT_NAMES}
                    cmds[joint_name] = sine_value
                    controller.give_direct_commands(cmds)

                    # Read IMU angle
                    angles = controller.get_joint_angles()
                    angles_recorded.append(angles[imu_idx])

                    next_run += loop_period
                    sleep_time = next_run - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_run = time.perf_counter()

                # Zero the valve
                controller.give_direct_commands({n: 0.0 for n in JOINT_NAMES})

                # Analyze
                angles_arr = np.array(angles_recorded)
                angle_min = angles_arr.min()
                angle_max = angles_arr.max()
                angle_range = angle_max - angle_min

                # Movement threshold: >0.5 degrees is "real" movement
                moved = "YES" if angle_range > 0.5 else "NO"

                lines.append(
                    f"  {joint_name:<12} {amp:5.2f} | "
                    f"{angle_min:+10.2f} {angle_max:+10.2f} "
                    f"{angle_range:7.2f} {moved:>10}")
                print(f" range={angle_range:.2f} deg {'OK' if moved == 'YES' else 'NO MOVEMENT'}")

        lines.append("")
        lines.append("  Movement threshold: >0.5 degrees = detected")
        lines.append("  If NO movement: check pump is on, valve is connected, amplitude too low")
        lines.append("")

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        lines.append(f"\n  ERROR: {e}")

    finally:
        if controller is not None:
            try:
                controller.give_direct_commands({n: 0.0 for n in JOINT_NAMES})
                time.sleep(0.3)
                controller.exit_direct_mode()
                controller.stop()
            except Exception:
                pass
        elif hardware is not None:
            try:
                hardware.shutdown()
            except Exception:
                pass

    return lines


# ============================================================
# PLOTS (offline, no hardware)
# ============================================================

def generate_plots(valve_configs):
    """Generate matplotlib plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sine_gen = SineExcitationGenerator()
    sine_gen.enabled = True
    sine_gen.start_time = 0.0
    sine_gen.amplitude_scale = 0.3

    # Plot 1: Sine signals for all joints over 30s
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t = np.arange(0, 30, 0.01)

    for i, name in enumerate(JOINT_NAMES):
        signals = [sine_gen.get_signal(name, ti) for ti in t]
        axes[i].plot(t, signals, linewidth=0.5)
        axes[i].set_ylabel(name)
        axes[i].set_ylim(-0.5, 0.5)
        axes[i].axhline(y=0, color='gray', linewidth=0.3)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Modulated sine excitation (amp=0.3, speed=1.0x)", fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_dir / "sine_signals_30s.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'sine_signals_30s.png'}")

    # Plot 2: Amplitude comparison
    fig, axes = plt.subplots(len(AMPLITUDE_PRESETS), 1, figsize=(14, 12), sharex=True)
    t_med = np.arange(0, 15, 0.01)

    for i, amp in enumerate(AMPLITUDE_PRESETS):
        sine_gen.amplitude_scale = amp
        sig = np.array([sine_gen.get_signal('lift_boom', ti) for ti in t_med])
        axes[i].plot(t_med, sig, linewidth=0.5)
        axes[i].set_ylabel(f"amp={amp}")
        axes[i].set_ylim(-0.6, 0.6)
        axes[i].axhline(y=0, color='gray', linewidth=0.3)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Sine at each amplitude preset (lift_boom)", fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_dir / "amplitude_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'amplitude_comparison.png'}")

    # Plot 3: Commanded pulse vs ramp-limited pulse (simulated)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t_short = np.arange(0, 10, 0.005)  # 200Hz to simulate ramp
    sine_gen.amplitude_scale = 0.3

    for i, name in enumerate(JOINT_NAMES):
        cfg = valve_configs[name]
        ramp_limit = cfg['ramp_limit_us_s']

        sigs = np.array([sine_gen.get_signal(name, ti) for ti in t_short])
        commanded = np.array([compute_base_pulse(cfg, s) for s in sigs])

        # Simulate ramp limiting
        ramped = np.copy(commanded)
        dt_sim = 0.005
        for j in range(1, len(ramped)):
            max_change = ramp_limit * dt_sim
            diff = commanded[j] - ramped[j-1]
            if abs(diff) > max_change:
                ramped[j] = ramped[j-1] + np.sign(diff) * max_change
            else:
                ramped[j] = commanded[j]

        axes[i].plot(t_short, commanded - cfg['center'], linewidth=0.5, label='commanded', alpha=0.7)
        axes[i].plot(t_short, ramped - cfg['center'], linewidth=0.5, label='after ramp limit')
        axes[i].set_ylabel(f"{name}\n(us from center)")
        axes[i].legend(loc='upper right', fontsize=7)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Commanded vs ramp-limited pulse (amp=0.3)", fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_dir / "ramp_limiting.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_dir / 'ramp_limiting.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sine excitation valve response test. "
                    "Runs offline analysis by default. "
                    "Use --hardware to command real valves and observe IMU.")
    parser.add_argument("--hardware", action="store_true",
                        help="Run real hardware test (commands valves, reads IMU)")
    parser.add_argument("--joint", type=str, default=None,
                        choices=JOINT_NAMES,
                        help="Test only this joint (default: all except rotate)")
    parser.add_argument("--amps", type=str, default=None,
                        help="Comma-separated amplitudes to test (default: all presets)")
    parser.add_argument("--duration-per-test", type=float, default=10.0,
                        help="Seconds per (joint, amplitude) test (default: 10)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate matplotlib plots (offline)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_lines = []

    # Always run offline analysis
    print("Loading valve config...")
    valve_configs = load_valve_config()

    print("Running offline analysis...\n")
    offline_lines = run_offline_analysis(valve_configs)
    all_lines.extend(offline_lines)
    for line in offline_lines:
        print(line)

    # Hardware test if requested
    if args.hardware:
        if args.joint:
            joints = [args.joint]
        else:
            # Default: all arm joints, skip rotate (safety — rotation can tip the machine)
            joints = ['lift_boom', 'tilt_boom', 'scoop']

        if args.amps:
            amps = [float(a) for a in args.amps.split(',')]
        else:
            amps = AMPLITUDE_PRESETS

        hw_lines = run_hardware_test(joints, amps, args.duration_per_test)
        all_lines.extend(hw_lines)
        for line in hw_lines:
            print(line)

    # Save results
    results_path = RESULTS_DIR / f"sine_valve_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    results_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {results_path}")

    if args.plot:
        print("\nGenerating plots...")
        generate_plots(valve_configs)


if __name__ == "__main__":
    main()
