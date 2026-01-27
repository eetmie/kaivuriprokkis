#!/usr/bin/env python3
"""
RT Full Workload Test - Mimics run_hw.py control loop with logging.

Runs the actual control loop workload including:
- Controller pose commands (IMU read, FK, IK, PWM)
- Data logging (similar to HardwareDataLogger)
- Perf metrics collection
- Periodic status output

This simulates real run_hw.py usage without waiting for goals.
"""

import argparse
import time
import sys
import os
import signal
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.hardware_interface import HardwareInterface
from modules.excavator_controller import ExcavatorController
from modules.rt_utils import (
    apply_rt_to_thread, reset_to_normal,
    get_system_info, SCHED_FIFO
)
from modules.perf_tracker import LoopPerfTracker


# Test parameters
DEFAULT_DURATION = 300.0  # 5 minutes
TARGET_HZ = 100.0
STATUS_INTERVAL = 10.0  # Print status every 10 seconds


class RTTestDataLogger:
    """Real data logger matching run_hw.py HardwareDataLogger format."""

    def __init__(self, output_dir: str = "logs_rt_test"):
        """Initialize the data logger.

        Args:
            output_dir: Directory to save log files
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(script_dir, output_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # Find next run number
        existing = [f for f in os.listdir(self.log_dir) if f.startswith("rt_test_")]
        if existing:
            nums = [int(f.split("_")[-1].split(".")[0]) for f in existing if f.split("_")[-1].split(".")[0].isdigit()]
            self.run_num = max(nums) + 1 if nums else 1
        else:
            self.run_num = 1

        self.samples = 0
        self.start_time = time.time()
        self.trajectory_log: List[List[float]] = []
        self.adc_log: List[List[float]] = []

        # Tracking metrics
        self.max_tracking_error = 0.0
        self._prev_pos: Optional[np.ndarray] = None
        self.total_distance = 0.0

    def log_step(
        self,
        planned_pos: np.ndarray,
        actual_pos: np.ndarray,
        planned_quat: np.ndarray,
        actual_quat: np.ndarray,
        joint_angles: np.ndarray,
        progress: float,
        adc_readings: Optional[Dict[str, float]] = None,
    ) -> float:
        """Log a step and return the time taken."""
        log_start = time.perf_counter()

        timestamp = time.time() - self.start_time

        # Trajectory row (same format as run_hw.py)
        row = [
            timestamp,
            float(planned_pos[0]), float(planned_pos[1]), float(planned_pos[2]),
            float(actual_pos[0]), float(actual_pos[1]), float(actual_pos[2]),
            float(planned_quat[0]), float(planned_quat[1]), float(planned_quat[2]), float(planned_quat[3]),
            float(actual_quat[0]), float(actual_quat[1]), float(actual_quat[2]), float(actual_quat[3]),
            float(joint_angles[0]), float(joint_angles[1]), float(joint_angles[2]), float(joint_angles[3]),
            float(progress),
        ]
        self.trajectory_log.append(row)

        # ADC row (if available)
        if adc_readings:
            adc_row = [timestamp]
            for name in sorted(adc_readings.keys()):
                adc_row.append(float(adc_readings[name]))
            self.adc_log.append(adc_row)

        # Update metrics
        error = float(np.linalg.norm(actual_pos - planned_pos))
        if error > self.max_tracking_error:
            self.max_tracking_error = error

        if self._prev_pos is not None:
            self.total_distance += float(np.linalg.norm(actual_pos - self._prev_pos))
        self._prev_pos = actual_pos.copy()

        self.samples += 1
        return time.perf_counter() - log_start

    def save(self, results: Dict[str, Any]) -> str:
        """Save logs to CSV files. Returns the trajectory CSV path."""
        if not self.trajectory_log:
            return ""

        # Trajectory CSV
        traj_columns = [
            "timestamp",
            "x_g", "y_g", "z_g",
            "x_e", "y_e", "z_e",
            "quat_g_w", "quat_g_x", "quat_g_y", "quat_g_z",
            "quat_e_w", "quat_e_x", "quat_e_y", "quat_e_z",
            "joint_1", "joint_2", "joint_3", "joint_4",
            "progress",
        ]
        traj_df = pd.DataFrame(self.trajectory_log, columns=traj_columns)
        traj_path = os.path.join(self.log_dir, f"rt_test_{self.run_num}_trajectory.csv")
        traj_df.to_csv(traj_path, index=False)

        # ADC CSV (if we have data)
        if self.adc_log:
            # Get column names from first row length
            adc_columns = ["timestamp",
                          "LiftBoom_retract_ps", "LiftBoom_extend_ps",
                          "TiltBoom_retract_ps", "TiltBoom_extend_ps",
                          "Scoop_extend_ps", "Scoop_retract_ps",
                          "Pump_ps", "Slew_encoder_rot"]
            # Adjust columns to match data
            if len(self.adc_log[0]) <= len(adc_columns):
                adc_columns = adc_columns[:len(self.adc_log[0])]
            adc_df = pd.DataFrame(self.adc_log, columns=adc_columns)
            adc_path = os.path.join(self.log_dir, f"rt_test_{self.run_num}_adc.csv")
            adc_df.to_csv(adc_path, index=False)

        # Metrics CSV
        metrics = {
            "run_id": self.run_num,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s": results.get('duration_s', 0),
            "samples": self.samples,
            "control_rt_config": results.get('config', 'unknown'),
            "imu_rt_config": results.get('imu_config', 'NORMAL'),
            "adc_rt_config": results.get('adc_config', 'NORMAL'),
            "loop_hz": results.get('loop_hz', 0),
            "loop_jitter_ms": results.get('loop_jitter_ms', 0),
            "loop_max_ms": results.get('loop_max_ms', 0),
            "imu_hz": results.get('imu_hz', 0),
            "imu_jitter_ms": results.get('imu_jitter_ms', 0),
            "adc_hz": results.get('adc_hz', 0),
            "adc_jitter_ms": results.get('adc_jitter_ms', 0),
            "max_tracking_error_m": self.max_tracking_error,
            "total_distance_m": self.total_distance,
        }
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(self.log_dir, "rt_test_metrics.csv")

        if os.path.exists(metrics_path):
            metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_path, index=False)

        return traj_path


def run_full_workload_test(
    duration: float,
    rt_priority: int,
    output_file: str,
    imu_rt_priority: int = 0,
    adc_rt_priority: int = 0,
    lock_memory: bool = False,
    pin_cores: bool = False,
) -> Dict[str, Any]:
    """Run full workload test mimicking run_hw.py.

    Args:
        duration: Test duration in seconds
        rt_priority: RT priority (0 = normal, 1-89 = SCHED_FIFO)
        output_file: Path to write results

    Returns:
        Dictionary with test results
    """
    results = {
        'success': False,
        'error': None,
        'config': f"RT-FIFO-{rt_priority}" if rt_priority > 0 else "NORMAL",
        'imu_config': f"RT-FIFO-{imu_rt_priority}" if imu_rt_priority > 0 else "NORMAL",
        'adc_config': f"RT-FIFO-{adc_rt_priority}" if adc_rt_priority > 0 else "NORMAL",
        'memory_locked': lock_memory,
        'cores_pinned': pin_cores,
        'duration_s': 0.0,
        'samples': 0,
    }

    # All 8 ADC channels
    all_adc_channels = [
        "LiftBoom retract ps",
        "LiftBoom extend ps",
        "TiltBoom retract ps",
        "TiltBoom extend ps",
        "Scoop extend ps",
        "Scoop retract ps",
        "Pump ps",
        "Slew encoder rot",
    ]

    # Initialize hardware
    # Core assignments for pinning (Pi 5 has cores 0-3)
    # Core 0: Linux housekeeping
    # Core 1: Control loop
    # Core 2: IMU thread
    # Core 3: ADC thread
    control_core = 1 if pin_cores else None
    imu_core = 2 if pin_cores else None
    adc_core = 3 if pin_cores else None

    print("Initializing hardware interface...")
    print(f"  ADC: 8 channels at 20Hz")
    print(f"  IMU RT: {imu_rt_priority if imu_rt_priority > 0 else 'NORMAL'}")
    print(f"  ADC RT: {adc_rt_priority if adc_rt_priority > 0 else 'NORMAL'}")
    if pin_cores:
        print(f"  Core pinning: Control=Core1, IMU=Core2, ADC=Core3")
    if lock_memory:
        print(f"  Memory locking: ENABLED")
    try:
        hardware = HardwareInterface(
            perf_enabled=True,
            adc_channels=all_adc_channels,
            adc_sample_hz=20.0,  # 20Hz for 8 channels
            imu_rt_priority=imu_rt_priority,
            adc_rt_priority=adc_rt_priority,
            imu_cpu_core=imu_core,
            adc_cpu_core=adc_core,
        )
    except Exception as e:
        results['error'] = f"Hardware init failed: {e}"
        return results

    print("Initializing excavator controller...")
    try:
        controller = ExcavatorController(
            hardware_interface=hardware,
            enable_perf_tracking=True,
            log_level="WARNING",
        )
        controller.start()
    except Exception as e:
        results['error'] = f"Controller init failed: {e}"
        return results

    print("Waiting for hardware to stabilize...")
    time.sleep(2.0)

    # Apply core pinning for control loop
    if pin_cores and control_core is not None:
        try:
            os.sched_setaffinity(0, {control_core})
            print(f"Control loop: Pinned to Core {control_core}")
        except Exception as e:
            print(f"Warning: Failed to pin control loop to Core {control_core}: {e}")

    # Apply RT priority and memory locking
    if rt_priority > 0 or lock_memory:
        print(f"Applying RT settings...")
        if rt_priority > 0:
            print(f"  RT priority: SCHED_FIFO-{rt_priority}")
        if lock_memory:
            print(f"  Memory locking: enabled")
        success = apply_rt_to_thread(
            priority=rt_priority if rt_priority > 0 else 0,
            policy=SCHED_FIFO if rt_priority > 0 else 0,
            lock_memory=lock_memory,
            quiet=False
        )
        if not success and rt_priority > 0:
            results['error'] = "Failed to apply RT settings"
            controller.stop()
            return results

    try:
        # Setup
        loop_tracker = LoopPerfTracker(enabled=True)
        data_logger = RTTestDataLogger()
        target_period = 1.0 / TARGET_HZ
        print(f"  Data logging to: {data_logger.log_dir}")

        # Fake trajectory (simulate path following)
        # Generate a simple back-and-forth motion
        path_duration = 30.0  # 30 second path cycle

        # Fake target poses
        point_a = np.array([0.35, -0.05, 0.15], dtype=np.float32)
        point_b = np.array([0.25, 0.05, 0.25], dtype=np.float32)

        controller.resume()
        hardware.reset_perf_stats()

        # Warmup
        print("Warming up (10s)...")
        warmup_end = time.perf_counter() + 10.0
        next_run = time.perf_counter()
        while time.perf_counter() < warmup_end:
            controller.give_pose(point_a, 0.0)
            next_run += target_period
            sleep_time = next_run - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run = time.perf_counter()

        # Reset stats after warmup
        loop_tracker.reset()
        hardware.reset_perf_stats()

        print(f"\nStarting {duration:.0f}s test run...")
        print(f"Config: {results['config']}")
        print("-" * 60)

        test_start = time.perf_counter()
        test_end = test_start + duration
        next_run = test_start
        last_status = test_start
        iterations = 0

        # Log timing stats
        log_times: List[float] = []

        while time.perf_counter() < test_end:
            loop_start = time.perf_counter()
            loop_tracker.tick_start()
            elapsed = loop_start - test_start

            # Simulate path progress (back and forth)
            cycle_progress = (elapsed % path_duration) / path_duration
            if int(elapsed / path_duration) % 2 == 0:
                progress = cycle_progress
            else:
                progress = 1.0 - cycle_progress

            # Interpolate target position
            target_pos = point_a + (point_b - point_a) * progress
            target_rot = 0.0

            # Main control work: send pose to controller
            controller.give_pose(target_pos, target_rot)

            # Get current state (like run_hw.py does for logging)
            current_pos, current_rot = controller.get_pose()
            current_pos = np.asarray(current_pos, dtype=np.float32)
            joint_angles = controller.get_joint_angles()
            if joint_angles is None or len(joint_angles) < 4:
                joint_angles = np.zeros(4, dtype=np.float32)
            else:
                joint_angles = np.asarray(joint_angles[:4], dtype=np.float32)

            # Fake quaternions (we only use Y rotation)
            planned_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            actual_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            # Get ADC readings (all 8 channels)
            try:
                adc_readings = hardware.get_latest_adc_readings()
            except Exception:
                adc_readings = None

            # Data logging work
            log_time = data_logger.log_step(
                planned_pos=target_pos,
                actual_pos=current_pos,
                planned_quat=planned_quat,
                actual_quat=actual_quat,
                joint_angles=joint_angles,
                progress=progress,
                adc_readings=adc_readings,
            )
            log_times.append(log_time)

            loop_tracker.tick_end(target_period_s=target_period)
            iterations += 1

            # Periodic status (like run_hw.py debug output)
            if loop_start - last_status >= STATUS_INTERVAL:
                last_status = loop_start
                loop_stats = loop_tracker.get_stats()
                hw_stats = hardware.get_perf_stats()
                ctrl_stats = controller.get_performance_stats()

                imu = hw_stats.get('imu', {})
                adc = hw_stats.get('adc', {})
                pwm = hw_stats.get('pwm', {})

                pct_complete = (elapsed / duration) * 100

                print(f"[{elapsed:6.1f}s / {duration:.0f}s] {pct_complete:5.1f}% | "
                      f"Loop: {loop_stats.get('hz', 0):.0f}Hz j={loop_stats.get('loop_std_ms', 0):.3f}ms "
                      f"max={loop_stats.get('loop_max_ms', 0):.2f}ms | "
                      f"IMU: {imu.get('hz', 0):.0f}Hz j={imu.get('std_interval_ms', 0):.2f}ms | "
                      f"ADC: {adc.get('hz', 0):.0f}Hz | "
                      f"Samples: {data_logger.samples}")

            # Maintain timing
            next_run += target_period
            sleep_time = next_run - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run = time.perf_counter()

        actual_duration = time.perf_counter() - test_start

        # Collect final stats
        loop_stats = loop_tracker.get_stats()
        hw_stats = hardware.get_perf_stats()
        ctrl_stats = controller.get_performance_stats()

        results['success'] = True
        results['duration_s'] = actual_duration
        results['samples'] = iterations
        results['loop_hz'] = loop_stats.get('hz', 0)
        results['loop_jitter_ms'] = loop_stats.get('loop_std_ms', 0)
        results['loop_max_ms'] = loop_stats.get('loop_max_ms', 0)
        results['loop_proc_ms'] = loop_stats.get('proc_avg_ms', 0)
        results['loop_headroom_ms'] = loop_stats.get('headroom_avg_ms', 0)
        results['loop_violations_pct'] = loop_stats.get('violation_pct', 0)

        imu = hw_stats.get('imu', {})
        adc = hw_stats.get('adc', {})
        pwm = hw_stats.get('pwm', {})

        results['imu_hz'] = imu.get('hz', 0)
        results['imu_jitter_ms'] = imu.get('std_interval_ms', 0)
        results['adc_hz'] = adc.get('hz', 0)
        results['adc_jitter_ms'] = adc.get('std_interval_ms', 0)
        results['pwm_hz'] = pwm.get('hz', 0)
        results['pwm_jitter_ms'] = pwm.get('loop_std_ms', 0)

        results['log_avg_ms'] = np.mean(log_times) * 1000 if log_times else 0
        results['log_max_ms'] = np.max(log_times) * 1000 if log_times else 0

        results['ctrl_stats'] = ctrl_stats

        # Save data logs
        print("\nSaving data logs...")
        log_path = data_logger.save(results)
        results['log_path'] = log_path
        print(f"  Trajectory: {log_path}")

    except Exception as e:
        results['error'] = str(e)
        import traceback
        traceback.print_exc()

    finally:
        reset_to_normal(quiet=True)
        print("\nShutting down controller...")
        try:
            controller.stop()
        except Exception:
            pass

    return results


def format_report(results: Dict[str, Any], system_info: Dict[str, Any]) -> str:
    """Format results as text report."""
    lines = []

    lines.append("=" * 70)
    lines.append("  RT FULL WORKLOAD TEST - SIMULATING run_hw.py")
    lines.append("=" * 70)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  SYSTEM INFORMATION")
    lines.append("-" * 70)
    lines.append(f"  Kernel: {system_info.get('kernel_version', 'unknown')[:55]}")
    lines.append(f"  RT Kernel: {'Yes' if system_info.get('is_rt_kernel') else 'No'}")
    lines.append(f"  Running as root: {'Yes' if system_info.get('is_root') else 'No'}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  TEST CONFIGURATION")
    lines.append("-" * 70)
    lines.append(f"  Control RT: {results.get('config', 'unknown')}")
    lines.append(f"  IMU RT:     {results.get('imu_config', 'NORMAL')}")
    lines.append(f"  ADC RT:     {results.get('adc_config', 'NORMAL')}")
    lines.append(f"  Memory locked: {'Yes' if results.get('memory_locked') else 'No'}")
    lines.append(f"  Cores pinned:  {'Yes (Ctrl=1, IMU=2, ADC=3)' if results.get('cores_pinned') else 'No'}")
    lines.append(f"  Target frequency: {TARGET_HZ:.0f} Hz")
    lines.append(f"  Duration: {results.get('duration_s', 0):.1f} seconds")
    lines.append(f"  Total samples: {results.get('samples', 0)}")
    lines.append("")
    lines.append("  Workload includes:")
    lines.append("    - Controller.give_pose() (IMU read, FK, IK, PWM send)")
    lines.append("    - Controller.get_pose() + get_joint_angles()")
    lines.append("    - ADC sampling (8 channels at 20Hz)")
    lines.append("    - Data logging (CSV: trajectory + ADC)")
    lines.append("    - Periodic status output")
    lines.append("")
    if results.get('log_path'):
        lines.append(f"  Data saved to: {results.get('log_path')}")
        lines.append("")

    if not results.get('success'):
        lines.append("-" * 70)
        lines.append("  TEST FAILED")
        lines.append("-" * 70)
        lines.append(f"  Error: {results.get('error', 'Unknown')}")
        return "\n".join(lines)

    lines.append("-" * 70)
    lines.append("  MAIN LOOP PERFORMANCE")
    lines.append("-" * 70)
    lines.append(f"  Frequency:    {results.get('loop_hz', 0):.1f} Hz (target: {TARGET_HZ:.0f} Hz)")
    lines.append(f"  Jitter:       {results.get('loop_jitter_ms', 0):.3f} ms (std dev of loop interval)")
    lines.append(f"  Max spike:    {results.get('loop_max_ms', 0):.3f} ms")
    lines.append(f"  Processing:   {results.get('loop_proc_ms', 0):.3f} ms avg")
    lines.append(f"  Headroom:     {results.get('loop_headroom_ms', 0):.2f} ms avg")
    lines.append(f"  Violations:   {results.get('loop_violations_pct', 0):.2f}%")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  SUBSYSTEM PERFORMANCE")
    lines.append("-" * 70)
    lines.append(f"  IMU:  {results.get('imu_hz', 0):6.1f} Hz, jitter = {results.get('imu_jitter_ms', 0):.3f} ms")
    lines.append(f"  ADC:  {results.get('adc_hz', 0):6.1f} Hz, jitter = {results.get('adc_jitter_ms', 0):.3f} ms")
    lines.append(f"  PWM:  {results.get('pwm_hz', 0):6.1f} Hz, jitter = {results.get('pwm_jitter_ms', 0):.3f} ms")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  DATA LOGGING OVERHEAD")
    lines.append("-" * 70)
    lines.append(f"  Avg log time: {results.get('log_avg_ms', 0):.4f} ms")
    lines.append(f"  Max log time: {results.get('log_max_ms', 0):.4f} ms")
    lines.append("")

    # Analysis
    lines.append("-" * 70)
    lines.append("  ANALYSIS")
    lines.append("-" * 70)

    jitter = results.get('loop_jitter_ms', 0)
    if jitter < 0.3:
        lines.append(f"  Jitter: EXCELLENT ({jitter:.3f}ms < 0.3ms)")
    elif jitter < 0.5:
        lines.append(f"  Jitter: VERY GOOD ({jitter:.3f}ms < 0.5ms)")
    elif jitter < 1.0:
        lines.append(f"  Jitter: GOOD ({jitter:.3f}ms < 1.0ms)")
    else:
        lines.append(f"  Jitter: ACCEPTABLE ({jitter:.3f}ms)")

    headroom = results.get('loop_headroom_ms', 0)
    target_ms = 1000.0 / TARGET_HZ
    headroom_pct = (headroom / target_ms) * 100
    lines.append(f"  Headroom: {headroom_pct:.1f}% of period unused ({headroom:.2f}ms / {target_ms:.1f}ms)")

    proc = results.get('loop_proc_ms', 0)
    log_avg = results.get('log_avg_ms', 0)
    lines.append(f"  Processing breakdown: {proc:.3f}ms total, {log_avg:.4f}ms logging")

    violations = results.get('loop_violations_pct', 0)
    if violations == 0:
        lines.append("  Timing violations: NONE")
    elif violations < 0.1:
        lines.append(f"  Timing violations: NEGLIGIBLE ({violations:.3f}%)")
    else:
        lines.append(f"  Timing violations: {violations:.2f}% - may need investigation")

    lines.append("")
    lines.append("-" * 70)
    lines.append("  CONCLUSION")
    lines.append("-" * 70)

    if jitter < 0.5 and violations < 0.1:
        lines.append("  System is STABLE and READY FOR PRODUCTION.")
        lines.append(f"  Recommended config: {results.get('config')}")
    elif jitter < 1.0:
        lines.append("  System is STABLE with minor jitter.")
        lines.append("  Acceptable for most control applications.")
    else:
        lines.append("  System shows some timing variability.")
        lines.append("  Consider investigating sources of jitter.")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Full workload RT test simulating run_hw.py")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help=f"Test duration in seconds (default: {DEFAULT_DURATION})")
    parser.add_argument("--priority", type=int, default=70,
                        help="RT priority for control loop (default: 70)")
    parser.add_argument("--imu-priority", type=int, default=0,
                        help="RT priority for IMU thread (default: 0 = normal, try 50)")
    parser.add_argument("--adc-priority", type=int, default=0,
                        help="RT priority for ADC thread (default: 0 = normal, try 50)")
    parser.add_argument("--lock-memory", action="store_true",
                        help="Lock all memory to prevent page faults (mlockall)")
    parser.add_argument("--pin-cores", action="store_true",
                        help="Pin threads to CPU cores (Control=1, IMU=2, ADC=3)")
    parser.add_argument("--output", type=str, default="rt_full_workload_results.txt",
                        help="Output file (default: rt_full_workload_results.txt)")

    args = parser.parse_args()

    def sigint_handler(sig, frame):
        print("\n\n[!] Interrupted - cleaning up...")
        reset_to_normal(quiet=True)
        sys.exit(1)
    signal.signal(signal.SIGINT, sigint_handler)

    system_info = get_system_info()

    print("\n" + "=" * 60)
    print("  RT FULL WORKLOAD TEST")
    print("  Simulates run_hw.py control loop with logging")
    print("=" * 60)

    results = run_full_workload_test(
        duration=args.duration,
        rt_priority=args.priority,
        output_file=args.output,
        imu_rt_priority=args.imu_priority,
        adc_rt_priority=args.adc_priority,
        lock_memory=args.lock_memory,
        pin_cores=args.pin_cores,
    )

    report = format_report(results, system_info)

    print("\n")
    print(report)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\nResults saved to: {output_path}")

    return 0 if results.get('success') else 1


if __name__ == "__main__":
    sys.exit(main())
