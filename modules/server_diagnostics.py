"""
Server-side diagnostics formatting for excv_gui.

ServerDiagnostics receives data and formats debug/perf/jacobian output.
It never calls controller or hardware directly.
"""

import enum
import numpy as np
from typing import Optional


class DiagMode(enum.Enum):
    QUIET = "quiet"
    NORMAL = "normal"
    PERF = "perf"
    JACOBIAN = "jacobian"


def _fmt_vec(v, n=3):
    """Format a numeric vector as a compact string."""
    return ", ".join(f"{float(x):+.3f}" for x in list(v)[:n])


class ServerDiagnostics:
    """Formats and prints server-side debug output.

    All data is passed in — this class never reaches into controller/hardware.
    """

    def __init__(self, mode: DiagMode):
        self.mode = mode

        # Jacobian debug state (accumulated across calls)
        self.prev_joint_angles: Optional[np.ndarray] = None
        self.prev_ee_pos: Optional[np.ndarray] = None

    def print_status(
        self,
        telemetry,
        last_command,
        debug_state: Optional[dict],
        network_times: list,
        packet_rate: float,
        packets_received: int,
        perf_only: bool = False,
    ):
        """Print one status line (called every 1-2 seconds from the main loop).

        Args:
            telemetry: RobotTelemetry from service.get_state()
            last_command: Last decoded ControlCommand (or None)
            debug_state: Dict from service.get_debug_state() (or None)
            network_times: List of network round-trip times in ms
            packet_rate: Packets/sec over the last print interval
            packets_received: Total packets received
            perf_only: If True, suppress non-perf info
        """
        if self.mode == DiagMode.QUIET:
            return

        if last_command is None:
            if self.mode != DiagMode.PERF:
                print("Waiting for position data...")
            return

        actual_pos = np.array([
            telemetry.measured_pose.x,
            telemetry.measured_pose.y,
            telemetry.measured_pose.z,
        ])
        actual_rot = telemetry.measured_pose.rot_y_deg

        if self.mode == DiagMode.PERF:
            self._print_perf(debug_state, network_times, packet_rate, perf_only)
        elif self.mode == DiagMode.JACOBIAN:
            self._print_normal(last_command, actual_pos, actual_rot, packet_rate, packets_received, debug_state)
            self._print_jacobian(telemetry, actual_pos, last_command, debug_state)
        else:
            self._print_normal(last_command, actual_pos, actual_rot, packet_rate, packets_received, debug_state)

    def _print_normal(self, last_command, actual_pos, actual_rot, packet_rate, packets_received, debug_state):
        """Print the normal status line."""
        cond = debug_state.get('condition_number', 0) if debug_state else 0
        cond_str = f" | Cond: {cond:.1f}" if cond > 0 else ""
        base_str = ""
        if debug_state:
            base_quat = debug_state.get('base_imu_quat')
            if base_quat is not None and len(base_quat) >= 4:
                # Pitch from quaternion: arcsin(2*(w*y - z*x))
                w, x, y, z = float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3])
                pitch_rad = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
                base_str = f" | Base: [{_fmt_vec(base_quat, n=4)}] pitch={np.degrees(pitch_rad):+.1f}\u00b0"
        print(f"[{packet_rate:.1f} Hz] "
              f"Target: [{last_command.pose.x:+.3f}, {last_command.pose.y:+.3f}, {last_command.pose.z:+.3f}]m "
              f"Rot: {last_command.pose.rot_y_deg:+.1f}\u00b0 | "
              f"Actual: [{actual_pos[0]:+.3f}, {actual_pos[1]:+.3f}, {actual_pos[2]:+.3f}]m "
              f"Rot: {actual_rot:+.1f}\u00b0 | "
              f"Packets: {packets_received}"
              f"{cond_str}"
              f"{base_str}")

    def _print_perf(self, debug_state, network_times, packet_rate, perf_only):
        """Print compact performance line with stage breakdown."""
        if not debug_state:
            return
        perf_stats = debug_state.get('perf_stats', {})
        if not perf_stats:
            return

        loop_std = perf_stats.get('std_loop_time_ms', 0.0)
        loop_max = perf_stats.get('max_loop_time_ms', 0.0)
        loop_p95 = perf_stats.get('jitter_p95_ms', 0.0)
        loop_p99 = perf_stats.get('jitter_p99_ms', 0.0)
        loop_step = perf_stats.get('max_step_ms', 0.0)
        actual_hz = perf_stats.get('actual_hz', perf_stats.get('hz', 0.0))
        process_cpu = perf_stats.get('process_cpu_pct', 0.0)
        loop_util = perf_stats.get('loop_util_pct', perf_stats.get('cpu_usage_pct', 0.0))
        line = (f"Loop: {perf_stats.get('avg_loop_time_ms', 0):.2f}ms "
                f"(min={perf_stats.get('min_loop_time_ms', 0):.2f} max={loop_max:.2f}) | "
                f"Timing: std={loop_std:.2f} p95={loop_p95:.2f} p99={loop_p99:.2f} max\u0394={loop_step:.2f}ms | "
                f"Hz: {actual_hz:.1f} | "
                f"ProcCPU: {process_cpu:.1f}% | "
                f"LoopUtil: {loop_util:.1f}% | "
                f"Headroom: {perf_stats.get('avg_headroom_ms', 0):.2f}ms")

        # Stage breakdown
        stages = []

        sensor_avg = perf_stats.get('avg_sensor_ms', 0)
        sensor_min = perf_stats.get('min_sensor_ms', 0)
        sensor_max = perf_stats.get('max_sensor_ms', 0)
        if sensor_avg > 0:
            stages.append(f"Sensors: {sensor_avg:.2f}ms ({sensor_min:.2f}-{sensor_max:.2f})")

        ik_avg = perf_stats.get('avg_ik_fk_ms', 0)
        ik_min = perf_stats.get('min_ik_fk_ms', 0)
        ik_max = perf_stats.get('max_ik_fk_ms', 0)
        if ik_avg > 0:
            stages.append(f"IK/FK: {ik_avg:.2f}ms ({ik_min:.2f}-{ik_max:.2f})")

        pwm_avg = perf_stats.get('avg_pwm_ms', 0)
        pwm_min = perf_stats.get('min_pwm_ms', 0)
        pwm_max = perf_stats.get('max_pwm_ms', 0)
        if pwm_avg > 0:
            stages.append(f"PWM: {pwm_avg:.2f}ms ({pwm_min:.2f}-{pwm_max:.2f})")

        if network_times and not perf_only:
            net_avg = np.mean(network_times)
            net_min = np.min(network_times)
            net_max = np.max(network_times)
            stages.append(f"Net: {net_avg:.2f}ms ({net_min:.2f}-{net_max:.2f})")

        if stages:
            line += " | " + " | ".join(stages)

        cond = debug_state.get('condition_number', 0)
        if cond > 0:
            line += f" | Cond: {cond:.1f}"

        late_cnt = perf_stats.get('deadline_miss_count_recent', perf_stats.get('overrun_count_recent', 0))
        late_pct = perf_stats.get('deadline_miss_pct_recent', perf_stats.get('overrun_pct_recent', 0.0))
        late_win = perf_stats.get('deadline_window_sec', perf_stats.get('overrun_window_sec', 0.0))
        comp_cnt = perf_stats.get('compute_overrun_count_recent', perf_stats.get('violation_count', 0))
        comp_pct = perf_stats.get('compute_overrun_pct_recent', perf_stats.get('violation_pct', 0.0))
        line += f" | Late: {int(late_cnt)} ({late_pct:.1f}% over {late_win:.1f}s)"
        line += f" | ComputeOV: {int(comp_cnt)} ({comp_pct:.1f}%)"

        if not perf_only:
            line += f" | Pkt: {packet_rate:.1f}Hz"

        # Joint velocity and limiter info
        if not perf_only:
            try:
                last_vel = perf_stats.get('last_joint_vel_degps', [])
                vel_cap = perf_stats.get('effective_vel_cap_degps', [])
                vel_lim_on = perf_stats.get('ik_vel_lim_enabled', False)
                if last_vel:
                    names = ['slew', 'boom', 'arm', 'bucket']
                    parts = [f"{n}={v:+.1f}" for n, v in zip(names, last_vel)]
                    line += " | Vel(deg/s): " + ", ".join(parts)
                    if vel_lim_on and vel_cap:
                        cap_parts = [f"{n}={c:.1f}" for n, c in zip(names, vel_cap)]
                        line += " | Cap(deg/s): " + ", ".join(cap_parts)
            except Exception:
                pass

        print(line)

    def _print_jacobian(self, telemetry, actual_pos, last_command, debug_state):
        """Print Jacobian debug block."""
        if not debug_state:
            return
        fk_quats = debug_state.get('fk_quaternions')
        robot_config = debug_state.get('robot_config')
        if fk_quats is None or len(fk_quats) < 4 or robot_config is None:
            return

        try:
            from .differential_ik import compute_jacobian, extract_axis_rotation, project_to_rotation_axes

            J = compute_jacobian(fk_quats, robot_config)
            J_pos = J[0:3, :]
            J_ang = J[3:6, :]
            s = np.linalg.svd(J_pos, compute_uv=False)
            sv = s.astype(np.float32)
            s_nonzero = s[s > 1e-9]
            cond = float(np.max(s_nonzero) / np.min(s_nonzero)) if len(s_nonzero) > 0 else float('inf')

            axes = np.array(robot_config.rotation_axes, dtype=np.float32)
            projected = project_to_rotation_axes(fk_quats, axes)
            joint_angles = np.array([
                extract_axis_rotation(q, axis) for q, axis in zip(projected, robot_config.rotation_axes)
            ], dtype=np.float32)

            ee_pos = np.asarray(telemetry.joint_positions[4], dtype=np.float32)
            joint_deg = np.degrees(joint_angles).astype(np.float32)

            dq = dq_deg = ee_pred = None
            if self.prev_joint_angles is not None:
                dq = (joint_angles - self.prev_joint_angles).astype(np.float32)
                dq_deg = np.degrees(dq).astype(np.float32)
                ee_pred = (J_pos @ dq).astype(np.float32)

            ee_actual = None
            if self.prev_ee_pos is not None:
                ee_actual = (ee_pos - self.prev_ee_pos).astype(np.float32)

            self.prev_joint_angles = joint_angles
            self.prev_ee_pos = ee_pos.astype(np.float32)

            # Print
            print(f"  [JAC] s=[{_fmt_vec(sv, n=min(3, len(sv)))}] cond={cond:.2f}")
            if dq is not None:
                print(f"  [JAC] dq_deg=[{_fmt_vec(dq_deg, n=4)}]; ee_pred=[{_fmt_vec(ee_pred, n=3)}] m")
            if ee_actual is not None:
                print(f"  [JAC] ee_act=[{_fmt_vec(ee_actual, n=3)}] m")
            jp = J_pos.astype(np.float32)
            print(f"  [Jpos] r0=[{_fmt_vec(jp[0], n=4)}] | r1=[{_fmt_vec(jp[1], n=4)}] | r2=[{_fmt_vec(jp[2], n=4)}]")
            ja = J_ang.astype(np.float32)
            print(f"  [Jang] r3=[{_fmt_vec(ja[0], n=4)}] | r4=[{_fmt_vec(ja[1], n=4)}] | r5=[{_fmt_vec(ja[2], n=4)}]")

            if joint_deg is not None:
                slew_deg = float(joint_deg[0]) if len(joint_deg) > 0 else 0.0
                dyaw = float(dq_deg[0]) if (dq_deg is not None and len(dq_deg) > 0) else 0.0
                # Use a rough dt of 1s (this is called once per print interval)
                print(
                    f"  [JNT] deg=[{_fmt_vec(joint_deg, n=4)}]; "
                    f"[SLEW] yaw={slew_deg:+.1f} deg dyaw={dyaw:+.2f} deg"
                )

            if last_command:
                target = np.array([last_command.pose.x, last_command.pose.y, last_command.pose.z], dtype=np.float32)
                pos_err = target - np.asarray(actual_pos, dtype=np.float32)
                print(f"  [ERR] pos=[{_fmt_vec(pos_err, n=3)}] m")
        except Exception:
            pass
