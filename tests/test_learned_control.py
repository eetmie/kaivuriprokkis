"""learned_control without hardware: circle timing, PID path, recording, IMU reading and the latched output gate."""

import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learned_control.circle import CircleJointPID, CircleTrajectory, PidGains, StartMove  # noqa: E402
from learned_control.hardware import ImuReader, OutputGate, raw_imu_values  # noqa: E402
from learned_control.recording import BufferedRecording  # noqa: E402
from learned_control.run_circle import (  # noqa: E402
    check_motion_envelope,
    compare_runs,
    operator_enabled,
    repeat_phase,
    summarize,
)
from learned_control.sensors import MOUNT_PITCH, aligned_rates, imu_positions, policy_joint_offset  # noqa: E402


def test_buffer_keeps_multiple_passes_and_writes_only_when_requested(tmp_path):
    fields = ["armed", "pass_index", "device_ts_us", "sample"]
    buffer = BufferedRecording(fields, 300)
    path = tmp_path / "passes.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        buffer.append([True, 0, 1234567890, 1.2])
        buffer.append([True, 1, 1234567891, 2.3])
        assert len(path.read_text().splitlines()) == 1
        buffer.write_to(writer)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["pass_index"] for r in rows] == ["0", "1"]
    assert rows[0]["armed"] == "True"
    assert rows[0]["device_ts_us"] == "1234567890"
    assert buffer.samples.nbytes < 30_000_000


def test_raw_imu_logs_all_axes_in_physical_sensor_order():
    snapshot = SimpleNamespace(
        raw_accel=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        raw_gyro=[[21, 22, 23], [24, 25, 26], [27, 28, 29], [30, 31, 32]],
    )
    values = np.array(raw_imu_values(snapshot)).reshape(4, 6)
    np.testing.assert_array_equal(values[1], [4, 5, 6, 24, 25, 26])
    snapshot.raw_accel = None
    with pytest.raises(ValueError, match="raw accelerometer/gyro"):
        raw_imu_values(snapshot)


def test_rocking_does_not_trip_the_driven_joint_rate_guard():
    q = np.array([0, 1, 0, math.radians(2)])
    v = np.array([0.1, 0.2, -0.3, 3.0])
    check_motion_envelope(q, v, 3, 2)
    with pytest.raises(ValueError, match="Carriage pitch"):
        check_motion_envelope(q, v, 1, 2)
    v[1] = 2.1
    with pytest.raises(ValueError, match="arm rate"):
        check_motion_envelope(q, v, 3, 2)
    v[1] = math.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        check_motion_envelope(q, v, 3, 2)


def test_start_move_is_smooth_rate_bounded_and_rejects_physical_bounds():
    initial = torch.tensor([[0.0, 1.0, -0.2, 0.0]])
    target = torch.tensor([[0.2, 1.5, -0.4, 0.0]])
    kin = SimpleNamespace(valid=lambda q: (q[:, :3].abs() <= 2).all(1))
    move = StartMove(initial, target, kin)
    torch.testing.assert_close(move.reference(0), initial[:, :3])
    torch.testing.assert_close(move.reference(move.duration), target[:, :3])
    times = np.linspace(0, move.duration, 1001)
    positions = torch.cat([move.reference(t) for t in times]).numpy()
    assert np.abs(np.diff(positions, axis=0) / np.diff(times)[:, None]).max() <= 0.0501
    target[0, 1] = 2.5
    with pytest.raises(ValueError, match="physical joint/collision"):
        StartMove(initial, target, kin)


def test_continuous_circle_keeps_original_center_and_restarts_after_settling():
    path = CircleTrajectory([0.5, 0.1, 2.7])
    for lap in (0, 1, 37):
        index, phase = repeat_phase(path, lap * path.duration + 2.0)
        assert index == lap
        np.testing.assert_allclose(path.reference(phase)[0], path.reference(2.0)[0], atol=1e-12)
    index, phase = repeat_phase(path, path.duration)
    assert index == 1 and phase == 0
    np.testing.assert_array_equal(path.reference(phase)[0], path.initial)


def test_autonomous_enable_survives_lb_release_b_logs_and_a_or_disconnect_stop():
    connected = [True]
    pad = SimpleNamespace(LeftBumper=False, A=False, B=True, is_connected=lambda: connected[0])
    assert not operator_enabled(pad, True, False)
    assert operator_enabled(pad, True, True)
    pad.A = True
    assert not operator_enabled(pad, True, True)
    pad.A = False
    connected[0] = False
    assert not operator_enabled(pad, True, True)
    connected[0] = True
    pad.B, pad.LeftBumper = False, True
    assert operator_enabled(pad, False, False)
    pad.LeftBumper = False
    assert not operator_enabled(pad, False, False)


def test_circle_restarts_at_zero_velocity_each_lap():
    path = CircleTrajectory([0.6, 0.03, 0], cycles=3)
    for lap in range(4):
        pose, twist = path.reference(path.lead + lap * path.lap_s)
        np.testing.assert_allclose(pose, path.initial, atol=1e-12)
        np.testing.assert_allclose(twist, 0, atol=1e-12)
    with pytest.raises(ValueError):
        CircleTrajectory([0.6, 0.03, 0], speed_m_s=float("nan"))


def test_gate_forces_slew_tracks_zero_and_latches_release():
    clock, enabled, written, pump = [10.0], [True], [], []
    hardware = SimpleNamespace(
        send_named_pwm_commands=lambda commands, **kwargs: written.append(commands.copy()) or True,
        set_pump_enabled=lambda value: pump.append(value),
        reset=lambda **kwargs: None,
    )
    gate = OutputGate(hardware, lambda: enabled[0], lambda: clock[0])
    gate.sensor_time = gate.policy_time = clock[0]
    gate.arm()
    gate.write({"boom": 0.2, "slew": 1, "trackL": 1, "trackR": 1})
    assert written[-1] == {"boom": 0.2, "arm": 0.0, "bucket": 0.0, "slew": 0.0, "trackL": 0.0, "trackR": 0.0}
    enabled[0] = False
    gate.check()
    enabled[0] = True
    gate.write({"boom": 1})
    assert not gate.armed and not any(written[-1].values()) and pump[-1] is False
    with pytest.raises(RuntimeError, match="latched"):
        gate.arm()


def test_summary_distinguishes_radial_error_from_timing_error_and_keeps_faults():
    path = CircleTrajectory([0.6, 0.03, 0])
    rows = [
        dict(
            armed=True,
            motion_t_s=2.0 + i * 0.01,
            error_m=0.01,
            radial_error_m=0.0,
            angle_error_rad=0,
            boom_emitted_u=0.2,
            arm_emitted_u=0.2,
            bucket_emitted_u=0.0,
            compute_ms=1.0,
        )
        for i in range(3)
    ]
    result = summarize(rows, path, "operator stop")
    assert not result["completed"] and result["fault"] == "operator stop"
    assert result["tracking_rmse_mm"] == pytest.approx(10)
    assert result["radial_max_abs_mm"] == 0


def test_arm_mounting_change_preserves_the_frozen_policy_observation_frame():
    runtime_mount = MOUNT_PITCH.copy()
    runtime_mount[2] = math.radians(0.612)
    profile = {
        "mounting_offsets_quat": {
            role: [math.cos(a / 2), 0, math.sin(a / 2), 0]
            for role, a in zip(("base", "boom", "arm", "bucket"), runtime_mount, strict=True)
        }
    }
    raw_pitch = np.array([0.0, -0.4, 0.7, -0.2])

    def quats(pitch):
        return np.stack([np.cos(pitch / 2), pitch * 0, np.sin(pitch / 2), pitch * 0], axis=-1)

    physical_q = imu_positions(quats(raw_pitch - runtime_mount), corrected=True)
    training_q = imu_positions(quats(raw_pitch), corrected=False)
    offset = policy_joint_offset(profile)
    np.testing.assert_allclose(physical_q + offset, training_q, atol=1e-7)
    # The arm changes by -0.388 degrees in policy coordinates; the bucket changes oppositely.
    assert math.degrees(offset[1]) == pytest.approx(-0.388, abs=1e-5)
    assert math.degrees(offset[2]) == pytest.approx(+0.388, abs=1e-5)
    assert offset[0] == 0 and offset[3] == 0


def test_comparison_keeps_stopped_trials_and_rejects_changed_calibration(tmp_path):
    logs = [tmp_path / "pid.csv", tmp_path / "mlp.csv"]
    reports = []
    for path, controller in zip(logs, ("pid_tuned", "mlp"), strict=True):
        report = {
            "controller": controller,
            "direction": "ccw",
            "radius_mm": 50,
            "speed_mm_s": 20,
            "cycles": 1,
            "profile_sha256": {"control_config.yaml": "same calibration"},
            "result": {
                "completed": controller == "pid_tuned",
                "fault": None if controller == "pid_tuned" else "stop",
                "tracking_rmse_mm": 5.0,
                "radial_max_abs_mm": 3.0,
            },
        }
        path.with_suffix(".json").write_text(json.dumps(report))
        reports.append(report)
    args = SimpleNamespace(logs=logs, out=tmp_path / "comparison", plot=False)
    compare_runs(args)
    result = json.loads(args.out.with_suffix(".json").read_text())
    assert len(result["runs"]) == 2 and result["runs"][1]["fault"] == "stop"
    assert not result["runs"][1]["completed"]
    reports[1]["profile_sha256"]["control_config.yaml"] = "changed calibration"
    logs[1].with_suffix(".json").write_text(json.dumps(reports[1]))
    with pytest.raises(ValueError, match="profile_sha256"):
        compare_runs(SimpleNamespace(logs=logs, out=tmp_path / "different", plot=False))
    assert not (tmp_path / "different.csv").exists()


def test_aligned_rates_remove_inherited_link_motion():
    np.testing.assert_allclose(
        aligned_rates(np.array([7.0, 17.0, 13.0, 33.0])), np.deg2rad([10, -4, 20, 7]), rtol=1e-6
    )
    np.testing.assert_allclose(aligned_rates(np.full(4, 20))[:3], 0)


def test_mounting_applied_exactly_once():
    pitch = np.array([0.1, -0.3, 0.7, -0.2])

    def quaternion(p):
        return np.stack((np.cos(p / 2), p * 0, np.sin(p / 2), p * 0), -1)

    raw = imu_positions(quaternion(pitch + MOUNT_PITCH))
    corrected = imu_positions(quaternion(pitch), corrected=True)
    np.testing.assert_allclose(raw, corrected, atol=1e-7)
    np.testing.assert_allclose(raw[:3], np.diff(pitch), atol=1e-7)


class FakeHardware:
    def __init__(self):
        self.writes = []
        self.pump = False
        self.resets = 0
        self.accept = True

    def send_named_pwm_commands(self, commands, **kwargs):
        self.writes.append(commands.copy())
        return self.accept

    def set_pump_enabled(self, enabled):
        self.pump = enabled
        return True

    def reset(self, reset_pump=False):
        self.resets += 1


@pytest.mark.parametrize("fault", ["deadman", "sensor", "policy", "nan", "range", "rejected"])
def test_output_fault_latches_pump_off_and_cannot_be_rearmed(fault):
    hardware = FakeHardware()
    now, enabled = [1.0], [True]
    gate = OutputGate(hardware, lambda: enabled[0], clock=lambda: now[0])
    gate.sensor_time = gate.policy_time = now[0]
    gate.arm()
    hardware.set_pump_enabled(True)
    gate.write({"boom": 0.2, "slew": 1.0, "trackL": 1.0})
    assert hardware.writes[-1]["boom"] == 0.2
    assert hardware.writes[-1]["slew"] == hardware.writes[-1]["trackL"] == 0
    if fault == "deadman":
        enabled[0] = False
    elif fault == "sensor":
        now[0] += 0.051
    elif fault == "policy":
        now[0] += 0.151
        gate.sensor_time = now[0]
    elif fault == "rejected":
        hardware.accept = False
    command = {"boom": float("nan") if fault == "nan" else 1.01 if fault == "range" else 0.2}
    gate.write(command)
    assert gate.fault and not gate.armed and not hardware.pump and hardware.resets
    enabled[0] = hardware.accept = True
    gate.sensor_time = gate.policy_time = now[0]
    gate.write({"boom": 1.0, "arm": 1.0, "bucket": 1.0})
    assert all(v == 0 for v in hardware.writes[-1].values())
    with pytest.raises(RuntimeError, match="latched"):
        gate.arm()


def imu_hardware(stamp=10000):
    snapshot = SimpleNamespace(
        device_ts=stamp,
        imu_by_role={r: [1.0, 0.0, 0.0, 0.0] for r in ("base", "boom", "arm", "bucket")},
        imu_gyro=[[0.0, 20.0, 0.0], [0.0, 30.0, 0.0], [0.0, 40.0, 0.0]],
        base_imu_gyro=[0.0, 10.0, 0.0],
    )
    return SimpleNamespace(
        _imu_snapshot=snapshot, _imu_joint_roles=["boom", "arm", "bucket"], is_hardware_ready=lambda: True
    )


def test_imu_reader_same_packet_units_wrap_and_freshness():
    hardware = imu_hardware(2**32 - 5000)
    reader = ImuReader(hardware)
    _, v, _ = reader.read(0.0)
    np.testing.assert_allclose(v, np.deg2rad([10.0] * 4), rtol=1e-6)
    hardware._imu_snapshot.device_ts = 5000
    reader.read(0.01)
    reader.read(0.04)
    with pytest.raises(RuntimeError, match="stale"):
        reader.read(0.061)


@pytest.mark.parametrize("failure", ["reverse", "gap", "slow", "nan"])
def test_imu_reader_rejects_bad_clocks_and_measurements(failure):
    hardware = imu_hardware()
    reader = ImuReader(hardware)
    reader.read(0.0)
    if failure == "reverse":
        hardware._imu_snapshot.device_ts -= 1
    elif failure == "gap":
        hardware._imu_snapshot.device_ts += 110000
    elif failure == "nan":
        hardware._imu_snapshot.imu_gyro[0][1] = float("nan")
    else:
        hardware._imu_snapshot.device_ts += 1000
    with pytest.raises((RuntimeError, ValueError)):
        reader.read(0.06 if failure == "slow" else 0.01)


def test_independent_monitor_stops_without_another_output_write():
    hardware = FakeHardware()
    now = [0.0]
    gate = OutputGate(hardware, lambda: True, clock=lambda: now[0])
    gate.sensor_time = gate.policy_time = now[0]
    gate.arm()
    hardware.set_pump_enabled(True)
    now[0] = 0.051
    gate.check()
    assert not gate.armed and not hardware.pump and gate.fault == "sensor timeout"
    assert not hardware.writes


def test_circle_pid_drives_the_robot_pid_like_the_ik_loop():
    """Each joint gets ``compute(0, -wrap(target - q))`` from modules.pid, the robot's own controller."""
    from modules.pid import PIDController

    gains = PidGains(kp=[10, 11, 5], ki=[0.4, 0.3, 0.25], kd=[0.2, 0.1, 0.05])
    hardware = CircleJointPID(gains, kin=None)
    reference = [PIDController(kp=p, ki=i, kd=d) for p, i, d in zip(gains.kp, gains.ki, gains.kd)]
    q = torch.tensor([[0.1, 1.2, -0.7, 0.0]])
    for k in range(50):
        target = q[:, :3] + 0.01 * math.sin(k / 7)
        expected = [pid.compute(0.0, -float(t - x), 0.01) for pid, t, x in zip(reference, target[0], q[0, :3])]
        np.testing.assert_allclose(hardware.joint_valves(q, target, 0.01)[0].numpy(), expected, rtol=1e-6, atol=1e-7)
