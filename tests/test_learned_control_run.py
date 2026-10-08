"""learned_control.run_circle run() end to end against a fake robot: bundles, valve write path, drive-log output."""

import json
import sys
import threading
import time
import shutil
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learned_control import run_circle  # noqa: E402
from learned_control.bundle import PolicyBundle  # noqa: E402
from learned_control.settings import HOME  # noqa: E402

BUNDLES = ROOT / "learned_control/bundles"
GAINS = ROOT / "learned_control/gains/pid_sim_tuned.yaml"
PROFILE = "jetson_bucket_dz0"
RATE = 0.3  # fake joint speed [rad/s] per unit valve; positive valves raise the angle, as on the robot


class FakeHardware:
    """Integrates written valves into joint motion; IMU reads see the result."""

    def __init__(self, **kwargs):
        self.q = np.array(HOME, dtype=np.float64)
        self.u = np.zeros(3)
        self.last = time.monotonic()
        self.lock = threading.Lock()
        self.writes = []  # (thread name, boom/arm/bucket)
        self._imu_joint_roles = ("boom", "arm", "bucket")

    def send_named_pwm_commands(self, commands, **kwargs):
        with self.lock:
            self.advance()
            self.u = np.array([commands[name] for name in ("boom", "arm", "bucket")])
            self.writes.append((threading.current_thread().name, self.u.copy()))
        return True

    def advance(self):
        now = time.monotonic()
        self.q[:3] += RATE * self.u * (now - self.last)
        self.last = now

    def set_pump_enabled(self, enabled):
        return True

    def reset(self, reset_pump=False):
        with self.lock:
            self.advance()
            self.u[:] = 0

    def is_hardware_ready(self):
        return True

    def shutdown(self):
        pass

    # Raw capture: one 200 Hz frame per 5 ms of wall time, four sensors of 10 values.
    def imu_stream_info(self):
        return {"roles_by_index": ["boom", "arm", "bucket", "base"], "ranges": None, "capture_dropped": 0}

    def start_imu_raw_capture(self):
        self.capture_us = int(time.monotonic() * 1e6) // 5000 * 5000
        return True

    def drain_imu_raw_capture(self):
        now = int(time.monotonic() * 1e6)
        frames = [(ts, [[1.0] + [0.0] * 9] * 4) for ts in range(self.capture_us + 5000, now + 1, 5000)]
        self.capture_us = frames[-1][0] if frames else self.capture_us
        return frames

    def try_read_imu_gyro(self):
        return None


class FakeReader:
    """Stands in for ImuReader: angles and rates straight from the fake plant."""

    stamp = 0

    def __init__(self, hardware, max_age_s=0.05):
        self.hardware, self.fresh_time = hardware, None
        quat = [1.0, 0.0, 0.0, 0.0]
        self.snapshot = SimpleNamespace(
            imu_by_role={role: quat for role in ("base", "boom", "arm", "bucket")},
            imu_gyro=[[0.0, 0.0, 0.0]] * 3,
            base_imu_gyro=[0.0, 0.0, 0.0],
        )

    def read(self, now=None):
        hw = self.hardware
        with hw.lock:
            hw.advance()
            q, v = hw.q.astype(np.float32), np.r_[RATE * hw.u, 0.0].astype(np.float32)
        FakeReader.stamp += 10000
        self.fresh_time = time.monotonic() if now is None else now
        return q, v, FakeReader.stamp


class FakePad:
    """Times are seconds after creation; None never presses. Continuous runs set B and A."""

    press_b = press_a = None

    def __init__(self):
        self.created = time.monotonic()

    def held(self, after):
        return after is not None and time.monotonic() - self.created > after

    @property
    def A(self):
        return self.held(self.press_a)

    @property
    def B(self):
        return self.held(self.press_b)

    @property
    def LeftBumper(self):  # released at first, held from 1.2 s on
        return time.monotonic() - self.created > 1.2

    def is_connected(self):
        return True

    def stop_monitoring(self):
        pass


class FakeController:
    """ExcavatorController stand-in; in direct-command mode its own thread writes the held setpoint."""

    instances = []

    def __init__(self, hardware, control_config_file=None):
        self.hardware, self.events, self.setpoint = hardware, [], None
        self.stop_event = threading.Event()
        FakeController.instances.append(self)

    def enter_direct_command_mode(self, **kwargs):
        self.events.append("direct")

    def suspend_ik_output(self):
        self.events.append("suspend")

    def give_direct_commands(self, commands):
        self.setpoint = dict(commands)

    def get_joint_angles(self):
        with self.hardware.lock:
            q = self.hardware.q.copy()
        return np.degrees(np.r_[0.0, q[:3]]), time.perf_counter(), FakeReader.stamp

    def get_joint_velocities_with_age(self):
        return list(np.degrees(np.r_[0.0, RATE * self.hardware.u])), 0.001

    def start(self):
        self.events.append("start")
        if "direct" in self.events:
            threading.Thread(target=self.loop, name="controller", daemon=True).start()

    def loop(self):
        while not self.stop_event.wait(0.01):
            self.hardware.send_named_pwm_commands(self.setpoint or {"boom": 0, "arm": 0, "bucket": 0})

    def stop(self):
        self.stop_event.set()


def drive_circle(monkeypatch, tmp_path, valve_writes, controller="pid_tuned", extra=()):
    # The real board profiles and DirectController; fakes for everything that opens a device.
    fakes = {
        "bringup": {"wait_for_hardware_ready": lambda hardware: None},
        "excavator_controller": {"ExcavatorController": FakeController},
        "gamepad": {"XboxController": FakePad},
        "hardware_interface": {"HardwareInterface": FakeHardware},
    }
    for name, attributes in fakes.items():
        module = types.ModuleType(f"modules.{name}")
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, f"modules.{name}", module)
    monkeypatch.setattr(run_circle, "ImuReader", FakeReader)
    FakeController.instances.clear()
    argv = [
        "run_circle.py", "run",
        "--bundle", str(BUNDLES / f"proto_{PROFILE}"),
        "--robot", PROFILE,
        "--controller", controller,
        "--pid_gains", str(GAINS),
        "--radius_mm", "5",
        "--valve_writes", valve_writes,
        "--out_dir", str(tmp_path),
        "--label", valve_writes,
        *extra,
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    run_circle.main()
    (log,) = tmp_path.glob(f"drive_log_*_circle_{controller}_ccw_{valve_writes}.csv")
    raw = tmp_path / log.name.replace("drive_log_", "imu_raw_", 1)
    report = json.loads(log.with_suffix(".json").read_text())
    return pd.read_csv(log), pd.read_csv(raw), report, FakeController.instances[0]


def test_loop_writes_each_command_in_the_tick_it_is_computed(monkeypatch, tmp_path):
    rows, _, report, controller = drive_circle(monkeypatch, tmp_path, "loop")
    assert report["result"]["completed"] and report["valve_writes"] == "loop"
    assert controller.events == ["suspend", "start"]
    writers = {name for name, _ in controller.hardware.writes}
    assert writers == {"MainThread"}  # only the measurement loop drives the valves
    requested = rows[[f"{j}_requested_u" for j in ("boom", "arm", "bucket")]].to_numpy()
    written = rows[list(run_circle.VALVE_COLUMNS)].to_numpy()
    armed = rows["armed"].to_numpy() > 0
    assert (armed & (np.abs(requested).sum(1) > 0)).sum() > 100
    # The logged valve command is what this tick computed and wrote, neutral while unarmed.
    np.testing.assert_allclose(written[armed], requested[armed], atol=1e-6)
    assert not written[~armed].any()
    assert (rows["cmd_age_s"][armed] < 0.005).all()


def test_circle_log_is_a_drive_log_the_training_loader_accepts(monkeypatch, tmp_path):
    rows, raw, report, _ = drive_circle(monkeypatch, tmp_path, "loop")
    # The training contract: Isaac-hydraulic-actuator training/dataset.py and gyro_transfer.py.
    required = [
        "timestamp", "sample_idx", "state_imu_ts_us", "cmd_stale", "cmd_age_s", "state_age_s", "vel_age_s",
        *(f"combined_cmd_{c}" for c in ("lift", "tilt", "scoop")),
        *(f"joint_pos_{j}" for j in ("boom", "arm", "bucket")),
        *(f"joint_vel_{j}" for j in ("boom", "arm", "bucket")),
    ]  # fmt: skip
    assert not set(required) - set(rows.columns)
    assert (np.diff(rows["sample_idx"]) == 1).all() and (np.diff(rows["timestamp"]) > 0).all()
    np.testing.assert_array_equal(rows["cmd_stale"] > 0, rows["armed"] == 0)
    assert set(rows["excitation_stage"]) <= {"wait", "approach", "circle", "stopped"}
    assert (rows["excitation_mode"] == "circle_pid_tuned").all()
    np.testing.assert_allclose(rows["joint_pos_boom"], rows["q_boom"], atol=0.02)
    # Every 200 Hz frame of the run, joinable on the Pico clock.
    assert (np.diff(raw["device_ts_us"]) == 5000).all()
    assert len(raw) >= 1.9 * len(rows)
    assert {"imu_base_qw", "imu_bucket_gy_dps"} <= set(raw.columns)
    assert report["drive_log"].startswith("drive_log_") and report["imu_raw"].startswith("imu_raw_")


def test_continuous_run_logs_from_b_and_scores_each_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(FakePad, "press_b", 2.5)
    monkeypatch.setattr(FakePad, "press_a", 4.5)
    rows, _, report, _ = drive_circle(monkeypatch, tmp_path, "loop", extra=["--continuous"])
    assert report["result"]["stopped_by"] == "A"
    # Only B onwards is recorded, with the drive-log clock starting at B.
    assert rows["timestamp"].iloc[0] < 0.02 and rows["run_t_s"].iloc[0] == pytest.approx(
        report["logging_started_at_s"], abs=0.02
    )
    assert 1.5 < rows["timestamp"].iloc[-1] < 2.5
    assert report["passes"] and all(p["pass_index"] in set(rows["pass_index"]) for p in report["passes"])


def test_thread_mode_keeps_the_direct_command_path(monkeypatch, tmp_path):
    rows, _, report, controller = drive_circle(monkeypatch, tmp_path, "thread")
    assert report["result"]["completed"] and report["valve_writes"] == "thread"
    assert controller.events == ["direct", "start"]
    assert any(name == "controller" for name, _ in controller.hardware.writes)


@pytest.mark.parametrize("bundle", sorted(p.name for p in BUNDLES.iterdir()))
def test_committed_bundles_replay_their_reference_cases(bundle):
    loaded = PolicyBundle(BUNDLES / bundle)
    assert loaded.verified
    profile = bundle.split("_", 1)[1]
    loaded.check_profile(ROOT / "configuration_files/profiles" / profile)


def test_bundle_refuses_a_changed_file_or_a_drifted_runtime(tmp_path, monkeypatch):
    copy = tmp_path / "bundle"
    shutil.copytree(BUNDLES / f"proto_{PROFILE}", copy)
    with (copy / "geometry.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Bundle file changed: geometry.json"):
        PolicyBundle(copy)
    from learned_control import kinematics

    original = kinematics.ExcavatorKinematics.pose_jacobian

    def drifted(self, q):
        pose, jacobian = original(self, q)
        return pose + 1e-3, jacobian

    monkeypatch.setattr(kinematics.ExcavatorKinematics, "pose_jacobian", drifted)
    with pytest.raises(ValueError, match="reference mismatch in pose"):
        PolicyBundle(BUNDLES / f"proto_{PROFILE}")
