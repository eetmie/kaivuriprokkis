"""Check, run and score the same bucket circle with PID or MLP.

Run on the robot with its .venv-lerobot Python. Only `run` opens hardware. Bundles come from
Isaac-hydraulic-actuator's export_robot_bundle.py, which also generates their robot profile.
Hold local gamepad Left Bumper to enable a finite run. Continuous mode starts
with LB, runs until A, and uses B to enable measurement logging. Gamepad
disconnect always stops. Slew, tracks and auxiliaries remain neutral.

A run records the same drive_log/imu_raw pair as simple_drive.py
(modules/drive_log.py), plus the circle columns in CIRCLE_FIELDS and a JSON
summary, so actuator-model training can read circle runs like operator data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import itertools
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_control.bundle import PolicyBundle  # noqa: E402
from learned_control.circle import CircleJointPID, CircleTrajectory, StartMove, load_robot_gains  # noqa: E402
from learned_control.hardware import ImuReader, OutputGate  # noqa: E402
from learned_control.robot_geometry import policy_twist  # noqa: E402
from learned_control.sensors import policy_joint_offset  # noqa: E402
from learned_control.settings import HOME, sha256  # noqa: E402
from modules.drive_log import COMMAND_CHANNELS, DriveLog, clean_suffix  # noqa: E402

PROFILES = ROOT / "configuration_files/profiles"
JOINTS = ("boom", "arm", "bucket")
CHANNELS = dict(COMMAND_CHANNELS)
VALVE_COLUMNS = tuple(f"combined_cmd_{CHANNELS[joint]}" for joint in JOINTS)
# Circle columns appended to each drive-log row. q/v and policy_device_ts_us are
# the run's own IMU state (what the controllers acted on); joint_pos_*/joint_vel_*
# come from the robot controller, as in simple_drive.py recordings.
CIRCLE_FIELDS = (
    "run_t_s",
    "motion_t_s",
    "armed",
    "pass_index",
    "policy_device_ts_us",
    "q_boom",
    "q_arm",
    "q_bucket",
    "q_pitch",
    "v_boom",
    "v_arm",
    "v_bucket",
    "v_pitch",
    "tip_x_m",
    "tip_z_m",
    "angle_rad",
    "ref_x_m",
    "ref_z_m",
    "ref_angle_rad",
    "ref_vx_m_s",
    "ref_vz_m_s",
    "ref_pitch_rad_s",
    "error_m",
    "radial_error_m",
    "angle_error_rad",
    "boom_requested_u",
    "arm_requested_u",
    "bucket_requested_u",
    "compute_ms",
    "lateness_ms",
)


def load_configuration(args):
    """Validate pinned profile and gains without opening robot devices."""
    if not math.isfinite(args.joint_margin_deg) or not 0 <= args.joint_margin_deg <= 10:
        raise ValueError("Extra joint margin must be finite and in [0, 10] degrees")
    bundle = PolicyBundle(args.bundle)
    for kin in (bundle.kin, bundle.robot_kin):
        kin.joint_margin = math.radians(args.joint_margin_deg)
    bundle.settings.joint_margin = math.radians(args.joint_margin_deg)
    profile_dir = PROFILES / args.robot
    bundle.check_profile(profile_dir)
    if not bundle.contract["sensor_trained"]:
        raise ValueError("The circle runner requires a gyro-trained policy")
    cfg = yaml.safe_load((profile_dir / "control_config.yaml").read_text())
    if cfg != bundle.robot_kin.profile:
        raise ValueError("Robot profile differs from the bundle's joint-based bucket geometry")
    offset = policy_joint_offset(cfg["imu"], bundle.contract["sensors"])
    if not np.allclose(offset, bundle.contract.get("policy_joint_offset_rad", [0] * 4), atol=1e-7, rtol=0):
        raise ValueError("Runtime-to-training joint calibration offset is missing or inconsistent")
    if cfg["rates"]["control_hz"] != 100 or bundle.settings.decimation != 5:
        raise ValueError("Use the trained 100 Hz state / 20 Hz policy contract")
    if any(abs(q[1]) + abs(q[3]) > 1e-6 for q in cfg["imu"]["mounting_offsets_quat"].values()):
        raise ValueError("This gyro adapter requires pure Y mounting corrections")
    gains_path = profile_dir / "control_config.yaml"
    if args.controller == "pid_tuned" and args.pid_gains is None:
        raise ValueError("pid_tuned requires --pid_gains; robot gains are not sim-tuned gains")
    if args.pid_gains is not None:
        override = yaml.safe_load(args.pid_gains.read_text())["pid"]
        for i in range(1, 4):
            values = override[f"joint{i}"]
            if any(not math.isfinite(float(values[k])) or float(values[k]) < 0 for k in ("kp", "ki", "kd")):
                raise ValueError("PID gains must be finite and nonnegative")
        # Load normal runtime limits/damping, then replace just the three arm gains.
        gains = load_robot_gains(gains_path)
        for key in ("kp", "ki", "kd"):
            setattr(gains, key, [float(override[f"joint{i}"][key]) for i in range(1, 4)])
    else:
        gains = load_robot_gains(gains_path)
    if not 0 < args.radius_mm <= 50 or not 0 < args.speed_mm_s <= 20 or not 1 <= args.cycles <= 3:
        raise ValueError("Initial demo bounds: radius <=50 mm, speed <=20 mm/s, 1..3 cycles")
    return bundle, CircleJointPID(gains, bundle.robot_kin)


def trajectory(args, pose) -> CircleTrajectory:
    return CircleTrajectory(pose, args.radius_mm / 1000, args.speed_mm_s / 1000, args.direction, args.cycles)


def start_target(args, kin, initial):
    """Use an earlier successful trial's fixed world tip pose, rather than the current pose."""
    if args.start_from_log is None:
        return initial, None
    with args.start_from_log.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    target = initial.new_tensor([[float(row[k]) for k in ("ref_x_m", "ref_z_m", "ref_angle_rad")]])
    solved, reached = kin.inverse(target, initial)
    if not bool(reached.all()):
        raise ValueError("Recorded circle start is outside physical joint/collision bounds")
    return solved, StartMove(initial, solved, kin)


def repeat_phase(path, motion_t):
    """Repeat a fixed circle without moving its center to follow tracking drift."""
    index = int(motion_t // path.duration)
    return index, motion_t - index * path.duration


def operator_enabled(pad, continuous, started):
    """Continuous motion is latched after LB start; A/disconnect always stop it."""
    if continuous:
        return pad.is_connected() and started and not bool(pad.A)
    return pad.is_connected() and bool(pad.LeftBumper) and not bool(pad.B)


def check_motion_envelope(q, v, max_pitch_deg, max_joint_velocity):
    """Bound driven joint rates separately from passive carriage rocking."""
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite measured joint state")
    for joint, rate in zip(JOINTS, v[:3], strict=True):
        if abs(rate) > max_joint_velocity:
            raise ValueError(f"{joint} rate {rate:.4f} rad/s exceeds {max_joint_velocity:.4f} rad/s")
    if abs(q[3]) > math.radians(max_pitch_deg):
        raise ValueError(f"Carriage pitch {math.degrees(q[3]):.3f} deg exceeds {max_pitch_deg:.3f} deg")


def check(args) -> None:
    """Run offline geometry/actor checks on this machine; no IMU/PWM initialization."""
    bundle, pid = load_configuration(args)
    q = torch.tensor([HOME], dtype=torch.float32)
    policy_q = q + torch.tensor(bundle.contract.get("policy_joint_offset_rad", [0] * 4))
    pose = bundle.robot_kin.pose_jacobian(q)[0][0].numpy()
    target_q, approach = start_target(args, bundle.robot_kin, q)
    path = trajectory(args, bundle.robot_kin.pose_jacobian(target_q)[0][0].numpy())
    path.validate(bundle.robot_kin, target_q)
    if approach is not None:
        print(
            f"Offline start-pose approach checked: {approach.duration:.2f}s at <=0.05 rad/s reference rate."
        )
    bundle.history.reset(torch.tensor([0]), policy_q)
    timings, actor_timings = [], []
    emitted, admitted = torch.zeros(1, 3), None
    for step in range(105):
        before = time.perf_counter()
        with torch.no_grad():
            bundle.history.push(policy_q, torch.zeros_like(q), emitted)
            bundle.robot_kin.valid(q)
            current = bundle.robot_kin.pose_jacobian(q)[0][0].numpy()
            if args.controller == "mlp":
                if admitted is None or step % 5 == 0:
                    physical = torch.tensor(path.command(step * 0.01, current)[None], dtype=torch.float32)
                    desired = policy_twist(q, physical, bundle.robot_kin, bundle.kin, policy_q=policy_q)
                    admitted, _ = bundle.governor(policy_q, torch.zeros_like(q), desired)
                if step % (100 // args.policy_hz) == 0:
                    actor_start = time.perf_counter()
                    emitted = bundle.valves(bundle.history.observe(bundle.kin, admitted))
                    actor_timings.append((time.perf_counter() - actor_start) * 1000)
            else:
                emitted = pid.valves(q, torch.tensor(pose[None]), 0.01)
        timings.append((time.perf_counter() - before) * 1000)
    print(f"Offline checks passed. HOME bucket tip: {pose[:2] * 1000} mm; lap {path.lap_s:.2f}s.")
    print(f"Warmed state/control ticks: mean {np.mean(timings[5:]):.2f} ms, max {max(timings[5:]):.2f} ms.")
    if actor_timings:
        print(
            f"Actor at {args.policy_hz} Hz: median {np.median(actor_timings[5:]):.2f} ms; projection 20 Hz."
        )


def summarize(rows: list[dict], path: CircleTrajectory, fault: str | None) -> dict:
    """Score timed tracking, geometric circle error and final settling in SI-derived units."""
    moving = [r for r in rows if r["armed"] and path.lead <= r["motion_t_s"] < path.move_end]
    settling = [r for r in rows if r["armed"] and r["motion_t_s"] >= path.move_end]
    result = {"completed": fault is None, "fault": fault, "moving_samples": len(moving)}
    if moving:
        errors = np.array([r["error_m"] for r in moving]) * 1000
        radial = np.array([r["radial_error_m"] for r in moving]) * 1000
        valves = np.array([[r[column] for column in VALVE_COLUMNS] for r in moving])
        duration = max(0.01, moving[-1]["motion_t_s"] - moving[0]["motion_t_s"])
        result.update(
            tracking_rmse_mm=float(np.sqrt(np.mean(errors**2))),
            tracking_p95_mm=float(np.percentile(errors, 95)),
            tracking_max_mm=float(errors.max()),
            radial_mean_abs_mm=float(np.abs(radial).mean()),
            radial_max_abs_mm=float(np.abs(radial).max()),
            angle_max_deg=float(np.rad2deg(max(abs(r["angle_error_rad"]) for r in moving))),
            valve_travel_per_s=float(np.abs(np.diff(valves, axis=0)).sum() / duration),
            saturation_fraction=float((np.abs(valves) >= 0.99).mean()),
            compute_max_ms=max(r["compute_ms"] for r in moving),
        )
    if settling:
        result["final_error_mm"] = settling[-1]["error_m"] * 1000
    return result


def run(args) -> None:
    """Run one operator-enabled experiment and save measurements even when it stops early."""
    bundle, pid = load_configuration(args)
    if not math.isfinite(args.approach_output_limit) or not 0 < args.approach_output_limit <= 1:
        raise ValueError("Approach output limit must be finite and in (0, 1]")
    approach_gains = copy.deepcopy(pid.gains)
    approach_gains.output_limits = (-args.approach_output_limit, args.approach_output_limit)
    approach_pid = CircleJointPID(approach_gains, bundle.robot_kin)
    policy_offset = torch.tensor(bundle.contract.get("policy_joint_offset_rad", [0] * 4))
    if not math.isfinite(args.max_error_mm) or not 0 < args.max_error_mm <= 50:
        raise ValueError("Use a finite tracking-error stop threshold in (0, 50] mm")
    if not math.isfinite(args.max_carriage_pitch_deg) or not 0 < args.max_carriage_pitch_deg <= 3:
        raise ValueError("Carriage pitch bound must be in (0, 3] degrees, within the model geometry")
    if not math.isfinite(args.max_joint_velocity_rad_s) or args.max_joint_velocity_rad_s <= 0:
        raise ValueError("Driven joint velocity bound must be finite and positive")
    if not math.isfinite(args.record_seconds) or not 0 < args.record_seconds <= 600:
        raise ValueError("Recording duration must be in (0, 600] seconds")
    # Reserve the summary before touching hardware; no artifact is ever overwritten.
    log_path, raw_path = run_paths(args)
    summary_path = log_path.with_suffix(".json")
    if log_path.exists() or raw_path.exists():
        raise FileExistsError(log_path)
    with summary_path.open("x") as summary_stream:
        from modules.board import resolve_profile
        from modules.bringup import wait_for_hardware_ready
        from modules.direct_controller import DirectController
        from modules.excavator_controller import ExcavatorController
        from modules.gamepad import XboxController
        from modules.hardware_interface import HardwareInterface

        profile = resolve_profile(args.robot)
        expected_dir = PROFILES / args.robot
        servo = (ROOT / profile["servo_config_file"]).resolve()
        control = (ROOT / profile["control_config_file"]).resolve()
        if servo != expected_dir / "servo_config.yaml" or control != expected_dir / "control_config.yaml":
            raise ValueError("Resolved board profile must use the checksum-pinned files")
        hardware = controller = direct = pad = gate = monitor = path = None
        monitor_stop = threading.Event()
        log, fault = DriveLog(), None
        imu_roles, stream_info = [], {}
        metadata = {
            "controller": args.controller,
            "robot": args.robot,
            "checkpoint_sha256": bundle.contract["checkpoint_sha256"],
            "contract_sha256": sha256(args.bundle / "contract.json"),
            "profile_sha256": bundle.contract["robot_profile_sha256"],
            "pid_gains_sha256": None if args.pid_gains is None else sha256(args.pid_gains),
            "pid_gains": vars(pid.gains),
            "direction": args.direction,
            "radius_mm": args.radius_mm,
            "speed_mm_s": args.speed_mm_s,
            "cycles": args.cycles,
            "origin": "slew_bearing",
            "slew_enabled": False,
            "policy_hz": args.policy_hz,
            "projection_hz": 20,
            "valve_writes": args.valve_writes,
            "continuous": args.continuous,
            "max_carriage_pitch_deg": args.max_carriage_pitch_deg,
            "max_joint_velocity_rad_s": args.max_joint_velocity_rad_s,
            "joint_margin_deg": args.joint_margin_deg,
            "start_from_log": None if args.start_from_log is None else str(args.start_from_log),
            "approach_output_limit": args.approach_output_limit,
            "passes": [],
            "imu_mapping": bundle.robot_kin.profile["imu"]["imu_mapping"],
            "drive_log": log_path.name,
            "imu_raw": raw_path.name,
            "allow_timing_overruns": args.allow_timing_overruns,
        }
        pass_index, stopped_by_a = 0, False
        recording = not args.continuous
        logging_started_at = None
        recording_complete = False
        metadata["record_seconds"] = args.record_seconds
        metadata["saving"] = "buffer in RAM; write and score after hardware shutdown"

        try:
            hardware = HardwareInterface(
                config_file=str(servo),
                control_config_file=str(control),
                enable_pwm=True,
                enable_imu=True,
                enable_adc=False,
                start_adc_reader=False,
                pump_auto_mode=False,
                toggle_channels=False,
                stale_timeout_s=0.15,
                cleanup_disable_osc=False,
                pwm_i2c_bus=profile["pwm_i2c_bus"],
                pwm_i2c_addr=profile["pwm_i2c_addr"],
            )
            if not hardware.set_pump_enabled(False):
                raise RuntimeError("Hardware rejected pump disable")
            hardware.reset(reset_pump=True)
            wait_for_hardware_ready(hardware)
            pad = XboxController()
            autonomous_started = False

            def deadman():
                return operator_enabled(pad, args.continuous, autonomous_started)

            gate = OutputGate(hardware, deadman)
            hardware.send_named_pwm_commands = gate.write
            controller = ExcavatorController(hardware, control_config_file=str(control))
            if args.valve_writes == "thread":
                controller.enter_direct_command_mode(
                    hold_timeout_s=0.15,
                    decay_s=0.0,
                    blend_s=0.0,
                    joint_names=JOINTS,
                )
            else:
                # This loop owns the bus and writes every tick, as simple_drive.py did when the actuator
                # data was recorded; the controller thread keeps its state but writes nothing.
                controller.suspend_ik_output()
                direct = DirectController(hardware)
            controller.start()

            def supervise():
                while not monitor_stop.wait(0.01):
                    try:
                        gate.check()
                    except Exception as exc:
                        gate.stop(exc)

            monitor = threading.Thread(target=supervise, daemon=True)
            monitor.start()
            # The same SCHED_FIFO priority simple_drive.py asks for when recording; it needs the realtime
            # limits from setup_jetson.sh and is recorded either way.
            from modules.rt_utils import SCHED_FIFO, apply_rt_to_thread

            metadata["realtime_fifo"] = bool(
                apply_rt_to_thread(priority=75, policy=SCHED_FIFO, lock_memory=False, quiet=True)
            )
            print(f"Measurement loop SCHED_FIFO 75: {'on' if metadata['realtime_fifo'] else 'unavailable'}")
            q, _, _ = ImuReader(hardware).read()
            qt = torch.tensor(q[None])
            bundle.history.reset(torch.tensor([0]), qt + policy_offset)
            with torch.no_grad():
                for _ in range(5):
                    bundle.valves(bundle.history.observe(bundle.kin, torch.zeros(1, 3)))
                    pose = bundle.robot_kin.pose_jacobian(qt)[0]
                    pid.valves(qt, pose, 0.01)
            q, _, _ = ImuReader(hardware).read()
            qt = torch.tensor(q[None])
            initial_pose = bundle.robot_kin.pose_jacobian(qt)[0][0].numpy()
            target_q, approach = start_target(args, bundle.robot_kin, qt)
            path = trajectory(args, bundle.robot_kin.pose_jacobian(target_q)[0][0].numpy())
            path.validate(bundle.robot_kin, target_q)
            metadata["preflight_q_rad"] = q.tolist()
            metadata["approach_duration_s"] = 0 if approach is None else approach.duration
            bundle.history.reset(torch.tensor([0]), qt + policy_offset)
            pid.reset()
            reader = ImuReader(hardware)
            reader.read()
            imu_roles = list(hardware.imu_stream_info()["roles_by_index"])
            if not hardware.start_imu_raw_capture():
                raise RuntimeError("Raw IMU capture is unavailable; the run would record no imu_raw strip")
            # Exempt the start-up heap (torch alone is large) from collection: a full pass over it
            # paused this loop for 20-40 ms. Later objects are still collected.
            gc.freeze()
            started = next_tick = previous_tick = time.monotonic()
            motion_start, released, step, compute_ms = None, False, 0, 0.0
            circle_start = None
            requested = np.zeros(3, dtype=np.float32)
            admitted = None
            print(
                "Circle preflight passed. Release LB, then hold LB to start after 1s history warmup. "
                + ("Then LB may be released: A stops; B starts logging." if args.continuous else "B stops."),
                flush=True,
            )
            while True:
                tick = time.monotonic()
                lateness = tick - next_tick
                if lateness > 0.03:
                    if motion_start is not None and not args.allow_timing_overruns:
                        raise RuntimeError("100 Hz measurement loop missed its deadline by >30 ms")
                    # Device startup can delay a neutral tick; establish a fresh
                    # schedule while pump/output remain off, before operator enable.
                    next_tick = tick
                dt = 0.01 if step == 0 else tick - previous_tick
                previous_tick = tick
                q, v, stamp = reader.read(tick)
                metadata["last_state"] = {
                    "q_rad": q.tolist(),
                    "velocity_rad_s": v.tolist(),
                    "device_ts_us": stamp,
                }
                with gate.lock:
                    gate.sensor_time = reader.fresh_time
                if args.continuous and pad.A:
                    stopped_by_a = True
                    break
                if gate.fault:
                    raise RuntimeError(gate.fault)
                if args.continuous and pad.B and not recording:
                    recording = True
                    logging_started_at = tick - started
                    hardware.drain_imu_raw_capture()  # frames before B are warm-up
                    print(
                        f"B: recording {args.record_seconds:g}s into memory; saving after pump-off.",
                        flush=True,
                    )
                if not args.continuous and pad.B:
                    raise RuntimeError("Operator pressed B")
                qt, vt = torch.tensor(q[None]), torch.tensor(v[None])
                check_motion_envelope(q, v, args.max_carriage_pitch_deg, args.max_joint_velocity_rad_s)
                if not bool(bundle.robot_kin.valid(qt)[0]):
                    raise ValueError("Measured joint state is outside joint/collision margins")
                policy_q = qt + policy_offset
                # Training histories hold the controller's own command for the preceding interval;
                # the emitted value trails it by the direct-command thread (~2 ticks), which the
                # actor never saw. A blocked gate sends nothing, so it records zeros.
                sent = requested if gate.armed else np.zeros(3, dtype=np.float32)
                bundle.history.push(policy_q, vt, torch.tensor(sent[None], dtype=torch.float32))
                pose = bundle.robot_kin.pose_jacobian(qt)[0][0].numpy()
                elapsed = tick - started
                released |= not bool(pad.LeftBumper)
                start_requested = pad.is_connected() and bool(pad.LeftBumper) and not bool(pad.A)
                if motion_start is None and elapsed >= 1 and released and start_requested:
                    displacement = np.linalg.norm(pose[:2] - initial_pose[:2])
                    angle_change = math.atan2(
                        math.sin(pose[2] - initial_pose[2]), math.cos(pose[2] - initial_pose[2])
                    )
                    if displacement > 0.005 or abs(angle_change) > math.radians(1):
                        raise RuntimeError("Robot moved since preflight; restart while stationary")
                    pid.reset()
                    motion_start = tick
                    autonomous_started = True
                approaching = motion_start is not None and approach is not None and circle_start is None
                if approaching and tick - motion_start >= approach.duration:
                    if np.linalg.norm(pose[:2] - path.initial[:2]) <= 0.005 and abs(
                        math.atan2(math.sin(pose[2] - path.initial[2]), math.cos(pose[2] - path.initial[2]))
                    ) <= math.radians(1):
                        circle_start = tick
                        approaching = False
                        pid.reset()
                        admitted = None
                        print("Start pose reached; starting circle.", flush=True)
                    elif tick - motion_start > approach.duration + 15:
                        raise TimeoutError("Start-pose approach did not settle within 15 seconds")
                if motion_start is not None and approach is None and circle_start is None:
                    circle_start = motion_start
                motion_t = -1.0 if circle_start is None else tick - circle_start
                if args.continuous and circle_start is not None:
                    new_pass, motion_t = repeat_phase(path, motion_t)
                    if new_pass != pass_index:
                        pass_index = new_pass
                        print(
                            f"Starting pass {pass_index + 1}; logging={'on' if recording else 'off'}",
                            flush=True,
                        )
                target, feedforward = path.reference(motion_t)
                if approaching:
                    approach_q = qt.clone()
                    approach_joint_target = approach.reference(tick - motion_start)
                    approach_q[:, :3] = approach_joint_target
                    target = bundle.robot_kin.pose_jacobian(approach_q)[0][0].numpy()
                error = float(np.linalg.norm(pose[:2] - target[:2]))
                angle_error = math.atan2(math.sin(target[2] - pose[2]), math.cos(target[2] - pose[2]))
                if circle_start is not None and (
                    error * 1000 > args.max_error_mm or abs(angle_error) > math.radians(10)
                ):
                    raise RuntimeError("Circle tracking error exceeds the configured position/angle envelope")
                if motion_start is not None and (
                    approaching or args.controller != "mlp" or step % (100 // args.policy_hz) == 0
                ):
                    before = time.monotonic()
                    with torch.no_grad():
                        if approaching:
                            requested = approach_pid.joint_valves(qt, approach_joint_target, dt)[0].numpy()
                        elif args.controller == "mlp":
                            # Keep the expensive training-time projection at 20 Hz;
                            # actor/history can consume fresh measurements at 100 Hz.
                            if admitted is None or step % 5 == 0:
                                physical = torch.tensor(
                                    path.command(motion_t, pose)[None], dtype=torch.float32
                                )
                                desired = policy_twist(
                                    qt, physical, bundle.robot_kin, bundle.kin, policy_q=policy_q
                                )
                                admitted, _ = bundle.governor(policy_q, vt, desired)
                            requested = bundle.valves(bundle.history.observe(bundle.kin, admitted))[0].numpy()
                        else:
                            requested = pid.valves(qt, torch.tensor(target[None], dtype=torch.float32), dt)[
                                0
                            ].numpy()
                    compute_ms = (time.monotonic() - before) * 1000
                    if compute_ms > 40 and not args.allow_timing_overruns:
                        raise RuntimeError("Controller computation exceeded 40 ms")
                    with gate.lock:
                        gate.policy_time = time.monotonic()
                        if not gate.armed:
                            gate.arm()
                            if not hardware.set_pump_enabled(True):
                                raise RuntimeError("Hardware rejected pump enable")
                    if args.valve_writes == "thread":
                        controller.give_direct_commands(dict(zip(JOINTS, requested.tolist(), strict=True)))
                if args.valve_writes == "loop":
                    # Every tick, held between actor updates; the gate zeroes it unless armed.
                    direct.give_commands(dict(zip(JOINTS, requested.tolist(), strict=True)))
                    direct.send_pending()
                if recording:
                    with gate.lock:
                        written, written_at = gate.last_command.copy(), gate.write_time
                    stage = (
                        "wait"
                        if motion_start is None
                        else "approach"
                        if approaching
                        else "circle"
                        if gate.armed
                        else "stopped"
                    )
                    radial = float(np.linalg.norm(pose[:2] - path.center) - path.radius)
                    vels, vel_age = controller.get_joint_velocities_with_age()
                    row = log.build_row(
                        tick - started - logging_started_at if args.continuous else tick - started,
                        {},
                        {},
                        dict(zip(JOINTS, written.tolist(), strict=True)),
                        *controller.get_joint_angles(),
                        vels,
                        vel_age,
                        hardware.try_read_imu_gyro(),
                        # Command age is that of the last valve write; unarmed rows hold neutral valves
                        # with the pump off and are marked stale so training cuts them out.
                        float("inf") if written_at is None else time.monotonic() - written_at,
                        not gate.armed,
                        gate.armed,
                        "all",
                        -1,
                        {
                            "mode": f"circle_{args.controller}",
                            "version": 1,
                            "block_id": pass_index,
                            "elapsed_s": motion_t,
                            "stage": stage,
                        },
                    )
                    row.update(
                        zip(
                            CIRCLE_FIELDS,
                            (
                                tick - started,
                                motion_t,
                                int(gate.armed),
                                pass_index,
                                stamp,
                                *q.tolist(),
                                *v.tolist(),
                                *pose.tolist(),
                                *target.tolist(),
                                *feedforward.tolist(),
                                error,
                                radial,
                                angle_error,
                                *requested.tolist(),
                                compute_ms,
                                lateness * 1000,
                            ),
                            strict=True,
                        )
                    )
                    log.append(row)
                    log.log_imu_raw(hardware.drain_imu_raw_capture(), len(imu_roles))
                else:
                    hardware.drain_imu_raw_capture()
                if step % 100 == 0:
                    print(
                        f"t={elapsed:.1f}s error={error * 1000:.1f}mm compute={compute_ms:.1f}ms "
                        f"{'ARMED' if gate.armed else 'waiting for LB'}",
                        flush=True,
                    )
                if not args.continuous and motion_start is not None and motion_t >= path.duration:
                    break
                if (
                    args.continuous
                    and logging_started_at is not None
                    and elapsed - logging_started_at >= args.record_seconds
                ):
                    recording_complete = True
                    break
                if motion_start is None and elapsed > 120:
                    raise TimeoutError("Operator enable was not received within 120 seconds")
                step += 1
                next_tick += 0.01
                time.sleep(max(0.0, next_tick - time.monotonic()))
        except BaseException as exc:
            fault = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Always save the trace, including failures, after output shutdown.
            try:
                if gate is not None:
                    gate.stop(fault or "run completed")
            finally:
                monitor_stop.set()
                if monitor is not None:
                    monitor.join(timeout=1)
                try:
                    if controller is not None:
                        controller.stop()
                finally:
                    try:
                        if pad is not None:
                            pad.stop_monitoring()
                    finally:
                        try:
                            if hardware is not None:
                                try:
                                    if recording:
                                        log.log_imu_raw(hardware.drain_imu_raw_capture(), len(imu_roles))
                                    stream_info = hardware.imu_stream_info()
                                except Exception as exc:  # keep the run's own fault; save what exists
                                    metadata["imu_raw_error"] = f"{type(exc).__name__}: {exc}"
                                finally:
                                    hardware.shutdown()
                        finally:
                            print(f"Pump off. Saving {log.n_samples()} samples...", flush=True)
                            if log.n_samples():
                                log.write_drive_log(log_path)
                                log.write_imu_raw(raw_path, imu_roles, stream_info)
                            dropped = stream_info.get("capture_dropped", 0) or 0
                            metadata["imu_raw_dropped_frames"] = int(dropped)
                            metadata["result"] = (
                                summarize(list(scoring_rows(log_path)), path, fault)
                                if path is not None and log.n_samples()
                                else {
                                    "completed": False,
                                    "fault": fault,
                                }
                            )
                            if args.continuous:
                                score_recorded_passes(log_path, path, metadata["passes"], fault)
                                metadata["logging_started_at_s"] = logging_started_at
                                metadata["result"] = {
                                    "completed": recording_complete,
                                    "stopped_by": "A"
                                    if stopped_by_a
                                    else "recording_duration"
                                    if recording_complete
                                    else "fault",
                                    "fault": fault,
                                    "passes_completed": pass_index,
                                    "logged_passes": len(metadata["passes"]),
                                }
                            json.dump(metadata, summary_stream, indent=2)
                            summary_stream.write("\n")
                            print(f"Saved {summary_path}", flush=True)


def run_paths(args):
    """drive_log/imu_raw names in simple_drive.py's pattern, labelled as a circle run."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tail = f"{stamp}_circle_{args.controller}_{args.direction}{clean_suffix(args.label)}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args.out_dir / f"drive_log_{tail}.csv", args.out_dir / f"imu_raw_{tail}.csv"


def scoring_rows(csv_path):
    """Yield the numeric fields summarize() needs, one drive-log row at a time."""
    numeric = ("motion_t_s", "error_m", "radial_error_m", "angle_error_rad", "compute_ms", *VALVE_COLUMNS)
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            yield {
                **{key: float(row[key]) for key in numeric},
                "armed": float(row["armed"]) > 0,
                "pass_index": int(row["pass_index"]),
            }


def score_recorded_passes(csv_path, path, reports, fault=None):
    """Score a CSV one pass at a time after hardware stops, keeping memory bounded."""

    if path is None or not csv_path.exists():
        return
    for index, group in itertools.groupby(scoring_rows(csv_path), lambda row: row["pass_index"]):
        rows = list(group)
        partial = rows[0]["motion_t_s"] > 0.02 or rows[-1]["motion_t_s"] < path.duration - 0.03
        reports.append(
            {"pass_index": index, "partial": partial, **summarize(rows, path, fault if partial else None)}
        )


def compare_runs(args) -> None:
    """Compare measured runs, retaining stopped trials alongside completed trials."""
    reports = [json.loads(p.with_suffix(".json").read_text()) for p in args.logs]
    first = reports[0]
    for report in reports[1:]:
        for key in ("radius_mm", "speed_mm_s", "cycles", "profile_sha256"):
            if report[key] != first[key]:
                raise ValueError(
                    f"Runs use different {key}; compare matching trajectory/calibration profiles"
                )
    records = [
        {
            "log": str(path),
            "controller": report["controller"],
            "direction": report["direction"],
            **report["result"],
        }
        for path, report in zip(args.logs, reports, strict=True)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in records for key in row))
    with args.out.with_suffix(".csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    with args.out.with_suffix(".json").open("x") as stream:
        json.dump({"runs": records}, stream, indent=2)
        stream.write("\n")
    for row in records:
        print(
            f"{row['controller']:10s} {row['direction']} "
            f"RMSE={row.get('tracking_rmse_mm', float('nan')):.2f}mm "
            f"radial max={row.get('radial_max_abs_mm', float('nan')):.2f}mm "
            f"{'completed' if row['completed'] else row['fault']}"
        )
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path = args.out.with_suffix(".png")
        if plot_path.exists():
            raise FileExistsError(plot_path)
        figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        angle = np.linspace(0, 2 * math.pi, 200)
        radius = first["radius_mm"]
        for column, direction in enumerate(("ccw", "cw")):
            axes[0, column].plot(radius * np.cos(angle), radius * np.sin(angle), "k--", label="reference")
            axes[0, column].set(
                title=direction, xlabel="X from center [mm]", ylabel="Z from center [mm]", aspect="equal"
            )
            axes[1, column].set(xlabel="Motion time [s]", ylabel="Timed tracking error [mm]")
        for path, report in zip(args.logs, reports, strict=True):
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            moving = [r for r in rows if float(r["armed"]) > 0 and float(r["motion_t_s"]) >= 1]
            if not rows or not moving:
                continue
            center = np.array([float(rows[0]["ref_x_m"]) - radius / 1000, float(rows[0]["ref_z_m"])])
            xy = np.array([[float(r["tip_x_m"]), float(r["tip_z_m"])] for r in moving])
            column = 0 if report["direction"] == "ccw" else 1
            label = f"{report['controller']} {path.stem}" + (
                " (stopped)" if not report["result"]["completed"] else ""
            )
            axes[0, column].plot(*(1000 * (xy - center)).T, label=label, alpha=0.8)
            axes[1, column].plot(
                [float(r["motion_t_s"]) for r in moving],
                [1000 * float(r["error_m"]) for r in moving],
                alpha=0.8,
            )
        for axis in axes.flat:
            axis.grid(alpha=0.3)
        for axis in axes[0]:
            axis.legend(fontsize=7)
        figure.suptitle(
            f"Real bucket circles: diameter {2 * radius:g} mm, speed {first['speed_mm_s']:g} mm/s"
        )
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
    print(f"Saved comparison under {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    comparison = commands.add_parser("compare", help="Compare recorded CSV/JSON runs; opens no hardware")
    comparison.add_argument("--logs", type=Path, nargs="+", required=True, help="Circle drive_log_ CSVs")
    comparison.add_argument(
        "--out", type=Path, required=True, help="New output prefix for CSV/JSON and optional PNG"
    )
    comparison.add_argument("--plot", action="store_true", help="Also save a figure; requires matplotlib")
    for mode in ("check", "run"):
        cmd = commands.add_parser(
            mode, help="Offline check (no devices)" if mode == "check" else "Operator-enabled motion"
        )
        cmd.add_argument("--bundle", type=Path, required=True, help="Exported bundle, e.g. learned_control/bundles/...")
        cmd.add_argument("--robot", default="jetson_bucket", help="Profile the bundle was exported for")
        cmd.add_argument("--controller", choices=("pid_robot", "pid_tuned", "mlp"), default="mlp")
        cmd.add_argument("--pid_gains", type=Path)
        cmd.add_argument("--direction", choices=("cw", "ccw"), default="ccw")
        cmd.add_argument("--radius_mm", type=float, default=50)
        cmd.add_argument("--speed_mm_s", type=float, default=20)
        cmd.add_argument("--cycles", type=int, default=1)
        cmd.add_argument(
            "--joint_margin_deg",
            type=float,
            default=math.degrees(0.06),
            help="Extra margin within physical joint limits; use 0 for the cleared hardware area",
        )
        cmd.add_argument(
            "--start_from_log",
            type=Path,
            help="Approach the fixed circle start from an earlier CSV before starting passes",
        )
        cmd.add_argument(
            "--policy_hz",
            type=int,
            choices=(20, 100),
            default=20,
            help="MLP actor update rate; history/PWM 100 Hz, projection 20 Hz",
        )
        if mode == "run":
            cmd.add_argument(
                "--approach_output_limit",
                type=float,
                default=0.25,
                help="Symmetric normalized valve cap for the start-pose PID approach",
            )
            cmd.add_argument(
                "--continuous",
                action="store_true",
                help="LB starts autonomous repeated passes; A stops; B starts logging",
            )
            cmd.add_argument("--max_error_mm", type=float, default=20)
            cmd.add_argument(
                "--valve_writes",
                choices=("loop", "thread"),
                default="loop",
                help="loop: this 100 Hz loop writes the valves in the tick it computes them, as the "
                "data recorder did; thread: the robot controller's direct-command thread writes them "
                "(~20 ms later, the 2026-10-07 behaviour)",
            )
            cmd.add_argument(
                "--allow_timing_overruns",
                action="store_true",
                help="Diagnostic: log/reschedule missed deadlines; independent freshness gate still stops stalls",
            )
            cmd.add_argument(
                "--record_seconds",
                type=float,
                default=300,
                help="Recording length after B; pump off then save all passes (max 600s)",
            )
            cmd.add_argument(
                "--max_carriage_pitch_deg",
                type=float,
                default=3,
                help="Passive rocking bound in degrees; model geometry supports +/-3",
            )
            cmd.add_argument(
                "--max_joint_velocity_rad_s",
                type=float,
                default=2,
                help="Boom/arm/bucket measured-rate bound; excludes passive carriage rate",
            )
            cmd.add_argument(
                "--out_dir",
                type=Path,
                default=ROOT / "data_collection/circle_logs",
                help="Writes drive_log_<time>_circle_<controller>_<direction>[_label].csv, its imu_raw_ "
                "strip and a .json summary",
            )
            cmd.add_argument("--label", help="Optional filename tail, e.g. v6_20hz")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    {"check": check, "run": run, "compare": compare_runs}[args.mode](args)


if __name__ == "__main__":
    main()
