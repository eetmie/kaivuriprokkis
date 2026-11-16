"""Drive boom, arm and bucket with the trained PPO valve controller, through kaivuri-control.

`check` loads the bundle (which replays its training reference cases) and times one policy step;
it opens no devices. `run` follows the controller's 100 Hz status, keeps the policy's measured
histories and sends one valve row per tick; the policy itself runs at its trained 20 Hz. Slew and
tracks stay neutral.

    python3 -B tools/ppo_live.py check
    python3 -B tools/ppo_live.py run --start-control --live                   # circle
    python3 -B tools/ppo_live.py run --start-control --live --mode gamepad    # relative edge control

--mode circle: an X/Z circle of the bucket cutting edge from the current pose, bucket angle held.
--mode gamepad: the left stick moves the edge target in X (right = away from the cabin) and Z (up);
the triggers turn the bucket (right = curl in, left = open). Sticks released holds the pose.

A on the pad arms and starts (pump permission on); A or B, Ctrl+C, a lost pad or any limit below
stops and disarms. Without a pad (--gamepad none) a circle starts by itself. Without --start-control,
start `kaivuri-control` first; without --live it writes no PWM and the machine will not move.

Stops: tracking error (--max-error-mm, 10 degrees), a joint within half the governor margin of its
limit or the self-collision grid, joint rate, carriage pitch, missed ticks, and every controller
fault (IMU, lease, deadline). Free-space motion only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

import client
from ppo_policy import Bundle, joint_state, wrap
from vla_operator import ABS_RZ, ABS_X, ABS_Y, ABS_Z, BTN_A, BTN_B, Pad, find_pad

NATIVE = Path(__file__).resolve().parents[1]
KP = 3.0             # position loop outside the policy [1/s], as in training playback and the RA-L paper
LEASH = (0.015, 0.1)  # gamepad target ahead of the measured edge at most this far [m, rad]
FIELDS = ("t_s", "tick", *(f"{n}_{j}" for n in ("q", "v") for j in ("boom", "arm", "bucket", "pitch")),
          *(f"{n}_{j}" for n in ("pose", "target", "feedforward", "admitted") for j in ("x", "z", "angle")),
          *(f"{n}_{j}" for n in ("valve", "applied") for j in ("boom", "arm", "bucket")),
          "error_mm", "angle_error_deg", "compute_ms")


class Stop(Exception):
    """The run ends: disarm and report why."""


class Circle:
    """X/Z circle of the cutting edge starting at its rightmost point, angle held, trapezoidal speed."""

    def __init__(self, start, radius, speed, direction, cycles, accel=0.5, lead=1.0, tail=2.0):
        self.start, self.radius, self.accel, self.lead = np.array(start, float), radius, accel, lead
        self.sense = 1 if direction == "ccw" else -1
        self.arc = 2 * math.pi * radius * cycles
        self.ramp = min(speed / accel, math.sqrt(self.arc / accel))
        self.peak = self.ramp * accel
        self.cruise = (self.arc - self.peak * self.ramp) / self.peak
        self.move_end = lead + 2 * self.ramp + self.cruise
        self.duration = self.move_end + tail

    def reference(self, t):
        """Edge pose [m, m, rad] and feedforward twist [m/s, m/s, rad/s] at run time t."""
        phase = min(max(t - self.lead, 0.0), self.move_end - self.lead)
        if phase < self.ramp:
            arc, speed = 0.5 * self.accel * phase**2, self.accel * phase
        elif phase < self.ramp + self.cruise:
            arc, speed = self.peak * (phase - 0.5 * self.ramp), self.peak
        else:
            left = 2 * self.ramp + self.cruise - phase
            arc, speed = self.arc - 0.5 * self.accel * left**2, self.accel * left
        angle = self.sense * arc / self.radius
        pose = self.start + [self.radius * (math.cos(angle) - 1), self.radius * math.sin(angle), 0]
        return pose, np.array([-math.sin(angle), math.cos(angle), 0]) * self.sense * speed

    def check(self, bundle, q):
        """The whole circle is reachable from q inside the governor's margins, away from singularities."""
        angle = np.linspace(0, 2 * math.pi, 129)
        targets = self.start + np.c_[self.radius * (np.cos(angle) - 1), self.radius * np.sin(angle), 0 * angle]
        solved, reached = bundle.kin.inverse(targets, q, bundle.settings["joint_margin"])
        if not reached.all():
            raise Stop("circle leaves the joint/collision margins from this pose; move the arm or shrink it")
        jac = bundle.kin.pose_jacobian(solved)[1][:, :, :3] * np.array([1, 1, 0.2])[:, None]
        if np.linalg.cond(jac).max() > 100:
            raise Stop("circle passes too close to a kinematic singularity")


def stick_reference(pad, target, pose, a, dt):
    """Advance the edge target by the sticks; returns their twist as feedforward."""
    twist = np.array([a.speed_mm_s / 1e3 * pad.axis(ABS_X), -a.speed_mm_s / 1e3 * pad.axis(ABS_Y),
                      a.pitch_rate * (pad.axis(ABS_RZ) - pad.axis(ABS_Z))])
    target += twist * dt
    # A target the machine cannot follow (limits, slow valves) must not run away from it.
    offset = target[:2] - pose[:2]
    target[:2] = pose[:2] + offset * min(1.0, LEASH[0] / max(np.linalg.norm(offset), 1e-9))
    target[2] = pose[2] + np.clip(wrap(target[2] - pose[2]), -LEASH[1], LEASH[1])
    return twist


def summarize(rows, outcome, missed):
    summary = {"outcome": outcome, "ticks": len(rows), "missed_ticks": missed}
    if len(rows) > 1:
        r = np.array(rows)
        col = FIELDS.index
        valves = r[:, col("valve_boom"):col("valve_boom") + 3]
        summary.update(
            tracking_rmse_mm=round(float(np.sqrt(np.mean(r[:, col("error_mm")] ** 2))), 2),
            tracking_max_mm=round(float(r[:, col("error_mm")].max()), 2),
            angle_max_deg=round(float(np.abs(r[:, col("angle_error_deg")]).max()), 2),
            valve_travel_per_s=round(float(np.abs(np.diff(valves, axis=0)).sum() / (r[-1, 0] - r[0, 0])), 2),
            compute_max_ms=round(float(r[:, col("compute_ms")].max()), 2))
    return summary


def run(a, control=None, pad=None):
    """One operator-started run; returns its summary. `control`/`pad` are injected by the test."""
    bundle = Bundle(a.bundle)
    cfg, dec, margin = bundle.settings, bundle.settings["decimation"], bundle.settings["joint_margin"] / 2
    dt = cfg["dt"] * dec
    procs, rows, missed, outcome = [], [], 0, "error"
    try:
        if a.start_control:
            Path(a.socket).parent.mkdir(mode=0o700, exist_ok=True)
            procs.append(subprocess.Popen(
                [str(Path(a.build) / "kaivuri-control"), "--profile", a.profile, "--safety", a.safety, "--socket",
                 a.socket, "--serial", a.serial, *(["--live-pwm"] if a.live else [])], start_new_session=True))
            deadline = time.monotonic() + 10
            while not Path(a.socket).exists():
                if procs[0].poll() is not None or time.monotonic() > deadline:
                    raise Stop("kaivuri-control did not start")
                time.sleep(0.1)
        control = control or client.Client(a.socket)
        path = find_pad(a.gamepad)
        pad = pad or (Pad(path, sticks=(ABS_X, ABS_Y, ABS_Z, ABS_RZ)) if path else None)
        if pad is None and a.mode == "gamepad":
            raise Stop("--mode gamepad needs a gamepad")
        st = control.status()
        if st["state"] != "disarmed" or st["control_hz"] != round(1 / cfg["dt"]):
            raise Stop(f"controller must be disarmed at {1 / cfg['dt']:.0f} Hz, is {st['state']} at {st['control_hz']} Hz")
        print(f"Controller: {'LIVE PWM' if st['live_pwm'] else 'dry, no PWM'}; filling the policy history.", flush=True)

        phase, last_tick, tick0, seen, announced, row = "wait", None, None, 0, False, None
        valves, target, feedforward, admitted, circle = np.zeros(3), None, np.zeros(3), np.zeros(3), None
        while True:
            # Served on the controller's next tick, so this loop keeps its 100 Hz and phase.
            st = control.status()
            began = time.perf_counter()
            if not st["imu_healthy"]:
                raise Stop("IMUs unhealthy")
            gap = 1 if last_tick is None else st["ticks"] - last_tick
            last_tick = st["ticks"]
            if not 1 <= gap <= 5:
                raise Stop(f"lost {gap - 1} controller ticks")
            missed += (gap - 1) * (phase == "run")
            q, v = joint_state(st["imu_quat_wxyz"], st["imu_gyro_dps"], bundle.sensors)
            for _ in range(gap):   # a missed tick repeats the sample, keeping the history's time base
                bundle.push(q, v, st["applied_action"][1:4])
            seen += gap
            pose = bundle.pose()
            pressed = pad.poll() if pad else []

            if phase == "wait" and seen >= len(bundle.u_history) + 40:
                if not announced:
                    announced = True
                    print(f"Edge at x {pose[0] * 1e3:.0f} mm, z {pose[1] * 1e3:.0f} mm, {math.degrees(pose[2]):.1f} deg. "
                          + ("A starts; A or B stops." if pad else "Starting."), flush=True)
                if BTN_A in pressed or not pad:
                    if not bundle.kin.valid(q[None], margin)[0]:
                        raise Stop("arm is at a joint limit or self-collision margin; move it clear first")
                    target = pose.copy()
                    if a.mode == "circle":
                        circle = Circle(pose, a.radius_mm / 1e3, a.speed_mm_s / 1e3, a.direction, a.cycles)
                        circle.check(bundle, q)
                    phase = "arming"
            elif phase != "wait" and (st["state"] == "fault" or (phase == "run" and st["state"] != "armed")):
                raise Stop(f"controller {st['state']}: {st['reason']}")
            elif phase == "arming" and st["state"] == "armed":
                phase, tick0 = "run", st["ticks"]
                print("Armed; running.", flush=True)

            if phase == "run":
                k = st["ticks"] - tick0
                t = k * cfg["dt"]
                if BTN_A in pressed or BTN_B in pressed:
                    outcome = "stopped by the operator"
                    break
                if not bundle.kin.valid(q[None], margin)[0]:
                    raise Stop("joint limit or self-collision margin reached")
                if np.abs(v[:3]).max() > cfg["velocity_limit"]:
                    raise Stop(f"joint rate {np.abs(v[:3]).max():.2f} rad/s")
                if k // dec != (k - gap) // dec or k == 0:
                    future = None
                    if circle:
                        target, feedforward = circle.reference(t)
                        future = np.array([circle.reference(t + j * dt)[1] for j in range(bundle.steps + 1)])
                    else:
                        feedforward = stick_reference(pad, target, pose, a, dt)
                    error = target - pose
                    error[2] = wrap(error[2])
                    valves, admitted = bundle.act(feedforward + KP * error, future)
                error = np.r_[np.linalg.norm(target[:2] - pose[:2]) * 1e3, math.degrees(wrap(target[2] - pose[2]))]
                if error[0] > a.max_error_mm or abs(error[1]) > 10:
                    raise Stop(f"tracking error {error[0]:.1f} mm, {error[1]:.1f} deg")
                if circle and t >= circle.duration:
                    outcome = "finished"
                    break
                row = (t, st["ticks"], *q, *v, *pose, *target, *feedforward, *admitted, *valves,
                       *st["applied_action"][1:4], *error)
                if k % 100 == 0:
                    print(f"t {t:5.1f} s  error {error[0]:5.1f} mm {error[1]:5.1f} deg  valves "
                          + " ".join(f"{u:+.2f}" for u in valves), flush=True)
            if phase != "wait":
                arm = client.ARM if phase == "arming" else 0
                control.send(client.DRIVE, [(0.0, *valves, 0.0, 0.0)], flags=client.DEADMAN | client.PUMP | arm)
            if phase == "run":
                rows.append((*row, (time.perf_counter() - began) * 1e3))
    except Stop as stop:
        outcome = str(stop)
    except KeyboardInterrupt:
        outcome = "interrupted"
    finally:
        try:
            if control is not None:
                control.send(client.DISARM)
                control.close()
        finally:
            client.stop(procs)
    summary = summarize(rows, outcome, missed)
    if a.out and rows:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=False)
        with open(out / "ticks.csv", "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(rows)
        settings = {k: v for k, v in vars(a).items() if k != "command"}
        (out / "summary.json").write_text(json.dumps(
            {**summary, "settings": settings, "checkpoint_sha256": bundle.contract["checkpoint_sha256"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def check(a):
    bundle = Bundle(a.bundle)
    q = bundle.arrays["ref_q"][0].astype(float)
    times = []
    for i in range(200):
        bundle.push(q + 1e-6 * i, np.zeros(4), np.zeros(3))   # a new measurement each step, as live
        began = time.perf_counter()
        bundle.act([0.02, 0.01, 0.05], np.zeros((bundle.steps + 1, 3)))
        times.append((time.perf_counter() - began) * 1e3)
    c = bundle.contract
    print(f"Bundle verified against its training reference: tracks the {bundle.point}, {bundle.steps} lookahead "
          f"steps, checkpoint {c['checkpoint_sha256'][:12]}.\n"
          f"Policy step p50 {np.median(times):.2f} ms, max {max(times):.2f} ms (budget {c['settings']['dt'] * 1e3:.0f} ms).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        s = sub.add_parser(name)
        s.add_argument("--bundle", default=str(NATIVE / "ppo_bundles/controller_pivot"),
                       help="Isaac-hydraulic-actuator export_robot_bundle.py native output")
    s.add_argument("--mode", choices=("circle", "gamepad"), default="circle")
    s.add_argument("--radius-mm", type=float, default=50)
    s.add_argument("--speed-mm-s", type=float, default=20, help="circle speed; full-stick speed in gamepad mode")
    s.add_argument("--direction", choices=("ccw", "cw"), default="ccw")
    s.add_argument("--cycles", type=int, default=1)
    s.add_argument("--pitch-rate", type=float, default=0.2, help="gamepad: full-trigger bucket rate [rad/s]")
    s.add_argument("--max-error-mm", type=float, default=20)
    s.add_argument("--gamepad", default="auto", help="evdev path, 'auto' (first by-id joystick) or 'none'")
    s.add_argument("--out", help="new directory for ticks.csv and summary.json (default: nothing is written)")
    s.add_argument("--socket", default=client.DEFAULT_SOCKET)
    s.add_argument("--start-control", action="store_true", help="start kaivuri-control here, stop it at the end")
    s.add_argument("--live", action="store_true", help="with --start-control: live PWM")
    s.add_argument("--build", default=str(NATIVE / "build"))
    s.add_argument("--profile", default=str(NATIVE.parent / "configuration_files/profiles/jetson"))
    s.add_argument("--safety", default=str(NATIVE / "config/safety.yaml"))
    s.add_argument("--serial", default="/dev/ttyACM0")
    a = parser.parse_args()
    if a.command == "check":
        return check(a)
    if not (0 < a.radius_mm <= 100 and 0 < a.speed_mm_s <= 60 and 1 <= a.cycles <= 20
            and 0 < a.pitch_rate <= 0.3 and 0 < a.max_error_mm <= 50):
        parser.error("bounds: radius <= 100 mm, speed <= 60 mm/s, 1..20 cycles, pitch rate <= 0.3 rad/s, error <= 50 mm")
    raise SystemExit(0 if run(a)["outcome"] in ("finished", "stopped by the operator") else 1)


if __name__ == "__main__":
    main()
