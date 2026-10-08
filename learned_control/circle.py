"""The timed bucket circle, its start-pose approach, and the robot's own joint PID behind a DLS step."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from learned_control.settings import wrap_angle
from modules.pid import PIDController


@dataclass
class PidGains:
    """Per-joint (boom, arm, bucket) gains in robot units: valve fraction per rad, per rad*s, per rad/s."""

    kp: list[float]
    ki: list[float]
    kd: list[float]
    deriv_filter_tau: list[float] = field(default_factory=lambda: [0.10] * 3)
    output_limits: tuple[float, float] = (-1.0, 1.0)
    ik_lambda: float = 0.001


def load_robot_gains(control_yaml: Path) -> PidGains:
    """Read boom/arm/bucket gains, output limits and the DLS lambda from a robot ``control_config.yaml``."""
    import yaml

    cfg = yaml.safe_load(Path(control_yaml).read_text())
    pid = [cfg["pid"][f"joint{i}"] for i in (1, 2, 3)]
    ctrl = cfg.get("controller", {})
    ik = cfg.get("ik", {})
    if ik.get("method", "dls") != "dls":
        raise ValueError(f"the circle PID models the dls IK method, the robot is set to {ik['method']!r}")
    return PidGains(
        kp=[float(j["kp"]) for j in pid],
        ki=[float(j["ki"]) for j in pid],
        kd=[float(j["kd"]) for j in pid],
        output_limits=(float(ctrl.get("output_limits_min", -1.0)), float(ctrl.get("output_limits_max", 1.0))),
        ik_lambda=float(ik.get("params", {}).get("lambda_val", 0.001)),
    )


def dls_step(kin, q: torch.Tensor, tip_target: torch.Tensor, lam: float) -> torch.Tensor:
    """Robot ``_ik_dls`` on the planar (x, z, pitch) task over boom/arm/bucket [rad]."""
    pose, jacobian = kin.pose_jacobian(q)
    j = jacobian[:, :, :3]
    error = tip_target - pose
    error[:, 2] = wrap_angle(error[:, 2])
    a = j @ j.transpose(1, 2) + lam * lam * torch.eye(3, device=q.device)
    return (j.transpose(1, 2) @ torch.linalg.solve(a, error[:, :, None])).squeeze(-1)


class CircleTrajectory:
    """X/Z circles [m], constant cutting-lip angle [rad], trapezoidal arc speed [m/s]."""

    def __init__(
        self,
        initial_pose,
        radius_m=0.05,
        speed_m_s=0.02,
        direction="ccw",
        cycles=1,
        accel_m_s2=0.5,
        lead_s=1.0,
        tail_s=2.0,
    ):
        self.initial = np.asarray(initial_pose, dtype=np.float64).copy()
        if self.initial.shape != (3,) or not np.isfinite(self.initial).all():
            raise ValueError("Expected finite [x, z, blade angle] in meters/radians")
        values = [radius_m, speed_m_s, accel_m_s2, lead_s, tail_s, cycles]
        if not np.isfinite(values).all() or min(values[:3]) <= 0 or min(values[3:5]) < 0:
            raise ValueError("Circle dimensions, timing and speed must be finite and positive")
        if direction not in ("cw", "ccw") or isinstance(cycles, bool) or int(cycles) != cycles or cycles < 1:
            raise ValueError("Use cw/ccw and a positive integer cycle count")
        self.radius, self.speed, self.accel = radius_m, speed_m_s, accel_m_s2
        self.direction, self.cycles = direction, int(cycles)
        self.lead, self.tail = lead_s, tail_s
        self.center = self.initial[:2] - np.array([radius_m, 0.0])
        self.distance = 2 * math.pi * radius_m
        self.ramp_s = min(speed_m_s / accel_m_s2, math.sqrt(self.distance / accel_m_s2))
        self.peak_speed = self.ramp_s * accel_m_s2
        self.cruise_s = max(0.0, (self.distance - self.peak_speed * self.ramp_s) / self.peak_speed)
        self.lap_s = 2 * self.ramp_s + self.cruise_s
        self.move_end = self.lead + self.cycles * self.lap_s
        self.duration = self.move_end + self.tail

    def reference(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return timed tip pose [m, m, rad] and feedforward twist [m/s, m/s, rad/s]."""
        if not math.isfinite(t):
            raise ValueError("Reference time must be finite")
        if t <= self.lead or t >= self.move_end:
            return self.initial.copy(), np.zeros(3)
        phase = (t - self.lead) % self.lap_s
        if phase < self.ramp_s:
            arc, speed = 0.5 * self.accel * phase**2, self.accel * phase
        elif phase < self.ramp_s + self.cruise_s:
            arc = 0.5 * self.peak_speed * self.ramp_s + self.peak_speed * (phase - self.ramp_s)
            speed = self.peak_speed
        else:
            decel_t = phase - self.ramp_s - self.cruise_s
            arc = self.distance - 0.5 * self.accel * (self.ramp_s - decel_t) ** 2
            speed = max(0.0, self.peak_speed - self.accel * decel_t)
        sense = 1 if self.direction == "ccw" else -1
        angle = sense * arc / self.radius
        pose = np.r_[
            self.center + self.radius * np.array([math.cos(angle), math.sin(angle)]), self.initial[2]
        ]
        twist = np.array([-math.sin(angle), math.cos(angle), 0.0]) * sense * speed
        return pose, twist

    def command(self, t: float, pose: np.ndarray, kp=3.0) -> np.ndarray:
        """Feedforward plus bounded position feedback [m/s, m/s, rad/s] for the MLP."""
        target, feedforward = self.reference(t)
        error = target - pose
        error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))
        command = feedforward + kp * error
        command[:2] *= min(1.0, 0.06 / max(1e-12, float(np.linalg.norm(command[:2]))))
        command[2] = np.clip(command[2], -0.3, 0.3)
        return command

    def validate(self, kin, q: torch.Tensor) -> None:
        """Check an entire circle from measured joint angles [rad], before enabling the pump."""
        angle = torch.linspace(0, 2 * math.pi, 129, device=q.device)
        targets = torch.tensor(self.initial, device=q.device, dtype=q.dtype).repeat(len(angle), 1)
        targets[:, 0] += self.radius * (angle.cos() - 1)
        targets[:, 1] += self.radius * angle.sin()
        solved, reached = kin.inverse(targets, q[:1])
        if not bool(reached.all()):
            raise ValueError("Circle is outside joint/collision margins at this starting pose")
        _, jac = kin.pose_jacobian(solved)
        weighted = jac[:, :, :3] * q.new_tensor([1.0, 1.0, 0.2])[None, :, None]
        if not torch.isfinite(weighted).all() or bool((torch.linalg.cond(weighted) > 100).any()):
            raise ValueError("Circle passes too close to a kinematic singularity")


class CircleJointPID:
    """Measured joint angles -> DLS target -> the robot's ``PIDController`` per joint -> normalized valves.

    Each PID is called as ``ExcavatorController`` calls it, ``compute(0, -wrap(target - q))``, so the gains in
    ``control_config.yaml`` (or a sim-tuned file) mean the same here as in the robot's IK loop.
    """

    def __init__(self, gains: PidGains, kin):
        self.gains, self.kin = gains, kin
        lo, hi = gains.output_limits
        self.pids = [
            PIDController(kp=kp, ki=ki, kd=kd, min_output=lo, max_output=hi, deriv_filter_tau=tau)
            for kp, ki, kd, tau in zip(gains.kp, gains.ki, gains.kd, gains.deriv_filter_tau, strict=True)
        ]

    def reset(self) -> None:
        """Clear integral and derivative state before motion."""
        for pid in self.pids:
            pid.reset()

    def valves(self, q: torch.Tensor, target: torch.Tensor, dt: float) -> torch.Tensor:
        """Return boom/arm/bucket valves [-1, 1] from measured q [rad] and tip target [m, m, rad]."""
        next_q = q[:, :3] + dls_step(self.kin, q, target, self.gains.ik_lambda)
        return self.joint_valves(q, next_q, dt)

    def joint_valves(self, q: torch.Tensor, target: torch.Tensor, dt: float) -> torch.Tensor:
        """PID toward joint targets [rad], also used for the smooth start-pose approach."""
        error = wrap_angle(target - q[:, :3])[0].tolist()
        out = [pid.compute(0.0, -e, dt) for pid, e in zip(self.pids, error, strict=True)]
        return torch.tensor([out], dtype=torch.float32)


class StartMove:
    """Quintic joint interpolation with zero endpoint speed, bounded to 0.05 rad/s."""

    def __init__(self, initial, target, kin):
        self.initial = initial[:1, :3].clone()
        self.target = target[:1, :3].clone()
        self.duration = max(3.0, 1.875 * float((self.target - self.initial).abs().max()) / 0.05)
        samples = initial[:1].repeat(129, 1)
        samples[:, :3] = self.initial + torch.linspace(0, 1, 129)[:, None] * (self.target - self.initial)
        if not bool(kin.valid(samples).all()):
            raise ValueError("Start-pose approach crosses a physical joint/collision bound")

    def reference(self, elapsed):
        phase = min(1.0, max(0.0, elapsed / self.duration))
        blend = phase**3 * (10 - 15 * phase + 6 * phase**2)
        return self.initial + blend * (self.target - self.initial)
