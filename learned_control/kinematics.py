"""Planar excavator kinematics from an exported geometry contract, and the training command governor.

Inference-only copy of ``hydraulic_controller.kinematics``: geometry comes from a bundle's ``geometry.json`` and
self-collision from its precomputed grid, so neither USD nor SciPy is needed. Only the bucket tip is tracked.
"""

from __future__ import annotations

import torch

from learned_control.settings import ControllerSettings, wrap_angle


class ExcavatorKinematics:
    """Batched Torch kinematics over the exported joint frames."""

    @classmethod
    def from_dict(cls, geometry: dict, device: str = "cpu"):
        """Load an exported geometry contract [m, rad]."""
        if geometry["version"] != 1:
            raise ValueError("Unsupported exported geometry")
        self = cls.__new__(cls)
        self.device = device
        self.path = None

        def tensor(value):
            return torch.tensor(value, dtype=torch.float32, device=device)

        self.frames = [(tensor(a), tensor(b), int(channel)) for a, b, channel in geometry["frames"]]
        self.links = geometry["links"]
        self.limits = tensor(geometry["limits_rad"])
        self.tip = tensor(geometry["tip"])
        self.polygons = [(name, int(index), tensor(corners)) for name, index, corners in geometry["polygons"]]
        self.collision_pairs = [tuple(pair) for pair in geometry["collision_pairs"]]
        return self

    def _transforms(self, q: torch.Tensor):
        n = len(q)
        transform = torch.eye(4, device=q.device).expand(n, 4, 4).clone()
        transforms = [transform]
        joint_frames = {}
        for parent, child_inverse, channel in self.frames:
            transform = transform @ parent
            if channel >= 0:
                joint_frames[channel] = transform
                angle = q[:, channel]
                c, s = angle.cos(), angle.sin()
                rotation = torch.eye(4, device=q.device).expand(n, 4, 4).clone()
                rotation[:, 0, 0] = c
                rotation[:, 2, 2] = c
                rotation[:, 0, 2] = s
                rotation[:, 2, 0] = -s
                transform = transform @ rotation
            transform = transform @ child_inverse
            transforms.append(transform)
        return transforms, joint_frames

    def pose_jacobian(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tip pose [m, m, rad] and Jacobian with respect to four angles [rad]."""
        transforms, joints = self._transforms(q)
        point = (transforms[-1] @ self.tip)[:, :3, 3]
        blade = transforms[-1][:, :3, 1]  # bucket-local +Y points toward the cutting lip
        angle = torch.atan2(-blade[:, 2], blade[:, 0])
        pose = torch.cat((point[:, ::2], angle[:, None]), dim=1)
        columns = []
        for i in range(4):
            frame = joints[i]
            axis = frame[:, :3, 1]
            linear = torch.linalg.cross(axis, point - frame[:, :3, 3])[:, ::2]
            columns.append(torch.cat((linear, axis[:, 1:2]), dim=1))
        return pose, torch.stack(columns, dim=-1)

    def colliding(self, q: torch.Tensor) -> torch.Tensor:
        """Conservatively flag nonadjacent collider intersections in the arm plane."""
        grid = getattr(self, "_grid", None)
        if grid is None:
            return self._colliding_exact(q)
        resolution = grid.shape[0]
        lo, hi = self.limits[:3, 0], self.limits[:3, 1]
        index = ((q[:, :3] - lo) / (hi - lo) * (resolution - 1)).round().long().clamp(0, resolution - 1)
        return grid[index[:, 0], index[:, 1], index[:, 2]]

    def _colliding_exact(self, q: torch.Tensor) -> torch.Tensor:
        transforms, _ = self._transforms(q)
        polygons = [
            (corners @ transforms[idx].transpose(1, 2))[:, :, [0, 2]] for _, idx, corners in self.polygons
        ]
        collided = torch.zeros(len(q), dtype=torch.bool, device=q.device)
        for i, j in self.collision_pairs:
            a, b = polygons[i], polygons[j]
            edges = torch.cat((torch.roll(a, 1, 1) - a, torch.roll(b, 1, 1) - b), dim=1)
            axes = torch.stack((-edges[:, :, 1], edges[:, :, 0]), dim=-1)
            axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            pa, pb = a @ axes.transpose(1, 2), b @ axes.transpose(1, 2)
            separated = (pa.amax(1) <= pb.amin(1) + 0.001) | (pb.amax(1) <= pa.amin(1) + 0.001)
            collided |= ~separated.any(dim=1)
        return collided

    def valid(self, q: torch.Tensor, margin: float | None = None) -> torch.Tensor:
        """Check finite, in-limit, nonintersecting configurations [rad]."""
        # The passive pitch channel has its own measured +/-3 degree bound, not an arm margin.
        margin = getattr(self, "joint_margin", 0.06) if margin is None else margin
        margins = q.new_tensor([margin, margin, margin, 0.0])
        inside = ((q >= self.limits[:, 0] + margins) & (q <= self.limits[:, 1] - margins)).all(1)
        return torch.isfinite(q).all(1) & inside & ~self.colliding(q)

    def inverse(self, targets: torch.Tensor, seed: torch.Tensor, iterations: int = 35):
        """Solve fixed-carriage-pitch IK for poses [m, m, rad], starting at seed [rad]."""
        q = seed.expand(len(targets), 4).clone()
        weights = q.new_tensor([1.0, 1.0, 0.2])
        eye = torch.eye(3, device=q.device)
        for _ in range(iterations):
            pose, jac = self.pose_jacobian(q)
            error = targets - pose
            error[:, 2] = wrap_angle(error[:, 2])
            j = jac[:, :, :3] * weights[None, :, None]
            dq = torch.linalg.solve(
                j.transpose(1, 2) @ j + 1e-6 * eye, (j.transpose(1, 2) @ (error * weights)[:, :, None])
            ).squeeze(-1)
            q[:, :3] += dq.clamp(-0.2, 0.2)
            q[:, :3] = q[:, :3].clamp(self.limits[:3, 0], self.limits[:3, 1])
        pose, _ = self.pose_jacobian(q)
        reached = (pose[:, :2] - targets[:, :2]).norm(dim=1) < 0.0005
        reached &= wrap_angle(pose[:, 2] - targets[:, 2]).abs() < 0.005
        return q, reached & self.valid(q)


class CommandGovernor:
    """Project requested tip velocities into a conservative local reachable set.

    With ``joint_speed_limits`` [rad/s], shape [3, 2] = (negative, positive), the whole twist is also scaled down so
    no joint is asked for more than ``speed_margin`` of its full-valve speed; the direction of motion is preserved.
    """

    def __init__(
        self,
        kinematics: ExcavatorKinematics,
        settings: ControllerSettings,
        joint_speed_limits: torch.Tensor | None = None,
        speed_margin: float = 0.8,
    ):
        self.kin = kinematics
        self.cfg = settings
        self.joint_speed_limits = joint_speed_limits
        self.speed_margin = speed_margin

    def __call__(self, q: torch.Tensor, v: torch.Tensor, requested: torch.Tensor):
        """Return admitted [m/s, m/s, rad/s] commands and intervention fractions."""
        cfg = self.cfg
        command = torch.nan_to_num(requested).clone()
        speed = command[:, :2].norm(dim=1, keepdim=True)
        command[:, :2] *= (cfg.speed_max / speed.clamp_min(1e-8)).clamp_max(1)
        command[:, 2].clamp_(-cfg.pitch_rate_max, cfg.pitch_rate_max)
        _, jac = self.kin.pose_jacobian(q)
        weights = q.new_tensor([1.0, 1.0, 0.2])
        j = jac[:, :, :3] * weights[None, :, None]
        # Commands describe the arm-driven twist; passive carriage rocking is neither commanded nor tracked.
        rhs = command * weights
        dq = torch.linalg.solve(
            j.transpose(1, 2) @ j + 1e-5 * torch.eye(3, device=q.device), j.transpose(1, 2) @ rhs[:, :, None]
        ).squeeze(-1)
        if self.joint_speed_limits is not None:
            limits = self.joint_speed_limits.to(q.device)
            allowed = torch.where(dq < 0, limits[:, 0], limits[:, 1]) * self.speed_margin
            dq = dq * (allowed / dq.abs().clamp_min(1e-9)).amin(dim=1, keepdim=True).clamp_max(1)
        predicted_q = q[:, :3] + 0.25 * v[:, :3]
        low = ((self.kin.limits[:3, 0] + cfg.joint_margin - predicted_q) / cfg.lookahead).clamp_max(0)
        high = ((self.kin.limits[:3, 1] - cfg.joint_margin - predicted_q) / cfg.lookahead).clamp_min(0)
        dq = dq.clamp(low, high)
        # Backtrack proposed configurations against conservative collision geometry.
        scale = torch.ones(len(q), device=q.device)
        for _ in range(5):
            proposed = q.clone()
            proposed[:, :3] += cfg.lookahead * dq * scale[:, None]
            safe = self.kin.valid(proposed)
            scale = torch.where(safe, scale, scale * 0.5)
        proposed = q.clone()
        proposed[:, :3] += cfg.lookahead * dq * scale[:, None]
        scale = torch.where(self.kin.valid(proposed), scale, 0.0)
        admitted = torch.einsum("nij,nj->ni", jac[:, :, :3], dq * scale[:, None])
        # Never increase the requested translational magnitude during projection.
        admitted[:, :2] *= (speed / admitted[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
        admitted[:, 2].clamp_(-cfg.pitch_rate_max, cfg.pitch_rate_max)
        intervention = ((admitted - command) * weights).norm(dim=1) / (command * weights).norm(
            dim=1
        ).clamp_min(0.005)
        return admitted, intervention.clamp_max(1)
