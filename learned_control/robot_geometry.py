"""Robot-profile cutting-tip geometry and twist conversion to the policy's training tip."""

from __future__ import annotations

import torch

from learned_control.kinematics import ExcavatorKinematics


class RobotKinematics(ExcavatorKinematics):
    """Fixed-slew planar FK using the physical robot profile [m, rad].

    Generated deployment profiles match the policy's authored USD. Legacy
    profiles can also be inspected while retaining the policy observation frame.
    Collision checks still use the conservative simulation geometry; this is a
    free-space adapter, not a newly validated physical collision model.
    """

    def __init__(self, profile: dict, safety: ExcavatorKinematics):
        self.device = safety.device
        self.safety = safety
        self.profile = profile
        self.limits = safety.limits.clone()
        limits = profile["ik"]["joint_limits_relative"]
        for i, bounds in enumerate(limits[1:4]):
            if bounds is not None:
                limit = torch.deg2rad(torch.tensor(bounds, device=self.device))
                self.limits[i, 0] = torch.maximum(self.limits[i, 0], limit[0])
                self.limits[i, 1] = torch.minimum(self.limits[i, 1], limit[1])
        if torch.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("Robot and policy joint limits do not overlap")
        eye = torch.eye(4, device=self.device)
        self.frames = [(eye.clone(), eye.clone(), 3)]
        joints = profile["robot"]["joints"]
        if [j["name"] for j in joints] != ["slew", "boom", "arm", "bucket"]:
            raise ValueError("Expected the fixed-slew boom/arm/bucket chain")
        for i, joint in enumerate(joints):
            expected = [0.0, 0.0, 1.0] if i == 0 else [0.0, 1.0, 0.0]
            if joint["axis"] != expected:
                raise ValueError("The hardware demo requires aligned Y-axis arm joints")
            frame = eye.clone()
            frame[:3, 3] = torch.tensor(joint["parent_to_joint_xyz"], device=self.device)
            self.frames.append((frame, eye.clone(), i - 1))
        self.tip = eye.clone()
        self.tip[:3, 3] = torch.tensor(profile["robot"]["tool"]["parent_to_tip_xyz"], device=self.device)

    def pose_jacobian(self, q):
        """Return physical tip pose [m, m, rad] and Jacobian [m/rad; rad/rad]."""
        transforms, joints = self._transforms(q)
        point = (transforms[-1] @ self.tip)[:, :3, 3]
        angle = q.sum(1, keepdim=True) + self.profile["robot"]["tool"].get("pitch_offset_rad", 0.0)
        pose = torch.cat((point[:, [0, 2]], angle), 1)
        columns = []
        for i in range(4):
            frame = joints[i]
            axis = frame[:, :3, 1]
            linear = torch.linalg.cross(axis, point - frame[:, :3, 3])[:, [0, 2]]
            columns.append(torch.cat((linear, axis[:, 1:2]), 1))
        return pose, torch.stack(columns, -1)

    def colliding(self, q):
        return self.safety.colliding(q)


def policy_twist(q, physical_twist, robot, policy_kin, *, policy_q=None):
    """Map physical cutting-tip twist [m/s, m/s, rad/s] to the policy's trained tip."""
    _, real_jac = robot.pose_jacobian(q)
    # A corrected mounting calibration changes measured relative joint zeros.
    # The frozen policy continues observing its original training calibration.
    _, learned_jac = policy_kin.pose_jacobian(q if policy_q is None else policy_q)
    j = real_jac[:, :, :3]
    # A weighted condition check avoids blindly amplifying commands at a singularity.
    weighted = j * q.new_tensor([1.0, 1.0, 0.2])[None, :, None]
    if not torch.isfinite(j).all() or bool((torch.linalg.cond(weighted) > 100).any()):
        raise ValueError("Physical tip Jacobian is too close to a singularity")
    rates = torch.linalg.solve(j, physical_twist[:, :, None]).squeeze(-1)
    return torch.einsum("nij,nj->ni", learned_jac[:, :, :3], rates)
