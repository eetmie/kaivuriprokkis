"""The policy observation layout, built from measured hardware state."""

from __future__ import annotations

import torch


def assemble_observation(q, velocity_history, command_history, u_stride, kin, command):
    """Assemble policy inputs from angles [rad], rates [rad/s], and twist commands [m/s, m/s, rad/s]."""
    pose, jac = kin.pose_jacobian(q)
    twist = torch.einsum("nij,nj->ni", jac[:, :, :3], velocity_history[:, 0, :3])
    return torch.cat(
        (
            q,
            velocity_history[:, ::2].flatten(1),
            command_history[:, ::u_stride].flatten(1),
            pose[:, :2],
            pose[:, 2:].sin(),
            pose[:, 2:].cos(),
            twist,
            command,
            command - twist,
        ),
        1,
    )


class MeasuredHistory:
    """100 Hz histories. Each measurement includes the command sent during the preceding interval."""

    def __init__(
        self,
        count: int,
        device: str,
        velocity_samples: int = 41,
        command_samples: int = 61,
        u_stride: int = 3,
    ):
        self.q = torch.zeros(count, 4, device=device)
        self.v = torch.zeros(count, velocity_samples, 4, device=device)
        self.u = torch.zeros(count, command_samples, 3, device=device)
        self.u_stride = u_stride

    def reset(self, ids, q):
        """Reset selected streams to angles [rad] with empty motion/command history."""
        self.q[ids] = q
        self.v[ids] = 0
        self.u[ids] = 0

    def push(self, q, velocity, preceding_command):
        """Append one 0.01 s measurement: angles [rad], rates [rad/s], normalized commands."""
        self.v[:, 1:] = self.v[:, :-1].clone()
        self.u[:, 1:] = self.u[:, :-1].clone()
        self.q.copy_(q)
        self.v[:, 0] = velocity
        self.u[:, 0] = preceding_command

    def observe(self, kin, command):
        """Return the shared observation vector for a twist command [m/s, m/s, rad/s]."""
        return assemble_observation(self.q, self.v, self.u, self.u_stride, kin, command)
