"""The trained PPO arm controller, NumPy only: IMU conversion, planar kinematics, command governor,
measured histories and the actor.

Inference copy of Isaac-hydraulic-actuator `hydraulic_controller` (sensors, kinematics, observations,
policy), loaded from a bundle written by its `export_robot_bundle.py native`. Every load replays the
bundle's reference cases, computed there with the training code, and refuses a mismatch.

Frame: lower-carriage X/Z, pitch about +Y. Joint order boom, arm, bucket, carriage pitch [rad]. Angles
use the training sensor contract (raw AHRS pitch minus its fixed mounting pitch), and the training USD
is retargeted to the robot profile in that calibration, so this one geometry is also the machine's.
Requests are bucket cutting-edge twists [m/s, m/s, rad/s]; a pivot policy converts them internally.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

WEIGHTS = np.array([1.0, 1.0, 0.2])   # task weights of the training governor and IK: x, z, angle


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def joint_state(quats_wxyz, gyro_dps, sensors: dict):
    """q [rad] and rates [rad/s] from one controller status: `imu_quat_wxyz` (raw AHRS) and
    `imu_gyro_dps`, rows base, boom, arm, bucket. All four gyro Y axes are parallel."""
    quats = np.asarray(quats_wxyz, np.float64)
    gyro_y = np.asarray(gyro_dps, np.float64)[:, 1]
    norm = np.linalg.norm(quats, axis=1)
    if quats.shape != (4, 4) or not np.isfinite(quats).all() or norm.min() < 0.5 or not np.isfinite(gyro_y).all():
        raise ValueError("expected four finite IMU quaternions and gyro rows")
    w, x, y, z = (quats / norm[:, None]).T
    pitch = np.arctan2(-2 * (x * z - w * y), 1 - 2 * (x * x + y * y)) - sensors["mount_pitch_rad"]
    q = np.r_[wrap(np.diff(pitch)), pitch[0] - sensors["carriage_pitch_zero_rad"]]
    return q, np.deg2rad(np.r_[np.diff(gyro_y), gyro_y[0]])


class Kinematics:
    """Batched planar chain over the exported joint frames; q is [N, 4]."""

    def __init__(self, geometry: dict, grid: np.ndarray):
        self.frames = [(np.array(a), np.array(b), int(channel)) for a, b, channel in geometry["frames"]]
        self.limits = np.array(geometry["limits_rad"])
        self.tip = np.array(geometry["tip"])
        if grid.dtype != np.bool_ or grid.ndim != 3 or len(set(grid.shape)) != 1 or grid.shape[0] < 2:
            raise ValueError("invalid collision lookup grid")
        self.grid = grid
        self.cache = {}

    def pose_jacobian(self, q, point="tip"):
        """Pose [x, z, bucket angle] of the cutting edge or the bucket pivot, and its [N, 3, 4] Jacobian."""
        # One tick asks for the same q several times; the chain costs ~0.5 ms on the Jetson.
        key = (q.tobytes(), point)
        if key not in self.cache:
            if len(self.cache) > 3:
                self.cache.clear()
            self.cache[key] = self._pose_jacobian(q, point)
        return self.cache[key]

    def _pose_jacobian(self, q, point):
        transform = np.broadcast_to(np.eye(4), (len(q), 4, 4))
        joints = {}
        for parent, child_inverse, channel in self.frames:
            transform = transform @ parent
            if channel >= 0:
                joints[channel] = transform
                c, s = np.cos(q[:, channel]), np.sin(q[:, channel])
                rotation = np.tile(np.eye(4), (len(q), 1, 1))
                rotation[:, 0, 0] = rotation[:, 2, 2] = c
                rotation[:, 0, 2], rotation[:, 2, 0] = s, -s
                transform = transform @ rotation
            transform = transform @ child_inverse
        position = (transform @ self.tip)[:, :3, 3] if point == "tip" else joints[2][:, :3, 3]
        blade = transform[:, :3, 1]           # bucket-local +Y points toward the cutting lip
        pose = np.c_[position[:, ::2], np.arctan2(-blade[:, 2], blade[:, 0])]
        columns = []
        for i in range(4):
            axis = joints[i][:, :3, 1]
            linear = np.cross(axis, position - joints[i][:, :3, 3])[:, ::2]
            columns.append(np.c_[linear, axis[:, 1]])
        return pose, np.stack(columns, -1)

    def tracking_twist(self, q, tip_twist, point):
        """An edge twist at the policy's tracked point: v_tip = v_pivot + omega * [offset_z, -offset_x]."""
        if point == "tip":
            return tip_twist
        offset = self.pose_jacobian(q)[0][:, :2] - self.pose_jacobian(q, "pivot")[0][:, :2]
        return np.c_[tip_twist[:, :2] - offset[:, ::-1] * [1, -1] * tip_twist[:, 2:], tip_twist[:, 2]]

    def valid(self, q, margin):
        """Finite, inside the joint limits by `margin` (arm joints) and clear of self-collision."""
        margins = np.array([margin, margin, margin, 0.0])   # the passive carriage pitch has its own bound
        lo, hi = self.limits[:, 0] + margins, self.limits[:, 1] - margins
        with np.errstate(invalid="ignore"):
            inside = np.isfinite(q).all(1) & ((q >= lo) & (q <= hi)).all(1)
        n = self.grid.shape[0]
        span = self.limits[:3, 1] - self.limits[:3, 0]
        cell = np.rint((np.nan_to_num(q[:, :3]) - self.limits[:3, 0]) / span * (n - 1)).clip(0, n - 1).astype(int)
        return inside & ~self.grid[cell[:, 0], cell[:, 1], cell[:, 2]]

    def inverse(self, targets, seed, margin, iterations=35):
        """Edge poses [N, 3] -> q at the seed's carriage pitch, and which were reached validly."""
        q = np.tile(seed, (len(targets), 1))
        for _ in range(iterations):
            pose, jac = self.pose_jacobian(q)
            error = targets - pose
            error[:, 2] = wrap(error[:, 2])
            j = jac[:, :, :3] * WEIGHTS[:, None]
            jt = j.transpose(0, 2, 1)
            dq = np.linalg.solve(jt @ j + 1e-6 * np.eye(3), jt @ (error * WEIGHTS)[:, :, None])[:, :, 0]
            q[:, :3] = (q[:, :3] + dq.clip(-0.2, 0.2)).clip(self.limits[:3, 0], self.limits[:3, 1])
        pose = self.pose_jacobian(q)[0]
        reached = (np.linalg.norm(pose[:, :2] - targets[:, :2], axis=1) < 0.0005) & (
            np.abs(wrap(pose[:, 2] - targets[:, 2])) < 0.005)
        return q, reached & self.valid(q, margin)


def observation(kin, point, q, v_history, u_history, u_stride, command):
    """The training observation layout (without lookahead) from 100 Hz histories, newest first."""
    pose, jac = kin.pose_jacobian(q, point)
    twist = np.einsum("nij,nj->ni", jac[:, :, :3], v_history[:, 0, :3])
    n = len(q)
    return np.c_[q, v_history[:, ::2].reshape(n, -1), u_history[:, ::u_stride].reshape(n, -1), pose[:, :2],
                 np.sin(pose[:, 2]), np.cos(pose[:, 2]), twist, command, command - twist]


class Bundle:
    """A verified controller: push() one measurement per 100 Hz tick, act() once per policy step."""

    def __init__(self, directory):
        directory = Path(directory)
        self.contract = c = json.loads((directory / "contract.json").read_text())
        if c.get("bundle_version") != 3 or c["action_transform"] != "tanh" or c["settings"]["dt"] != 0.01:
            raise ValueError(f"{directory}: not a native PPO bundle (export_robot_bundle.py native)")
        raw = (directory / "arrays.npz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != c["arrays_sha256"]:
            raise ValueError(f"{directory}: arrays.npz changed since export")
        a = self.arrays = dict(np.load(directory / "arrays.npz"))
        self.kin = Kinematics(c["geometry"], a["collision_grid"])
        self.settings, self.sensors = c["settings"], c["sensors"]
        self.point, self.steps, self.u_stride = c["tracked_point"], c["lookahead_steps"], c["u_stride"]
        self.speed_limits = np.array(c["joint_speed_limits_rad_s"]) * c.get("governor_joint_speed_margin", 0.8)
        self.layers = [(a[f"w{i}"].astype(np.float64), a[f"b{i}"].astype(np.float64))
                       for i in range(sum(k.startswith("w") for k in a))]
        self.q = np.zeros(4)
        self.v_history = np.zeros((c["velocity_samples"], 4))
        self.u_history = np.zeros((c["command_samples"], 3))
        self.replay_reference()

    def admit(self, q, v, requested):
        """The training command governor: the requested tracked-point twist, scaled to the joint speed
        limits and kept inside the joint margins and clear of self-collision over its lookahead."""
        cfg, kin = self.settings, self.kin
        command = np.nan_to_num(requested)
        speed = np.linalg.norm(command[:, :2], axis=1, keepdims=True)
        command[:, :2] *= np.minimum(cfg["speed_max"] / np.maximum(speed, 1e-8), 1)
        command[:, 2] = command[:, 2].clip(-cfg["pitch_rate_max"], cfg["pitch_rate_max"])
        jac = kin.pose_jacobian(q, self.point)[1][:, :, :3]
        j = jac * WEIGHTS[:, None]
        jt = j.transpose(0, 2, 1)
        dq = np.linalg.solve(jt @ j + 1e-5 * np.eye(3), jt @ (command * WEIGHTS)[:, :, None])[:, :, 0]
        allowed = np.where(dq < 0, self.speed_limits[:, 0], self.speed_limits[:, 1])
        dq *= np.minimum((allowed / np.maximum(np.abs(dq), 1e-9)).min(1, keepdims=True), 1)
        margin, horizon = cfg["joint_margin"], cfg["lookahead"]
        predicted = q[:, :3] + 0.25 * v[:, :3]
        dq = dq.clip(np.minimum((kin.limits[:3, 0] + margin - predicted) / horizon, 0),
                     np.maximum((kin.limits[:3, 1] - margin - predicted) / horizon, 0))

        def safe(scale):
            proposed = q.copy()
            proposed[:, :3] += horizon * dq * scale[:, None]
            return kin.valid(proposed, margin)

        scale = np.ones(len(q))
        for _ in range(5):
            scale = np.where(safe(scale), scale, scale * 0.5)
        scale = np.where(safe(scale), scale, 0.0)
        admitted = np.einsum("nij,nj->ni", jac, dq * scale[:, None])
        # Never more translation than requested.
        admitted[:, :2] *= np.minimum(speed / np.maximum(np.linalg.norm(admitted[:, :2], axis=1, keepdims=True), 1e-8), 1)
        admitted[:, 2] = admitted[:, 2].clip(-cfg["pitch_rate_max"], cfg["pitch_rate_max"])
        return admitted

    def actor(self, observations):
        """Raw actions; the valves are their tanh."""
        x = (observations - self.arrays["obs_mean"]) / self.arrays["obs_scale"]
        for weight, bias in self.layers[:-1]:
            x = np.tanh(x @ weight.T + bias)
        return x @ self.layers[-1][0].T + self.layers[-1][1]

    def replay_reference(self):
        a = self.arrays
        ref = {k[4:]: v.astype(np.float64) for k, v in a.items() if k.startswith("ref_")}
        q, n = ref["q"], len(ref["q"])
        tracking = self.kin.tracking_twist(q, ref["requested"], self.point)
        admitted = self.admit(q, ref["v"], tracking)
        observed = np.c_[observation(self.kin, self.point, q, ref["v_history"], ref["u_history"], self.u_stride,
                                     admitted), ref["lookahead"].reshape(n, -1)]
        for v, u in zip(ref["seq_v"], ref["seq_u"]):
            self.push(q[0], v, u)
        rolled = observation(self.kin, self.point, q[:1], self.v_history[None], self.u_history[None],
                             self.u_stride, admitted[:1])
        self.v_history[:], self.u_history[:] = 0, 0
        for name, actual in (("tip_pose", self.kin.pose_jacobian(q)[0]), ("tracking", tracking),
                             ("admitted", admitted), ("observation", observed), ("action", self.actor(observed)),
                             ("seq_observation", rolled)):
            if actual.shape != ref[name].shape or not np.allclose(actual, ref[name], atol=1e-4, rtol=1e-4):
                raise ValueError(f"bundle reference mismatch in {name}: this runtime differs from training")

    def push(self, q, v, applied):
        """One 100 Hz measurement and the valve command [boom, arm, bucket] applied before it."""
        self.q = np.asarray(q, np.float64)
        for history, sample in ((self.v_history, v), (self.u_history, applied)):
            history[1:] = history[:-1].copy()
            history[0] = sample

    def pose(self):
        """Cutting-edge pose [x m, z m, angle rad] at the last measurement."""
        return self.kin.pose_jacobian(self.q[None])[0][0]

    def act(self, request, feedforward=None):
        """Edge twist request -> (valves [boom, arm, bucket] in [-1, 1], admitted tracked-point twist).

        `feedforward` [1 + lookahead steps, 3] is the reference's own edge twist now and at the next
        policy steps; without it the request is taken to hold.
        """
        q = self.q[None]
        admitted = self.admit(q, self.v_history[:1], self.kin.tracking_twist(q, np.array([request], float), self.point))
        ahead = np.zeros((self.steps, 3))
        if feedforward is not None and self.steps:
            ahead = self.kin.tracking_twist(q, feedforward[1:] - feedforward[0], self.point)   # q broadcasts
        observed = np.c_[observation(self.kin, self.point, q, self.v_history[None], self.u_history[None],
                                     self.u_stride, admitted), ahead.reshape(1, -1)]
        action = self.actor(observed)[0]
        if observed.shape[1] != self.contract["observation_dim"] or not np.isfinite(action).all():
            raise ValueError("invalid observation or actor output")
        return np.tanh(action), admitted[0]
