"""Load an exported controller bundle: TorchScript actor, geometry, collision grid and its contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from learned_control.kinematics import CommandGovernor, ExcavatorKinematics
from learned_control.observations import MeasuredHistory
from learned_control.robot_geometry import RobotKinematics
from learned_control.sensors import SENSOR_CONTRACT
from learned_control.settings import ControllerSettings, sha256

FILES = ("actor.pt", "geometry.json", "robot_geometry.json", "bucket_control_config.yaml", "collision_grid.npy")
REFERENCE = "reference_io.npz"


class PolicyBundle:
    """Deterministic CPU actor; ``valves`` includes the single required tanh transform.

    A bundle exported with ``reference_io.npz`` is replayed at load: kinematics, governor, observation layout and
    actor must reproduce what the training code computed, so a drifted copy here or a changed numeric library
    refuses to run instead of driving the valves differently.
    """

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        self.directory = directory.resolve()
        self.contract = c = json.loads((directory / "contract.json").read_text())
        if c["bundle_version"] != 2 or c["action_transform"] != "tanh" or c["sensors"] != SENSOR_CONTRACT:
            raise ValueError("Unsupported hardware contract")
        if c.get("tracked_point", "tip") != "tip":
            raise ValueError("This runtime sends bucket-tip requests; the bundle tracks another point")
        if set(c["robot_profile_sha256"]) != {"servo_config.yaml", "control_config.yaml", "profile.yaml"}:
            raise ValueError("Bundle must pin calibration and board configuration")
        for name in (*FILES, *((REFERENCE,) if REFERENCE in c["files"] else ())):
            if sha256(directory / name) != c["files"][name]:
                raise ValueError(f"Bundle file changed: {name}")
        self.actor = torch.jit.load(str(directory / "actor.pt"), map_location="cpu").eval()
        self.kin = ExcavatorKinematics.from_dict(json.loads((directory / "geometry.json").read_text()))
        grid = np.load(directory / "collision_grid.npy", allow_pickle=False)
        if grid.dtype != np.bool_ or grid.ndim != 3 or len(set(grid.shape)) != 1 or grid.shape[0] < 2:
            raise ValueError("Invalid collision lookup grid")
        self.kin._grid = torch.from_numpy(grid.copy())
        self.robot_kin = RobotKinematics(json.loads((directory / "robot_geometry.json").read_text()), self.kin)
        self.settings = ControllerSettings(**c["settings"])
        self.history = MeasuredHistory(1, "cpu", c["velocity_samples"], c["command_samples"], c["u_stride"])
        self.governor = CommandGovernor(
            self.kin,
            self.settings,
            None if "joint_speed_limits_rad_s" not in c else torch.tensor(c["joint_speed_limits_rad_s"]),
            c.get("governor_joint_speed_margin", 0.8),
        )
        self.verified = REFERENCE in c["files"]
        if self.verified:
            self.replay_reference()

    @torch.inference_mode()
    def replay_reference(self) -> None:
        """Recompute the export's reference cases with this runtime and refuse any mismatch."""
        ref = {key: torch.from_numpy(value) for key, value in np.load(self.directory / REFERENCE).items()}
        pose, jacobian = self.kin.pose_jacobian(ref["q"])
        robot_pose, robot_jacobian = self.robot_kin.pose_jacobian(ref["q"])
        admitted, _ = self.governor(ref["q"], ref["v"], ref["requested"])
        history = MeasuredHistory(
            len(ref["q"]), "cpu", ref["v_history"].shape[1], ref["u_history"].shape[1], self.contract["u_stride"]
        )
        history.q, history.v, history.u = ref["q"], ref["v_history"], ref["u_history"]
        observation = history.observe(self.kin, admitted)
        checks = {
            "pose": (pose, ref["pose"]),
            "jacobian": (jacobian, ref["jacobian"]),
            "robot_pose": (robot_pose, ref["robot_pose"]),
            "robot_jacobian": (robot_jacobian, ref["robot_jacobian"]),
            "admitted": (admitted, ref["admitted"]),
            "observation": (observation, ref["observation"]),
            "action": (self.actor(observation), ref["action"]),
        }
        for name, (actual, expected) in checks.items():
            if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=1e-4, rtol=1e-4):
                error = (actual - expected).abs().max().item() if actual.shape == expected.shape else "shape"
                raise ValueError(f"Bundle reference mismatch in {name} ({error}); runtime differs from training")

    @torch.inference_mode()
    def valves(self, observations):
        """Return boom/arm/bucket normalized valve values [-1, 1] from a finite observation vector."""
        if observations.shape != (1, self.contract["observation_dim"]) or not torch.isfinite(observations).all():
            raise ValueError("Invalid hardware observation")
        action = self.actor(observations)
        if action.shape != (1, 3) or not torch.isfinite(action).all():
            raise ValueError("Invalid actor output")
        return torch.tanh(action)

    def check_profile(self, profile_dir):
        """Refuse changed IMU/geometry/PWM calibration before accessing hardware."""
        for name, fingerprint in self.contract["robot_profile_sha256"].items():
            if sha256(Path(profile_dir) / name) != fingerprint:
                raise ValueError(f"Robot configuration changed since export: {name}")
