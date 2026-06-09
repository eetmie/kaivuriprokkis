"""
URDF-style excavator kinematic model.

The chain is described as a list of revolute joints plus a rigid tool tip
offset. Each joint has:
  - a name (slew, boom, arm, bucket, optional tool joints)
  - a unit axis vector (rotation axis in the parent's rotated frame)
  - parent_to_joint_xyz: fixed translation from the parent's rotated frame
    to this joint's origin (URDF "origin xyz")

The tool has a single rigid offset from the last joint's rotated frame to
the working point (bucket tip).

YAML shape::

    robot:
      joints:
        - name: slew
          axis: [0, 0, 1]
          parent_to_joint_xyz: [0.0, 0.0, 0.07]
        - name: boom
          axis: [0, 1, 0]
          parent_to_joint_xyz: [0.0165, 0.0, 0.0645]
        - name: arm
          axis: [0, 1, 0]
          parent_to_joint_xyz: [0.468, 0.0, 0.0]
        - name: bucket
          axis: [0, 1, 0]
          parent_to_joint_xyz: [0.250, 0.0, 0.0]
      tool:
        parent_to_tip_xyz: [0.031, 0.0, -0.142]

Adding a joint (rototilt etc.) is one extra entry in the joints list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


@dataclass(frozen=True)
class Joint:
    """One revolute joint in the URDF-style chain."""

    name: str
    axis: np.ndarray              # shape (3,), float32, unit vector
    parent_to_joint_xyz: np.ndarray  # shape (3,), float32, meters


@dataclass(frozen=True)
class Tool:
    """Rigid offset from the last joint's rotated frame to the tool tip."""

    parent_to_tip_xyz: np.ndarray  # shape (3,), float32, meters


@dataclass(frozen=True)
class ExcavatorModel:
    """Full kinematic chain.

    Holds both per-joint dataclasses (for code that wants names + structure)
    and packed numpy arrays (for the numba FK/Jacobian hot path).
    """

    joints: tuple[Joint, ...]
    tool: Tool

    # Packed views for the hot path; populated in __post_init__.
    axes: np.ndarray             # (n, 3) float32
    offsets: np.ndarray          # (n, 3) float32  parent_to_joint_xyz per joint
    tip_offset: np.ndarray       # (3,)  float32

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)


def _to_xyz(value: Any, field_name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a list of 3 numbers, got {value!r}")
    return np.asarray(value, dtype=np.float32)


def _to_unit_axis(value: Any, field_name: str) -> np.ndarray:
    vec = _to_xyz(value, field_name)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError(f"{field_name} must be non-zero, got {value!r}")
    return (vec / norm).astype(np.float32)


def build_excavator_model(
    joint_specs: Sequence[Mapping[str, Any]],
    tool_spec: Mapping[str, Any],
) -> ExcavatorModel:
    """Build an ExcavatorModel from parsed-dict specs.

    Useful in tests / programmatic construction without YAML.
    """
    if not joint_specs:
        raise ValueError("robot.joints must contain at least one joint")

    joints: list[Joint] = []
    seen: set[str] = set()
    for i, spec in enumerate(joint_specs):
        if not isinstance(spec, Mapping):
            raise ValueError(f"robot.joints[{i}] must be a mapping, got {type(spec).__name__}")
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"robot.joints[{i}].name must be a non-empty string")
        if name in seen:
            raise ValueError(f"robot.joints[{i}].name='{name}' is duplicated")
        seen.add(name)

        axis = _to_unit_axis(spec.get("axis"), f"robot.joints[{i}].axis")
        offset = _to_xyz(
            spec.get("parent_to_joint_xyz"),
            f"robot.joints[{i}].parent_to_joint_xyz",
        )
        joints.append(Joint(name=name, axis=axis, parent_to_joint_xyz=offset))

    if not isinstance(tool_spec, Mapping):
        raise ValueError("robot.tool must be a mapping")
    tip = _to_xyz(tool_spec.get("parent_to_tip_xyz"), "robot.tool.parent_to_tip_xyz")
    tool = Tool(parent_to_tip_xyz=tip)

    axes = np.asarray([j.axis for j in joints], dtype=np.float32)
    offsets = np.asarray([j.parent_to_joint_xyz for j in joints], dtype=np.float32)

    return ExcavatorModel(
        joints=tuple(joints),
        tool=tool,
        axes=axes,
        offsets=offsets,
        tip_offset=tip,
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        candidates = [p, Path(__file__).resolve().parents[2] / p]
        p = next((c for c in candidates if c.exists()), p)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {p} must be a mapping")
    return data


def load_excavator_model(path: str | Path) -> ExcavatorModel:
    """Load the URDF-style excavator model from a control_config.yaml file."""
    cfg = _load_yaml(path)
    robot = cfg.get("robot")
    if not isinstance(robot, dict):
        raise ValueError("control_config.yaml is missing top-level 'robot' section")

    joints = robot.get("joints")
    if not isinstance(joints, list):
        raise ValueError("robot.joints must be a list")

    tool = robot.get("tool")
    if not isinstance(tool, dict):
        raise ValueError("robot.tool must be a mapping")

    return build_excavator_model(joints, tool)
