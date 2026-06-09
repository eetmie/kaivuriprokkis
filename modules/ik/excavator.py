"""
IMU glue for the excavator: convert corrected absolute IMU quaternions to
canonical joint angles in radians.

This module is the boundary between the sensor layer and the kinematics
package. Inputs are corrected absolute IMU quaternions (mounting offsets
already removed by the hardware layer) in a known role order. Output is
``q_rad`` in the model's joint order.

Extraction rules supported in ``IMUConfig.chain``:

  average_z_yaw          — average Z twist across all IMUs (slew yaw)
  gravity_pitch_delta    — pitch of child against gravity, minus parent pitch
  relative_axis_twist    — twist of child relative to parent about a local axis

A joint without a chain entry remains zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import yaml

from .kinematics import (
    _fk_core,
    _jacobian_core,
    _jacobian_metrics,
)
from .math import (
    extract_axis_rotation,
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
)
from .model import ExcavatorModel


# ----------------------------
# IMU config
# ----------------------------

@dataclass(frozen=True)
class IMUChainStep:
    joint: str
    output_index: int
    extraction: str                  # 'average_z_yaw' | 'gravity_pitch_delta' | 'relative_axis_twist'
    role: Optional[str] = None       # sensor role for child link (None for slew)
    parent_role: Optional[str] = None
    axis: Optional[str] = None       # 'x' | 'y' | 'z' for twist extractions


@dataclass(frozen=True)
class IMUConfig:
    """IMU layout and extraction rules.

    sensor_order is the order the ``imu_quats`` array uses when calling
    :func:`joint_angles_from_imus`. ``mapping`` (role -> physical sensor
    index) and ``mounting_offsets`` (role -> [w, x, y, z]) are kept here
    for the hardware layer to use; this module does not apply them.
    """

    sensor_order: Tuple[str, ...]
    mapping: Mapping[str, int]
    chain: Tuple[IMUChainStep, ...]
    mounting_offsets: Mapping[str, np.ndarray]


def _to_quat(value: Any, field_name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field_name} must be a list of 4 numbers, got {value!r}")
    return np.asarray(value, dtype=np.float32)


def build_imu_config(imu_section: Mapping[str, Any]) -> IMUConfig:
    """Build an IMUConfig from a parsed YAML 'imu' section dict."""
    mapping = imu_section.get("imu_mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("imu.imu_mapping must be a non-empty mapping of role -> index")

    chain_raw = imu_section.get("chain")
    if not isinstance(chain_raw, list) or not chain_raw:
        raise ValueError("imu.chain must be a non-empty list")

    steps: list[IMUChainStep] = []
    for i, item in enumerate(chain_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"imu.chain[{i}] must be a mapping")
        extraction = item.get("extraction")
        if extraction not in ("average_z_yaw", "gravity_pitch_delta", "relative_axis_twist"):
            raise ValueError(
                f"imu.chain[{i}].extraction must be one of "
                f"average_z_yaw/gravity_pitch_delta/relative_axis_twist, got {extraction!r}"
            )
        if "output_index" not in item:
            raise ValueError(f"imu.chain[{i}].output_index is required")
        steps.append(IMUChainStep(
            joint=str(item.get("joint", f"joint_{i}")),
            output_index=int(item["output_index"]),
            extraction=str(extraction),
            role=item.get("role"),
            parent_role=item.get("parent_role"),
            axis=item.get("axis"),
        ))

    # Sensor order: collect role + parent_role mentions in chain order,
    # filtered to roles that exist in the mapping.
    sensor_order: list[str] = []
    def _add(role: Optional[str]) -> None:
        if role and role in mapping and role not in sensor_order:
            sensor_order.append(role)
    for step in steps:
        _add(step.parent_role)
        _add(step.role)
    if not sensor_order:
        sensor_order = list(mapping.keys())

    offsets_raw = imu_section.get("mounting_offsets_quat") or {}
    if not isinstance(offsets_raw, dict):
        raise ValueError("imu.mounting_offsets_quat must be a mapping when present")
    offsets = {
        role: _to_quat(q, f"imu.mounting_offsets_quat.{role}")
        for role, q in offsets_raw.items()
    }

    return IMUConfig(
        sensor_order=tuple(sensor_order),
        mapping=dict(mapping),
        chain=tuple(steps),
        mounting_offsets=offsets,
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


def load_imu_config(path: str | Path) -> IMUConfig:
    """Load IMUConfig from a control_config.yaml file."""
    cfg = _load_yaml(path)
    imu = cfg.get("imu")
    if not isinstance(imu, dict):
        raise ValueError("control_config.yaml is missing top-level 'imu' section")
    return build_imu_config(imu)


# ----------------------------
# Extraction primitives
# ----------------------------

def average_axis_twist_quaternion(quats: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Hemisphere-aligned average of per-quaternion twists about ``axis``."""
    quats = np.asarray(quats, dtype=np.float32)
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-12)
    if quats.ndim != 2 or quats.shape[1] != 4 or len(quats) == 0:
        raise ValueError("Expected quats with shape (n, 4)")

    accum = np.zeros(4, dtype=np.float32)
    reference: Optional[np.ndarray] = None
    for q in quats:
        angle = extract_axis_rotation(q, axis)
        twist = quat_from_axis_angle(axis, np.float32(angle))
        if reference is None:
            reference = twist.copy()
        elif float(np.dot(reference, twist)) < 0.0:
            twist = -twist
        accum += twist
    if float(np.linalg.norm(accum)) < 1e-9:
        return reference if reference is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat_normalize(accum)


def gravity_pitch_from_quat(quat: np.ndarray) -> np.float32:
    """Extract link pitch against gravity from a corrected IMU quaternion."""
    q = quat_normalize(np.asarray(quat, dtype=np.float32))
    w, x, y, z = q[0], q[1], q[2], q[3]
    gx = np.float32(2.0) * (x * z - w * y)
    gz = np.float32(1.0) - np.float32(2.0) * (x * x + y * y)
    return np.float32(np.arctan2(-gx, gz))


def _wrap_angle_pi(angle: float) -> float:
    a = float(angle)
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


_AXIS_LOOKUP = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


def _axis_from_name(name: Optional[str], fallback: np.ndarray) -> np.ndarray:
    if name in _AXIS_LOOKUP:
        return _AXIS_LOOKUP[name]
    return np.asarray(fallback, dtype=np.float32)


# ----------------------------
# Main API
# ----------------------------

def joint_angles_from_imus(
    imu_quats: np.ndarray,
    imu_cfg: IMUConfig,
    model: ExcavatorModel,
) -> np.ndarray:
    """Convert corrected absolute IMU quats to canonical joint angles.

    ``imu_quats`` must be shape (len(imu_cfg.sensor_order), 4), in the role
    order ``imu_cfg.sensor_order`` (e.g. base, boom, arm, bucket).

    Returns ``q_rad`` of shape (model.num_joints,), with joint i populated
    only when the chain has an entry whose ``output_index == i``. Joints
    without a chain entry stay at 0.
    """
    imu_quats = np.asarray(imu_quats, dtype=np.float32)
    if imu_quats.ndim != 2 or imu_quats.shape[1] != 4:
        raise ValueError(f"imu_quats must have shape (n, 4), got {imu_quats.shape}")
    if imu_quats.shape[0] != len(imu_cfg.sensor_order):
        raise ValueError(
            f"Expected {len(imu_cfg.sensor_order)} IMU quaternions (one per role "
            f"in {imu_cfg.sensor_order}), got {imu_quats.shape[0]}"
        )

    role_quats = {role: imu_quats[i] for i, role in enumerate(imu_cfg.sensor_order)}

    angles = np.zeros(model.num_joints, dtype=np.float32)

    # Cache average-Z-yaw quat lazily (used by slew step).
    slew_axis_local = model.axes[0]  # first joint axis in its parent frame
    z_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    averaged_yaw_quat: Optional[np.ndarray] = None

    for step in imu_cfg.chain:
        i = step.output_index
        if i < 0 or i >= angles.shape[0]:
            raise ValueError(
                f"imu.chain step for joint '{step.joint}' has output_index={i} "
                f"out of range for model with {angles.shape[0]} joints"
            )

        if step.extraction == "average_z_yaw":
            axis = _axis_from_name(step.axis, z_world)
            if averaged_yaw_quat is None:
                averaged_yaw_quat = average_axis_twist_quaternion(imu_quats, axis)
            angles[i] = extract_axis_rotation(averaged_yaw_quat, slew_axis_local if i == 0 else axis)
            continue

        if step.role is None or step.role not in role_quats:
            raise ValueError(
                f"imu.chain step '{step.joint}' (extraction={step.extraction}) "
                f"requires role '{step.role}' to be in sensor_order {imu_cfg.sensor_order}"
            )
        child_q = role_quats[step.role]
        parent_q = role_quats[step.parent_role] if step.parent_role in role_quats else None

        if step.extraction == "gravity_pitch_delta":
            child_pitch = float(gravity_pitch_from_quat(child_q))
            parent_pitch = float(gravity_pitch_from_quat(parent_q)) if parent_q is not None else 0.0
            angles[i] = np.float32(_wrap_angle_pi(child_pitch - parent_pitch))
        elif step.extraction == "relative_axis_twist":
            axis = _axis_from_name(step.axis, model.axes[i])
            if parent_q is not None:
                rel = quat_normalize(quat_multiply(quat_conjugate(parent_q), child_q))
            else:
                rel = child_q
            angles[i] = extract_axis_rotation(rel, axis)
        else:
            raise ValueError(f"Unsupported extraction mode {step.extraction!r}")

    return angles


# ----------------------------
# Numba warmup
# ----------------------------

def warmup_numba_functions() -> None:
    """Prime the numba JIT for the FK/Jacobian/metric cores.

    Call this once at startup so the first real IK tick doesn't pay the
    compile cost. Uses a 3-joint dummy chain — input layout matches the
    real model so all branches compile.
    """
    q = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    axes = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    offsets = np.array([
        [0.0, 0.0, 0.05],
        [0.02, 0.0, 0.06],
        [0.4, 0.0, 0.0],
    ], dtype=np.float32)
    tip = np.array([0.03, 0.0, -0.14], dtype=np.float32)

    try:
        for _ in range(2):
            origins, axes_w, _, ee_pos, _ = _fk_core(q, axes, offsets, tip)
            J = _jacobian_core(origins, axes_w, ee_pos)
            _jacobian_metrics(J)
    except Exception as e:  # pragma: no cover
        print(f"Numba warmup failed: {e}")


__all__ = [
    "IMUChainStep",
    "IMUConfig",
    "build_imu_config",
    "load_imu_config",
    "joint_angles_from_imus",
    "average_axis_twist_quaternion",
    "gravity_pitch_from_quat",
    "warmup_numba_functions",
]
