"""Small, dependency-free GR00T bundle inspection and integrity checks."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or external bundle artifact: {name}")
    return path


def verify_manifest(root: Path) -> dict:
    root = root.resolve()
    covered = set()
    manifest = inside(root, "MANIFEST.sha256")
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        if len(digest) != 64 or name in covered:
            raise ValueError(f"invalid or duplicate manifest entry: {name}")
        if sha256(inside(root, name)) != digest:
            raise ValueError(f"bundle checksum mismatch: {name}")
        covered.add(name)
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
             and p.name != "MANIFEST.sha256"
             and not (p.name.startswith("_meta_") and p.suffix == ".json")}
    if files - covered:
        raise ValueError(f"bundle has unlisted artifacts: {sorted(files - covered)[:5]}")
    if "bundle.json" not in covered:
        raise ValueError("bundle.json must be covered by MANIFEST.sha256")
    return {"manifest_sha256": sha256(manifest), "files_verified": len(covered)}


def load_bundle(root: Path) -> dict:
    root = root.expanduser().resolve()
    b = json.loads(inside(root, "bundle.json").read_text())
    if b.get("schema") != "groot-n1.7-split/2":
        raise ValueError("this probe currently supports groot-n1.7-split/2 bundles")
    for key in ("views", "frames", "action_horizon", "state_dim", "action_dim"):
        value = b.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"bundle {key} must be a positive integer")
    offsets = b.get("video_delta_indices")
    if (not isinstance(offsets, list) or len(offsets) != b["frames"]
            or any(type(x) is not int for x in offsets)
            or offsets != sorted(set(offsets)) or offsets[-1] != 0):
        raise ValueError("video_delta_indices must be ordered, unique and end at zero")
    if not isinstance(b.get("task"), str) or not b["task"].strip():
        raise ValueError("bundle lacks its baked task text")
    if (not isinstance(b.get("timesteps"), list) or not b["timesteps"]
            or any(not isinstance(t, (int, float)) or not math.isfinite(t)
                   for t in b["timesteps"])):
        raise ValueError("bundle lacks a finite denoising schedule")
    names = set()
    for graph in b.get("graphs", []):
        if graph["name"] in names:
            raise ValueError("duplicate graph name")
        names.add(graph["name"])
        inside(root, graph["file"])
    if not names or not any(n.startswith("vision_") for n in names):
        raise ValueError("bundle lacks model graphs")
    return b


def describe(root: Path, verify=False) -> dict:
    root = root.expanduser().resolve()
    b = load_bundle(root)
    return {
        "bundle": str(root), "bundle_sha256": sha256(root / "bundle.json"),
        "schema": b["schema"], "model": b.get("model"),
        "views": b["views"], "frames_per_view": b["frames"],
        "video_delta_indices": b["video_delta_indices"],
        "history_timebase": "training sampling rate required; do not assume live camera FPS",
        "action_horizon": b["action_horizon"], "model_action_dim": b["action_dim"],
        "real_action_dim": b.get("action_used"), "model_state_dim": b["state_dim"],
        "embodiment": b.get("embodiment"), "denoising_steps": len(b["timesteps"]),
        "prompt": {"mode": "baked_in_export", "text": b["task"],
                   "arbitrary_runtime_text": False},
        "actuator_output": False,
        "robot_boundary": "not implemented by this probe; outputs are model-space diagnostics",
        "integrity": verify_manifest(root) if verify else {"verified": False},
    }
