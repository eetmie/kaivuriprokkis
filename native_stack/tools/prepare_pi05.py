"""Verify an OpenPI pi0.5 bundle (pi05_libero or pi05_openpi_state) and unpack its parity
fixtures for C++.

Build-time utility only. Use the validated VLA numpy environment; the inference
executable does not import Python, Torch, Triton or ONNX Runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def libero_values(fixture, info):
    return {
        "image0": fixture["image_" + info["image_keys"][0]].astype(np.float16),
        "image1": fixture["image_" + info["image_keys"][1]].astype(np.float16),
        "raw0": fixture["image"].astype(np.uint8),
        "raw1": fixture["wrist"].astype(np.uint8),
        "tokens": fixture["tokens"].astype(np.int64),
        "mask": fixture["token_mask"].astype(np.bool_),
        "noise": fixture["noise"].astype(np.float32),
        "reference": fixture["fp16_actions_normalized"].astype(np.float32),
        "mixed_reference": fixture["actions"].astype(np.float32),
    }, None


def state_values(fixture, info):
    """pi05_openpi_state: real dataset frames, raw joint angles and the task's prompt."""
    tokens = fixture["tokens"].reshape(-1).astype(np.int64)
    mask = fixture["token_mask"].reshape(-1).astype(np.bool_)
    real = list(tokens[mask])
    tasks = [t for t, v in info["tasks"].items() if real[:len(v["head"])] == v["head"]]
    if len(tasks) != 1:
        raise ValueError(f"fixture prompt matches {len(tasks)} bundle task heads")
    if fixture["raw_state"].dtype != np.float32:
        raise ValueError("raw_state must be the dataset's float32 state")
    return {
        "image0": fixture["image_" + info["image_keys"][0]].astype(np.float16),
        "image1": fixture["image_" + info["image_keys"][1]].astype(np.float16),
        "raw0": fixture["raw_cam1"].astype(np.uint8),
        "raw1": fixture["raw_cam2"].astype(np.uint8),
        "tokens": tokens,
        "mask": mask,
        "noise": fixture["noise"].astype(np.float32),
        "reference": fixture["fp16_actions_normalized"].astype(np.float32),
        "mixed_reference": fixture["actions"].astype(np.float32),
        "state": fixture["raw_state"],
        "state_normalized": fixture["state"].reshape(-1)[:info["state_dim"]].astype(np.float64),
        "robot_reference": fixture["fp16_actions_robot"].astype(np.float64),
    }, tasks[0]


def prepare(bundle, output):
    bundle = bundle.resolve()
    info = json.loads((bundle / "bundle.json").read_text())
    policies = {"pi05_libero": libero_values, "pi05_openpi_state": state_values}
    if info.get("policy") not in policies or not info.get("complete"):
        raise ValueError("requires a complete OpenPI pi05_libero or pi05_openpi_state bundle")
    # Check every original artifact before deriving C++ fixtures.
    for line in (bundle / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = (bundle / name.lstrip(" *")).resolve()
        if not path.is_relative_to(bundle) or sha(path) != digest:
            raise ValueError(f"bundle checksum mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    record = {"bundle": str(bundle), "policy": info["policy"],
              "bundle_sha256": sha(bundle / "bundle.json"),
              "source_manifest_sha256": sha(bundle / "MANIFEST.sha256"), "fixtures": []}
    for source in sorted((bundle / "fixtures").glob("fixture_*.npz")):
        target = output / source.stem
        target.mkdir()
        with np.load(source, allow_pickle=False) as fixture:
            values, task = policies[info["policy"]](fixture, info)
            files = {}
            for name, value in values.items():
                path = target / (name + ".npy")
                np.save(path, np.ascontiguousarray(value), allow_pickle=False)
                files[path.name] = sha(path)
            entry = {"directory": source.stem, "source_sha256": sha(source), "files": files}
            if task is not None:
                entry["task"] = task
            record["fixtures"].append(entry)
    if not record["fixtures"]:
        raise ValueError("bundle has no parity fixtures")
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Verified bundle and prepared {len(record['fixtures'])} fixtures at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.bundle, args.out)
