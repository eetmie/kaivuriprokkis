"""Measure a verified X-VLA bundle without opening cameras or robot hardware."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from lerobot_vla.runtime.xvla import load_bundle
from lerobot_vla.runtime.xvla_native import XVLANativePolicy


def rss_mib():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--chain", choices=("host", "device", "graph"), default="graph")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    bundle = load_bundle(args.split_dir)
    before = rss_mib()
    t0 = time.perf_counter()
    policy = XVLANativePolicy(args.split_dir, cache_dir=args.cache_dir, chain=args.chain)
    try:
        load_s = time.perf_counter() - t0
        fx = bundle["fixture"]
        with np.load(Path(args.split_dir) / fx["file"]) as fixture:
            images = [np.ascontiguousarray(x) for x in fixture["raw"]]
            # Feasibility bundles use padded model state. A physical checkpoint
            # needs inverse normalization to start from its saved model fixture.
            state = fixture["proprio"][0, :policy.state_dim].astype(np.float32)
            if policy.processor_contract:
                from lerobot_vla.runtime.vendor.xvla_bundle_contract import unnormalize_vector
                state = unnormalize_vector(state, policy.processor_contract["state"])
            noise = fixture["x1"].copy()
        for _ in range(5):
            policy.sample_actions(images, fx["task"], state, noise)
        after = rss_mib()
        samples, stages = [], []
        start = time.perf_counter()
        while time.perf_counter() - start < args.seconds:
            t0 = time.perf_counter()
            chunk = policy.sample_actions(images, fx["task"], state, noise)
            samples.append((time.perf_counter() - t0) * 1000)
            stages.append(dict(policy.last_timings))
        report = {
            "host": platform.node(), "chain": args.chain, "bundle": args.split_dir,
            "purpose": "Model benchmark; no camera, controller, PWM or pump is opened",
            "load_s": load_s, "rss_before_mib": before, "rss_loaded_mib": after,
            "rss_after_benchmark_mib": rss_mib(), "samples": len(samples),
            "latency_ms_p50_p95_p99": np.percentile(samples, [50, 95, 99]).tolist(),
            "stage_mean_ms": {k: float(np.mean([t.get(k, 0) for t in stages])) for k in stages[0]},
            "chunk_shape": list(chunk.shape), "fixture": policy.fixture_parity,
            "torch_imported": "torch" in sys.modules, "onnxruntime_imported": "onnxruntime" in sys.modules,
            "scratch_mib": policy.engines.scratch_bytes / 2**20,
        }
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        policy.close()


if __name__ == "__main__":
    main()
