#!/usr/bin/env python3
"""Optional CPU parity check against pinned upstream RTC source and torch autograd.

Executes the GR00T seed block and PI prefix-weight function from trusted checkouts.
No model weights, CUDA, camera or robot service are used. Requires CPU torch + numpy.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch


def block(path, predicate):
    source = path.read_text()
    matches = [node for node in ast.walk(ast.parse(source)) if predicate(node)]
    if len(matches) != 1:
        raise ValueError(f"Expected one upstream code block in {path}, got {len(matches)}")
    return compile(ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[])),
                   str(path), "exec")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--groot-source", required=True, type=Path,
                        help="Pinned gr00t/model/gr00t_n1d7/gr00t_n1d7.py")
    parser.add_argument("--rtc-source", required=True, type=Path, help="PI RTC src/model.py")
    args = parser.parse_args()
    results = {}
    for mode in ("groot17", "openpi"):
        probe = json.loads(subprocess.check_output([str(args.probe), mode], text=True))
        overlap, frozen = probe["overlap"], probe["frozen"]
        if mode == "groot17":
            # Run NVIDIA's actual RTC seed block, keeping its torch.linspace/ramp math.
            namespace = dict(torch=torch, actions=torch.full((1, 40, 4), -.5),
                             vel_strength=torch.ones((1, 40, 4)), device="cpu",
                             action_input={"action": torch.full((1, 40, 4), .1)},
                             options={"action_horizon": 40, "rtc_overlap_steps": overlap,
                                      "rtc_frozen_steps": frozen, "rtc_ramp_rate": 2.})
            exec(block(args.groot_source, lambda n: isinstance(n, ast.If)
                       and ast.unparse(n.test) == "'action' in action_input"), namespace)
            x, strength = namespace["actions"], namespace["vel_strength"]
            for _ in range(4):
                x = x + .25 * (.3 - .2 * x) * strength
            reference = x.detach().numpy().ravel()
        else:
            # Actual PI prefix weights. NumPy substitutes only for these JAX array ops.
            namespace = dict(jnp=np, jax=SimpleNamespace(Array=object),
                             PrefixAttentionSchedule=str)
            exec(block(args.rtc_source, lambda n: isinstance(n, ast.FunctionDef)
                       and n.name == "get_prefix_weights"), namespace)
            weights = torch.tensor(namespace["get_prefix_weights"](frozen, overlap, 40, "exp"),
                                   dtype=torch.float32).reshape(1, 40, 1)
            target = torch.zeros((1, 40, 4))
            target[:, :overlap] = .1
            x = torch.full((1, 40, 4), -.5)
            # Paper's FORWARD progress convention; C++ uses OpenPI's reversed time.
            for step in range(10):
                t = step / 10
                x = x.detach().requires_grad_(True)
                velocity = .3 - .2 * x
                endpoint = x + (1 - t) * velocity
                error = ((target - endpoint) * weights).detach()
                correction = torch.autograd.grad(endpoint, x, grad_outputs=error)[0]
                gamma = 5. if t == 0 else min(5., (t * t + (1 - t) ** 2) / (t * (1 - t)))
                x = x + .1 * (velocity + gamma * correction)
            reference = x.detach().numpy().ravel()
        actual = np.asarray(probe["actions"], dtype=np.float32)
        error = float(np.max(np.abs(actual - reference)))
        if not np.isfinite(error) or error > 2e-5:
            raise ValueError(f"{mode}: max absolute error {error} exceeds 2e-5")
        results[mode] = dict(max_abs_error=error, elements=int(actual.size), status="PASS")
    print(json.dumps(dict(status="PASS", analytic_only=True, results=results,
                          source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (args.groot_source, args.rtc_source)}), indent=2))


if __name__ == "__main__":
    main()
