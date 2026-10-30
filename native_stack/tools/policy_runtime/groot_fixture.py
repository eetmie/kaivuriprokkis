"""Load an existing GR00T AOT cache directly, without building or patching the CLI.

This is a short-lived Python adapter for bringing up the engine boundary. Control
and capture remain native services. The final C++ backend can use the same fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from .bundle import describe, inside, sha256


def comparison(actual, expected):
    import numpy as np

    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite or incorrectly shaped inference result")
    a, b = a.ravel(), b.ravel()
    delta = np.abs(a - b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(a @ b / denom) if denom else float(np.array_equal(a, b))
    return {"cosine": cosine, "max_abs": float(delta.max()),
            "max_pct_range": float(delta.max() / max(float(np.ptp(b)), 1e-8) * 100)}


def run_fixture(bundle_dir: Path, cache: Path, vla_root: Path, output: Path,
                iterations=5, graph=True):
    bundle_dir, cache, vla_root = (p.expanduser().resolve() for p in (bundle_dir, cache, vla_root))
    # All validation before loading the model; no actuator client is imported here.
    report = describe(bundle_dir, verify=True)
    if report["frames_per_view"] != 2:
        raise ValueError("this initial engine adapter handles two temporal slots; capture supports other offsets")
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    if not (vla_root / "bench/vendor/trt_device.py").is_file():
        raise ValueError("--vla-root must point to the jetson-orin-nano-vla source")
    sys.path.insert(0, str(vla_root))
    import numpy as np
    import tensorrt as trt
    from bench.aot.groot.candidate import verify
    from bench.vendor import groot_trt
    from bench.vendor.groot17_trt import Bundle, FrameHistory
    from bench.vendor.trt_device import Groot17Device

    candidate = verify(cache, bundle_dir)
    if candidate.get("tensorrt") != trt.__version__:
        raise ValueError("cached TensorRT version differs from the running version")
    runtime_version = groot_trt._cudart_version()
    stack = candidate.get("platform_stack", "")
    if f"cudart={runtime_version} " not in stack:
        raise ValueError("cache lacks matching CUDA runtime identity")
    groot_trt.ACCUMULATE = candidate["accumulate"]
    constants = cache / "step_constants.npz"
    if candidate.get("constants_sha256", {}).get(constants.name) != sha256(constants):
        raise ValueError("cached timestep constants are not pinned by candidate.json")

    bd = Bundle(bundle_dir)
    offsets = bd.b["video_delta_indices"]
    if offsets[0] >= 0:
        raise ValueError("two-slot engine fixture requires a negative historical index")
    with np.load(constants, allow_pickle=False) as z:
        bd.consts = [{} for _ in bd.b["timesteps"]]
        for key in z.files:
            step, name = key.split("/", 1)
            bd.consts[int(step)][name] = z[key]
    if not all(bd.consts):
        raise ValueError("empty timestep constants")

    fixture = inside(bundle_dir, bd.b["fixture"]["file"])
    with np.load(fixture, allow_pickle=False) as z:
        raw = z["raw"]
        pixels = np.stack([bd.preprocess(np.ascontiguousarray(im)) for im in raw])
        from bench.vendor.groot17_trt import _unpatchify
        grid = z["image_grid_thw"]
        expected_pixels = _unpatchify(z["pixel_values"].astype(np.float32), grid.shape[0],
                                     int(grid[0, 1]), int(grid[0, 2]))
        preprocess = comparison(pixels, expected_pixels)
        if preprocess["max_abs"] > 1e-6:
            raise ValueError(f"raw image preprocessing parity failed: {preprocess}")
        state, noise, reference = z["state"].copy(), z["noise"].copy(), z["action_pred"].copy()
    if pixels.shape[0] != bd.b["views"] * 2:
        raise ValueError("fixture must contain two real frames per view")

    output.mkdir(parents=True, exist_ok=False)
    initial = {**report, "status": "loading", "runtime": "Python TensorRT adapter; native services",
               "input_mode": "export_fixture", "cache": str(cache),
               "candidate_sha256": sha256(cache / "candidate.json"),
               "platform_stack": stack,
               "vla_source": str(vla_root), "actuator_output": False}
    (output / "summary.json").write_text(json.dumps(initial, indent=2) + "\n")
    try:
        started = time.monotonic()
        # Every engine already exists. Do not call prebuild_engines or load_step_constants:
        # those helpers can mutate a cache or spawn builders.
        engines = groot_trt.Engines(bd, cache, skip=["time", *bd.mod])
        # Fixture slots have a synthetic, explicitly declared one-second separation.
        # This is not a guess about the checkpoint's training or live camera FPS.
        lag = 1.0
        device = Groot17Device(bd, engines, lag, graph=graph)
        views = bd.b["views"]
        rows = []
        chunk = None
        for i in range(iterations + 1):
            device.history = FrameHistory(lag)
            device.infer(pixels[:views], state, noise, 0.0)
            timing = {}
            chunk, age = device.infer(pixels[views:], state, noise, lag, timings=timing)
            full = comparison(chunk, reference)
            used = bd.b.get("action_used")
            active = comparison(chunk[..., :used], reference[..., :used]) if used else full
            if any(r["cosine"] < .99999 or r["max_pct_range"] > .13 for r in (full, active)):
                raise ValueError(f"GR00T action fixture parity failed: full={full}, active={active}")
            if abs(age - lag) > 1e-9:
                raise ValueError("fixture used the wrong historical frame")
            if i:
                rows.append({"iteration": i, "history_age_s": age, **timing})
        np.save(output / "model_actions.npy", chunk, allow_pickle=False)
        total = np.asarray([row["total"] for row in rows])
        summary = {**initial, "status": "PASS", "elapsed_with_load_s": time.monotonic() - started,
                   "iterations": iterations, "cuda_graph": graph,
                   "fixture_sha256": sha256(fixture), "preprocess_parity": preprocess,
                   "full_chunk_parity": full, "active_chunk_parity": active,
                   "history": {"source": "real fixture frames in two slots", "fixture_separation_s": lag},
                   "latency_ms": {"p50": float(np.median(total)), "p99": float(np.quantile(total, .99)),
                                  "max": float(total.max())},
                   "output_units": "model space; not excavator valve commands"}
        (output / "timings.json").write_text(json.dumps(rows, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    except BaseException as error:
        initial.update(status="FAIL", error=str(error))
        (output / "summary.json").write_text(json.dumps(initial, indent=2) + "\n")
        raise
