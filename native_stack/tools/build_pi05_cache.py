"""Build the native TensorRT template cache for a pi0.5 split bundle on the Jetson.

    python build_pi05_cache.py --bundle BUNDLE --out CACHE [--vla-root ~/vla-clean-install-20261012/source]

Same plans and cache keys as jetson-orin-nano-vla's baseline-cache (bench/vendor/pi05_trt.py
prebuild_templates), without its policy whitelist: one weight-stripped, individually refittable plan
per layer kind, built by that checkout's build_template in its own subprocess. Fine-tuned bundles
with 50 action rows use native TensorRT attention (no AOT Triton plugin).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vla-root", type=Path, default=Path.home() / "vla-clean-install-20261012" / "source")
    a = ap.parse_args()
    info = json.loads((a.bundle / "bundle.json").read_text())
    if not info.get("complete"):
        sys.exit(f"{a.bundle} is not a complete bundle")
    lines = (a.bundle / "MANIFEST.sha256").read_text().splitlines()
    bad = [rel for digest, rel in (l.split("  ", 1) for l in lines) if sha256(a.bundle / rel) != digest]
    if bad:
        sys.exit(f"bundle manifest mismatch: {bad[:5]}")
    sys.path.insert(0, str(a.vla_root))
    from bench.vendor.pi05_trt import _cudart_version, _trt_version
    stack = (f"tensorrt={_trt_version()} cudart={_cudart_version()} opt_level=2 "
             "workspace_mb=512 strongly_typed refit_individual strip_plan no_tf32")
    a.out.mkdir(parents=True, exist_ok=True)
    kinds = list(dict.fromkeys(c["kind"] for c in info["components"]))
    built = {}
    for kind in kinds:
        onnx = a.bundle / "templates" / f"{kind}.onnx"
        want = f"{sha256(onnx)} {stack}"
        engine, key = a.out / f"{kind}.engine", a.out / f"{kind}.onnx.sha256"
        if engine.exists() and key.exists() and key.read_text().strip() == want:
            continue
        t0 = time.time()
        subprocess.run([sys.executable, "-c",
                        "import sys; from bench.vendor.pi05_trt import build_template; "
                        "build_template(sys.argv[1], sys.argv[2], sys.argv[3])",
                        str(onnx), str(onnx.with_suffix(".json")), str(engine)],
                       check=True, cwd=str(a.vla_root), env={**os.environ, "PYTHONNOUSERSITE": "1"})
        key.write_text(want + "\n")
        built[kind] = round(time.time() - t0, 1)
        print(f"built {kind} in {built[kind]} s", flush=True)
    (a.out / "cache.json").write_text(json.dumps({
        "bundle": str(a.bundle.resolve()), "bundle_manifest_sha256": sha256(a.bundle / "MANIFEST.sha256"),
        "policy": info["policy"], "attention": "native", "stack": stack, "built_s": built}, indent=2) + "\n")
    print(f"cache ready: {a.out}", flush=True)


if __name__ == "__main__":
    main()
