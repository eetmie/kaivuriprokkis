#!/usr/bin/env python3
"""IR cam1 exposure sweep — find the exposure value to lock for recording.

Headless-friendly: captures one frame per exposure setting (emitter off, same
stream config as recording) and writes PNGs + a stats table, so the images can
be inspected from another machine.

    .venv-lerobot/bin/python -m lerobot_vla.tune_exposure \
        --exposures 500,1000,2000,4000,8000,16000,33000 --out /tmp/ir_sweep

Reading the stats: mean around 80-130 with clip_hi below ~1% is usually right.
At 30 fps anything above ~33000 us caps the frame rate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep IR cam1 exposure values.")
    p.add_argument("--exposures", default="500,1000,2000,4000,8000,16000,33000",
                   help="Comma-separated exposure values in microseconds")
    p.add_argument("--gain", type=float, default=None, help="Fixed gain (16..248)")
    p.add_argument("--out", default="/tmp/ir_exposure_sweep", help="Output dir")
    p.add_argument("--auto-too", action="store_true", default=True,
                   help="Also capture an auto-exposure reference frame")
    p.add_argument("--settle-frames", type=int, default=8,
                   help="Frames to discard after changing exposure")
    args = p.parse_args()

    import cv2
    import pyrealsense2 as rs

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exposures = [float(e) for e in args.exposures.split(",")]

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    profile = pipe.start(cfg)
    sensor = profile.get_device().first_depth_sensor()
    sensor.set_option(rs.option.emitter_enabled, 0)

    def grab() -> np.ndarray:
        for _ in range(args.settle_frames):
            pipe.wait_for_frames(timeout_ms=2000)
        f = pipe.wait_for_frames(timeout_ms=2000).get_infrared_frame(1)
        return np.asanyarray(f.get_data())

    results = []

    def record(label: str, img: np.ndarray, note: str = ""):
        path = out / f"ir_{label}.png"
        cv2.imwrite(str(path), img)
        clip_lo = float((img <= 2).mean() * 100)
        clip_hi = float((img >= 253).mean() * 100)
        results.append((label, float(img.mean()), float(img.std()),
                        clip_lo, clip_hi, note))

    try:
        if args.auto_too:
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            time.sleep(0.5)
            record("auto", grab(),
                   note=f"AE exposure={sensor.get_option(rs.option.exposure):.0f}us "
                        f"gain={sensor.get_option(rs.option.gain):.0f}")

        sensor.set_option(rs.option.enable_auto_exposure, 0)
        if args.gain is not None:
            sensor.set_option(rs.option.gain, args.gain)
        gain = sensor.get_option(rs.option.gain)

        for e in exposures:
            sensor.set_option(rs.option.exposure, e)
            record(f"{int(e):06d}us", grab(), note=f"gain={gain:.0f}")
    finally:
        pipe.stop()

    print(f"\n{'setting':<12} {'mean':>6} {'std':>6} {'%clip_lo':>9} {'%clip_hi':>9}  note")
    for label, mean, std, lo, hi, note in results:
        print(f"{label:<12} {mean:6.1f} {std:6.1f} {lo:9.2f} {hi:9.2f}  {note}")
    print(f"\nPNGs in {out}/ — pick the exposure, then record with "
          f"--exposure-us <value>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
