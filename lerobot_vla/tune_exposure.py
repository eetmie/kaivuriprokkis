#!/usr/bin/env python3
"""Exposure/gain sweep for the D435i — find the values to lock for recording.

Sweeps the IR (cam1) and colour (cam2) imagers, both on ONE pipeline so every
frame of the sweep sees the same scene under the same light. Headless-friendly:
writes a PNG per setting plus a stats table, so the images can be inspected from
another machine.

    .venv-lerobot/bin/python -m lerobot_vla.tune_exposure --out /tmp/cam_sweep

    # IR only, finer exposure ladder, one gain
    .venv-lerobot/bin/python -m lerobot_vla.tune_exposure --camera ir \
        --exposures 8000,12000,16000,20000 --gains-ir 16

Reading the stats: mean around 80-130 with clip_hi below ~1% is usually right,
and rows meeting that are marked with a star. At 30 fps anything above
~33000 us caps the frame rate, whichever imager it is.

The two sensors take exposure in DIFFERENT units (stereo in microseconds, colour
in 100 us ticks) and have different gain ranges (IR 16..248, RGB 0..128).
--exposures is given in microseconds for both; the conversion is done per sensor
from the range it reports. Gains are per-camera because their scales differ.
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

from lerobot_vla.camera import exposure_units_per_us

# What a good frame looks like; rows inside this get starred in the table.
TARGET_MEAN = (80.0, 130.0)
MAX_CLIP_HI_PCT = 1.0


def frame_stats(img: np.ndarray) -> tuple[float, float, float, float]:
    """mean, std, %pixels crushed to black, %pixels blown to white.

    Clipping is measured on the brightest channel so a single blown colour
    channel is not averaged away by the other two.
    """
    hottest = img.max(axis=2) if img.ndim == 3 else img
    coldest = img.min(axis=2) if img.ndim == 3 else img
    return (float(img.mean()), float(img.std()),
            float((coldest <= 2).mean() * 100), float((hottest >= 253).mean() * 100))


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep D435i exposure/gain for cam1/cam2.")
    p.add_argument("--camera", choices=("ir", "rgb", "both"), default="both",
                   help="Which imager(s) to sweep (default: both)")
    p.add_argument("--exposures", default="2000,4000,8000,16000,24000,33000",
                   help="Comma-separated exposure values in MICROSECONDS, "
                        "applied to whichever sensor is being swept")
    p.add_argument("--gains-ir", default="16,48,96",
                   help="Comma-separated IR gains to cross with --exposures (16..248)")
    p.add_argument("--gains-rgb", default="16,64,128",
                   help="Comma-separated RGB gains to cross with --exposures (0..128)")
    p.add_argument("--out", default="/tmp/cam_exposure_sweep", help="Output dir")
    p.add_argument("--no-auto", action="store_true",
                   help="Skip the auto-exposure reference frame")
    p.add_argument("--settle-frames", type=int, default=12,
                   help="Frames to discard after changing a setting")
    args = p.parse_args()

    import cv2
    import pyrealsense2 as rs

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exposures = [float(e) for e in args.exposures.split(",")]
    want_ir = args.camera in ("ir", "both")
    want_rgb = args.camera in ("rgb", "both")

    # Both streams on one pipeline: the sweep should compare imagers, not
    # compare two different moments of a scene with people moving in it.
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    if want_rgb:
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = pipe.start(cfg)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    color_sensor = profile.get_device().first_color_sensor() if want_rgb else None

    def grab(kind: str) -> np.ndarray:
        for _ in range(args.settle_frames):
            pipe.wait_for_frames(timeout_ms=2000)
        frames = pipe.wait_for_frames(timeout_ms=2000)
        if kind == "ir":
            return np.asanyarray(frames.get_infrared_frame(1).get_data())
        return np.asanyarray(frames.get_color_frame().get_data())

    results: list[tuple] = []

    def record(kind: str, label: str, img: np.ndarray, note: str = ""):
        # cv2 writes BGR; the colour stream is rgb8, so it needs flipping or the
        # PNGs come out with red and blue swapped.
        png = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
        cv2.imwrite(str(out / f"{kind}_{label}.png"), png)
        results.append((kind, label, *frame_stats(img), note))

    def sweep(kind: str, sensor, gains: list[float]) -> None:
        units = exposure_units_per_us(rs, sensor)
        e_rng = sensor.get_option_range(rs.option.exposure)
        g_rng = sensor.get_option_range(rs.option.gain)
        print(f"[{kind}] exposure {e_rng.min / units:.0f}..{e_rng.max / units:.0f} us, "
              f"gain {g_rng.min:.0f}..{g_rng.max:.0f}")

        if not args.no_auto:
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            time.sleep(0.5)
            record(kind, "auto", grab(kind),
                   note=f"AE chose {sensor.get_option(rs.option.exposure) / units:.0f}us "
                        f"gain={sensor.get_option(rs.option.gain):.0f}")
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        # Leaving auto-exposure takes effect several frames later than the
        # option write returns, so the first manual setting would otherwise be
        # measured on a frame still exposed by AE — which reads as a wildly
        # out-of-order first row (2000 us looking brighter than 4000 us).
        grab(kind)

        for g in gains:
            if not (g_rng.min <= g <= g_rng.max):
                print(f"[{kind}] skipping gain {g:.0f}: outside "
                      f"{g_rng.min:.0f}..{g_rng.max:.0f}")
                continue
            sensor.set_option(rs.option.gain, float(g))
            for e in exposures:
                value = e * units
                if not (e_rng.min <= value <= e_rng.max):
                    print(f"[{kind}] skipping exposure {e:.0f}us: outside sensor range")
                    continue
                sensor.set_option(rs.option.exposure, float(value))
                record(kind, f"e{int(e):06d}us_g{int(g):03d}", grab(kind))

    try:
        if want_ir:
            sweep("ir", depth_sensor, [float(g) for g in args.gains_ir.split(",")])
        if want_rgb:
            sweep("rgb", color_sensor, [float(g) for g in args.gains_rgb.split(",")])
    finally:
        pipe.stop()

    print(f"\n{'cam':<4} {'setting':<20} {'mean':>6} {'std':>6} {'%lo':>6} {'%hi':>6}   note")
    print("-" * 78)
    for kind, label, mean, std, lo, hi, note in results:
        good = (TARGET_MEAN[0] <= mean <= TARGET_MEAN[1]
                and hi <= MAX_CLIP_HI_PCT and label != "auto")
        print(f"{kind:<4} {label:<20} {mean:6.1f} {std:6.1f} {lo:6.2f} {hi:6.2f} "
              f"{'*' if good else ' '} {note}")

    # Best = closest to the middle of the target band among rows that pass.
    mid = sum(TARGET_MEAN) / 2
    for kind in ("ir", "rgb"):
        rows = [r for r in results if r[0] == kind and r[1] != "auto"
                and TARGET_MEAN[0] <= r[2] <= TARGET_MEAN[1] and r[5] <= MAX_CLIP_HI_PCT]
        if rows:
            best = min(rows, key=lambda r: abs(r[2] - mid))
            flag = "--exposure-ir" if kind == "ir" else "--exposure-rgb"
            gainflag = "--gain-ir" if kind == "ir" else "--gain-rgb"
            e, g = best[1].replace("us", "").split("_g")
            print(f"\n[{kind}] best in band: {best[1]} (mean {best[2]:.1f}) -> "
                  f"{flag} {int(e[1:])} {gainflag} {int(g)}")
        else:
            print(f"\n[{kind}] nothing landed in mean {TARGET_MEAN[0]:.0f}-"
                  f"{TARGET_MEAN[1]:.0f} with clip_hi <= {MAX_CLIP_HI_PCT}% — "
                  f"widen --exposures or change the lighting")

    print(f"\nPNGs in {out}/ — eyeball them, then record with the flags above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
