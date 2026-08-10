#!/usr/bin/env python3
"""SmolVLA inference loop for the MASI excavator (split TRT engines).

    IR cam1 + joint angles + instruction -> SmolVLA (split ONNX/TRT) -> valves

Modes (safety ladder):
    --synthetic     no robot, no camera — synthetic image + zero state. Proves
                    the model pipeline + engine builds. Safe anywhere.
    (default)       robot connected, observations real, actions PRINTED ONLY.
    --live          actions are actually sent to the valves. Base (un-finetuned)
                    weights produce garbage actions — do not use --live until a
                    finetuned checkpoint is in place and the pump is your call.

The policy emits a 50-step action chunk per observation; we execute the first
--n-action-steps of it open-loop at --fps, then re-infer. With ~0.2 s
inference and 25 steps at 30 Hz this re-plans roughly every 1 s.

First run builds the TRT engines (minutes); later runs load from
--cache-dir in seconds.

Usage (model-only check, today's goal 1):
    .venv-lerobot/bin/python -m lerobot_vla.run_inference --synthetic --loops 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lerobot_vla.smolvla_split import NormStats, SmolVLASplitPolicy

DEFAULT_SPLIT_DIR = Path.home() / "GitHub/spark-projects/orin-nano/smolvla-runtime/exports/ainekko_base_split"
DEFAULT_TOKENIZER = Path.home() / "GitHub/spark-projects/orin-nano/smolvla-runtime/exports/tokenizer"

LOG = logging.getLogger("run_inference")


def load_norm(stats_path: str | None) -> NormStats:
    if not stats_path:
        LOG.warning("No --dataset-stats given: state/action normalization is "
                    "IDENTITY. Fine for base-weight plumbing tests only.")
        return NormStats()
    stats = json.loads(Path(stats_path).read_text())
    norm = NormStats.from_lerobot_stats(stats)
    LOG.info("Loaded normalization stats from %s", stats_path)
    return norm


def main() -> int:
    p = argparse.ArgumentParser(description="SmolVLA split-engine inference loop.")
    p.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--cache-dir", default="/tmp/smolvla_split_cache")
    p.add_argument("--instruction", default="scoop sand and dump it to the left")
    p.add_argument("--num-steps", type=int, default=10, help="Denoise steps")
    p.add_argument("--n-action-steps", type=int, default=25,
                   help="How many of the 50 chunk actions to execute before re-inferring")
    p.add_argument("--fps", type=float, default=30.0, help="Action execution rate")
    p.add_argument("--dataset-stats", default=None,
                   help="Path to a LeRobot dataset stats.json for state/action normalization")
    p.add_argument("--robot", default="auto")
    p.add_argument("--exposure-us", type=float, default=None,
                   help="Lock IR exposure (us) — use the SAME value the dataset "
                        "was recorded with")
    p.add_argument("--gain", type=float, default=None, help="IR sensor gain 16..248")
    p.add_argument("--synthetic", action="store_true",
                   help="No robot/camera; synthetic observation (model-only test)")
    p.add_argument("--live", action="store_true",
                   help="SEND actions to the valves (default: print only)")
    p.add_argument("--loops", type=int, default=0,
                   help="Stop after N infer+execute cycles (0 = run until Ctrl+C)")
    p.add_argument("--seed", type=int, default=None, help="Fix the denoise noise seed")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    policy = SmolVLASplitPolicy(
        split_dir=args.split_dir,
        tokenizer_dir=args.tokenizer,
        cache_dir=args.cache_dir,
        num_steps=args.num_steps,
        action_dim=4,
        norm=load_norm(args.dataset_stats),
        seed=args.seed,
    )

    robot = None
    if not args.synthetic:
        from lerobot_vla.excavator_robot import MasiExcavator
        from lerobot_vla.ir_camera import IRCameraConfig
        robot = MasiExcavator(
            profile=args.robot,
            camera_config=IRCameraConfig(exposure_us=args.exposure_us,
                                         gain=args.gain))
        robot.connect()
        if args.live:
            LOG.warning("LIVE MODE: actions will drive the valves. Pump is under "
                        "your control (it is NOT auto-enabled).")
        else:
            LOG.info("Dry run: observations are real, actions are printed only.")

    def get_observation():
        if robot is not None:
            obs = robot.get_observation()
            return obs["observation.images.cam1"], obs["observation.state"]
        img = (np.random.default_rng(0).integers(0, 255, (480, 640, 3))
               .astype(np.uint8))
        return img, np.zeros(4, dtype=np.float32)

    # Warmup / engine build happens on the first sample_actions call.
    LOG.info("Warmup inference (builds TRT engines on first ever run)...")
    img, state = get_observation()
    t0 = time.perf_counter()
    policy.sample_actions(img, args.instruction, state)
    LOG.info("Warmup done in %.1fs", time.perf_counter() - t0)

    period = 1.0 / args.fps
    cycle = 0
    try:
        while True:
            img, state = get_observation()
            t0 = time.perf_counter()
            chunk = policy.sample_actions(img, args.instruction, state)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            n_exec = min(args.n_action_steps, len(chunk))
            LOG.info("cycle=%d infer=%.0fms state=[%s] a0=[%s] a%d=[%s]",
                     cycle, infer_ms,
                     " ".join(f"{v:+.1f}" for v in state),
                     " ".join(f"{v:+.2f}" for v in chunk[0]),
                     n_exec - 1,
                     " ".join(f"{v:+.2f}" for v in chunk[n_exec - 1]))

            next_tick = time.perf_counter()
            for i in range(n_exec):
                if robot is not None and args.live:
                    robot.send_action(chunk[i])
                next_tick += period
                dt = next_tick - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)

            cycle += 1
            if args.loops and cycle >= args.loops:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if robot is not None:
            robot.disconnect()

    LOG.info("Done: %d cycles.", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
