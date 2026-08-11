#!/usr/bin/env python3
"""LeRobot dataset collection for the MASI excavator (gamepad teleop).

The training-data twin of simple_drive.py: drives the valves open-loop
straight from the gamepad while recording synchronized episodes into the
LeRobot v3 dataset format (lerobot 0.5.1 — the same version pinned on the
DGX Spark finetune side).

Per frame (at --fps, default 30):
    observation.state        float32[4]  joint angles [slew, lift, tilt, scoop] deg
    observation.images.cam1  uint8 480x640x3  D435i infrared left imager (emitter OFF)
    action                   float32[4]  normalized valve cmds actually sent [-1,1]
    task                     the natural-language instruction (--task)

This loop runs at --fps because that is the dataset frame rate. It does NOT
drive the valves at that rate: the recorded action is handed to
ExcavatorController's direct-command mode as a setpoint, and its 100 Hz thread
writes the hardware. That is what makes the dataset valid for deployment —
recording and inference push setpoints through the identical fixed-rate writer,
so the action->motion mapping the policy learns is the one it will get back.
Driving I2C straight from this loop (the old behaviour) made that mapping
depend on whatever rate the loop happened to hit between video encodes.

Button controls (same layout as simple_drive.py):
    A: start episode / stop + SAVE episode
    B: stop + DISCARD current episode (re-record)
    X: toggle hydraulic pump
    Y: reload servo config from disk

Usage:
    .venv-lerobot/bin/python -m lerobot_vla.record_episodes \
        --repo-id masi/excavator_sand_v0 \
        --task "scoop sand and dump it to the left"

Videos are encoded H.264 (not the AV1 default) because the Spark's LeRobot
loader needed an H.264 transcode to read AV1 reliably.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lerobot_vla.excavator_robot import (
    ACTION_KEY, CAMERA_KEY, JOINT_NAMES, STATE_KEY, MasiExcavator,
)
from lerobot_vla.ir_camera import IRCameraConfig
from simple_drive import (
    BTN_A, BTN_B, BTN_X, BTN_Y, LocalGamepadInput,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data_collection" / "lerobot_datasets"


def build_features(cam_h: int, cam_w: int) -> dict:
    return {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        CAMERA_KEY: {
            "dtype": "video",
            "shape": (cam_h, cam_w, 3),
            "names": ["height", "width", "channels"],
        },
    }


def manual_action_from_axes(axes: dict) -> np.ndarray:
    """Map gamepad axes to [slew, lift, tilt, scoop] — same sticks as simple_drive."""
    return np.array([
        axes["left_rl"],    # slew
        axes["right_ud"],   # lift (boom)
        axes["left_ud"],    # tilt (arm)
        axes["right_rl"],   # scoop (bucket)
    ], dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description="Record LeRobot episodes with gamepad teleop.")
    p.add_argument("--repo-id", default="masi/excavator_sand_v0",
                   help="Dataset repo id (also the folder name under --root)")
    p.add_argument("--task", required=True,
                   help='Instruction for the episodes, e.g. "scoop sand and dump it to the left"')
    p.add_argument("--root", default=None,
                   help=f"Dataset root dir (default {DEFAULT_ROOT}/<repo-id>)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--robot", default="auto", help="Board profile (auto = detect)")
    p.add_argument("--max-episode-s", type=float, default=120.0,
                   help="Auto-stop + save an episode after this long")
    p.add_argument("--no-slew", action="store_true", help="Disable the slew channel")
    p.add_argument("--exposure-us", type=float, default=None,
                   help="Lock IR exposure to this value in microseconds "
                        "(default: auto-exposure; find a value with tune_exposure.py)")
    p.add_argument("--gain", type=float, default=None,
                   help="IR sensor gain 16..248 (only with --exposure-us)")
    p.add_argument("--resume", action="store_true",
                   help="Append to an existing dataset instead of creating a new one")
    p.add_argument("--legacy-direct-write", action="store_true",
                   help="Drive valves from this loop instead of the control "
                        "thread (bench A/B only — do not record with this)")
    args = p.parse_args()

    root = Path(args.root) if args.root else DEFAULT_ROOT / args.repo_id.replace("/", "_")

    cam_cfg = IRCameraConfig(width=640, height=480, fps=args.fps,
                             exposure_us=args.exposure_us, gain=args.gain)
    if args.exposure_us:
        print(f"[camera] exposure locked: {args.exposure_us:.0f} us"
              + (f", gain {args.gain:.0f}" if args.gain else ""))
    else:
        print("[camera] AUTO exposure — lock it with --exposure-us for "
              "consistent training data")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.resume:
        if not root.exists():
            print(f"--resume given but {root} does not exist"); return 1
        dataset = LeRobotDataset.resume(args.repo_id, root=root, vcodec="h264")
        print(f"[dataset] Resumed {root} at episode {dataset.meta.total_episodes}")
    else:
        if root.exists():
            print(f"Dataset root {root} already exists. Use --resume to append, "
                  f"or remove it / pick another --repo-id.")
            return 1
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=build_features(cam_cfg.height, cam_cfg.width),
            root=root,
            robot_type=MasiExcavator.robot_type,
            use_videos=True,
            vcodec="h264",
        )
        print(f"[dataset] Created {root}")

    # ── hardware ─────────────────────────────────────────────────────────────
    # The gamepad is polled at --fps, so the setpoint hold has to cover a few
    # missed frames without decaying. 0.15 s ≈ 4 frames at 30 Hz: long enough to
    # ride out a video-encode hiccup, short enough that a dropped gamepad or a
    # wedged loop ramps the valves down in well under a second.
    robot = MasiExcavator(profile=args.robot,
                          camera_config=cam_cfg,
                          enable_slew=not args.no_slew,
                          use_control_thread=not args.legacy_direct_write,
                          setpoint_hold_s=max(0.1, 4.0 / args.fps),
                          setpoint_decay_s=0.2)
    robot.connect()

    pad = LocalGamepadInput()
    if not pad.open():
        robot.disconnect()
        return 1

    print(f"\nTask: {args.task!r}")
    print("A=start/save episode  B=discard episode  X=pump  Y=reload-config  Ctrl+C=quit\n")

    period = 1.0 / args.fps
    next_tick = time.perf_counter()
    mask_prev = 0
    recording = False
    ep_start = 0.0
    ep_frames = 0
    saved_eps = dataset.meta.total_episodes
    last_status = time.time()

    def stop_episode(save: bool, reason: str):
        nonlocal recording, ep_frames
        recording = False
        robot.stop_motion()
        if save and ep_frames > 0:
            print(f"\n[EP] saving {ep_frames} frames ({ep_frames / args.fps:.1f}s) "
                  f"({reason})... ", end="", flush=True)
            dataset.save_episode()
            print("done.")
        else:
            dataset.clear_episode_buffer()
            print(f"\n[EP] discarded ({reason}).")
        ep_frames = 0

    try:
        while True:
            axes, mask = pad.poll()
            if axes is not None:
                def btn(b): return bool(mask & (1 << b))
                def prev(b): return bool(mask_prev & (1 << b))

                if btn(BTN_A) and not prev(BTN_A):
                    if not recording:
                        recording = True
                        ep_start = time.time()
                        ep_frames = 0
                        print(f"\n[EP] recording episode {saved_eps}...")
                    else:
                        stop_episode(save=True, reason="button A")
                        saved_eps += 1
                if btn(BTN_B) and not prev(BTN_B) and recording:
                    stop_episode(save=False, reason="button B")
                if btn(BTN_X) and not prev(BTN_X):
                    print(f"\n[pump] {'ON' if robot.toggle_pump() else 'OFF'}")
                if btn(BTN_Y) and not prev(BTN_Y):
                    print(f"\n[config] reload {'OK' if robot.reload_config() else 'FAILED'}")
                mask_prev = mask

                action = manual_action_from_axes(axes)
            else:
                action = np.zeros(4, dtype=np.float32)

            if not pad.is_live():
                action = np.zeros(4, dtype=np.float32)

            sent = robot.send_action(action)

            if recording:
                obs = robot.get_observation()
                if obs[CAMERA_KEY] is not None:
                    dataset.add_frame({
                        STATE_KEY: obs[STATE_KEY],
                        CAMERA_KEY: obs[CAMERA_KEY],
                        ACTION_KEY: sent,
                        "task": args.task,
                    })
                    ep_frames += 1
                if time.time() - ep_start >= args.max_episode_s:
                    stop_episode(save=True, reason=f"{args.max_episode_s:.0f}s limit")
                    saved_eps += 1

            now = time.time()
            if now - last_status >= 5.0:
                last_status = now
                ja = robot.get_joint_angles()
                state = (f"REC ep{saved_eps} {ep_frames} frames" if recording else "idle")
                sp = robot.get_setpoint_status()
                # decay < 1 means this loop is not feeding setpoints fast enough
                # and the control thread is ramping the valves down.
                sp_str = ("" if sp['decay'] >= 1.0
                          else f" | *** SETPOINT STALE {sp['age_s']:.2f}s "
                               f"decay={sp['decay']:.2f} ***")
                print(f"[STATUS] {state} | pump={'ON' if robot.pump_enabled else 'OFF'} | "
                      f"slew={ja[0]:+.1f} lift={ja[1]:+.1f} tilt={ja[2]:+.1f} "
                      f"scoop={ja[3]:+.1f} deg"
                      + ("" if pad.is_live() else " | *** GAMEPAD LOST ***")
                      + sp_str)

            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if recording and ep_frames > 0:
            stop_episode(save=True, reason="shutdown")
            saved_eps += 1
        pad.close()
        robot.disconnect()
        print("[dataset] finalizing...")
        try:
            dataset.finalize()
        except Exception as exc:
            print(f"[dataset] finalize failed: {exc}")
        print(f"[dataset] {saved_eps} episodes total in {root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
