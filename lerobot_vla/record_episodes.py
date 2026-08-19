#!/usr/bin/env python3
"""LeRobot dataset collection for the MASI excavator (gamepad teleop).

The training-data twin of simple_drive.py: drives the valves open-loop
straight from the gamepad while recording synchronized episodes into the
LeRobot v3 dataset format (lerobot 0.5.1 — the same version pinned on the
DGX Spark finetune side).

Per frame (at --fps, default 30):
    observation.state        float32[4]  joint angles [slew, lift, tilt, scoop] deg
    observation.images.cam1  uint8 480x640x3  D435i infrared left imager (emitter OFF)
    observation.images.cam2  uint8 480x640x3  D435i color imager
    action                   float32[4]  normalized valve cmds actually sent [-1,1]
    task                     the natural-language instruction (--task)
    clock.loop               float64     seconds since episode start (perf_counter)
    clock.cam1_age           float32     seconds this cam1 frame had been sitting
    clock.cam2_age           float32     seconds this cam2 frame had been sitting
    clock.state_age          float32     seconds since the IMU frame behind the pose
    clock.imu_us             int64       Pico device clock, microseconds

The clock.* columns are diagnostics, not observations. lerobot's own
``timestamp`` is computed as ``frame_index / fps``, so it reports a flawless
33.33 ms whatever the loop actually did, and the camera and control threads both
sit behind latest-value caches this loop samples at its own rate. Without these
a repeated frame, a late tick and a clean recording all look identical
afterwards. Training ignores them: policies select their features by name.

Both imagers are always recorded, off one RealSense pipeline, so a cam1/cam2
pair shares a capture instant. They are separate sensors with separate exposure
controls: --exposure-ir and --exposure-rgb. Drop whichever camera a given
training run does not want when deriving its dataset, not at record time.

This loop runs at --fps because that is the dataset frame rate. It does NOT
drive the valves at that rate: the recorded action is handed to
ExcavatorController's direct-command mode as a setpoint, and its 100 Hz thread
writes the hardware. That is what makes the dataset valid for deployment —
recording and inference push setpoints through the identical fixed-rate writer,
so the action->motion mapping the policy learns is the one it will get back.
Driving I2C straight from this loop (the old behaviour) made that mapping
depend on whatever rate the loop happened to hit between video encodes.

The hydraulic pump is cut while an episode is being written and switched back on
when the write finishes, which is also the operator's signal that the board is
ready for the next take.

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
    ACTION_KEY, CAMERA_KEY, CAMERA_KEY_RGB, DEFAULT_STATE_JOINTS, JOINT_NAMES,
    STATE_KEY, MasiExcavator,
)
from lerobot_vla.camera import CameraConfig
from simple_drive import (
    BTN_A, BTN_B, BTN_X, BTN_Y, LocalGamepadInput,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data_collection" / "lerobot_datasets"


CLOCK_LOOP = "clock.loop"
CLOCK_CAM1_AGE = "clock.cam1_age"
CLOCK_CAM2_AGE = "clock.cam2_age"
CLOCK_STATE_AGE = "clock.state_age"
CLOCK_IMU_US = "clock.imu_us"


def clock_fields(obs: dict, tick: float, ep_perf0: float) -> dict:
    """The per-frame capture clocks, in dataset units.

    ``tick`` is this iteration's perf_counter, read straight after the
    observation: the ages then measure how stale the data already was when it
    was written, not how long the rest of the tick took.

    A source that has not reported yet gives NaN for its age. The device clock
    uses -1 instead because int64 has no NaN, and 0 is a real Pico timestamp.
    """
    def age(ts):
        return np.float32("nan") if ts is None else np.float32(tick - ts)

    imu_us = obs.get("imu_device_us")
    return {
        CLOCK_LOOP: np.array([tick - ep_perf0], dtype=np.float64),
        CLOCK_CAM1_AGE: np.array([age(obs.get("img_ts"))], dtype=np.float32),
        CLOCK_CAM2_AGE: np.array([age(obs.get("rgb_ts"))], dtype=np.float32),
        CLOCK_STATE_AGE: np.array([age(obs.get("state_ts"))], dtype=np.float32),
        CLOCK_IMU_US: np.array([-1 if imu_us is None else int(imu_us)], dtype=np.int64),
    }


def build_features(cam_h: int, cam_w: int,
                   state_joints: list[str] | None = None) -> dict:
    # `names` is the record of which joints observation.state carries — the
    # dataset documents its own state layout, so training and inference cannot
    # silently disagree about whether slew is in there.
    state_joints = list(state_joints or DEFAULT_STATE_JOINTS)
    video = {"dtype": "video", "shape": (cam_h, cam_w, 3),
             "names": ["height", "width", "channels"]}
    return {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (len(state_joints),),
            "names": state_joints,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        CAMERA_KEY: dict(video),
        CAMERA_KEY_RGB: dict(video),
        CLOCK_LOOP: {"dtype": "float64", "shape": (1,), "names": None},
        CLOCK_CAM1_AGE: {"dtype": "float32", "shape": (1,), "names": None},
        CLOCK_CAM2_AGE: {"dtype": "float32", "shape": (1,), "names": None},
        CLOCK_STATE_AGE: {"dtype": "float32", "shape": (1,), "names": None},
        CLOCK_IMU_US: {"dtype": "int64", "shape": (1,), "names": None},
    }


def unresumable_reason(root: Path) -> str | None:
    """Why ``root`` cannot be resumed, or None if it can.

    Worth checking here rather than letting lerobot raise: its
    LeRobotDatasetMetadata.__init__ catches FileNotFoundError from the local
    metadata load and falls back to *downloading* ``repo_id`` from the Hugging
    Face Hub. For a local-only dataset that turns "meta/tasks.parquet is
    missing" into a 401 RepositoryNotFoundError against huggingface.co, which
    names neither the missing file nor the real problem. These datasets never
    live on the Hub, so the failure should stay local and legible.

    A run that creates the dataset and exits before saving an episode leaves
    exactly the stub this detects: meta/info.json and nothing else.
    """
    if not root.exists():
        return f"{root} does not exist"
    missing = [rel for rel in ("meta/info.json", "meta/tasks.parquet")
               if not (root / rel).exists()]
    if missing:
        return (f"{root} holds no saved episodes "
                f"(missing {', '.join(missing)}) — an aborted run leaves this stub")
    return None


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
    p.add_argument("--repo-id", required=True,
                   help="Dataset repo id, e.g. masi/excavator_sand_v0. Also the "
                        "folder name under --root, with '/' replaced by '_'. "
                        "Required: there is no default, so a session can never "
                        "silently land in a leftover dataset.")
    p.add_argument("--task", required=True,
                   help='Instruction for the episodes, e.g. "scoop sand and dump it to the left"')
    p.add_argument("--root", default=None,
                   help=f"Dataset root dir (default {DEFAULT_ROOT}/<repo-id>)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--robot", default="auto", help="Board profile (auto = detect)")
    p.add_argument("--max-episode-s", type=float, default=120.0,
                   help="Auto-stop + save an episode after this long")
    p.add_argument("--no-slew", action="store_true",
                   help="Disable the slew ACTION channel (the machine will not slew)")
    p.add_argument("--state-joints", default=",".join(DEFAULT_STATE_JOINTS),
                   help="Joints recorded into observation.state. Slew is out by "
                        "default: nothing zeroes its yaw and there is no "
                        "magnetometer, so its origin is arbitrary across power "
                        "cycles. Actions stay 4-dim either way. "
                        f"Pass '{','.join(JOINT_NAMES)}' to record it anyway.")
    # Both imagers are always recorded: they ride one pipeline, the extra cost
    # is one video encode, and which one a policy should use is a training-time
    # question. Drop the unwanted camera when deriving the training dataset.
    p.add_argument("--exposure-ir", type=float, default=None,
                   help="Lock cam1 (IR) exposure, microseconds "
                        "(default: auto; find a value with tune_exposure.py)")
    p.add_argument("--gain-ir", type=float, default=None,
                   help="cam1 (IR) sensor gain 16..248 (only with --exposure-ir)")
    p.add_argument("--exposure-rgb", type=float, default=None,
                   help="Lock cam2 (RGB) exposure, microseconds. Separate sensor "
                        "from the IR one, so --exposure-ir does not apply to it")
    p.add_argument("--gain-rgb", type=float, default=None,
                   help="cam2 (RGB) sensor gain 0..128 (only with --exposure-rgb)")
    p.add_argument("--resume", action="store_true",
                   help="Append to an existing dataset instead of creating a new one")
    args = p.parse_args()

    root = Path(args.root) if args.root else DEFAULT_ROOT / args.repo_id.replace("/", "_")

    state_joints = [j.strip() for j in args.state_joints.split(",") if j.strip()]
    unknown = [j for j in state_joints if j not in JOINT_NAMES]
    if unknown:
        print(f"--state-joints {unknown} not in {JOINT_NAMES}"); return 1
    print(f"[state] observation.state = {state_joints}"
          + ("" if "slew" in state_joints else "   (slew excluded: unanchored yaw)"))

    cam_cfg = CameraConfig(width=640, height=480, fps=args.fps,
                             exposure_us=args.exposure_ir, gain=args.gain_ir,
                             enable_color=True,
                             color_exposure_us=args.exposure_rgb,
                             color_gain=args.gain_rgb)
    print(f"[camera] cam1 (IR) -> {CAMERA_KEY}")
    print(f"[camera] cam2 (RGB) -> {CAMERA_KEY_RGB}")
    for lbl, exp, flag in (("cam1 (IR)", args.exposure_ir, "--exposure-ir"),
                           ("cam2 (RGB)", args.exposure_rgb, "--exposure-rgb")):
        if not exp:
            print(f"[camera] {lbl}: AUTO exposure — lock it with {flag} "
                  f"for consistent training data")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    stub = unresumable_reason(root)
    if args.resume:
        if stub:
            print(f"Cannot --resume: {stub}.")
            if root.exists():
                print(f"There is no episode data to lose. Remove it and record "
                      f"without --resume:\n    rm -rf {root}")
            return 1
        dataset = LeRobotDataset.resume(args.repo_id, root=root, vcodec="h264",
                                        streaming_encoding=True)
        # Catch a schema change here rather than on the first add_frame. lerobot
        # validates every frame against the stored features, so appending to a
        # dataset recorded before the clock.* columns existed fails mid-episode
        # -- after the operator has already driven the take.
        want = set(build_features(cam_cfg.height, cam_cfg.width,
                                  state_joints=state_joints))
        have = set(dataset.meta.features) - {"timestamp", "frame_index",
                                             "episode_index", "index", "task_index"}
        if want != have:
            print(f"Cannot --resume: {root} was recorded with a different schema.")
            for label, diff in (("missing from it", want - have),
                                ("extra in it", have - want)):
                if diff:
                    print(f"  {label}: {', '.join(sorted(diff))}")
            print("Record into a new --repo-id; the two shapes cannot share a dataset.")
            return 1
        print(f"[dataset] Resumed {root} at episode {dataset.meta.total_episodes}")
    else:
        if root.exists():
            print(f"Dataset root {root} already exists. Use --resume to append, "
                  f"or remove it / pick another --repo-id.")
            if stub:
                print(f"({stub}, so removing it loses nothing:\n"
                      f"    rm -rf {root})")
            return 1
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=build_features(cam_cfg.height, cam_cfg.width,
                                    state_joints=state_joints),
            root=root,
            robot_type=MasiExcavator.robot_type,
            use_videos=True,
            vcodec="h264",
            # Without this lerobot writes a PNG per camera per frame, on this
            # thread: measured 46 ms each, so two cameras alone blow the 33.3 ms
            # budget and the loop settles at ~14.5 Hz while still stamping
            # timestamps as frame_index/fps. Every dataset recorded that way
            # claims 30 fps for motion that happened at half that, and replays
            # (and trains, and executes) about twice too fast. The streaming
            # encoder hands frames to a background encoder instead: 12 ms.
            streaming_encoding=True,
        )
        print(f"[dataset] Created {root}")

    saved_eps = dataset.meta.total_episodes

    def discard_if_empty() -> bool:
        """Drop a dataset this run created but never saved an episode into.

        record_episodes.py doubles as a way to just drive around, and every such
        run used to leave a metadata-only directory behind: not a resumable
        dataset (no meta/tasks.parquet) and enough to block the same --repo-id
        next time.

        Removing `root` is safe *because* --resume was not given: that path
        refuses to start when the directory already exists, so everything under
        it was created by this process moments ago. A --resume run points at
        data somebody recorded earlier and is never touched.
        """
        if saved_eps > 0 or args.resume:
            return False
        shutil.rmtree(root, ignore_errors=True)
        print(f"[dataset] no episodes saved — removed {root}")
        return True

    # ── hardware ─────────────────────────────────────────────────────────────
    # The gamepad is polled at --fps, so the setpoint hold has to cover a few
    # missed frames without decaying. 0.15 s ≈ 4 frames at 30 Hz: long enough to
    # ride out a video-encode hiccup, short enough that a dropped gamepad or a
    # wedged loop ramps the valves down in well under a second.
    robot = MasiExcavator(profile=args.robot,
                          camera_config=cam_cfg,
                          enable_slew=not args.no_slew,
                          use_control_thread=True,
                          setpoint_hold_s=max(0.1, 4.0 / args.fps),
                          setpoint_decay_s=0.2,
                          state_joints=state_joints)
    try:
        robot.connect()
    except BaseException:
        # connect() can fail part-way — hardware and the 100 Hz control thread
        # up, camera not — so bring it down explicitly instead of leaving the
        # valves owned by a thread nobody is feeding until the process exits.
        try:
            robot.disconnect()
        except Exception:
            pass
        discard_if_empty()
        raise

    pad = LocalGamepadInput()
    if not pad.open():
        robot.disconnect()
        discard_if_empty()
        return 1

    print(f"\nTask: {args.task!r}")
    print("A=start/save episode  B=discard episode  X=pump  Y=reload-config  Ctrl+C=quit\n")

    period = 1.0 / args.fps
    # Capture time of the last frame this loop consumed. The loop paces on the
    # camera publishing a frame newer than this, rather than on its own sleep.
    last_cam_ts = 0.0
    stalls = 0
    mask_prev = 0
    recording = False
    ep_start = 0.0
    ep_perf0 = 0.0   # same instant as ep_start, on the monotonic clock the ages use
    ep_frames = 0
    last_status = time.time()

    def stop_episode(save: bool, reason: str):
        nonlocal recording, ep_frames
        recording = False
        robot.stop_motion()
        if save and ep_frames > 0:
            # Pump off across the save, back on the moment it finishes.
            # save_episode() blocks this loop for seconds encoding video with the
            # valves already zeroed by stop_motion, so the pump would only
            # dead-head against closed valves. Restoring it right afterwards
            # doubles as the operator's cue that the board has finished writing
            # and the next episode can start.
            #
            # Only on the save path: a discard (B) encodes nothing, so there is
            # nothing to wait out.
            pump_was_on = robot.pump_enabled
            if pump_was_on:
                robot.set_pump(False)
                print("\n[pump] OFF (saving)")
            # The rate this episode actually achieved, against the one the
            # dataset is about to assert. lerobot stamps `timestamp` as
            # frame_index / fps no matter what the loop managed, so a loop that
            # cannot hold --fps produces a dataset claiming a duration it never
            # took: it plays back fast, trains on a compressed timebase, and
            # run_inference commands the machine by that same wrong factor.
            # Nothing downstream can detect it, so it has to be said here.
            elapsed = time.perf_counter() - ep_perf0
            achieved = ep_frames / elapsed if elapsed > 0 else 0.0
            if achieved < args.fps * 0.98:
                print(f"\n*** RATE: recorded {achieved:.1f} Hz, dataset will claim "
                      f"{args.fps} — motion will read {args.fps / max(achieved, 1e-6):.2f}x "
                      f"too fast ***")
                print(f"*** Re-record at --fps {int(achieved)} or find what is "
                      f"stalling the loop. clock.loop vs timestamp shows it. ***")
            print(f"\n[EP] saving {ep_frames} frames "
                  f"({elapsed:.1f}s wall, {achieved:.1f} Hz) "
                  f"({reason})... ", end="", flush=True)
            dataset.save_episode()
            print("done.")
            if pump_was_on:
                robot.set_pump(True)
                print("[pump] ON (save complete — ready to record)")
        else:
            dataset.clear_episode_buffer()
            print(f"\n[EP] discarded ({reason}).")
        ep_frames = 0

    try:
        while True:
            # Pace on the camera, not on a sleep. A 30.000 Hz sleep loop is a
            # second clock beating against a sensor that actually delivers at
            # ~29.976 Hz: measured over one 35 s episode the frame age walked
            # 22 ms -> 5 ms, and continuing that slide means alternately
            # reusing and skipping frames. Waiting for the frame removes the
            # second clock instead of correcting for it.
            #
            # On timeout the loop still runs: the gamepad has to be polled and
            # the setpoint fed (or visibly stop, so the valves decay) rather
            # than block behind a stalled USB pipe. Nothing is recorded for a
            # tick with no new frame -- a duplicate image would be a worse lie
            # than a short gap, and the gap shows up in the achieved rate.
            cam_ts = robot.wait_for_next_frame(last_cam_ts, timeout_s=4 * period)
            fresh = cam_ts > 0.0
            if fresh:
                last_cam_ts = cam_ts
            elif recording:
                stalls += 1
                if stalls % 30 == 1:
                    print(f"\n*** CAMERA: no new frame for {4 * period * 1000:.0f} ms "
                          f"({stalls} skipped this run) ***")

            axes, mask = pad.poll()
            if axes is not None:
                def btn(b): return bool(mask & (1 << b))
                def prev(b): return bool(mask_prev & (1 << b))

                if btn(BTN_A) and not prev(BTN_A):
                    if not recording:
                        recording = True
                        ep_start = time.time()
                        ep_perf0 = time.perf_counter()
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

            if recording and fresh:
                obs = robot.get_observation()
                tick = time.perf_counter()
                # Every declared camera must be present in the frame, so a
                # missing cam2 drops the whole frame rather than desynchronizing
                # the two video streams against the parquet rows.
                cam_keys = [CAMERA_KEY, CAMERA_KEY_RGB]
                if all(obs.get(k) is not None for k in cam_keys):
                    frame = {
                        STATE_KEY: obs[STATE_KEY],
                        ACTION_KEY: sent,
                        "task": args.task,
                    }
                    frame.update({k: obs[k] for k in cam_keys})
                    frame.update(clock_fields(obs, tick, ep_perf0))
                    dataset.add_frame(frame)
                    ep_frames += 1

            # Outside the fresh-frame gate on purpose: a camera that stalls mid
            # take must not also disable the auto-stop and leave the operator
            # recording into a dead stream.
            if recording and time.time() - ep_start >= args.max_episode_s:
                stop_episode(save=True, reason=f"{args.max_episode_s:.0f}s limit")
                saved_eps += 1

            now = time.time()
            if now - last_status >= 5.0:
                last_status = now
                ja, _, _ = robot.get_joint_angles()
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

            # No sleep here: wait_for_next_frame() at the top of the loop is
            # what paces it, and it blocks until the camera has something new.

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if recording and ep_frames > 0:
            stop_episode(save=True, reason="shutdown")
            saved_eps += 1
        pad.close()
        robot.disconnect()
        if not discard_if_empty():
            print("[dataset] finalizing...")
            try:
                dataset.finalize()
            except Exception as exc:
                print(f"[dataset] finalize failed: {exc}")
            print(f"[dataset] {saved_eps} episodes total in {root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
