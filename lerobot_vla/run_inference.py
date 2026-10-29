#!/usr/bin/env python3
"""X-VLA split-engine inference for the MASI excavator.

--synthetic checks the model only; default mode reads real observations and
prints actions; --live sends chunks to the existing 100 Hz Python controller.
Camera order, action rate, task and normalization come from the X-VLA bundle.
Dual IR capture uses the same identities and orientation as record_episodes.
A toggles manual takeover during live inference; D-pad cycles trained tasks.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lerobot_vla.runtime.action_log import ActionLogger
# Control-layer channel names. Column-aligned with JOINT_NAMES (the dataset
# order), and the keys the setpoint status dict uses — so an action-log
# column means the same thing in the chunk stream and the emitted stream.
from lerobot_vla.excavator_robot import JOINT_NAMES, action_names
from lerobot_vla.excavator_robot import _CONTROL_CHANNELS as CONTROL_CHANNELS
from lerobot_vla.excavator_robot import _TRACK_CHANNELS as TRACK_CHANNELS
from lerobot_vla.record_episodes import manual_action_from_axes
from lerobot_vla.runtime import xvla
from lerobot_vla.policy import detect_architecture, make_policy
from lerobot_vla.gamepad import BTN_A, BTN_DPAD_LEFT, BTN_DPAD_RIGHT, LocalGamepadInput

LOG = logging.getLogger("run_inference")

#: --camera choice -> the dataset image key it selects.
CAMERA_KEYS = {"ir": "observation.images.cam1", "cam1": "observation.images.cam1",
               "cam2": "observation.images.cam2"}

#: Rate the manual setpoint is fed at while the operator drives. Deliberately
#: NOT --fps: that is the checkpoint's action rate, and a 6 fps checkpoint would
#: leave the setpoint stale between ticks and decay the valves mid-stroke.
TELEOP_HZ = 30.0

#: Button poll rate. Fast enough that a tap is never missed, and it only reads.
BUTTON_POLL_HZ = 50.0

#: How long to wait for a pad at startup. Short, because it is optional.
GAMEPAD_TIMEOUT_S = 3.0


def check_action_width(width: int, enable_tracks: bool, source: str) -> list[str]:
    """The checkpoint's action layout, refused if the run cannot use it.

    --enable-tracks decides only whether the tracks are DRIVEN. A 6-wide
    checkpoint runs either way -- without the flag its track columns are simply
    not sent -- but the flag on a 4-wide one is asking for columns the model
    does not have.
    """
    for names in (action_names(False), action_names(True)):
        if width == len(names):
            break
    else:
        raise SystemExit(
            f"{source} has a {width}-dim action, which is neither "
            f"{action_names(False)} nor {action_names(True)}. Check the training "
            f"dataset's meta/info.json -> features['action']['names'].")
    if enable_tracks and names != action_names(True):
        raise SystemExit(
            f"--enable-tracks refused: {source} has a {width}-dim action "
            f"{names}, with no track columns to drive. Drop the flag.")
    return names


class GamepadTeleop:
    """The pad's two jobs in a run: hand the machine over, and pick the task.

    A toggles control: press it while the policy is driving and the operator takes
    over until it is pressed again. Useful for repositioning between takes, and
    for rescuing a run that has driven itself somewhere useless.

    The two sources are exclusive, not summed. That is what makes the hand-over
    clean in both directions. Manual control goes out through ``send_action``,
    whose ``set_point`` DROPS the chunk the control thread is playing
    (modules/setpoint_schedule.py), so the operator never fights a stale plan;
    and because no inference runs while they drive, the chunk that resumes
    control was inferred from the pose they left the machine in rather than
    from one taken before they touched it.

    D-pad right/left cycles the active instruction when the run carries more than
    one (see resolve_tasks). That is a selector, not a command: it changes the
    string the NEXT inference conditions on and never writes a valve itself, which
    is why it is offered in a dry run too — there ``allow_takeover`` is False and
    A does nothing. Taking the machine over stays live-only, because the sticks
    DO write setpoints and the safety ladder in the module docstring must not have
    a hole in it for the pad.

    The pad is polled on its own thread only to catch the buttons. The inference
    loop comes around once per chunk (~0.5 s), and the pad holds a button's
    state only until its release event arrives, so a tap read from that loop is
    simply missed. That thread never commands the valves.
    """

    def __init__(self, robot, pad, allow_takeover: bool = True) -> None:
        self._robot = robot
        self._pad = pad
        self._allow_takeover = allow_takeover
        self._edge = threading.Event()
        self._task_step = 0                 # net D-pad clicks not yet acted on
        self._task_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch_button, daemon=True)
        self._thread.start()

    def _watch_button(self) -> None:
        prev = 0
        while not self._stop.wait(1.0 / BUTTON_POLL_HZ):
            _, mask = self._pad.poll()
            hit = mask & ~prev              # rising edges only
            prev = mask
            if self._allow_takeover and hit & (1 << BTN_A):
                self._edge.set()
            step = (bool(hit & (1 << BTN_DPAD_RIGHT))
                    - bool(hit & (1 << BTN_DPAD_LEFT)))
            if step:
                with self._task_lock:
                    self._task_step += step

    def pressed(self) -> bool:
        """True once per press of A, whenever it is next asked for."""
        if self._edge.is_set():
            self._edge.clear()
            return True
        return False

    def take_task_step(self) -> int:
        """Net D-pad clicks since this was last asked, and clear them.

        Net rather than one per call: the loop asks once per chunk, so two clicks
        landing inside one inference should move two tasks along rather than queue
        the second switch behind the next chunk.
        """
        with self._task_lock:
            step, self._task_step = self._task_step, 0
        return step

    def run(self) -> None:
        """Drive from the sticks until A is pressed again. Blocks the caller.

        ``stop_motion`` on entry zeroes the valves and drops the policy chunk in
        the same call, so control transfers on a stopped machine rather than
        mid-stroke. On exit it zeroes again: the policy needs one inference
        (~0.4 s) before it has anything to say, and the operator's last stick
        value must not coast through that gap.

        A pad that disconnects mid-teleop leaves the machine stopped -- read()
        zeroes every axis while it is gone -- and control stays manual until it
        is back. Resuming the policy on its own here would start autonomous
        motion on a machine nobody is holding.
        """
        self._robot.stop_motion()
        period = 1.0 / TELEOP_HZ
        # The robot's layout, not a flag of our own: send_action refuses a vector
        # of any other width, so the sticks cannot disagree with the policy about it.
        tracks = self._robot.enable_tracks
        while not self.pressed():
            axes, _ = self._pad.poll()
            action = (manual_action_from_axes(axes, tracks)
                      if axes is not None and self._pad.is_live()
                      else np.zeros(len(self._robot.action_names), dtype=np.float32))
            self._robot.send_action(action)
            time.sleep(period)
        self._robot.stop_motion()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._pad.close()


def make_teleop(robot, allow_takeover: bool = True) -> GamepadTeleop | None:
    """The run's pad, or None if no pad answers.

    Optional on purpose: a pad that is absent, unplugged or unreadable leaves
    the run exactly as it would have been rather than failing it, so an
    inference session never depends on one being connected.
    """
    pad = LocalGamepadInput(connect_timeout_s=GAMEPAD_TIMEOUT_S)
    if not pad.open():
        return None
    return GamepadTeleop(robot, pad, allow_takeover=allow_takeover)


def main() -> int:
    p = argparse.ArgumentParser(description="X-VLA split-engine inference loop.")
    p.add_argument("--split-dir", required=True, help="Fine-tuned X-VLA export bundle directory")
    p.add_argument("--tokenizer", default=None, help="Override <split-dir>/tokenizer")
    p.add_argument("--cache-dir", default=None, help="Override <split-dir>/trt_native_cache")
    p.add_argument("--trt-chain", choices=("graph", "device", "host"), default="graph",
                   help="TensorRT execution; graph is fastest, host is for diagnostics")
    p.add_argument("--rebuild", action="store_true",
                   help="Wipe this bundle's engine cache and rebuild it "
                        "(minutes). The cache is dropped automatically when the "
                        "bundle's weights change on disk, so this is only for a "
                        "cache you suspect rather than one you changed.")
    p.add_argument("--task", action="append", dest="tasks", default=None,
                   metavar="INSTRUCTION",
                   help="Instruction the policy is conditioned on, e.g. "
                        '"scoop sand and dump it to the left". Same flag, and the '
                        "same string, as record_episodes --task. Normally omitted "
                        "— it is read from the export bundle, since the phrasing "
                        "is a property of the checkpoint. There is NO default: a "
                        "wrong instruction is out of distribution and silent. "
                        "REPEAT the flag for a multi-task checkpoint: the run "
                        "starts on the first and the pad's D-pad cycles the rest")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Denoise steps. Left alone, an X-VLA bundle uses the "
                        "num_denoising_steps it was exported with")
    p.add_argument("--min-replan-s", type=float, default=0.0,
                   help="Minimum seconds between chunk hand-overs; 0 (default) "
                        "replans as fast as inference allows. A THROTTLE, not an "
                        "execution limit -- the full chunk is always handed to the "
                        "scheduler, so what actually plays is set by inference "
                        "latency (~infer_s * fps), not by this flag")
    p.add_argument("--fps", type=float, default=None,
                   help="Action execution rate. Normally omitted — it is read from "
                        "the X-VLA bundle; no default is assumed")
    p.add_argument("--state-blind", action="store_true",
                   help="Explicitly feed zero state for a camera-only checkpoint")
    p.add_argument("--allow-base-bundle", action="store_true",
                   help="Load an X-VLA bundle that carries no physical boundary "
                        "(the base ee6d checkpoint, or an export missing its "
                        "normalization stats). Its action columns are arm "
                        "dimensions, NOT valve commands, so the run is "
                        "model-only: --live is refused. Use this only for "
                        "model/engine diagnostics")
    p.add_argument("--robot", default="auto")
    p.add_argument("--state-joints", default="lift,tilt,scoop",
                   help="Joints fed to the policy as observation.state. MUST "
                        "match what the checkpoint was trained on — checked "
                        "against the X-VLA processor contract. Slew is out "
                        "by default (unanchored yaw; see excavator_robot.py)")
    from lerobot_vla.camera import add_camera_arguments, config_from_args, fresh_observation
    add_camera_arguments(p)
    p.add_argument("--camera", choices=("auto", "ir", "cam1", "cam2", "both"), default="auto",
                   help="auto uses checkpoint view order; both selects wrist + cabin IR")
    p.add_argument("--synthetic", action="store_true",
                   help="No robot/camera; synthetic observation (model-only test)")
    p.add_argument("--live", action="store_true",
                   help="SEND actions to the valves (default: print only)")
    p.add_argument("--no-gamepad", action="store_true",
                   help="Do not look for a gamepad. Without it, an attached pad "
                        "can take manual control with A during a --live run "
                        "(exclusive with the policy, not summed) and cycle the "
                        "task with the D-pad when the run carries more than one; "
                        "no pad attached is not an error either way")
    p.add_argument("--loops", type=int, default=0,
                   help="Stop after N infer+execute cycles (0 = run until Ctrl+C)")
    p.add_argument("--seed", type=int, default=None, help="Fix the denoise noise seed")
    p.add_argument("--setpoint-hold-s", type=float, default=0.25,
                   help="How long a chunk's last action holds full authority "
                        "after the chunk runs out, before decaying to zero")
    p.add_argument("--setpoint-decay-s", type=float, default=0.25,
                   help="Ramp-to-zero window once the setpoint goes stale")
    p.add_argument("--blend-s", type=float, default=0.0,
                   help="Cross-fade into each new chunk, to soften the step at "
                        "a chunk boundary (0 = apply immediately)")
    p.add_argument("--lag-compensation", default="off", metavar="off|auto|SECONDS",
                   help="Skip the part of each chunk that is already in the past "
                        "by the time it is handed over. The chunk is planned for "
                        "the world as it looked at the OBSERVATION, but playback "
                        "starts at hand-over, one inference later -- so index 0 is "
                        "executed ~infer_s late and the machine follows commands "
                        "meant for a frame it has already left behind. "
                        "'auto' (recommended) times the chunk from the observation, "
                        "so playback starts at the step whose timestamp is now and "
                        "self-corrects as inference time varies; a float forces a "
                        "fixed compensation in seconds; 'off' (default) is the "
                        "historical behaviour. NOTE this consumes the chunk from "
                        "both ends -- it needs chunk_size >= 2 * infer_s * fps "
                        "before any reserve is left for a late replan")
    p.add_argument("--enable-tracks", action="store_true",
                   help="Drive the tracks from the checkpoint's [trackL, "
                        "trackR] columns, and from the triggers/bumpers during "
                        "gamepad takeover, as in record_episodes. Refused for a "
                        "4-wide checkpoint. Without it a 6-wide checkpoint still "
                        "runs, with its track columns logged but not driven")
    p.add_argument("--legacy-direct-write", action="store_true",
                   help="Drive valves from this thread instead of the control "
                        "thread (bench A/B only — has the 30 Hz rate problems)")
    p.add_argument("--log-actions", metavar="PATH", nargs="?", const=".",
                   default=None,
                   help="Record every inference to a JSONL file: the whole "
                        "action chunk, the state that produced it and the "
                        "timing. Bare --log-actions writes "
                        "infer_<timestamp>.jsonl into the current directory; "
                        "any directory given instead gets the timestamped file, "
                        "and a filename is used as-is. Works in dry-run too — "
                        "that is the safe way to collect the command "
                        "distribution for a position. Analyse with "
                        "tools/analyze_action_log.py")
    p.add_argument("--log-emitted-hz", type=float, default=100.0,
                   help="With --log-actions on a real robot, also poll the "
                        "setpoint the control thread is writing at this rate. "
                        "This is the stream the valves see — interpolation, "
                        "chunk cross-fade and stale-decay all make small "
                        "values that never appear in the chunk itself. The "
                        "default matches the control thread; polling is "
                        "unsynchronized, so measured spool travel is a floor, "
                        "not an exact figure. 0 disables")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if detect_architecture(args.split_dir) != "xvla":
        raise SystemExit("This checkout runs X-VLA bundles only.")
    selected_cameras = (["observation.images.cam1", "observation.images.cam2"]
                        if args.camera == "both" else
                        None if args.camera == "auto" else [CAMERA_KEYS[args.camera]])
    camera_keys = xvla.resolve_cameras(args.split_dir, selected_cameras,
        allow_unknown=args.synthetic and args.allow_base_bundle)
    act_names = action_names(args.enable_tracks)
    state_joints = [j.strip() for j in args.state_joints.split(",") if j.strip()]
    unknown = [j for j in state_joints if j not in JOINT_NAMES]
    if unknown:
        raise SystemExit(f"--state-joints {unknown} not in {JOINT_NAMES}")

    # --lag-compensation resolves to None (off), "auto", or a float of seconds.
    # Done here, with the other fail-fast checks, so a typo costs a second rather
    # than surfacing after a multi-minute TRT build with the machine armed.
    _lag = str(args.lag_compensation).strip().lower()
    if _lag in ("off", "none", "false"):
        args.lag_comp = None
    elif _lag == "auto":
        args.lag_comp = "auto"
    else:
        try:
            args.lag_comp = float(_lag)
        except ValueError:
            raise SystemExit("--lag-compensation wants 'off', 'auto' or seconds, "
                             f"got {args.lag_compensation!r}")
        if args.lag_comp < 0:
            raise SystemExit("--lag-compensation seconds must be >= 0, got "
                             f"{args.lag_comp}")
        if args.lag_comp == 0:
            args.lag_comp = None

    if args.live and args.allow_base_bundle:
        raise SystemExit("--live cannot be combined with --allow-base-bundle.")
    tasks = xvla.resolve_tasks(args.split_dir, args.tasks)
    args.fps = xvla.resolve_fps(args.split_dir, args.fps)
    state_blind = args.state_blind
    policy = make_policy(
        "xvla", args.split_dir,
        cache_dir=args.cache_dir,
        rebuild=args.rebuild,
        num_steps=args.num_steps,
        seed=args.seed,
        tokenizer_dir=args.tokenizer,
        state_joints=state_joints,
        allow_base_bundle=args.allow_base_bundle,
        chain=args.trt_chain,
    )

    # A bundle with no physical boundary emits arm dimensions, not valve commands
    # (see runtime/xvla.py). It is loadable for model/engine diagnostics, but it
    # never drives.
    if getattr(policy, "feasibility_only", False) and args.live:
        raise SystemExit(
            "--live refused: this bundle is a model-only feasibility export whose "
            "action columns are not valve commands. Drop --live to watch it run "
            "against real observations.")
    # X-VLA's width comes from the bundle's physical boundary, so it can only be
    # checked once the policy is built. A feasibility bundle's columns are arm
    # dimensions with no layout to check; it never drives.
    model_names = (act_names if policy.feasibility_only else
                   check_action_width(policy.action_dim, args.enable_tracks,
                                      f"The bundle {args.split_dir}"))
    ignored_tracks = len(model_names) > len(act_names) and not getattr(
        policy, "feasibility_only", False)
    if ignored_tracks:
        LOG.info("The checkpoint also emits %s; without --enable-tracks they are "
                 "logged but not driven, so the tracks stay stopped.",
                 model_names[len(act_names):])

    robot = None
    if not args.synthetic:
        from lerobot_vla.excavator_robot import MasiExcavator
        robot = MasiExcavator(
            profile=args.robot,
            camera_config=config_from_args(args,
                enable_cam2="observation.images.cam2" in camera_keys),
            use_control_thread=not args.legacy_direct_write,
            setpoint_hold_s=args.setpoint_hold_s,
            setpoint_decay_s=args.setpoint_decay_s,
            setpoint_blend_s=args.blend_s,
            state_joints=state_joints,
            enable_tracks=args.enable_tracks)
        try:
            robot.connect()
        except BaseException:
            robot.disconnect()
            policy.close()
            raise
        if args.live:
            LOG.warning("LIVE MODE: actions will drive the valves. Pump is under "
                        "your control (it is NOT auto-enabled).")
        else:
            LOG.info("Dry run: observations are real, actions are printed only.")

    # Takeover is live only: without --live no source drives the valves, and the
    # safety ladder in the module docstring should not have a hole in it for the
    # pad. A multi-task run still opens the pad without --live, with takeover off
    # -- the D-pad only picks the instruction the next inference conditions on, so
    # it writes no valve, and it is how a two-task bundle gets checked on the
    # bench before anyone stands next to a live machine.
    takeover = bool(args.live) and robot is not None
    teleop = (make_teleop(robot, allow_takeover=takeover)
              if not args.no_gamepad and (takeover or len(tasks) > 1)
              else None)
    if teleop is not None:
        LOG.info("Gamepad attached: %s%s.",
                 "A hands control between the policy and the sticks (exclusive, "
                 "not summed)" if takeover
                 else "takeover is live-only, so A does nothing in this mode",
                 "; D-pad right/left cycles the task" if len(tasks) > 1 else "")
    elif len(tasks) > 1 and not args.no_gamepad:
        LOG.warning("%d tasks are loaded but no gamepad answered, so this run "
                    "stays on %r for its whole length. --task <instruction> puts "
                    "another one first.", len(tasks), tasks[0])

    if not args.synthetic:
        LOG.info("Policy IR cameras in checkpoint order: %s", camera_keys)
    LOG.info("observation.state = %s%s", state_joints,
             "" if "slew" in state_joints else "   (slew excluded: unanchored yaw)")

    def get_observation():
        if robot is not None:
            obs = fresh_observation(robot.get_observation, camera_keys)
            images = [obs[key] for key in camera_keys]
            return images, obs["observation.state"]
        img = (np.random.default_rng(0).integers(0, 255, (480, 640, 3))
               .astype(np.uint8))
        images = [img.copy() for _ in camera_keys]
        return images, np.zeros(len(state_joints), dtype=np.float32)

    def for_policy(state):
        """What the policy actually receives. A camera-only checkpoint gets zeros;
        the real reading is still logged, so the IMU stays visible for diagnostics."""
        return np.zeros_like(state) if state_blind else state

    # Opened before the warmup so the meta record carries the resolved settings
    # even if the warmup is what blows up.
    action_log = None
    if args.log_actions:
        # The model's layout, not the driven one: an ignored track column is still
        # what the policy asked for, and its emitted value reads 0 -- which is what
        # the tracks got.
        channels = CONTROL_CHANNELS + (TRACK_CHANNELS if len(model_names) > len(CONTROL_CHANNELS)
                                       else [])
        action_log = ActionLogger(args.log_actions, channels, meta={
            "split_dir": args.split_dir,
            "task": tasks[0],
            "tasks": list(tasks),
            "fps": args.fps,
            "live": bool(args.live),
            "synthetic": bool(args.synthetic),
            "camera": args.camera,
            "camera_keys": camera_keys,
            "state_joints": list(state_joints),
            "state_blind": bool(state_blind),
            "min_replan_s": args.min_replan_s,
            "setpoint_hold_s": args.setpoint_hold_s,
            "setpoint_decay_s": args.setpoint_decay_s,
            "blend_s": args.blend_s,
            "robot_profile": None if robot is None else args.robot,
        })
        LOG.info("Logging actions to %s", action_log.path)
        if robot is not None and args.log_emitted_hz > 0:
            action_log.start_emitted_sampler(robot.get_setpoint_status,
                                             args.log_emitted_hz)

    # Warmup / engine build happens on the first sample_actions call.
    LOG.info("Warmup inference (builds TRT engines on first ever run)...")
    try:
        img, state = get_observation()
        t0 = time.perf_counter()
        policy.sample_actions(img, tasks[0], for_policy(state))
        LOG.info("Warmup done in %.1fs", time.perf_counter() - t0)
    except BaseException:
        if action_log is not None:
            action_log.close()
        if teleop is not None:
            teleop.close()
        if robot is not None:
            robot.disconnect()
        policy.close()
        raise

    cycle = 0
    # Index into `tasks`. Only the D-pad moves it, and only between chunks: the
    # string is an input to inference like the image and the state, so switching
    # is a one-line change of what the next prefix says. The language embedding is
    # cached per string inside the policy, so the first use of each instruction
    # costs one text-encoder run (~ms) and every later one costs nothing.
    task_index = 0
    # Start re-inference this far before the current chunk ends, so the next one
    # is ready when it runs out. Tracked as a decaying max of measured inference
    # time; the first cycle sets it from its own measurement. Not seeded from
    # the warmup — that one includes the TRT engine build.
    infer_lead_s = 0.0
    lag_mode = args.lag_comp          # resolved and validated in parse_args
    if lag_mode is not None:
        LOG.info("lag compensation: %s -- playback starts at the chunk step whose "
                 "timestamp is now, not at index 0",
                 "auto (measured per cycle)" if lag_mode == "auto" else f"{lag_mode:.3f}s fixed")
    chunk_t0 = 0.0          # set on every hand-over; only read once cycle > 0
    # When a chunk stopped playing mid-play -- teleop took the machine over, or
    # the operator switched task. `played` counts up to this instead of to the
    # next hand-over, so the stopped stretch is not billed to the chunk.
    chunk_cut = None
    try:
        while True:
            img, state = get_observation()
            # Monotonic clock on purpose: SetpointSchedule times chunks with
            # time.monotonic(), and perf_counter has a different epoch.
            t_obs = time.monotonic()
            t0 = time.perf_counter()
            chunk = policy.sample_actions(img, tasks[task_index], for_policy(state))
            infer_s = time.perf_counter() - t0
            infer_lead_s = max(infer_s, infer_lead_s * 0.9)

            # Hand over the FULL chunk. --min-replan-s only gates when the next
            # inference may START; it never truncates what is executed. The part
            # of the chunk that has not played when the next one lands is the
            # late-inference fallback: a slow replan keeps following the
            # predicted plan instead of falling into hold->decay.
            # With compensation the chunk is timestamped from the observation, so
            # the scheduler starts at the step that is due now instead of replaying
            # the ~infer_s of plan that expired while we were computing it.
            if lag_mode is None:
                play_t0 = None
            elif lag_mode == "auto":
                play_t0 = t_obs
            else:
                play_t0 = time.monotonic() - lag_mode
            skipped = 0.0 if play_t0 is None else max(
                0.0, (time.monotonic() - play_t0) * args.fps)
            if robot is not None and args.live:
                # Only the driven columns: without --enable-tracks the robot has
                # no track channel, and send_action_chunk refuses a wider chunk.
                robot.send_action_chunk(chunk[:, :len(act_names)], fps=args.fps,
                                        t0=play_t0)
            now = time.perf_counter()
            # Steps of the PREVIOUS chunk that actually played before this one
            # replaced it -- measured, not assumed. This is the real execution
            # horizon; with --min-replan-s 0 it is inference latency alone.
            played_until = chunk_cut if chunk_cut is not None else now
            played = (min(round((played_until - chunk_t0) * args.fps), len(chunk))
                      if cycle else None)
            chunk_t0 = now
            chunk_cut = None

            LOG.info("cycle=%d%s infer=%.0fms%s played=%s state%s=[%s] a0=[%s]%s",
                     cycle,
                     f" task={task_index}" if len(tasks) > 1 else "",
                     infer_s * 1000.0,
                     "" if lag_mode is None else f" skip={skipped:.1f}",
                     played if played is not None else "-",
                     "(unused)" if state_blind else "",
                     " ".join(f"{v:+.1f}" for v in state),
                     " ".join(f"{v:+.2f}" for v in chunk[0][:len(act_names)]),
                     (" tracks(ignored)=[%s]" % " ".join(
                         f"{v:+.2f}" for v in chunk[0][len(act_names):])
                      if ignored_tracks else
                      f" (+{chunk.shape[1] - len(act_names)} more columns; "
                      "not valve commands)"
                      if chunk.shape[1] > len(act_names) else ""))

            if action_log is not None:
                action_log.log_chunk(cycle, infer_s, played, state, chunk)

            cycle += 1
            if args.loops and cycle >= args.loops:
                break

            # Blocks here for as long as the operator drives. The chunk handed
            # over above is dropped by the first manual setpoint, so record when
            # it stopped playing before giving the machine away.
            if teleop is not None and teleop.pressed():
                chunk_cut = time.perf_counter()
                if action_log is not None:
                    action_log.log_event("teleop_start", cycle=cycle)
                teleop.run()
                if action_log is not None:
                    action_log.log_event("teleop_end", cycle=cycle)

            # A task switch lands on the NEXT inference rather than at the end of
            # the current chunk: the operator pressed the D-pad to change what the
            # machine is doing, and a chunk planned under the old instruction is
            # exactly what they no longer want. So it is dropped the way a teleop
            # hand-over drops it -- stopped machine, then a fresh plan -- and the
            # replan throttle is skipped, since the whole point of the press is
            # that the next chunk should be for the new task.
            switched = False
            if teleop is not None and len(tasks) > 1:
                step = teleop.take_task_step()
                if step:
                    task_index = (task_index + step) % len(tasks)
                    switched = True
                    LOG.info("Task -> [%d/%d] %r", task_index + 1, len(tasks),
                             tasks[task_index])
                    if robot is not None and args.live:
                        robot.stop_motion()
                        # Same accounting as the teleop hand-over: the chunk
                        # stopped playing here, so `played` is billed to here
                        # rather than to the next hand-over.
                        if chunk_cut is None:
                            chunk_cut = time.perf_counter()
                    if action_log is not None:
                        action_log.log_event("task_switch", cycle=cycle,
                                             index=task_index,
                                             task=tasks[task_index])

            # Sleep until it is time to start the next inference. The control
            # thread is driving the valves from the chunk meanwhile. The default
            # --min-replan-s 0 means no sleep at all: replan flat out.
            sleep_s = args.min_replan_s - infer_lead_s - (time.perf_counter() - chunk_t0)
            if sleep_s > 0 and not switched:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if action_log is not None:
            action_log.close()
            LOG.info("Action log written: %s", action_log.path)
        if teleop is not None:
            teleop.close()
        if robot is not None:
            try:
                robot.stop_motion()
            except Exception:
                pass
            robot.disconnect()
        policy.close()

    LOG.info("Done: %d cycles.", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
