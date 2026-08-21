#!/usr/bin/env python3
"""SmolVLA inference loop for the MASI excavator (split TRT engines).

    IR cam1 + joint angles + task -> SmolVLA (split ONNX/TRT) -> valves

Modes (safety ladder):
    --synthetic     no robot, no camera — synthetic image + zero state. Proves
                    the model pipeline + engine builds. Safe anywhere.
    (default)       robot connected, observations real, actions PRINTED ONLY.
    --live          actions are actually sent to the valves. Base (un-finetuned)
                    weights produce garbage actions — do not use --live until a
                    finetuned checkpoint is in place and the pump is your call.

The policy emits a 50-step action chunk per observation and the WHOLE chunk is
handed to the controller as a trajectory at --fps. Re-inference starts as soon
as the previous one finishes, unless --min-replan-s throttles it. Whatever part
of the chunk has not played when the next one lands is simply replaced, so the
EXECUTED horizon is set by inference latency (~infer_s * fps), not by any flag:
at ~0.4 s inference and 30 Hz that is ~12 steps. The unplayed tail is the
late-inference fallback — a slow replan keeps following the predicted plan
instead of degrading into hold->decay-to-zero.

--fps is normally NOT passed. The chunk is rate commands sampled at the rate the
checkpoint was trained on, so the playback rate belongs to the checkpoint: it is
read from the export bundle (export_info.json / info.json). Checkpoints trained
on decimated data (10 or 6 fps) played back at 30 Hz move the machine at a third
or a fifth of the intended speed, silently. An explicit --fps that contradicts
the bundle is refused.

The chunk is NOT replayed step-by-step from this thread. It is handed to
ExcavatorController's direct-command mode, whose 100 Hz control thread indexes
into it by elapsed time and interpolates. This thread only has to produce the
next chunk before the current one runs out — inference being far slower than
the valve update rate is exactly what that split is for. Re-inference starts
early by the measured inference time, so the schedule does not go stale in the
gap; if it does anyway, the held command decays to zero rather than latching.

--task is the same flag, and must carry the same string, as
record_episodes.py --task: the policy conditions on that language embedding, so
a phrasing the checkpoint was not finetuned on is out of distribution and
nothing downstream notices. It has no default. Like --fps it is read from the
export bundle when the bundle records one, and so are the tokenizer and the
normalization stats — a finetuned bundle ships all three, which is why the
correct invocation is --split-dir and (almost) nothing else.

A gamepad, if one is plugged in, can take the machine over during a --live
run: A toggles between the policy and the sticks. The two are exclusive rather
than summed — manual control drops the chunk in flight, and no inference runs
while the operator drives, so the chunk that resumes control is inferred from
the pose they left the machine in. A missing pad is not an error; the run
proceeds exactly as it would without one.

First run builds the TRT engines (minutes); later runs load from
--cache-dir in seconds.

Usage (model-only check):
    .venv-lerobot/bin/python -m lerobot_vla.run_inference \
        --split-dir <bundle> --synthetic --loops 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lerobot_vla.excavator_robot import JOINT_NAMES
from lerobot_vla.record_episodes import manual_action_from_axes
from lerobot_vla.smolvla_split import NormStats, SmolVLASplitPolicy
from simple_drive import BTN_A, LocalGamepadInput

DEFAULT_SPLIT_DIR = Path.home() / "GitHub/spark-projects/orin-nano/smolvla-runtime/exports/ainekko_base_split"
DEFAULT_TOKENIZER = Path.home() / "GitHub/spark-projects/orin-nano/smolvla-runtime/exports/tokenizer"

LOG = logging.getLogger("run_inference")

#: --camera choice -> the dataset image key it selects.
CAMERA_KEYS = {"ir": "observation.images.cam1", "rgb": "observation.images.cam2"}

#: Used only when nothing in the export bundle records the training rate.
DEFAULT_FPS = 30.0

#: Rate the manual setpoint is fed at while the operator drives. Deliberately
#: NOT --fps: that is the checkpoint's action rate, and a 6 fps checkpoint would
#: leave the setpoint stale between ticks and decay the valves mid-stroke.
TELEOP_HZ = 30.0

#: Button poll rate. Fast enough that a tap is never missed, and it only reads.
BUTTON_POLL_HZ = 50.0

#: How long to wait for a pad at startup. Short, because it is optional.
GAMEPAD_TIMEOUT_S = 3.0


def resolve_policy_fps(split_dir: str, stats_path: str | None,
                       explicit_fps: float | None) -> float:
    """Work out the control rate this checkpoint was trained at.

    A chunk is a sequence of RATE commands, played back at ``--fps``. That rate
    is a property of the checkpoint, not a free choice: a model trained on 10 fps
    data expects each action to be held for 100 ms, so replaying it at 30 Hz holds
    each one for 33 ms and the machine travels a third of the intended distance.
    Nothing about that failure is loud — the excavator just moves feebly.

    Since we now have checkpoints trained at 30, 10 and 6 fps, the rate is read
    from the export bundle where possible instead of defaulting. Search order:

        1. ``fps`` in <split-dir>/export_info.json   (written by the exporter)
        2. ``fps`` in <split-dir>/info.json          (a copied LeRobot info.json)
        3. ``fps`` in <stats-dir>/info.json          (stats copied with its info)

    An explicit --fps that disagrees with the bundle is treated as a mistake and
    refused. If nothing records the rate we fall back to 30 with a warning.
    """
    candidates = [Path(split_dir) / "export_info.json", Path(split_dir) / "info.json"]
    if stats_path:
        candidates.append(Path(stats_path).parent / "info.json")

    recorded, source = None, None
    for path in candidates:
        if not path.is_file():
            continue
        try:
            fps = json.loads(path.read_text()).get("fps")
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("Could not read %s for the training fps: %s", path, exc)
            continue
        if fps:
            recorded, source = float(fps), path
            break

    if recorded is not None:
        if explicit_fps is not None and abs(explicit_fps - recorded) > 1e-6:
            raise SystemExit(
                f"--fps {explicit_fps:g} contradicts the training rate recorded in "
                f"{source} ({recorded:g} fps).\n"
                f"The action chunk is rate commands sampled at {recorded:g} Hz; playing it "
                f"at {explicit_fps:g} Hz scales the executed motion by "
                f"{recorded / explicit_fps:.2f}x.\n"
                f"Drop --fps to use the recorded rate, or pass --fps {recorded:g}.")
        LOG.info("Control rate %.4g Hz (training rate recorded in %s)", recorded, source.name)
        return recorded

    if explicit_fps is not None:
        LOG.warning("No training fps recorded in the export bundle; using --fps %.4g. "
                    "Make sure this matches the rate the checkpoint was trained at.",
                    explicit_fps)
        return explicit_fps

    LOG.warning("No training fps recorded in the export bundle and no --fps given; "
                "assuming %.4g Hz. If this checkpoint was trained on decimated data "
                "(10 or 6 fps) the machine will move at the wrong speed. Record the "
                "rate in <split-dir>/export_info.json to remove this guess.",
                DEFAULT_FPS)
    return DEFAULT_FPS


def resolve_state_blind(split_dir: str, explicit: bool) -> bool:
    """Whether this checkpoint was trained camera-only, so state must be fed as zeros.

    A state-blind checkpoint (the excav_E* bundles) saw ``observation.state == 0`` at
    every training frame, and its normalizer was patched to mean 0 / std 1. That makes
    normalization the identity, so a real joint reading is NOT scaled down on its way
    into ``state_proj`` — raw degrees land in a projector whose weight never received a
    gradient and still sits at its pretrained value. Measured on the E@017500 bundle,
    feeding plausible angles instead of zeros moves the commanded action by up to 0.434
    on the [-1, 1] joystick scale. Like the fps mismatch above, nothing about that
    failure is loud: the machine just drives somewhere else.

    Read from ``state_blind`` in <split-dir>/export_info.json; --state-blind forces it
    on for a bundle that predates the flag.
    """
    path = Path(split_dir) / "export_info.json"
    recorded = None
    if path.is_file():
        try:
            recorded = json.loads(path.read_text()).get("state_blind")
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("Could not read %s for the state_blind flag: %s", path, exc)

    if recorded:
        LOG.warning("CAMERA-ONLY checkpoint (state_blind in %s): observation.state is "
                    "fed to the policy as ZEROS; the IMU reading is logged but unused.",
                    path.name)
        return True
    if explicit:
        LOG.warning("--state-blind given: observation.state is fed to the policy as "
                    "ZEROS even though %s does not record state_blind.", path.name)
        return True
    return False


def resolve_task(split_dir: str, explicit: str | None) -> str:
    """The instruction string this checkpoint was finetuned against.

    The task is a property of the checkpoint, not a free choice: SmolVLA
    conditions on the language embedding, so a phrasing the model never trained
    on is simply out of distribution. Unlike a wrong --fps or another run's
    stats, nothing downstream can detect it — the prefix is well-formed, the
    chunk is well-shaped, and the machine just drives somewhere else.

    There used to be a default here (``"scoop sand and dump it to the left"``,
    the kaivuri task). It silently mislabelled every run against any other
    checkpoint, so it is gone: the string now comes from ``task`` in
    <split-dir>/export_info.json, or from --task, or the run refuses to start.

    An explicit --task that disagrees with the bundle is allowed but warned
    about. Passing the flag is a statement of intent — probing how much the
    phrasing matters is a real thing to want — where omitting it is just an
    omission, and is refused rather than guessed.
    """
    path = Path(split_dir) / "export_info.json"
    recorded = None
    if path.is_file():
        try:
            recorded = json.loads(path.read_text()).get("task")
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("Could not read %s for the task string: %s", path, exc)

    if explicit:
        if recorded and explicit != recorded:
            LOG.warning("--task %r differs from the task recorded in %s (%r). The "
                        "policy conditions on this string; a phrasing the "
                        "checkpoint was not finetuned on is out of distribution.",
                        explicit, path.name, recorded)
        return explicit
    if recorded:
        LOG.info("Task %r (recorded in %s)", recorded, path.name)
        return recorded
    raise SystemExit(
        f"No --task given and {path} records none.\n"
        f"The policy conditions on the instruction string, so there is no safe "
        f"default: a phrasing this checkpoint was not finetuned on is out of "
        f"distribution and nothing downstream will notice.\n"
        f'Pass --task "<the instruction the training dataset was recorded with>", '
        f'or add a "task" key to the bundle\'s export_info.json.')


def resolve_bundle_path(split_dir: str, explicit: str | None, name: str,
                        fallback: str | None, what: str) -> str | None:
    """Prefer what the export bundle ships over a historical default path.

    The finetuned bundles carry their own ``tokenizer/`` and ``stats.json``, so
    the correct invocation is ``--split-dir <bundle>`` and nothing else. Every
    path an operator has to remember separately is a chance to pair a checkpoint
    with another run's normalization — the failure check_state_blind_stats
    exists to catch, and the one that turns real degrees into an
    out-of-distribution state token when --dataset-stats is simply forgotten.
    """
    if explicit:
        return explicit
    shipped = Path(split_dir) / name
    if shipped.exists():
        LOG.info("Using the %s shipped in the bundle: %s", what, shipped)
        return str(shipped)
    return fallback


def check_camera_against_stats(camera_key: str, stats_path: str | None) -> None:
    """Refuse an imager the training dataset never carried.

    record_episodes always writes both cam1 and cam2 off one pipeline, but a
    training run derives its dataset from one of them, and the stats.json that
    comes with it has an entry per image key that survived. That settles which
    imager the checkpoint saw. A cam1-trained policy fed colour frames is out of
    distribution in exactly the silent way a wrong --task is, so check it.
    """
    if not stats_path:
        return
    try:
        stats = json.loads(Path(stats_path).read_text())
    except (json.JSONDecodeError, OSError):
        return  # load_norm reports the real error
    image_keys = sorted(k for k in stats if k.startswith("observation.images."))
    if not image_keys or camera_key in image_keys:
        return
    suggest = next((flag for flag, key in CAMERA_KEYS.items() if key in image_keys), None)
    raise SystemExit(
        f"--camera selects {camera_key}, but {stats_path} was built from "
        f"{image_keys}.\n"
        f"This checkpoint never saw that imager; feeding it is out of distribution "
        f"and it will drive badly."
        + (f"\nPass --camera {suggest} to match." if suggest else ""))


def check_state_blind_stats(norm: NormStats, stats_path: str | None) -> None:
    """Refuse a camera-only checkpoint paired with another run's normalization stats.

    Zeroing the state in ``for_policy`` is not sufficient on its own: the zeros are
    normalized afterwards, inside ``sample_actions``. Pointing --dataset-stats at the
    state-fed dataset (masi_kaivuri_juusto) turns those zeros back into
    ``(0 - mean)/std = [0.172, 1.028, -4.104, 0.869]`` — a state token four sigma out,
    which is precisely what feeding the raw IMU would have done. The state-blind bundle
    ships its own stats.json (mean 0 / std 1) for this reason.

    Checked as the invariant that actually matters — zeros must normalize to zeros —
    rather than by comparing file paths.
    """
    if norm.state_mean is None:
        return
    probe = norm.normalize_state(np.zeros_like(norm.state_mean))
    if np.allclose(probe, 0.0, atol=1e-6):
        return
    raise SystemExit(
        f"--dataset-stats {stats_path} cannot be used with a state_blind (camera-only) "
        f"checkpoint.\n"
        f"This model was trained with observation.state == 0 and expects identity "
        f"normalization (mean 0 / std 1), but these stats normalize zeros to "
        f"[{' '.join(f'{v:+.3f}' for v in probe)}].\n"
        f"That puts an out-of-distribution state token into the prefix — the same failure "
        f"as feeding the raw IMU.\n"
        f"Use the stats.json shipped inside the bundle: "
        f"--dataset-stats {Path(DEFAULT_SPLIT_DIR).parent}/<bundle>/stats.json")


STATE_KEY_NAME = "observation.state"


def load_norm(stats_path: str | None) -> NormStats:
    if not stats_path:
        LOG.warning("No --dataset-stats given: state/action normalization is "
                    "IDENTITY. Fine for base-weight plumbing tests only.")
        return NormStats()
    stats = json.loads(Path(stats_path).read_text())
    norm = NormStats.from_lerobot_stats(stats)
    LOG.info("Loaded normalization stats from %s", stats_path)
    return norm


class GamepadTeleop:
    """Hand the machine between the policy and the sticks, one or the other.

    A toggles: press it while the policy is driving and the operator takes over
    until it is pressed again. Useful for repositioning between takes, and for
    rescuing a run that has driven itself somewhere useless.

    The two sources are exclusive, not summed. That is what makes the hand-over
    clean in both directions. Manual control goes out through ``send_action``,
    whose ``set_point`` DROPS the chunk the control thread is playing
    (modules/setpoint_schedule.py), so the operator never fights a stale plan;
    and because no inference runs while they drive, the chunk that resumes
    control was inferred from the pose they left the machine in rather than
    from one taken before they touched it.

    The pad is polled on its own thread only to catch the button. The inference
    loop comes around once per chunk (~0.5 s), and the pad holds a button's
    state only until its release event arrives, so a tap read from that loop is
    simply missed. That thread never commands the valves.
    """

    def __init__(self, robot, pad) -> None:
        self._robot = robot
        self._pad = pad
        self._edge = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch_button, daemon=True)
        self._thread.start()

    def _watch_button(self) -> None:
        prev = False
        while not self._stop.wait(1.0 / BUTTON_POLL_HZ):
            _, mask = self._pad.poll()
            down = bool(mask & (1 << BTN_A))
            if down and not prev:
                self._edge.set()
            prev = down

    def pressed(self) -> bool:
        """True once per press of A, whenever it is next asked for."""
        if self._edge.is_set():
            self._edge.clear()
            return True
        return False

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
        while not self.pressed():
            axes, _ = self._pad.poll()
            action = (manual_action_from_axes(axes)
                      if axes is not None and self._pad.is_live()
                      else np.zeros(len(JOINT_NAMES), dtype=np.float32))
            self._robot.send_action(action)
            time.sleep(period)
        self._robot.stop_motion()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._pad.close()


def make_teleop(robot) -> GamepadTeleop | None:
    """Gamepad override for a live run, or None if no pad answers.

    Optional on purpose: a pad that is absent, unplugged or unreadable leaves
    the run exactly as it would have been rather than failing it, so an
    inference session never depends on one being connected.
    """
    pad = LocalGamepadInput(connect_timeout_s=GAMEPAD_TIMEOUT_S)
    if not pad.open():
        return None
    return GamepadTeleop(robot, pad)


def main() -> int:
    p = argparse.ArgumentParser(description="SmolVLA split-engine inference loop.")
    p.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer dir. Normally omitted — <split-dir>/tokenizer "
                        f"is used when the bundle ships one (fallback: {DEFAULT_TOKENIZER})")
    p.add_argument("--cache-dir", default="/tmp/smolvla_split_cache")
    p.add_argument("--task", default=None,
                   help="Instruction the policy is conditioned on, e.g. "
                        '"scoop sand and dump it to the left". Same flag, and the '
                        "same string, as record_episodes --task. Normally omitted "
                        "— it is read from the export bundle, since the phrasing "
                        "is a property of the checkpoint. There is NO default: a "
                        "wrong instruction is out of distribution and silent")
    p.add_argument("--num-steps", type=int, default=10, help="Denoise steps")
    p.add_argument("--min-replan-s", type=float, default=0.0,
                   help="Minimum seconds between chunk hand-overs; 0 (default) "
                        "replans as fast as inference allows. A THROTTLE, not an "
                        "execution limit -- the full chunk is always handed to the "
                        "scheduler, so what actually plays is set by inference "
                        "latency (~infer_s * fps), not by this flag")
    p.add_argument("--fps", type=float, default=None,
                   help="Action execution rate. Normally omitted — it is read from "
                        "the export bundle, since it is a property of the checkpoint "
                        f"(default if nothing records it: {DEFAULT_FPS:g})")
    p.add_argument("--dataset-stats", default=None,
                   help="LeRobot stats.json for state/action normalization. "
                        "Normally omitted — <split-dir>/stats.json is used when "
                        "the bundle ships one")
    p.add_argument("--state-blind", action="store_true",
                   help="Feed observation.state to the policy as zeros (camera-only "
                        "checkpoint). Normally omitted — it is read from the export "
                        "bundle's state_blind flag")
    p.add_argument("--robot", default="auto")
    p.add_argument("--state-joints", default="lift,tilt,scoop",
                   help="Joints fed to the policy as observation.state. MUST "
                        "match what the checkpoint was trained on — checked "
                        "against --dataset-stats when that is given. Slew is out "
                        "by default (unanchored yaw; see excavator_robot.py)")
    p.add_argument("--exposure-ir", type=float, default=None,
                   help="Lock cam1 (IR) exposure, microseconds — use the SAME "
                        "value the dataset was recorded with")
    p.add_argument("--gain-ir", type=float, default=None,
                   help="cam1 (IR) sensor gain 16..248")
    p.add_argument("--camera", choices=("ir", "rgb"), default="ir",
                   help="Which imager the policy sees: ir = cam1, rgb = cam2. "
                        "Must match what the checkpoint was TRAINED on — a "
                        "cam1-trained policy fed colour frames is out of "
                        "distribution and will drive badly")
    p.add_argument("--exposure-rgb", type=float, default=None,
                   help="Lock cam2 (RGB) exposure, microseconds — separate sensor "
                        "from the IR one, so --exposure-ir does not apply to it")
    p.add_argument("--gain-rgb", type=float, default=None,
                   help="cam2 (RGB) sensor gain 0..128")
    p.add_argument("--synthetic", action="store_true",
                   help="No robot/camera; synthetic observation (model-only test)")
    p.add_argument("--live", action="store_true",
                   help="SEND actions to the valves (default: print only)")
    p.add_argument("--no-gamepad", action="store_true",
                   help="Do not look for a gamepad. Without it, an attached pad "
                        "can take manual control with A during a --live run "
                        "(exclusive with the policy, not summed); no pad "
                        "attached is not an error either way")
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
    p.add_argument("--legacy-direct-write", action="store_true",
                   help="Drive valves from this thread instead of the control "
                        "thread (bench A/B only — has the 30 Hz rate problems)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Resolve before building engines so a rate mismatch fails in a second rather
    # than after a multi-minute TRT build.
    args.tokenizer = resolve_bundle_path(args.split_dir, args.tokenizer,
                                         "tokenizer", str(DEFAULT_TOKENIZER),
                                         "tokenizer")
    args.dataset_stats = resolve_bundle_path(args.split_dir, args.dataset_stats,
                                             "stats.json", None,
                                             "normalization stats")
    args.task = resolve_task(args.split_dir, args.task)
    args.fps = resolve_policy_fps(args.split_dir, args.dataset_stats, args.fps)
    state_blind = resolve_state_blind(args.split_dir, args.state_blind)

    camera_key = CAMERA_KEYS[args.camera]
    check_camera_against_stats(camera_key, args.dataset_stats)

    state_joints = [j.strip() for j in args.state_joints.split(",") if j.strip()]
    unknown = [j for j in state_joints if j not in JOINT_NAMES]
    if unknown:
        raise SystemExit(f"--state-joints {unknown} not in {JOINT_NAMES}")

    norm = load_norm(args.dataset_stats)
    # The state layout is a contract between the training dataset and this loop.
    # Getting it wrong is silent: a 3-dim policy fed 4 values, or the same count
    # in a different order, still runs and still drives — just wrongly. The stats
    # file carries one number that settles it, so check it.
    if norm.state_mean is not None and len(norm.state_mean) != len(state_joints):
        raise SystemExit(
            f"--state-joints has {len(state_joints)} joints {state_joints} but "
            f"--dataset-stats {args.dataset_stats} was built from a "
            f"{len(norm.state_mean)}-dim observation.state.\n"
            f"Check the training dataset's meta/info.json -> "
            f"features['{STATE_KEY_NAME}']['names'] and pass the same list.")
    if state_blind:
        check_state_blind_stats(norm, args.dataset_stats)

    policy = SmolVLASplitPolicy(
        split_dir=args.split_dir,
        tokenizer_dir=args.tokenizer,
        cache_dir=args.cache_dir,
        num_steps=args.num_steps,
        action_dim=4,
        norm=norm,
        seed=args.seed,
    )

    robot = None
    if not args.synthetic:
        from lerobot_vla.excavator_robot import MasiExcavator
        from lerobot_vla.camera import CameraConfig
        robot = MasiExcavator(
            profile=args.robot,
            camera_config=CameraConfig(exposure_us=args.exposure_ir,
                                         gain=args.gain_ir,
                                         enable_color=args.camera == "rgb",
                                         color_exposure_us=args.exposure_rgb,
                                         color_gain=args.gain_rgb),
            use_control_thread=not args.legacy_direct_write,
            setpoint_hold_s=args.setpoint_hold_s,
            setpoint_decay_s=args.setpoint_decay_s,
            setpoint_blend_s=args.blend_s,
            state_joints=state_joints)
        robot.connect()
        if args.live:
            LOG.warning("LIVE MODE: actions will drive the valves. Pump is under "
                        "your control (it is NOT auto-enabled).")
        else:
            LOG.info("Dry run: observations are real, actions are printed only.")

    # Live only: without --live no source drives the valves, and the safety
    # ladder in the module docstring should not have a hole in it for the pad.
    teleop = (make_teleop(robot)
              if robot is not None and args.live and not args.no_gamepad
              else None)
    if teleop is not None:
        LOG.info("Gamepad attached: A hands control between the policy and the "
                 "sticks (exclusive, not summed).")

    if not args.synthetic:
        LOG.info("Policy camera: %s (%s)", camera_key,
                 "IR cam1" if args.camera == "ir" else "colour cam2")
    LOG.info("observation.state = %s%s", state_joints,
             "" if "slew" in state_joints else "   (slew excluded: unanchored yaw)")

    def get_observation():
        if robot is not None:
            obs = robot.get_observation()
            return obs[camera_key], obs["observation.state"]
        img = (np.random.default_rng(0).integers(0, 255, (480, 640, 3))
               .astype(np.uint8))
        return img, np.zeros(len(state_joints), dtype=np.float32)

    def for_policy(state):
        """What the policy actually receives. A camera-only checkpoint gets zeros;
        the real reading is still logged, so the IMU stays visible for diagnostics."""
        return np.zeros_like(state) if state_blind else state

    # Warmup / engine build happens on the first sample_actions call.
    LOG.info("Warmup inference (builds TRT engines on first ever run)...")
    img, state = get_observation()
    t0 = time.perf_counter()
    policy.sample_actions(img, args.task, for_policy(state))
    LOG.info("Warmup done in %.1fs", time.perf_counter() - t0)

    cycle = 0
    # Start re-inference this far before the current chunk ends, so the next one
    # is ready when it runs out. Tracked as a decaying max of measured inference
    # time; the first cycle sets it from its own measurement. Not seeded from
    # the warmup — that one includes the TRT engine build.
    infer_lead_s = 0.0
    chunk_t0 = 0.0          # set on every hand-over; only read once cycle > 0
    # When teleop took a chunk over mid-play. `played` counts up to this instead
    # of to the next hand-over, so the manual stretch is not billed to the chunk.
    chunk_cut = None
    try:
        while True:
            img, state = get_observation()
            t0 = time.perf_counter()
            chunk = policy.sample_actions(img, args.task, for_policy(state))
            infer_s = time.perf_counter() - t0
            infer_lead_s = max(infer_s, infer_lead_s * 0.9)

            # Hand over the FULL chunk. --min-replan-s only gates when the next
            # inference may START; it never truncates what is executed. The part
            # of the chunk that has not played when the next one lands is the
            # late-inference fallback: a slow replan keeps following the
            # predicted plan instead of falling into hold->decay.
            if robot is not None and args.live:
                robot.send_action_chunk(chunk, fps=args.fps)
            now = time.perf_counter()
            # Steps of the PREVIOUS chunk that actually played before this one
            # replaced it -- measured, not assumed. This is the real execution
            # horizon; with --min-replan-s 0 it is inference latency alone.
            played_until = chunk_cut if chunk_cut is not None else now
            played = (min(round((played_until - chunk_t0) * args.fps), len(chunk))
                      if cycle else None)
            chunk_t0 = now
            chunk_cut = None

            LOG.info("cycle=%d infer=%.0fms played=%s state%s=[%s] a0=[%s]",
                     cycle, infer_s * 1000.0,
                     played if played is not None else "-",
                     "(unused)" if state_blind else "",
                     " ".join(f"{v:+.1f}" for v in state),
                     " ".join(f"{v:+.2f}" for v in chunk[0]))

            cycle += 1
            if args.loops and cycle >= args.loops:
                break

            # Blocks here for as long as the operator drives. The chunk handed
            # over above is dropped by the first manual setpoint, so record when
            # it stopped playing before giving the machine away.
            if teleop is not None and teleop.pressed():
                chunk_cut = time.perf_counter()
                teleop.run()

            # Sleep until it is time to start the next inference. The control
            # thread is driving the valves from the chunk meanwhile. The default
            # --min-replan-s 0 means no sleep at all: replan flat out.
            sleep_s = args.min_replan_s - infer_lead_s - (time.perf_counter() - chunk_t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if teleop is not None:
            teleop.close()
        if robot is not None:
            try:
                robot.stop_motion()
            except Exception:
                pass
            robot.disconnect()

    LOG.info("Done: %d cycles.", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
