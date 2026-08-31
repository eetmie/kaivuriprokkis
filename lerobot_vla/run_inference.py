#!/usr/bin/env python3
"""VLA inference loop for the MASI excavator (split TRT engines).

    IR cam1 + joint angles + task -> SmolVLA or X-VLA (split ONNX/TRT) -> valves

Which architecture runs is read from the bundle rather than chosen with a flag —
see policy.py. SmolVLA is the deployed path; X-VLA is 12 engines instead of 9,
~3x the inference time, and 2.5x the resident memory, and only one of the two
fits in this board's 8 GB at a time, so switching models means restarting.

Modes (safety ladder):
    --synthetic     no robot, no camera — synthetic image + zero state. Proves
                    the model pipeline + engine builds. Safe anywhere.
    (default)       robot connected, observations real, actions PRINTED ONLY.
    --live          actions are actually sent to the valves. Base (un-finetuned)
                    weights produce garbage actions — do not use --live until a
                    finetuned checkpoint is in place and the pump is your call.

    --allow-base-bundle loads an X-VLA export that carries no physical boundary
                    (the base ee6d checkpoint). Its chunk is arm dimensions, not
                    valve commands, so that mode refuses --live outright: it
                    exists only for model/engine diagnostics.

The policy emits an action chunk per observation (length is a property of the
bundle -- 50 for the base export, 12 for the deployed digging bundle) and the
WHOLE chunk is handed to the controller as a trajectory at --fps. Re-inference
starts as soon as the previous one finishes, unless --min-replan-s throttles it.
Whatever part of the chunk has not played when the next one lands is simply
replaced, so the EXECUTED horizon is set by inference latency (~infer_s * fps),
not by any flag: at the measured 0.12 s inference and 30 Hz that is ~4 steps.
The unplayed tail is the late-inference fallback — a slow replan keeps following
the predicted plan instead of degrading into hold->decay-to-zero.

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

A checkpoint finetuned on a multi-task dataset has more than one phrasing in
distribution, and its export records them as a "tasks" LIST (the same strings as
the dataset's meta/tasks.parquet). Then the run carries all of them and the D-pad
picks which one is live: right = next, left = previous. Only the string changes —
one policy, one set of engines, one process — so a switch costs one text-encoder
run the first time an instruction is used and nothing afterwards; it is nothing
like the model switch in the paragraph above. Repeating --task does the same from
the command line, in the order given — and a --task the bundle already records is a
REORDERING, not a filter: it goes first, the rest stay behind it on the D-pad, so
naming the task to start on never costs the operator the others. Switching drops
the chunk in flight and
stops the machine first: that chunk was planned under the old instruction, and
half of one task's plan followed by half of another's is a trajectory neither of
them meant.

A gamepad, if one is plugged in, can take the machine over during a --live
run: A toggles between the policy and the sticks. The two are exclusive rather
than summed — manual control drops the chunk in flight, and no inference runs
while the operator drives, so the chunk that resumes control is inferred from
the pose they left the machine in. A missing pad is not an error; the run
proceeds exactly as it would without one. Takeover is live-only, but the D-pad's
task selector is not: a multi-task run opens the pad without --live too, with A
disabled, because picking the next inference's instruction never writes a valve.

First run builds the TRT engines (minutes); later runs load from
--cache-dir in seconds. The cache is per bundle — a second --split-dir builds
its own engines beside the first one's rather than colliding with them, and
either model then starts in seconds. --rebuild forces one bundle's rebuild.

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

from lerobot_vla.action_log import ActionLogger
# Control-layer channel names. Column-aligned with JOINT_NAMES (the dataset
# order), and the keys the setpoint status dict uses — so an action-log
# column means the same thing in the chunk stream and the emitted stream.
from lerobot_vla.excavator_robot import JOINT_NAMES
from lerobot_vla.excavator_robot import _CONTROL_CHANNELS as CONTROL_CHANNELS
from lerobot_vla.record_episodes import manual_action_from_axes
from lerobot_vla import xvla_split
from lerobot_vla.policy import (
    bundle_tasks, detect_architecture, make_policy, merge_tasks, warn_off_bundle,
)
from lerobot_vla.smolvla_split import NormStats
from simple_drive import BTN_A, BTN_DPAD_LEFT, BTN_DPAD_RIGHT, LocalGamepadInput

#: Where deployable export bundles live on the Jetson. Only used to make error
#: messages concrete — there is deliberately NO default bundle. A checkpoint is the
#: single biggest thing deciding what the machine does, so it gets named out loud
#: rather than inherited from an argparse default. (The old default was a public
#: base-weight split that did not work here; see the README.)
BUNDLE_ROOT = Path.home() / "bundles"
DEFAULT_TOKENIZER = BUNDLE_ROOT / "smolvlm2-tokenizer"

LOG = logging.getLogger("run_inference")

#: --camera choice -> the dataset image key it selects.
CAMERA_KEYS = {"ir": "observation.images.cam1", "rgb": "observation.images.cam2"}

#: Used only when nothing in the export bundle records the training rate.
DEFAULT_FPS = 30.0

#: Argparse defaults kept as names because the X-VLA path has to tell "left at
#: the default" from "asked for explicitly": its runtime derives both from the
#: bundle (engine cache beside the graphs, denoise steps from bundle.json), so a
#: default carried over from the SmolVLA side would silently override them.
DEFAULT_CACHE_DIR = "/tmp/smolvla_split_cache"
DEFAULT_NUM_STEPS = 10

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


def resolve_tasks(split_dir: str, explicit: list[str] | None) -> list[str]:
    """The instruction strings this run may condition on, in order.

    The task is a property of the checkpoint, not a free choice: SmolVLA
    conditions on the language embedding, so a phrasing the model never trained
    on is simply out of distribution. Unlike a wrong --fps or another run's
    stats, nothing downstream can detect it — the prefix is well-formed, the
    chunk is well-shaped, and the machine just drives somewhere else.

    There used to be a default here (``"scoop sand and dump it to the left"``,
    the kaivuri task). It silently mislabelled every run against any other
    checkpoint, so it is gone: the strings now come from ``tasks``/``task`` in
    <split-dir>/export_info.json, or from --task, or the run refuses to start.

    A checkpoint finetuned on a multi-task dataset has more than one phrasing IN
    distribution, and its export records them as a list (see policy.bundle_tasks).
    A single-task bundle is the one-element case of that, so both come back as a
    list and the loop never branches on which kind it was handed. The first entry
    is the one the run starts on; the operator cycles the rest with the D-pad.

    An explicit --task goes to the FRONT of the list rather than replacing it
    (policy.merge_tasks): the flag picks which instruction the run starts on, and
    the bundle's other tasks stay on the D-pad behind it. An explicit --task the
    bundle does not record is allowed but warned about. Passing the flag is a
    statement of intent — probing how much the phrasing matters is a real thing to
    want — where omitting it is just an omission, and is refused rather than
    guessed.
    """
    path = Path(split_dir) / "export_info.json"
    recorded: list[str] = []
    if path.is_file():
        try:
            recorded = bundle_tasks(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("Could not read %s for the task string: %s", path, exc)

    if explicit:
        warn_off_bundle(explicit, recorded, path.name)
        tasks = merge_tasks(explicit, recorded)
        LOG.info("Task%s %s (--task first, then the rest of %s)",
                 "s" if len(tasks) > 1 else "",
                 ", ".join(repr(t) for t in tasks), path.name)
        return tasks
    if recorded:
        LOG.info("Task%s %s (recorded in %s)", "s" if len(recorded) > 1 else "",
                 ", ".join(repr(t) for t in recorded), path.name)
        return recorded
    raise SystemExit(
        f"No --task given and {path} records none.\n"
        f"The policy conditions on the instruction string, so there is no safe "
        f"default: a phrasing this checkpoint was not finetuned on is out of "
        f"distribution and nothing downstream will notice.\n"
        f'Pass --task "<the instruction the training dataset was recorded with>", '
        f'or add a "task" key (or a "tasks" list, for a multi-task checkpoint) '
        f"to the bundle's export_info.json.")


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
        f"--dataset-stats {BUNDLE_ROOT}/<bundle>/stats.json")


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
    p = argparse.ArgumentParser(description="SmolVLA split-engine inference loop.")
    p.add_argument("--split-dir", required=True,
                   help="The export bundle to drive with, e.g. "
                        f"{BUNDLE_ROOT}/smolvla-digging-clean-ir12-35k. Required: "
                        "the checkpoint is the single biggest thing that decides "
                        "what the machine does, so it is never defaulted")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer dir. Normally omitted — <split-dir>/tokenizer "
                        f"is used when the bundle ships one (SmolVLA fallback: "
                        f"{DEFAULT_TOKENIZER}; an X-VLA bundle that ships none is "
                        f"a base export and needs this passed explicitly)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                   help="Root of the TRT engine cache. Each bundle gets its own "
                        "subdirectory under it, so switching --split-dir needs "
                        "no extra flag and switching back is still instant. "
                        "Left alone, an X-VLA bundle caches into its own "
                        "<split-dir>/trt_cache instead, which survives a reboot.")
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
    p.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS,
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
                        "(exclusive with the policy, not summed) and cycle the "
                        "task with the D-pad when the run carries more than one; "
                        "no pad attached is not an error either way")
    p.add_argument("--loops", type=int, default=0,
                   help="Stop after N infer+execute cycles (0 = run until Ctrl+C)")
    p.add_argument("--seed", type=int, default=None, help="Fix the denoise noise seed")
    # Both default to the settings validated on this board in
    # jetson-orin-nano-vla (docs/06-optimization-backlog.md): together they cut
    # p50 from 229.8 to 134.8 ms with bit-identical output. The opt-outs exist
    # so a regression can be A/B'd against the old loop, not because there is a
    # reason to run them.
    p.add_argument("--projectors", choices=("gpu", "cpu"), default="gpu",
                   help="Where the four per-step action/time projectors run. "
                        "gpu (default) keeps the denoise loop off the CPU; "
                        "cpu reproduces the pre-benchmark loop")
    p.add_argument("--no-iobinding", dest="iobinding", action="store_false",
                   help="Re-feed the KV cache as numpy on every denoise step "
                        "instead of binding it to the device once. Slower, and "
                        "bit-identical — for regression comparison only")
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

    # The architecture is a property of the bundle, not a flag — see policy.py.
    architecture = detect_architecture(args.split_dir)
    LOG.info("Architecture: %s (detected from %s)", architecture, args.split_dir)

    camera_key = CAMERA_KEYS[args.camera]
    state_joints = [j.strip() for j in args.state_joints.split(",") if j.strip()]
    unknown = [j for j in state_joints if j not in JOINT_NAMES]
    if unknown:
        raise SystemExit(f"--state-joints {unknown} not in {JOINT_NAMES}")

    # Everything below resolves BEFORE any engine is built, so a wrong rate, task
    # or camera fails in a second rather than after a multi-minute TRT build.
    # The two architectures record the same facts in different files: SmolVLA in
    # export_info.json + stats.json, X-VLA in its bundle.json + processor
    # contract. Each resolver lives beside the runtime that owns that format.
    if architecture == "xvla":
        if args.dataset_stats:
            raise SystemExit(
                "--dataset-stats does not apply to an X-VLA bundle: its "
                "normalization is carried in bundle.json's processor contract, "
                "which the runtime applies at the physical boundary itself.")
        # Refuse the non-drivable combination BEFORE the engines load: the
        # authoritative check is on the constructed policy below, but that costs a
        # 25 s engine load first, and --allow-base-bundle is the only way a
        # feasibility bundle gets in at all.
        if args.live and args.allow_base_bundle:
            raise SystemExit(
                "--live and --allow-base-bundle are contradictory: a bundle that "
                "needs --allow-base-bundle emits arm dimensions, not valve "
                "commands. Run it without --live to watch it against real "
                "observations.")
        tasks = xvla_split.resolve_tasks(args.split_dir, args.tasks)
        args.fps = xvla_split.resolve_fps(args.split_dir, args.fps)
        xvla_split.check_camera(args.split_dir, camera_key)
        state_blind = args.state_blind
        if args.projectors != "gpu" or not args.iobinding:
            LOG.warning("--projectors / --no-iobinding are SmolVLA-only knobs and "
                        "are ignored for an X-VLA bundle.")
        policy = make_policy(
            "xvla", args.split_dir,
            cache_dir=args.cache_dir if args.cache_dir != DEFAULT_CACHE_DIR else None,
            num_steps=args.num_steps if args.num_steps != DEFAULT_NUM_STEPS else None,
            seed=args.seed,
            tokenizer_dir=args.tokenizer,
            state_joints=state_joints,
            allow_base_bundle=args.allow_base_bundle,
        )
    else:
        args.tokenizer = resolve_bundle_path(args.split_dir, args.tokenizer,
                                             "tokenizer", str(DEFAULT_TOKENIZER),
                                             "tokenizer")
        args.dataset_stats = resolve_bundle_path(args.split_dir, args.dataset_stats,
                                                 "stats.json", None,
                                                 "normalization stats")
        tasks = resolve_tasks(args.split_dir, args.tasks)
        args.fps = resolve_policy_fps(args.split_dir, args.dataset_stats, args.fps)
        state_blind = resolve_state_blind(args.split_dir, args.state_blind)
        check_camera_against_stats(camera_key, args.dataset_stats)

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

        policy = make_policy(
            "smolvla", args.split_dir,
            tokenizer_dir=args.tokenizer,
            cache_dir=args.cache_dir,
            rebuild=args.rebuild,
            num_steps=args.num_steps,
            action_dim=4,
            norm=norm,
            seed=args.seed,
            projectors=args.projectors,
            iobinding=args.iobinding,
        )

    # A bundle with no physical boundary emits arm dimensions, not valve commands
    # (see xvla_split.py). It is loadable for model/engine diagnostics, but it
    # never drives.
    if getattr(policy, "feasibility_only", False) and args.live:
        raise SystemExit(
            "--live refused: this bundle is a model-only feasibility export whose "
            "action columns are not valve commands. Drop --live to watch it run "
            "against real observations.")

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

    # Opened before the warmup so the meta record carries the resolved settings
    # even if the warmup is what blows up.
    action_log = None
    if args.log_actions:
        action_log = ActionLogger(args.log_actions, CONTROL_CHANNELS, meta={
            "split_dir": args.split_dir,
            "task": tasks[0],
            "tasks": list(tasks),
            "fps": args.fps,
            "live": bool(args.live),
            "synthetic": bool(args.synthetic),
            "camera": args.camera,
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
    img, state = get_observation()
    t0 = time.perf_counter()
    policy.sample_actions(img, tasks[0], for_policy(state))
    LOG.info("Warmup done in %.1fs", time.perf_counter() - t0)

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
    chunk_t0 = 0.0          # set on every hand-over; only read once cycle > 0
    # When a chunk stopped playing mid-play -- teleop took the machine over, or
    # the operator switched task. `played` counts up to this instead of to the
    # next hand-over, so the stopped stretch is not billed to the chunk.
    chunk_cut = None
    try:
        while True:
            img, state = get_observation()
            t0 = time.perf_counter()
            chunk = policy.sample_actions(img, tasks[task_index], for_policy(state))
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

            LOG.info("cycle=%d%s infer=%.0fms played=%s state%s=[%s] a0=[%s]%s",
                     cycle,
                     f" task={task_index}" if len(tasks) > 1 else "",
                     infer_s * 1000.0,
                     played if played is not None else "-",
                     "(unused)" if state_blind else "",
                     " ".join(f"{v:+.1f}" for v in state),
                     " ".join(f"{v:+.2f}" for v in chunk[0][:4]),
                     f" (+{chunk.shape[1] - 4} more columns; not valve commands)"
                     if chunk.shape[1] > 4 else "")

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

    LOG.info("Done: %d cycles.", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
