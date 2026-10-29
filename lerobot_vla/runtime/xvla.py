"""X-VLA bundle contracts and the excavator's direct TensorRT policy wrapper.

The saved processor contract defines physical state/action dimensions and
normalization. Incomplete/base exports remain model-only, with --live refused.
The accelerated implementation is in xvla_native.py; shared upstream TensorRT
helpers are vendored so this project runs independently of the benchmark repo.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

# How a bundle spells its instruction list is the one thing the two export
# formats have to agree on, so it is defined once, next to the loader that picks
# between them.
from lerobot_vla.policy import bundle_tasks, merge_tasks, warn_off_bundle

LOG = logging.getLogger("xvla_split")

#: Where a bundle's TRT engines are cached. Beside the graphs, not in /tmp: twelve
#: engines are a ~5 minute cold build and /tmp is cleared on reboot, which would
#: mean paying that at every boot. (The SmolVLA side still uses /tmp and does
#: exactly that.)
ENGINE_CACHE_DIRNAME = "trt_native_cache"


def is_xvla_bundle(split_dir: str | Path) -> bool:
    """Whether <split-dir> looks like an X-VLA split export.

    The architecture is a property of the bundle, not a free choice, so it is
    detected from what the bundle ships rather than asked for on the command
    line: an X-VLA export carries `bundle.json`, a SmolVLA one carries
    `export_info.json` and `smolvlm_vision.onnx`.
    """
    return (Path(split_dir) / "bundle.json").is_file()


def load_bundle(split_dir: str | Path) -> dict:
    """The bundle document, with schema-v2 identity checks applied.

    `verify_bundle` fails CLOSED on a v2 bundle whose checkpoint hash, processor
    artifacts or tokenizer tree do not match what it claims — before any engine is
    built, which is the only point where failing is cheap. A v1 bundle predates all
    of that and is returned as-is; `XVLAExcavatorPolicy` is what refuses to drive
    with one.
    """
    from lerobot_vla.runtime.vendor.xvla_bundle_contract import verify_bundle

    split_dir = Path(split_dir)
    if not (split_dir / "bundle.json").is_file():
        raise SystemExit(
            f"{split_dir} has no bundle.json, so it is not an X-VLA export.\n"
            f"Export one with vla-onnx/xvla/export_split_onnx.py.")
    try:
        return verify_bundle(split_dir, verify_manifest=False)
    except ValueError as exc:
        raise SystemExit(
            f"{split_dir}/bundle.json failed its own identity check: {exc}\n"
            f"The bundle does not match the checkpoint or processor artifacts it "
            f"claims to carry. Re-export it rather than running it.") from exc


def contract_of(bundle: dict) -> dict | None:
    """The processor contract, or None for a base/legacy bundle that has none."""
    contract = bundle.get("processor_contract")
    return contract if contract and int(contract.get("version") or 0) == 1 else None


def resolve_tasks(split_dir: str | Path, explicit: list[str] | None) -> list[str]:
    """The instruction strings this checkpoint was fine-tuned against, in order.

    Same rule, and the same reasoning, as `run_inference.resolve_tasks` on the
    SmolVLA side: the policy conditions on the language embedding, so a phrasing
    the checkpoint never trained on is out of distribution and *nothing downstream
    can detect it* — the prefix is well-formed, the chunk is well-shaped, and the
    machine simply drives somewhere else. There is therefore no default: the
    strings come from the bundle, or from --task, or the run refuses to start.

    A multi-task fine-tune records a ``tasks`` list; a single-task one records
    ``task``, which is the one-element case (policy.bundle_tasks). The run starts
    on the first and the operator cycles the rest with the pad's D-pad — including
    when --task is given, which puts the named instruction first rather than
    throwing the bundle's others away (policy.merge_tasks).
    """
    bundle = load_bundle(split_dir)
    recorded = bundle_tasks(bundle)
    if explicit:
        warn_off_bundle(explicit, recorded, "bundle.json")
        tasks = merge_tasks(explicit, recorded)
        LOG.info("Task%s %s (--task first, then the rest of bundle.json)",
                 "s" if len(tasks) > 1 else "",
                 ", ".join(repr(t) for t in tasks))
        return tasks
    if recorded:
        LOG.info("Task%s %s (recorded in bundle.json)",
                 "s" if len(recorded) > 1 else "",
                 ", ".join(repr(t) for t in recorded))
        return recorded
    raise SystemExit(
        f"No --task given and {Path(split_dir)}/bundle.json records none.\n"
        f"The policy conditions on the instruction string, so there is no safe "
        f"default: a phrasing this checkpoint was not fine-tuned on is out of "
        f"distribution and nothing downstream will notice.\n"
        f'Pass --task "<the instruction the training dataset was recorded with>", '
        f'or have the exporter write a "task" key — or a "tasks" list, for a '
        f'multi-task fine-tune — into bundle.json (export_split_onnx.py --task).')


def resolve_fps(split_dir: str | Path, explicit_fps: float | None) -> float:
    """The rate the chunk's actions were sampled at.

    A chunk is a sequence of RATE commands played back at this rate, so it belongs
    to the checkpoint and not to the operator: a model trained on 10 fps data
    replayed at 30 Hz holds each action for a third as long and the machine travels
    a third of the intended distance. Nothing about that failure is loud.

    X-VLA's bundle.json historically carried no `fps` — `xvla-base` has no training
    rate for our robot — so the fine-tune export has to write one, exactly as
    `export_info.json` does on the SmolVLA side. Until it does, this refuses rather
    than assuming 30: the SmolVLA path can fall back on a warning because its
    bundles have carried the key for a while and 30 is the rate its datasets were
    actually recorded at, whereas an X-VLA bundle with no fps has never been
    validated at any rate here.
    """
    bundle = load_bundle(split_dir)
    recorded = bundle.get("fps")
    if recorded:
        recorded = float(recorded)
        if explicit_fps is not None and abs(explicit_fps - recorded) > 1e-6:
            raise SystemExit(
                f"--fps {explicit_fps:g} contradicts the training rate recorded in "
                f"{Path(split_dir)}/bundle.json ({recorded:g} fps).\n"
                f"The action chunk is rate commands sampled at {recorded:g} Hz; playing "
                f"it at {explicit_fps:g} Hz scales the executed motion by "
                f"{recorded / explicit_fps:.2f}x.\n"
                f"Drop --fps to use the recorded rate, or pass --fps {recorded:g}.")
        LOG.info("Control rate %.4g Hz (training rate recorded in bundle.json)", recorded)
        return recorded

    if explicit_fps is not None:
        LOG.warning("No training fps in bundle.json; using --fps %.4g. Make sure this "
                    "matches the rate the checkpoint was trained at — the chunk is "
                    "rate commands, so a wrong value scales the machine's speed.",
                    explicit_fps)
        return explicit_fps

    raise SystemExit(
        f"{Path(split_dir)}/bundle.json records no `fps` and no --fps was given.\n"
        f"The action chunk is rate commands sampled at the training rate; playing it "
        f"at the wrong rate scales every motion the machine makes, silently.\n"
        f"Have the exporter write it (export_split_onnx.py --fps <dataset fps>), or "
        f"pass --fps explicitly if you know the rate the dataset was recorded at.")


def resolve_cameras(split_dir: str | Path, selected=None, *, allow_unknown=False) -> list[str]:
    """Use checkpoint VISUAL feature order, never filesystem/device order."""
    bundle = load_bundle(split_dir)
    contract = contract_of(bundle) or {}
    features = contract.get("input_features") or {}
    trained = [name for name, spec in features.items()
               if str(spec.get("type", "")).upper() == "VISUAL"]
    requested = ([selected] if isinstance(selected, str) else list(selected)) if selected else None
    keys = requested or trained or ["observation.images.cam1"]
    if trained and keys != trained:
        raise SystemExit(f"Camera selection {keys} does not match checkpoint view order {trained}.")
    unsupported = set(keys) - {"observation.images.cam1", "observation.images.cam2"}
    if (unsupported and not allow_unknown) or len(set(keys)) != len(keys):
        raise SystemExit(f"Unsupported or duplicate camera features: {keys}")
    valid = bundle.get("valid_views")
    if valid is not None and int(valid) != len(keys):
        raise SystemExit(f"Bundle requires {valid} camera views but selection supplies {len(keys)}.")
    return keys


def check_camera(split_dir: str | Path, camera_key: str) -> None:
    resolve_cameras(split_dir, camera_key)


class XVLAExcavatorPolicy:
    """The 12-engine X-VLA split policy, called the way `run_inference` calls SmolVLA."""

    def __init__(self,
                 split_dir: str | Path,
                 cache_dir: str | Path | None = None,
                 rebuild: bool = False,
                 precision: str = "fp16",
                 num_steps: int | None = None,
                 seed: int | None = None,
                 tokenizer_dir: str | Path | None = None,
                 state_joints: list[str] | None = None,
                 allow_base_bundle: bool = False,
                 chain: str = "graph"):
        split_dir = Path(split_dir)
        self.bundle = load_bundle(split_dir)
        self.contract = contract_of(self.bundle)
        self.split_dir = split_dir

        # A bundle with no physical boundary cannot produce a valve command. Say
        # which of the two cases it is, because the fixes differ: a v1 export is
        # the base ee6d checkpoint and needs a fine-tune, while a v2 export whose
        # boundary is incomplete has IDENTITY-normalized stats it should have
        # carried and needs re-exporting from a checkpoint that saved them.
        self.feasibility_only = (
            self.contract is None
            or not self.contract.get("physical_boundary_complete"))
        if self.feasibility_only and not allow_base_bundle:
            raise SystemExit(self._base_bundle_message())

        from lerobot_vla.runtime.xvla_native import XVLANativePolicy
        cache_dir = Path(cache_dir) if cache_dir else split_dir / ENGINE_CACHE_DIRNAME
        if rebuild and cache_dir.exists():
            # Only generated runtime cache artifacts belong to the rebuild.
            for pattern in ("*.engine", "*.onnx.sha256", "*.engine.sha256", "timing.cache"):
                for path in cache_dir.glob(pattern):
                    path.unlink()

        # A schema-v2 bundle materializes its own tokenizer and verify_bundle
        # checks the tree hash, so the bundle's copy is the right one and an
        # explicit --tokenizer is only ever needed for a v1 feasibility export,
        # which ships none.
        tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else split_dir / "tokenizer"
        if not tokenizer_dir.is_dir():
            raise SystemExit(
                f"No tokenizer at {tokenizer_dir}.\n"
                f"A fine-tuned bundle ships its own under <split-dir>/tokenizer; this "
                f"one does not, so pass --tokenizer <dir> (the X-VLA base tokenizer "
                f"lives in the bundle's tokenizer/ dir).")

        self.policy = XVLANativePolicy(
            split_dir,
            cache_dir=str(cache_dir),
            precision=precision,
            tokenizer_dir=str(tokenizer_dir),
            num_denoising_steps=num_steps,
            seed=seed,
            chain=chain,
        )

        self.chunk_size = self.policy.chunk_size
        #: Valve channels this policy commands. 4 once the contract trims the
        #: model's padded 20; the raw model width in feasibility mode, where the
        #: columns are arm dims and mean nothing here.
        self.action_dim = (self.policy.real_action_dim
                           if self.policy.real_action_dim is not None
                           else self.policy.action_dim)
        self.state_dim = self.policy.state_dim

        # The state layout is a contract between the training dataset and this
        # loop, and getting it wrong is silent: the same count in a different
        # order still runs and still drives. The bundle carries the one number
        # that settles it.
        if state_joints is not None and not self.feasibility_only:
            if len(state_joints) != self.state_dim:
                raise SystemExit(
                    f"--state-joints has {len(state_joints)} joints {state_joints} but "
                    f"{split_dir}/bundle.json was exported from a {self.state_dim}-dim "
                    f"{self.contract['state']['feature']}.\n"
                    f"Check the training dataset's meta/info.json -> features -> names "
                    f"and pass the same list.")

        LOG.info("X-VLA bundle: chunk %d x %d action dims, state %d, "
                 "%d/%d camera views, %d denoise steps%s",
                 self.chunk_size, self.action_dim, self.state_dim,
                 self.policy.valid_views, self.policy.num_views, self.policy.steps,
                 "   [FEASIBILITY ONLY — not drivable]" if self.feasibility_only else "")

    def _base_bundle_message(self) -> str:
        if self.contract is None:
            why = (f"It is a schema-v{int(self.bundle.get('schema_version') or 1)} "
                   f"export with no processor contract — i.e. the base "
                   f"{self.bundle.get('action_mode', 'ee6d')!r} checkpoint, whose "
                   f"{self.bundle.get('max_state_dim', 20)} action columns are arm "
                   f"end-effector dimensions, not valve commands.")
            fix = ("Fine-tune it on excavator data first "
                   "(spark-projects/vla-onnx/xvla), then re-export.")
        else:
            why = ("Its processor contract is incomplete "
                   "(`physical_boundary_complete: false`): the checkpoint did not "
                   "carry the normalization statistics the boundary needs, so state "
                   "and action would pass through unnormalized.")
            fix = ("Re-export from a checkpoint whose policy_preprocessor/"
                   "postprocessor saved MEAN_STD statistics.")
        return (
            f"{self.split_dir} cannot drive this machine.\n{why}\n"
            f"Slicing its chunk to 4 columns would produce a perfectly well-formed "
            f"action chunk that is nonsense on a hydraulic excavator, and nothing "
            f"downstream would notice.\n{fix}\n"
            f"To run it anyway as a model-only feasibility test (engines, latency, "
            f"memory), pass --allow-base-bundle. That mode refuses --live.")

    def sample_actions(self, image_hwc_uint8: np.ndarray | list[np.ndarray], instruction: str,
                       state: np.ndarray) -> np.ndarray:
        """One observation -> (chunk_size, action_dim) chunk in valve units.

        The vendored runtime does the whole physical boundary: it normalizes the
        state with the bundle's own statistics, runs the vision/text/cond cold path
        once and the 24-block policy transformer `steps` times over it, trims the
        model's padded action width back to the real one, and unnormalizes. What is
        left for us is the call shape: one ordered entry per real camera view.
        A single ndarray remains accepted for single-view checkpoints.
        """
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if self.feasibility_only:
            # No contract, so the runtime wants the model's full padded proprio
            # vector rather than physical axes. Zero-pad, exactly as a fine-tune's
            # _prepare_state would. The chunk that comes back is arm dims.
            padded = np.zeros(self.state_dim, dtype=np.float32)
            padded[: min(state.size, self.state_dim)] = state[: self.state_dim]
            state = padded
        images = (list(image_hwc_uint8) if isinstance(image_hwc_uint8, (list, tuple))
                  else [image_hwc_uint8])
        if len(images) != self.policy.valid_views:
            raise ValueError(f"Expected {self.policy.valid_views} camera views, got {len(images)}")
        return self.policy.sample_actions(images, instruction, state)

    @property
    def last_timings(self) -> dict:
        """Per-stage milliseconds from the last inference (vision/text/cond/denoise)."""
        return self.policy.last_timings

    def close(self):
        self.policy.close()
