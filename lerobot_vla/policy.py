"""One entry point for loading a VLA policy: the model is a bundle, not a code path.

`run_inference` drives whatever `make_policy` returns, and both architectures
expose the same three things:

    policy.sample_actions(image_hwc_uint8, task, state) -> (chunk_size, action_dim)
    policy.chunk_size
    policy.action_dim

Which architecture loads is **detected from the bundle**, not selected with a
flag. That follows the rule the rest of this loop already works by: a checkpoint
is the single biggest thing deciding what the machine does, so it is named out
loud on the command line — and everything else about it (its task phrasing, its
control rate, its normalization, and now its architecture) is a property *of that
checkpoint*, read from what it ships. An `--arch` flag would only add a way for
the operator to contradict the bundle.

    X-VLA   <split-dir>/bundle.json
    SmolVLA <split-dir>/export_info.json + smolvlm_vision.onnx

**Only one policy per process.** Measured on this board on 2026-08-31: X-VLA's
twelve engines sit at 5.47 GB resident with the D435i reader and the 100 Hz
control thread beside them, against 7.4 GB total — an available floor of 1.48 GB.
SmolVLA's nine engines (2.2 GB with the same robot stack) do not fit in what is
left. So switching models is a restart, not a runtime toggle, and constructing a
second policy in one process raises here rather than being discovered as an OOM
part-way through a --live run.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger("policy")

#: Set to the architecture already loaded in this process. See the module
#: docstring: two split policies do not fit in 8 GB alongside the robot stack.
_LOADED: str | None = None


def bundle_tasks(info: dict) -> list[str]:
    """The instruction strings a bundle records, in the order it records them.

    Lives here rather than beside either resolver because the two architectures
    spell everything else differently (export_info.json vs bundle.json) but must
    spell *this* the same: a checkpoint finetuned on a multi-task dataset records
    a ``tasks`` LIST, and every bundle written before multi-task existed records a
    single ``task`` string, which means exactly the one-element list. Resolving
    both to a list here is what lets the inference loop stop branching on which
    kind of bundle it was handed.

    The strings are the dataset's own, letter for letter — meta/tasks.parquet is
    where they come from — because that is what the language embedding was
    finetuned against.
    """
    tasks = info.get("tasks")
    if isinstance(tasks, str):          # an exporter that used the plural key for one
        return [tasks]
    if tasks:
        return [str(t) for t in tasks]
    task = info.get("task")
    return [str(task)] if task else []


def merge_tasks(explicit: list[str], recorded: list[str]) -> list[str]:
    """--task first, then every other task the bundle records.

    --task says which instruction the run STARTS on; it does not narrow what the
    checkpoint knows. Dropping the bundle's other tasks would mean naming one of
    them costs the operator the D-pad, which is the wrong trade: on a multi-task
    checkpoint every recorded phrasing is in distribution, and which one is live
    is a decision for the machine's side of the fence, not the command line.

    Duplicates collapse with the first spelling winning, so naming a recorded task
    REORDERS the list rather than doubling it.
    """
    merged = list(dict.fromkeys(explicit))
    merged += [task for task in recorded if task not in merged]
    return merged


def warn_off_bundle(explicit: list[str], recorded: list[str], where: str) -> None:
    """Warn about each --task the bundle does not list.

    Not an error: passing --task is a statement of intent, and probing how much
    the phrasing matters is a real thing to want. But it is the one setting whose
    being wrong is completely silent downstream, so it says so out loud.
    """
    for task in explicit:
        if recorded and task not in recorded:
            LOG.warning("--task %r is not among the tasks recorded in %s (%s). The "
                        "policy conditions on this string; a phrasing the "
                        "checkpoint was not finetuned on is out of distribution.",
                        task, where, ", ".join(repr(r) for r in recorded))


def detect_architecture(split_dir: str | Path) -> str:
    """"smolvla" or "xvla", from what the bundle ships. Refuses to guess."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise SystemExit(f"--split-dir {split_dir} does not exist.")

    is_xvla = (split_dir / "bundle.json").is_file()
    is_smolvla = ((split_dir / "smolvlm_vision.onnx").is_file()
                  or (split_dir / "export_info.json").is_file())

    if is_xvla and is_smolvla:
        raise SystemExit(
            f"{split_dir} carries both an X-VLA bundle.json and SmolVLA export "
            f"files, so the architecture is ambiguous. A bundle holds one model; "
            f"split them into separate directories.")
    if is_xvla:
        return "xvla"
    if is_smolvla:
        return "smolvla"
    raise SystemExit(
        f"{split_dir} is not a recognisable export bundle: it has neither "
        f"bundle.json (X-VLA) nor export_info.json / smolvlm_vision.onnx "
        f"(SmolVLA).\nPoint --split-dir at an export bundle, e.g. "
        f"lerobot_vla/model/smolvla-digging-clean-ir12-35k.")


def make_policy(architecture: str, split_dir: str | Path, **kwargs):
    """Construct the one policy this process will run.

    Architecture-specific keyword arguments are passed through; giving one that
    belongs to the other architecture is an error rather than a silent no-op,
    because every one of them changes what the machine does.
    """
    global _LOADED
    if _LOADED is not None:
        raise RuntimeError(
            f"A {_LOADED} policy is already loaded in this process and a "
            f"{architecture} one was requested. The two do not fit in 8 GB "
            f"together (see the module docstring) — switching models is a "
            f"restart, not a runtime toggle.")

    if architecture == "smolvla":
        from lerobot_vla.smolvla_split import SmolVLASplitPolicy
        policy = SmolVLASplitPolicy(split_dir=str(split_dir), **kwargs)
    elif architecture == "xvla":
        from lerobot_vla.xvla_split import XVLAExcavatorPolicy
        policy = XVLAExcavatorPolicy(split_dir=split_dir, **kwargs)
    else:
        raise ValueError(f"unknown architecture {architecture!r}")

    _LOADED = architecture
    return policy
