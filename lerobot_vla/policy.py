"""X-VLA task metadata and the single-policy loader for this Python project.

Models are explicit bundles. Keep one policy per process and restart to change
models, so a model replacement cannot overlap GPU allocations during robot use.
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
    """Recognize an X-VLA bundle and reject incompatible export formats."""
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
        raise SystemExit("This Python project runs X-VLA bundles only; expected bundle.json.")
    raise SystemExit(f"{split_dir} is not an X-VLA bundle: bundle.json is missing.")


def make_policy(architecture: str, split_dir: str | Path, **kwargs):
    """Construct the one policy this process will run.

    Architecture-specific keyword arguments are passed through; giving one that
    belongs to the other architecture is an error rather than a silent no-op,
    because every one of them changes what the machine does.
    """
    global _LOADED
    if _LOADED is not None:
        raise RuntimeError(f"A {_LOADED} policy is already loaded; switch models with a restart.")

    if architecture == "xvla":
        from lerobot_vla.runtime.xvla import XVLAExcavatorPolicy
        policy = XVLAExcavatorPolicy(split_dir=split_dir, **kwargs)
    else:
        raise ValueError(f"unknown architecture {architecture!r}")

    _LOADED = architecture
    return policy
