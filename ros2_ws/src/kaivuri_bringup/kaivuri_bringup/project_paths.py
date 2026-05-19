import os
import sys
from pathlib import Path


def resolve_project_root(configured_root: str) -> Path:
    """Return the Kaivuri project root, falling back from stale launch defaults."""
    candidates = []

    if configured_root:
        candidates.append(Path(configured_root).expanduser())

    env_root = os.environ.get("KAIVURI_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    for parent in Path(__file__).resolve().parents:
        candidates.append(parent)

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "modules").is_dir() and (resolved / "configuration_files").is_dir():
            return resolved

    return Path(configured_root or ".").expanduser().resolve()


def add_project_import_path(project_root: Path) -> None:
    root = str(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
