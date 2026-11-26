"""Rebuild native session folders (symlinks, no copies) from a full LeRobot export's native/ tree,
so replay.py can re-run mapping and arm filters offline without the robot's raw sessions.

usage: sessions_from_export.py EXPORT_ROOT OUT_DIR   -> prints the session dirs to pass to replay.py
"""
import json
from pathlib import Path
import sys

export, out = (Path(a).resolve() for a in sys.argv[1:3])
native = export / "native"
sessions = {}
for episode in sorted(native.glob("episode-*")):
    source = json.loads((episode / "source.json").read_text())
    original = Path(source["source"])  # .../native_sessions/<name>/episode-<id>.ready
    src_session = native / f"session-{source['session']:03d}"
    dst = out / original.parent.name
    if dst not in sessions:
        dst.mkdir(parents=True, exist_ok=True)
        for item in ("session.json", "cam1.depth.json", "cam2.depth.json", "provenance"):
            if not (dst / item).exists():
                (dst / item).symlink_to(src_session / item)
        sessions[dst] = 0
    link = dst / original.name
    if not link.exists():
        link.symlink_to(episode)
    sessions[dst] += 1
assert sessions, f"{native}: no episode-* folders"
for path, n in sessions.items():
    print(path, f"# {n} episodes", file=sys.stderr)
print(" ".join(str(p) for p in sessions))
