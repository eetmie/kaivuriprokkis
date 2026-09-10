#!/usr/bin/env python3
"""One-off migration: fold a dataset's `action.tracks` column into `action`.

Recordings made between 2026-09-09 and 2026-09-10 carried the track commands in
a sibling `action.tracks` feature. That is not how lerobot lays out an action:
`hw_to_dataset_features` puts every actuator -- grippers included -- into one
flat `action` vector and lists them in `names`, and the tooling reads only that.
A sibling column is prefix-matched into FeatureType.ACTION, lands in the
policy's output_features, and is then never fed to the model (the pipeline only
ever transforms the literal key `action`); the dataset viewer skips it too. The
recorder now appends [trackL, trackR] to `action` under --enable-tracks, and
this brings the already-recorded datasets to that layout.

    .venv-lerobot/bin/python -m lerobot_vla.tools.fold_tracks_into_action \
        data_collection/lerobot_datasets/masi_digging_new_imu --backup /tmp/bk

Rewrites, in place: every data parquet (`action` 4 -> 6 wide, `action.tracks`
dropped), the per-episode stats in meta/episodes/, meta/stats.json and
meta/info.json. Videos are untouched, which is why --backup only has to copy
data/ and meta/ -- a few MB against a few hundred.

Nothing is recomputed. Every statistic lerobot keeps is per-dimension, so the
6-wide value is the 4-wide one with the 2-wide one appended, exactly what a
recompute would produce; `count` is frames, identical for both, and is kept.

Idempotent: a dataset with no `action.tracks` is reported and left alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ACTION = "action"
TRACKS = "action.tracks"
TRACK_NAMES = ["trackL", "trackR"]

# Per-episode stats live as `stats/<feature>/<stat>` columns. Every one of these
# is per-dimension and so concatenates, except count, which is a frame tally.
_CONCAT_STATS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def _write_atomic(table: pa.Table, path: Path) -> None:
    """Write beside the target and rename, so an interrupted run loses nothing.

    The rename is atomic within the directory; a half-written .tmp left behind
    by a kill is inert, since nothing reads it.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="snappy")
    tmp.replace(path)


def _fold_column(table: pa.Table) -> pa.Table:
    """Replace `action` with action||tracks, in place in the column order."""
    n = table.num_rows
    act = np.asarray(table[ACTION].to_numpy(zero_copy_only=False).tolist(),
                     dtype=np.float32).reshape(n, -1)
    trk = np.asarray(table[TRACKS].to_numpy(zero_copy_only=False).tolist(),
                     dtype=np.float32).reshape(n, -1)
    merged = np.concatenate([act, trk], axis=1)
    width = merged.shape[1]
    col = pa.FixedSizeListArray.from_arrays(
        pa.array(merged.reshape(-1), type=pa.float32()), width)
    return (table
            .set_column(table.schema.get_field_index(ACTION), ACTION, col)
            .drop_columns([TRACKS]))


def _fold_episode_stats(table: pa.Table) -> pa.Table:
    """Append the track dimension to every per-episode `stats/action/*` array."""
    for stat in _CONCAT_STATS:
        a_col, t_col = f"stats/{ACTION}/{stat}", f"stats/{TRACKS}/{stat}"
        if a_col not in table.column_names or t_col not in table.column_names:
            continue
        merged = [None if a is None or t is None else list(a) + list(t)
                  for a, t in zip(table[a_col].to_pylist(), table[t_col].to_pylist())]
        table = table.set_column(table.schema.get_field_index(a_col), a_col,
                                 pa.array(merged, type=table.schema.field(a_col).type))
    drop = [c for c in table.column_names if c.startswith(f"stats/{TRACKS}/")]
    return table.drop_columns(drop)


def _fold_stats_json(stats: dict) -> dict:
    action, tracks = stats[ACTION], stats.pop(TRACKS)
    for stat in _CONCAT_STATS:
        if stat in action and stat in tracks:
            action[stat] = list(action[stat]) + list(tracks[stat])
    return stats


def migrate(root: Path, backup: Path | None = None, dry_run: bool = False) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"{root}: no meta/info.json — not a LeRobot dataset")
        return 1
    info = json.loads(info_path.read_text())
    if TRACKS not in info["features"]:
        print(f"{root}: no {TRACKS} feature — already folded, nothing to do")
        return 0

    old_names = list(info["features"][ACTION]["names"])
    new_names = old_names + list(info["features"][TRACKS]["names"])
    data_files = sorted((root / "data").rglob("*.parquet"))
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    print(f"{root}:")
    print(f"  action {tuple(info['features'][ACTION]['shape'])} -> ({len(new_names)},)"
          f"  {','.join(old_names)} -> {','.join(new_names)}")
    print(f"  {len(data_files)} data file(s), {len(ep_files)} episode-meta file(s)")
    if dry_run:
        print("  --dry-run: nothing written")
        return 0

    if backup is not None:
        dest = backup / root.name
        if dest.exists():
            print(f"  backup {dest} already exists — refusing to overwrite it")
            return 1
        # data/ and meta/ only: videos are not touched by this migration, and
        # copying them would need hundreds of MB the disk does not have.
        dest.mkdir(parents=True)
        for sub in ("data", "meta"):
            shutil.copytree(root / sub, dest / sub)
        print(f"  backed up data/ + meta/ to {dest}")

    for path in data_files:
        _write_atomic(_fold_column(pq.read_table(path)), path)
        print(f"  rewrote {path.relative_to(root)}")
    for path in ep_files:
        _write_atomic(_fold_episode_stats(pq.read_table(path)), path)
        print(f"  rewrote {path.relative_to(root)}")

    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        stats_path.write_text(json.dumps(
            _fold_stats_json(json.loads(stats_path.read_text())), indent=4))
        print("  rewrote meta/stats.json")

    info["features"][ACTION]["shape"] = [len(new_names)]
    info["features"][ACTION]["names"] = new_names
    del info["features"][TRACKS]
    info_path.write_text(json.dumps(info, indent=4))
    print("  rewrote meta/info.json")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("roots", nargs="+", type=Path,
                   help="Dataset root directories (the ones holding meta/info.json).")
    p.add_argument("--backup", type=Path, default=None,
                   help="Copy each dataset's data/ and meta/ under here first. "
                        "Videos are not copied; the migration never touches them.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change and exit.")
    args = p.parse_args()
    return max(migrate(r, args.backup, args.dry_run) for r in args.roots)


if __name__ == "__main__":
    sys.exit(main())
