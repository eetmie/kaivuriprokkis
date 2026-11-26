"""Add an existing heightmap replay to a matching full native LeRobot export."""
import argparse
import errno
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ALIGNMENT = ("index", "episode_index", "frame_index", "timestamp")
MAP_VIEWS = ("observation.images.heightmap_front", "observation.images.heightmap_surround")
MAP_COLUMNS = ("diagnostics.mapping", "clock.depth_host_ns", "clock.native_frame")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    # The unchanged files are hardlinked to their inputs. Replace edited files.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parquet_paths(root, pattern):
    paths = sorted(root.glob(pattern))
    if not paths:
        raise ValueError(f"Missing {pattern} under {root}")
    return paths


def table_at(paths, columns=None):
    return pa.concat_tables([pq.read_table(p, columns=columns) for p in paths])


def clone(source, target):
    def link_or_copy(src, dst):
        try:
            return os.link(src, dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            return shutil.copy2(src, dst)
    shutil.copytree(source, target, copy_function=link_or_copy)


def write_columns(source, target, extra, source_root):
    offset = 0
    for path in source:
        table = pq.read_table(path)
        segment = extra.slice(offset, len(table))
        if len(segment) != len(table):
            raise ValueError("Chunk length mismatch")
        for key in segment.column_names:
            table = table.append_column(key, segment[key])
        destination = target / path.relative_to(source_root)
        temporary = destination.with_suffix(".tmp")
        pq.write_table(table.replace_schema_metadata(None), temporary)
        temporary.replace(destination)
        offset += len(table)
    if offset != len(extra):
        raise ValueError("Unconsumed map rows")


def augment(native, maps, out):
    native_info = read_json(native / "meta/info.json")
    map_info = read_json(maps / "meta/info.json")
    for key in ("fps", "total_episodes", "total_frames", "video_path", "codebase_version"):
        if native_info[key] != map_info[key]:
            raise ValueError(f"Native and map datasets disagree on {key}")

    native_data = parquet_paths(native, "data/*/*.parquet")
    map_data = parquet_paths(maps, "data/*/*.parquet")
    keys = [*ALIGNMENT, "observation.state", "action"]
    a = table_at(native_data, keys)
    b = table_at(map_data, [*keys, *MAP_COLUMNS])
    for key in ALIGNMENT:
        if a[key].to_pylist() != b[key].to_pylist():
            raise ValueError(f"Frame alignment mismatch in {key}")
    for key in keys[len(ALIGNMENT):]:
        np.testing.assert_allclose(a[key].to_pylist(), b[key].to_pylist(), atol=1e-5)

    native_episodes = parquet_paths(native, "meta/episodes/*/*.parquet")
    map_episodes = table_at(parquet_paths(maps, "meta/episodes/*/*.parquet"))
    episode_keys = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
    if table_at(native_episodes, episode_keys).to_pydict() != map_episodes.select(episode_keys).to_pydict():
        raise ValueError("Episode boundaries differ")
    map_meta = read_json(maps / "meta/heightmaps.json")
    for i, episode in enumerate(map_meta["episodes"]):
        source = read_json(native / "native" / f"episode-{i:06d}" / "source.json")
        if Path(source["source"]).name != episode["source"] or source["frames"] != episode["frames"]:
            raise ValueError(f"Source episode {i} differs")

    clone(native, out)
    for key in MAP_VIEWS:
        clone(maps / "videos" / key, out / "videos" / key)
    clone(maps / "heightmaps", out / "heightmaps")
    shutil.copy2(maps / "meta/heightmaps.json", out / "meta/heightmaps.json")

    write_columns(native_data, out, b.select(MAP_COLUMNS), native)
    added_meta = [name for name in map_episodes.column_names
                  if any(f"/{key}/" in name for key in (*MAP_VIEWS, *MAP_COLUMNS))]
    write_columns(native_episodes, out, map_episodes.select(added_meta), native)
    features = (*MAP_VIEWS, *MAP_COLUMNS)
    native_info["features"].update({key: map_info["features"][key] for key in features})
    write_json(out / "meta/info.json", native_info)
    native_stats = read_json(native / "meta/stats.json")
    map_stats = read_json(maps / "meta/stats.json")
    native_stats.update({key: map_stats[key] for key in features})
    write_json(out / "meta/stats.json", native_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("maps", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    augment(args.native, args.maps, args.out)
