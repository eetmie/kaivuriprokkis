"""Shared fixed-grid geometry and FastDEM adapter for offline and live use."""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

import sand_height as sh


STRIDE = 4
FRONT_SIZE = 64
FRONT_CELL = .0125
SURROUND_SIZE = 160
SURROUND_CELL = .01
NEAR_RADIUS = .14
FAR_RADIUS = .80
HEIGHT_LIMIT = .12
UNKNOWN_MM = -32768
KEYS = ("cam2", "cam1")  # cabin, arm
MODES = ("cabin-only", "cabin-arm-fused")
SURROUND_X = .8 - (np.arange(SURROUND_SIZE) + .5) * SURROUND_CELL
ANNULUS = (np.hypot(SURROUND_X[:, None], SURROUND_X) >= NEAR_RADIUS) & (
    np.hypot(SURROUND_X[:, None], SURROUND_X) <= FAR_RADIUS
)


def rotation_lerp(a, b, weight):
    a, b = np.asarray(a), np.asarray(b)
    dot = (a * b).sum(axis=-1, keepdims=True)
    q = (1 - weight) * a + weight * np.where(dot < 0, -b, b)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def pose_at(t, times, poses, pivot):
    after = int(np.searchsorted(times, t))
    if after == 0 or after == len(times):
        st = poses[0] if after == 0 else poses[-1]
    else:
        a, b = poses[after - 1], poses[after]
        w = (t - times[after - 1]) / (times[after] - times[after - 1])
        st = dict(a)
        for k in ("fk_link_quat_wxyz", "cabin_tilt_quat_wxyz"):
            st[k] = rotation_lerp(a[k], b[k], w)
        for k in ("fk_joint_xyz_m", "fk_xyz_m"):
            st[k] = (1 - w) * np.asarray(a[k]) + w * np.asarray(b[k])
        delta_yaw = (b["slew_yaw_rad"] - a["slew_yaw_rad"] + np.pi) % (2 * np.pi) - np.pi
        st["slew_yaw_rad"] = a["slew_yaw_rad"] + w * delta_yaw
    if not pivot:
        return st
    axis = sh.quat_matrix(st["cabin_tilt_quat_wxyz"])[:, 0]
    end = -pivot if axis[2] > 0 else pivot
    shift = end * (np.array([1, 0, 0]) - axis)
    return {**st, "fk_joint_xyz_m": np.asarray(st["fk_joint_xyz_m"]) + shift,
            "fk_xyz_m": np.asarray(st["fk_xyz_m"]) + shift}


def pipeline_defaults(parser, sections):
    candidates = (Path(__file__).parent / "config/heightmap_pipeline.json",
                  Path(__file__).resolve().parents[2] / "config/heightmap_pipeline.json")
    path = next((p for p in candidates if p.is_file()), candidates[0])
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--pipeline-config", type=Path, default=path)
    selected, _ = selector.parse_known_args()
    parser.add_argument("--pipeline-config", type=Path, default=path)
    try:
        config = json.loads(selected.pipeline_config.read_text())
        parser.set_defaults(**{key: value for section in sections for key, value in config[section].items()})
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"{selected.pipeline_config}: {error}")


def projection_arguments(parser):
    sh.arm_arguments(parser)
    parser.add_argument("--stride", type=int, choices=range(1, 9), default=STRIDE)
    parser.add_argument("--arm-radius-m", type=float, default=.11,
                        help="capsule radius when --arm-model none")


def track_pivot(calibration, profile):
    """Apply the fallback only when native FK lacks track-end translation."""
    return (0 if profile.is_file() and "tilt_pivot_dist:" in profile.read_text()
            else calibration["track_pivot_m"])


class DepthProjector:
    def __init__(self, calibration, camera_files, args):
        self.stride = args.stride
        self.radius = args.arm_radius_m
        self.rays, self.mounts, self.metadata = {}, {}, {}
        for key, path in camera_files.items():
            info = json.loads(path.read_text())
            if info["serial"] != calibration["serials"][key]:
                raise ValueError(f"{path}: calibration serial mismatch for {key}")
            self.metadata[key] = info
            ray = sh.Rays(path)
            ray.set_correction(calibration["depth_correction"].get(key))
            self.rays[key] = ray
            self.mounts[key] = sh.Mount.from_json(calibration["cameras"][key])

    def scan(self, key, data, pose):
        return scan(data, self.rays[key], self.mounts[key], pose, self.stride, self.radius)


def scan(data, ray, mount, st, stride, arm_radius):
    p, pixels = ray.points(data, stride, .90)
    link_r, link_t = sh.link_pose(st, mount.link)
    r = link_r @ mount.R
    t = link_t + link_r @ mount.t
    base = p @ r.T + t
    self_hit = sh.near_arm(base, st, arm_radius)
    t[2] += mount.d
    world = base.copy()
    world[:, 2] += mount.d
    radius = np.linalg.norm(world[:, :2], axis=1)
    use = (~self_hit & (radius >= NEAR_RADIUS) & (radius <= FAR_RADIUS)
           & (np.abs(world[:, 2]) <= .30) & (p[:, 2] >= .10))
    mask = np.zeros(((ray.h + stride - 1) // stride, (ray.w + stride - 1) // stride), np.uint8)
    mask.flat[pixels] = 1
    mask.flat[pixels[self_hit]] = 2
    depth = ray.depth(data, stride)
    return p[use], world[use], np.column_stack((r, t)), depth, mask


def cells(xy, xmax, ymax, size, cell):
    indices = np.floor((np.array([xmax, ymax]) - xy) / cell).astype(int)
    good = ((indices >= 0) & (indices < size)).all(axis=1)
    return good, indices[:, 0] * size + indices[:, 1]


def raster(world, yaw):
    xy = world[:, :2] @ sh.rz(yaw)[:2, :2]
    good, indices = cells(xy, .9, .4, FRONT_SIZE, FRONT_CELL)
    idx, z = indices[good], world[good, 2]
    order = np.lexsort((z, idx))
    idx, z = idx[order], z[order]
    ids, starts, counts = np.unique(idx, return_index=True, return_counts=True)
    result = np.full(FRONT_SIZE**2, np.nan, dtype=np.float32)
    result[ids] = (z[starts + (counts - 1)//2] + z[starts + counts//2]) / 2
    return result.reshape(FRONT_SIZE, FRONT_SIZE)


def arm_fill(cab_world, arm_world):
    cg, ci = cells(cab_world[:, :2], .8, .8, SURROUND_SIZE, SURROUND_CELL)
    ag, ai = cells(arm_world[:, :2], .8, .8, SURROUND_SIZE, SURROUND_CELL)
    seen = np.zeros(SURROUND_SIZE**2, bool)
    seen[ci[cg]] = True
    fill = ag.copy()
    fill[ag] = ~seen[ai[ag]]
    return fill


class FastDEM:
    def __init__(self, exe, config):
        self.exe = exe.resolve()
        self.config = config

    def __enter__(self):
        self.proc = subprocess.Popen([self.exe, self.config], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        return self

    def __exit__(self, kind, value, trace):
        self.proc.stdin.close()
        self.proc.stdout.close()
        if self.proc.wait() and kind is None:
            raise RuntimeError(f"FastDEM exited with {self.proc.returncode}")

    def update(self, *inputs):
        stream = self.proc.stdin
        stream.write(struct.pack("<I", len(inputs)))
        for points, transform in inputs:
            stream.write(struct.pack("<I", len(points)))
            stream.write(np.asarray(transform, dtype="<f8").tobytes())
            stream.write(np.asarray(points, dtype="<f4").tobytes())
        stream.flush()
        size = SURROUND_SIZE**2
        output = self.proc.stdout.read(size * 6)
        if len(output) != size * 6:
            raise RuntimeError("incomplete response from FastDEM")
        h = np.frombuffer(output, "<f4", size).copy().reshape(ANNULUS.shape)
        age = np.frombuffer(output, "<u2", size, size * 4).copy().reshape(ANNULUS.shape)
        h[~ANNULUS] = np.nan
        age[~np.isfinite(h)] = 65535
        return h, age


def render(height):
    valid = np.isfinite(height)
    gray = np.zeros(height.shape, np.uint8)
    gray[valid] = np.round(128 + 112 * np.clip(height[valid] / HEIGHT_LIMIT, -1, 1)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def update(engine, scans, yaw):
    """Integrate one latest-frame observation and return front, surround and age."""
    pc, wc, tc, _, _ = scans[0]
    front = raster(wc, yaw)
    inputs = [(pc, tc)]
    if len(scans) == 2:
        pa, wa, ta, _, _ = scans[1]
        front = np.where(np.isfinite(front), front, raster(wa, yaw))
        inputs.append((pa[arm_fill(wc, wa)], ta))
    surround, age = engine.update(*inputs)
    return front, surround, age
