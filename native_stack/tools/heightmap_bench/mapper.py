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
    parser.add_argument("--arm-min-view-deg", type=float, default=None,
                        help="drop the arm camera when its optical axis tips back past this "
                             "(0 = straight down, + forward, - toward the cabin); unset = never")
    parser.add_argument("--arm-silhouette-cells", type=int, default=-1,
                        help="drop depth pixels inside the arm model's projected silhouette grown by this many "
                             "strided cells; catches returns with wrong depth on the arm; -1 = off")
    parser.add_argument("--camera-mask-radius-m", type=float, default=0,
                        help="drop points within this of another camera (its body and cable); 0 = off")
    parser.add_argument("--arm-min-forward-m", type=float, default=None,
                        help="drop arm-camera points closer than this in front of the slew axis "
                             "(the cabin camera's near edge); unset = keep all")


def track_pivot(calibration, profile):
    """Apply the fallback only when native FK lacks track-end translation."""
    return (0 if profile.is_file() and "tilt_pivot_dist:" in profile.read_text()
            else calibration["track_pivot_m"])


class DepthProjector:
    def __init__(self, calibration, camera_files, args):
        self.stride = args.stride
        self.radius = args.arm_radius_m
        self.camera_radius = getattr(args, "camera_mask_radius_m", 0)
        self.silhouette = getattr(args, "arm_silhouette_cells", -1)
        self.solids = None  # arm model voxels per link, built on first use (needs sh.load_arm)
        # Every calibrated camera, even one not streaming here, so the other camera's body is masked.
        self.all_mounts = {k: sh.Mount.from_json(c) for k, c in calibration["cameras"].items()}
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
        others = []
        if self.camera_radius > 0:
            for other, mount in self.all_mounts.items():
                if other != key:
                    link_r, link_t = sh.link_pose(pose, mount.link)
                    others.append(link_t + link_r @ mount.t)
        sil = self.arm_silhouette(key, pose) if self.silhouette >= 0 and sh.ARM is not None else None
        return scan(data, self.rays[key], self.mounts[key], pose, self.stride, self.radius,
                    others, self.camera_radius, sil)

    def arm_silhouette(self, key, pose):
        """Strided-grid cells the boom/arm/bucket model covers in this camera, grown by self.silhouette."""
        arm = sh.ARM
        if self.solids is None:
            self.solids = [(k, lo + (np.argwhere(grid == 0) + .5) * arm.voxel, length)
                           for k, grid, lo, length in arm.links]
            self.pinholes = {}
        ray, mount = self.rays[key], self.mounts[key]
        if key not in self.pinholes:  # the stored (possibly rotated) image's pinhole, from its ray tables
            self.pinholes[key] = (np.polyfit(ray.x[ray.h // 2], np.arange(ray.w), 1),
                                  np.polyfit(ray.y[:, ray.w // 2], np.arange(ray.h), 1))
        chain = sh.fk_chain(pose)
        lateral = sh.link_pose(pose, 0)[0][:, 1]
        model = []
        for k, Q, length in self.solids:  # same link frame and stretch as ArmModel.near, inverted
            E = sh.link_frame(chain[k], chain[k + 1], lateral)
            R = Q.copy()
            R[:, 0] *= ((chain[k + 1] - chain[k]) @ E[0]) / length
            model.append(chain[k] + R @ E)
        link_r, link_t = sh.link_pose(pose, mount.link)
        M = (np.vstack(model) - (link_t + link_r @ mount.t)) @ (link_r @ mount.R)
        M = M[M[:, 2] > .05]
        (ax, bx), (ay, by) = self.pinholes[key]
        s = self.stride
        cu = np.round((ax * M[:, 0] / M[:, 2] + bx) / s).astype(int)
        cv = np.round((ay * M[:, 1] / M[:, 2] + by) / s).astype(int)
        gh, gw = (ray.h + s - 1) // s, (ray.w + s - 1) // s
        ok = (cu >= 0) & (cu < gw) & (cv >= 0) & (cv < gh)
        sil = np.zeros((gh, gw), bool)
        sil[cv[ok], cu[ok]] = True
        for _ in range(self.silhouette):
            grown = sil.copy()
            grown[1:] |= sil[:-1]; grown[:-1] |= sil[1:]; grown[:, 1:] |= sil[:, :-1]; grown[:, :-1] |= sil[:, 1:]
            sil = grown
        return sil


def scan(data, ray, mount, st, stride, arm_radius, cameras=(), camera_radius=0, silhouette=None):
    p, pixels = ray.points(data, stride, .90)
    link_r, link_t = sh.link_pose(st, mount.link)
    r = link_r @ mount.R
    t = link_t + link_r @ mount.t
    base = p @ r.T + t
    self_hit = sh.near_arm(base, st, arm_radius)
    for centre in cameras:  # base-frame origins of the other cameras
        self_hit |= np.einsum("ij,ij->i", base - centre, base - centre) < camera_radius ** 2
    if silhouette is not None:  # the arm hides everything behind it, whatever depth a pixel reports
        self_hit |= silhouette.reshape(-1)[pixels]
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


def view_deg(transform, yaw):
    """Optical-axis angle in the cabin plane: 0 = straight down, + forward, - back toward the cabin."""
    axis = np.asarray(transform)[:, 2]
    forward = axis[:2] @ sh.rz(yaw)[:2, :2]
    return float(np.degrees(np.arctan2(forward[0], -axis[2])))


def update(engine, scans, yaw, arm_min_view_deg=None, arm_min_forward_m=None):
    """Integrate one latest-frame observation and return front, surround and age."""
    pc, wc, tc, _, _ = scans[0]
    front = raster(wc, yaw)
    inputs = [(pc, tc)]
    # Looking back past the limit the arm camera sees the machine and far clutter, not the dig area.
    if len(scans) == 2 and (arm_min_view_deg is None or view_deg(scans[1][2], yaw) >= arm_min_view_deg):
        pa, wa, ta, _, _ = scans[1]
        if arm_min_forward_m is not None:  # nothing nearer than where the cabin camera starts seeing
            keep = (wa[:, :2] @ sh.rz(yaw)[:2, :2])[:, 0] >= arm_min_forward_m
            pa, wa = pa[keep], wa[keep]
        front = np.where(np.isfinite(front), front, raster(wa, yaw))
        inputs.append((pa[arm_fill(wc, wa)], ta))
    surround, age = engine.update(*inputs)
    return front, surround, age
