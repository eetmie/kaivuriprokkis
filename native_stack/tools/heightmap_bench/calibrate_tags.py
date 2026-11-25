"""Fit camera-to-link mounts from fixed AprilTags, native IR timestamps and FK."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from export_episodes import frame_rows, samples
from native_sessions import committed_episodes
import sand_height as sh
from mapper import pose_at, track_pivot

CAMERAS = ("cam1", "cam2")


def load_episode(episode, number, detections, prior, stride):
    poses = {s["host_monotonic_ns"]: s for row in samples(episode)
             if (s := row["status"])["fk_valid"] and s["imu_healthy"]}
    times = sorted(poses)
    statuses = [poses[t] for t in times]
    frames = {r["frame_index"]: r for r in frame_rows(episode)}
    profile = episode.parent / "provenance/configuration_files/profiles/jetson/control_config.yaml"
    pivot = track_pivot(prior, profile)
    cameras, tags, pixels, rotations, translations, intrinsics, ir_r, ir_t = ([] for _ in range(8))
    for c, key in enumerate(CAMERAS):
        info = json.loads((episode.parent / f"{key}.depth.json").read_text())
        if info["serial"] != prior["serials"][key]:
            raise ValueError(f"Camera serial mismatch: {key}")
        intr = info["ir_intrinsics"]
        if any(intr["coeffs"]):
            raise ValueError("This calibrator expects rectified IR")
        ext = info["depth_to_ir_extrinsics"]
        for row in np.load(detections / f"ep{number}-{key}.npy"):
            f, tag, hamming = map(int, row[:3])
            if hamming or f % stride:
                continue
            stamp = frames[f][key]["host_monotonic_ns"]
            pose = pose_at(stamp, times, statuses, pivot)
            r, t = sh.link_pose(pose, prior["cameras"][key]["link"])
            uv = row[6:].reshape(4, 2)
            if info["rotation_deg"] == 180:
                uv = np.array([intr["width"] - 1, intr["height"] - 1]) - uv
            cameras.append(c)
            tags.append(tag)
            pixels.append(uv)
            rotations.append(r)
            translations.append(t)
            intrinsics.append([intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]])
            ir_r.append(np.reshape(ext["rotation_column_major"], (3, 3), order="F"))
            ir_t.append(ext["translation_m"])
    return tuple(map(np.asarray, (cameras, tags, pixels, rotations, translations, intrinsics, ir_r, ir_t)))


def load_ground(episode, prior, every):
    poses = {s["host_monotonic_ns"]: s for row in samples(episode)
             if (s := row["status"])["fk_valid"] and s["imu_healthy"]}
    times = sorted(poses)
    statuses = [poses[t] for t in times]
    frames = list(frame_rows(episode))
    profile = episode.parent / "provenance/configuration_files/profiles/jetson/control_config.yaml"
    pivot = track_pivot(prior, profile)
    ground = []
    for key in CAMERAS:
        ray = sh.Rays(episode.parent, key)
        ray.set_correction(prior["depth_correction"][key])
        points, up, offset = [], [], []
        for index, _, cloud, _, _ in sh.flat_samples(episode, ray, every=every, per_frame=80):
            stamp = frames[index][f"{key}_depth"]["host_monotonic_ns"]
            pose = pose_at(stamp, times, statuses, pivot)
            r, t = sh.link_pose(pose, prior["cameras"][key]["link"])
            points.extend(cloud)
            up.extend(np.broadcast_to(r[2], cloud.shape))
            offset.extend(np.full(len(cloud), t[2]))
        ground.append(tuple(map(np.asarray, (points, up, offset))))
    return ground


def ground_error(ground, mounts, delta):
    result = []
    for (points, up, offset), mount, d in zip(ground, mounts, delta):
        rotation = Rotation.from_rotvec(d[:3]).as_matrix() @ mount.R
        local = points @ rotation.T + mount.t + d[3:]
        result.append((np.einsum("ij,ij->i", local, up) + offset + mount.d) * 1000)
    return result


def ground_stats(ground, mounts):
    return {k: dict(zip(("p50_abs_mm", "p90_abs_mm", "median_signed_mm"),
                       [*np.percentile(np.abs(z), [50, 90]).tolist(), float(np.median(z))]))
            for k, z in zip(CAMERAS, ground_error(ground, mounts, np.zeros((2, 6))))}


def error_stats(predicted, pixels, camera):
    error = np.linalg.norm(predicted - pixels, axis=-1)
    return {k: dict(zip(("p50_px", "p90_px"), np.percentile(error[camera == i], [50, 90]).tolist()))
            for i, k in enumerate(CAMERAS) if np.any(camera == i)}


def solve(data, mounts, corners, fit_mounts=False, target_camera=None, ground=None, ground_weight=1.):
    camera, tag, uv, link_r, link_t, intr, ir_r, ir_t = data
    ids, tag_idx = np.unique(tag, return_inverse=True)
    mount_r = np.stack([m.R for m in mounts])
    mount_t = np.stack([m.t for m in mounts])
    x0 = np.zeros((len(ids), 6))
    used = np.ones(len(tag), bool) if target_camera is None else camera == target_camera
    for i in range(len(ids)):
        candidates = []
        for j in np.flatnonzero((tag_idx == i) & used)[::4]:
            fx, fy, px, py = intr[j]
            matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
            ok, rv, tv = cv2.solvePnP(corners, uv[j].copy(), matrix, None)
            if ok:
                r = link_r[j] @ mount_r[camera[j]] @ ir_r[j].T
                t = link_t[j] + link_r[j] @ mount_t[camera[j]]
                candidates.append(np.r_[Rotation.from_matrix(r @ Rotation.from_rotvec(rv[:, 0]).as_matrix()).as_rotvec(),
                                         r @ (tv[:, 0] - ir_t[j]) + t])
        if not candidates:
            raise ValueError(f"No initializer for tag {ids[i]}")
        candidates = np.asarray(candidates)
        center = np.median(candidates[:, 3:], axis=0)
        closest = np.argmin(np.linalg.norm(candidates[:, 3:] - center, axis=1))
        x0[i] = np.r_[candidates[closest, :3], center]

    def model(x):
        camera_delta = x[:12].reshape(2, 6) if fit_mounts else np.zeros((2, 6))
        target = x[12 if fit_mounts else 0:].reshape(-1, 6)
        r = (Rotation.from_rotvec(camera_delta[:, :3]).as_matrix() @ mount_r)[camera]
        t = (mount_t + camera_delta[:, 3:])[camera]
        world = corners @ Rotation.from_rotvec(target[:, :3]).as_matrix()[tag_idx].transpose(0, 2, 1)
        world += target[tag_idx, None, 3:]
        depth = ((world - link_t[:, None]) @ link_r - t[:, None]) @ r
        ir = depth @ ir_r.transpose(0, 2, 1) + ir_t[:, None]
        return ir[:, :, :2] / np.maximum(ir[:, :, 2:3], .03) * intr[:, None, :2] + intr[:, None, 2:]

    def residual(x):
        pixel = (model(x)[used] - uv[used]).ravel()
        if not fit_mounts:
            return pixel
        delta = x[:12].reshape(2, 6)
        regularization = (delta * [20, 20, 20, 80, 80, 80]).ravel()
        floor = np.concatenate(ground_error(ground, mounts, delta)) * ground_weight if ground else []
        return np.r_[pixel, regularization, floor]

    offset = 12 if fit_mounts else 0
    chosen = np.flatnonzero(used)
    ground_rows = sum(len(g[0]) for g in ground) if ground else 0
    sparsity = lil_matrix((8 * len(chosen) + offset + ground_rows, 6 * len(ids) + offset), dtype=int)
    for i, j in enumerate(chosen):
        rows = slice(8 * i, 8 * (i + 1))
        start = offset + 6 * tag_idx[j]
        sparsity[rows, start:start + 6] = 1
        if fit_mounts:
            sparsity[rows, 6 * camera[j]:6 * (camera[j] + 1)] = 1
    if fit_mounts:
        for i in range(12):
            sparsity[8 * len(chosen) + i, i] = 1
    start = 8 * len(chosen) + offset
    for i, g in enumerate(ground or []):
        stop = start + len(g[0])
        sparsity[start:stop, 6*i:6*(i+1)] = 1
        start = stop
    initial = np.r_[np.zeros(offset), x0.ravel()]
    fit = least_squares(residual, initial, jac_sparsity=sparsity.tocsr(), loss="soft_l1",
                        f_scale=2, x_scale="jac", ftol=1e-5, max_nfev=350)
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x, error_stats(model(fit.x), uv, camera)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session", type=Path)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag-edge-mm", type=float, required=True)
    p.add_argument("--fit-episode", type=int, default=1)
    p.add_argument("--test-episode", type=int, default=3)
    p.add_argument("--flat-session", type=Path)
    p.add_argument("--ground-weight", type=float, default=4.,
                   help="Pixel residual units per millimetre of zero-height error")
    args = p.parse_args()
    if args.fit_episode == args.test_episode or 2 in (args.fit_episode, args.test_episode):
        p.error("Use separate fixed-target takes. Episode 2 contains a moved target.")
    prior = json.loads(args.prior.read_text())
    episodes = {json.loads((ep / "episode.json").read_text())["episode_number"]: ep
                for ep in committed_episodes(args.session)}
    train, test = (load_episode(episodes[n], n, args.detections, prior, stride)
                   for n, stride in ((args.fit_episode, 25), (args.test_episode, 5)))
    corners = np.array([[-1., -1., 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * args.tag_edge_mm / 2000
    mounts = [sh.Mount.from_json(prior["cameras"][k]) for k in CAMERAS]
    ground = None
    if args.flat_session:
        floor_episodes = committed_episodes(args.flat_session)
        ground = load_ground(floor_episodes[1], prior, every=20)
        test_ground = load_ground(floor_episodes[0], prior, every=20)
    delta, fit_errors = solve(train, mounts, corners, fit_mounts=True,
                              ground=ground, ground_weight=args.ground_weight)
    delta = delta[:12].reshape(2, 6)
    updated = [sh.Mount(m.link, Rotation.from_rotvec(d[:3]).as_matrix() @ m.R,
                        m.t + d[3:], m.n, m.d) for m, d in zip(mounts, delta)]

    def held_out(candidate):
        return {k: solve(test, candidate, corners, target_camera=i)[1][CAMERAS[1-i]]
                for i, k in enumerate(CAMERAS)}

    report = {"fit_episode": args.fit_episode, "test_episode": args.test_episode,
              "excluded_episodes": [2], "tag_edge_mm": args.tag_edge_mm,
              "fit_observations": len(train[0]), "test_observations": len(test[0]),
              "train_prior": solve(train, mounts, corners)[1], "train_candidate": fit_errors,
              "held_out_prior_by_source": held_out(mounts),
              "held_out_candidate_by_source": held_out(updated),
              "delta": {k: {"rotation_deg": float(np.linalg.norm(np.rad2deg(d[:3]))),
                            "translation_mm": (d[3:] * 1000).tolist()} for k, d in zip(CAMERAS, delta)}}
    if ground:
        report["ground"] = {
            "weight": args.ground_weight,
            "train_prior": ground_stats(ground, mounts),
            "train_candidate": ground_stats(ground, updated),
            "held_out_prior": ground_stats(test_ground, mounts),
            "held_out_candidate": ground_stats(test_ground, updated),
        }
    result = {**prior, "cameras": {k: {**prior["cameras"][k], **m.to_json()}
                                   for k, m in zip(CAMERAS, updated)}, "april_tag_calibration": report}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
