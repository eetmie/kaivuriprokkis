#!/usr/bin/env python3
"""Offline feasibility test: can the D435i IR stream correct slew-gyro drift?

Reads one episode of a LeRobot dataset recorded by record_episodes.py and
measures how well plain phase correlation on the cab camera tracks the slew
yaw. Nothing here touches the hardware; it is the go/no-go before building a
real-time yaw filter.

The idea being tested: a cab-mounted camera that slews only *rotates*, and a
rotation about the vertical axis is (to first order) a horizontal image shift,
    yaw = atan(shift_px / fx)
so no map or SLAM is needed. The gyro stays the 100 Hz yaw source; the camera
only has to supply an occasional absolute fix against keyframes.

Three tests, each printed in the summary:

  1. incremental  frame-to-frame shift vs the gyro's yaw change. Fits the sign
                  and effective fx, and shows the per-frame noise.
  2. keyframe     keyframes are taken during the first --sweep-s seconds (the
                  operator slews through the working range there); every later
                  sample is matched to the nearest keyframe for an ABSOLUTE yaw.
                  Its error against the gyro is the number that matters.
  3. fusion       a gyro with --inject-drift-dps of artificial bias is corrected
                  by the keyframe fixes in a complementary filter, to show the
                  drift is actually removed (real drift over one episode is too
                  small to see).

Tests 2 and 3 need slew in observation.state. Datasets recorded with the
default --state-joints lack it; on those the tool falls back to checking the
image shift against the slew valve command (sign and timing only).

Record a test episode (slew back and forth over the working range in the first
~20 s, then dig normally, boom moving):
    .venv-lerobot/bin/python -m lerobot_vla.record_episodes \\
        --repo-id masi/yaw_probe --task "yaw probe" \\
        --state-joints slew,lift,tilt,scoop

Analyse it:
    .venv-lerobot/bin/python -m lerobot_vla.tools.visual_yaw_probe \\
        data_collection/lerobot_datasets/masi_yaw_probe --episode 0

Check --out/roi_preview.png first: the ROI (green box) must not contain the
boom, arm or bucket. Those rotate with the cab, look stationary to the camera,
and pull every measurement toward zero rotation.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import av
import cv2

CAM_KEYS = {"cam1": "observation.images.cam1", "cam2": "observation.images.cam2"}


# ── dataset access ───────────────────────────────────────────────────────────

def load_episode_meta(root: Path, episode: int) -> pd.Series:
    files = sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))
    eps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    row = eps[eps["episode_index"] == episode]
    if row.empty:
        raise SystemExit(f"episode {episode} not in {root} "
                         f"(have 0..{int(eps['episode_index'].max())})")
    return row.iloc[0]


def load_episode_rows(root: Path, ep: pd.Series) -> pd.DataFrame:
    path = root / f"data/chunk-{int(ep['data/chunk_index']):03d}/file-{int(ep['data/file_index']):03d}.parquet"
    df = pd.read_parquet(path)
    df = df[df["episode_index"] == int(ep["episode_index"])].sort_values("frame_index")
    return df.reset_index(drop=True)


def decode_episode_gray(root: Path, ep: pd.Series, video_key: str,
                        n_frames: int, fps: float) -> list[np.ndarray]:
    """Decode this episode's frames as HxW uint8 gray.

    v3 datasets concatenate episodes into shared mp4 files, so the episode is
    the [from_timestamp, to_timestamp) slice of its file.
    """
    pre = f"videos/{video_key}"
    path = root / f"{pre}/chunk-{int(ep[pre + '/chunk_index']):03d}/file-{int(ep[pre + '/file_index']):03d}.mp4"
    t0 = float(ep[pre + "/from_timestamp"])
    half = 0.5 / fps
    frames: list[np.ndarray] = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        c.seek(int(max(0.0, t0 - 1.0) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            if fr.pts is None or fr.pts * st.time_base < t0 - half:
                continue
            frames.append(fr.to_ndarray(format="gray"))
            if len(frames) >= n_frames:
                break
    if len(frames) < n_frames:
        print(f"[warn] decoded {len(frames)} of {n_frames} frames")
    return frames


# ── vision ───────────────────────────────────────────────────────────────────

class Matcher:
    """Crop + downsample + windowed phase correlation, as it would run online."""

    def __init__(self, shape: tuple[int, int], roi: tuple[float, float, float, float],
                 scale: int):
        h, w = shape
        y0, y1, x0, x1 = roi
        self.sl = (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))
        ch = self.sl[0].stop - self.sl[0].start
        cw = self.sl[1].stop - self.sl[1].start
        self.size = (max(16, cw // scale), max(16, ch // scale))   # cv2 wants (w, h)
        self.scale = cw / self.size[0]                              # full-res px per small px
        self.win = cv2.createHanningWindow(self.size, cv2.CV_32F)

    def prep(self, gray: np.ndarray) -> np.ndarray:
        small = cv2.resize(gray[self.sl], self.size, interpolation=cv2.INTER_AREA)
        return small.astype(np.float32)

    def shift(self, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        """Horizontal shift b relative to a in FULL-RES pixels, and the peak response."""
        (dx, _dy), resp = cv2.phaseCorrelate(a, b, self.win)
        return dx * self.scale, resp


def px_to_deg(dx_px: np.ndarray | float, fx: float) -> np.ndarray | float:
    return np.degrees(np.arctan(np.asarray(dx_px) / fx))


# ── analysis ─────────────────────────────────────────────────────────────────

def fit_gain(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares k in y ≈ k·x (through the origin)."""
    den = float(np.dot(x, x))
    return float(np.dot(x, y) / den) if den > 0 else float("nan")


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a)))) if len(a) else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dataset", type=Path, help="Dataset root (the dir holding meta/, data/, videos/)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--cam", choices=sorted(CAM_KEYS), default="cam1",
                   help="cam1 = IR left imager (default), cam2 = RGB")
    p.add_argument("--roi", default="0.0,0.45,0.0,1.0",
                   help="y0,y1,x0,x1 as image fractions. Keep the boom OUT of it. "
                        "Default: top 45%% of the frame")
    p.add_argument("--scale", type=int, default=4, help="Downsample factor before matching")
    p.add_argument("--fx", type=float, default=None,
                   help="Focal length in px at the recorded resolution "
                        "(default: ~385 IR / ~615 RGB at 640x480; the incremental "
                        "fit reports the effective value)")
    p.add_argument("--min-response", type=float, default=0.08,
                   help="Reject matches whose phase-correlation peak is below this")
    p.add_argument("--sweep-s", type=float, default=20.0,
                   help="Keyframes are only taken in the first N seconds")
    p.add_argument("--kf-step-deg", type=float, default=15.0,
                   help="Keyframe spacing in yaw")
    p.add_argument("--kf-max-rate", type=float, default=15.0,
                   help="Only take keyframes / fixes when |slew rate| is below this, deg/s "
                        "(fast slews blur the image)")
    p.add_argument("--fix-hz", type=float, default=5.0, help="Keyframe fix rate")
    p.add_argument("--inject-drift-dps", type=float, default=0.05,
                   help="Artificial gyro bias for the fusion test, deg/s")
    p.add_argument("--fusion-gain", type=float, default=0.2,
                   help="Complementary-filter gain per accepted fix (0..1)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: <dataset>/../yaw_probe_<name>_ep<N>)")
    args = p.parse_args()

    root = args.dataset.resolve()
    roi = tuple(float(v) for v in args.roi.split(","))
    if len(roi) != 4:
        print("--roi needs 4 values"); return 1
    video_key = CAM_KEYS[args.cam]
    fx = args.fx or (385.0 if args.cam == "cam1" else 615.0)
    out = args.out or root.parent / f"yaw_probe_{root.name}_ep{args.episode}"
    out.mkdir(parents=True, exist_ok=True)

    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info["fps"])
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    has_slew = "slew" in state_names

    ep = load_episode_meta(root, args.episode)
    rows = load_episode_rows(root, ep)
    n = len(rows)
    t = rows["clock.loop"].map(lambda v: float(np.asarray(v).ravel()[0])).to_numpy() \
        if "clock.loop" in rows else np.arange(n) / fps
    state = np.stack(rows["observation.state"].to_numpy())
    action = np.stack(rows["action"].to_numpy())

    print(f"[data] {root.name} ep{args.episode}: {n} frames, {t[-1]:.1f} s, "
          f"state={state_names}, cam={args.cam}, fx={fx:.0f}")
    tic = time.perf_counter()
    gray = decode_episode_gray(root, ep, video_key, n, fps)
    print(f"[data] decoded in {time.perf_counter() - tic:.1f} s")
    n = min(n, len(gray))
    t, state, action, gray = t[:n], state[:n], action[:n], gray[:n]

    m = Matcher(gray[0].shape, roi, args.scale)
    prev = cv2.cvtColor(gray[0], cv2.COLOR_GRAY2BGR)
    s = m.sl
    cv2.rectangle(prev, (s[1].start, s[0].start), (s[1].stop - 1, s[0].stop - 1), (0, 255, 0), 2)
    cv2.imwrite(str(out / "roi_preview.png"), prev)
    print(f"[roi] {out / 'roi_preview.png'}  match size {m.size[0]}x{m.size[1]}  "
          "<- check the boom is outside the box")

    # ── 1. incremental ──────────────────────────────────────────────────────
    small = [m.prep(g) for g in gray]
    dx = np.zeros(n); resp = np.zeros(n); cost = []
    for i in range(1, n):
        c0 = time.perf_counter()
        dx[i], resp[i] = m.shift(small[i - 1], small[i])
        cost.append(time.perf_counter() - c0)
    prep_cost = []
    for g in gray[: min(200, n)]:
        c0 = time.perf_counter(); m.prep(g); prep_cost.append(time.perf_counter() - c0)
    vis_d = px_to_deg(dx, fx)
    ok = resp >= args.min_response

    print("\n=== cost (one core, Python) ===")
    print(f"  prep  {1e3 * np.median(prep_cost):.2f} ms median   "
          f"match {1e3 * np.median(cost):.2f} ms median, {1e3 * np.percentile(cost, 99):.2f} ms p99")
    print(f"  match response: median {np.median(resp[1:]):.3f}, "
          f"{100 * (1 - ok[1:].mean()):.1f}% below --min-response {args.min_response}")

    if not has_slew:
        a = action[:, action_names.index("slew")]
        moving = (np.abs(a) > 0.2) & ok
        print("\n=== slew not in observation.state: action-only check ===")
        if moving.sum() < 10:
            print("  too few frames with a slew command to judge"); return 0
        corr = float(np.corrcoef(a[moving], vis_d[moving])[0, 1])
        still = (np.abs(a) < 0.02) & ok
        print(f"  frames commanding slew: {moving.sum()}   corr(slew cmd, image yaw rate) = {corr:+.2f}")
        print(f"  image yaw rate while slewing: {np.median(np.abs(vis_d[moving])) * fps:.1f} deg/s median")
        print(f"  image yaw rate with no slew cmd: {np.median(np.abs(vis_d[still])) * fps:.2f} deg/s median "
              "(boom motion leaking into the ROI shows up here)")
        print("  A strong |corr| means the shift is seen and the sign is stable. For the real\n"
              "  test re-record with --state-joints slew,lift,tilt,scoop.")
        pd.DataFrame({"t": t, "slew_cmd": a, "vis_dyaw_deg": vis_d, "resp": resp}) \
            .to_csv(out / "frames.csv", index=False)
        return 0

    yaw = np.degrees(np.unwrap(np.radians(state[:, state_names.index("slew")])))
    dyaw = np.diff(yaw, prepend=yaw[0])
    rate = np.gradient(yaw, t)
    moving = ok & (np.abs(rate) > 2.0)
    if moving.sum() < 10:
        print("\nNot enough slewing in this episode to fit anything."); return 1
    k = fit_gain(dyaw[moving], vis_d[moving])
    sign = 1.0 if k >= 0 else -1.0
    fx_eff = fx * abs(k)
    vis_d_cal = sign * px_to_deg(dx, fx_eff)
    res = (vis_d_cal - dyaw)[ok & (np.arange(n) > 0)]
    print("\n=== 1. incremental (frame to frame) ===")
    print(f"  sign {'+' if sign > 0 else '-'}, fitted gain {abs(k):.3f} -> effective fx {fx_eff:.0f} px "
          f"(expected ~{fx:.0f}; far off = parallax, boom in ROI, or the camera is not on the cab)")
    print(f"  corr(gyro dyaw, image dyaw) = {np.corrcoef(dyaw[moving], vis_d_cal[moving])[0, 1]:+.3f}")
    print(f"  per-frame residual: {rms(res):.3f} deg RMS")
    print(f"  integrated over the episode: image {np.sum(vis_d_cal[ok]):+.1f} deg vs gyro {yaw[-1] - yaw[0]:+.1f} deg "
          "(dead reckoning; drifts, which is why keyframes exist)")

    # ── 2. keyframes + 3. fusion ────────────────────────────────────────────
    drift = args.inject_drift_dps * (t - t[0])
    gyro_d = yaw + drift                      # what a biased gyro would integrate to
    kfs: list[tuple[float, np.ndarray]] = []  # (yaw label, prepped image)
    fused = np.empty(n); fused[0] = gyro_d[0]
    fix_t = []; fix_err = []; fix_resp = []
    next_fix = t[0]
    for i in range(n):
        if i > 0:
            fused[i] = fused[i - 1] + (gyro_d[i] - gyro_d[i - 1])
        slow = abs(rate[i]) < args.kf_max_rate
        if t[i] - t[0] < args.sweep_s:
            if slow and (not kfs or min(abs(fused[i] - y) for y, _ in kfs) >= args.kf_step_deg):
                kfs.append((fused[i], small[i]))
            continue
        if not kfs or t[i] < next_fix or not slow:
            continue
        next_fix = t[i] + 1.0 / args.fix_hz
        y_kf, img_kf = min(kfs, key=lambda kf: abs(kf[0] - fused[i]))
        d, r = m.shift(img_kf, small[i])
        if r < args.min_response:
            continue
        y_vis = y_kf + sign * float(px_to_deg(d, fx_eff))
        fix_t.append(t[i] - t[0]); fix_err.append(y_vis - yaw[i]); fix_resp.append(r)
        fused[i] += args.fusion_gain * (y_vis - fused[i])

    kf_span = (min(y for y, _ in kfs), max(y for y, _ in kfs)) if kfs else (0, 0)
    fix_err = np.asarray(fix_err)
    after = (t - t[0]) >= args.sweep_s
    print(f"\n=== 2. keyframe absolute fixes (after the {args.sweep_s:.0f} s sweep) ===")
    print(f"  {len(kfs)} keyframes spanning {kf_span[0]:+.0f}..{kf_span[1]:+.0f} deg; "
          f"episode yaw range {yaw.min():+.0f}..{yaw.max():+.0f} deg")
    if len(fix_err):
        print(f"  {len(fix_err)} fixes accepted: error vs gyro {np.median(fix_err):+.2f} deg median, "
              f"{rms(fix_err - np.median(fix_err)):.2f} deg RMS scatter, "
              f"{np.max(np.abs(fix_err)):.2f} deg worst")
        print("  (the median includes the real gyro drift since the sweep; the scatter is the camera)")
    else:
        print("  no fixes accepted — episode too short after the sweep, or matches all rejected")

    print(f"\n=== 3. fusion with {args.inject_drift_dps:+.3f} deg/s injected gyro bias ===")
    if after.any():
        print(f"  end of episode: gyro alone off by {gyro_d[-1] - yaw[-1]:+.2f} deg, "
              f"fused off by {fused[-1] - yaw[-1]:+.2f} deg")
        print(f"  after sweep: gyro alone {rms((gyro_d - yaw)[after]):.2f} deg RMS, "
              f"fused {rms((fused - yaw)[after]):.2f} deg RMS")

    pd.DataFrame({"t": t - t[0], "gyro_yaw": yaw, "gyro_drifted": gyro_d, "fused": fused,
                  "gyro_dyaw": dyaw, "vis_dyaw": vis_d_cal, "resp": resp}) \
        .to_csv(out / "frames.csv", index=False)
    pd.DataFrame({"t": fix_t, "err_deg": fix_err, "resp": fix_resp}) \
        .to_csv(out / "fixes.csv", index=False)
    print(f"\n[out] {out}/frames.csv, fixes.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
