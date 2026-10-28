"""GR00T N1.7 fine-tune on the native stack: physical parity and a live drive loop.

Engines and the host chain come from jetson-orin-nano-vla (`--vla-root`, imported read-only).
This file owns the excavator boundary that Gr00tPolicy applies on the Spark:

    cameras  native IR Y8 -> gray repeated to RGB -> the bundle's eval transform
    state    angles_rad[boom, arm, bucket] -> float32 degrees -> min-max [lo, hi] -> [-1, 1], clipped
    actions  model [-1, 1] -> clipped -> [lo, hi] (the contract's stats) -> [slew, lift, tilt, scoop]

`parity` runs a physical fixture (spark-projects groot/excavator/physical_fixture.py) from raw
frames and degrees to valve commands. `run` reads live cameras and the controller, infers
continuously and, with --drive, sends each chunk as a timed plan (t0 = observation time, so
the rows that went stale during inference are skipped by the controller). A supervisor thread
renews the plan every 100 ms and disarms on chunk exhaustion, on observation age, on any
controller state but armed and on any inference error. It never re-arms and never sends pump
permission.

    python groot_live.py parity --bundle B --contract C.json --fixture F.npz
    python groot_live.py run --bundle B --contract C.json --start-services --drive --seconds 60 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAM1, CAM2 = "233522073919", "141722076647"        # digging_native_640: cam1 = ir2, rotated 180
JOINTS = {"slew": 0, "lift": 1, "tilt": 2, "scoop": 3}  # controller angles_rad / plan columns


# ── boundary ────────────────────────────────────────────────────────────────


class Boundary:
    def __init__(self, contract: dict, state_names=("lift", "tilt", "scoop"),
                 action_names=("slew", "lift", "tilt", "scoop")):
        (sk, s), = contract["state"].items()
        (ak, a), = contract["action"].items()
        self.s_lo, self.s_hi = np.asarray(s["min"]), np.asarray(s["max"])
        self.a_lo, self.a_hi = np.asarray(a["min"]), np.asarray(a["max"])
        self.clip_state = contract["clip_state"]
        self.state_idx = [JOINTS[n] for n in state_names]
        self.action_cols = [JOINTS[n] for n in action_names]
        assert len(self.s_lo) == len(state_names) and len(self.a_lo) == len(action_names)

    def state(self, degrees: np.ndarray, width: int) -> np.ndarray:
        """float32 degrees -> [1,1,width] model state (gr00t.data.utils.normalize_values_minmax)."""
        x = np.asarray(degrees, np.float32)
        n = np.zeros_like(x)
        m = ~np.isclose(self.s_hi, self.s_lo)
        n[m] = 2 * ((x[m] - self.s_lo[m]) / (self.s_hi[m] - self.s_lo[m])) - 1
        if self.clip_state:
            n = np.clip(n, -1.0, 1.0)
        out = np.zeros((1, 1, width), np.float32)
        out[0, 0, :n.size] = n
        return out

    def actions(self, model: np.ndarray) -> np.ndarray:
        """[1,H,W] model actions -> [H,A] valve commands (unnormalize_values_minmax)."""
        x = np.clip(model[0, :, :len(self.a_lo)].astype(np.float64), -1.0, 1.0)
        return ((x + 1.0) / 2.0 * (self.a_hi - self.a_lo) + self.a_lo).astype(np.float32)

    def degrees(self, status: dict) -> np.ndarray:
        return np.rad2deg(np.asarray([status["angles_rad"][i] for i in self.state_idx],
                                     np.float64)).astype(np.float32)


class Policy:
    """Bundle + engines + boundary; one current frame per camera (video_delta_indices [0])."""

    def __init__(self, bundle: Path, contract: dict, cache: Path, vla_root: Path):
        sys.path.insert(0, str(vla_root))
        os.chdir(vla_root)      # the engine builders run `-c "from bench.vendor..."` subprocesses
        from bench.vendor import groot17_trt as g17
        from bench.vendor import groot_trt as g16

        self.g17, self.g16 = g17, g16
        self.b = g17.Bundle(bundle)
        bb = self.b.b
        if bb["frames"] != 1 or bb["video_delta_indices"] != [0]:
            raise ValueError("this runtime handles current-frame-only bundles")
        if bb["embodiment"] != contract["embodiment"]:
            raise ValueError("contract and bundle embodiments differ")
        self.boundary = Boundary(contract)
        t0 = time.time()
        built = g16.prebuild_engines(self.b, cache)
        if self.b.mod:
            g16.load_step_constants(self.b, cache)
        self.engines = g16.Engines(self.b, cache, skip=self.b.mod)
        self.load_s = round(time.time() - t0, 1)
        self.built = built
        self.fixture = g17.validate_fixture(self.b, self.engines)
        if self.fixture["status"] != "PASS":
            raise RuntimeError(f"bundle fixture gate failed: {self.fixture}")
        self.state_width = self.engine_shape("cond_0", "state")[-1]
        self.noise_shape = (1, bb["action_horizon"], bb["action_dim"])

    def engine_shape(self, graph, tensor):
        for n, _, shape, _, _ in self.engines.io[graph]:
            if n == tensor:
                return shape
        raise KeyError(tensor)

    def infer(self, frames_rgb: list[np.ndarray], degrees: np.ndarray, noise: np.ndarray,
              timings: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        t = {} if timings is None else timings
        t0 = time.perf_counter()
        pv = np.stack([self.b.preprocess(f) for f in frames_rgb])
        state = self.boundary.state(degrees, self.state_width)
        t1 = time.perf_counter()
        enc = self.g17.encode_frames(self.b, self.engines.run, pv)
        t2 = time.perf_counter()
        model = self.g17.infer(self.b, self.engines.run, [enc], state, noise, timings=t)
        t3 = time.perf_counter()
        t.update(preprocess=(t1 - t0) * 1e3, vision=(t2 - t1) * 1e3, total=(t3 - t0) * 1e3)
        return self.boundary.actions(model), model


def gray_to_rgb(y8: np.ndarray) -> np.ndarray:
    return np.repeat(y8[..., None], 3, axis=2)


# ── parity ──────────────────────────────────────────────────────────────────


def parity(a) -> int:
    contract = json.loads(Path(a.contract).read_text())
    pol = Policy(Path(a.bundle), contract, Path(a.cache).expanduser(), Path(a.vla_root))
    fx = np.load(a.fixture)
    meta = json.loads(str(fx["meta"]))
    if meta["contract"] != contract:
        raise SystemExit("fixture contract differs from --contract")
    bad = set(meta["tasks"]) - {pol.b.b["task"]}
    if bad:
        raise SystemExit(f"fixture tasks {bad} differ from the baked prompt {pol.b.b['task']!r}")
    env = np.load(a.envelope) if a.envelope else None
    if env is not None and not np.array_equal(env["noise"], fx["noise"]):
        raise SystemExit("--envelope fixture uses different inputs")
    rows, worst, ok = [], 0.0, True
    for i in range(len(fx["raw"])):
        raw = fx["raw"][i]
        # The dataset decodes gray IR as identical RGB channels; the live path does too.
        gray_ok = all(np.array_equal(raw[v][..., 0], raw[v][..., c]) for v in range(len(raw)) for c in (1, 2))
        frames = [gray_to_rgb(raw[v][..., 0]) if gray_ok else raw[v] for v in range(len(raw))]
        t = {}
        act, _ = pol.infer(frames, fx["state_deg"][i], fx["noise"][i][None], t)
        ref = fx["actions"][i]
        d = np.abs(act - ref)
        cos = float((act.ravel() @ ref.ravel()) / (np.linalg.norm(act) * np.linalg.norm(ref)))
        worst = max(worst, float(d.max()))
        if env is not None:     # no further from FP32 than Gr00tPolicy's own bfloat16 inference
            e = np.abs(env["actions"][i] - ref)
            lim_max, lim_mean = float(e.max()), float(e.mean())
        else:
            lim_max, lim_mean = a.max_abs, np.inf
        passed = bool(d.max() <= lim_max and d.mean() <= lim_mean and cos >= a.min_cosine)
        ok &= passed
        rows.append({"pass": passed, "limit_max_abs": round(lim_max, 4),
                     "limit_mean_abs": round(lim_mean, 5) if np.isfinite(lim_mean) else None,"episode": int(fx["episode"][i]), "frame": int(fx["frame"][i]), "gray_rgb": gray_ok,
                     "cosine": round(cos, 6), "max_abs": round(float(d.max()), 4),
                     "mean_abs": round(float(d.mean()), 5),
                     "mean_abs_vs_recorded": round(float(np.abs(act - fx["gt"][i]).mean()), 4),
                     "ms": round(t["total"], 1)})
        print(json.dumps(rows[-1]), flush=True)
    summary = {"status": "PASS" if ok else "FAIL", "cosine_gate": a.min_cosine,
               "gate": "bfloat16 Gr00tPolicy envelope" if env is not None else f"max_abs {a.max_abs}",
               "worst_max_abs": worst, "bundle_fixture": pol.fixture, "engines_built_s": pol.built,
               "load_s": pol.load_s, "samples": rows}
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "worst_max_abs", "load_s")}))
    return 0 if ok else 1


# ── live ────────────────────────────────────────────────────────────────────


class Supervisor(threading.Thread):
    """Owns the drive connection: arms, renews the newest plan, stops on any doubt."""

    def __init__(self, client_mod, socket_path, fps, max_obs_age_s=1.4, renew_s=0.1, slew=True):
        super().__init__(daemon=True)
        self.c = client_mod
        self.control = client_mod.Client(socket_path)
        self.fps, self.max_age, self.renew = fps, max_obs_age_s, renew_s
        self.slew = slew
        self.lock = threading.Lock()
        self.plan = None          # (rows [N,6], t0_ns, obs_ns)
        self.stop_reason = None
        self.sent = self.renewals = 0
        self.applied = []         # (row position, planned [4] interpolated, controller applied_action [4])
        self.by_sequence = {}     # drive packet sequence -> (t0 ns, rows)
        self.halt = threading.Event()

    def arm(self):
        st = self.control.status()
        if st["state"] != "disarmed":
            raise RuntimeError(f"controller is {st['state']} ({st.get('reason')}), not disarmed")
        self.control.arm(pump=False)

    def offer(self, rows: np.ndarray, t0_ns: int, obs_ns: int):
        with self.lock:
            if self.plan is not None and obs_ns <= self.plan[2]:
                return False
            self.plan = (rows, t0_ns, obs_ns)
            self.fresh = True
            return True

    def stop(self, reason):
        if self.stop_reason is None:
            self.stop_reason = reason
        self.halt.set()

    def run(self):
        try:
            while not self.halt.is_set():
                now = time.monotonic_ns()
                with self.lock:
                    plan = self.plan
                if plan is None:          # centered hold until the first chunk
                    self.control.send(self.c.DRIVE, [(0.0,) * 6], flags=self.c.DEADMAN)
                else:
                    rows, t0, obs = plan
                    if (now - obs) / 1e9 > self.max_age:
                        self.stop(f"observation age {(now - obs) / 1e9:.2f} s")
                        break
                    if now >= t0 + int((len(rows) - 1) / self.fps * 1e9):
                        self.stop("chunk exhausted")
                        break
                    self.control.send(self.c.DRIVE, [tuple(map(float, r)) for r in rows],
                                      flags=self.c.DEADMAN, fps=self.fps, t0_ns=t0, observation_ns=obs)
                    self.by_sequence[self.control.sequence] = (t0, rows)
                    self.sent += 1
                st = self.control.status()
                applied_ns = st["applied_host_monotonic_ns"]
                held = self.by_sequence.get(st["applied_command_sequence"])
                if held:   # the controller's command at that tick: the packet it held, interpolated
                    t0, rows = held
                    pos = min(max((applied_ns - t0) * self.fps / 1e9, 0.0), len(rows) - 1)
                    i = int(pos)
                    j = min(i + 1, len(rows) - 1)
                    due = rows[i][:4] + (rows[j][:4] - rows[i][:4]) * (pos - i)
                    self.applied.append((round(pos, 2), due.tolist(), st["applied_action"][:4]))
                if st["state"] != "armed":
                    self.stop(f"controller {st['state']}: {st.get('reason')}")
                    break
                self.halt.wait(self.renew)
        except Exception as e:  # noqa: BLE001
            self.stop(f"supervisor error: {e!r}")
        finally:
            try:
                self.control.send(self.c.DISARM)
            finally:
                self.control.close()


def start_services(a, runtime: Path):
    build = Path(a.build)
    control = [build / "kaivuri-control", "--profile", a.profile, "--safety", a.safety,
               "--socket", runtime / "control.sock", "--control-hz", str(a.control_hz),
               "--serial", a.serial]
    if a.live_pwm:
        control.append("--live-pwm")
    if a.enable_slew:
        control.append("--enable-slew")
    cams = [build / "kaivuri-camera", "--camera", f"{CAM1}:2", "--camera", f"{CAM2}:1",
            "--output-dir", runtime / "cameras", "--fps", "30", "--width", "640", "--height", "480",
            "--rotate-180", CAM1]
    procs = []
    log = open(runtime / "services.log", "w")
    for cmd, ready in ((control, [runtime / "control.sock"]),
                       (cams, [runtime / "cameras" / f"{CAM1}-ir2.frame", runtime / "cameras" / f"{CAM2}-ir1.frame"])):
        p = subprocess.Popen([str(x) for x in cmd], start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        end = time.time() + 30
        while not all(r.exists() for r in ready):
            if p.poll() is not None or time.time() > end:
                raise RuntimeError(f"{cmd[0].name} did not start (see {runtime / 'services.log'})")
            time.sleep(0.1)
    time.sleep(1.5)       # USB streams start one after the other
    return procs, runtime / "control.sock", [runtime / "cameras" / f"{CAM1}-ir2.frame",
                                             runtime / "cameras" / f"{CAM2}-ir1.frame"]


def run_live(a) -> int:
    sys.path.insert(0, str(Path(a.tools)))
    import client as client_mod
    from frame_reader import FrameReader

    contract = json.loads(Path(a.contract).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    procs, tmp = [], None
    sup = None
    try:
        if a.start_services:
            tmp = tempfile.TemporaryDirectory(prefix="kvr-groot-")
            procs, sock, cam_files = start_services(a, Path(tmp.name))
        else:
            sock, cam_files = a.control_socket, a.camera_frame
        pol = Policy(Path(a.bundle), contract, Path(a.cache).expanduser(), Path(a.vla_root))
        print(f"[groot] engines ready in {pol.load_s} s, fixture {pol.fixture['status']} "
              f"(chunk cos {pol.fixture['reports']['chunk']['cosine']:.6f})", flush=True)
        readers = [FrameReader(p, Path(a.build) / "libkaivuri_frames.so", max_age_s=a.max_frame_age_ms / 1e3)
                   for p in cam_files]
        status_conn = client_mod.Client(sock)
        rng = np.random.default_rng(a.seed)
        fps = float(a.fps)
        log = open(out / "chunks.jsonl", "w")
        chunks = []

        def observe():
            frames, arrivals = [], []
            for r in readers:
                snap = None
                end = time.monotonic() + 1.0
                while snap is None:
                    snap = r.read()
                    if snap is None:
                        if time.monotonic() > end:
                            raise RuntimeError("no fresh camera frame")
                        time.sleep(0.002)
                m, px = snap
                frames.append(np.frombuffer(px, np.uint8).reshape(m.height, m.width).copy())
                arrivals.append(m.host_monotonic_ns)
            st = status_conn.status()
            now = time.monotonic_ns()
            if not (st["imu_healthy"] and st["fk_valid"]):
                raise RuntimeError("IMU unhealthy / FK invalid")
            if (now - st["imu_host_monotonic_ns"]) / 1e6 > a.max_imu_age_ms:
                raise RuntimeError("IMU sample too old")
            skew = (max(arrivals) - min(arrivals)) / 1e6
            return frames, st, min([*arrivals, st["imu_host_monotonic_ns"]]), skew

        # Warm-up: a few inferences before arming, so the first plan's latency is known.
        for _ in range(a.warmup):
            frames, st, obs_ns, _ = observe()
            pol.infer([gray_to_rgb(f) for f in frames], pol.boundary.degrees(st),
                      rng.standard_normal(pol.noise_shape).astype(np.float32))
        if a.drive:
            sup = Supervisor(client_mod, sock, fps, a.max_observation_age_ms / 1e3, slew=a.enable_slew)
            sup.arm()
            sup.start()
            print("[groot] armed (no pump permission)", flush=True)
        t_end = time.monotonic() + a.seconds
        next_start = time.monotonic()
        while time.monotonic() < t_end and not (sup and sup.halt.is_set()):
            if a.rate_hz > 0:
                time.sleep(max(0.0, next_start - time.monotonic()))
                next_start += 1.0 / a.rate_hz
            frames, st, obs_ns, skew = observe()
            if not chunks:        # what the model saw first, for an orientation check
                for k, f in enumerate(frames, 1):
                    (out / f"cam{k}_first.pgm").write_bytes(
                        f"P5\n{f.shape[1]} {f.shape[0]}\n255\n".encode() + f.tobytes())
            deg = pol.boundary.degrees(st)
            t = {}
            noise = rng.standard_normal(pol.noise_shape).astype(np.float32)
            act, model = pol.infer([gray_to_rgb(f) for f in frames], deg, noise, t)
            if not np.isfinite(act).all():
                raise RuntimeError("non-finite actions")
            done_ns = time.monotonic_ns()
            rows = np.zeros((len(act), 6), np.float32)
            rows[:, pol.boundary.action_cols] = np.clip(act, -1.0, 1.0)
            if not a.enable_slew:
                rows[:, 0] = 0.0
            accepted = sup.offer(rows, obs_ns + int(a.action_offset_ms * 1e6), obs_ns) if sup else None
            lag_rows = (done_ns - obs_ns) / 1e9 * fps
            rec = {"i": len(chunks), "obs_ns": obs_ns, "done_ns": done_ns,
                   "obs_to_plan_ms": round((done_ns - obs_ns) / 1e6, 1), "stale_rows": round(lag_rows, 1),
                   "camera_skew_ms": round(skew, 1), "state_deg": deg.round(2).tolist(),
                   "row_now": act[min(int(lag_rows), len(act) - 1)].round(3).tolist(),
                   "row0": act[0].round(3).tolist(), "abs_max": np.abs(act).max(0).round(3).tolist(),
                   "timings_ms": {k: round(v, 1) for k, v in t.items()}, "accepted": accepted}
            chunks.append(act)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{rec['i']:4d}] obs->plan {rec['obs_to_plan_ms']:6.1f} ms  infer {t['total']:6.1f}  "
                  f"deg {deg.round(1)}  now {np.round(rec['row_now'], 2)}", flush=True)
        np.save(out / "chunks.npy", np.stack(chunks) if chunks else np.zeros((0,)))
        summary = {"chunks": len(chunks), "drive": a.drive,
                   "stop_reason": sup.stop_reason if sup else None,
                   "plans_sent": sup.sent if sup else 0}
        if sup and sup.applied:
            planned = np.array([p for _, p, _ in sup.applied])
            applied = np.array([q for _, _, q in sup.applied])
            summary["applied_vs_planned"] = {
                "samples": len(applied), "mean_abs": float(np.abs(applied - planned).mean()),
                "max_abs": float(np.abs(applied - planned).max()),
                "row_due": {"min": min(k for k, _, _ in sup.applied), "max": max(k for k, _, _ in sup.applied)}}
            json.dump(sup.applied, open(out / "applied.json", "w"))
        lat = [json.loads(l)["obs_to_plan_ms"] for l in open(out / "chunks.jsonl")]
        inf = [json.loads(l)["timings_ms"]["total"] for l in open(out / "chunks.jsonl")]
        if lat:
            summary.update(obs_to_plan_ms={"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99)),
                                           "max": max(lat)},
                           infer_ms={"p50": float(np.percentile(inf, 50)), "p99": float(np.percentile(inf, 99))})
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary), flush=True)
        return 0
    finally:
        if sup:
            sup.stop(sup.stop_reason or "run finished")
            sup.join(timeout=2)
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        if tmp:
            tmp.cleanup()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("parity", "run"):
        s = sub.add_parser(name)
        s.add_argument("--bundle", required=True)
        s.add_argument("--contract", required=True)
        s.add_argument("--cache", default="~/.cache/kaivuri-groot/default")
        s.add_argument("--vla-root", default=str(Path.home() / "jetson-orin-nano-vla"))
        s.add_argument("--out")
    s = sub.choices["parity"]
    s.add_argument("--fixture", required=True)
    s.add_argument("--envelope", help="the same fixture from bfloat16 Gr00tPolicy: per-sample error limits")
    s.add_argument("--max-abs", type=float, default=0.03, help="valve-command units, without --envelope")
    s.add_argument("--min-cosine", type=float, default=0.999)
    s = sub.choices["run"]
    s.add_argument("--tools", default=str(Path.home() / "kaivuriprokkis/native_stack/tools"))
    s.add_argument("--build", default=str(Path.home() / "kaivuriprokkis/native_stack/build"))
    s.add_argument("--profile", default=str(Path.home() / "kaivuriprokkis/configuration_files/profiles/jetson"))
    s.add_argument("--safety", default=str(Path.home() / "kaivuriprokkis/native_stack/config/safety.yaml"))
    s.add_argument("--serial", default="/dev/ttyACM0")
    s.add_argument("--control-hz", type=int, default=200)
    s.add_argument("--start-services", action="store_true")
    s.add_argument("--live-pwm", action="store_true")
    s.add_argument("--enable-slew", action="store_true")
    s.add_argument("--control-socket")
    s.add_argument("--camera-frame", action="append", help="cam1 then cam2 frame files")
    s.add_argument("--drive", action="store_true")
    s.add_argument("--seconds", type=float, default=30)
    s.add_argument("--rate-hz", type=float, default=0, help="0 = back to back")
    s.add_argument("--fps", type=float, default=30)
    s.add_argument("--warmup", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--action-offset-ms", type=float, default=0)
    s.add_argument("--max-frame-age-ms", type=float, default=200)
    s.add_argument("--max-imu-age-ms", type=float, default=50)
    s.add_argument("--max-observation-age-ms", type=float, default=1400)
    a = p.parse_args()
    if a.cmd == "run" and not a.out:
        p.error("run needs --out")
    # Policy() changes into --vla-root (its engine builders import from there).
    for k in ("bundle", "contract", "cache", "vla_root", "out", "fixture", "envelope", "tools", "build",
              "profile", "safety", "control_socket"):
        if getattr(a, k, None):
            setattr(a, k, str(Path(getattr(a, k)).expanduser().resolve()))
    if getattr(a, "camera_frame", None):
        a.camera_frame = [str(Path(x).resolve()) for x in a.camera_frame]
    return parity(a) if a.cmd == "parity" else run_live(a)


if __name__ == "__main__":
    sys.exit(main())
