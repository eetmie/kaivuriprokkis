"""One live runner for the excavator's VLA fine-tunes: GR00T N1.7 and pi0.5 (OpenPI).

The bundle picks the model (bundle.json `policy: pi05_openpi_state` -> vla_pi05.py, a GR00T
export -> vla_groot.py); everything else is shared. `parity` checks the model against its
fixtures. `run` reads live cameras and the controller, infers continuously and, with --live,
sends each chunk as a timed plan (t0 = observation time, so the rows that went stale during
inference are skipped by the controller). A supervisor thread renews the plan every 100 ms and
disarms on chunk exhaustion, on observation age, on any controller state but armed and on any
inference error. It never re-arms. With an Xbox pad (--gamepad auto): A toggles manual (sticks
drive, inference paused) and model; a stale or exhausted chunk then falls back to manual instead
of disarming. X toggles pump permission. --live drives with pump permission on and, with
--start-services, starts the controller with live PWM. Ctrl+C disarms, stops the services and
still writes the summary (with --out; without it nothing is written and the outcome is printed).

Prompts: the run starts on --task or the model's first task; D-pad right/left cycles the list,
and a line typed in the terminal becomes a prompt (GR00T bundles that tokenize at runtime). A
change drops the current plan; the machine holds until the first chunk observed under the new
prompt, at most --first-plan-timeout-s.

RTC (GR00T bundles with an `rtc` section): each inference commits the first d rows of its chunk
to what the running plan executes at those times (read back from the plan the controller holds,
in normalized model units), and the model denoises only the rest. d covers the observation ->
new-plan-applied latency (recent maximum + --rtc-margin-ms), capped at the trained maximum.
`--rtc off` runs the stock sampler on the same engines.

Exit status: 0 run finished, 130 Ctrl+C, 3 gamepad lost, 4 controller fault, 5 plan stopped
(observation age, chunk exhausted, no first plan), 1 any other error.

    python vla_live.py parity --bundle B [--fixtures PREPARED]      # pi0.5 needs its prepared fixtures
    python vla_live.py run --bundle B [--fixtures PREPARED] --start-services --live --enable-slew [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from vla_observe import HistoryFilling, Incoherent, Observer, frame_ns, start_services, stop_services
from vla_operator import Pad, Supervisor, TaskList, find_pad, interp_rows

HERE = Path(__file__).resolve().parent        # defaults below are this checkout's, never another's
NATIVE = HERE.parent


def bundle_kind(bundle: Path) -> str:
    b = json.loads((Path(bundle) / "bundle.json").read_text())
    if b.get("policy") == "pi05_openpi_state":
        return "pi05"
    if "embodiment" in b and "video_delta_indices" in b:
        return "groot"
    raise ValueError(f"{bundle}: neither a pi05_openpi_state nor a GR00T N1.7 bundle")


def load_model(a):
    """The bundle's model, loaded and gated against its fixtures."""
    if a.kind == "pi05":
        import vla_pi05
        return vla_pi05.Model(Path(a.bundle), Path(a.cache), Path(a.fixtures) if a.fixtures else None,
                              Path(a.build), a.min_cosine, a.max_abs)
    import vla_groot
    contract = json.loads(Path(a.contract).read_text()) if a.contract else None
    return vla_groot.Policy(Path(a.bundle), contract, Path(a.cache), Path(a.vla_root),
                            history_tolerance_s=a.history_tolerance_ms / 1e3)


# ── RTC ─────────────────────────────────────────────────────────────────────


class RtcDelay:
    """Committed rows d: from the observation (row 0) until the controller applies the new plan.
    The recent maximum observation -> result latency plus a dispatch margin, capped at the trained
    maximum; the prefix must also lie inside the running plan."""

    def __init__(self, max_delay: int, fps: float, margin_ms: float, window: int = 20):
        self.max, self.fps, self.margin = max_delay, fps, margin_ms
        self.lat = collections.deque(maxlen=window)

    def add(self, ms: float):
        self.lat.append(ms)

    def want(self) -> int:
        if not self.lat or self.max == 0:
            return self.max
        return min(self.max, math.ceil((max(self.lat) + self.margin) * self.fps / 1e3))


def committed_rows(pol, running, t0_ns: int, want: int, fps: float):
    """(d, prefix [d,A] model units, plan id): what `running` (rows, t0, id) executes at rows
    0..d-1 of a chunk anchored at t0_ns, as the controller interpolates it."""
    if running is None or want == 0:
        return 0, None, None
    rows, t0_old, pid = running
    pos = (t0_ns - t0_old) * fps / 1e9 + np.arange(want)
    d = int(np.sum((pos >= 0) & (pos <= len(rows) - 1)))
    if d == 0 or pos[0] < 0:
        return 0, None, pid
    valves = interp_rows(rows[:, pol.action_cols], pos[:d])
    return d, pol.model_actions(valves), pid


# ── live ────────────────────────────────────────────────────────────────────

EXIT = {"finished": 0, "interrupted": 130, "gamepad": 3, "controller": 4, "plan": 5, "error": 1}


def sha256(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def stats(xs) -> dict | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "p50": round(float(np.percentile(xs, 50)), 3),
            "p99": round(float(np.percentile(xs, 99)), 3), "max": round(float(max(xs)), 3)}


def write_summary(out: Path, a, chunks, sup, outcome: dict, tasks=None, extra=None) -> dict:
    np.save(out / "chunks.npy", np.stack(chunks) if chunks else np.zeros((0,)))
    summary = {"outcome": outcome, "model": a.kind, "chunks": len(chunks), "drive": a.live,
               "plans_sent": sup.sent if sup else 0, "pad_events": list(sup.events) if sup else [],
               "tasks": tasks.items if tasks else None, **(extra or {})}
    if sup and sup.applied:
        planned = np.array([p for _, p, _ in sup.applied])
        applied = np.array([q for _, _, q in sup.applied])
        summary["applied_vs_planned"] = {
            "samples": len(applied), "mean_abs": float(np.abs(applied - planned).mean()),
            "max_abs": float(np.abs(applied - planned).max()),
            "row_due": {"min": min(k for k, _, _ in sup.applied), "max": max(k for k, _, _ in sup.applied)}}
        json.dump(list(sup.applied), open(out / "applied.json", "w"))
    if sup:
        json.dump(list(sup.dispatch), open(out / "dispatch.json", "w"))
        jumps = [max(r["jump"]) for r in sup.dispatch if "jump" in r]
        summary["handover"] = {"max_jump": stats(jumps), "row": stats([r["row"] for r in sup.dispatch]),
                               "over_0.5": sum(j > 0.5 for j in jumps)}
        summary["supervisor_tick_ms"] = stats(list(sup.ticks))
    recs = [json.loads(l) for l in open(out / "chunks.jsonl")] if (out / "chunks.jsonl").exists() else []
    inferred = [r for r in recs if "obs_to_plan_ms" in r]
    if recs:
        summary["rejected_observations"] = collections.Counter(r["rejected"] for r in recs if "rejected" in r)
        summary["offers"] = collections.Counter(r.get("reason") or ("accepted" if r.get("accepted") else "not offered")
                                                for r in inferred)
    if inferred:
        summary["chunks_per_task"] = {t: sum(r.get("task") == t for r in inferred)
                                      for t in dict.fromkeys(r.get("task") for r in inferred)}
        summary.update(obs_to_plan_ms=stats([r["obs_to_plan_ms"] for r in inferred]),
                       infer_ms=stats([r["timings_ms"]["total"] for r in inferred]),
                       camera_skew_ms=stats([r["camera_skew_ms"] for r in inferred]),
                       rtc_delay=stats([r["rtc_delay"] for r in inferred]),
                       rtc_short=sum(r["rtc_delay"] < r["rtc_want"] for r in inferred),
                       history_age_ms=stats([r["timings_ms"].get("history_age") for r in inferred]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def run_live(a) -> int:
    # Started in the background by a non-interactive shell, SIGINT arrives ignored; restore it
    # so Ctrl+C / kill -INT always takes the graceful path below.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    import client as client_mod
    from frame_reader import FrameReader

    out = Path(a.out) if a.out else None       # None: nothing is written
    if out:
        out.mkdir(parents=True, exist_ok=False)
    identity = {"argv": sys.argv, "model": a.kind,
                "runner_sha256": {f.name: sha256(f) for f in sorted(HERE.glob("vla_*.py"))}, "bundle": a.bundle,
                "bundle_json_sha256": sha256(Path(a.bundle) / "bundle.json"), "cache": a.cache,
                "vla_root": a.vla_root, "profile": a.profile, "safety": a.safety,
                "safety_sha256": sha256(a.safety), "control_hz": a.control_hz, "fps": a.fps,
                "start_monotonic_ns": time.monotonic_ns(), "start_unix": time.time()}
    if out:
        (out / "run.json").write_text(json.dumps(identity, indent=1) + "\n")
    procs, tmp, sup, pad, tasks, status_conn, readers, pol = [], None, None, None, None, None, [], None
    chunks = collections.deque(maxlen=20_000)
    outcome = {"kind": "finished", "reason": "run finished"}
    rtc_on = False
    try:
        pad_path = find_pad(a.gamepad) if a.live else None
        if pad_path:
            pad = Pad(pad_path)
            print(f"[pad] {pad_path}: A = model/manual, X = pump on/off, Ctrl+C = stop", flush=True)
        elif a.live:
            print("[pad] no gamepad: model drives until the run ends; Ctrl+C = stop", flush=True)
        if a.start_services:
            tmp = tempfile.TemporaryDirectory(prefix="kvr-vla-")
            sock, cam_files = start_services(a, Path(tmp.name), procs)
        else:
            sock, cam_files = a.control_socket, a.camera_frame
        pol = load_model(a)
        rtc_on = bool(pol.rtc) and a.rtc != "off"
        if a.rtc == "on" and not pol.rtc:
            raise ValueError("--rtc on: this bundle has no RTC inputs")
        print(pol.describe(rtc_on, a), flush=True)
        if a.task:
            pol.check_prompt(a.task)                    # refuse an unusable --task before arming
        tasks = TaskList(pol.tasks, a.task, same=pol.same_prompt)
        print(f"[task] {pol.prompt_note()}; D-pad right/left cycles\n{tasks.show()}", flush=True)
        if hasattr(pol, "check_cameras"):
            pol.check_cameras(cam_files)
        for p in cam_files:
            readers.append(FrameReader(p, Path(a.build) / "libkaivuri_frames.so",
                                       max_age_s=a.max_frame_age_ms / 1e3))
        status_conn = client_mod.Client(sock)
        observe = Observer(readers, cam_files, status_conn, a)
        rng = np.random.default_rng(a.seed)
        fps = float(a.fps or pol.fps)
        delay = RtcDelay(pol.rtc_max if rtc_on else 0, fps, a.rtc_margin_ms)
        log = open(out / "chunks.jsonl" if out else os.devnull, "w")

        def observe_retry(deadline_s=2.0):
            end = time.monotonic() + deadline_s
            while True:
                try:
                    return observe()
                except Incoherent:
                    if time.monotonic() > end:
                        raise
                    time.sleep(0.005)

        # Warm-up: one inference before arming, so the first plan's latency (RTC delay) is known; a
        # frame-history bundle first fills its history (no plan is made until then).
        fill_end = time.monotonic() + pol.lag_s + 10
        while True:
            frames, st, obs_ns, _, src = observe_retry()
            try:
                pol.step(frames, st, rng.standard_normal(pol.noise_shape).astype(np.float32), {},
                         capture_ns=frame_ns(src))
            except HistoryFilling:
                if time.monotonic() > fill_end:
                    raise RuntimeError("the frame history did not fill during the warm-up")
                continue
            delay.add((time.monotonic_ns() - obs_ns) / 1e6)
            break
        if a.live:
            sup = Supervisor(client_mod, sock, fps, a.max_observation_age_ms / 1e3, slew=a.enable_slew,
                             pump=True, pad=pad, start_model=a.start_model, tasks=tasks,
                             first_plan_s=a.first_plan_timeout_s)
            if pad:
                pad.drain()
            sup.arm()
            sup.start()
            print(f"[vla] armed, {sup.mode.upper()} mode, pump ON", flush=True)

        def read_prompts():
            """Terminal lines become prompts: appended to the task list and selected."""
            for line in sys.stdin:
                text = line.strip()
                if not text:
                    continue
                if text == "?":
                    print(tasks.show(), flush=True)
                    continue
                try:
                    n = pol.check_prompt(text)
                except ValueError as e:
                    print(f"[task] not used: {e}", flush=True)
                    continue
                if sup:
                    sup.drop_plan()
                k = tasks.add(text)
                print(f"[task] {k + 1}: {text} ({n} text tokens)", flush=True)

        if sys.stdin and not sys.stdin.closed:
            threading.Thread(target=read_prompts, daemon=True).start()
        applied, virtual, first_written = None, None, False
        t_end = time.monotonic() + a.seconds if a.seconds > 0 else float("inf")
        next_start = time.monotonic()
        n = 0
        while time.monotonic() < t_end and not (sup and sup.halt.is_set()):
            if sup and sup.mode != "model":       # manual: no inference, sticks drive
                if pol.history:                   # keep the history warm for the switch back
                    try:
                        frames, _, _, _, src = observe()
                        pol.keep_warm(frames, frame_ns(src))
                    except Incoherent:
                        pass
                time.sleep(0.02)
                next_start = time.monotonic()
                continue
            if a.rate_hz > 0:
                time.sleep(max(0.0, next_start - time.monotonic()))
                next_start += 1.0 / a.rate_hz
            k, text, version = tasks.current()
            if version != applied:
                pol.set_prompt(text)
                applied, virtual = version, None
                print(f"[task {k + 1}/{len(tasks.items)}] {text!r} ({pol.task_tokens} prompt tokens)", flush=True)
            try:
                frames, st, obs_ns, skew, sources = observe()
            except Incoherent as e:
                log.write(json.dumps({"rejected": e.kind, "detail": str(e), "ns": time.monotonic_ns()}) + "\n")
                time.sleep(0.005)
                continue
            if out and not first_written:        # what the model saw first, for an orientation check
                for c, f in enumerate(frames, 1):
                    (out / f"cam{c}_first.pgm").write_bytes(
                        f"P5\n{f.shape[1]} {f.shape[0]}\n255\n".encode() + f.tobytes())
                first_written = True
            t = {}
            noise = rng.standard_normal(pol.noise_shape).astype(np.float32)
            t0_ns = obs_ns + int(a.action_offset_ms * 1e6)
            want = delay.want() if rtc_on else 0
            running = sup.running_plan() if sup else virtual
            d, prefix, prefix_plan = committed_rows(pol, running, t0_ns, want, fps)
            try:
                act, deg = pol.step(frames, st, noise, t, rtc_delay=d, rtc_prefix=prefix,
                                    capture_ns=frame_ns(sources))
            except HistoryFilling as e:           # the encode is stored; no plan from this frame
                log.write(json.dumps({"rejected": "history filling", "detail": str(e),
                                      "ns": time.monotonic_ns()}) + "\n")
                continue
            done_ns = time.monotonic_ns()
            delay.add((done_ns - obs_ns) / 1e6)
            rows = np.zeros((len(act), 6), np.float32)
            rows[:, pol.action_cols] = act
            if not a.enable_slew:
                rows[:, 0] = 0.0
            # A chunk of an instruction that changed meanwhile is never offered.
            if tasks.current()[2] != applied:
                accepted, reason, pid = False, "prompt changed during inference", None
            elif sup:
                accepted, reason, pid = sup.offer(rows, t0_ns, obs_ns)
            else:                         # no drive: the chunk becomes the virtual running plan
                accepted, reason, pid = None, None, n + 1
                virtual = (rows, t0_ns, pid)
            lag_rows = (done_ns - obs_ns) / 1e9 * fps
            rec = {"i": n, "obs_ns": obs_ns, "t0_ns": t0_ns, "done_ns": done_ns,
                   "obs_to_plan_ms": round((done_ns - obs_ns) / 1e6, 1), "stale_rows": round(lag_rows, 1),
                   "camera_skew_ms": round(skew, 1), "sources": sources, "state_deg": deg.round(2).tolist(),
                   "rtc_delay": d, "rtc_want": want, "prefix_plan": prefix_plan,
                   "row_now": act[min(int(lag_rows), len(act) - 1)].round(3).tolist(),
                   "row0": act[0].round(3).tolist(), "abs_max": np.abs(act).max(0).round(3).tolist(),
                   "timings_ms": {kk: round(v, 1) for kk, v in t.items()}, "accepted": accepted,
                   "reason": reason, "plan_id": pid, "pump": sup.pump if sup else None, "task": pol.task}
            chunks.append(act)
            n += 1
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{rec['i']:4d}] obs->plan {rec['obs_to_plan_ms']:6.1f} ms  infer {t['total']:6.1f}  "
                  + (f"past {t['history_age']:5.0f} ms  " if "history_age" in t else "")
                  + f"rtc {d:2d}/{want:2d}  deg {deg.round(1)}  now {np.round(rec['row_now'], 2)}"
                  + (f"  clamped {t['clamped']}" if "clamped" in t else ""), flush=True)
        if sup and sup.stop_kind:
            outcome = {"kind": sup.stop_kind, "reason": sup.stop_reason}
    except KeyboardInterrupt:
        outcome = {"kind": "interrupted", "reason": "Ctrl+C"}
        print("\n[vla] Ctrl+C: disarming and stopping services", flush=True)
    except Exception as e:  # noqa: BLE001
        outcome = {"kind": "error", "reason": repr(e)}
        import traceback
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # let the cleanup finish
        cleanup = []

        def step(name, fn):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                cleanup.append(f"{name}: {e!r}")

        if sup:
            step("supervisor stop", lambda: sup.stop(outcome["reason"], outcome["kind"]))
            if sup.ident is not None:
                step("supervisor join", lambda: sup.join(timeout=2))
            if sup.is_alive():
                cleanup.append("supervisor did not finish in 2 s")
            step("disarm", sup.shutdown)        # no-op after run(); disarms when arming failed
            if sup.stop_kind and outcome["kind"] == "finished":
                outcome = {"kind": sup.stop_kind, "reason": sup.stop_reason}
        final = {}
        if status_conn:
            def final_status():
                """The controller's own state after the disarm; armed past 1 s is a cleanup error."""
                end = time.monotonic() + 1.0
                while True:
                    st = status_conn.status()
                    if st["state"] != "armed" or not sup or time.monotonic() > end:
                        break
                    time.sleep(0.02)
                final.update({k: st.get(k) for k in ("state", "reason", "pump_on")})
                if sup and st["state"] == "armed":
                    raise RuntimeError("controller still armed 1 s after the disarm")
            step("final status", final_status)
            step("status close", status_conn.close)
        step("services", lambda: stop_services(procs))
        for r in readers:
            step("frame reader", r.close)
        if tmp:
            step("runtime dir", tmp.cleanup)
        if pad:
            step("gamepad", pad.close)
        if pol is not None and hasattr(pol, "close"):
            step("model", pol.close)
        outcome["final_controller"] = final or None
        outcome["cleanup_errors"] = cleanup
        outcome["exit"] = EXIT.get(outcome["kind"], 1)
        if not out:
            print(json.dumps({"outcome": outcome, "chunks": len(chunks),
                              "plans_sent": sup.sent if sup else 0}), flush=True)
        else:
            try:
                summary = write_summary(out, a, chunks, sup, outcome, tasks,
                                        {"rtc": {"on": rtc_on, "max_delay": pol.rtc_max if rtc_on else 0,
                                                 "margin_ms": a.rtc_margin_ms} if pol else None})
                print(json.dumps({k: v for k, v in summary.items() if k not in ("pad_events", "tasks")}),
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[vla] summary not written: {e!r}", flush=True)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    return outcome["exit"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("parity", "run"):
        s = sub.add_parser(name)
        s.add_argument("--bundle", required=True, help="a GR00T N1.7 or pi05_openpi_state bundle")
        s.add_argument("--cache", help="engine cache (default: GR00T ~/.cache/kaivuri-groot/<bundle name>, "
                                       "pi0.5 <bundle>.cache)")
        s.add_argument("--out", help="report path (default: nothing is written)")
        s.add_argument("--build", default=str(NATIVE / "build"), help="native_stack build of this checkout")
        s.add_argument("--fixtures", help="pi0.5: prepare_pi05.py output, run through the engines at every load")
        s.add_argument("--min-cosine", type=float,
                       help="fixture gate (default GR00T parity 0.999 on valves, pi0.5 0.9998 on the chunk)")
        s.add_argument("--max-abs", type=float,
                       help="fixture gate (default GR00T parity 0.02 on valves, pi0.5 0.01 on the chunk)")
        s.add_argument("--contract", help="GR00T: normalization contract JSON (default: the bundle's)")
        s.add_argument("--vla-root", default=str(Path.home() / "jetson-orin-nano-vla"),
                       help="GR00T: engines and host chain")
        s.add_argument("--history-tolerance-ms", type=float, default=150,
                       help="GR00T frame-history bundles: how far the past frame may sit from the trained lag "
                            "(back-to-back inference stores one every ~250 ms)")
    s = sub.choices["parity"]
    s.add_argument("--fixture", help="GR00T: one physical fixture (default: every prompt fixture in the bundle)")
    s.add_argument("--envelope", help="GR00T: the same fixture from bfloat16 Gr00tPolicy: per-sample error limits")
    s = sub.choices["run"]
    s.add_argument("--profile", default=str(NATIVE.parent / "configuration_files/profiles/jetson"))
    s.add_argument("--safety", default=str(NATIVE / "config/safety.yaml"))
    s.add_argument("--serial", default="/dev/ttyACM0")
    s.add_argument("--control-hz", type=int, default=200)
    s.add_argument("--start-services", action="store_true",
                   help="start kaivuri-control and both IR cameras here, stop them at the end "
                        "(otherwise pass --control-socket and two --camera-frame)")
    s.add_argument("--enable-slew", action="store_true")
    s.add_argument("--control-socket")
    s.add_argument("--camera-frame", action="append", help="cam1 then cam2 frame files")
    s.add_argument("--live", action="store_true",
                   help="drive the machine: plans, pump permission on (X toggles it on the pad), "
                        "live PWM with --start-services")
    s.add_argument("--gamepad", default="auto", help="evdev path, 'auto' (first by-id joystick) or 'none'")
    s.add_argument("--start-model", action="store_true", help="with a gamepad, start in model mode, not manual")
    s.add_argument("--task", help="starting prompt (default: the first trained task); added to the list")
    s.add_argument("--seconds", type=float, default=30, help="0 = until Ctrl+C")
    s.add_argument("--rate-hz", type=float, default=0, help="0 = back to back")
    s.add_argument("--fps", type=float, help="plan row rate (default: the bundle's dataset fps)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--action-offset-ms", type=float, default=0)
    s.add_argument("--max-frame-age-ms", type=float, default=200)
    s.add_argument("--max-imu-age-ms", type=float, default=50)
    s.add_argument("--max-observation-age-ms", type=float, default=1400)
    s.add_argument("--max-camera-skew-ms", type=float, default=50,
                   help="camera-to-camera arrival skew (recorded live max 22.8 ms; 30 fps = 33 ms/frame)")
    s.add_argument("--max-state-skew-ms", type=float, default=100, help="IMU sample vs each camera frame")
    s.add_argument("--first-plan-timeout-s", type=float, default=2.0,
                   help="model mode without a plan (start, A, prompt change) holds centred at most this long")
    s.add_argument("--rtc", choices=("auto", "on", "off"), default="auto",
                   help="auto: training-time RTC when the bundle has RTC inputs")
    s.add_argument("--rtc-margin-ms", type=float, default=40,
                   help="added to the recent max observation->result latency: supervisor tick + dispatch")
    a = p.parse_args()
    if getattr(a, "task", None) is not None and not a.task.strip():
        p.error("--task is empty (unset shell variable?); omit it to start on the first trained task")
    # The GR00T Policy changes into --vla-root (its engine builders import from there).
    for k in ("bundle", "contract", "cache", "vla_root", "out", "fixture", "envelope", "fixtures", "build",
              "profile", "safety", "control_socket"):
        if getattr(a, k, None):
            setattr(a, k, str(Path(getattr(a, k)).expanduser().resolve()))
    if getattr(a, "camera_frame", None):
        a.camera_frame = [str(Path(x).resolve()) for x in a.camera_frame]
    a.kind = bundle_kind(Path(a.bundle))
    if a.kind == "pi05":
        for k in ("contract", "fixture", "envelope"):
            if getattr(a, k, None):
                p.error(f"--{k} is for GR00T bundles")
        a.cache = a.cache or a.bundle + ".cache"
    else:
        if a.fixtures:
            p.error("--fixtures is for pi0.5 bundles (GR00T checks the fixtures inside its bundle)")
        a.cache = a.cache or str(Path.home() / ".cache/kaivuri-groot" / Path(a.bundle).name)
        a.min_cosine = 0.999 if a.min_cosine is None else a.min_cosine
        a.max_abs = 0.02 if a.max_abs is None else a.max_abs
    if a.cmd == "run":
        return run_live(a)
    if a.kind == "pi05":
        import vla_pi05
        return vla_pi05.parity(a)
    import vla_groot
    return vla_groot.parity(a)


if __name__ == "__main__":
    sys.exit(main())
