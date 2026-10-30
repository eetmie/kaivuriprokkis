#!/usr/bin/env python3
"""Inspect GR00T bundles, capture native temporal observations, or check prebuilt engines.

There is no drive/arm/PWM operation. Capture sends status requests only. Model
fixtures run in a separate command and return model-space diagnostics only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from policy_runtime.bundle import describe
from policy_runtime.history import Frame, FrameHistory, NotReady, robot_state


SOURCE = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def task_text(args):
    text = args.task_file.read_text().rstrip("\r\n") if args.task_file else args.task
    if not text or not text.strip() or len(text.encode()) > 8192:
        raise ValueError("provide nonempty task text of at most 8192 UTF-8 bytes")
    return text


def stop_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def synthetic_sources(args, stack):
    runtime = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="policy-probe-")))
    args.control_socket = runtime / "control.sock"
    args.camera_frame = [runtime / f"cameras/{name}-ir1.frame" for name in ("probe1", "probe2")]
    profile = SOURCE.parent / "configuration_files/profiles/jetson"
    commands = [
        [str(args.build / "kaivuri-control"), "--simulate-imu", "--control-hz", "200",
         "--profile", str(profile), "--safety", str(SOURCE / "config/safety.yaml"),
         "--socket", str(args.control_socket)],
        [str(args.build / "kaivuri-camera"), "--synthetic", "--width", "64", "--height", "48",
         "--camera", "probe1:1", "--camera", "probe2:1", "--output-dir", str(runtime / "cameras")],
    ]
    children = []
    for i, command in enumerate(commands):
        log = stack.enter_context((args.out / f"synthetic-{i}.log").open("w"))
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        stack.callback(stop_child, child)
    deadline = time.monotonic() + 5
    while not args.control_socket.exists() or not all(p.exists() for p in args.camera_frame):
        if any(child.poll() is not None for child in children):
            raise RuntimeError("synthetic service exited; see output logs")
        if time.monotonic() > deadline:
            raise TimeoutError("synthetic services did not start")
        time.sleep(.01)


def save_observation(root, index, slots, offsets, status, state, task, history):
    partial = root / f"{index:06d}.partial"
    partial.mkdir()
    files = []
    current = slots[-1]
    for slot_i, (offset, frames) in enumerate(zip(offsets, slots)):
        views = []
        for view_i, frame in enumerate(frames):
            m = frame.metadata
            name = f"slot{slot_i}-view{view_i}.pgm"
            data = f"P5\n{m['width']} {m['height']}\n255\n".encode() + frame.pixels
            (partial / name).write_bytes(data)
            views.append({"file": name, "sha256": hashlib.sha256(data).hexdigest(),
                          "metadata": m,
                          "actual_offset_ns": current[view_i].timestamp - frame.timestamp})
        files.append({"requested_offset_ns": offset, "views": views})
    record = {
        "schema": "kaivuri-observation/1", "index": index, "task": task,
        "captured_monotonic_ns": time.monotonic_ns(),
        # Old visual context is intentional: it is not the freshness of current input.
        "observation_ns": min(state["imu_host_monotonic_ns"], *(f.timestamp for f in current)),
        "state": state, "controller_status": status, "slots": files,
        "history_bytes": history.bytes, "history_memory_evictions": history.evictions,
        "actuator_output": False,
    }
    dump(partial / "observation.json", record)
    target = partial.with_suffix(".ready")
    partial.rename(target)
    return target


def capture(args):
    # Imports remain stdlib-only; FrameReader calls the native shared-memory C ABI.
    from client import Client
    from frame_reader import FrameReader, Metadata

    if args.out.exists():
        raise FileExistsError(f"choose a new output directory: {args.out}")
    if args.synthetic and (args.camera_frame or args.control_socket):
        raise ValueError("--synthetic cannot attach to externally supplied services")
    if not args.synthetic and not args.camera_frame:
        raise ValueError("--camera-frame is required when attaching to native services")
    if args.camera_frame and len({p.resolve() for p in args.camera_frame}) != len(args.camera_frame):
        raise ValueError("each camera view must use a distinct frame store")
    task_text(args)  # fail before creating output or opening services
    offsets = args.history_ms if args.history_ms is not None else [1000]
    history = FrameHistory(2 if args.synthetic else len(args.camera_frame), offsets,
                           args.history_tolerance_ms, int(args.history_memory_mb * 1024**2))
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {"schema": "kaivuri-probe-capture/1", "status": "starting",
               "actuator_output": False, "synthetic": args.synthetic,
               "history_offsets_ns": list(history.offsets), "snapshots": 0,
               "prompt_source": str(args.task_file) if args.task_file else "--task",
               "clock": "host CLOCK_MONOTONIC arrival timestamps; not exposure-aligned"}
    dump(args.out / "session.json", summary)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signals = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ExitStack() as stack:
            if args.synthetic:
                synthetic_sources(args, stack)
            socket_path = args.control_socket or Path(f"/tmp/kaivuri-native-{os.geteuid()}/control.sock")
            client = stack.enter_context(Client(socket_path))
            readers = [stack.enter_context(FrameReader(p, args.library, max_age_s=args.max_age_ms / 1000))
                       for p in args.camera_frame]
            summary.update(camera_frames=[str(p) for p in args.camera_frame],
                           control_socket=str(socket_path))
            dump(args.out / "session.json", summary)
            index = stack.enter_context((args.out / "observations.jsonl").open("x"))
            start = time.monotonic()
            next_sample, last_good = start, start
            deadline = start + args.duration_s
            last_sequences = None
            waiting = "waiting for cameras"
            while not stopped and time.monotonic() < deadline:
                for i, reader in enumerate(readers):
                    snapshot = reader.read()
                    if snapshot:
                        metadata, pixels = snapshot
                        history.push(i, Frame({n: getattr(metadata, n) for n, _ in Metadata._fields_},
                                              bytes(pixels)))
                now = time.monotonic()
                if now < next_sample:
                    time.sleep(.005)
                    continue
                try:
                    slots = history.select(time.monotonic_ns(), int(args.max_age_ms * 1e6),
                                           int(args.max_skew_ms * 1e6))
                    sequences = tuple(f.metadata["sequence"] for f in slots[-1])
                    if sequences == last_sequences:
                        raise NotReady("waiting for new camera frames")
                    status = client.status()  # the only control protocol operation
                    state = robot_state(status, time.monotonic_ns(), int(args.max_state_age_ms * 1e6))
                    # The status round trip also counts toward current camera age.
                    now_ns = time.monotonic_ns()
                    if any(now_ns - f.timestamp > args.max_age_ms * 1e6 for f in slots[-1]):
                        raise NotReady("camera frames aged during state read")
                except NotReady as error:
                    waiting = str(error)
                    if now - last_good > args.source_timeout_s:
                        raise TimeoutError(waiting) from error
                    time.sleep(.005)
                    continue
                task = task_text(args)
                path = save_observation(args.out, summary["snapshots"], slots, history.offsets,
                                        status, state, task, history)
                index.write(json.dumps({"directory": path.name, "task": task}) + "\n")
                index.flush()
                summary["snapshots"] += 1
                last_sequences, last_good = sequences, time.monotonic()
                next_sample = last_good + 1 / args.rate_hz
            if not summary["snapshots"]:
                raise RuntimeError(f"no complete temporal observation: {waiting}")
            summary.update(status="complete", interrupted=stopped,
                           history_budget_bytes=history.max_bytes,
                           history_memory_evictions=history.evictions)
            dump(args.out / "session.json", summary)
            return summary
    except BaseException as error:
        summary.update(status="incomplete", error=str(error))
        dump(args.out / "session.json", summary)
        raise
    finally:
        for s, previous in signals.items():
            signal.signal(s, previous)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="inspect the actual bundle; no GPU imports")
    inspect.add_argument("--bundle", type=Path, required=True)
    inspect.add_argument("--verify", action="store_true", help="hash all portable artifacts")
    cap = commands.add_parser("capture", help="capture real timestamped history; status-only IPC")
    cap.add_argument("--out", type=Path, required=True)
    prompt = cap.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--task", help="arbitrary instruction; no fixed task list")
    prompt.add_argument("--task-file", type=Path, help="UTF-8 text, reread for each snapshot")
    cap.add_argument("--camera-frame", type=Path, action="append")
    cap.add_argument("--control-socket", type=Path)
    cap.add_argument("--library", type=Path, default=SOURCE / "build/libkaivuri_frames.so")
    cap.add_argument("--build", type=Path, default=SOURCE / "build")
    cap.add_argument("--synthetic", action="store_true", help="start isolated simulated services")
    cap.add_argument("--history-ms", type=float, action="append",
                     help="past offsets; repeatable, default 1000 ms; zero adds current-only mode")
    cap.add_argument("--history-tolerance-ms", type=float, default=60)
    cap.add_argument("--history-memory-mb", type=float, default=128)
    cap.add_argument("--duration-s", type=float, default=10)
    cap.add_argument("--rate-hz", type=float, default=2)
    cap.add_argument("--max-age-ms", type=float, default=200)
    cap.add_argument("--max-state-age-ms", type=float, default=50)
    cap.add_argument("--max-skew-ms", type=float, default=40)
    cap.add_argument("--source-timeout-s", type=float, default=8)
    fixture = commands.add_parser("fixture", help="GR00T AOT engine check; model-space output only")
    fixture.add_argument("--bundle", type=Path, required=True)
    fixture.add_argument("--cache", type=Path, required=True)
    fixture.add_argument("--vla-root", type=Path, required=True)
    fixture.add_argument("--out", type=Path, required=True)
    fixture.add_argument("--iters", type=int, default=5)
    fixture.add_argument("--no-graph", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "inspect":
            result = describe(args.bundle, args.verify)
        elif args.command == "capture":
            for name in ("duration_s", "rate_hz", "history_memory_mb", "max_age_ms",
                         "max_state_age_ms", "max_skew_ms", "source_timeout_s"):
                value = getattr(args, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
            if args.rate_hz > 30 or args.max_state_age_ms > 200 or args.history_memory_mb > 1024:
                raise ValueError("rate <= 30 Hz, state age <= 200 ms and history budget <= 1024 MB required")
            result = capture(args)
        else:
            if not 1 <= args.iters <= 100:
                raise ValueError("--iters must be between 1 and 100")
            from policy_runtime.groot_fixture import run_fixture
            result = run_fixture(args.bundle, args.cache, args.vla_root, args.out,
                                 args.iters, not args.no_graph)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f"policy-probe: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
