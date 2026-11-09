#!/usr/bin/env python3
"""Valtra T171 front-loader X-VLA demo: one USB camera -> policy -> ISOBUS valve commands.

    check   load the bundle (first run builds TensorRT engines, ~5 min), run one chunk
    camera  open the camera, report size/fps, save camera.jpg (compare reference_front.jpg)
    listen  decode valve 7/8 commands already on the bus (verify adapter, IDs, signs)
    run     camera -> policy -> valve commands; prints only, --live transmits

Model contract (bundle trained on AleksiL/valtra_t171_pallet, trimmed):
  image  observation.images.front, RGB, 1280x720 as recorded
  state  [0, 0] (no proprioception was recorded)
  action [boom, bucket] flow, scaled /100 in training -> x100 here = signed valve %
CAN: ISO 11783-7 Auxiliary Valve Command, valve 7 = boom (0CFE3728), valve 8 = bucket
(0CFE3828), 250 kbit/s, flow 0.4 %/bit. Dataset signs: boom + = raise, bucket + = curl in.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VALVES = {"boom": 7, "bucket": 8}
STATES = {0: "block", 1: "extend", 2: "retract", 3: "float"}


# ---- ISOBUS auxiliary valve command -------------------------------------------------

def avc_id(valve: int, sa: int) -> int:
    """Priority 3, PGN 0xFE30 + valve, source address sa."""
    return (3 << 26) | ((0xFE30 + valve) << 8) | sa


def encode(pct: float, plus_is_extend: bool) -> bytes:
    flow = min(250, round(abs(pct) / 0.4))
    state = 0 if flow == 0 else 1 if (pct > 0) == plus_is_extend else 2
    # Same layout as the Valtra terminal (captured 2026-11-07): idle 00 00 C0 00 00 00 00 00
    return bytes([flow, 0x00, 0xC0 | state, 0x00, 0x00, 0x00, 0x00, 0x00])


def decode(data: bytes, plus_is_extend: bool) -> tuple[float, int]:
    flow = data[0] * 0.4 if data[0] <= 250 else float("nan")
    state = data[2] & 0x0F
    sign = {1: 1, 2: -1}.get(state, 0) * (1 if plus_is_extend else -1)
    return sign * flow, state


def _selftest() -> None:
    assert avc_id(7, 0x28) == 0x0CFE3728 and avc_id(8, 0x28) == 0x0CFE3828
    for pct in (0.0, 0.1, 12.4, -22.0, 50.0, 100.0, -100.0):
        for ext in (True, False):
            back, _ = decode(encode(pct, ext), ext)
            assert abs(back - pct) <= 0.2 + 1e-9, (pct, ext, back)
    assert encode(130, True)[0] == 250
    assert encode(0, True) == bytes.fromhex("0000c00000000000")       # terminal idle frame
    assert encode(94.8, True) == bytes.fromhex("ed00c10000000000")    # terminal boom-up frame


def open_bus(spec: str, bitrate: int, tries: int = 6):
    """socketcan:can0 | slcan:/dev/ttyACM0 | gs_usb:0 (python-can interface:channel)."""
    import can
    iface, _, chan = spec.partition(":")
    if iface == "socketcan":   # bitrate is set on the link: sudo ip link set can0 up type can bitrate 250000
        return can.Bus(interface="socketcan", channel=chan or "can0")
    if iface == "gs_usb":      # candleLight: no kernel driver on this board, libusb instead
        # start() resets the USB device, which re-enumerates it; that fails ("Entity not found")
        # about every other time on this board, and the next attempt works.
        for attempt in range(1, tries + 1):
            try:
                return can.Bus(interface="gs_usb", channel="gs_usb", index=int(chan or 0), bitrate=bitrate)
            except Exception as e:
                logging.warning("gs_usb open failed (%d/%d): %s", attempt, tries, e)
                if attempt == tries:
                    raise
                time.sleep(1.5)
    return can.Bus(interface=iface, channel=chan, bitrate=bitrate)


def valve_frames(bus, seconds: float):
    """Yield (name, sa, data) for valve 7/8 commands seen within `seconds` (or forever if None)."""
    end = None if seconds is None else time.monotonic() + seconds
    pgns = {0xFE30 + v: n for n, v in VALVES.items()}
    while end is None or time.monotonic() < end:
        msg = bus.recv(0.2)
        if msg is None:
            yield None
            continue
        name = pgns.get((msg.arbitration_id >> 8) & 0xFFFF) if msg.is_extended_id else None
        yield (name, msg.arbitration_id & 0xFF, bytes(msg.data)) if name else ("other", 0, b"")


# ---- camera ------------------------------------------------------------------------

class Camera:
    """Latest-frame USB camera reader (MJPEG 1280x720 @ 30 like the recording)."""

    def __init__(self, dev: str, width=1280, height=720, fps=30):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(int(dev) if dev.isdigit() else dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera {dev} (list them: v4l2-ctl --list-devices)")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.size != (width, height):
            logging.warning("Camera gives %dx%d, model was trained on %dx%d: framing/padding differ",
                            *self.size, width, height)
        self.frame, self.t, self.count, self.running = None, 0.0, 0, True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self.frame, self.t, self.count = frame, time.monotonic(), self.count + 1
            else:
                time.sleep(0.01)

    def latest(self):
        """(RGB HxWx3 uint8, capture time) or (None, 0)."""
        frame = self.frame
        return (None, 0.0) if frame is None else (self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB), self.t)

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)  # release() under a running read() crashes OpenCV
        self.cap.release()


# ---- policy ------------------------------------------------------------------------

def load_policy(bundle: Path):
    sys.path.insert(0, str(HERE))
    from lerobot_vla.runtime.xvla import XVLAExcavatorPolicy
    policy = XVLAExcavatorPolicy(split_dir=bundle)
    info = json.loads((bundle / "bundle.json").read_text())
    if policy.action_dim != 2 or policy.state_dim != 2:
        raise SystemExit(f"Expected a 2-state/2-action Valtra bundle, got {policy.state_dim}/{policy.action_dim}")
    logging.info("fixture parity: %s", policy.policy.fixture_parity["status"])
    return policy, info["tasks"], float(info["fps"])


def open_pad():
    """Local Xbox pad, or None (the run works without one)."""
    sys.path.insert(0, str(HERE))
    from lerobot_vla.gamepad import LocalGamepadInput
    pad = LocalGamepadInput(connect_timeout_s=3.0)
    return pad if pad.open() else None


ZERO_STATE = np.zeros(2, np.float32)


def predict(policy, rgb, task):
    """Chunk [30, 2] in signed valve % (training scaled actions by 1/100)."""
    return policy.sample_actions(rgb, task, ZERO_STATE) * 100.0


# ---- commands ----------------------------------------------------------------------

def cmd_check(a):
    import cv2
    policy, tasks, fps = load_policy(a.bundle)
    rgb = cv2.cvtColor(cv2.imread(str(HERE / "reference_front.jpg")), cv2.COLOR_BGR2RGB)
    print(f"reference_front.jpg is a 'lift the pallet' frame at loader start; fps {fps:g}")
    for task in tasks:
        for _ in range(3):
            chunk = predict(policy, rgb, task)
        ms = policy.last_timings.get("total_ms", float("nan"))
        print(f"{task!r:22} {ms:6.1f} ms  boom % first 10: {np.round(chunk[:10, 0], 1).tolist()}")
        print(f"{'':22} {'':9} bucket %:         {np.round(chunk[:10, 1], 1).tolist()}")
    policy.close()


def cmd_camera(a):
    cam = Camera(a.camera)
    time.sleep(2.0)
    n0, t0 = cam.count, time.monotonic()
    time.sleep(3.0)
    rate = (cam.count - n0) / (time.monotonic() - t0)
    rgb, _ = cam.latest()
    cam.close()
    if rgb is None:
        raise SystemExit("No frames from the camera")
    out = HERE / "camera.jpg"
    cam.cv2.imwrite(str(out), cam.cv2.cvtColor(rgb, cam.cv2.COLOR_RGB2BGR))
    print(f"{a.camera}: {cam.size[0]}x{cam.size[1]} at {rate:.1f} fps -> {out}")
    print(f"Compare with {HERE / 'reference_front.jpg'}: fork carriage at the bottom, looking forward/down.")


def cmd_listen(a):
    bus = open_bus(a.can, a.bitrate)
    sign = {"boom": a.boom_raise == "extend", "bucket": a.bucket_curl == "extend"}
    last, senders, n, t_print = {}, Counter(), 0, 0.0
    print("Move the loader with the tractor's own joystick. Boom up must read +, bucket curl-in +;\n"
          "if not, flip --boom-raise / --bucket-curl. Ctrl+C to stop.")
    try:
        for item in valve_frames(bus, None):
            if item:
                name, sa, data = item
                n += 1
                if name != "other":
                    last[name] = (*decode(data, sign[name]), data.hex(" "))
                    senders[(name, sa)] += 1
            if time.monotonic() - t_print > 0.2:
                t_print = time.monotonic()
                parts = [f"{k} {p:+6.1f}% {STATES.get(s, s):7} [{raw}]" for k, (p, s, raw) in sorted(last.items())]
                src = " ".join(f"{k}@SA{sa:02X}" for k, sa in senders)
                print(f"\r{n:7d} frames  {' | '.join(parts) or 'no valve 7/8 commands yet'}  {src}   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        bus.shutdown()


def cmd_run(a):
    sign = {"boom": a.boom_raise == "extend", "bucket": a.bucket_curl == "extend"}
    ids = {k: avc_id(v, a.sa) for k, v in VALVES.items()}
    bus = None
    if a.live:
        import can
        bus = open_bus(a.can, a.bitrate)
        others = {(n, sa) for item in valve_frames(bus, 1.0) if item for n, sa, _ in [item] if n != "other"}
        if others and not a.allow_other_sender:
            bus.shutdown()
            seen = ", ".join(f"{n} from SA 0x{sa:02X}" for n, sa in sorted(others))
            raise SystemExit(f"Another node is already commanding the valves ({seen}). Two senders "
                             f"fight over the valves. Disconnect it, or pass --allow-other-sender.")
    policy, tasks, fps = load_policy(a.bundle)
    cam = Camera(a.camera)

    pad = None if a.no_gamepad else open_pad()
    lock = threading.Lock()
    # mode: STOP (zeros) | POLICY (chunks) | MANUAL (right stick). With a pad we start in MANUAL:
    # centred sticks send 0, A hands over to the policy (run_inference's takeover toggle).
    st = {"mode": "MANUAL" if pad else "STOP", "task": 0, "chunk": None, "t0": 0.0, "ms": 0.0,
          "running": True}

    def say(msg):
        print(f"\n>>> {msg}", flush=True)

    def keyboard():
        for line in sys.stdin:
            line = line.strip()
            with lock:
                if line.isdigit() and 1 <= int(line) <= len(tasks):
                    st.update(mode="POLICY", task=int(line) - 1, chunk=None)
                else:
                    st.update(mode="STOP", chunk=None)
                say(st["mode"] if st["mode"] == "STOP" else f"POLICY {tasks[st['task']]!r}")

    def buttons():   # edges only, polled on their own thread so a tap is never missed
        from lerobot_vla.gamepad import BTN_A, BTN_DPAD_LEFT, BTN_DPAD_RIGHT
        prev = 0
        while st["running"]:
            _, mask = pad.poll()
            hit, prev = mask & ~prev, mask
            step = bool(hit & (1 << BTN_DPAD_RIGHT)) - bool(hit & (1 << BTN_DPAD_LEFT))
            with lock:
                if hit & (1 << BTN_A):
                    st.update(mode="MANUAL" if st["mode"] == "POLICY" else "POLICY", chunk=None)
                    say(f"{st['mode']} {tasks[st['task']]!r}" if st["mode"] == "POLICY" else "MANUAL (right stick)")
                if step:
                    st.update(task=(st["task"] + step) % len(tasks), chunk=None)
                    say(f"task {tasks[st['task']]!r}")
            time.sleep(0.02)

    def infer():
        while st["running"]:
            mode, task = st["mode"], st["task"]
            rgb, t = cam.latest()
            if mode != "POLICY" or rgb is None or time.monotonic() - t > a.max_frame_age:
                time.sleep(0.01)
                continue
            chunk = predict(policy, rgb, tasks[task])
            with lock:
                if (st["mode"], st["task"]) == (mode, task):   # drop a chunk the operator just left
                    st.update(chunk=chunk, t0=t, ms=policy.last_timings.get("total_ms", 0.0))

    menu = "  ".join(f"{i + 1}={t!r}" for i, t in enumerate(tasks))
    print(f"\nKeyboard: {menu} + Enter starts the policy; empty Enter = STOP. Ctrl+C quits.")
    if pad:
        print("Gamepad: A = policy <-> manual, D-pad left/right = task, right stick up/down = boom, "
              "left/right = bucket (left = +).")
    threading.Thread(target=keyboard, daemon=True).start()
    if pad:
        threading.Thread(target=buttons, daemon=True).start()
    worker = threading.Thread(target=infer, daemon=True)
    worker.start()

    def send(cmd):
        nonlocal bus
        if bus is None:
            return
        for attempt in (0, 1):
            try:
                for k, pct in zip(VALVES, cmd):
                    bus.send(can.Message(arbitration_id=ids[k], data=encode(pct, sign[k]), is_extended_id=True))
                return
            except can.CanError as e:   # adapter wedged / USB timeout: reopen it once, then give up
                if attempt:
                    raise
                print(f"\nCAN send failed ({e}); reconnecting...", flush=True)
                try:
                    bus.shutdown()
                except Exception:
                    pass
                bus = open_bus(a.can, a.bitrate)
                print("CAN reconnected", flush=True)

    period, t_next, t_print = 1.0 / (a.send_hz or fps), time.monotonic(), 0.0
    print(f"{'LIVE: transmitting on ' + a.can if a.live else 'DRY RUN: printing only (add --live to drive)'}; "
          f"send {1 / period:g} Hz (policy {fps:g}), clamp +-{a.max_flow:g} %, deadband {a.deadband:g} %, SA 0x{a.sa:02X}")
    try:
        while True:
            now = time.monotonic()
            with lock:
                mode, task, chunk, t0, ms = st["mode"], st["task"], st["chunk"], st["t0"], st["ms"]
            k = int((now - t0) * fps)   # step of the chunk that belongs to "now" (skips inference latency)
            cmd = np.zeros(2)
            if mode == "POLICY" and chunk is not None and k < len(chunk):
                cmd = chunk[k].astype(float)
            elif mode == "MANUAL" and pad.is_live():   # a lost pad sends 0 and stays manual
                axes, _ = pad.poll()
                cmd = np.array([axes["right_ud"], axes["right_rl"]]) * a.manual_flow
            cmd = np.clip(cmd, -a.max_flow, a.max_flow)
            cmd[np.abs(cmd) < a.deadband] = 0.0
            send(cmd)
            if now - t_print > 0.2:
                t_print = now
                label = f"POLICY {tasks[task]}" if mode == "POLICY" else mode
                step = f"step {k:2d}  " if mode == "POLICY" else ""
                print(f"\r{label:24} boom {cmd[0]:+6.1f}%  bucket {cmd[1]:+6.1f}%  {step}"
                      f"infer {ms:5.0f} ms  frame age {now - cam.t:4.2f} s   ", end="", flush=True)
            t_next += period
            time.sleep(max(0.0, t_next - time.monotonic()))
    except KeyboardInterrupt:
        print()
    finally:
        st["running"] = False
        for _ in range(5):             # leave both valves blocked
            try:
                send(np.zeros(2))
            except Exception as e:
                print(f"could not send the final zero frames: {e}", flush=True)
                break
            time.sleep(period)
        worker.join(timeout=2)
        # each step guarded: one failure must not skip the rest (an unclosed bus/policy segfaults at exit)
        for close in (lambda: bus and bus.shutdown(), lambda: pad and pad.close(), cam.close, policy.close):
            try:
                close()
            except Exception as e:
                print(f"cleanup: {e}", flush=True)


def main():
    _selftest()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "camera", "listen", "run"])
    p.add_argument("--bundle", type=Path, default=HERE / "bundle")
    p.add_argument("--camera", default="/dev/video0", help="V4L2 device or index")
    p.add_argument("--can", default="socketcan:can0", help="socketcan:can0 | slcan:/dev/ttyACM0 | gs_usb:0")
    p.add_argument("--bitrate", type=int, default=250000)
    p.add_argument("--sa", type=lambda s: int(s, 0), default=0x28, help="our source address (dataset: 0x28)")
    p.add_argument("--boom-raise", choices=["extend", "retract"], default="extend",
                   help="valve state that raises the boom (check with `listen`)")
    p.add_argument("--bucket-curl", choices=["extend", "retract"], default="extend",
                   help="valve state that curls the bucket in (check with `listen`)")
    p.add_argument("--max-flow", type=float, default=50.0, help="clamp, %% (training range -22..+50)")
    p.add_argument("--deadband", type=float, default=1.0, help="|cmd| below this %% is sent as 0")
    p.add_argument("--max-frame-age", type=float, default=0.3, help="s; older frames are not used")
    p.add_argument("--manual-flow", type=float, default=50.0, help="%% at full stick in MANUAL")
    p.add_argument("--send-hz", type=float, default=0,
                   help="CAN send rate, 0 = policy fps; above the other sender's rate our frames win")
    p.add_argument("--no-gamepad", action="store_true")
    p.add_argument("--live", action="store_true", help="transmit on CAN (default prints only)")
    p.add_argument("--allow-other-sender", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    {"check": cmd_check, "camera": cmd_camera, "listen": cmd_listen, "run": cmd_run}[a.command](a)


if __name__ == "__main__":
    main()
