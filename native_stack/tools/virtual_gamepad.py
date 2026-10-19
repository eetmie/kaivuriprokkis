#!/usr/bin/env python3
"""Play an operator through a FIFO that kaivuri-record opens as its gamepad.

This kernel has no uinput, so kaivuri-record treats a FIFO given to --gamepad as a virtual
Xbox pad: raw Linux input_event structs, xpad axis ranges. Everything after the event read
(deadzone, signs, buttons, arming, pump permission, live PWM) is the real teleop path.

Start this first; it creates the FIFO and blocks until the recorder opens it:

    python3 tools/virtual_gamepad.py /tmp/pad.fifo --episodes 4 --seconds 20 &
    python3 tools/record_native.py --session S --task T --gamepad /tmp/pad.fifo --enable-slew

Each take: A (arm + start), X (pump permission), smooth stick motion, A (save). With
--discard-every N, every Nth take ends with B instead. After the last take the script
sends SIGINT to --stop-pid (the launcher), which shuts the session down cleanly.
"""
import argparse
import math
import os
import random
import signal
import struct
import time

EV_SYN, EV_KEY, EV_ABS = 0, 1, 3
ABS_X, ABS_Y, ABS_RX, ABS_RY = 0x00, 0x01, 0x03, 0x04
BTN_SOUTH, BTN_EAST, BTN_NORTH, BTN_WEST = 0x130, 0x131, 0x133, 0x134
EVENT = struct.Struct('llHHi')  # struct input_event on 64-bit Linux


class Pad:
    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY)  # blocks until the recorder opens the FIFO

    def emit(self, events):
        now = time.time()
        sec, usec = int(now), int((now % 1) * 1e6)
        data = b''.join(EVENT.pack(sec, usec, t, c, v) for t, c, v in events)
        os.write(self.fd, data + EVENT.pack(sec, usec, EV_SYN, 0, 0))

    def press(self, button, hold=0.15):
        self.emit([(EV_KEY, button, 1)])
        time.sleep(hold)
        self.emit([(EV_KEY, button, 0)])
        time.sleep(0.2)

    def sticks(self, slew=0.0, lift=0.0, tilt=0.0, scoop=0.0):
        # Inverse of Pad::action: slew = -X, lift = -RY, tilt = Y, scoop = -RX.
        raw = lambda v: int(max(-1.0, min(1.0, v)) * 32767)
        self.emit([(EV_ABS, ABS_X, raw(-slew)), (EV_ABS, ABS_RY, raw(-lift)),
                   (EV_ABS, ABS_Y, raw(tilt)), (EV_ABS, ABS_RX, raw(-scoop))])


def drive(pad, seconds, rate, rng):
    """Smooth stick motion: per-joint sums of two sines with random phase, eased in and out."""
    params = [[(rng.uniform(0.1, 0.6), rng.uniform(0, 2 * math.pi)) for _ in range(2)] for _ in range(4)]
    start = time.monotonic()
    while (t := time.monotonic() - start) < seconds:
        ease = min(1.0, t / 1.5, (seconds - t) / 1.5)
        values = [ease * 0.5 * sum(math.sin(2 * math.pi * f * t + p) for f, p in joint) for joint in params]
        pad.sticks(*values)
        time.sleep(1 / rate)
    pad.sticks()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('fifo')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--seconds', type=float, default=20)
    p.add_argument('--startup-s', type=float, default=8, help='wait for IMUs/cameras before the first take')
    p.add_argument('--gap-s', type=float, default=4, help='pause after each take for the disk flush')
    p.add_argument('--discard-every', type=int, default=0)
    p.add_argument('--rate-hz', type=float, default=60)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stop-pid', type=int, help='send SIGINT here after the last take')
    args = p.parse_args()
    if not os.path.exists(args.fifo):
        os.mkfifo(args.fifo, 0o600)
    rng = random.Random(args.seed)
    pad = Pad(args.fifo)
    pad.sticks()
    time.sleep(args.startup_s)
    for episode in range(args.episodes):
        pad.press(BTN_SOUTH)            # A: start (the recorder arms itself)
        time.sleep(0.6)
        pad.press(BTN_NORTH)            # X: pump permission (Linux BTN_NORTH is the X label)
        drive(pad, args.seconds, args.rate_hz, rng)
        discard = args.discard_every and (episode + 1) % args.discard_every == 0
        pad.press(BTN_EAST if discard else BTN_SOUTH)   # B discard / A save
        print(f'virtual pad: take {episode + 1} {"discarded" if discard else "saved"}', flush=True)
        time.sleep(args.gap_s)
    if args.stop_pid:
        os.kill(args.stop_pid, signal.SIGINT)
    time.sleep(5)  # keep the FIFO open until the launcher has stopped the recorder
    os.close(pad.fd)


if __name__ == '__main__':
    main()
