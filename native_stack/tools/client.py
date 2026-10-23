"""Small stdlib-only producer for the native control protocol. No hardware imports."""
import argparse
import json
import os
import socket
import struct
import subprocess
import time

HEADER = struct.Struct("<4sHHIQQIfQQ")
DRIVE, DISARM, ESTOP, STATUS = 1, 2, 3, 4
DEADMAN, ARM, PUMP = 1, 2, 4
RUNTIME_DIR = f"/tmp/kaivuri-native-{os.geteuid()}"  # sockets and the controller lock
DEFAULT_SOCKET = f"{RUNTIME_DIR}/control.sock"


class Client:
    def __init__(self, path=DEFAULT_SOCKET):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.socket.settimeout(1)
        self.socket.connect(str(path))
        self.sequence = 0

    def send(self, op, steps=(), *, flags=0, fps=30, t0_ns=0, observation_ns=0):
        self.sequence += 1
        packet = HEADER.pack(b"KVR1", 1, op, flags, self.sequence,
                             time.monotonic_ns(), len(steps), fps, t0_ns, observation_ns)
        for step in steps:
            if len(step) != 6:
                raise ValueError("six controls required: slew,boom,arm,bucket,trackL,trackR")
            packet += struct.pack("<6f", *step)
        self.socket.sendall(packet)

    def status(self):
        self.send(STATUS)
        data, _, flags, _ = self.socket.recvmsg(8192)
        if flags & socket.MSG_TRUNC:
            raise RuntimeError("native status reply exceeds the 8192-byte receive buffer")
        return json.loads(data)

    def arm(self, *, pump=False, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.send(DRIVE, [(0,) * 6], flags=DEADMAN | ARM | (PUMP if pump else 0))
            time.sleep(.025)
            status = self.status()
            if status["state"] == "armed":
                return status
            if status["state"] == "fault":
                raise RuntimeError(status["reason"])
        raise TimeoutError("arming did not complete")

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def stop(processes, timeout=10):
    """Terminate every running process, then wait for each; kill any still alive after timeout."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "disarm", "estop"):
        sub.add_parser(name)
    exercise = sub.add_parser("exercise", help="arm, drive bounded values, then disarm")
    exercise.add_argument("--values", default="0,0,0,0,0,0", help="six comma-separated controls")
    exercise.add_argument("--seconds", type=float, default=1)
    exercise.add_argument("--pump", action="store_true", help="explicit pump permission")
    exercise.add_argument("--max-abs", type=float, default=.15, help="bench command bound")
    args = parser.parse_args()
    with Client(args.socket) as client:
        if args.command == "status":
            print(json.dumps(client.status(), indent=2))
        elif args.command in ("disarm", "estop"):
            client.send(DISARM if args.command == "disarm" else ESTOP)
            print(json.dumps(client.status(), indent=2))
        else:
            values = tuple(float(v) for v in args.values.split(","))
            if len(values) != 6 or not 0 < args.seconds <= 10 or not 0 < args.max_abs <= 1:
                parser.error("six controls, 0 < seconds <= 10, 0 < max-abs <= 1 required")
            if any(not abs(v) <= args.max_abs for v in values):
                parser.error("control exceeds --max-abs")
            try:
                client.send(DISARM)
                client.arm(pump=args.pump)
                end = time.monotonic() + args.seconds
                while time.monotonic() < end:
                    client.send(DRIVE, [values], flags=DEADMAN | (PUMP if args.pump else 0))
                    time.sleep(.025)
                print(json.dumps(client.status(), indent=2))
            finally:
                client.send(DISARM)


if __name__ == "__main__":
    main()
