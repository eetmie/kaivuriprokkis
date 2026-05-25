#!/usr/bin/env python3
"""Gamepad -> UDP velocity client for the smooth-reachability prototype.

Sends relative EE velocity commands `(vx, vy, vz, vyaw_dps)` over UDP,
piggy-backing on the existing ControlCommand wire format -- the four
`pose` floats are REINTERPRETED as velocity here.

This client only talks to control_prototype/excv_gui_relative.py
(default port 8081). It will NOT talk to the production excv_gui.py
on 8080 -- different port on purpose, see control_prototype/README.md.

Mapping (Xbox-style):
    LeftJoystickY     -> +vx   (forward / back)
    LeftJoystickX     -> +vy   (left / right)
    RightJoystickY    -> +vz   (up / down)
    RightJoystickX    -> +vyaw_dps
    LeftBumper held   -> slow mode (0.3x scale)
    Start             -> request pause/resume (toggle, debounced)

Run:
    python control_prototype/gamepad_velocity_client.py --host 192.168.0.132
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for p in (str(_HERE), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from modules.control_protocol import (
    COMMAND_PACKET_SIZE,
    ControlCommand,
    ControlMode,
    PoseTarget,
    TELEMETRY_PACKET_SIZE,
    decode_telemetry_message,
    encode_command_message,
)
from modules.udp_socket import UDPSocket


def _try_open_gamepad():
    try:
        from modules.gamepad import XboxController
    except Exception as exc:
        print(f"[client] gamepad import failed: {exc!r}")
        return None
    try:
        return XboxController(max_reconnect=3, deadzone=15.0)
    except Exception as exc:
        print(f"[client] gamepad open failed: {exc!r}")
        return None


def _build_velocity_command(
    gamepad,
    *,
    seq: int,
    max_lin_mps: float,
    max_yaw_dps: float,
    pause_state: dict,
) -> ControlCommand:
    state = gamepad.read() if gamepad is not None else {}

    slow_scale = 0.3 if int(state.get("LeftBumper", 0)) else 1.0
    vx = float(state.get("LeftJoystickY", 0.0)) * max_lin_mps * slow_scale
    vy = float(state.get("LeftJoystickX", 0.0)) * max_lin_mps * slow_scale
    vz = float(state.get("RightJoystickY", 0.0)) * max_lin_mps * slow_scale
    vyaw = float(state.get("RightJoystickX", 0.0)) * max_yaw_dps * slow_scale

    # Edge-triggered pause toggle on Start.
    start_now = bool(int(state.get("Start", 0)))
    pause_req = False
    resume_req = False
    if start_now and not pause_state["last"]:
        if pause_state["paused"]:
            resume_req = True
            pause_state["paused"] = False
        else:
            pause_req = True
            pause_state["paused"] = True
    pause_state["last"] = start_now

    # Reinterpret the four pose floats as (vx, vy, vz, vyaw_dps).
    return ControlCommand(
        sequence=seq,
        mode=ControlMode.IK,
        pose=PoseTarget(x=vx, y=vy, z=vz, rot_y_deg=vyaw),
        pause=pause_req,
        resume=resume_req,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Gamepad UDP velocity client for the prototype")
    parser.add_argument("--host", default="192.168.0.132")
    parser.add_argument("--port", type=int, default=8081,
                        help="prototype server port (production stack uses 8080)")
    parser.add_argument("--rate", type=float, default=50.0, help="UDP send rate (Hz)")
    parser.add_argument("--max-lin", type=float, default=0.10, help="max linear velocity (m/s) at full stick")
    parser.add_argument("--max-yaw", type=float, default=20.0, help="max yaw rate (deg/s) at full stick")
    parser.add_argument("--dry-run", action="store_true",
                        help="don't open the gamepad; send zeros (useful for handshake/protocol tests)")
    args = parser.parse_args()

    gamepad = None if args.dry_run else _try_open_gamepad()
    if gamepad is None and not args.dry_run:
        print("[client] no gamepad; falling back to dry-run (zero velocity).")

    udp = UDPSocket(local_id=3, max_age_seconds=0.5, nominal_rate_hz=args.rate)
    udp.setup(args.host, args.port,
              inputs=f'{TELEMETRY_PACKET_SIZE}b',
              outputs=f'{COMMAND_PACKET_SIZE}b',
              is_server=False)

    print(f"[client] handshaking with {args.host}:{args.port} at {args.rate:.1f} Hz...")
    if not udp.handshake(timeout=10.0):
        print("[client] handshake failed.")
        return 1
    print("[client] connected.")

    udp.start_receiving()

    period = 1.0 / args.rate
    next_t = time.perf_counter()
    seq = 0
    pause_state = {"last": False, "paused": False}
    last_print = time.time()

    try:
        while True:
            cmd = _build_velocity_command(
                gamepad,
                seq=seq,
                max_lin_mps=args.max_lin,
                max_yaw_dps=args.max_yaw,
                pause_state=pause_state,
            )
            udp.send(encode_command_message(cmd))
            seq += 1

            # Read whatever telemetry came back.
            data = udp.get_latest()
            now = time.time()
            if data and (now - last_print) >= 0.5:
                try:
                    tlm = decode_telemetry_message(data)
                    p = tlm.measured_pose
                    print(
                        f"[client] v=({cmd.pose.x:+.3f},{cmd.pose.y:+.3f},{cmd.pose.z:+.3f}) m/s, "
                        f"yaw={cmd.pose.rot_y_deg:+.1f} dps  |  "
                        f"measured=({p.x:+.3f},{p.y:+.3f},{p.z:+.3f}) m, rot={p.rot_y_deg:+.1f} deg  |  "
                        f"paused={tlm.paused} hw_ready={tlm.hardware_ready}"
                    )
                except Exception:
                    pass
                last_print = now

            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[client] interrupted.")
    finally:
        try:
            udp.close()
        except Exception:
            pass
        if gamepad is not None:
            try:
                gamepad.stop_monitoring()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
