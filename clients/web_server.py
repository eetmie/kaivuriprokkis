#!/usr/bin/env python3
"""
Browser-based Excavator controller — lightweight HTTP + SSE server.

Drop-in replacement for the tkinter/matplotlib client_gui.py. Moves all
rendering to the browser (Three.js + the real URDF/OBJ meshes) and keeps
Python doing only what it's good at: UDP I/O and the IK tick loop.

Uses stdlib only — no Flask, no websockets, no external Python deps.

Architecture:
    Browser keyboard events  --POST /input-->  server keys_down set
    Server 30Hz tick         --deltas-->       target_state / direct_commands
    UDP send thread (20Hz)   --ControlCommand--> excavator
    UDP recv thread          --telemetry-->     server state
    Server state             --SSE /events-->  browser UI

Run:
    python clients/web_server.py
Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

from modules.control_protocol import (
    ControlCommand,
    ControlMode,
    DirectCommand,
    PoseTarget,
    RobotTelemetry,
)
from modules.excavator_target_state import ExcavatorTargetState, IKControlSpace
from clients.input_handler import InputHandler
from clients.udp_client import ExcavatorUDPClient


DEFAULT_X = 0.6
DEFAULT_Y = 0.0
DEFAULT_Z = 0.0

UDP_HOST = "192.168.0.132"
UDP_PORT = 8080
UDP_SEND_RATE_HZ = 20.0

KEY_STEP_POS = 0.001
KEY_STEP_ROT = 0.1

TICK_HZ = 30.0
BROADCAST_HZ = 15.0

WEB_HOST = "0.0.0.0"
WEB_PORT = 8000

_HERE = Path(__file__).resolve().parent


def _robot_static_root() -> Path:
    """Find the robot folder, preferring the current/generated model location."""
    candidates = [
        Path.cwd() / "robot",
        _ROOT_DIR / "models" / "robot",
        _HERE / "robot",
    ]
    for candidate in candidates:
        if (candidate / "robot.urdf").is_file():
            return candidate
    return _HERE / "robot"


STATIC_ROOTS = {
    # URL prefix -> directory on disk. Longest prefix wins.
    "/robot/": _robot_static_root(),
    "/vendor/urdf-loader/": _HERE / "urdf-loaders-master" / "javascript" / "src",
    "/": _HERE / "web",
}


class ExcavatorApp:
    """Server-side state mirroring what ExcavatorGUI3D used to hold."""

    def __init__(self):
        self.target_state = ExcavatorTargetState.from_cartesian_pose(
            DEFAULT_X, DEFAULT_Y, DEFAULT_Z, 0.0,
        )
        self.ik_control_space = IKControlSpace.CARTESIAN

        self.measured_x = DEFAULT_X
        self.measured_y = DEFAULT_Y
        self.measured_z = DEFAULT_Z
        self.measured_rot_y = 0.0
        self._has_telemetry = False
        self._sync_target_from_telemetry = False

        self.keys_down: set[str] = set()
        self.step_pos = KEY_STEP_POS
        self.step_rot = KEY_STEP_ROT

        self.direct_mode = False
        self.direct_commands = [0.0, 0.0, 0.0, 0.0]

        self.joint_positions: Optional[List[List[float]]] = None
        self.joint_angles_deg: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self.command_sequence = 0

        self.reload_flag = 0
        self.pause_flag = 0
        self.resume_flag = 0

        self.host = UDP_HOST
        self.port = UDP_PORT
        self.rate_hz = UDP_SEND_RATE_HZ

        self.send_hz_actual = 0.0
        self.status_message = "Disconnected"
        self.recv_label = "None"

        self.input_handler = InputHandler()
        self.udp_client = ExcavatorUDPClient()

        self._lock = threading.Lock()

        self._subscribers: List[queue.Queue] = []
        self._subscribers_lock = threading.Lock()

        self._running = True
        self._last_broadcast = 0.0
        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def _target_pose(self) -> tuple[float, float, float, float]:
        return self.target_state.compose_cartesian_pose()

    def _sync_target_from_measured(self) -> bool:
        if not self._has_telemetry:
            return False
        self.target_state.sync_from_cartesian_pose(
            (self.measured_x, self.measured_y, self.measured_z),
            self.measured_rot_y,
        )
        return True

    def _tick_loop(self):
        period = 1.0 / TICK_HZ
        next_t = time.perf_counter()
        while self._running:
            self._tick()
            self._maybe_broadcast()
            next_t += period
            remaining = next_t - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_t = time.perf_counter()

    def _tick(self):
        with self._lock:
            keys = set(self.keys_down)
            space = self.ik_control_space
            direct = self.direct_mode
            step_pos = self.step_pos
            step_rot = self.step_rot

        if direct:
            slew, boom, arm, bucket = self.input_handler.tick_direct(keys, None)
            with self._lock:
                self.direct_commands = [slew, boom, arm, bucket]
            return

        d0, d1, dz, drot = self.input_handler.tick_ik(
            keys, None, step_pos, step_rot, control_space=space,
            radius=self.target_state.radius,
        )
        if not (d0 or d1 or dz or drot):
            return
        with self._lock:
            if space == IKControlSpace.CARTESIAN:
                self.target_state.apply_cartesian_delta(d0, d1, dz, drot)
            else:
                self.target_state.apply_radial_delta(d0, d1, dz, drot)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=8)
        with self._subscribers_lock:
            self._subscribers.append(q)
        q.put(self.snapshot())
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._subscribers_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _maybe_broadcast(self):
        now = time.perf_counter()
        if now - self._last_broadcast < 1.0 / BROADCAST_HZ:
            return
        self._last_broadcast = now
        snap = self.snapshot()
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(snap)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(snap)
                except queue.Empty:
                    pass

    def snapshot(self) -> dict:
        with self._lock:
            tx, ty, tz, trot = self._target_pose()
            joints = (
                [list(p) for p in self.joint_positions]
                if self.joint_positions is not None else None
            )
            return {
                "target": {"x": tx, "y": ty, "z": tz, "rot_y": trot},
                "measured": {
                    "x": self.measured_x,
                    "y": self.measured_y,
                    "z": self.measured_z,
                    "rot_y": self.measured_rot_y,
                    "valid": self._has_telemetry,
                },
                "joints": joints,
                "joint_angles_deg": list(self.joint_angles_deg),
                "direct": self.direct_commands[:],
                "direct_mode": self.direct_mode,
                "ik_space": self.ik_control_space.value,
                "connected": self.udp_client.is_connected,
                "streaming": self.udp_client.is_streaming,
                "status": self.status_message,
                "recv": self.recv_label,
                "send_hz": self.send_hz_actual,
                "host": self.host,
                "port": self.port,
                "rate_hz": self.rate_hz,
                "step_pos": self.step_pos,
                "step_rot": self.step_rot,
            }

    def _build_command(self) -> ControlCommand:
        with self._lock:
            self.command_sequence += 1
            tx, ty, tz, trot = self._target_pose()
            cmd = ControlCommand(
                sequence=self.command_sequence,
                mode=ControlMode.DIRECT if self.direct_mode else ControlMode.IK,
                pose=PoseTarget(tx, ty, tz, trot),
                direct=DirectCommand(*self.direct_commands),
                reload_config=bool(self.reload_flag),
                pause=bool(self.pause_flag),
                resume=bool(self.resume_flag),
            )
            self.reload_flag = 0
            self.pause_flag = 0
            self.resume_flag = 0
        return cmd

    def _on_telemetry(self, telemetry: RobotTelemetry):
        mode_name = "DIRECT" if telemetry.mode == ControlMode.DIRECT else "IK"
        with self._lock:
            self.joint_positions = [list(p) for p in telemetry.joint_positions]
            self.joint_angles_deg = telemetry.joint_angles_deg
            self.measured_x = telemetry.measured_pose.x
            self.measured_y = telemetry.measured_pose.y
            self.measured_z = telemetry.measured_pose.z
            self.measured_rot_y = telemetry.measured_pose.rot_y_deg
            self._has_telemetry = True
            self.recv_label = mode_name
            if self._sync_target_from_telemetry and not self.direct_mode:
                if self._sync_target_from_measured():
                    self._sync_target_from_telemetry = False

    def _on_hz_update(self, hz: float):
        with self._lock:
            self.send_hz_actual = hz

    def connect(self, host: str, port: int, rate_hz: float) -> tuple[bool, str]:
        if not self.udp_client.connect(host, port, rate_hz):
            return False, "Failed to connect / handshake"
        with self._lock:
            self.host = host
            self.port = port
            self.rate_hz = rate_hz
            self.status_message = f"Connected to {host}:{port}"
        return True, "ok"

    def disconnect(self):
        self.stop_streaming()
        self.udp_client.disconnect()
        with self._lock:
            self._has_telemetry = False
            self.status_message = "Disconnected"

    def start_streaming(self) -> tuple[bool, str]:
        if not self.udp_client.is_connected:
            return False, "Not connected"
        with self._lock:
            self._sync_target_from_telemetry = not self._sync_target_from_measured()
            self.status_message = "Streaming"
        self.udp_client.start_streaming(
            get_command=self._build_command,
            on_telemetry=self._on_telemetry,
            on_hz_update=self._on_hz_update,
        )
        return True, "ok"

    def stop_streaming(self):
        if not self.udp_client.is_streaming:
            return
        self.udp_client.stop_streaming()
        with self._lock:
            self._sync_target_from_telemetry = False
            self._has_telemetry = False
            self.send_hz_actual = 0.0
            if self.direct_mode:
                self.direct_mode = False
                self.direct_commands = [0.0, 0.0, 0.0, 0.0]
            self.status_message = (
                "Connected (Idle)" if self.udp_client.is_connected else "Disconnected"
            )

    def set_keys(self, keys: List[str]):
        with self._lock:
            self.keys_down = set(k.lower() for k in keys)

    def toggle_ik_space(self):
        with self._lock:
            self.ik_control_space = (
                IKControlSpace.RADIAL
                if self.ik_control_space == IKControlSpace.CARTESIAN
                else IKControlSpace.CARTESIAN
            )

    def toggle_direct_mode(self):
        with self._lock:
            self.direct_mode = not self.direct_mode
            self.direct_commands = [0.0, 0.0, 0.0, 0.0]
            if self.direct_mode:
                self.status_message = "DIRECT MODE"
            else:
                self._sync_target_from_measured()
                if self.udp_client.is_streaming:
                    self.status_message = "Streaming"

    def reset_target(self):
        with self._lock:
            self.target_state.set_cartesian_pose(DEFAULT_X, DEFAULT_Y, DEFAULT_Z, 0.0)

    def trigger_reload(self):
        with self._lock:
            self.reload_flag = 1

    def trigger_pause(self):
        with self._lock:
            self.pause_flag = 1

    def trigger_resume(self):
        with self._lock:
            self._sync_target_from_measured()
            self.resume_flag = 1

    def set_step(self, pos: Optional[float], rot: Optional[float]):
        with self._lock:
            if pos is not None and pos > 0:
                self.step_pos = min(float(pos), 0.025)
            if rot is not None and rot > 0:
                self.step_rot = max(0.1, min(float(rot), 1.0))

    def shutdown(self):
        self._running = False
        self.disconnect()


def _json_response(handler: BaseHTTPRequestHandler, code: int, body: dict):
    payload = json.dumps(body).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(payload)


def _resolve_static(url_path: str) -> Optional[Path]:
    """Map URL path to a file on disk using STATIC_ROOTS (longest prefix wins)."""
    for prefix in sorted(STATIC_ROOTS, key=len, reverse=True):
        if url_path.startswith(prefix):
            root = STATIC_ROOTS[prefix]
            rel = url_path[len(prefix):].lstrip("/") or "index.html"
            candidate = (root / rel).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                return None
            if candidate.is_file():
                return candidate
    return None


_CTYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".urdf": "application/xml; charset=utf-8",
    ".xml": "application/xml; charset=utf-8",
    ".obj": "text/plain; charset=utf-8",
    ".mtl": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
}


_CACHEABLE_EXTS = {".obj", ".mtl", ".stl", ".dae", ".png", ".jpg", ".jpeg"}


def _serve_file(handler: BaseHTTPRequestHandler, path: Path):
    data = path.read_bytes()
    ctype = _CTYPES.get(path.suffix.lower(), "application/octet-stream")
    # Only cache bulky static assets (meshes/textures). HTML/JS/CSS/URDF
    # change during development — must not be cached or users see stale UI.
    cacheable = path.suffix.lower() in _CACHEABLE_EXTS
    cache_ctl = "public, max-age=3600" if cacheable else "no-store"
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", cache_ctl)
    handler.end_headers()
    handler.wfile.write(data)


def make_handler(app: ExcavatorApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/events":
                self._serve_events()
                return
            if path == "/snapshot":
                _json_response(self, HTTPStatus.OK, app.snapshot())
                return
            resolved = _resolve_static(path)
            if resolved is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                _serve_file(self, resolved)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            body = self._read_json()
            try:
                if path == "/input":
                    keys = body.get("keys", [])
                    if not isinstance(keys, list):
                        return _json_response(self, 400, {"error": "keys must be a list"})
                    app.set_keys([str(k) for k in keys])
                    return _json_response(self, 200, {"ok": True})

                if path == "/action":
                    return self._handle_action(body.get("action", ""), body)

                if path == "/config":
                    pos = body.get("step_pos")
                    rot = body.get("step_rot")
                    app.set_step(
                        float(pos) if pos is not None else None,
                        float(rot) if rot is not None else None,
                    )
                    return _json_response(self, 200, {"ok": True})
            except Exception as e:
                return _json_response(self, 500, {"error": str(e)})
            self.send_error(HTTPStatus.NOT_FOUND)

        def _handle_action(self, action: str, body: dict):
            if action == "connect":
                host = str(body.get("host", "")).strip()
                if not host:
                    return _json_response(self, 400, {
                        "ok": False, "msg": "Host is required",
                    })
                try:
                    port = int(body.get("port"))
                    rate = float(body.get("rate_hz"))
                except (TypeError, ValueError):
                    return _json_response(self, 400, {
                        "ok": False, "msg": "Port and rate must be numbers",
                    })
                ok, msg = app.connect(host, port, rate)
                return _json_response(self, 200, {"ok": ok, "msg": msg})
            if action == "disconnect":
                app.disconnect()
                return _json_response(self, 200, {"ok": True})
            if action == "start_stream":
                ok, msg = app.start_streaming()
                return _json_response(self, 200, {"ok": ok, "msg": msg})
            if action == "stop_stream":
                app.stop_streaming()
                return _json_response(self, 200, {"ok": True})
            if action == "toggle_ik_space":
                app.toggle_ik_space()
                return _json_response(self, 200, {"ok": True})
            if action == "toggle_direct":
                app.toggle_direct_mode()
                return _json_response(self, 200, {"ok": True})
            if action == "reset":
                app.reset_target()
                return _json_response(self, 200, {"ok": True})
            if action == "reload":
                app.trigger_reload()
                return _json_response(self, 200, {"ok": True})
            if action == "pause":
                app.trigger_pause()
                return _json_response(self, 200, {"ok": True})
            if action == "resume":
                app.trigger_resume()
                return _json_response(self, 200, {"ok": True})
            return _json_response(self, 400, {"error": f"unknown action: {action}"})

        def _serve_events(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            q = app.subscribe()
            try:
                last_heartbeat = time.perf_counter()
                while True:
                    try:
                        snap = q.get(timeout=1.0)
                        self.wfile.write(f"data: {json.dumps(snap)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        pass
                    if time.perf_counter() - last_heartbeat > 15.0:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_heartbeat = time.perf_counter()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                app.unsubscribe(q)

    return Handler


def main():
    app = ExcavatorApp()
    handler_cls = make_handler(app)
    server = ThreadingHTTPServer((WEB_HOST, WEB_PORT), handler_cls)
    print(f"[web] Excavator web GUI at http://{WEB_HOST}:{WEB_PORT}  (Ctrl-C to quit)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] Shutting down.")
    finally:
        app.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
