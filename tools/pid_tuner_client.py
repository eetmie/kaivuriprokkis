#!/usr/bin/env python3
"""Client GUI for tools/pid_tuner_robot.py."""

from __future__ import annotations

import sys
import time
import tkinter as tk
import argparse
from collections import deque
from pathlib import Path
from tkinter import ttk
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.udp_socket import UDPSocket


COMMAND_SIZE = 10
TELEMETRY_SIZE = 12
JOINT_NAMES = ["slew", "boom", "arm", "bucket"]
PID_KEYS = ["joint0", "joint1", "joint2", "joint3"]


def _load_initial_gains() -> list[tuple[float, float, float]]:
    path = ROOT_DIR / "configuration_files" / "control_config.yaml"
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        pid = cfg.get("pid", {}) if isinstance(cfg, dict) else {}
        gains = []
        for key in PID_KEYS:
            item = pid.get(key, {}) if isinstance(pid, dict) else {}
            gains.append((float(item.get("kp", 0.0)), float(item.get("ki", 0.0)), float(item.get("kd", 0.0))))
        return gains
    except Exception:
        return [(0.0, 0.0, 0.0) for _ in JOINT_NAMES]


class PIDTunerClient:
    def __init__(self, host: str = "192.168.0.132", port: int = 8090):
        self.sock: UDPSocket | None = None
        self.default_host = host
        self.default_port = port
        self.initial_gains = _load_initial_gains()

        self.root = tk.Tk()
        self.root.title("Excavator PID Tuner")

        self.selected_joint = tk.IntVar(value=0)
        self.target_deg = tk.DoubleVar(value=0.0)
        self.kp = tk.DoubleVar(value=self.initial_gains[0][0])
        self.ki = tk.DoubleVar(value=self.initial_gains[0][1])
        self.kd = tk.DoubleVar(value=self.initial_gains[0][2])
        self.max_output = tk.DoubleVar(value=1.0)
        self.enabled = tk.BooleanVar(value=False)
        self.pump_enabled = tk.BooleanVar(value=False)
        self.host_var = tk.StringVar(value=host)
        self.port_var = tk.StringVar(value=str(port))
        self.status = tk.StringVar(value="Disconnected")
        self.angle_text = tk.StringVar(value="slew -- | boom -- | arm -- | bucket --")
        self.gain_text = tk.StringVar(value="")

        self.reload_once = False
        self.reset_once = False
        self.telemetry: list[float] | None = None

        self.max_points = 600
        self.target_hist = deque(maxlen=self.max_points)
        self.measured_hist = deque(maxlen=self.max_points)
        self.output_hist = deque(maxlen=self.max_points)
        self.period_ms = 50

        self._build_ui()
        self._schedule_tick()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Host").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.host_var, width=16).grid(row=0, column=1, sticky="ew")
        ttk.Label(frame, text="Port").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Entry(frame, textvariable=self.port_var, width=8).grid(row=0, column=3, sticky="w")
        ttk.Button(frame, text="Connect", command=self._connect).grid(row=0, column=4, padx=(8, 0))
        ttk.Button(frame, text="Disconnect", command=self._disconnect).grid(row=0, column=5, padx=(4, 0))

        ttk.Label(frame, text="Joint").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.joint_combo = ttk.Combobox(frame, values=JOINT_NAMES, state="readonly", width=12)
        self.joint_combo.current(0)
        self.joint_combo.grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.joint_combo.bind("<<ComboboxSelected>>", lambda _e: self._set_joint(self.joint_combo.current()))

        ttk.Checkbutton(frame, text="Enable Output", variable=self.enabled).grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(frame, text="Pump", variable=self.pump_enabled).grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Button(frame, text="Sync Target", command=self._sync_target).grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Button(frame, text="Reset PID", command=self._reset_pid).grid(row=1, column=5, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="Target deg").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(frame, from_=-180.0, to=180.0, variable=self.target_deg, orient=tk.HORIZONTAL).grid(
            row=2, column=1, columnspan=4, sticky="ew", pady=(8, 0)
        )
        self.target_label = ttk.Label(frame, width=9)
        self.target_label.grid(row=2, column=5, sticky="w", pady=(8, 0))

        self._add_gain_row(frame, 3, "kp", self.kp, 0.0, 20.0)
        self._add_gain_row(frame, 4, "ki", self.ki, 0.0, 5.0)
        self._add_gain_row(frame, 5, "kd", self.kd, 0.0, 5.0)
        self._add_gain_row(frame, 6, "max output", self.max_output, 0.05, 1.0)

        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Load Config Gains", command=self._load_selected_config_gain).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Reload Servo Config", command=self._reload_config).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Zero Target", command=lambda: self.target_deg.set(0.0)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Snapshot", command=self._snapshot).pack(side=tk.LEFT)

        ttk.Label(frame, textvariable=self.angle_text, font=("Courier", 11)).grid(
            row=8, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )
        ttk.Label(frame, textvariable=self.gain_text, font=("Courier", 10)).grid(
            row=9, column=0, columnspan=6, sticky="w"
        )
        ttk.Label(frame, textvariable=self.status).grid(row=10, column=0, columnspan=6, sticky="w", pady=(4, 0))

        self.canvas = tk.Canvas(frame, width=1200, height=420, bg="white")
        self.canvas.grid(row=11, column=0, columnspan=6, sticky="nsew", pady=(8, 0))

        for col in (1, 2, 3, 4):
            frame.columnconfigure(col, weight=1)
        frame.rowconfigure(11, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _add_gain_row(self, parent, row: int, label: str, var: tk.DoubleVar, lo: float, hi: float):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Scale(parent, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL).grid(
            row=row, column=1, columnspan=4, sticky="ew", pady=(4, 0)
        )
        value_label = ttk.Label(parent, width=9)
        value_label.grid(row=row, column=5, sticky="w", pady=(4, 0))

        def update(*_args):
            value_label.config(text=f"{float(var.get()):.3f}")

        var.trace_add("write", update)
        update()

    def _schedule_tick(self):
        self.root.after(self.period_ms, self._tick)

    def _tick(self):
        self.target_label.config(text=f"{float(self.target_deg.get()):+.1f}")
        if self.sock is not None:
            self._send_command()
            latest = self.sock.get_latest()
            if latest and len(latest) >= TELEMETRY_SIZE:
                self.telemetry = [float(v) for v in latest]
                self._apply_telemetry(self.telemetry)
        self._redraw_plot()
        self._schedule_tick()

    def _send_command(self):
        if self.sock is None:
            return
        packet = [
            float(self.selected_joint.get()),
            float(self.target_deg.get()),
            float(self.kp.get()),
            float(self.ki.get()),
            float(self.kd.get()),
            1.0 if self.enabled.get() else 0.0,
            1.0 if self.pump_enabled.get() else 0.0,
            1.0 if self.reload_once else 0.0,
            1.0 if self.reset_once else 0.0,
            float(self.max_output.get()),
        ]
        try:
            self.sock.send(packet)
            self.reload_once = False
            self.reset_once = False
        except Exception as exc:
            self.status.set(f"Send error: {exc}")

    def _apply_telemetry(self, values: list[float]):
        angles = values[1:5]
        joint = self.selected_joint.get()
        target = values[5]
        measured = angles[joint]
        output = values[6]
        self.target_hist.append(target)
        self.measured_hist.append(measured)
        self.output_hist.append(output * 90.0)
        self.angle_text.set(
            " | ".join(f"{name} {angle:+7.2f}" for name, angle in zip(JOINT_NAMES, angles))
        )
        self.gain_text.set(
            f"robot: joint={JOINT_NAMES[int(values[0])]} target={target:+.2f} output={output:+.3f} "
            f"kp={values[7]:.3f} ki={values[8]:.3f} kd={values[9]:.3f} "
            f"enabled={bool(values[10] > 0.5)} pump={bool(values[11] > 0.5)}"
        )
        self.status.set("Connected")

    def _connect(self):
        self._disconnect()
        host = self.host_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            self.status.set("Invalid port")
            return
        try:
            sock = UDPSocket(local_id=11, max_age_seconds=0.5, data_format="f", nominal_rate_hz=1000.0 / self.period_ms)
            sock.setup(host, port, num_inputs=TELEMETRY_SIZE, num_outputs=COMMAND_SIZE, is_server=False, data_format="f")
            if not sock.handshake(timeout=10.0):
                self.status.set("Handshake failed")
                sock.close()
                return
            sock.start_receiving()
            self.sock = sock
            self.status.set(f"Connected to {host}:{port}")
        except Exception as exc:
            self.status.set(f"Connect failed: {exc}")

    def _disconnect(self):
        self.enabled.set(False)
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None

    def _set_joint(self, idx: int):
        idx = max(0, min(3, int(idx)))
        self.selected_joint.set(idx)
        self.enabled.set(False)
        self._load_selected_config_gain()
        self.target_hist.clear()
        self.measured_hist.clear()
        self.output_hist.clear()
        self._sync_target()

    def _load_selected_config_gain(self):
        kp, ki, kd = self.initial_gains[self.selected_joint.get()]
        self.kp.set(kp)
        self.ki.set(ki)
        self.kd.set(kd)
        self._reset_pid()

    def _sync_target(self):
        if self.telemetry is None:
            return
        joint = self.selected_joint.get()
        self.target_deg.set(float(self.telemetry[1 + joint]))
        self._reset_pid()

    def _reset_pid(self):
        self.reset_once = True

    def _reload_config(self):
        self.reload_once = True


    def _snapshot(self):
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:
            self.status.set(f"matplotlib required for snapshot: {exc}")
            return
        target = list(self.target_hist)
        measured = list(self.measured_hist)
        output = list(self.output_hist)
        if not measured:
            self.status.set("No data to snapshot")
            return
        dt = self.period_ms / 1000.0
        x = [-(len(measured) - 1 - i) * dt for i in range(len(measured))]
        fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
        ax.plot(x, target, label="target deg", color="#1f77b4")
        ax.plot(x, measured, label="measured deg", color="#d62728")
        ax.plot(x, output, label="output x90", color="#2ca02c", alpha=0.6)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("deg")
        ax.set_title(
            f"{JOINT_NAMES[self.selected_joint.get()]} kp={self.kp.get():.3f} "
            f"ki={self.ki.get():.3f} kd={self.kd.get():.3f}"
        )
        path = ROOT_DIR / f"pid_tuner_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.status.set(f"Saved {path.name}")

    def _redraw_plot(self):
        self.canvas.delete("all")
        w = int(self.canvas["width"])
        h = int(self.canvas["height"])
        left = 45
        top = 10
        bottom = h - 25
        right = w - 10
        self.canvas.create_line(left, bottom, right, bottom, fill="#888")
        self.canvas.create_line(left, top, left, bottom, fill="#888")
        ymin, ymax = -90.0, 90.0

        def points(series):
            values = list(series)
            if len(values) < 2:
                return []
            out = []
            for i, value in enumerate(values):
                x = left + (i / (len(values) - 1)) * (right - left)
                y_norm = (float(value) - ymin) / (ymax - ymin)
                y = bottom - y_norm * (bottom - top)
                out.extend([x, y])
            return out

        for series, color in ((self.target_hist, "#1f77b4"), (self.measured_hist, "#d62728"), (self.output_hist, "#2ca02c")):
            pts = points(series)
            if len(pts) >= 4:
                self.canvas.create_line(*pts, fill=color, width=2)
        self.canvas.create_text(left + 4, top, anchor="nw", text="+90 deg", fill="#444")
        self.canvas.create_text(left + 4, bottom, anchor="sw", text="-90 deg", fill="#444")
        self.canvas.create_text(right - 250, top, anchor="nw", text="blue=target red=measured green=output x90", fill="#444")

    def _on_close(self):
        self._disconnect()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client GUI for tools/pid_tuner_robot.py")
    parser.add_argument("host", nargs="?", default="192.168.0.132")
    parser.add_argument("port", nargs="?", type=int, default=8090)
    args = parser.parse_args()
    PIDTunerClient(args.host, args.port).run()
