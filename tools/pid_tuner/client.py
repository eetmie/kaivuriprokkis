#!/usr/bin/env python3
"""PID tuner — client GUI.

Two-tab window connecting to ``tools/pid_tuner/robot.py`` running on the robot:

* **Per-Joint tab** (port 8090): single-joint angle tuning.
* **EE Tracking tab** (port 8091): EE-plane auto-tuner (boom + arm + bucket).

Typical workflow:

1. Per-Joint tab — rough-tune each joint PID individually.
2. EE Tracking tab — auto-tune all three joints together against cartesian X
   tracking error.

Run with no args to connect to the rpi default host::

    python -m tools.pid_tuner.client

or::

    python tools/pid_tuner/client.py
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk
from typing import Deque, List, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.udp_socket import UDPSocket

from tools.pid_tuner.common import (
    CMD_COST_W_MAX,
    CMD_COST_W_RMSE,
    CMD_FLAGS,
    CMD_JOINT_MASK,
    CMD_KD_FWD,
    CMD_KD_REV,
    CMD_KI_FWD,
    CMD_KI_REV,
    CMD_KP_FWD,
    CMD_KP_REV,
    CMD_NUM_RUNS,
    CMD_SPEED,
    CMD_SPEED_MAX,
    CMD_SPEED_MIN,
    CMD_STROKES_PER_RUN,
    CMD_TUNED_JOINT,
    CMD_X_MAX,
    CMD_X_MIN,
    CMD_Y,
    CMD_Z,
    CMD_Z_MAX,
    CMD_Z_MIN,
    DEFAULT_COST_WEIGHTS,
    DEFAULT_HOST,
    DEFAULT_NUM_RUNS,
    DEFAULT_SPEED,
    DEFAULT_SPEED_MAX,
    DEFAULT_SPEED_MIN,
    DEFAULT_STROKES_PER_RUN,
    DEFAULT_X_MAX,
    DEFAULT_X_MIN,
    DEFAULT_Y,
    DEFAULT_Z,
    DEFAULT_Z_MAX,
    DEFAULT_Z_MIN,
    EE_COMMAND_SIZE,
    EE_PORT,
    EE_TELEMETRY_SIZE,
    EE_TUNABLE_JOINTS,
    EE_TUNABLE_NAMES,
    EECommandFlag,
    JOINT_COMMAND_SIZE,
    JOINT_NAMES,
    JOINT_PORT,
    JOINT_TELEMETRY_SIZE,
    TLM_BEST_COST,
    TLM_BEST_COST_FWD,
    TLM_BEST_COST_REV,
    TLM_CMD_X,
    TLM_CMD_Z,
    TLM_CUR_SPEED,
    TLM_CUR_X_MAX,
    TLM_CUR_X_MIN,
    TLM_CUR_Z,
    TLM_DIR_PID,
    TLM_EE_X,
    TLM_EE_Z,
    TLM_ERR_NOW,
    TLM_HW_READY,
    TLM_ITER,
    TLM_KD_FWD,
    TLM_KD_REV,
    TLM_KI_FWD,
    TLM_KI_REV,
    TLM_KP_FWD,
    TLM_KP_REV,
    TLM_LAST_COST,
    TLM_LAST_COST_FWD,
    TLM_LAST_COST_REV,
    TLM_MAXERR_LAST,
    TLM_MAXERR_LAST_FWD,
    TLM_MAXERR_LAST_REV,
    TLM_MEANERR_LAST,
    TLM_MEANERR_LAST_FWD,
    TLM_MEANERR_LAST_REV,
    TLM_NUM_RUNS,
    TLM_RAMP_DIR,
    TLM_RMSE_LAST,
    TLM_RMSE_LAST_FWD,
    TLM_RMSE_LAST_REV,
    TLM_RUN_INDEX,
    TLM_STATE,
    TLM_STROKE_INDEX,
    TLM_TOTAL_STROKES,
    TLM_TUNED_JOINT,
    ee_state_label,
)

PID_KEYS = ["joint0", "joint1", "joint2", "joint3"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_initial_gains() -> list:
    path = ROOT_DIR / "configuration_files" / "profiles" / "rpi" / "control_config.yaml"
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        pid = cfg.get("pid", {}) if isinstance(cfg, dict) else {}
        gains = []
        for key in PID_KEYS:
            item = pid.get(key, {}) if isinstance(pid, dict) else {}
            gains.append((
                float(item.get("kp", 0.0)),
                float(item.get("ki", 0.0)),
                float(item.get("kd", 0.0)),
            ))
        return gains
    except Exception:
        return [(0.0, 0.0, 0.0) for _ in JOINT_NAMES]


# ---------------------------------------------------------------------------
# Per-Joint tab
# ---------------------------------------------------------------------------


class _JointTab:
    PERIOD_MS = 50

    def __init__(self, parent: ttk.Notebook, host: str, port: int) -> None:
        self.frame = ttk.Frame(parent)
        self.host_default = host
        self.port_default = port
        self.initial_gains = _load_initial_gains()
        self.sock: Optional[UDPSocket] = None

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
        self.telemetry: Optional[List[float]] = None

        self.max_points = 600
        self.target_hist: Deque[float] = deque(maxlen=self.max_points)
        self.measured_hist: Deque[float] = deque(maxlen=self.max_points)
        self.output_hist: Deque[float] = deque(maxlen=self.max_points)

        self._build_ui()
        self.frame.after(self.PERIOD_MS, self._tick)

    def _build_ui(self) -> None:
        f = ttk.Frame(self.frame, padding=10)
        f.grid(row=0, column=0, sticky="nsew")
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        ttk.Label(f, text="Host").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.host_var, width=16).grid(row=0, column=1, sticky="ew")
        ttk.Label(f, text="Port").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Entry(f, textvariable=self.port_var, width=8).grid(row=0, column=3, sticky="w")
        ttk.Button(f, text="Connect", command=self._connect).grid(row=0, column=4, padx=(8, 0))
        ttk.Button(f, text="Disconnect", command=self._disconnect).grid(row=0, column=5, padx=(4, 0))

        ttk.Label(f, text="Joint").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.joint_combo = ttk.Combobox(f, values=JOINT_NAMES, state="readonly", width=12)
        self.joint_combo.current(0)
        self.joint_combo.grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.joint_combo.bind("<<ComboboxSelected>>",
                              lambda _e: self._set_joint(self.joint_combo.current()))

        ttk.Checkbutton(f, text="Enable Output", variable=self.enabled).grid(
            row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(f, text="Pump", variable=self.pump_enabled).grid(
            row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Button(f, text="Sync Target", command=self._sync_target).grid(
            row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Button(f, text="Reset PID", command=self._reset_pid).grid(
            row=1, column=5, sticky="w", pady=(8, 0))

        ttk.Label(f, text="Target deg").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(f, from_=-180.0, to=180.0, variable=self.target_deg,
                  orient=tk.HORIZONTAL).grid(row=2, column=1, columnspan=4, sticky="ew", pady=(8, 0))
        self.target_label = ttk.Label(f, width=9)
        self.target_label.grid(row=2, column=5, sticky="w", pady=(8, 0))

        self._add_gain_row(f, 3, "kp", self.kp, 0.0, 20.0)
        self._add_gain_row(f, 4, "ki", self.ki, 0.0, 5.0)
        self._add_gain_row(f, 5, "kd", self.kd, 0.0, 5.0)
        self._add_gain_row(f, 6, "max output", self.max_output, 0.05, 1.0)

        btns = ttk.Frame(f)
        btns.grid(row=7, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Button(btns, text="Load Config Gains",
                   command=self._load_selected_config_gain).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Reload Servo Config",
                   command=self._reload_config).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Zero Target",
                   command=lambda: self.target_deg.set(0.0)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Snapshot", command=self._snapshot).pack(side=tk.LEFT)

        ttk.Label(f, textvariable=self.angle_text, font=("Courier", 11)).grid(
            row=8, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Label(f, textvariable=self.gain_text, font=("Courier", 10)).grid(
            row=9, column=0, columnspan=6, sticky="w")
        ttk.Label(f, textvariable=self.status).grid(
            row=10, column=0, columnspan=6, sticky="w", pady=(4, 0))

        self.canvas = tk.Canvas(f, width=1100, height=400, bg="white")
        self.canvas.grid(row=11, column=0, columnspan=6, sticky="nsew", pady=(8, 0))

        for col in (1, 2, 3, 4):
            f.columnconfigure(col, weight=1)
        f.rowconfigure(11, weight=1)

    def _add_gain_row(self, parent, row: int, label: str,
                      var: tk.DoubleVar, lo: float, hi: float) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Scale(parent, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL).grid(
            row=row, column=1, columnspan=4, sticky="ew", pady=(4, 0))
        lbl = ttk.Label(parent, width=9)
        lbl.grid(row=row, column=5, sticky="w", pady=(4, 0))

        def update(*_):
            lbl.config(text=f"{float(var.get()):.3f}")

        var.trace_add("write", update)
        update()

    def _tick(self) -> None:
        self.target_label.config(text=f"{float(self.target_deg.get()):+.1f}")
        if self.sock is not None:
            self._send_command()
            latest = self.sock.get_latest()
            if latest and len(latest) >= JOINT_TELEMETRY_SIZE:
                self.telemetry = [float(v) for v in latest]
                self._apply_telemetry(self.telemetry)
        self._redraw_plot()
        self.frame.after(self.PERIOD_MS, self._tick)

    def _send_command(self) -> None:
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

    def _apply_telemetry(self, values: List[float]) -> None:
        angles = values[1:5]
        joint = self.selected_joint.get()
        target = values[5]
        measured = angles[joint]
        output = values[6]
        self.target_hist.append(target)
        self.measured_hist.append(measured)
        self.output_hist.append(output * 90.0)
        self.angle_text.set(
            " | ".join(f"{name} {angle:+7.2f}"
                       for name, angle in zip(JOINT_NAMES, angles))
        )
        self.gain_text.set(
            f"robot: joint={JOINT_NAMES[int(values[0])]} target={target:+.2f} "
            f"output={output:+.3f} "
            f"kp={values[7]:.3f} ki={values[8]:.3f} kd={values[9]:.3f} "
            f"enabled={bool(values[10] > 0.5)} pump={bool(values[11] > 0.5)}"
        )
        self.status.set("Connected")

    def _connect(self) -> None:
        self._disconnect()
        host = self.host_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            self.status.set("Invalid port")
            return
        try:
            sock = UDPSocket(local_id=11, max_age_seconds=0.5,
                             nominal_rate_hz=1000.0 / self.PERIOD_MS)
            sock.setup(host, port,
                       inputs=f"{JOINT_TELEMETRY_SIZE}f",
                       outputs=f"{JOINT_COMMAND_SIZE}f",
                       is_server=False)
            if not sock.handshake(timeout=10.0):
                self.status.set("Handshake failed")
                sock.close()
                return
            sock.start_receiving()
            self.sock = sock
            self.status.set(f"Connected to {host}:{port}")
        except Exception as exc:
            self.status.set(f"Connect failed: {exc}")

    def _disconnect(self) -> None:
        self.enabled.set(False)
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None

    def cleanup(self) -> None:
        self._disconnect()

    def _set_joint(self, idx: int) -> None:
        idx = max(0, min(3, int(idx)))
        self.selected_joint.set(idx)
        self.enabled.set(False)
        self._load_selected_config_gain()
        self.target_hist.clear()
        self.measured_hist.clear()
        self.output_hist.clear()
        self._sync_target()

    def _load_selected_config_gain(self) -> None:
        kp, ki, kd = self.initial_gains[self.selected_joint.get()]
        self.kp.set(kp)
        self.ki.set(ki)
        self.kd.set(kd)
        self._reset_pid()

    def _sync_target(self) -> None:
        if self.telemetry is None:
            return
        joint = self.selected_joint.get()
        self.target_deg.set(float(self.telemetry[1 + joint]))
        self._reset_pid()

    def _reset_pid(self) -> None:
        self.reset_once = True

    def _reload_config(self) -> None:
        self.reload_once = True

    def _snapshot(self) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:
            self.status.set(f"matplotlib required: {exc}")
            return
        target = list(self.target_hist)
        measured = list(self.measured_hist)
        output = list(self.output_hist)
        if not measured:
            self.status.set("No data to snapshot")
            return
        dt = self.PERIOD_MS / 1000.0
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
            f"{JOINT_NAMES[self.selected_joint.get()]} "
            f"kp={self.kp.get():.3f} ki={self.ki.get():.3f} kd={self.kd.get():.3f}"
        )
        path = ROOT_DIR / f"pid_tuner_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.status.set(f"Saved {path.name}")

    def _redraw_plot(self) -> None:
        c = self.canvas
        c.delete("all")
        w = int(c["width"]); h = int(c["height"])
        left, top, bottom, right = 45, 10, h - 25, w - 10
        c.create_line(left, bottom, right, bottom, fill="#888")
        c.create_line(left, top, left, bottom, fill="#888")
        ymin, ymax = -90.0, 90.0

        def points(series):
            vals = list(series)
            if len(vals) < 2:
                return []
            out = []
            for i, v in enumerate(vals):
                xp = left + (i / (len(vals) - 1)) * (right - left)
                yp = bottom - ((float(v) - ymin) / (ymax - ymin)) * (bottom - top)
                out.extend([xp, yp])
            return out

        for series, color in (
            (self.target_hist, "#1f77b4"),
            (self.measured_hist, "#d62728"),
            (self.output_hist, "#2ca02c"),
        ):
            pts = points(series)
            if len(pts) >= 4:
                c.create_line(*pts, fill=color, width=2)

        c.create_text(left + 4, top, anchor="nw", text="+90 deg", fill="#444")
        c.create_text(left + 4, bottom, anchor="sw", text="-90 deg", fill="#444")
        c.create_text(right - 280, top, anchor="nw",
                      text="blue=target  red=measured  green=output×90", fill="#444")


# ---------------------------------------------------------------------------
# EE Tracking tab
# ---------------------------------------------------------------------------


class _EETab:
    PERIOD_MS = 50
    HISTORY_POINTS = 1200

    def __init__(self, parent: ttk.Notebook, host: str, port: int) -> None:
        self.frame = ttk.Frame(parent)
        self.sock: Optional[UDPSocket] = None
        self.host_default = host
        self.port_default = port

        self.host_var = tk.StringVar(value=host)
        self.port_var = tk.StringVar(value=str(port))
        self.status_var = tk.StringVar(value="Disconnected")

        self.x_min_var = tk.DoubleVar(value=DEFAULT_X_MIN)
        self.x_max_var = tk.DoubleVar(value=DEFAULT_X_MAX)
        self.y_var = tk.DoubleVar(value=DEFAULT_Y)
        self.z_var = tk.DoubleVar(value=DEFAULT_Z)
        self.speed_var = tk.DoubleVar(value=DEFAULT_SPEED)
        self.strokes_var = tk.IntVar(value=DEFAULT_STROKES_PER_RUN)
        self.num_runs_var = tk.IntVar(value=DEFAULT_NUM_RUNS)
        self.z_min_var = tk.DoubleVar(value=DEFAULT_Z_MIN)
        self.z_max_var = tk.DoubleVar(value=DEFAULT_Z_MAX)
        self.speed_min_var = tk.DoubleVar(value=DEFAULT_SPEED_MIN)
        self.speed_max_var = tk.DoubleVar(value=DEFAULT_SPEED_MAX)

        self.joint_var = tk.IntVar(value=1)
        self.dir_pid_var = tk.BooleanVar(value=True)
        self.auto_tune_var = tk.BooleanVar(value=False)

        self.kp_fwd_var = tk.DoubleVar(value=0.0)
        self.ki_fwd_var = tk.DoubleVar(value=0.0)
        self.kd_fwd_var = tk.DoubleVar(value=0.0)
        self.kp_rev_var = tk.DoubleVar(value=0.0)
        self.ki_rev_var = tk.DoubleVar(value=0.0)
        self.kd_rev_var = tk.DoubleVar(value=0.0)

        self.cost_w_rmse_var = tk.DoubleVar(value=DEFAULT_COST_WEIGHTS[0])
        self.cost_w_max_var = tk.DoubleVar(value=DEFAULT_COST_WEIGHTS[1])

        self._pending_flags = 0

        self.t_hist: Deque[float] = deque(maxlen=self.HISTORY_POINTS)
        self.cmd_x_hist: Deque[float] = deque(maxlen=self.HISTORY_POINTS)
        self.ee_x_hist: Deque[float] = deque(maxlen=self.HISTORY_POINTS)
        self.err_hist: Deque[float] = deque(maxlen=self.HISTORY_POINTS)
        self.xz_cmd_hist: Deque[tuple] = deque(maxlen=self.HISTORY_POINTS)
        self.xz_ee_hist: Deque[tuple] = deque(maxlen=self.HISTORY_POINTS)
        self.last_telemetry: Optional[List[float]] = None
        self.connect_t0 = time.perf_counter()

        self._build_ui()
        self.frame.after(self.PERIOD_MS, self._tick)

    def _build_ui(self) -> None:
        root = ttk.Frame(self.frame, padding=8)
        root.grid(row=0, column=0, sticky="nsew")
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        self._build_connection_row(root)
        self._build_params_panel(root)
        self._build_plot_area(root)
        self._build_status_row(root)

    def _build_connection_row(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(row, text="Host").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.host_var, width=16).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="Port").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.port_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Connect", command=self._connect).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Disconnect", command=self._disconnect).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.status_var, foreground="#444").pack(side=tk.LEFT, padx=8)

    def _build_params_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent)
        panel.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        for col in range(8):
            panel.columnconfigure(col, weight=1)

        sweep = ttk.LabelFrame(panel, text="Sweep", padding=6)
        sweep.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=(0, 4))
        self._lentry(sweep, 0, "x_min (m)", self.x_min_var, "A point (near)")
        self._lentry(sweep, 1, "x_max (m)", self.x_max_var, "B point (far)")
        self._lentry(sweep, 2, "y      (m)", self.y_var)
        self._lentry(sweep, 3, "z      (m)", self.z_var, "EE height for manual runs")
        self._lentry(sweep, 4, "speed (m/s)", self.speed_var,
                     "EE speed for MANUAL runs (0.020..0.070)")
        self._lentry(sweep, 5, "strokes/run", self.strokes_var)

        atune = ttk.LabelFrame(panel, text="Auto-tune", padding=6)
        atune.grid(row=0, column=2, columnspan=2, sticky="nsew", padx=(0, 4))
        self._lentry(atune, 0, "runs", self.num_runs_var, "Max auto-tune iterations")
        self._lentry(atune, 1, "z_min (m)", self.z_min_var, "Z sweep lower bound")
        self._lentry(atune, 2, "z_max (m)", self.z_max_var, "Z sweep upper bound")
        self._lentry(atune, 3, "v_min (m/s)", self.speed_min_var, "Speed lower bound")
        self._lentry(atune, 4, "v_max (m/s)", self.speed_max_var, "Speed upper bound")

        sel = ttk.LabelFrame(panel, text="Joint / PID", padding=6)
        sel.grid(row=0, column=4, columnspan=2, sticky="nsew", padx=(0, 4))
        ttk.Label(sel, text="Display joint").grid(row=0, column=0, sticky="w")
        joint_combo = ttk.Combobox(sel, values=list(EE_TUNABLE_NAMES),
                                   state="readonly", width=10)
        try:
            joint_combo.current(list(EE_TUNABLE_JOINTS).index(self.joint_var.get()))
        except ValueError:
            joint_combo.current(0)
            self.joint_var.set(EE_TUNABLE_JOINTS[0])
        joint_combo.grid(row=0, column=1, sticky="w")
        joint_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.joint_var.set(EE_TUNABLE_JOINTS[joint_combo.current()]),
        )
        ttk.Checkbutton(sel, text="Directional PID", variable=self.dir_pid_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(sel, text="Auto-tune", variable=self.auto_tune_var).grid(
            row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(sel, text="Reset PIDs", command=self._reset_pids).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(sel, text="Reload servo config", command=self._reload_config).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        gains = ttk.LabelFrame(panel, text="Manual gains (per direction)", padding=6)
        gains.grid(row=0, column=6, columnspan=2, sticky="nsew")
        ttk.Label(gains, text="").grid(row=0, column=0)
        ttk.Label(gains, text="fwd").grid(row=0, column=1)
        ttk.Label(gains, text="rev").grid(row=0, column=2)
        for r, (lbl, fv, rv) in enumerate((
            ("kp", self.kp_fwd_var, self.kp_rev_var),
            ("ki", self.ki_fwd_var, self.ki_rev_var),
            ("kd", self.kd_fwd_var, self.kd_rev_var),
        ), start=1):
            ttk.Label(gains, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(gains, textvariable=fv, width=8).grid(row=r, column=1, padx=2)
            ttk.Entry(gains, textvariable=rv, width=8).grid(row=r, column=2, padx=2)
        ttk.Button(gains, text="Apply gains", command=self._apply_gains).grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(gains, text="Pull gains from host", command=self._pull_gains).grid(
            row=5, column=0, columnspan=3, sticky="ew")

        runrow = ttk.Frame(parent)
        runrow.grid(row=2, column=0, sticky="ew", pady=(2, 4))
        ttk.Button(runrow, text="START", command=self._start).pack(side=tk.LEFT)
        ttk.Button(runrow, text="STOP", command=self._stop).pack(side=tk.LEFT, padx=4)
        ttk.Label(runrow, text="cost weights — rmse:").pack(side=tk.LEFT, padx=(16, 2))
        ttk.Entry(runrow, textvariable=self.cost_w_rmse_var, width=6).pack(side=tk.LEFT)
        ttk.Label(runrow, text="max:").pack(side=tk.LEFT, padx=(4, 2))
        ttk.Entry(runrow, textvariable=self.cost_w_max_var, width=6).pack(side=tk.LEFT)
        ttk.Button(runrow, text="Save snapshot", command=self._snapshot).pack(side=tk.LEFT, padx=12)

    def _lentry(self, parent, row: int, label: str,
                var: tk.Variable, tooltip: Optional[str] = None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        e = ttk.Entry(parent, textvariable=var, width=10)
        e.grid(row=row, column=1, sticky="w", padx=2)
        if tooltip:
            e.bind("<Enter>", lambda _e, t=tooltip: self.status_var.set(t))

    def _build_plot_area(self, parent: ttk.Frame) -> None:
        plots = ttk.Frame(parent)
        plots.grid(row=3, column=0, sticky="nsew")
        plots.columnconfigure(0, weight=2)
        plots.columnconfigure(1, weight=1)
        plots.rowconfigure(0, weight=1)

        self.canvas_time = tk.Canvas(plots, width=860, height=360, bg="white",
                                     highlightthickness=1, highlightbackground="#bbb")
        self.canvas_time.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        self.canvas_xz = tk.Canvas(plots, width=360, height=360, bg="white",
                                   highlightthickness=1, highlightbackground="#bbb")
        self.canvas_xz.grid(row=0, column=1, sticky="nsew")

    def _build_status_row(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        self.metrics_var = tk.StringVar(value="no telemetry yet")
        self.gains_var = tk.StringVar(value="")
        self.split_var = tk.StringVar(value="OUT/IN breakdown will appear after first scored run")
        ttk.Label(row, textvariable=self.metrics_var,
                  font=("Courier", 10)).pack(side=tk.TOP, anchor="w")
        ttk.Label(row, textvariable=self.gains_var,
                  font=("Courier", 10)).pack(side=tk.TOP, anchor="w")
        ttk.Label(row, textvariable=self.split_var, font=("Courier", 10),
                  foreground="#0a4").pack(side=tk.TOP, anchor="w")

    def _connect(self) -> None:
        self._disconnect()
        host = self.host_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            self.status_var.set("Invalid port")
            return
        try:
            sock = UDPSocket(local_id=13, max_age_seconds=0.5,
                             nominal_rate_hz=1000.0 / self.PERIOD_MS)
            sock.setup(host, port,
                       inputs=f"{EE_TELEMETRY_SIZE}f",
                       outputs=f"{EE_COMMAND_SIZE}f",
                       is_server=False)
            if not sock.handshake(timeout=10.0):
                self.status_var.set("Handshake failed")
                sock.close()
                return
            sock.start_receiving()
            self.sock = sock
            self.connect_t0 = time.perf_counter()
            self.status_var.set(f"Connected to {host}:{port}")
        except Exception as exc:
            self.status_var.set(f"Connect failed: {exc}")

    def _disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def cleanup(self) -> None:
        try:
            self._stop()
            if self.sock is not None:
                self.sock.send(self._build_command())
        except Exception:
            pass
        self._disconnect()

    def _start(self) -> None:
        flag = int(EECommandFlag.START)
        if self.auto_tune_var.get():
            flag |= int(EECommandFlag.AUTO_TUNE)
        if self.dir_pid_var.get():
            flag |= int(EECommandFlag.DIR_PID)
        for h in (self.t_hist, self.cmd_x_hist, self.ee_x_hist,
                  self.err_hist, self.xz_cmd_hist, self.xz_ee_hist):
            h.clear()
        self._pending_flags |= flag

    def _stop(self) -> None:
        self._pending_flags |= int(EECommandFlag.STOP)

    def _reset_pids(self) -> None:
        self._pending_flags |= int(EECommandFlag.RESET_PIDS)

    def _reload_config(self) -> None:
        self._pending_flags |= int(EECommandFlag.RELOAD_CONFIG)

    def _apply_gains(self) -> None:
        self._pending_flags |= int(EECommandFlag.APPLY_GAINS)
        if self.dir_pid_var.get():
            self._pending_flags |= int(EECommandFlag.DIR_PID)

    def _pull_gains(self) -> None:
        if self.last_telemetry is None:
            self.status_var.set("No telemetry yet")
            return
        t = self.last_telemetry
        self.kp_fwd_var.set(float(t[TLM_KP_FWD]))
        self.ki_fwd_var.set(float(t[TLM_KI_FWD]))
        self.kd_fwd_var.set(float(t[TLM_KD_FWD]))
        self.kp_rev_var.set(float(t[TLM_KP_REV]))
        self.ki_rev_var.set(float(t[TLM_KI_REV]))
        self.kd_rev_var.set(float(t[TLM_KD_REV]))

    def _snapshot(self) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:
            self.status_var.set(f"matplotlib required: {exc}")
            return
        if not self.cmd_x_hist:
            self.status_var.set("Nothing to snapshot")
            return
        t = list(self.t_hist)
        cmd = list(self.cmd_x_hist)
        ee = list(self.ee_x_hist)
        err = list(self.err_hist)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=100, sharex=True)
        ax1.plot(t, cmd, label="cmd x (m)", color="#1f77b4")
        ax1.plot(t, ee, label="measured x (m)", color="#d62728")
        ax1.legend(loc="upper right")
        ax1.set_ylabel("x (m)")
        ax1.grid(True, alpha=0.3)
        ax2.plot(t, err, color="#2ca02c")
        ax2.set_ylabel("|err| (mm)")
        ax2.set_xlabel("time (s)")
        ax2.grid(True, alpha=0.3)
        title_bits = [f"joint={JOINT_NAMES[self.joint_var.get()]}"]
        if self.last_telemetry is not None:
            t_ = self.last_telemetry
            title_bits.append(
                f"fwd kp/ki/kd={t_[TLM_KP_FWD]:.2f}/{t_[TLM_KI_FWD]:.2f}/{t_[TLM_KD_FWD]:.2f}"
            )
            title_bits.append(
                f"rev kp/ki/kd={t_[TLM_KP_REV]:.2f}/{t_[TLM_KI_REV]:.2f}/{t_[TLM_KD_REV]:.2f}"
            )
        fig.suptitle("  ".join(title_bits))
        path = ROOT_DIR / f"pid_tuner_ee_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.status_var.set(f"Saved {path.name}")

    def _build_command(self) -> List[float]:
        cmd = [0.0] * EE_COMMAND_SIZE
        cmd[CMD_FLAGS] = float(self._pending_flags)
        cmd[CMD_X_MIN] = float(self.x_min_var.get())
        cmd[CMD_X_MAX] = float(self.x_max_var.get())
        cmd[CMD_Z] = float(self.z_var.get())
        cmd[CMD_Y] = float(self.y_var.get())
        cmd[CMD_SPEED] = float(self.speed_var.get())
        cmd[CMD_STROKES_PER_RUN] = float(self.strokes_var.get())
        cmd[CMD_NUM_RUNS] = float(self.num_runs_var.get())
        cmd[CMD_TUNED_JOINT] = float(self.joint_var.get())
        cmd[CMD_JOINT_MASK] = float(1 << int(self.joint_var.get()))
        cmd[CMD_KP_FWD] = float(self.kp_fwd_var.get())
        cmd[CMD_KI_FWD] = float(self.ki_fwd_var.get())
        cmd[CMD_KD_FWD] = float(self.kd_fwd_var.get())
        cmd[CMD_KP_REV] = float(self.kp_rev_var.get())
        cmd[CMD_KI_REV] = float(self.ki_rev_var.get())
        cmd[CMD_KD_REV] = float(self.kd_rev_var.get())
        cmd[CMD_COST_W_RMSE] = float(self.cost_w_rmse_var.get())
        cmd[CMD_COST_W_MAX] = float(self.cost_w_max_var.get())
        cmd[CMD_Z_MIN] = float(self.z_min_var.get())
        cmd[CMD_Z_MAX] = float(self.z_max_var.get())
        cmd[CMD_SPEED_MIN] = float(self.speed_min_var.get())
        cmd[CMD_SPEED_MAX] = float(self.speed_max_var.get())
        self._pending_flags = 0
        return cmd

    def _tick(self) -> None:
        try:
            if self.sock is not None:
                try:
                    self.sock.send(self._build_command())
                except Exception as exc:
                    self.status_var.set(f"Send error: {exc}")
                pkt = self.sock.get_latest()
                if pkt is not None and len(pkt) >= EE_TELEMETRY_SIZE:
                    self._absorb_telemetry([float(v) for v in pkt[:EE_TELEMETRY_SIZE]])
            self._redraw_time_plot()
            self._redraw_xz_plot()
        finally:
            self.frame.after(self.PERIOD_MS, self._tick)

    def _absorb_telemetry(self, t: List[float]) -> None:
        self.last_telemetry = t
        now = time.perf_counter() - self.connect_t0
        self.t_hist.append(now)
        cmd_x = t[TLM_CMD_X]
        ee_x = t[TLM_EE_X]
        self.cmd_x_hist.append(cmd_x)
        self.ee_x_hist.append(ee_x)
        self.err_hist.append(t[TLM_ERR_NOW])
        self.xz_cmd_hist.append((cmd_x, t[TLM_CMD_Z]))
        self.xz_ee_hist.append((ee_x, t[TLM_EE_Z]))

        state_txt = ee_state_label(t[TLM_STATE])
        joint_txt = JOINT_NAMES[int(t[TLM_TUNED_JOINT]) & 0x3]
        hw = "ready" if t[TLM_HW_READY] > 0.5 else "not-ready"
        dirp = "dir-pid" if t[TLM_DIR_PID] > 0.5 else "single-pid"
        self.metrics_var.set(
            f"state={state_txt:<11} hw={hw:<9} joint={joint_txt:<6} {dirp:<10} "
            f"run={int(t[TLM_RUN_INDEX]):3d}/{int(t[TLM_NUM_RUNS]):3d}  "
            f"stroke={int(t[TLM_STROKE_INDEX]):2d}/{int(t[TLM_TOTAL_STROKES]):2d}  "
            f"speed={t[TLM_CUR_SPEED]*1000:5.1f} mm/s  z={t[TLM_CUR_Z]:.3f} m  "
            f"x=[{t[TLM_CUR_X_MIN]:.3f}..{t[TLM_CUR_X_MAX]:.3f}]"
        )
        self.gains_var.set(
            f"err_now={t[TLM_ERR_NOW]:6.2f}mm  rmse={t[TLM_RMSE_LAST]:6.2f}mm  "
            f"max={t[TLM_MAXERR_LAST]:6.2f}mm  mean={t[TLM_MEANERR_LAST]:6.2f}mm  "
            f"cost_last={t[TLM_LAST_COST]:7.2f}  best={t[TLM_BEST_COST]:7.2f}  "
            f"iter={int(t[TLM_ITER])}  ||  "
            f"fwd kp/ki/kd={t[TLM_KP_FWD]:.3f}/{t[TLM_KI_FWD]:.3f}/{t[TLM_KD_FWD]:.3f}  "
            f"rev kp/ki/kd={t[TLM_KP_REV]:.3f}/{t[TLM_KI_REV]:.3f}/{t[TLM_KD_REV]:.3f}"
        )
        self.split_var.set(
            f"OUT (+X)  rmse={t[TLM_RMSE_LAST_FWD]:6.2f}mm  "
            f"max={t[TLM_MAXERR_LAST_FWD]:6.2f}mm  "
            f"mean={t[TLM_MEANERR_LAST_FWD]:6.2f}mm  "
            f"cost={t[TLM_LAST_COST_FWD]:7.2f}  best={t[TLM_BEST_COST_FWD]:7.2f}\n"
            f"IN  (-X)  rmse={t[TLM_RMSE_LAST_REV]:6.2f}mm  "
            f"max={t[TLM_MAXERR_LAST_REV]:6.2f}mm  "
            f"mean={t[TLM_MEANERR_LAST_REV]:6.2f}mm  "
            f"cost={t[TLM_LAST_COST_REV]:7.2f}  best={t[TLM_BEST_COST_REV]:7.2f}"
        )

    def _redraw_time_plot(self) -> None:
        c = self.canvas_time
        c.delete("all")
        w = int(c["width"]); h = int(c["height"])
        left, top, right, bottom = 55, 12, w - 10, h - 60
        c.create_rectangle(left, top, right, bottom, outline="#bbb")
        err_top = bottom + 8
        err_bottom = h - 6
        c.create_rectangle(left, err_top, right, err_bottom, outline="#bbb")

        if not self.t_hist:
            c.create_text((left + right) / 2, (top + bottom) / 2,
                          text="waiting for telemetry", fill="#888")
            return

        ts = list(self.t_hist)
        t_lo, t_hi = ts[0], ts[-1]
        if t_hi - t_lo < 1.0:
            t_hi = t_lo + 1.0

        cmd = list(self.cmd_x_hist)
        ee = list(self.ee_x_hist)
        err = list(self.err_hist)

        if self.last_telemetry is not None:
            x_lo = float(self.last_telemetry[TLM_CUR_X_MIN])
            x_hi = float(self.last_telemetry[TLM_CUR_X_MAX])
        else:
            x_lo = min(min(cmd), min(ee))
            x_hi = max(max(cmd), max(ee))
        span = max(1e-3, x_hi - x_lo)
        x_lo -= 0.1 * span
        x_hi += 0.1 * span

        def to_xy(t: float, v: float) -> tuple:
            xp = left + (t - t_lo) / (t_hi - t_lo) * (right - left)
            yp = bottom - ((v - x_lo) / (x_hi - x_lo)) * (bottom - top)
            return xp, yp

        if self.last_telemetry is not None:
            for x_ref in (self.last_telemetry[TLM_CUR_X_MIN],
                          self.last_telemetry[TLM_CUR_X_MAX]):
                _, y = to_xy(t_lo, x_ref)
                c.create_line(left, y, right, y, fill="#cccccc", dash=(2, 4))
                c.create_text(left - 4, y, text=f"{x_ref:.3f}",
                              anchor="e", fill="#888", font=("Courier", 8))

        def draw_series(values, color, width=2):
            pts = []
            for t, v in zip(ts, values):
                pts.extend(to_xy(t, v))
            if len(pts) >= 4:
                c.create_line(*pts, fill=color, width=width)

        draw_series(cmd, "#1f77b4")
        draw_series(ee, "#d62728")

        err_max = max(2.0, max(err) if err else 2.0)

        def err_xy(t, v):
            xp = left + (t - t_lo) / (t_hi - t_lo) * (right - left)
            yp = err_bottom - (v / err_max) * (err_bottom - err_top)
            return xp, yp

        pts = []
        for t, v in zip(ts, err):
            pts.extend(err_xy(t, v))
        if len(pts) >= 4:
            c.create_line(*pts, fill="#2ca02c", width=1)

        c.create_text(left + 4, top + 2, anchor="nw",
                      text="blue=cmd_x  red=measured_x", fill="#555")
        c.create_text(left + 4, err_top + 2, anchor="nw",
                      text=f"|err| (mm), full-scale {err_max:.1f}", fill="#555")

    def _redraw_xz_plot(self) -> None:
        c = self.canvas_xz
        c.delete("all")
        w = int(c["width"]); h = int(c["height"])
        left, top, right, bottom = 50, 12, w - 10, h - 30
        c.create_rectangle(left, top, right, bottom, outline="#bbb")

        if not self.xz_ee_hist or self.last_telemetry is None:
            c.create_text((left + right) / 2, (top + bottom) / 2,
                          text="waiting for telemetry", fill="#888")
            return

        t = self.last_telemetry
        x_lo = min(t[TLM_CUR_X_MIN], min(p[0] for p in self.xz_ee_hist))
        x_hi = max(t[TLM_CUR_X_MAX], max(p[0] for p in self.xz_ee_hist))
        z_lo = min(t[TLM_CUR_Z] - 0.05, min(p[1] for p in self.xz_ee_hist))
        z_hi = max(t[TLM_CUR_Z] + 0.05, max(p[1] for p in self.xz_ee_hist))
        x_span = max(1e-3, x_hi - x_lo)
        z_span = max(1e-3, z_hi - z_lo)
        x_lo -= 0.05 * x_span; x_hi += 0.05 * x_span
        z_lo -= 0.05 * z_span; z_hi += 0.05 * z_span

        def to_xy(x: float, z: float) -> tuple:
            xp = left + (x - x_lo) / (x_hi - x_lo) * (right - left)
            yp = bottom - (z - z_lo) / (z_hi - z_lo) * (bottom - top)
            return xp, yp

        z_cmd = float(t[TLM_CUR_Z])
        x0, y0 = to_xy(t[TLM_CUR_X_MIN], z_cmd)
        x1, y1 = to_xy(t[TLM_CUR_X_MAX], z_cmd)
        c.create_line(x0, y0, x1, y1, fill="#1f77b4", width=2)
        c.create_oval(x0 - 3, y0 - 3, x0 + 3, y0 + 3, outline="#1f77b4")
        c.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, outline="#1f77b4")

        pts = []
        for x, z in self.xz_ee_hist:
            xp, yp = to_xy(x, z)
            pts.extend([xp, yp])
        if len(pts) >= 4:
            c.create_line(*pts, fill="#d62728", width=1)

        c.create_text(left, bottom + 14, anchor="w",
                      text=f"x: {x_lo:.3f}..{x_hi:.3f} m", fill="#555")
        c.create_text(left, top - 2, anchor="sw",
                      text=f"z: {z_lo:.3f}..{z_hi:.3f} m", fill="#555")
        c.create_text(right, top + 2, anchor="ne",
                      text="blue=x-line  red=measured", fill="#555")


# ---------------------------------------------------------------------------
# Top-level window
# ---------------------------------------------------------------------------


class CombinedTunerClient:
    def __init__(self, host: str, joint_port: int, ee_port: int) -> None:
        self.root = tk.Tk()
        self.root.title("Excavator PID Tuner")

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.joint_tab = _JointTab(nb, host, joint_port)
        self.ee_tab = _EETab(nb, host, ee_port)

        nb.add(self.joint_tab.frame, text="  Per-Joint  ")
        nb.add(self.ee_tab.frame, text="  EE Tracking  ")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        try:
            self.joint_tab.cleanup()
        except Exception:
            pass
        try:
            self.ee_tab.cleanup()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PID tuner — client GUI")
    parser.add_argument("host", nargs="?", default=DEFAULT_HOST)
    parser.add_argument("--joint-port", type=int, default=JOINT_PORT)
    parser.add_argument("--ee-port", type=int, default=EE_PORT)
    args = parser.parse_args(argv)
    CombinedTunerClient(args.host, args.joint_port, args.ee_port).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
