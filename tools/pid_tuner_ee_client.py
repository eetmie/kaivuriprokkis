#!/usr/bin/env python3
"""End-effector PID tuner — client GUI.

Pair with :mod:`tools.pid_tuner_ee` running on the robot. The client is a
thin Tk window matching the style of ``tools/pid_tuner_client.py``:

* connect / disconnect to the host over UDP
* set the X-line sweep (``x_min``, ``x_max``), ``z`` height and target speed
* pick which joint's PID to mutate and whether to install a directional PID
* push gains manually, or hand control to the host-side auto-tuner
* live plot of the X-line vs. the measured EE travel + tracking-error stats

The GUI is presentation-only; all motion, tuning and bookkeeping live on the
host. See :mod:`tools.pid_tuner_ee_common` for the wire format and the host
module's docstring for the auto-tune algorithm.

Run with no args to bind to the rpi default
(``DEFAULT_HOST``/``DEFAULT_PORT``); ``-h`` lists every option.
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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.udp_socket import UDPSocket

from tools.pid_tuner_ee_common import (
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
    COMMAND_SIZE,
    CommandFlag,
    DEFAULT_COST_WEIGHTS,
    DEFAULT_HOST,
    DEFAULT_NUM_RUNS,
    DEFAULT_PORT,
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
    TELEMETRY_SIZE,
    TLM_BEST_COST,
    TLM_BEST_COST_FWD,
    TLM_BEST_COST_REV,
    TLM_CMD_X,
    TLM_CMD_Y,
    TLM_CMD_Z,
    TLM_CUR_SPEED,
    TLM_CUR_X_MAX,
    TLM_CUR_X_MIN,
    TLM_CUR_Z,
    TLM_DIR_PID,
    TLM_EE_X,
    TLM_EE_Y,
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
    state_label,
)


JOINT_NAMES = ["slew", "boom", "arm", "bucket"]
# X-plane tuning only covers the joints that actually move the EE in X:
# lift (boom), tilt (arm), scoop (bucket). Slew is excluded — it has no
# effect on +X / -X tracking when the cab faces forward.
TUNABLE_JOINTS: tuple = (1, 2, 3)
TUNABLE_NAMES: tuple = tuple(JOINT_NAMES[i] for i in TUNABLE_JOINTS)


class PIDTunerEEClient:
    """Top-level Tk app."""

    PERIOD_MS = 50         # UI tick + UDP send rate (20 Hz)
    HISTORY_POINTS = 1200  # 60 s of trace at 20 Hz

    def __init__(self, host: str, port: int) -> None:
        self.sock: Optional[UDPSocket] = None
        self.host_default = host
        self.port_default = port

        self.root = tk.Tk()
        self.root.title("Excavator EE PID Tuner")

        # ----- Tk variables --------------------------------------------------
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
        # Auto-tune sweep envelopes — host cycles (z, speed) across these.
        self.z_min_var = tk.DoubleVar(value=DEFAULT_Z_MIN)
        self.z_max_var = tk.DoubleVar(value=DEFAULT_Z_MAX)
        self.speed_min_var = tk.DoubleVar(value=DEFAULT_SPEED_MIN)
        self.speed_max_var = tk.DoubleVar(value=DEFAULT_SPEED_MAX)

        self.joint_var = tk.IntVar(value=1)  # boom
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

        # Pending one-shot flags. The Tk tick OR-s these into the next packet.
        self._pending_flags = 0

        # Telemetry buffers (deques, x in metres).
        self.t_hist: Deque[float] = deque(maxlen=self.HISTORY_POINTS)
        self.cmd_x_hist: Deque[float] = deque(maxlen=self.HISTORY_POINTS)
        self.ee_x_hist: Deque[float] = deque(maxlen=self.HISTORY_POINTS)
        self.err_hist: Deque[float] = deque(maxlen=self.HISTORY_POINTS)
        # XZ trace for the trajectory view.
        self.xz_cmd_hist: Deque[tuple] = deque(maxlen=self.HISTORY_POINTS)
        self.xz_ee_hist: Deque[tuple] = deque(maxlen=self.HISTORY_POINTS)

        self.last_telemetry: Optional[List[float]] = None
        self.connect_t0 = time.perf_counter()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self.PERIOD_MS, self._tick)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = ttk.Frame(self.root, padding=8)
        root.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

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

        # Program parameters
        prog = ttk.LabelFrame(panel, text="Program", padding=6)
        prog.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=(0, 4))
        self._labeled_entry(prog, 0, "x_min (m)", self.x_min_var)
        self._labeled_entry(prog, 1, "x_max (m)", self.x_max_var)
        self._labeled_entry(prog, 2, "y      (m)", self.y_var)
        self._labeled_entry(prog, 3, "z      (m)", self.z_var)
        self._labeled_entry(prog, 4, "speed (m/s)", self.speed_var,
                            tooltip="EE linear speed for MANUAL runs, range 0.020 .. 0.070")
        self._labeled_entry(prog, 5, "strokes/run", self.strokes_var)
        self._labeled_entry(prog, 6, "auto-tune runs", self.num_runs_var)
        self._labeled_entry(prog, 7, "z_min (m)", self.z_min_var,
                            tooltip="Auto-tune Z sweep lower bound")
        self._labeled_entry(prog, 8, "z_max (m)", self.z_max_var,
                            tooltip="Auto-tune Z sweep upper bound")
        self._labeled_entry(prog, 9, "v_min (m/s)", self.speed_min_var,
                            tooltip="Auto-tune speed sweep lower bound (e.g. 0.020)")
        self._labeled_entry(prog, 10, "v_max (m/s)", self.speed_max_var,
                            tooltip="Auto-tune speed sweep upper bound (e.g. 0.070)")

        # Joint + PID selection
        sel = ttk.LabelFrame(panel, text="Joint / PID", padding=6)
        sel.grid(row=0, column=4, columnspan=2, sticky="nsew", padx=(0, 4))
        ttk.Label(sel, text="Tuned joint").grid(row=0, column=0, sticky="w")
        joint_combo = ttk.Combobox(sel, values=list(TUNABLE_NAMES), state="readonly", width=10)
        try:
            joint_combo.current(TUNABLE_JOINTS.index(self.joint_var.get()))
        except ValueError:
            joint_combo.current(0)
            self.joint_var.set(TUNABLE_JOINTS[0])
        joint_combo.grid(row=0, column=1, sticky="w")
        joint_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.joint_var.set(TUNABLE_JOINTS[joint_combo.current()]),
        )
        ttk.Checkbutton(sel, text="Directional PID", variable=self.dir_pid_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Checkbutton(sel, text="Auto-tune", variable=self.auto_tune_var).grid(
            row=2, column=0, columnspan=2, sticky="w"
        )
        ttk.Button(sel, text="Reset PIDs", command=self._reset_pids).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(sel, text="Reload servo config", command=self._reload_config).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(2, 0)
        )

        # Gains
        gains = ttk.LabelFrame(panel, text="Manual gains (per direction)", padding=6)
        gains.grid(row=0, column=6, columnspan=2, sticky="nsew")
        ttk.Label(gains, text="").grid(row=0, column=0)
        ttk.Label(gains, text="fwd").grid(row=0, column=1)
        ttk.Label(gains, text="rev").grid(row=0, column=2)
        for r, (label, fv, rv) in enumerate((
            ("kp", self.kp_fwd_var, self.kp_rev_var),
            ("ki", self.ki_fwd_var, self.ki_rev_var),
            ("kd", self.kd_fwd_var, self.kd_rev_var),
        ), start=1):
            ttk.Label(gains, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(gains, textvariable=fv, width=8).grid(row=r, column=1, padx=2)
            ttk.Entry(gains, textvariable=rv, width=8).grid(row=r, column=2, padx=2)
        ttk.Button(gains, text="Apply gains", command=self._apply_gains).grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )
        ttk.Button(gains, text="Pull gains from host", command=self._pull_gains).grid(
            row=5, column=0, columnspan=3, sticky="ew"
        )

        # Run control row
        runrow = ttk.Frame(parent)
        runrow.grid(row=1, column=0, sticky="ew", pady=(2, 4))
        ttk.Button(runrow, text="START", command=self._start).pack(side=tk.LEFT)
        ttk.Button(runrow, text="STOP", command=self._stop).pack(side=tk.LEFT, padx=4)
        ttk.Label(runrow, text="cost weights — rmse:").pack(side=tk.LEFT, padx=(16, 2))
        ttk.Entry(runrow, textvariable=self.cost_w_rmse_var, width=6).pack(side=tk.LEFT)
        ttk.Label(runrow, text="max:").pack(side=tk.LEFT, padx=(4, 2))
        ttk.Entry(runrow, textvariable=self.cost_w_max_var, width=6).pack(side=tk.LEFT)
        ttk.Button(runrow, text="Save snapshot", command=self._snapshot).pack(side=tk.LEFT, padx=12)

    def _labeled_entry(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        var: tk.Variable,
        tooltip: Optional[str] = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        e = ttk.Entry(parent, textvariable=var, width=10)
        e.grid(row=row, column=1, sticky="w", padx=2)
        if tooltip:
            e.bind("<Enter>", lambda _e, t=tooltip: self.status_var.set(t))

    def _build_plot_area(self, parent: ttk.Frame) -> None:
        plots = ttk.Frame(parent)
        plots.grid(row=2, column=0, sticky="nsew")
        plots.columnconfigure(0, weight=2)
        plots.columnconfigure(1, weight=1)
        plots.rowconfigure(0, weight=1)

        # Time plot: cmd vs measured x over time + error trace
        self.canvas_time = tk.Canvas(plots, width=900, height=380, bg="white", highlightthickness=1, highlightbackground="#bbb")
        self.canvas_time.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        # XZ trajectory view (top down isn't right — this is side view: x horizontal, z vertical)
        self.canvas_xz = tk.Canvas(plots, width=380, height=380, bg="white", highlightthickness=1, highlightbackground="#bbb")
        self.canvas_xz.grid(row=0, column=1, sticky="nsew")

    def _build_status_row(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self.metrics_var = tk.StringVar(value="no telemetry yet")
        self.gains_var = tk.StringVar(value="")
        self.split_var = tk.StringVar(value="OUT/IN breakdown will appear after first scored run")
        ttk.Label(row, textvariable=self.metrics_var, font=("Courier", 10)).pack(side=tk.TOP, anchor="w")
        ttk.Label(row, textvariable=self.gains_var, font=("Courier", 10)).pack(side=tk.TOP, anchor="w")
        ttk.Label(row, textvariable=self.split_var, font=("Courier", 10),
                  foreground="#0a4").pack(side=tk.TOP, anchor="w")

    # ---------------------------------------------------------------- UDP

    def _connect(self) -> None:
        self._disconnect()
        host = self.host_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            self.status_var.set("Invalid port")
            return
        try:
            sock = UDPSocket(
                local_id=13,
                max_age_seconds=0.5,
                nominal_rate_hz=1000.0 / self.PERIOD_MS,
            )
            sock.setup(
                host, port,
                inputs=f"{TELEMETRY_SIZE}f",
                outputs=f"{COMMAND_SIZE}f",
                is_server=False,
            )
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

    # ----------------------------------------------------------- commands

    def _start(self) -> None:
        flag = int(CommandFlag.START)
        if self.auto_tune_var.get():
            flag |= int(CommandFlag.AUTO_TUNE)
        if self.dir_pid_var.get():
            flag |= int(CommandFlag.DIR_PID)
        # Clear history at start so the plot reflects only the new run.
        self.t_hist.clear()
        self.cmd_x_hist.clear()
        self.ee_x_hist.clear()
        self.err_hist.clear()
        self.xz_cmd_hist.clear()
        self.xz_ee_hist.clear()
        self._pending_flags |= flag

    def _stop(self) -> None:
        self._pending_flags |= int(CommandFlag.STOP)

    def _reset_pids(self) -> None:
        self._pending_flags |= int(CommandFlag.RESET_PIDS)

    def _reload_config(self) -> None:
        self._pending_flags |= int(CommandFlag.RELOAD_CONFIG)

    def _apply_gains(self) -> None:
        self._pending_flags |= int(CommandFlag.APPLY_GAINS)
        if self.dir_pid_var.get():
            self._pending_flags |= int(CommandFlag.DIR_PID)

    def _pull_gains(self) -> None:
        # Take whatever the host last reported as the installed gains.
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
            self.status_var.set(f"matplotlib required for snapshot: {exc}")
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

    # --------------------------------------------------------------- tick

    def _build_command(self) -> List[float]:
        cmd = [0.0] * COMMAND_SIZE
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
        # One-shot flags are consumed each tick.
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
                if pkt is not None and len(pkt) >= TELEMETRY_SIZE:
                    self._absorb_telemetry([float(v) for v in pkt[:TELEMETRY_SIZE]])
            self._redraw_time_plot()
            self._redraw_xz_plot()
        finally:
            self.root.after(self.PERIOD_MS, self._tick)

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

        state_txt = state_label(t[TLM_STATE])
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
        # Combined-run line — quick at-a-glance.
        self.gains_var.set(
            f"err_now={t[TLM_ERR_NOW]:6.2f}mm  rmse={t[TLM_RMSE_LAST]:6.2f}mm  "
            f"max={t[TLM_MAXERR_LAST]:6.2f}mm  mean={t[TLM_MEANERR_LAST]:6.2f}mm  "
            f"cost_last={t[TLM_LAST_COST]:7.2f}  best={t[TLM_BEST_COST]:7.2f}  "
            f"iter={int(t[TLM_ITER])}  ||  "
            f"fwd kp/ki/kd={t[TLM_KP_FWD]:.3f}/{t[TLM_KI_FWD]:.3f}/{t[TLM_KD_FWD]:.3f}  "
            f"rev kp/ki/kd={t[TLM_KP_REV]:.3f}/{t[TLM_KI_REV]:.3f}/{t[TLM_KD_REV]:.3f}"
        )
        # Per-direction breakdown — the auto-tuner scores each gain triple
        # against its own direction, so the GUI shows them separately.
        self.split_var.set(
            f"OUT (+X)  rmse={t[TLM_RMSE_LAST_FWD]:6.2f}mm  max={t[TLM_MAXERR_LAST_FWD]:6.2f}mm  "
            f"mean={t[TLM_MEANERR_LAST_FWD]:6.2f}mm  cost={t[TLM_LAST_COST_FWD]:7.2f}  "
            f"best={t[TLM_BEST_COST_FWD]:7.2f}\n"
            f"IN  (-X)  rmse={t[TLM_RMSE_LAST_REV]:6.2f}mm  max={t[TLM_MAXERR_LAST_REV]:6.2f}mm  "
            f"mean={t[TLM_MEANERR_LAST_REV]:6.2f}mm  cost={t[TLM_LAST_COST_REV]:7.2f}  "
            f"best={t[TLM_BEST_COST_REV]:7.2f}"
        )

    # ---------------------------------------------------------- drawing

    def _redraw_time_plot(self) -> None:
        c = self.canvas_time
        c.delete("all")
        w = int(c["width"]); h = int(c["height"])
        left, top, right, bottom = 55, 12, w - 10, h - 60
        # Frame
        c.create_rectangle(left, top, right, bottom, outline="#bbb")
        # Lower error band
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

        cmd = list(self.cmd_x_hist); ee = list(self.ee_x_hist); err = list(self.err_hist)
        # Y bounds for x: use telemetry x range +/- margin.
        if self.last_telemetry is not None:
            x_lo = float(self.last_telemetry[TLM_CUR_X_MIN])
            x_hi = float(self.last_telemetry[TLM_CUR_X_MAX])
        else:
            x_lo, x_hi = min(min(cmd), min(ee)), max(max(cmd), max(ee))
        span = max(1e-3, x_hi - x_lo)
        x_lo -= 0.1 * span
        x_hi += 0.1 * span

        def to_xy(t: float, v: float) -> tuple:
            x = left + (t - t_lo) / (t_hi - t_lo) * (right - left)
            y_norm = (v - x_lo) / (x_hi - x_lo)
            y = bottom - y_norm * (bottom - top)
            return x, y

        # Reference line at x_min and x_max
        for x_ref in (
            float(self.last_telemetry[TLM_CUR_X_MIN]) if self.last_telemetry else None,
            float(self.last_telemetry[TLM_CUR_X_MAX]) if self.last_telemetry else None,
        ):
            if x_ref is None:
                continue
            _, y = to_xy(t_lo, x_ref)
            c.create_line(left, y, right, y, fill="#cccccc", dash=(2, 4))
            c.create_text(left - 4, y, text=f"{x_ref:.3f}", anchor="e", fill="#888", font=("Courier", 8))

        def draw_series(values, color, width=2):
            pts = []
            for t, v in zip(ts, values):
                pts.extend(to_xy(t, v))
            if len(pts) >= 4:
                c.create_line(*pts, fill=color, width=width)

        draw_series(cmd, "#1f77b4")
        draw_series(ee, "#d62728")

        # Error band
        err_max = max(2.0, max(err) if err else 2.0)
        def err_xy(t, v):
            x = left + (t - t_lo) / (t_hi - t_lo) * (right - left)
            y = err_bottom - (v / err_max) * (err_bottom - err_top)
            return x, y
        pts = []
        for t, v in zip(ts, err):
            pts.extend(err_xy(t, v))
        if len(pts) >= 4:
            c.create_line(*pts, fill="#2ca02c", width=1)

        c.create_text(left + 4, top + 2, anchor="nw", text="blue=cmd_x  red=measured_x", fill="#555")
        c.create_text(left + 4, err_top + 2, anchor="nw", text=f"|err| (mm), full-scale {err_max:.1f}", fill="#555")

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
            xpx = left + (x - x_lo) / (x_hi - x_lo) * (right - left)
            ypx = bottom - (z - z_lo) / (z_hi - z_lo) * (bottom - top)
            return xpx, ypx

        # Ideal X-line at the commanded Z
        z_cmd = float(t[TLM_CUR_Z])
        x0, y0 = to_xy(t[TLM_CUR_X_MIN], z_cmd)
        x1, y1 = to_xy(t[TLM_CUR_X_MAX], z_cmd)
        c.create_line(x0, y0, x1, y1, fill="#1f77b4", width=2)
        c.create_oval(x0 - 3, y0 - 3, x0 + 3, y0 + 3, outline="#1f77b4")
        c.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, outline="#1f77b4")

        # Measured EE trace
        pts = []
        for x, z in self.xz_ee_hist:
            xp, yp = to_xy(x, z)
            pts.extend([xp, yp])
        if len(pts) >= 4:
            c.create_line(*pts, fill="#d62728", width=1)

        c.create_text(left, bottom + 14, anchor="w",
                      text=f"x: {x_lo:.3f} .. {x_hi:.3f} m", fill="#555")
        c.create_text(left, top - 2, anchor="sw",
                      text=f"z: {z_lo:.3f} .. {z_hi:.3f} m", fill="#555")
        c.create_text(right, top + 2, anchor="ne",
                      text="blue=x-line  red=measured", fill="#555")

    # ----------------------------------------------------------- close

    def _on_close(self) -> None:
        try:
            self._stop()
            if self.sock is not None:
                self.sock.send(self._build_command())
        except Exception:
            pass
        self._disconnect()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EE-plane PID tuner — client GUI")
    parser.add_argument("host", nargs="?", default=DEFAULT_HOST)
    parser.add_argument("port", nargs="?", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    PIDTunerEEClient(args.host, args.port).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
