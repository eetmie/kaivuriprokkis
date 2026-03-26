#!/usr/bin/env python3
"""
Excavator 3D Position GUI — Keyboard control with UDP streaming.

Orchestrates InputHandler, ArmVisualizer, and ExcavatorUDPClient.

Controls:
    IK:     A/D=X, W/S=Z, Q/E=Y, R/F=Rot | Shift=5x
    Direct: A/D=Slew, W/S=Boom, Q/E=Arm, R/F=Bucket | Shift=full
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

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
from clients.input_handler import InputHandler
from clients.arm_visualizer import ArmVisualizer
from clients.udp_client import ExcavatorUDPClient


class ExcavatorGUI3D:
    """Interactive 3D GUI for selecting excavator end-effector pose."""

    WORKSPACE_LIMITS = {
        'x_min': 0.350, 'x_max': 0.680,
        'y_min': -0.150, 'y_max': 0.150,
        'z_min': -0.300, 'z_max': 0.150,
    }

    DEFAULT_X = 0.6
    DEFAULT_Y = 0.0
    DEFAULT_Z = 0.0

    UDP_HOST = "192.168.0.132"
    UDP_PORT = 8080
    UDP_SEND_RATE_HZ = 20.0

    KEY_STEP_POS = 0.001
    KEY_STEP_ROT = 0.1

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Excavator 3D Position Control GUI - UDP Streaming")

        # Target pose
        self.current_x = self.DEFAULT_X
        self.current_y = self.DEFAULT_Y
        self.current_z = self.DEFAULT_Z
        self.current_rot_y = 0.0

        # Measured pose from telemetry
        self.measured_x = self.DEFAULT_X
        self.measured_y = self.DEFAULT_Y
        self.measured_z = self.DEFAULT_Z
        self.measured_rot_y = 0.0

        # Key state
        self.keys_down = set()

        # Step sizes
        self.step_pos = self.KEY_STEP_POS
        self.step_rot = self.KEY_STEP_ROT

        # Direct mode
        self.direct_mode = False
        self.direct_commands = [0.0, 0.0, 0.0, 0.0]

        # Telemetry
        self.joint_positions = None
        self.command_sequence = 0

        # Components
        self.input_handler = InputHandler()
        self.udp_client = ExcavatorUDPClient()

        # Xbox controller (optional)
        self._controller = None
        try:
            from modules.gamepad import XboxController
            self._controller = XboxController(max_reconnect=None, deadzone=20.0, padding=5.0)
        except Exception as e:
            print(f"[GUI] Gamepad not available: {e}")

        # One-shot flags
        self.reload_flag = 0
        self.pause_flag = 0
        self.resume_flag = 0

        # Build UI
        self._create_widgets()
        self._bind_events()
        self.visualizer.update(
            (self.current_x, self.current_y, self.current_z, self.current_rot_y),
            self.joint_positions,
        )

        # Start tick loop (~30 FPS)
        self._schedule_tick()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ---- UI Layout ----
    def _create_widgets(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        title = ttk.Label(main, text="Excavator 3D Position Control", font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 8))

        # 3D Visualizer
        self.visualizer = ArmVisualizer(main, self.WORKSPACE_LIMITS)
        canvas_widget = self.visualizer.get_canvas_widget()
        canvas_widget.grid(row=1, column=0, columnspan=3, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Pose labels
        info = ttk.LabelFrame(main, text="Target / Measured Pose", padding="8")
        info.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E))
        row = ttk.Frame(info)
        row.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.x_label = ttk.Label(row, text="X T:+0.000 M:+0.000", font=('Courier', 12))
        self.y_label = ttk.Label(row, text="Y T:+0.000 M:+0.000", font=('Courier', 12))
        self.z_label = ttk.Label(row, text="Z T:+0.000 M:+0.000", font=('Courier', 12))
        self.rot_label = ttk.Label(row, text="Rot T:+0.0 M:+0.0", font=('Courier', 12), width=22)
        self.x_label.pack(side=tk.LEFT, padx=(0, 14))
        self.y_label.pack(side=tk.LEFT, padx=(0, 14))
        self.z_label.pack(side=tk.LEFT, padx=(0, 14))
        self.rot_label.pack(side=tk.LEFT)

        # Control settings
        settings = ttk.LabelFrame(main, text="Control Settings", padding="8")
        settings.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(8, 0))

        ttk.Label(settings, text="Move step (m):").grid(row=0, column=0, sticky=tk.W)
        self.pos_step_var = tk.StringVar(value=f"{self.step_pos:.3f}")
        self.pos_step_spin = tk.Spinbox(
            settings, from_=0.001, to=0.025, increment=0.001,
            textvariable=self.pos_step_var, width=8, command=self._apply_pos_step,
        )
        self.pos_step_spin.grid(row=0, column=1, padx=(6, 12))
        self.pos_step_spin.bind('<Return>', lambda e: self._apply_pos_step())
        self.pos_step_spin.bind('<FocusOut>', lambda e: self._apply_pos_step())

        ttk.Label(settings, text="Rot step (deg):").grid(row=0, column=2, sticky=tk.W)
        self.rot_step_var = tk.StringVar(value=f"{self.step_rot:.1f}")
        self.rot_step_spin = tk.Spinbox(
            settings, from_=0.1, to=1.0, increment=0.1,
            textvariable=self.rot_step_var, width=8, command=self._apply_rot_step,
        )
        self.rot_step_spin.grid(row=0, column=3, padx=(6, 12))
        self.rot_step_spin.bind('<Return>', lambda e: self._apply_rot_step())
        self.rot_step_spin.bind('<FocusOut>', lambda e: self._apply_rot_step())

        ttk.Label(settings, text="Hold Shift for 5\u00d7 step size").grid(
            row=0, column=4, sticky=tk.W, padx=(16, 0))

        # UDP settings
        cfg = ttk.LabelFrame(main, text="UDP Config", padding="8")
        cfg.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(8, 0))

        ttk.Label(cfg, text="Host:").grid(row=0, column=0, sticky=tk.W)
        self.host_var = tk.StringVar(value=self.UDP_HOST)
        ttk.Entry(cfg, textvariable=self.host_var, width=15).grid(row=0, column=1, padx=(4, 12))

        ttk.Label(cfg, text="Port:").grid(row=0, column=2, sticky=tk.W)
        self.port_var = tk.StringVar(value=str(self.UDP_PORT))
        ttk.Entry(cfg, textvariable=self.port_var, width=7).grid(row=0, column=3, padx=(4, 12))

        ttk.Label(cfg, text="Send rate (Hz):").grid(row=0, column=4, sticky=tk.W)
        self.rate_var = tk.StringVar(value=f"{self.UDP_SEND_RATE_HZ:.1f}")
        ttk.Entry(cfg, textvariable=self.rate_var, width=7).grid(row=0, column=5, padx=(4, 12))

        # Buttons
        btns = ttk.Frame(cfg)
        btns.grid(row=1, column=0, columnspan=6, pady=(8, 0), sticky=tk.W)

        self.connect_btn = ttk.Button(btns, text="Connect", command=self._toggle_connect)
        self.stream_btn = ttk.Button(btns, text="Start Streaming", command=self._toggle_streaming, state=tk.DISABLED)
        self.reload_btn = ttk.Button(btns, text="Reload", command=self._trigger_reload, state=tk.DISABLED)
        self.pause_btn = ttk.Button(btns, text="Pause", command=self._trigger_pause, state=tk.DISABLED)
        self.resume_btn = ttk.Button(btns, text="Resume", command=self._trigger_resume, state=tk.DISABLED)
        self.direct_btn = ttk.Button(btns, text="Direct Mode", command=self._toggle_direct_mode, state=tk.DISABLED)
        self.copy_btn = ttk.Button(btns, text="Copy Position", command=self._copy_command)
        self.reset_btn = ttk.Button(btns, text="Reset (0.6, 0.0, 0\u00b0)", command=self._reset_position)

        for i, btn in enumerate([self.connect_btn, self.stream_btn, self.reload_btn,
                                  self.pause_btn, self.resume_btn, self.direct_btn,
                                  self.copy_btn, self.reset_btn]):
            btn.grid(row=0, column=i, padx=(0, 4))

        # UDP info
        udp_frame = ttk.LabelFrame(main, text="UDP Streaming Control", padding="8")
        udp_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(8, 0))

        top = ttk.Frame(udp_frame)
        top.grid(row=0, column=0, columnspan=3, pady=(0, 6))
        self.target_label = ttk.Label(top, text=f"Target: {self.UDP_HOST}:{self.UDP_PORT}", font=('Courier', 9))
        self.target_label.grid(row=0, column=0, sticky=tk.W)
        self.rate_label = ttk.Label(top, text=f"Target Rate: {self.UDP_SEND_RATE_HZ:.1f} Hz", font=('Courier', 9))
        self.rate_label.grid(row=0, column=1, sticky=tk.W, padx=(20, 0))

        stat = ttk.Frame(udp_frame)
        stat.grid(row=1, column=0, sticky=tk.W)
        self.status_label = ttk.Label(stat, text="Status: Disconnected", font=('Courier', 10))
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        self.hz_label = ttk.Label(stat, text="Send Hz: 0.0", font=('Courier', 10))
        self.hz_label.grid(row=0, column=1, sticky=tk.W, padx=(12, 0))
        self.recv_label = ttk.Label(stat, text="Recv: None", font=('Courier', 10))
        self.recv_label.grid(row=0, column=2, sticky=tk.W, padx=(12, 0))

        # Hint
        ttk.Label(
            main,
            text="IK: A/D=X, W/S=Z, Q/E=Y, R/F=Rot | Direct: A/D=Slew, W/S=Boom, Q/E=Arm, R/F=Bucket | Shift=fast",
            font=('Courier', 9),
        ).grid(row=6, column=0, columnspan=3, pady=(6, 0), sticky=tk.W)

        # Resizing
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        for c in range(3):
            main.columnconfigure(c, weight=1)

    def _bind_events(self):
        self.root.bind('<KeyPress>', self._on_key_press)
        self.root.bind('<KeyRelease>', self._on_key_release)

    # ---- Keyboard ----
    def _on_key_press(self, event):
        key = (event.keysym or '').lower()
        if key in ('shift_l', 'shift_r'):
            self.keys_down.add('shift')
        else:
            self.keys_down.add(key)

    def _on_key_release(self, event):
        key = (event.keysym or '').lower()
        if key in ('shift_l', 'shift_r'):
            self.keys_down.discard('shift')
        else:
            self.keys_down.discard(key)

    # ---- Tick loop ----
    def _schedule_tick(self):
        self._tick()
        self.root.after(33, self._schedule_tick)

    def _tick(self):
        controller_state = None
        if self._controller is not None:
            try:
                controller_state = self._controller.read()
            except Exception:
                pass

        if self.direct_mode:
            slew, boom, arm, bucket = self.input_handler.tick_direct(self.keys_down, controller_state)
            self.direct_commands = [slew, boom, arm, bucket]
            self.visualizer.set_state(
                target_pose=(self.current_x, self.current_y, self.current_z, self.current_rot_y),
                joint_positions=self.joint_positions,
            )
            self.visualizer.request_redraw(self.root)
        else:
            dx, dy, dz, drot = self.input_handler.tick_ik(
                self.keys_down, controller_state, self.step_pos, self.step_rot)
            if dx or dy or dz or drot:
                self.current_x += dx
                self.current_y += dy
                self.current_z += dz
                self.current_rot_y += drot
                self._clamp_pose()
                self.visualizer.set_state(
                    target_pose=(self.current_x, self.current_y, self.current_z, self.current_rot_y),
                    joint_positions=self.joint_positions,
                )
                self.visualizer.request_redraw(self.root)

        self._update_labels()

    def _clamp_pose(self):
        wl = self.WORKSPACE_LIMITS
        self.current_x = max(wl['x_min'], min(self.current_x, wl['x_max']))
        self.current_y = max(wl['y_min'], min(self.current_y, wl['y_max']))
        self.current_z = max(wl['z_min'], min(self.current_z, wl['z_max']))
        self.current_rot_y = max(-45.0, min(45.0, self.current_rot_y))

    def _update_labels(self):
        self.x_label.config(text=f"X T:{self.current_x:+6.3f} M:{self.measured_x:+6.3f}")
        self.y_label.config(text=f"Y T:{self.current_y:+6.3f} M:{self.measured_y:+6.3f}")
        self.z_label.config(text=f"Z T:{self.current_z:+6.3f} M:{self.measured_z:+6.3f}")
        self.rot_label.config(text=f"Rot T:{self.current_rot_y:+5.1f} M:{self.measured_rot_y:+5.1f}")

    # ---- Settings ----
    def _apply_pos_step(self):
        try:
            val = float(self.pos_step_var.get())
            if val <= 0:
                raise ValueError
            self.step_pos = min(val, 0.025)
            self.pos_step_var.set(f"{self.step_pos:.3f}")
        except ValueError:
            messagebox.showerror("Invalid value", "Move step must be a positive number.")
            self.pos_step_var.set(f"{self.step_pos:.3f}")

    def _apply_rot_step(self):
        try:
            val = float(self.rot_step_var.get())
            if val <= 0:
                raise ValueError
            self.step_rot = val
        except ValueError:
            messagebox.showerror("Invalid value", "Rot step must be a positive number.")
            self.rot_step_var.set(f"{self.step_rot:.1f}")

    def _copy_command(self):
        cmd = f"{self.current_x:.3f}, {self.current_y:.3f}, {self.current_z:.3f}, {self.current_rot_y:.1f}"
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.root.update()
        messagebox.showinfo("Copied", f"Copied pose: {cmd}")

    def _reset_position(self):
        self.current_x = self.DEFAULT_X
        self.current_y = self.DEFAULT_Y
        self.current_z = self.DEFAULT_Z
        self.current_rot_y = 0.0
        self._clamp_pose()
        self.visualizer.request_redraw(self.root)

    # ---- Direct mode ----
    def _toggle_direct_mode(self):
        self.direct_mode = not self.direct_mode
        if self.direct_mode:
            self.direct_commands = [0.0, 0.0, 0.0, 0.0]
            self.direct_btn.config(text="IK Mode")
            self.status_label.config(text="Status: DIRECT MODE")
        else:
            self.direct_commands = [0.0, 0.0, 0.0, 0.0]
            self.current_x = self.measured_x
            self.current_y = self.measured_y
            self.current_z = self.measured_z
            self.current_rot_y = self.measured_rot_y
            self.direct_btn.config(text="Direct Mode")
            if self.udp_client.is_streaming:
                self.status_label.config(text="Status: Streaming")

    # ---- UDP / Streaming ----
    def _toggle_connect(self):
        if not self.udp_client.is_connected:
            self._connect()
        else:
            self._disconnect()

    def _connect(self):
        try:
            host = self.host_var.get().strip()
            port = int(self.port_var.get().strip())
            rate = float(self.rate_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid config", "Host/port/rate settings are invalid.")
            return

        if not self.udp_client.connect(host, port, rate):
            messagebox.showerror("Connection failed", "Failed to connect / handshake.")
            return

        self.connect_btn.config(text="Disconnect")
        self.stream_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Status: Connected to {host}:{port}")
        self.target_label.config(text=f"Target: {host}:{port}")

    def _disconnect(self):
        self._stop_streaming()
        self.udp_client.disconnect()
        self.connect_btn.config(text="Connect")
        self.stream_btn.config(state=tk.DISABLED)
        self.reload_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.resume_btn.config(state=tk.DISABLED)
        self.direct_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Disconnected")

    def _toggle_streaming(self):
        if not self.udp_client.is_streaming:
            self._start_streaming()
        else:
            self._stop_streaming()

    def _start_streaming(self):
        if not self.udp_client.is_connected:
            messagebox.showwarning("Not connected", "Connect to the UDP server first.")
            return

        self.udp_client.start_streaming(
            get_command=self._build_command,
            on_telemetry=self._on_telemetry,
            on_hz_update=lambda hz: self.root.after(0, self._update_hz, hz),
        )

        self.stream_btn.config(text="Stop Streaming")
        self.reload_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)
        self.resume_btn.config(state=tk.NORMAL)
        self.direct_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Streaming")

    def _stop_streaming(self):
        if not self.udp_client.is_streaming:
            return
        self.udp_client.stop_streaming()
        self.stream_btn.config(text="Start Streaming")
        self.reload_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.resume_btn.config(state=tk.DISABLED)
        self.direct_btn.config(state=tk.DISABLED)
        if self.direct_mode:
            self.direct_mode = False
            self.direct_commands = [0.0, 0.0, 0.0, 0.0]
            self.direct_btn.config(text="Direct Mode")
        self.hz_label.config(text="Send Hz: 0.0")
        if self.udp_client.is_connected:
            self.status_label.config(text="Status: Connected (Idle)")
        else:
            self.status_label.config(text="Status: Disconnected")

    def _build_command(self) -> ControlCommand:
        """Called by UDP send thread each tick."""
        self.command_sequence += 1
        cmd = ControlCommand(
            sequence=self.command_sequence,
            mode=ControlMode.DIRECT if self.direct_mode else ControlMode.IK,
            pose=PoseTarget(self.current_x, self.current_y, self.current_z, self.current_rot_y),
            direct=DirectCommand(*self.direct_commands),
            reload_config=bool(self.reload_flag),
            pause=bool(self.pause_flag),
            resume=bool(self.resume_flag),
        )
        if self.reload_flag or self.pause_flag or self.resume_flag:
            self.reload_flag = 0
            self.pause_flag = 0
            self.resume_flag = 0
        return cmd

    def _on_telemetry(self, telemetry: RobotTelemetry):
        """Called by UDP receive thread."""
        self.joint_positions = [list(pos) for pos in telemetry.joint_positions]
        self.measured_x = telemetry.measured_pose.x
        self.measured_y = telemetry.measured_pose.y
        self.measured_z = telemetry.measured_pose.z
        self.measured_rot_y = telemetry.measured_pose.rot_y_deg
        mode_name = "DIRECT" if telemetry.mode == ControlMode.DIRECT else "IK"
        fusion = "fused" if telemetry.slew_fusion_active else ("enc" if telemetry.slew_fusion_enabled else "off")
        self.root.after(0, self._apply_telemetry_to_view, self.joint_positions, mode_name, fusion)

    def _apply_telemetry_to_view(self, joint_positions, mode_name: str, fusion: str):
        self.visualizer.set_state(
            target_pose=(self.current_x, self.current_y, self.current_z, self.current_rot_y),
            joint_positions=joint_positions,
        )
        self.visualizer.request_redraw(self.root)
        self.recv_label.config(text=f"Recv: {mode_name} {fusion}")

    def _update_hz(self, hz: float):
        self.hz_label.config(text=f"Send Hz: {hz:4.1f}")

    # ---- One-shot flags ----
    def _trigger_reload(self):
        self.reload_flag = 1

    def _trigger_pause(self):
        self.pause_flag = 1

    def _trigger_resume(self):
        self.resume_flag = 1

    # ---- Shutdown ----
    def _on_closing(self):
        if self.udp_client.is_streaming:
            self._stop_streaming()
        if self.udp_client.is_connected:
            self._disconnect()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ExcavatorGUI3D(root)
    root.mainloop()


if __name__ == "__main__":
    main()
