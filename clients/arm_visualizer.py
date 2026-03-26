"""
3D arm visualization for the excavator GUI.

Extracted from client_gui.py — handles matplotlib 3D plot, workspace box,
robot chain drawing, and target arrow rendering.
"""

import math
import time
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Dict, List, Optional, Tuple


class ArmVisualizer:
    """Manages the 3D matplotlib plot showing the excavator arm and target."""

    def __init__(self, parent_frame: tk.Widget, workspace_limits: Dict[str, float],
                 figsize=(7.5, 4.6), dpi=100, max_fps=30):
        self.workspace_limits = workspace_limits
        self._redraw_interval = 1.0 / max_fps
        self._last_redraw_time = 0.0
        self._mouse_dragging = False

        # Create matplotlib figure
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)

        # Bind mouse events for drag detection
        widget = self.canvas.get_tk_widget()
        widget.bind("<Configure>", self._on_canvas_resize)
        widget.bind("<ButtonPress>", lambda e: setattr(self, '_mouse_dragging', True))
        widget.bind("<ButtonRelease>", self._on_mouse_release)

        self._init_plot()

    def get_canvas_widget(self) -> tk.Widget:
        return self.canvas.get_tk_widget()

    def _init_plot(self):
        self.ax.view_init(elev=25, azim=-60)
        self._set_axes_limits()
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        try:
            self.fig.tight_layout()
        except Exception:
            pass

    def _set_axes_limits(self):
        wl = self.workspace_limits
        xl = (wl['x_min'], wl['x_max'])
        yl = (wl['y_min'], wl['y_max'])
        zl = (wl['z_min'], wl['z_max'])
        max_range = max(xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0])
        r = max_range / 2
        x_mid = sum(xl) / 2
        y_mid = sum(yl) / 2
        z_mid = sum(zl) / 2
        self.ax.set_xlim(x_mid - r, x_mid + r)
        self.ax.set_ylim(y_mid - r, y_mid + r)
        self.ax.set_zlim(z_mid - r, z_mid + r)

    def _on_canvas_resize(self, event):
        try:
            dpi = self.fig.get_dpi()
            w = max(event.width / dpi, 1)
            h = max(event.height / dpi, 1)
            self.fig.set_size_inches(w, h, forward=False)
            try:
                self.fig.tight_layout()
            except Exception:
                pass
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_mouse_release(self, event):
        self._mouse_dragging = False
        self.request_redraw()

    def request_redraw(self, root: Optional[tk.Tk] = None):
        """Request a throttled redraw. Pass root for after_idle scheduling."""
        if self._mouse_dragging:
            return
        now = time.time()
        if now - self._last_redraw_time >= self._redraw_interval:
            self._last_redraw_time = now
            if root:
                root.after_idle(self._do_redraw)
            else:
                self._do_redraw()

    def update(self, target_pose: Tuple[float, float, float, float],
               joint_positions: Optional[List[List[float]]]):
        """Full update: store data then redraw."""
        self._target_pose = target_pose
        self._joint_positions = joint_positions
        self._do_redraw()

    def set_state(
        self,
        target_pose: Optional[Tuple[float, float, float, float]] = None,
        joint_positions: Optional[List[List[float]]] = None,
    ) -> None:
        """Store new scene data without forcing an immediate redraw."""
        if target_pose is not None:
            self._target_pose = target_pose
        if joint_positions is not None:
            self._joint_positions = joint_positions

    def _do_redraw(self):
        """Redraw the 3D scene while preserving camera orientation."""
        try:
            elev = self.ax.elev
            azim = self.ax.azim
            dist = self.ax.dist
            xlim = self.ax.get_xlim3d()
            ylim = self.ax.get_ylim3d()
            zlim = self.ax.get_zlim3d()
        except Exception:
            elev, azim, dist = 25, -60, 10
            xlim = ylim = zlim = None

        self.ax.cla()

        if xlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.set_zlim(zlim)
        else:
            self._set_axes_limits()

        self._draw_workspace_box()
        self._draw_robot()
        self._draw_target()

        self.ax.view_init(elev=elev, azim=azim)
        try:
            self.ax.dist = dist
        except Exception:
            pass

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')

        try:
            self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        except Exception:
            pass

        self.canvas.draw_idle()

    def _draw_workspace_box(self):
        wl = self.workspace_limits
        x0, x1 = wl['x_min'], wl['x_max']
        y0, y1 = wl['y_min'], wl['y_max']
        z0, z1 = wl['z_min'], wl['z_max']
        edges = [
            ((x0, y0, z0), (x1, y0, z0)), ((x0, y1, z0), (x1, y1, z0)),
            ((x0, y0, z1), (x1, y0, z1)), ((x0, y1, z1), (x1, y1, z1)),
            ((x0, y0, z0), (x0, y1, z0)), ((x1, y0, z0), (x1, y1, z0)),
            ((x0, y0, z1), (x0, y1, z1)), ((x1, y0, z1), (x1, y1, z1)),
            ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)),
            ((x0, y1, z0), (x0, y1, z1)), ((x1, y1, z0), (x1, y1, z1)),
        ]
        lc = Line3DCollection(edges, colors='gray', linewidths=1.0, linestyles='dashed')
        self.ax.add_collection3d(lc)

    def _draw_robot(self):
        jp = getattr(self, '_joint_positions', None)
        if not jp or len(jp) < 2:
            return

        # Origin offset (green)
        first_joint = jp[0]
        if np.linalg.norm(first_joint) > 0.001:
            self.ax.plot([0, first_joint[0]], [0, first_joint[1]], [0, first_joint[2]],
                         '-', color='green', linewidth=2.5, label='Origin offset')
            self.ax.scatter([0], [0], [0], c='green', s=30, marker='s')

        # Joint links (blue)
        num_joints = min(4, len(jp) - 1) if len(jp) == 5 else len(jp)
        xs = [p[0] for p in jp[:num_joints]]
        ys = [p[1] for p in jp[:num_joints]]
        zs = [p[2] for p in jp[:num_joints]]
        self.ax.plot(xs, ys, zs, '-o', color='blue', linewidth=2, markersize=4, label='Links')

        # EE offset (orange)
        if len(jp) == 5:
            last_joint = jp[3]
            ee_pos = jp[4]
            if np.linalg.norm(np.array(ee_pos) - np.array(last_joint)) > 0.001:
                self.ax.plot([last_joint[0], ee_pos[0]], [last_joint[1], ee_pos[1]],
                             [last_joint[2], ee_pos[2]], '-', color='orange', linewidth=2.5,
                             label='EE offset')
                self.ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], c='orange', s=40, marker='D')

    def _draw_target(self):
        pose = getattr(self, '_target_pose', None)
        if pose is None:
            return
        x, y, z, rot_y = pose
        pitch = -math.radians(rot_y)
        dx = 0.05 * math.cos(pitch)
        dz = 0.05 * math.sin(pitch)
        self.ax.scatter([x], [y], [z], c='red', s=40, label='Target')
        self.ax.quiver(x, y, z, dx, 0.0, dz, length=1.0, normalize=False,
                       arrow_length_ratio=0.2, color='red')
