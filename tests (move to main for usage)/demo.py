"""
Signal Processor Demo GUI

Interactive demonstration of the signal processing chain used in PWM valve control:
1. Normalization - maps raw input to [-1, 1]
2. Deadzone - ignores small inputs
3. Gamma shaping - power curve for valve flow compensation
4. Deadband - compresses command range to skip dead zone
5. Ramp limiting - slew rate limiting to smooth step inputs
6. Dither - vibration to prevent valve stiction

Mimics the control hierarchy from PCA9685_controller.py
"""

import math
import time
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass, field
from typing import Tuple, Optional
from collections import deque


@dataclass
class InputProcessor:
    input_min: float = -1.0
    input_max: float = 1.0
    deadzone_percent: float = 0.0
    gamma: float = 1.0
    direction: int = 1
    value_center: float = 1500.0
    value_min: float = 1000.0
    value_max: float = 2000.0
    deadband_pos: float = 0.0
    deadband_neg: float = 0.0
    dither_amp_us: float = 8.0 # set to 0.0 to disable dither
    dither_hz: float = 40.0
    dither_taper: bool = False
    ramp_limit: float = 0.0 # set to 0.0 to disable ramping
    s_curve_enable: bool = False
    accel_limit: float = 0.0 # us/s^2; used when s_curve_enable is True

    @property
    def deadzone_threshold(self) -> float:
        # Match PCA9685_controller: deadzone is percent of full-scale input
        return (self.deadzone_percent / 100.0) * 2.0

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))

    def normalize(self, raw_value: float) -> float:
        clamped = self.clamp(raw_value, self.input_min, self.input_max)
        return ((clamped - self.input_min) / (self.input_max - self.input_min)) * 2.0 - 1.0

    def apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone_threshold:
            return 0.0
        return value

    def apply_gamma(self, value: float) -> float:
        """
        Valve flow is often proportional to spool * sqrt(delta_p).
        Using gamma = 0.5 applies inverse sqrt(delta_p)-type compensation
        to reduce load-pressure-dependent gain and improve velocity linearity.
        """
        if self.gamma == 1.0 or value == 0.0:
            return value
        sign = 1.0 if value >= 0 else -1.0
        return sign * (abs(value) ** self.gamma)

    def apply_direction(self, value: float) -> float:
        # small check that allows user to set wonky direction values
        return value if self.direction > 0 else -value

    def apply_deadband(self,
                       value: float) -> float:
        s = self.apply_direction(self.clamp(value, -1.0, 1.0))

        if s == 0.0:
            return self.value_center
        if s > 0.0:
            base = self.value_center + self.deadband_pos
            working_range = self.value_max - base
            return base + abs(s) * working_range

        # s < 0.0
        base = self.value_center - self.deadband_neg
        working_range = base - self.value_min
        return base - abs(s) * working_range

    def to_pulse(self, raw_value: float) -> float:
        value = self.normalize(raw_value)
        value = self.apply_deadzone(value)
        value = self.apply_gamma(value)
        return self.apply_deadband(value)

    def apply_dither(self,
                     pulse: float,
                     value: float,
                     timestamp: float,
                     channel_index: int = 0) -> float:
        if self.dither_amp_us == 0.0 or self.dither_hz == 0.0:
            return pulse

        # remove idle dither
        if abs(value) < self.deadzone_threshold:
            return pulse

        phase_offset = channel_index * (math.pi / 3.0)
        phase = 2.0 * math.pi * self.dither_hz * timestamp + phase_offset
        if self.dither_taper:
            headroom = min(pulse - self.value_min, self.value_max - pulse)
            allowed_amp = max(0.0, min(self.dither_amp_us, headroom))
            dither = allowed_amp * math.sin(phase)
        else:
            dither = self.dither_amp_us * math.sin(phase)

        return pulse + dither

    def apply_ramp(self,
                   target_pulse: float,
                   last_pulse: float,
                   last_time: float,
                   current_time: float,
                   last_dt: float = 0.0,
                   last_vel: float = 0.0) -> Tuple[float, float, float]:
        if self.ramp_limit <= 0.0:
            return target_pulse, 0.0, 0.0

        dt_raw = current_time - last_time
        if last_dt > 0.0:
            dt = min(dt_raw, last_dt * 2.0)
        else:
            dt = dt_raw

        if dt <= 0.0:
            return last_pulse, dt_raw, last_vel

        if self.s_curve_enable and self.accel_limit > 0.0 and self.ramp_limit > 0.0:
            v_max = self.ramp_limit
            a_max = self.accel_limit
            target_vel = (target_pulse - last_pulse) / dt
            target_vel = max(-v_max, min(v_max, target_vel))
            if last_vel > 0.0:
                dist_to_limit = self.value_max - last_pulse
            elif last_vel < 0.0:
                dist_to_limit = last_pulse - self.value_min
            else:
                dist_to_limit = float("inf")
            if dist_to_limit <= (last_vel * last_vel) / (2.0 * a_max):
                target_vel = 0.0
            dv_max = a_max * dt
            dv = target_vel - last_vel
            if dv > dv_max:
                new_vel = last_vel + dv_max
            elif dv < -dv_max:
                new_vel = last_vel - dv_max
            else:
                new_vel = target_vel
            new_pulse = last_pulse + new_vel * dt
            if new_pulse < self.value_min:
                new_pulse = self.value_min
                new_vel = 0.0
            elif new_pulse > self.value_max:
                new_pulse = self.value_max
                new_vel = 0.0
            return new_pulse, dt_raw, new_vel

        allowed_step = self.ramp_limit * dt
        delta = target_pulse - last_pulse

        if abs(delta) <= allowed_step:
            new_pulse = target_pulse
        else:
            new_pulse = last_pulse + allowed_step * (1.0 if delta > 0 else -1.0)
        if new_pulse < self.value_min:
            new_pulse = self.value_min
        elif new_pulse > self.value_max:
            new_pulse = self.value_max

        return new_pulse, dt_raw, 0.0


class PlotCanvas(tk.Canvas):
    """Custom canvas for plotting time series data."""

    def __init__(self, parent, width=600, height=300, **kwargs):
        super().__init__(parent, width=width, height=height, bg='white', **kwargs)
        self.width = width
        self.height = height
        self.margin = {'left': 60, 'right': 20, 'top': 20, 'bottom': 40}

        # Data storage
        self.max_points = 500
        self.input_data: deque = deque(maxlen=self.max_points)
        self.output_data: deque = deque(maxlen=self.max_points)
        self.time_data: deque = deque(maxlen=self.max_points)

        # Y-axis range (pulse values in us)
        self.y_min = 1000.0
        self.y_max = 2000.0

        self.bind('<Configure>', self._on_resize)

    def _on_resize(self, event):
        self.width = event.width
        self.height = event.height

    def set_y_range(self, y_min: float, y_max: float):
        self.y_min = y_min
        self.y_max = y_max

    def add_point(self, t: float, input_pulse: float, output_pulse: float):
        self.time_data.append(t)
        self.input_data.append(input_pulse)
        self.output_data.append(output_pulse)

    def clear_data(self):
        self.input_data.clear()
        self.output_data.clear()
        self.time_data.clear()

    def redraw(self):
        self.delete('all')

        plot_left = self.margin['left']
        plot_right = self.width - self.margin['right']
        plot_top = self.margin['top']
        plot_bottom = self.height - self.margin['bottom']
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top

        # Background
        self.create_rectangle(plot_left, plot_top, plot_right, plot_bottom,
                            fill='#f8f8f8', outline='#ccc')

        # Center line
        center_y = plot_top + plot_height * (self.y_max - 1500) / (self.y_max - self.y_min)
        self.create_line(plot_left, center_y, plot_right, center_y,
                        fill='#ddd', dash=(4, 4))

        # Grid lines and labels
        for i in range(5):
            y_val = self.y_min + (self.y_max - self.y_min) * i / 4
            y_pos = plot_bottom - plot_height * (y_val - self.y_min) / (self.y_max - self.y_min)
            self.create_line(plot_left, y_pos, plot_right, y_pos, fill='#eee')
            self.create_text(plot_left - 5, y_pos, text=f'{y_val:.0f}',
                           anchor='e', font=('Arial', 8))

        # Axis labels
        self.create_text(15, (plot_top + plot_bottom) / 2, text='Pulse (us)',
                        anchor='center', angle=90, font=('Arial', 9))
        self.create_text((plot_left + plot_right) / 2, self.height - 5,
                        text='Time', anchor='center', font=('Arial', 9))

        if len(self.time_data) < 2:
            return

        t_max = self.time_data[-1]
        t_min = t_max - 5.0

        def to_screen(t, val):
            x = plot_left + plot_width * (t - t_min) / (t_max - t_min + 0.001)
            y = plot_bottom - plot_height * (val - self.y_min) / (self.y_max - self.y_min)
            return x, y

        # Input line (blue, dashed)
        input_coords = []
        for t, val in zip(self.time_data, self.input_data):
            if t >= t_min:
                x, y = to_screen(t, val)
                input_coords.extend([x, y])

        if len(input_coords) >= 4:
            self.create_line(input_coords, fill='#2196F3', width=2, dash=(6, 3))

        # Output line (red, solid)
        output_coords = []
        for t, val in zip(self.time_data, self.output_data):
            if t >= t_min:
                x, y = to_screen(t, val)
                output_coords.extend([x, y])

        if len(output_coords) >= 4:
            self.create_line(output_coords, fill='#F44336', width=2)

        # Legend
        legend_x = plot_right - 120
        legend_y = plot_top + 10
        self.create_line(legend_x, legend_y + 5, legend_x + 20, legend_y + 5,
                        fill='#2196F3', width=2, dash=(6, 3))
        self.create_text(legend_x + 25, legend_y + 5, text='Input',
                        anchor='w', font=('Arial', 9))
        self.create_line(legend_x, legend_y + 20, legend_x + 20, legend_y + 20,
                        fill='#F44336', width=2)
        self.create_text(legend_x + 25, legend_y + 20, text='Output',
                        anchor='w', font=('Arial', 9))


class SignalProcessorDemo:
    """Main GUI application demonstrating the signal processing chain."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Signal Processor Demo")
        self.root.geometry("850x750")

        # Create processor with default settings
        self.processor = InputProcessor()

        # Ramp state tracking
        self._last_pulse = self.processor.value_center
        self._last_time = time.time()
        self._last_dt = 0.0
        self._last_vel = 0.0

        # State
        self.input_value = 0.0
        self.running = True
        self.refresh_rate = 60
        self.start_time = time.time()

        self._build_ui()
        self._update_loop()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)

        # Demo tab
        self.main_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.main_tab, text="Demo")

        # Settings tab
        self.settings_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.settings_tab, text="Settings")

        self._build_main_tab()
        self._build_settings_tab()

    def _build_main_tab(self):
        # Plot frame
        plot_frame = ttk.LabelFrame(self.main_tab, text="Signal Plot", padding="5")
        plot_frame.pack(fill='both', expand=True, pady=(0, 10))

        self.plot = PlotCanvas(plot_frame, width=700, height=280)
        self.plot.pack(fill='both', expand=True)
        self._update_plot_range()

        # Input slider
        slider_frame = ttk.LabelFrame(self.main_tab, text="Input Control", padding="10")
        slider_frame.pack(fill='x', pady=(0, 10))

        self.input_var = tk.DoubleVar(value=0.0)

        slider_container = ttk.Frame(slider_frame)
        slider_container.pack(fill='x')

        ttk.Label(slider_container, text="-1.0").pack(side='left')
        self.input_slider = ttk.Scale(
            slider_container, from_=-1.0, to=1.0,
            variable=self.input_var, orient='horizontal',
            command=self._on_input_change
        )
        self.input_slider.pack(side='left', fill='x', expand=True, padx=10)
        ttk.Label(slider_container, text="+1.0").pack(side='left')

        self.input_label = ttk.Label(slider_frame, text="Input: 0.000",
                                        font=('Arial', 12, 'bold'))
        self.input_label.pack(pady=(5, 0))

        # Processing stages display
        values_frame = ttk.LabelFrame(self.main_tab, text="Processing Stages", padding="10")
        values_frame.pack(fill='x', pady=(0, 10))

        self.values_text = tk.Text(values_frame, height=7, width=80, font=('Consolas', 10))
        self.values_text.pack(fill='x')
        self.values_text.config(state='disabled')

        # Control buttons
        btn_frame = ttk.Frame(self.main_tab)
        btn_frame.pack(fill='x')

        ttk.Button(btn_frame, text="Reset Input",
                  command=self._reset_input).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Clear Plot",
                  command=self._clear_plot).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Reset Ramp State",
                  command=self._reset_ramp).pack(side='left', padx=5)

    def _build_settings_tab(self):
        # Scrollable frame
        canvas = tk.Canvas(self.settings_tab)
        scrollbar = ttk.Scrollbar(self.settings_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # GUI Settings
        gui_frame = ttk.LabelFrame(scrollable_frame, text="GUI Settings", padding="10")
        gui_frame.pack(fill='x', pady=5, padx=5)

        ttk.Label(gui_frame, text="Refresh Rate (Hz):").grid(row=0, column=0, sticky='w', pady=2)
        self.refresh_rate_var = tk.IntVar(value=self.refresh_rate)
        refresh_spin = ttk.Spinbox(gui_frame, from_=10, to=120, textvariable=self.refresh_rate_var,
                                   width=10, command=self._on_refresh_rate_change)
        refresh_spin.grid(row=0, column=1, sticky='w', pady=2, padx=10)
        refresh_spin.bind('<Return>', lambda e: self._on_refresh_rate_change())

        # Input Range
        input_frame = ttk.LabelFrame(scrollable_frame, text="Input Range", padding="10")
        input_frame.pack(fill='x', pady=5, padx=5)

        ttk.Label(input_frame, text="Input Min:").grid(row=0, column=0, sticky='w', pady=2)
        self.input_min_var = tk.DoubleVar(value=self.processor.input_min)
        ttk.Entry(input_frame, textvariable=self.input_min_var, width=10).grid(row=0, column=1, pady=2, padx=10)

        ttk.Label(input_frame, text="Input Max:").grid(row=1, column=0, sticky='w', pady=2)
        self.input_max_var = tk.DoubleVar(value=self.processor.input_max)
        ttk.Entry(input_frame, textvariable=self.input_max_var, width=10).grid(row=1, column=1, pady=2, padx=10)

        # Pulse Range
        range_frame = ttk.LabelFrame(scrollable_frame, text="Pulse Range (us)", padding="10")
        range_frame.pack(fill='x', pady=5, padx=5)

        ttk.Label(range_frame, text="Minimum:").grid(row=0, column=0, sticky='w', pady=2)
        self.pulse_min_var = tk.DoubleVar(value=self.processor.value_min)
        ttk.Entry(range_frame, textvariable=self.pulse_min_var, width=10).grid(row=0, column=1, pady=2, padx=10)

        ttk.Label(range_frame, text="Maximum:").grid(row=1, column=0, sticky='w', pady=2)
        self.pulse_max_var = tk.DoubleVar(value=self.processor.value_max)
        ttk.Entry(range_frame, textvariable=self.pulse_max_var, width=10).grid(row=1, column=1, pady=2, padx=10)

        ttk.Label(range_frame, text="Center:").grid(row=2, column=0, sticky='w', pady=2)
        self.center_var = tk.DoubleVar(value=self.processor.value_center)
        ttk.Entry(range_frame, textvariable=self.center_var, width=10).grid(row=2, column=1, pady=2, padx=10)

        # Deadzone
        deadzone_frame = ttk.LabelFrame(scrollable_frame, text="Deadzone (percent of full scale)", padding="10")
        deadzone_frame.pack(fill='x', pady=5, padx=5)

        ttk.Label(deadzone_frame, text="Threshold:").grid(row=0, column=0, sticky='w', pady=2)
        self.deadzone_var = tk.DoubleVar(value=self.processor.deadzone_percent)
        deadzone_scale = ttk.Scale(deadzone_frame, from_=0.0, to=100.0, variable=self.deadzone_var,
                                   orient='horizontal', length=200)
        deadzone_scale.grid(row=0, column=1, pady=2, padx=10)
        self.deadzone_label = ttk.Label(deadzone_frame, text=f"{self.processor.deadzone_percent:.1f}%")
        self.deadzone_label.grid(row=0, column=2, pady=2)
        deadzone_scale.bind('<Motion>', lambda e: self.deadzone_label.config(text=f"{self.deadzone_var.get():.1f}%"))

        # Gamma
        gamma_frame = ttk.LabelFrame(scrollable_frame, text="Gamma Shaping", padding="10")
        gamma_frame.pack(fill='x', pady=5, padx=5)

        ttk.Label(gamma_frame, text="Gamma (1.0 = linear):").grid(row=0, column=0, sticky='w', pady=2)
        self.gamma_var = tk.DoubleVar(value=self.processor.gamma)
        gamma_scale = ttk.Scale(gamma_frame, from_=0.1, to=3.0, variable=self.gamma_var,
                               orient='horizontal', length=200)
        gamma_scale.grid(row=0, column=1, pady=2, padx=10)
        self.gamma_label = ttk.Label(gamma_frame, text=f"{self.processor.gamma:.2f}")
        self.gamma_label.grid(row=0, column=2, pady=2)
        gamma_scale.bind('<Motion>', lambda e: self.gamma_label.config(text=f"{self.gamma_var.get():.2f}"))

        ttk.Label(gamma_frame, text="< 1: More sensitive near center\n> 1: More sensitive near extremes",
                 font=('Arial', 8)).grid(row=1, column=0, columnspan=3, sticky='w', pady=2)

        # Direction
        dir_frame = ttk.LabelFrame(scrollable_frame, text="Direction", padding="10")
        dir_frame.pack(fill='x', pady=5, padx=5)

        self.direction_var = tk.IntVar(value=self.processor.direction)
        ttk.Radiobutton(dir_frame, text="Normal (+1)", variable=self.direction_var,
                       value=1).pack(side='left', padx=10)
        ttk.Radiobutton(dir_frame, text="Inverted (-1)", variable=self.direction_var,
                       value=-1).pack(side='left', padx=10)

        # Deadband
        deadband_frame = ttk.LabelFrame(scrollable_frame, text="Deadband (us from center)", padding="10")
        deadband_frame.pack(fill='x', pady=5, padx=5)

        ttk.Label(deadband_frame, text="Positive direction:").grid(row=0, column=0, sticky='w', pady=2)
        self.deadband_pos_var = tk.DoubleVar(value=self.processor.deadband_pos)
        ttk.Entry(deadband_frame, textvariable=self.deadband_pos_var, width=10).grid(row=0, column=1, pady=2, padx=10)

        ttk.Label(deadband_frame, text="Negative direction:").grid(row=1, column=0, sticky='w', pady=2)
        self.deadband_neg_var = tk.DoubleVar(value=self.processor.deadband_neg)
        ttk.Entry(deadband_frame, textvariable=self.deadband_neg_var, width=10).grid(row=1, column=1, pady=2, padx=10)

        # Ramp
        ramp_frame = ttk.LabelFrame(scrollable_frame, text="Ramp / Slew Limiting", padding="10")
        ramp_frame.pack(fill='x', pady=5, padx=5)

        ttk.Label(ramp_frame, text="Rate Limit (us/s):").grid(row=0, column=0, sticky='w', pady=2)
        self.ramp_limit_var = tk.DoubleVar(value=self.processor.ramp_limit)
        ttk.Entry(ramp_frame, textvariable=self.ramp_limit_var, width=10).grid(row=0, column=1, pady=2, padx=10)
        ttk.Label(ramp_frame, text="(0 = disabled)").grid(row=0, column=2, sticky='w', pady=2)

        self.s_curve_var = tk.IntVar(value=1 if self.processor.s_curve_enable else 0)
        ttk.Checkbutton(ramp_frame, text="Enable S-curve", variable=self.s_curve_var).grid(row=1, column=0, sticky='w', pady=2)

        ttk.Label(ramp_frame, text="Accel Limit (us/s^2):").grid(row=2, column=0, sticky='w', pady=2)
        self.accel_limit_var = tk.DoubleVar(value=self.processor.accel_limit)
        ttk.Entry(ramp_frame, textvariable=self.accel_limit_var, width=10).grid(row=2, column=1, pady=2, padx=10)

        # Dither
        dither_frame = ttk.LabelFrame(scrollable_frame, text="Dither (Anti-Stiction)", padding="10")
        dither_frame.pack(fill='x', pady=5, padx=5)

        ttk.Label(dither_frame, text="Amplitude (us):").grid(row=0, column=0, sticky='w', pady=2)
        self.dither_amp_var = tk.DoubleVar(value=self.processor.dither_amp_us)
        ttk.Entry(dither_frame, textvariable=self.dither_amp_var, width=10).grid(row=0, column=1, pady=2, padx=10)
        ttk.Label(dither_frame, text="(0 = disabled)").grid(row=0, column=2, sticky='w', pady=2)

        ttk.Label(dither_frame, text="Frequency (Hz):").grid(row=1, column=0, sticky='w', pady=2)
        self.dither_hz_var = tk.DoubleVar(value=self.processor.dither_hz)
        ttk.Entry(dither_frame, textvariable=self.dither_hz_var, width=10).grid(row=1, column=1, pady=2, padx=10)

        self.dither_taper_var = tk.IntVar(value=1 if self.processor.dither_taper else 0)
        ttk.Checkbutton(dither_frame, text="Taper near limits", variable=self.dither_taper_var).grid(row=2, column=0, sticky='w', pady=2)

        # Buttons
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill='x', pady=10, padx=5)

        ttk.Button(btn_frame, text="Apply Settings",
                  command=self._apply_settings).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Reset to Defaults",
                  command=self._reset_settings).pack(side='left', padx=5)

    def _on_input_change(self, value=None):
        self.input_value = self.input_var.get()
        self.input_label.config(text=f"Input: {self.input_value:.3f}")

    def _on_refresh_rate_change(self):
        try:
            self.refresh_rate = max(10, min(120, self.refresh_rate_var.get()))
        except:
            pass

    def _apply_settings(self):
        try:
            self.processor.input_min = self.input_min_var.get()
            self.processor.input_max = self.input_max_var.get()
            self.processor.value_min = self.pulse_min_var.get()
            self.processor.value_max = self.pulse_max_var.get()
            self.processor.value_center = self.center_var.get()
            self.processor.deadzone_percent = self.deadzone_var.get()
            self.processor.gamma = self.gamma_var.get()
            self.processor.direction = self.direction_var.get()
            self.processor.deadband_pos = self.deadband_pos_var.get()
            self.processor.deadband_neg = self.deadband_neg_var.get()
            self.processor.ramp_limit = self.ramp_limit_var.get()
            self.processor.s_curve_enable = bool(self.s_curve_var.get())
            self.processor.accel_limit = self.accel_limit_var.get()
            self.processor.dither_amp_us = self.dither_amp_var.get()
            self.processor.dither_hz = self.dither_hz_var.get()
            self.processor.dither_taper = bool(self.dither_taper_var.get())

            self._update_plot_range()
            self._reset_ramp()

        except Exception as e:
            print(f"Error applying settings: {e}")

    def _reset_settings(self):
        self.processor = InputProcessor()
        self._update_settings_ui()
        self._update_plot_range()
        self._reset_ramp()

    def _update_settings_ui(self):
        self.input_min_var.set(self.processor.input_min)
        self.input_max_var.set(self.processor.input_max)
        self.pulse_min_var.set(self.processor.value_min)
        self.pulse_max_var.set(self.processor.value_max)
        self.center_var.set(self.processor.value_center)
        self.deadzone_var.set(self.processor.deadzone_percent)
        self.deadzone_label.config(text=f"{self.processor.deadzone_percent:.1f}%")
        self.gamma_var.set(self.processor.gamma)
        self.gamma_label.config(text=f"{self.processor.gamma:.2f}")
        self.direction_var.set(self.processor.direction)
        self.deadband_pos_var.set(self.processor.deadband_pos)
        self.deadband_neg_var.set(self.processor.deadband_neg)
        self.ramp_limit_var.set(self.processor.ramp_limit)
        self.s_curve_var.set(1 if self.processor.s_curve_enable else 0)
        self.accel_limit_var.set(self.processor.accel_limit)
        self.dither_amp_var.set(self.processor.dither_amp_us)
        self.dither_hz_var.set(self.processor.dither_hz)
        self.dither_taper_var.set(1 if self.processor.dither_taper else 0)

    def _reset_input(self):
        self.input_var.set(0.0)
        self.input_value = 0.0
        self.input_label.config(text="Input: 0.000")

    def _clear_plot(self):
        self.plot.clear_data()
        self.start_time = time.time()

    def _reset_ramp(self):
        self._last_pulse = self.processor.value_center
        self._last_time = time.time()
        self._last_dt = 0.0
        self._last_vel = 0.0

    def _update_plot_range(self):
        if self.processor.dither_taper:
            self.plot.set_y_range(self.processor.value_min, self.processor.value_max)
        else:
            amp = max(0.0, float(self.processor.dither_amp_us))
            self.plot.set_y_range(self.processor.value_min - amp, self.processor.value_max + amp)

    def _update_loop(self):
        if not self.running:
            return

        now = time.time()
        t = now - self.start_time

        # Process through the signal chain
        intermediates = {}

        # Step 1: Normalize
        normalized = self.processor.normalize(self.input_value)
        intermediates['normalized'] = normalized

        # Step 2: Deadzone
        after_deadzone = self.processor.apply_deadzone(normalized)
        intermediates['after_deadzone'] = after_deadzone

        # Step 3: Gamma
        after_gamma = self.processor.apply_gamma(after_deadzone)
        intermediates['after_gamma'] = after_gamma

        # Step 4: Deadband (to pulse)
        base_pulse = self.processor.apply_deadband(after_gamma)
        intermediates['after_deadband'] = base_pulse

        # Step 5: Ramp limiting
        ramped_pulse, new_dt, new_vel = self.processor.apply_ramp(
            base_pulse, self._last_pulse, self._last_time, now, self._last_dt, self._last_vel
        )
        self._last_pulse = ramped_pulse
        self._last_time = now
        if new_dt > 0:
            self._last_dt = new_dt
        self._last_vel = new_vel
        intermediates['after_ramp'] = ramped_pulse

        # Step 6: Dither
        final_pulse = self.processor.apply_dither(ramped_pulse, after_deadzone, now)
        intermediates['after_dither'] = final_pulse

        # Clamp final (expanded when dither is active to preserve full amplitude)
        if self.processor.dither_amp_us > 0.0 and abs(after_deadzone) >= self.processor.deadzone_threshold:
            if self.processor.dither_taper:
                min_pulse = self.processor.value_min
                max_pulse = self.processor.value_max
            else:
                amp = self.processor.dither_amp_us
                min_pulse = self.processor.value_min - amp
                max_pulse = self.processor.value_max + amp
        else:
            min_pulse = self.processor.value_min
            max_pulse = self.processor.value_max
        final_pulse = max(min_pulse, min(max_pulse, final_pulse))
        intermediates['final'] = final_pulse

        # Calculate input pulse for comparison (no ramp/dither)
        input_pulse = self.processor.to_pulse(self.input_value)

        # Add to plot
        self.plot.add_point(t, input_pulse, final_pulse)
        self.plot.redraw()

        # Update display
        self._update_values_display(intermediates)

        # Schedule next
        interval_ms = max(1, int(1000 / self.refresh_rate))
        self.root.after(interval_ms, self._update_loop)

    def _update_values_display(self, intermediates: dict):
        self.values_text.config(state='normal')
        self.values_text.delete('1.0', tk.END)

        lines = [
            f"1. Input:          {self.input_value:.4f}",
            f"2. Normalized:     {intermediates.get('normalized', 0):.4f}",
            f"3. After Deadzone: {intermediates.get('after_deadzone', 0):.4f}",
            f"4. After Gamma:    {intermediates.get('after_gamma', 0):.4f}",
            f"5. After Deadband: {intermediates.get('after_deadband', 0):.2f} us",
            f"6. After Ramp:     {intermediates.get('after_ramp', 0):.2f} us",
            f"7. After Dither:   {intermediates.get('after_dither', 0):.2f} us  (Final Output)",
        ]

        self.values_text.insert('1.0', '\n'.join(lines))
        self.values_text.config(state='disabled')

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.running = False
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SignalProcessorDemo(root)
    app.run()


if __name__ == "__main__":
    main()
