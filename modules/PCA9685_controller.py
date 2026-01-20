"""
Valve testing PWM controller with:
- Derived PWM period from actual PCA9685 frequency
- Simple deadband: compresses command range to skip dead zone (linear throughout!)
- Optional dither (per-channel) to prevent valve stiction
- Optional per-channel ramp/slew limiting to smooth step inputs

Maintains linearity by packing commands into the working range around the dead zone.
"""

import atexit
import threading
import time
import logging
import yaml
import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from pathlib import Path

from adafruit_pca9685 import PCA9685
import board
import busio


# ============================================================================
# Direct I2C PWM Writer (optimized for real-time performance)
# ============================================================================

class DirectPWMWriter:
    """Direct I2C writer for PCA9685 using pre-allocated buffer.

    Writes only configured channels in a single I2C transaction, eliminating
    per-channel allocation and lock overhead from the Adafruit library.

    At 1MHz I2C: 21 bytes (5 channels) takes ~250µs vs ~540µs with Adafruit.
    """

    LED0_ON_L = 0x06  # First PWM register (channel 0)

    def __init__(self, i2c_bus, address: int = 0x40,
                 min_channel: int = 0, max_channel: int = 15):
        """Initialize the direct PWM writer.

        Args:
            i2c_bus: The I2C bus instance
            address: PCA9685 I2C address (default 0x40)
            min_channel: Lowest channel number to write (0-15)
            max_channel: Highest channel number to write (0-15)
        """
        self._i2c = i2c_bus
        self._addr = address
        self._min_ch = min_channel
        self._max_ch = max_channel
        self._num_channels = max_channel - min_channel + 1

        # Pre-allocated buffer: 1 byte register addr + 4 bytes per channel
        self._buf = bytearray(1 + 4 * self._num_channels)
        self._buf[0] = self.LED0_ON_L + (min_channel * 4)  # Start register
        self._duty_cycles = [0] * 16  # Full array; only min_ch..max_ch are written

    def set_channel(self, channel: int, duty_cycle: int):
        """Queue a duty cycle update (0-65535).

        Note: Only channels within [min_channel, max_channel] are written by flush().
        # TODO: Add bounds check `if 0 <= channel < 16` to prevent IndexError on invalid channel.
        #       Skipped for now since PWMController validates channels at config load time.
        """
        self._duty_cycles[channel] = duty_cycle

    def set_channel_range(self, min_channel: int, max_channel: int):
        """Update the channel range. For benchmarking only - allocates memory."""
        self._min_ch = min_channel
        self._max_ch = max_channel
        self._num_channels = max_channel - min_channel + 1
        self._buf = bytearray(1 + 4 * self._num_channels)
        self._buf[0] = self.LED0_ON_L + (min_channel * 4)

    def flush(self):
        """Write configured channels in a single I2C transaction."""
        buf = self._buf
        duty_cycles = self._duty_cycles
        min_ch = self._min_ch

        for i in range(self._num_channels):
            duty = duty_cycles[min_ch + i]
            # Convert 16-bit duty to PCA9685 ON/OFF register values
            if duty >= 0xFFF0:
                # Fully on: set bit 4 of ON_H
                on_val, off_val = 0x1000, 0
            elif duty < 0x0010:
                # Fully off: set bit 4 of OFF_H
                on_val, off_val = 0, 0x1000
            else:
                # Normal PWM: ON at 0, OFF at duty>>4 (12-bit)
                on_val = 0
                off_val = duty >> 4

            offset = 1 + i * 4
            buf[offset] = on_val & 0xFF
            buf[offset + 1] = (on_val >> 8) & 0xFF
            buf[offset + 2] = off_val & 0xFF
            buf[offset + 3] = (off_val >> 8) & 0xFF

        while not self._i2c.try_lock():
            pass
        try:
            self._i2c.writeto(self._addr, buf)
        finally:
            self._i2c.unlock()


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class ChannelConfig:
    """Configuration for a single PWM channel."""
    output_channel: int
    pulse_min: int
    pulse_max: int
    direction: int  # +1 or -1; maps sign of input to physical pulse side
    center: Optional[float] = None
    deadzone: float = 0.0  # percent of full-scale input mapped to zero (rarely used here)
    affects_pump: bool = False
    toggleable: bool = False

    # Simple deadband (separate per sign) - jumps over dead zone by compressing command range
    # Offsets from center (us) where the working range starts for positive/negative inputs
    deadband_us_pos: float = 0.0
    deadband_us_neg: float = 0.0

    # Dither settings to prevent valve stiction - DISABLED by default
    dither_enable: bool = False
    dither_amp_us: float = 8.0  # vibration amplitude in microseconds
    dither_hz: float = 40.0  # vibration frequency
    dither_taper: bool = False  # taper near pulse bounds instead of expanding clamp

    # Slew rate limiting (microseconds per second) to soften command steps
    ramp_enable: bool = False
    ramp_limit: float = 0.0  # us/s; ignored when ramp_enable is False
    # Optional S-curve ramping (jerk-limited) using accel limit on rate changes
    s_curve_enable: bool = False
    accel_limit: float = 0.0  # us/s^2; used when s_curve_enable is True

    # Symmetric gamma shaping (1.0 = linear). Applied to magnitude for both directions.
    gamma: float = 1.0

    def __post_init__(self):
        self.pulse_range = self.pulse_max - self.pulse_min
        if self.center is None:
            self.center = self.pulse_min + (self.pulse_range / 2)
        self.deadzone_threshold = self.deadzone / 100.0 * 2
        self._dither_omega = 2.0 * math.pi * float(self.dither_hz)
        self._dither_phase_offset = self.output_channel * 1.0471975512


@dataclass
class PumpConfig:
    """Configuration specific to pump control (not typically used in valve_testing)."""
    output_channel: int
    pulse_min: int
    pulse_max: int
    idle: float
    multiplier: float
    # Manual pump via input channel removed in testing controller.


class PWMConstants:
    """Hardware and timing constants."""
    PWM_FREQUENCY_DEFAULT = 200  # 50 Hz standard, now 2x input ish.
    MAX_CHANNELS = 16
    DUTY_CYCLE_MAX = 65535
    RAMP_DT_MAX = 0.05

    # Validation limits
    PULSE_MIN = 0
    PULSE_MAX = 4095
    PUMP_IDLE_MIN = -1.0
    PUMP_IDLE_MAX = 0.6
    PUMP_MULTIPLIER_MAX = 1.0

    # Safety parameters
    DEFAULT_TIME_WINDOW = 5  # seconds
    SAFE_STATE_THRESHOLD = 0.25


class PWMController:
    """Simple PWM controller with piecewise deadband and dither for valve testing."""

    def __init__(self, config_file: str, pump_variable: bool = False,
                 toggle_channels: bool = True, input_rate_threshold: float = 0,
                 default_unset_to_zero: bool = True, log_level: str = "INFO",
                 stale_timeout_s: float = 0.0, watchdog_channel: Optional[int] = None,
                 watchdog_toggle_hz: float = 0.0):
        """Initialize PWM controller.

        Args:
            config_file: Path to YAML configuration file
            pump_variable: Enable variable pump speed
            toggle_channels: Enable toggleable channels
            input_rate_threshold: Input rate threshold for safety monitoring
            default_unset_to_zero: Default unset channels to zero
            log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR"
        """
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.PWMController")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Add console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self._lock = threading.RLock()
        self.pump_variable = pump_variable
        self.toggle_channels = toggle_channels
        self.pump_enabled = True
        self.manual_pump_load = 0.0
        self.pump_variable_sum = 0.0
        self.pump_variable_count = 0
        self._pump_override_throttle: Optional[float] = None
        self._stale_timeout_s = max(0.0, float(stale_timeout_s))
        self._last_command_ts = time.monotonic()
        self._watchdog_channel = watchdog_channel
        self._watchdog_toggle_hz = max(0.0, float(watchdog_toggle_hz))
        self._watchdog_last_toggle = time.monotonic()
        self._watchdog_state = False

        # Rate monitoring (optional)
        self.input_rate_threshold = input_rate_threshold
        self.skip_rate_checking = (input_rate_threshold == 0)
        self.is_safe_state = not self.skip_rate_checking
        self.time_window = PWMConstants.DEFAULT_TIME_WINDOW

        # Threads/monitoring
        self.running = False
        self.input_event = threading.Event()
        self.monitor_thread = None
        self.input_count = 0
        self.last_input_time = time.time()

        # Load config
        self._load_config(config_file)

        # Hardware init
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = PWMConstants.PWM_FREQUENCY_DEFAULT
        self._pwm_period_us = 1e6 / float(self.pca.frequency)

        # Compute channel range for optimized partial writes
        all_channels = [cfg.output_channel for cfg in self.channel_configs.values()]
        if self.pump_config:
            all_channels.append(self.pump_config.output_channel)
        min_ch = min(all_channels) if all_channels else 0
        max_ch = max(all_channels) if all_channels else 15
        self._direct_writer = DirectPWMWriter(i2c, min_channel=min_ch, max_channel=max_ch)
        self.logger.info(f"DirectPWMWriter using channels {min_ch}-{max_ch} "
                        f"({max_ch - min_ch + 1} channels, {1 + 4*(max_ch-min_ch+1)} bytes)")

        # Current normalized values per channel
        self.values = [0.0] * PWMConstants.MAX_CHANNELS

        # Monitoring counters
        self.input_counter = 0
        self.rate_window_start = time.time()

        # Register simple cleanup and start monitoring
        atexit.register(self._simple_cleanup)
        self.reset()

        if not self.skip_rate_checking:
            self._start_monitoring()
        
        # Behavior defaults
        self._default_unset_to_zero = default_unset_to_zero
        self._affects_pump_channels = [cfg for cfg in self.channel_configs.values() if cfg.affects_pump]

    def _load_config(self, config_file: str):
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file '{config_file}' not found")
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        self.channel_configs, self.pump_config = self._parse_config(raw_config)
        self._validate_config()

    def _parse_config(self, raw_config: Dict) -> tuple[Dict[str, ChannelConfig], Optional[PumpConfig]]:
        channel_configs: Dict[str, ChannelConfig] = {}
        pump_config = None
        config_errors: List[str] = []

        def _err(scope: str, key: str, msg: str) -> None:
            config_errors.append(f"{scope}: {key} {msg}")

        def _as_int(scope: str, key: str, value: Any) -> Optional[int]:
            if isinstance(value, bool):
                _err(scope, key, "must be int (got bool)")
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            _err(scope, key, f"must be int (got {type(value).__name__})")
            return None

        def _as_float(scope: str, key: str, value: Any) -> Optional[float]:
            if isinstance(value, bool):
                _err(scope, key, "must be number (got bool)")
                return None
            if isinstance(value, (int, float)):
                return float(value)
            _err(scope, key, f"must be number (got {type(value).__name__})")
            return None

        def _as_bool(scope: str, key: str, value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            if isinstance(value, int) and value in (0, 1):
                return bool(value)
            _err(scope, key, f"must be bool (got {type(value).__name__})")
            return None

        required_pump_keys = [
            'output_channel', 'pulse_min', 'pulse_max', 'idle', 'multiplier'
        ]
        required_channel_keys = [
            'output_channel', 'pulse_min', 'pulse_max', 'direction', 'center',
            'deadzone', 'affects_pump', 'toggleable', 'deadband_us_pos', 'deadband_us_neg',
            'dither_enable', 'dither_amp_us', 'dither_hz', 'dither_taper',
            'ramp_enable', 'ramp_limit', 's_curve_enable', 'accel_limit', 'gamma'
        ]

        for name, cfg in raw_config['CHANNEL_CONFIGS'].items():
            if name == 'pump':
                missing = [k for k in required_pump_keys if k not in cfg]
                if missing:
                    config_errors.append(
                        f"Pump config missing required keys: {', '.join(missing)}"
                    )
                    continue
                scope = "Pump"
                output_channel = _as_int(scope, 'output_channel', cfg['output_channel'])
                pulse_min = _as_int(scope, 'pulse_min', cfg['pulse_min'])
                pulse_max = _as_int(scope, 'pulse_max', cfg['pulse_max'])
                idle = _as_float(scope, 'idle', cfg['idle'])
                multiplier = _as_float(scope, 'multiplier', cfg['multiplier'])
                if config_errors:
                    continue
                pump_config = PumpConfig(
                    output_channel=output_channel,
                    pulse_min=pulse_min,
                    pulse_max=pulse_max,
                    idle=idle,
                    multiplier=multiplier,
                )
            else:
                missing = [k for k in required_channel_keys if k not in cfg]
                if missing:
                    config_errors.append(
                        f"Channel '{name}' missing required keys: {', '.join(missing)}"
                    )
                    continue
                scope = f"Channel '{name}'"
                output_channel = _as_int(scope, 'output_channel', cfg['output_channel'])
                pulse_min = _as_int(scope, 'pulse_min', cfg['pulse_min'])
                pulse_max = _as_int(scope, 'pulse_max', cfg['pulse_max'])
                direction = _as_int(scope, 'direction', cfg['direction'])
                center = self._normalize_none(cfg['center'])
                if center is not None:
                    center = _as_float(scope, 'center', center)
                deadzone = _as_float(scope, 'deadzone', cfg['deadzone'])
                affects_pump = _as_bool(scope, 'affects_pump', cfg['affects_pump'])
                toggleable = _as_bool(scope, 'toggleable', cfg['toggleable'])
                deadband_us_pos = _as_float(scope, 'deadband_us_pos', cfg['deadband_us_pos'])
                deadband_us_neg = _as_float(scope, 'deadband_us_neg', cfg['deadband_us_neg'])
                dither_enable = _as_bool(scope, 'dither_enable', cfg['dither_enable'])
                dither_amp_us = _as_float(scope, 'dither_amp_us', cfg['dither_amp_us'])
                dither_hz = _as_float(scope, 'dither_hz', cfg['dither_hz'])
                dither_taper = _as_bool(scope, 'dither_taper', cfg['dither_taper'])
                ramp_enable = _as_bool(scope, 'ramp_enable', cfg['ramp_enable'])
                ramp_limit = _as_float(scope, 'ramp_limit', cfg['ramp_limit'])
                s_curve_enable = _as_bool(scope, 's_curve_enable', cfg['s_curve_enable'])
                accel_limit = _as_float(scope, 'accel_limit', cfg['accel_limit'])
                gamma = _as_float(scope, 'gamma', cfg['gamma'])
                if config_errors:
                    continue
                channel_configs[name] = ChannelConfig(
                    output_channel=output_channel,
                    pulse_min=pulse_min,
                    pulse_max=pulse_max,
                    direction=direction,
                    center=center,
                    deadzone=deadzone,
                    affects_pump=affects_pump,
                    toggleable=toggleable,
                    # Simple deadband (separate per sign)
                    deadband_us_pos=float(deadband_us_pos),
                    deadband_us_neg=float(deadband_us_neg),
                    # Dither settings - opt-in only
                    dither_enable=dither_enable,
                    dither_amp_us=float(dither_amp_us),
                    dither_hz=float(dither_hz),
                    dither_taper=dither_taper,
                    # Ramp/slew limiting - opt-in per channel
                    ramp_enable=ramp_enable,
                    ramp_limit=float(ramp_limit),
                    # S-curve (jerk-limited) ramping - opt-in per channel
                    s_curve_enable=s_curve_enable,
                    accel_limit=float(accel_limit),
                    # Symmetric gamma shaping
                    gamma=float(gamma),
                )

        if config_errors:
            raise ValueError("Invalid PWM config:\n- " + "\n- ".join(config_errors))

        return channel_configs, pump_config

    @staticmethod
    def _normalize_none(value: Any) -> Optional[Any]:
        none_values = [None, "None", "none", "null", "NONE", "Null", "NULL", "", "n/a", "N/A"]
        return None if value in none_values else value

    def _validate_config(self):
        errors = []
        used_outputs = {}

        for name, config in self.channel_configs.items():
            if config.direction not in [-1, 1]:
                errors.append(f"Channel '{name}': direction must be -1 or 1")

            if config.output_channel in used_outputs:
                errors.append(f"Channel '{name}': output {config.output_channel} already used")
            elif not 0 <= config.output_channel < PWMConstants.MAX_CHANNELS:
                errors.append(f"Channel '{name}': output must be 0-{PWMConstants.MAX_CHANNELS - 1}")
            else:
                used_outputs[config.output_channel] = name

            if not PWMConstants.PULSE_MIN <= config.pulse_min <= PWMConstants.PULSE_MAX:
                errors.append(f"Channel '{name}': pulse_min out of range")
            if not PWMConstants.PULSE_MIN <= config.pulse_max <= PWMConstants.PULSE_MAX:
                errors.append(f"Channel '{name}': pulse_max out of range")
            if config.pulse_min >= config.pulse_max:
                errors.append(f"Channel '{name}': pulse_min must be less than pulse_max")

            # Center sanity
            if config.center is not None and not (config.pulse_min <= float(config.center) <= config.pulse_max):
                errors.append(f"Channel '{name}': center must be within [pulse_min, pulse_max]")

            # Deadband and dither bounds
            rng = config.pulse_max - config.pulse_min
            # deadband_us_pos/neg should not exceed half of span and must be >=0
            if float(config.deadband_us_pos) < 0.0 or float(config.deadband_us_pos) > (rng * 0.5):
                errors.append(f"Channel '{name}': deadband_us_pos is unrealistic (0 .. {rng*0.5:.1f}us)")
            if float(config.deadband_us_neg) < 0.0 or float(config.deadband_us_neg) > (rng * 0.5):
                errors.append(f"Channel '{name}': deadband_us_neg is unrealistic (0 .. {rng*0.5:.1f}us)")
            # dither amplitude reasonable vs span
            if float(config.dither_amp_us) < 0.0 or float(config.dither_amp_us) > (rng * 0.25):
                errors.append(f"Channel '{name}': dither_amp_us is unrealistic (0 .. {rng*0.25:.1f}us)")
            # dither frequency sensible
            if float(config.dither_hz) <= 0.0 or float(config.dither_hz) > 200.0:
                errors.append(f"Channel '{name}': dither_hz must be within (0, 200]")
            # ramp limits: enabled channels need a positive rate
            if config.ramp_enable and float(config.ramp_limit) <= 0.0:
                errors.append(f"Channel '{name}': ramp_limit must be > 0 when ramp_enable is true")
            # s-curve limits: require both rate and accel
            if config.s_curve_enable:
                if float(config.ramp_limit) <= 0.0:
                    errors.append(f"Channel '{name}': ramp_limit must be > 0 when s_curve_enable is true")
                if float(config.accel_limit) <= 0.0:
                    errors.append(f"Channel '{name}': accel_limit must be > 0 when s_curve_enable is true")
            # gamma shaping bounds (keep reasonable)
            if float(config.gamma) <= 0.0 or float(config.gamma) > 5.0:
                errors.append(f"Channel '{name}': gamma must be within (0, 5]")

        if self.pump_config:
            if self.pump_config.output_channel in used_outputs:
                errors.append(f"Pump: output {self.pump_config.output_channel} already used")
            if not PWMConstants.PUMP_IDLE_MIN <= self.pump_config.idle <= PWMConstants.PUMP_IDLE_MAX:
                errors.append("Pump: idle out of range")
            if not 0 < self.pump_config.multiplier <= PWMConstants.PUMP_MULTIPLIER_MAX:
                errors.append("Pump: multiplier out of range")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

    def _start_monitoring(self):
        if self.skip_rate_checking or (self.monitor_thread and self.monitor_thread.is_alive()):
            return
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _stop_monitoring(self):
        if not self.running:
            return
        self.running = False
        self.input_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.monitor_thread = None

    def _monitor_loop(self):
        while self.running:
            wait_timeout = 0.2 if self.input_rate_threshold <= 0 else min(0.2, 1.0 / self.input_rate_threshold)
            if self.input_event.wait(timeout=wait_timeout):
                self.input_event.clear()
                current_time = time.time()
                self.last_input_time = current_time
                self.input_counter += 1
            else:
                current_time = time.time()

            # Rolling window safety rate check (less jitter-sensitive)
            elapsed = current_time - self.rate_window_start
            if elapsed >= self.time_window:
                rate = self.input_counter / elapsed if elapsed > 0 else 0.0
                required_rate = self.input_rate_threshold * PWMConstants.SAFE_STATE_THRESHOLD
                with self._lock:
                    self.is_safe_state = (rate >= required_rate)
                self.input_counter = 0
                self.rate_window_start = current_time

            # Stale watchdog: if no updates for too long, reset outputs
            if self._stale_timeout_s > 0.0:
                if (time.monotonic() - self._last_command_ts) > self._stale_timeout_s:
                    with self._lock:
                        if self.is_safe_state:
                            self.reset(reset_pump=False)
                            self.is_safe_state = False
                    continue

            if (current_time - self.last_input_time) > self.time_window:
                with self._lock:
                    if self.is_safe_state:
                        self.reset(reset_pump=False)
                        self.is_safe_state = False

    def update_named(self, commands: Dict[str, float], *, unset_to_zero: Optional[bool] = None,
                     one_shot_pump_override: bool = True, command_ts: Optional[float] = None):
        with self._lock:
            now_mono = time.monotonic()
            if command_ts is not None and self._stale_timeout_s > 0.0:
                if (now_mono - float(command_ts)) > self._stale_timeout_s:
                    return
            self._last_command_ts = now_mono

            if not self.skip_rate_checking:
                self.input_event.set()
                if not self.is_safe_state:
                    return

            if 'pump' in commands and self.pump_config:
                try:
                    pump_val = float(commands['pump'])
                    self._pump_override_throttle = max(-1.0, min(1.0, pump_val))
                except Exception:
                    pass

            do_zero = self._default_unset_to_zero if unset_to_zero is None else unset_to_zero
            if do_zero:
                for cfg in self.channel_configs.values():
                    self.values[cfg.output_channel] = 0.0

            self.pump_variable_sum = 0.0
            self.pump_variable_count = 0
            for name, val in commands.items():
                cfg = self.channel_configs.get(name)
                if cfg is None:
                    continue
                try:
                    value = float(val)
                except Exception:
                    continue
                value = max(-1.0, min(1.0, value))
                if abs(value) < cfg.deadzone_threshold:
                    value = 0.0
                self.values[cfg.output_channel] = value

            for cfg in self._affects_pump_channels:
                self.pump_variable_sum += abs(self.values[cfg.output_channel])
                self.pump_variable_count += 1

            self._update_channels()
            self._update_pump()
            self._direct_writer.flush()  # Single I2C transaction for all channels + pump

            if one_shot_pump_override:
                self._pump_override_throttle = None

    def _update_channels(self):
        with self._lock:
            now = time.time()
            for name, config in self.channel_configs.items():
                if not self.toggle_channels and config.toggleable:
                    continue

                value = self.values[config.output_channel]
                pulse = self._pulse_from_value(config, value, now, apply_ramp=True)

                # Convert to duty and queue for batch write
                duty_cycle = int((pulse / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
                duty_cycle = max(0, min(PWMConstants.DUTY_CYCLE_MAX, duty_cycle))
                self._direct_writer.set_channel(config.output_channel, duty_cycle)

            self._update_watchdog_queued()
            # Note: flush is called by update_named after _update_pump queues pump

    def _pulse_from_value(self, config: ChannelConfig, value: float, now: Optional[float] = None,
                          apply_ramp: bool = False) -> float:
        """Compute output pulse width (us) from normalized value using current config.

        Applies simple deadband by compressing command range into working area.
        Optional dither adds vibration to prevent valve stiction.
        Optional ramp limits slew rate so even deadband jumps are spread over time.
        """
        if now is None:
            now = time.time()

        # Enforce input deadzone locally so preview/compute_pulse() honors it too
        if abs(value) < float(getattr(config, 'deadzone_threshold', 0.0)):
            value = 0.0
        # Apply symmetric gamma shaping for both directions
        value = self._apply_gamma(value, float(config.gamma))

        base_pulse = self._compute_base_pulse(config, value)
        if apply_ramp and (config.ramp_enable or config.s_curve_enable):
            base_pulse = self._apply_ramp(config, base_pulse, now)

        use_dither = (
            config.dither_enable
            and float(config.dither_amp_us) != 0.0
            and float(config.dither_hz) != 0.0
            and abs(value) >= float(getattr(config, 'deadzone_threshold', 0.0))
        )
        pulse = self._apply_dither(config, base_pulse, value, now) if use_dither else base_pulse

        # Clamp to limits (optionally expanded when dither is active to preserve full amplitude)
        if use_dither:
            if config.dither_taper:
                pulse_min = float(config.pulse_min)
                pulse_max = float(config.pulse_max)
            else:
                amp = max(0.0, float(config.dither_amp_us))
                pulse_min = float(config.pulse_min) - amp
                pulse_max = float(config.pulse_max) + amp
        else:
            pulse_min = float(config.pulse_min)
            pulse_max = float(config.pulse_max)
        pulse = max(pulse_min, min(pulse_max, pulse))
        return pulse

    def _compute_base_pulse(self, config: ChannelConfig, value: float) -> float:
        """Map normalized value to physical pulse without dither or ramp."""
        # Simple deadband by physical sign (value * direction):
        # - s > 0 => physical positive: jump to center + deadband_us_pos, then scale to pulse_max
        # - s < 0 => physical negative: jump to center - deadband_us_neg, then scale to pulse_min
        # - s == 0 => center
        s = float(value) * float(config.direction)
        if s == 0.0:
            return float(config.center)
        elif s > 0.0:
            base = float(config.center) + float(config.deadband_us_pos)
            working_range = float(config.pulse_max) - base
            return base + abs(float(value)) * working_range
        else:  # s < 0.0
            base = float(config.center) - float(config.deadband_us_neg)
            working_range = base - float(config.pulse_min)
            return base - abs(float(value)) * working_range

    def _apply_dither(self, config: ChannelConfig, pulse: float, value: float, now: float) -> float:
        # Dither to prevent valve stiction (only when actively commanding)
        if config.dither_enable and value != 0.0 and abs(value) > float(getattr(config, 'deadzone_threshold', 0.0)):
            # Per-channel phase offset using output_channel index to avoid perfect sync
            phase = config._dither_omega * now + config._dither_phase_offset
            if config.dither_taper:
                headroom = min(pulse - float(config.pulse_min), float(config.pulse_max) - pulse)
                allowed_amp = max(0.0, min(float(config.dither_amp_us), headroom))
                dither = allowed_amp * math.sin(phase)
            else:
                dither = float(config.dither_amp_us) * math.sin(phase)
            pulse += dither
        return pulse

    def _apply_ramp(self, config: ChannelConfig, target_pulse: float, now: float) -> float:
        """Limit slew rate so large steps are spread over time."""
        state_container = getattr(self, "_channel_ramp_state", None)
        if state_container is None:
            self._channel_ramp_state = {}
            state_container = self._channel_ramp_state

        state = state_container.get(config.output_channel)
        if state is None:
            # Initialize state lazily if a channel was added later
            self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
            return target_pulse

        last_pulse, last_time, last_vel = state
        if config.s_curve_enable and float(config.ramp_limit) > 0.0 and float(config.accel_limit) > 0.0:
            ramp_mode = "s_curve"
        elif config.ramp_enable and float(config.ramp_limit) > 0.0:
            ramp_mode = "linear"
        else:
            self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
            return target_pulse

        dt_raw = max(0.0, now - last_time)
        # Clamp dt so a stalled loop cannot create a giant one-shot jump.
        dt = min(dt_raw, PWMConstants.RAMP_DT_MAX)
        if dt <= 0.0:
            self._channel_ramp_state[config.output_channel] = (last_pulse, now, last_vel)
            return last_pulse

        if ramp_mode == "s_curve":
            # Jerk-limited: limit acceleration of the slew rate
            v_max = float(config.ramp_limit)
            a_max = float(config.accel_limit)
            target_vel = (target_pulse - last_pulse) / dt
            if target_vel > v_max:
                target_vel = v_max
            elif target_vel < -v_max:
                target_vel = -v_max
            # Brake if we're too close to a limit to stop safely
            if last_vel > 0.0:
                dist_to_limit = float(config.pulse_max) - last_pulse
            elif last_vel < 0.0:
                dist_to_limit = last_pulse - float(config.pulse_min)
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
            # Clamp to user-configured limits and zero velocity if we hit a stop
            pulse_min = float(config.pulse_min)
            pulse_max = float(config.pulse_max)
            if new_pulse < pulse_min:
                new_pulse = pulse_min
                new_vel = 0.0
            elif new_pulse > pulse_max:
                new_pulse = pulse_max
                new_vel = 0.0
            self._channel_ramp_state[config.output_channel] = (new_pulse, now, new_vel)
        else:
            allowed_step = float(config.ramp_limit) * dt  # microseconds permitted in this interval
            delta = target_pulse - last_pulse
            if abs(delta) <= allowed_step:
                new_pulse = target_pulse
            else:
                new_pulse = last_pulse + allowed_step * (1 if delta > 0 else -1)
            # Clamp to user-configured limits
            pulse_min = float(config.pulse_min)
            pulse_max = float(config.pulse_max)
            if new_pulse < pulse_min:
                new_pulse = pulse_min
            elif new_pulse > pulse_max:
                new_pulse = pulse_max
            self._channel_ramp_state[config.output_channel] = (new_pulse, now, 0.0)
        return new_pulse

    # Public helper for testers to preview the pulse for a value
    def compute_pulse(self, name: str, value: float, now: Optional[float] = None) -> Optional[float]:
        cfg = self.channel_configs.get(name)
        if cfg is None:
            return None
        value = max(-1.0, min(1.0, float(value)))
        return self._pulse_from_value(cfg, value, now)

    @staticmethod
    def _apply_gamma(value: float, gamma: float) -> float:
        """Apply symmetric gamma shaping to a normalized command."""
        if gamma == 1.0 or value == 0.0:
            return value
        sign = 1.0 if value >= 0 else -1.0
        return sign * (abs(value) ** gamma)

    def _update_pump(self, flush: bool = False):
        """Update pump channel. Set flush=True for standalone calls outside update_named."""
        with self._lock:
            if not self.pump_config:
                return
            if self._pump_override_throttle is not None:
                throttle = self._pump_override_throttle
            elif not self.pump_enabled:
                throttle = -1.0
            else:
                if self.pump_variable:
                    denom = max(1, self.pump_variable_count)
                    throttle = self.pump_config.idle + (self.pump_config.multiplier * self.pump_variable_sum / denom)
                else:
                    throttle = self.pump_config.idle + (self.pump_config.multiplier / 10)
                throttle += self.manual_pump_load

            throttle = max(-1.0, min(1.0, throttle))
            pulse_range = self.pump_config.pulse_max - self.pump_config.pulse_min
            pulse = self.pump_config.pulse_min + pulse_range * ((throttle + 1) / 2)
            duty_cycle = int((pulse / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
            duty_cycle = max(0, min(PWMConstants.DUTY_CYCLE_MAX, duty_cycle))
            self._direct_writer.set_channel(self.pump_config.output_channel, duty_cycle)
            if flush:
                self._direct_writer.flush()

    def reset(self, reset_pump: bool = True):
        with self._lock:
            for name, config in self.channel_configs.items():
                duty_cycle = int((config.center / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
                duty_cycle = max(0, min(PWMConstants.DUTY_CYCLE_MAX, duty_cycle))
                self._direct_writer.set_channel(config.output_channel, duty_cycle)
            self._init_ramp_state()
            if reset_pump and self.pump_config:
                duty_cycle = int((self.pump_config.pulse_min / self._pwm_period_us) * PWMConstants.DUTY_CYCLE_MAX)
                duty_cycle = max(0, min(PWMConstants.DUTY_CYCLE_MAX, duty_cycle))
                self._direct_writer.set_channel(self.pump_config.output_channel, duty_cycle)
            self._direct_writer.flush()
            self.is_safe_state = False
            self.input_count = 0
            self._pump_override_throttle = None

    def get_average_input_rate(self) -> float:
        current_time = time.time()
        elapsed = current_time - self.rate_window_start
        if elapsed <= 0:
            return 0.0
        rate = self.input_counter / elapsed
        if elapsed >= self.time_window:
            self.input_counter = 0
            self.rate_window_start = current_time
        return rate

    def set_pump(self, enabled: bool):
        self.pump_enabled = enabled

    def toggle_pump_variable(self, variable: bool):
        self.pump_variable = variable

    def update_pump_load(self, adjustment: float):
        self.manual_pump_load = max(-1.0, min(0.3, self.manual_pump_load + adjustment / 10))

    def reset_pump_load(self):
        self.manual_pump_load = 0.0
        self._update_pump(flush=True)

    def disable_channels(self, disabled: bool):
        self.toggle_channels = not disabled

    def clear_pump_override(self):
        self._pump_override_throttle = None

    def get_channel_names(self, include_pump: bool = True) -> List[str]:
        names = list(self.channel_configs.keys())
        if include_pump and self.pump_config:
            names.append('pump')
        return names

    def build_zero_commands(self, include_toggleable: bool = True, include_pump: bool = False) -> Dict[str, float]:
        commands: Dict[str, float] = {}
        for name, cfg in self.channel_configs.items():
            if include_toggleable or not cfg.toggleable:
                commands[name] = 0.0
        if include_pump and self.pump_config:
            commands['pump'] = 0.0
        return commands

    def reload_config(self, config_file: str) -> bool:
        self.reset(reset_pump=True)
        try:
            was_monitoring = self.running
            if was_monitoring:
                self._stop_monitoring()
            self._load_config(config_file)
            self.reset(reset_pump=True)
            if was_monitoring:
                self._start_monitoring()
            return True
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False

    def set_log_level(self, level: str) -> None:
        """Change the logging level at runtime.

        Args:
            level: One of "DEBUG", "INFO", "WARNING", "ERROR"
        """
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def _init_ramp_state(self):
        now = time.time()
        self._channel_ramp_state: Dict[int, tuple[float, float, float]] = {}
        for cfg in self.channel_configs.values():
            self._channel_ramp_state[cfg.output_channel] = (float(cfg.center), now, 0.0)

    def _simple_cleanup(self):
        try:
            with self._lock:
                self._stop_monitoring()
                self.reset(reset_pump=True)
        except:
            pass

    def _update_watchdog_queued(self) -> None:
        """Queue watchdog toggle (called before flush)."""
        if self._watchdog_channel is None or self._watchdog_toggle_hz <= 0.0:
            return
        now_mono = time.monotonic()
        period = 1.0 / self._watchdog_toggle_hz
        if (now_mono - self._watchdog_last_toggle) < period:
            return
        self._watchdog_last_toggle = now_mono
        self._watchdog_state = not self._watchdog_state
        duty_cycle = PWMConstants.DUTY_CYCLE_MAX if self._watchdog_state else 0
        self._direct_writer.set_channel(int(self._watchdog_channel), duty_cycle)
