"""
PWM controller.
"""

import atexit
import threading
import time
import logging
import yaml
import math
import sys
from typing import Optional, Dict, List, Any
from pathlib import Path

from ..perf_tracker import LoopPerfTracker
from .config import ChannelConfig, PumpConfig
from .constants import (
    CANONICAL_CHANNEL_NAMES,
    PCA9685_I2C_BUS,
    PCA9685_I2C_ADDRESS,
    PWMConstants,
    resolve_i2c_bus,
)
from .errors import PWMConfigError, PWMConfigLoadError, PWMConfigValidationError
from .validation import collect_config_validation_errors
from .writer import DirectPWMWriter, _open_smbus as _writer_open_smbus


def _open_smbus(bus_number: int | None = None):
    """Open SMBus, honoring the legacy facade monkeypatch target.

    The single real implementation lives in ``modules.pwm.writer``; this wrapper
    exists only so that tests patching ``modules.PCA9685_controller._open_smbus``
    still intercept opens made through this module.
    """
    compat_module = sys.modules.get("modules.PCA9685_controller")
    compat_open = getattr(compat_module, "_open_smbus", None)
    if compat_open is not None and compat_open is not _open_smbus:
        return compat_open(bus_number)
    return _writer_open_smbus(bus_number)

class PWMController:
    """Simple PWM controller with piecewise deadband and dither for valve testing."""

    def __init__(self, config_file: str, pump_auto_mode: bool = False,
                 toggle_channels: bool = True, input_rate_threshold: float = 0,
                 default_unset_to_zero: bool = True, log_level: str = "INFO",
                 stale_timeout_s: float = 0.0, perf_enabled: bool = False,
                 cleanup_disable_osc: bool = True, pwm_frequency: Optional[int] = None,
                 bus: Optional[int] = None,
                 i2c_addr: int = PCA9685_I2C_ADDRESS):
        """Initialize PWM controller.

        Args:
            config_file: Path to YAML configuration file
            pump_auto_mode: Enable auto pump speed (simple scaling with valve activity).
            toggle_channels: Enable toggleable channels
            input_rate_threshold: Input rate threshold for safety monitoring
            default_unset_to_zero: Default unset channels to zero
            log_level: Logging level - "DEBUG", "INFO", "WARNING", "ERROR"
            stale_timeout_s: Timeout for stale commands (0 = disabled)
            perf_enabled: Enable performance tracking (loop time, jitter, headroom)
            cleanup_disable_osc: If True, stop PCA9685 oscillator on cleanup (outputs go LOW).
                                 If False, keep oscillator running (outputs stay at center).
            pwm_frequency: PWM frequency in Hz (default: from PWMConstants.PWM_FREQUENCY_DEFAULT)
            bus: I2C bus number for PCA9685. None (default) resolves from the
                 active board profile -- 1 on Raspberry Pi, 7 on Orin Nano.
                 Only pass an explicit value to override that (CLI flag, ROS param).
            i2c_addr: PCA9685 I2C address.
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
        self._io_lock = threading.Lock()
        self.pump_auto_mode = bool(pump_auto_mode)
        self.toggle_channels = toggle_channels
        self.pump_enabled = True
        self._pump_direct_us: Optional[float] = None  # set by set_pump_speed_us; None = auto/static
        self.pump_activity_sum = 0.0
        self.pump_activity_count = 0
        self._stale_timeout_s = max(0.0, float(stale_timeout_s))
        self._last_command_ts = time.monotonic()

        # Rate monitoring (optional)
        self.input_rate_threshold = input_rate_threshold
        self.skip_rate_checking = (input_rate_threshold == 0)
        self.is_safe_state = not self.skip_rate_checking
        self.time_window = PWMConstants.DEFAULT_TIME_WINDOW

        # Threads/monitoring
        self.running = False
        self.monitor_thread = None
        self.last_input_time = time.time()

        # Load config
        self._load_config(config_file)

        # Hardware init. Resolve the bus once here so the number is available
        # for logging and for error messages raised by DirectPWMWriter.
        self._i2c_bus = resolve_i2c_bus(bus)
        self.logger.info(
            f"PCA9685 on i2c-{self._i2c_bus} address 0x{i2c_addr:02X}"
            f"{'' if bus is not None else ' (from board profile)'}"
        )
        self._i2c = _open_smbus(self._i2c_bus)
        self._i2c_addr = i2c_addr
        self._direct_writer = None
        self._pwm_period_us = 0.0
        self._rebuild_writer(pwm_frequency=pwm_frequency)

        # Current normalized values per channel
        self.values = [0.0] * PWMConstants.MAX_CHANNELS
        self._current_pulse_us = [0.0] * PWMConstants.MAX_CHANNELS

        # Rate monitoring counter — incremented in update_named() under _lock,
        # read and reset by _monitor_loop each window. Avoids event coalescing.
        self._cmd_counter = 0
        self.rate_window_start = time.time()

        # Register simple cleanup and start monitoring
        atexit.register(self._simple_cleanup)
        self.reset()

        if not self.skip_rate_checking or self._stale_timeout_s > 0.0:
            self._start_monitoring()

        # Behavior defaults
        self._default_unset_to_zero = default_unset_to_zero

        # Performance tracking (lightweight, opt-in)
        self._perf_tracker = LoopPerfTracker(enabled=perf_enabled)

        # Cleanup behavior: whether to stop oscillator (outputs go LOW) or keep running (outputs stay at center)
        self._cleanup_disable_osc = cleanup_disable_osc

    def _load_config(self, config_file: str):
        config_path, channel_configs, pump_config, pwm_frequency = self._read_config_data(config_file)
        self._config_path = config_path
        self.channel_configs = channel_configs
        self.pump_config = pump_config
        self._config_pwm_frequency = pwm_frequency

    def _resolve_config_path(self, config_file: str) -> Path:
        path = Path(config_file)
        candidates = [path]
        if not path.is_absolute():
            candidates.append(Path(__file__).resolve().parent.parent / path)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        searched = ", ".join(str(candidate) for candidate in candidates)
        raise PWMConfigLoadError(f"PWM config file '{config_file}' not found. Searched: {searched}")

    def _read_config_data(self, config_file: str) -> tuple[Path, Dict[str, ChannelConfig], Optional[PumpConfig], Any]:
        config_path = self._resolve_config_path(config_file)

        try:
            with config_path.open('r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise PWMConfigLoadError(f"Failed to parse YAML in '{config_path}': {exc}") from exc
        except OSError as exc:
            raise PWMConfigLoadError(f"Failed to read PWM config '{config_path}': {exc}") from exc

        if raw_config is None:
            raise PWMConfigLoadError(f"PWM config '{config_path}' is empty")
        if not isinstance(raw_config, dict):
            raise PWMConfigLoadError(
                f"PWM config '{config_path}' must contain a top-level mapping, got {type(raw_config).__name__}"
            )
        if 'CHANNEL_CONFIGS' not in raw_config:
            raise PWMConfigValidationError(
                f"PWM config '{config_path}' is missing required top-level key 'CHANNEL_CONFIGS'"
            )
        if not isinstance(raw_config['CHANNEL_CONFIGS'], dict):
            raise PWMConfigValidationError(
                f"PWM config '{config_path}': 'CHANNEL_CONFIGS' must be a mapping"
            )

        missing_canonical = [name for name in CANONICAL_CHANNEL_NAMES if name not in raw_config['CHANNEL_CONFIGS']]
        if missing_canonical:
            raise PWMConfigValidationError(
                f"PWM config '{config_path}' is missing canonical channel(s): "
                f"{', '.join(missing_canonical)}. Add entries with enabled: false if unused."
            )

        channel_configs, pump_config = self._parse_config(raw_config)
        pwm_frequency = raw_config.get('pwm_frequency', None)
        self._validate_config_data(channel_configs, pump_config, pwm_frequency)
        return config_path, channel_configs, pump_config, pwm_frequency

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
            'output_channel', 'pulse_min', 'pulse_max', 'static_pulse_us', 'base_pulse_us', 'activity_gain_us'
        ]
        required_channel_keys = [
            'output_channel', 'pulse_min', 'pulse_max', 'direction', 'center',
            'deadzone', 'affects_pump', 'toggleable', 'deadband_us_pos', 'deadband_us_neg',
            'dither_enable', 'dither_amp_us', 'dither_hz', 'dither_taper', 'dither_taper_us',
            'ramp_enable', 'ramp_limit', 'ramp_skip_deadband', 'gamma'
        ]

        for name, cfg in raw_config['CHANNEL_CONFIGS'].items():
            if not isinstance(cfg, dict):
                config_errors.append(
                    f"Entry '{name}' in CHANNEL_CONFIGS must be a mapping (got {type(cfg).__name__})"
                )
                continue

            if cfg.get('enabled', True) is False:
                continue

            item_errors_before = len(config_errors)
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
                static_pulse_us = _as_float(scope, 'static_pulse_us', cfg['static_pulse_us'])
                base_pulse_us = _as_float(scope, 'base_pulse_us', cfg['base_pulse_us'])
                activity_gain_us = _as_float(scope, 'activity_gain_us', cfg['activity_gain_us'])
                if len(config_errors) != item_errors_before:
                    continue
                pump_config = PumpConfig(
                    output_channel=output_channel,
                    pulse_min=pulse_min,
                    pulse_max=pulse_max,
                    static_pulse_us=static_pulse_us,
                    base_pulse_us=base_pulse_us,
                    activity_gain_us=activity_gain_us,
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
                dither_taper_us = _as_float(scope, 'dither_taper_us', cfg['dither_taper_us'])
                ramp_enable = _as_bool(scope, 'ramp_enable', cfg['ramp_enable'])
                ramp_limit = _as_float(scope, 'ramp_limit', cfg['ramp_limit'])
                ramp_skip_deadband = _as_bool(scope, 'ramp_skip_deadband', cfg['ramp_skip_deadband'])
                gamma = _as_float(scope, 'gamma', cfg['gamma'])
                if len(config_errors) != item_errors_before:
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
                    dither_taper_us=float(dither_taper_us),
                    # Ramp/slew limiting - opt-in per channel
                    ramp_enable=ramp_enable,
                    ramp_limit=float(ramp_limit),
                    ramp_skip_deadband=ramp_skip_deadband,
                    # Symmetric gamma shaping
                    gamma=float(gamma),
                )

        if config_errors:
            raise PWMConfigValidationError("Invalid PWM config:\n- " + "\n- ".join(config_errors))

        return channel_configs, pump_config

    @staticmethod
    def _normalize_none(value: Any) -> Optional[Any]:
        none_values = [None, "None", "none", "null", "NONE", "Null", "NULL", "", "n/a", "N/A"]
        return None if value in none_values else value

    def _validate_config(self):
        self._validate_config_data(self.channel_configs, self.pump_config, self._config_pwm_frequency)

    def _validate_config_data(
        self,
        channel_configs: Dict[str, ChannelConfig],
        pump_config: Optional[PumpConfig],
        pwm_frequency: Any,
    ):
        errors = collect_config_validation_errors(
            channel_configs,
            pump_config,
            pwm_frequency,
            PWMConstants,
        )
        if errors:
            raise PWMConfigValidationError("Configuration validation failed:\n" + "\n".join(errors))

    def _build_writer_state(
        self,
        channel_configs: Optional[Dict[str, ChannelConfig]] = None,
        pump_config: Optional[PumpConfig] = None,
        pwm_frequency: Optional[int] = None,
    ) -> tuple[DirectPWMWriter, float, List[ChannelConfig], int, int]:
        channel_configs = self.channel_configs if channel_configs is None else channel_configs
        pump_config = self.pump_config if pump_config is None else pump_config

        all_channels = [cfg.output_channel for cfg in channel_configs.values()]
        if pump_config:
            all_channels.append(pump_config.output_channel)
        min_ch = min(all_channels) if all_channels else 0
        max_ch = max(all_channels) if all_channels else 15

        freq = int(pwm_frequency if pwm_frequency is not None else self._config_pwm_frequency)
        writer = DirectPWMWriter(
            self._i2c,
            address=self._i2c_addr,
            min_channel=min_ch,
            max_channel=max_ch,
            frequency=freq,
            bus_number=getattr(self, "_i2c_bus", None),
        )
        pwm_period_us = 1e6 / float(writer.frequency)
        affects_pump_channels = [cfg for cfg in channel_configs.values() if cfg.affects_pump]
        return writer, pwm_period_us, affects_pump_channels, min_ch, max_ch

    def _rebuild_writer(self, pwm_frequency: Optional[int] = None) -> None:
        writer, pwm_period_us, affects_pump_channels, min_ch, max_ch = self._build_writer_state(
            pwm_frequency=pwm_frequency,
        )
        self._direct_writer = writer
        self._pwm_period_us = pwm_period_us
        self._affects_pump_channels = affects_pump_channels
        self.logger.info(
            f"DirectPWMWriter using channels {min_ch}-{max_ch} "
            f"({max_ch - min_ch + 1} channels, {1 + 4 * (max_ch - min_ch + 1)} bytes)"
        )

    def _start_monitoring(self):
        if (self.skip_rate_checking and self._stale_timeout_s <= 0.0) or (self.monitor_thread and self.monitor_thread.is_alive()):
            return
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _stop_monitoring(self):
        if not self.running:
            return
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.monitor_thread = None

    def _monitor_loop(self):
        _stale_reset_sent = False
        while self.running:
            time.sleep(0.05)
            current_time = time.time()

            # Rate check: count actual update_named() calls per window.
            # _cmd_counter is incremented under _lock in update_named(), so read
            # and reset it atomically here to get an accurate rate regardless of
            # how many commands arrived between monitor wakeups.
            if not self.skip_rate_checking:
                elapsed = current_time - self.rate_window_start
                if elapsed >= self.time_window:
                    with self._lock:
                        count = self._cmd_counter
                        self._cmd_counter = 0
                    rate = count / elapsed if elapsed > 0 else 0.0
                    with self._lock:
                        self.is_safe_state = (rate >= self.input_rate_threshold)
                    self.rate_window_start = current_time

            # Stale command timeout: reset outputs if update_named() stops being called.
            # Uses its own flag so it fires exactly once per stale period.
            if self._stale_timeout_s > 0.0:
                if (time.monotonic() - self._last_command_ts) > self._stale_timeout_s:
                    if not _stale_reset_sent:
                        with self._lock:
                            self.reset(reset_pump=False)
                            self.is_safe_state = False
                        _stale_reset_sent = True
                    continue
                else:
                    _stale_reset_sent = False

    def update_named(self, commands: Dict[str, float], *, unset_to_zero: Optional[bool] = None,
                     command_ts: Optional[float] = None):
        # Performance tracking (before lock to capture full update time)
        self._perf_tracker.tick_start()

        count_updates = []
        io_locked = False
        with self._lock:
            now_mono = time.monotonic()
            if command_ts is not None and self._stale_timeout_s > 0.0:
                if (now_mono - float(command_ts)) > self._stale_timeout_s:
                    return
            self._last_command_ts = now_mono

            if not self.skip_rate_checking:
                self._cmd_counter += 1
                if not self.is_safe_state:
                    return

            do_zero = self._default_unset_to_zero if unset_to_zero is None else unset_to_zero
            if do_zero:
                for cfg in self.channel_configs.values():
                    self.values[cfg.output_channel] = 0.0

            self.pump_activity_sum = 0.0
            self.pump_activity_count = 0
            for name, val in commands.items():
                cfg = self.channel_configs.get(name)
                if cfg is None:
                    continue
                try:
                    value = float(val)
                except Exception:
                    continue
                value = max(-1.0, min(1.0, value))
                self.values[cfg.output_channel] = value

            for cfg in self._affects_pump_channels:
                self.pump_activity_sum += abs(self.values[cfg.output_channel])
                self.pump_activity_count += 1

            now = time.monotonic()
            count_updates = self._collect_channel_counts(now)
            pump_update = self._pump_count_update()
            if pump_update is not None:
                count_updates.append(pump_update)

            # Preserve write ordering but do not hold the PWM state lock during I2C.
            self._io_lock.acquire()
            io_locked = True

        try:
            self._flush_count_updates(count_updates)
        finally:
            if io_locked:
                self._io_lock.release()

        # Performance tracking (after lock released, captures full update cycle)
        self._perf_tracker.tick_end()

    def _collect_channel_counts(self, now: float) -> List[tuple[int, int]]:
        updates: List[tuple[int, int]] = []
        for name, config in self.channel_configs.items():
            if not self.toggle_channels and config.toggleable:
                continue

            value = self.values[config.output_channel]
            pulse = self._pulse_from_value(config, value, now, apply_ramp=True)
            count = self._count_from_pulse_us(pulse)
            self._current_pulse_us[config.output_channel] = pulse
            updates.append((config.output_channel, count))
        return updates

    def _count_from_pulse_us(self, pulse_us: float) -> int:
        count = int((float(pulse_us) / self._pwm_period_us) * PWMConstants.PCA9685_COUNT_MAX)
        return max(0, min(PWMConstants.PCA9685_COUNT_MAX, count))

    @staticmethod
    def _clamp_channel_pulse_us(config: ChannelConfig, pulse_us: float) -> float:
        return max(float(config.pulse_min), min(float(config.pulse_max), float(pulse_us)))

    @staticmethod
    def _activity_from_pulse_us(config: ChannelConfig, pulse_us: float) -> float:
        center = float(config.center)
        max_span = max(abs(float(config.pulse_max) - center), abs(center - float(config.pulse_min)))
        if max_span <= 0.0:
            return 0.0
        return max(0.0, min(1.0, abs(float(pulse_us) - center) / max_span))

    def _flush_count_updates(self, updates: List[tuple[int, int]]) -> None:
        if not updates:
            return
        for channel, count in updates:
            self._direct_writer.set_channel(channel, count)
        self._direct_writer.flush()

    def update_named_us(self, commands: Dict[str, float], *, unset_to_center: Optional[bool] = None,
                        command_ts: Optional[float] = None):
        """Send name-based direct pulse-width commands in microseconds.

        This bypasses normalized value mapping, deadband, gamma, ramp, and dither.
        Pulses are clamped to each channel's configured [pulse_min, pulse_max].
        """
        self._perf_tracker.tick_start()

        count_updates = []
        io_locked = False
        with self._lock:
            now_mono = time.monotonic()
            if command_ts is not None and self._stale_timeout_s > 0.0:
                if (now_mono - float(command_ts)) > self._stale_timeout_s:
                    return
            self._last_command_ts = now_mono

            if not self.skip_rate_checking:
                self._cmd_counter += 1
                if not self.is_safe_state:
                    return

            do_center = self._default_unset_to_zero if unset_to_center is None else unset_to_center
            selected: Dict[int, float] = {}
            if do_center:
                for cfg in self.channel_configs.values():
                    if self.toggle_channels or not cfg.toggleable:
                        selected[cfg.output_channel] = float(cfg.center)

            for name, val in commands.items():
                cfg = self.channel_configs.get(name)
                if cfg is None:
                    continue
                if not self.toggle_channels and cfg.toggleable:
                    continue
                try:
                    pulse = self._clamp_channel_pulse_us(cfg, float(val))
                except Exception:
                    continue
                selected[cfg.output_channel] = pulse

            self.pump_activity_sum = 0.0
            self.pump_activity_count = 0
            for cfg in self._affects_pump_channels:
                pulse = selected.get(cfg.output_channel, self._current_pulse_us[cfg.output_channel])
                self.pump_activity_sum += self._activity_from_pulse_us(cfg, pulse)
                self.pump_activity_count += 1

            for output_channel, pulse in selected.items():
                self._current_pulse_us[output_channel] = pulse
                self.values[output_channel] = 0.0
                count_updates.append((output_channel, self._count_from_pulse_us(pulse)))

            pump_update = self._pump_count_update()
            if pump_update is not None:
                count_updates.append(pump_update)

            self._io_lock.acquire()
            io_locked = True

        try:
            self._flush_count_updates(count_updates)
        finally:
            if io_locked:
                self._io_lock.release()

        self._perf_tracker.tick_end()

    def set_named_pulse_us(self, commands: Dict[str, float], *, unset_to_center: Optional[bool] = None,
                           command_ts: Optional[float] = None):
        """Alias for update_named_us() with explicit direct-set naming."""
        self.update_named_us(commands, unset_to_center=unset_to_center, command_ts=command_ts)

    def _pulse_from_value(self, config: ChannelConfig, value: float, now: Optional[float] = None,
                          apply_ramp: bool = False) -> float:
        """Compute output pulse width (us) from normalized value using current config.

        Applies simple deadband by compressing command range into working area.
        Optional dither adds vibration to prevent valve stiction.
        Optional ramp limits slew rate so even deadband jumps are spread over time.
        """
        if now is None:
            now = time.monotonic()

        # Enforce input deadzone locally so preview/compute_pulse() honors it too
        value = self._apply_deadzone(value, float(getattr(config, 'deadzone_threshold', 0.0)))
        # Apply symmetric gamma shaping for both directions
        value = self._apply_gamma(value, float(config.gamma))

        base_pulse = self._compute_base_pulse(config, value)
        if apply_ramp and config.ramp_enable:
            base_pulse = self._apply_ramp(config, base_pulse, now)

        dither_active = abs(base_pulse - float(config.center)) > 1e-6
        use_dither = (
            config.dither_enable
            and float(config.dither_amp_us) != 0.0
            and float(config.dither_hz) != 0.0
            and dither_active
        )
        if use_dither:
            dither_amp = self._dither_amplitude(config, base_pulse)
            base_pulse = self._clamp_base_for_dither(config, base_pulse, dither_amp)
            dither_amp = self._dither_amplitude(config, base_pulse)
            pulse = self._apply_dither(config, base_pulse, now, dither_amp)
        else:
            pulse = base_pulse

        # Final safety clamp: configured pulse limits are hard limits.
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

    def _dither_amplitude(self, config: ChannelConfig, pulse: float) -> float:
        amp = float(config.dither_amp_us)
        if not config.dither_taper:
            return amp

        center = float(config.center)
        if pulse > center:
            deadband_edge = center + float(config.deadband_us_pos)
            distance_from_edge = pulse - deadband_edge
        else:
            deadband_edge = center - float(config.deadband_us_neg)
            distance_from_edge = deadband_edge - pulse
        taper_us = max(float(config.dither_taper_us), 1e-6)
        taper_gain = max(0.0, min(1.0, distance_from_edge / taper_us))
        return amp * taper_gain

    def _clamp_base_for_dither(self, config: ChannelConfig, pulse: float, dither_amp: float) -> float:
        pulse_min = float(config.pulse_min)
        pulse_max = float(config.pulse_max)
        lower = pulse_min + max(0.0, dither_amp)
        upper = pulse_max - max(0.0, dither_amp)
        if lower > upper:
            return (pulse_min + pulse_max) * 0.5
        return max(lower, min(upper, pulse))

    def _apply_dither(self, config: ChannelConfig, pulse: float, now: float, dither_amp: float) -> float:
        # Dither to prevent valve stiction (only when actively commanding)
        # Per-channel phase offset using output_channel index to avoid perfect sync
        phase = config._dither_omega * now + config._dither_phase_offset
        dither = dither_amp * math.sin(phase)
        return pulse + dither

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
        if not (config.ramp_enable and float(config.ramp_limit) > 0.0):
            self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
            return target_pulse

        # Deadband skip: slew limit only applies to usable range, not deadband jump
        center = float(config.center)
        if config.ramp_skip_deadband:
            pos_edge = center + float(config.deadband_us_pos)
            neg_edge = center - float(config.deadband_us_neg)

            # Transitioning from center to active: start from deadband edge
            if last_pulse == center and target_pulse != center:
                if target_pulse > center:
                    last_pulse = pos_edge
                else:
                    last_pulse = neg_edge
                # If target is exactly at the edge, we're done
                if target_pulse == last_pulse:
                    self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
                    return target_pulse

            # Crossing from one active side to the other: snap through center
            elif last_pulse > center and target_pulse < center:
                # Was positive, going negative - snap to negative edge
                last_pulse = neg_edge
                if target_pulse >= last_pulse:
                    self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
                    return target_pulse
            elif last_pulse < center and target_pulse > center:
                # Was negative, going positive - snap to positive edge
                last_pulse = pos_edge
                if target_pulse <= last_pulse:
                    self._channel_ramp_state[config.output_channel] = (target_pulse, now, 0.0)
                    return target_pulse

        dt_raw = max(0.0, now - last_time)
        # Clamp dt so a stalled loop cannot create a giant one-shot jump.
        dt = min(dt_raw, PWMConstants.RAMP_DT_MAX)
        if dt <= 0.0:
            self._channel_ramp_state[config.output_channel] = (last_pulse, now, 0.0)
            return last_pulse

        delta = target_pulse - last_pulse
        allowed_step = float(config.ramp_limit) * dt  # microseconds permitted in this interval
        if abs(delta) <= allowed_step:
            new_pulse = target_pulse
        else:
            new_pulse = last_pulse + allowed_step * (1 if delta > 0 else -1)

        # If ramping toward center and we'd enter deadband region, snap to center
        if config.ramp_skip_deadband and target_pulse == center:
            pos_edge = center + float(config.deadband_us_pos)
            neg_edge = center - float(config.deadband_us_neg)
            if neg_edge < new_pulse < pos_edge:
                new_pulse = center

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
    def _apply_deadzone(value: float, threshold: float) -> float:
        """Apply deadzone with rescaling so output uses full range after threshold.

        Values within [-threshold, +threshold] map to 0.
        Values outside are rescaled so that threshold->0 and 1.0->1.0.
        """
        if threshold <= 0.0:
            return value
        abs_val = abs(value)
        if abs_val < threshold:
            return 0.0
        if threshold >= 1.0:
            return 0.0
        sign = 1.0 if value >= 0 else -1.0
        return sign * (abs_val - threshold) / (1.0 - threshold)

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
            update = self._pump_count_update()
            if update is None:
                return
            if not flush:
                self._direct_writer.set_channel(update[0], update[1])
                return

        with self._io_lock:
            self._flush_count_updates([update])

    def _pump_count_update(self) -> Optional[tuple[int, int]]:
        if not self.pump_config:
            return None
        if not self.pump_enabled:
            pulse = float(self.pump_config.pulse_min)
        elif self._pump_direct_us is not None:
            # Direct control: caller owns the pulse, no auto logic applied
            pulse = max(float(self.pump_config.pulse_min),
                        min(float(self.pump_config.pulse_max), self._pump_direct_us))
        elif self.pump_auto_mode:
            # Auto mode: scale between base_pulse_us and base+activity_gain based on valve activity
            denom = max(1, self.pump_activity_count)
            avg_activity = self.pump_activity_sum / denom  # 0.0 - 1.0
            pulse = self.pump_config.base_pulse_us + (
                self.pump_config.activity_gain_us * avg_activity
            )
            pulse = max(float(self.pump_config.pulse_min),
                        min(float(self.pump_config.pulse_max), pulse))
        else:
            # Static mode: fixed speed from config
            pulse = max(float(self.pump_config.pulse_min),
                        min(float(self.pump_config.pulse_max), self.pump_config.static_pulse_us))

        count = self._count_from_pulse_us(pulse)
        self._current_pulse_us[self.pump_config.output_channel] = pulse
        return self.pump_config.output_channel, count

    def reset(self, reset_pump: bool = True):
        count_updates = []
        io_locked = False
        with self._lock:
            self.values = [0.0] * PWMConstants.MAX_CHANNELS
            self.pump_activity_sum = 0.0
            self.pump_activity_count = 0
            self._pump_direct_us = None
            for name, config in self.channel_configs.items():
                pulse = float(config.center)
                count = self._count_from_pulse_us(pulse)
                self._current_pulse_us[config.output_channel] = pulse
                count_updates.append((config.output_channel, count))
            self._init_ramp_state()
            if reset_pump and self.pump_config:
                pulse = float(self.pump_config.pulse_min)
                count = self._count_from_pulse_us(pulse)
                self._current_pulse_us[self.pump_config.output_channel] = pulse
                count_updates.append((self.pump_config.output_channel, count))
            self.is_safe_state = False
            self.input_counter = 0
            self._io_lock.acquire()
            io_locked = True

        try:
            self._flush_count_updates(count_updates)
        finally:
            if io_locked:
                self._io_lock.release()

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

    # =========================================================================
    # Performance Tracking API
    # =========================================================================

    def set_perf_enabled(self, enabled: bool) -> None:
        """Enable or disable performance tracking at runtime."""
        self._perf_tracker.set_enabled(enabled)

    def get_perf_stats(self) -> Dict[str, Any]:
        """Get PWM update performance statistics.

        Returns:
            Dictionary with timing stats (all in milliseconds):
            - hz: Measured update rate
            - loop_avg_ms, loop_std_ms, loop_min_ms, loop_max_ms: Update interval stats
            - proc_avg_ms, proc_std_ms, proc_min_ms, proc_max_ms: Processing time stats
            - samples: Number of samples collected
        """
        return self._perf_tracker.get_stats()

    def reset_perf_stats(self) -> None:
        """Reset performance statistics."""
        self._perf_tracker.reset()

    def set_pump_enabled(self, enabled: bool, flush: bool = True):
        with self._lock:
            self.pump_enabled = bool(enabled)
        if flush:
            self._update_pump(flush=True)

    def set_pump_auto(self, auto: bool):
        """Enable (True) or disable (False) valve-activity-based auto speed scaling."""
        with self._lock:
            self.pump_auto_mode = bool(auto)

    def set_pump_activity_gain_us(self, gain_us: float) -> float:
        """Set the auto-mode activity_gain_us at runtime. Returns the clamped value."""
        with self._lock:
            if not self.pump_config:
                return 0.0
            clamped = max(0.0, min(float(gain_us), float(self.pump_config.pulse_max - self.pump_config.pulse_min)))
            self.pump_config.activity_gain_us = clamped
            return clamped

    def get_pump_activity_gain_us(self) -> float:
        """Return current auto-mode activity_gain_us, or 0 if no pump configured."""
        with self._lock:
            return float(self.pump_config.activity_gain_us) if self.pump_config else 0.0

    def set_pump_speed_us(self, pulse_us: Optional[float], flush: bool = True):
        """Set pump speed directly in microseconds, bypassing auto/static logic.

        The pulse is clamped to [pulse_min, pulse_max] from config.
        Pass None to release direct control and return to auto/static mode.
        """
        with self._lock:
            if pulse_us is None:
                self._pump_direct_us = None
            else:
                if self.pump_config:
                    self._pump_direct_us = max(float(self.pump_config.pulse_min),
                                               min(float(self.pump_config.pulse_max), float(pulse_us)))
                else:
                    self._pump_direct_us = float(pulse_us)
        if flush:
            self._update_pump(flush=True)

    def disable_channels(self, disabled: bool):
        self.toggle_channels = not disabled

    def get_channel_names(self, include_pump: bool = True) -> List[str]:
        names = list(self.channel_configs.keys())
        if include_pump and self.pump_config:
            names.append('pump')
        return names

    def get_current_pulses_us(self) -> Dict[str, float]:
        """Return last commanded pulse widths by channel name, in microseconds."""
        with self._lock:
            pulses = {
                name: float(self._current_pulse_us[cfg.output_channel])
                for name, cfg in self.channel_configs.items()
            }
            if self.pump_config:
                pulses['pump'] = float(self._current_pulse_us[self.pump_config.output_channel])
            return pulses

    def build_zero_commands(self, include_toggleable: bool = True, include_pump: bool = False) -> Dict[str, float]:
        commands: Dict[str, float] = {}
        for name, cfg in self.channel_configs.items():
            if include_toggleable or not cfg.toggleable:
                commands[name] = 0.0
        if include_pump and self.pump_config:
            commands['pump'] = 0.0
        return commands

    def reload_config(self, config_file: str) -> bool:
        config_path = None
        was_monitoring = self.running
        try:
            if was_monitoring:
                self._stop_monitoring()

            with self._lock:
                # Hold outputs neutral while the config file is parsed/swapped.
                # This blocks concurrent command writes until reload completes.
                self.reset(reset_pump=True)

                config_path, channel_configs, pump_config, pwm_frequency = self._read_config_data(config_file)
                writer_state = self._build_writer_state(
                    channel_configs=channel_configs,
                    pump_config=pump_config,
                    pwm_frequency=int(pwm_frequency),
                )
                self._config_path = config_path
                self.channel_configs = channel_configs
                self.pump_config = pump_config
                self._config_pwm_frequency = pwm_frequency
                self._direct_writer, self._pwm_period_us, self._affects_pump_channels, min_ch, max_ch = writer_state
                self.logger.info(
                    f"DirectPWMWriter using channels {min_ch}-{max_ch} "
                    f"({max_ch - min_ch + 1} channels, {1 + 4 * (max_ch - min_ch + 1)} bytes)"
                )
                self.reset(reset_pump=True)

            if was_monitoring:
                self._start_monitoring()
            return True
        except PWMConfigError as exc:
            location = config_path if config_path is not None else config_file
            self.logger.error(f"Configuration reload failed for '{location}': {exc}")
            if was_monitoring:
                self._start_monitoring()
            return False
        except Exception as exc:
            self.logger.error(f"Unexpected error reloading configuration '{config_file}': {exc}")
            if was_monitoring:
                self._start_monitoring()
            return False

    def set_log_level(self, level: str) -> None:
        """Change the logging level at runtime.

        Args:
            level: One of "DEBUG", "INFO", "WARNING", "ERROR"
        """
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def _init_ramp_state(self):
        # monotonic, to match the clock _collect_channel_counts stamps ramp
        # state with. Seeding from time.time() made the first ramped tick after
        # every reset()/reload_config() see a negative dt (epoch vs uptime),
        # clamp it to zero, and hold the channel at center for that tick.
        now = time.monotonic()
        self._channel_ramp_state: Dict[int, tuple[float, float, float]] = {}
        for cfg in self.channel_configs.values():
            self._channel_ramp_state[cfg.output_channel] = (float(cfg.center), now, 0.0)

    def _simple_cleanup(self):
        """Cleanup on exit: reset channels to center, optionally stop oscillator."""
        try:
            self._stop_monitoring()
            with self._lock:
                # Reset all channels to center (safe neutral position)
                self.reset(reset_pump=True)
                # Small delay for hardware to settle at center position
                time.sleep(0.05)
                # Optionally stop oscillator (outputs go LOW) or keep running (outputs stay at center)
                if self._cleanup_disable_osc:
                    self._direct_writer.sleep()
        except Exception as exc:
            self.logger.warning(f"PWM cleanup failed: {exc}")

    def set_cleanup_disable_osc(self, enabled: bool) -> None:
        """Configure whether oscillator stops on cleanup.

        Args:
            enabled: If True, stop oscillator on cleanup (outputs go LOW).
                     If False, keep oscillator running (outputs stay at center).
        """
        self._cleanup_disable_osc = enabled
