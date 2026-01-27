# Hardware Interface Arguments Handbook

This document describes all configuration arguments for `HardwareInterface` and the underlying `PWMController`. Understanding these parameters is essential for safe operation of hydraulic actuators.

---

## Table of Contents

1. [HardwareInterface.__init__() Arguments](#hardwareinterface__init__-arguments)
2. [HardwareInterface.send_named_pwm_commands() Arguments](#send_named_pwm_commands-arguments)
3. [PWMController Arguments (Lower Level)](#pwmcontroller-arguments-lower-level)
4. [Channel Configuration (YAML)](#channel-configuration-yaml)
5. [General Config File Options](#general-config-file-options)
6. [Safety Parameter Recommendations](#safety-parameter-recommendations)

---

## HardwareInterface.__init__() Arguments

### Configuration Files

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_file` | str | `"configuration_files/servo_config.yaml"` | Path to PWM channel configuration YAML |
| `hardware_config_path` | str | `"configuration_files/hardware_config.yaml"` | Path to hardware/driver config (rates, imu, pwm, safety) |

### Safety Parameters

| Parameter | Type | Default | Safety Impact | Recommended |
|-----------|------|---------|---------------|-------------|
| `input_rate_threshold` | int | `0` (disabled) | If > 0, monitors command rate. PWM resets if rate drops below `threshold * 0.25` for 1 second. | `5` or higher for remote control |
| `stale_timeout_s` | float | `0.0` (disabled) | If > 0, rejects commands older than this timeout and resets PWM if no commands received. Requires `command_ts` in `send_named_pwm_commands()`. | `0.3` - `1.0` seconds |
| `watchdog_channel` | int | `None` (disabled) | PWM channel (0-15) for external hardware watchdog. Toggles at `watchdog_toggle_hz` during operation. | Dedicated channel (e.g., 15) |
| `watchdog_toggle_hz` | float | `0.0` | Toggle frequency for watchdog signal. External relay detects missing pulses → cuts power. | `5` - `20` Hz |

**Why these matter:**
- `input_rate_threshold=0` means **no safety monitoring** - if controller disconnects, actuators keep last state
- `stale_timeout_s=0` means **no staleness check** - delayed/buffered commands are still executed
- `watchdog_channel=None` means **no hardware failsafe** - software crash leaves hydraulics powered

### Hardware Watchdog (External Failsafe)

The watchdog provides a **hardware-level safety cutoff** independent of all software:

```
┌─────────────────┐                    ┌─────────────────┐                    ┌─────────────────┐
│  Raspberry Pi   │   PWM toggles      │   Watchdog      │   Power enable     │   Hydraulic     │
│  PWM Channel 15 │ ────────────────── │   Relay Board   │ ────────────────── │   Pump/Valves   │
│                 │   at 10 Hz         │                 │                    │                 │
└─────────────────┘                    └─────────────────┘                    └─────────────────┘
                                              │
                                              │ No pulses for ~200ms
                                              ▼
                                       RELAY OPENS → POWER CUT
```

**When watchdog triggers:**
- Python hangs / crashes
- Kernel panic
- I2C bus lockup
- Any software failure that stops `update_named()` from being called

**Example setup:**
```python
hardware = HardwareInterface(
    config_file="configuration_files/servo_config.yaml",
    watchdog_channel=15,        # Use PWM channel 15 for watchdog
    watchdog_toggle_hz=10.0,    # Toggle at 10 Hz
    input_rate_threshold=5,     # Software safety (backup)
    stale_timeout_s=0.5,        # Software safety (backup)
)
```

The watchdog is the **last line of defense** - software safeties (`input_rate_threshold`, `stale_timeout_s`) handle normal disconnects, but the hardware watchdog handles catastrophic failures.

### PWM Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pump_variable` | bool | `False` | If `True`, pump speed varies with actuator demand. If `False`, pump runs at fixed idle + multiplier. |
| `toggle_channels` | bool | `False` | If `True`, enables "toggleable" channels (tracks, rotation). If `False`, these channels are disabled (useful for IK-only control). |
| `default_unset_to_zero` | bool | `True` | **Important:** If `True`, channels not mentioned in a command update are set to zero (neutral). If `False`, they keep their previous value (latching behavior). |

**`default_unset_to_zero` explained:**
```python
# With default_unset_to_zero=True (SAFER):
send_named_pwm_commands({'lift_boom': 0.5})
# Result: lift_boom=0.5, ALL other channels=0 (neutral)

# With default_unset_to_zero=False (DANGEROUS):
send_named_pwm_commands({'lift_boom': 0.5})
# Result: lift_boom=0.5, other channels KEEP PREVIOUS VALUE
# If you forget to send a channel, it keeps moving!
```

### Subsystem Enable/Disable

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_pwm` | bool | `True` | If `False`, skip PWM controller initialization (sensor-only mode) |
| `enable_imu` | bool | `True` | If `False`, skip IMU initialization and threads |
| `enable_adc` | bool | `True` | If `False`, skip ADC/encoder initialization and threads |
| `start_imu_reader` | bool | `True` | If `False`, IMU thread not auto-started (call `start_imu_streaming()` manually) |
| `start_adc_reader` | bool | `True` | If `False`, ADC thread not auto-started (call `start_adc_streaming()` manually) |

### Sensor Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imu_expected_hz` | float | `None` (from config) | Target IMU sample rate. If `None`, reads from `hardware_config.yaml` → `rates.imu_hz` |
| `adc_sample_hz` | float | `None` (from config) | Target ADC sample rate. If `None`, reads from `hardware_config.yaml` → `rates.adc_hz` |
| `adc_channels` | list | `None` | List of ADC channels to sample. Accepts sensor names (e.g., `"LiftBoom extend ps"`) or tuples (e.g., `("b1", 3)`). If `None`, defaults to slew encoder only. |

### Debugging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | str | `"INFO"` | Logging verbosity: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |
| `perf_enabled` | bool | `False` | If `True`, enables low-overhead performance metrics (IMU/ADC Hz, jitter stats) |

---

## send_named_pwm_commands() Arguments

```python
hardware.send_named_pwm_commands(
    commands: Dict[str, float],
    unset_to_zero: Optional[bool] = None,
    one_shot_pump_override: bool = True,
    command_ts: Optional[float] = None
) -> bool
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `commands` | dict | (required) | Channel name → value mapping. Values in `[-1.0, 1.0]`. Unknown names ignored. |
| `unset_to_zero` | bool/None | `None` | Override `default_unset_to_zero` for this call. `None` = use default. |
| `one_shot_pump_override` | bool | `True` | If `True` and `'pump'` is in commands, the pump override is cleared after this update. If `False`, pump override persists. |
| `command_ts` | float | `None` | Monotonic timestamp (`time.monotonic()`) of when command was generated. Used with `stale_timeout_s` to reject old commands. |

**Example with timestamp for safety:**
```python
# Capture timestamp when UDP packet received
receive_ts = time.monotonic()
data = udp_socket.get_latest_floats()

if data:
    commands = {'lift_boom': data[0], 'tilt_boom': data[1]}
    # Pass timestamp - if command is older than stale_timeout_s, it's rejected
    hardware.send_named_pwm_commands(commands, command_ts=receive_ts)
```

---

## PWMController Arguments (Lower Level)

These are passed through from `HardwareInterface` or can be set directly if using `PWMController` standalone.

| Parameter | Type | Default | Exposed in HardwareInterface? | Description |
|-----------|------|---------|------------------------------|-------------|
| `config_file` | str | (required) | Yes | Path to servo config YAML |
| `pump_variable` | bool | `False` | Yes | Variable vs fixed pump speed |
| `toggle_channels` | bool | `True` | Yes | Enable toggleable channels |
| `input_rate_threshold` | float | `0` | Yes | Safety rate monitoring threshold |
| `stale_timeout_s` | float | `0.0` | Yes | Command staleness timeout |
| `default_unset_to_zero` | bool | `True` | Yes | Zero unmentioned channels |
| `log_level` | str | `"INFO"` | Yes (via HardwareInterface) | Logging verbosity |
| `watchdog_channel` | int | `None` | Yes | PWM channel to toggle as external watchdog signal |
| `watchdog_toggle_hz` | float | `0.0` | Yes | Frequency to toggle watchdog channel |

All PWMController safety parameters are now exposed through HardwareInterface.

---

## Channel Configuration (YAML)

Each channel in `servo_config.yaml` has these fields:

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `output_channel` | int | PCA9685 channel number (0-15) |
| `pulse_min` | int | Minimum pulse width in microseconds |
| `pulse_max` | int | Maximum pulse width in microseconds |
| `direction` | int | `+1` or `-1`: maps input sign to physical direction |
| `center` | float/null | Center pulse width (µs). If `null`, computed as midpoint. |

### Behavior Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deadzone` | float | `0.0` | Input deadzone as percentage of full scale. Inputs below this are treated as zero. |
| `affects_pump` | bool | `False` | If `True`, this channel's activity influences variable pump speed. |
| `toggleable` | bool | `False` | If `True`, channel is disabled when `toggle_channels=False`. Used for tracks/rotation. |

### Deadband (Valve Dead Zone Compensation)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deadband_us_pos` | float | `0.0` | Microseconds to jump past center for positive commands (compensates valve dead zone) |
| `deadband_us_neg` | float | `0.0` | Microseconds to jump past center for negative commands |

### Dither (Anti-Stiction)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dither_enable` | bool | `False` | Enable high-frequency vibration to prevent valve stiction |
| `dither_amp_us` | float | `8.0` | Dither amplitude in microseconds |
| `dither_hz` | float | `40.0` | Dither frequency in Hz |
| `dither_taper` | bool | `False` | If `True`, reduce dither near pulse limits instead of expanding clamp range |

### Ramping (Slew Rate Limiting)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ramp_enable` | bool | `False` | Enable linear slew rate limiting |
| `ramp_limit` | float | `0.0` | Maximum rate of change in µs/second |
| `ramp_skip_deadband` | bool | `False` | If `True`, deadband jump is instant (slew limit only applies to usable range) |

### Input Shaping

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gamma` | float | `1.0` | Gamma curve exponent. `1.0` = linear. `>1.0` = less sensitive near center. `<1.0` = more sensitive near center. |

---

## Configuration Files

Configuration has been split into two files for clarity:

### hardware_config.yaml (Used by HardwareInterface)

```yaml
rates:
  imu_hz: 120.0        # IMU sample rate (sent via handshake)
  adc_hz: 120.0        # ADC/encoder read frequency
  pwm_update_hz: 100.0 # PWM update cadence

perf:
  enabled_default: false

imu:
  lpf_enabled: 1       # Enable low-pass filter on IMU
  lpf_alpha: 0.995     # LPF smoothing factor (0-1, higher = more smoothing)
  qmode: FULL          # Quaternion output mode

pwm:
  pump_variable: false
  toggle_channels: false
  input_rate_threshold: 0    # Recommended: 5+ for safety
  stale_timeout_s: 0.0       # Recommended: 0.3-0.5 for safety
  # watchdog_channel: 15     # Uncomment to enable hardware watchdog
  # watchdog_toggle_hz: 10.0
```

**Note:** Values in `hardware_config.yaml` → `pwm.*` override the corresponding `HardwareInterface` constructor arguments.

### control_config.yaml (Used by ExcavatorController)

```yaml
rates:
  control_hz: 100.0    # Main control loop frequency

pid:
  joint0: {kp: 4.5, ki: 0.25, kd: 0.75}   # Slew
  joint1: {kp: 15.0, ki: 0.5, kd: 0.0}    # Boom
  joint2: {kp: 7.0, ki: 0.5, kd: 0.15}    # Arm
  joint3: {kp: 5.0, ki: 0.5, kd: 0.15}    # Bucket

controller:
  output_limits_min: -1.0
  output_limits_max: 1.0
  enable_velocity_limiting: true
  per_joint_max_velocity: [0.5, 1.0, 1.0, 1.0]
```

---

## Safety Parameter Recommendations

### For Remote-Controlled Operation (UDP/Network)

```python
hardware = HardwareInterface(
    config_file="configuration_files/servo_config.yaml",
    input_rate_threshold=5,      # Require ~1.25 Hz minimum (5 * 0.25)
    stale_timeout_s=0.5,         # Reset if no commands for 500ms
    toggle_channels=True,        # Enable all channels
    default_unset_to_zero=True,  # Safe default
)

# In control loop:
receive_ts = time.monotonic()
data = server.get_latest_floats()
if data:
    hardware.send_named_pwm_commands(commands, command_ts=receive_ts)
```

### For Autonomous/IK Control

```python
hardware = HardwareInterface(
    config_file="configuration_files/servo_config.yaml",
    input_rate_threshold=10,     # Higher rate expected from IK loop
    stale_timeout_s=0.2,         # Tighter timeout for autonomous
    toggle_channels=False,       # Disable tracks (IK controls arm only)
    default_unset_to_zero=True,
)
```

### For Testing/Development (Less Strict)

```python
hardware = HardwareInterface(
    config_file="configuration_files/servo_config.yaml",
    input_rate_threshold=0,      # Disable rate checking (CAREFUL!)
    stale_timeout_s=0.0,         # Disable staleness check
    log_level="DEBUG",           # Verbose logging
)
```

---

## Safety Timeline Summary

When controller disconnects (with recommended settings):

| Time | Event |
|------|-------|
| 0ms | Last valid command received |
| 500ms | `stale_timeout_s` triggers → PWM reset (pump stays on) |
| 1000ms | `input_rate_threshold` window triggers → additional safety check |
| On shutdown | `HardwareInterface.shutdown()` calls `reset(reset_pump=True)` |
| On crash | `PWMController` atexit handler calls `reset(reset_pump=True)` |

---

## Configuration Layering (Why Parameters "Overlap")

Several parameters appear in multiple places: `HardwareInterface.__init__()`, `PWMController.__init__()`, and `hardware_config.yaml`. This is **intentional layered configuration**, not bad design.

### The Hierarchy

```
Priority (highest to lowest):
┌──────────────────────────────────────────────────────────────┐
│ 1. hardware_config.yaml (pwm.* section)    ← Wins if present │
├──────────────────────────────────────────────────────────────┤
│ 2. HardwareInterface constructor argument  ← Fallback        │
├──────────────────────────────────────────────────────────────┤
│ 3. Code default in function signature      ← Last resort     │
└──────────────────────────────────────────────────────────────┘
```

### How It Works

In `HardwareInterface.__init__()`:
```python
self.pwm_controller = PWMController(
    pump_variable=self._g('pwm.pump_variable', pump_variable),  # ← _g() does the lookup
    ...
)
```

The `_g()` helper means: "Get `pwm.pump_variable` from config file, OR use `pump_variable` argument as default."

### Why This Design?

| Approach | Use Case |
|----------|----------|
| **Config file** | Production settings, shared across scripts, version controlled |
| **Constructor args** | Script-specific overrides, testing, one-off changes |
| **Code defaults** | Sensible fallbacks, self-documenting API |

### Example

```yaml
# hardware_config.yaml
pwm:
  input_rate_threshold: 10  # Production safety setting
```

```python
# Script A: Uses config file value (10)
hardware = HardwareInterface()

# Script B: Overrides for testing (config file ignored for this param)
hardware = HardwareInterface(input_rate_threshold=0)  # Still uses 10! Config wins.
```

**Wait, that's confusing!** Yes - if the config file has a value, it **always wins**. To truly override, you must either:
1. Remove the value from `hardware_config.yaml`, or
2. Modify the code to not use `_g()` for that parameter

### Is This Good Design?

**Pros:**
- Single source of truth for production (config file)
- Scripts don't accidentally override critical safety settings
- Easy to change settings without code changes

**Cons:**
- Confusing when constructor arg is silently ignored
- Harder to write test scripts with different settings
- "Spooky action at a distance" - behavior depends on file state

**Verdict:** It's a deliberate design choice for **production safety** (config file controls critical settings), but the override behavior should be better documented. Consider adding logging when config overrides a constructor arg.

---

## Quick Reference Card

```
SAFE DEFAULTS (SOFTWARE):
  input_rate_threshold = 5      # Enable rate monitoring
  stale_timeout_s = 0.5         # Enable staleness check
  default_unset_to_zero = True  # Zero unmentioned channels
  command_ts = time.monotonic() # Always pass timestamps

SAFE DEFAULTS (HARDWARE FAILSAFE):
  watchdog_channel = 15         # Dedicated PWM channel for watchdog
  watchdog_toggle_hz = 10.0     # 10 Hz toggle signal to external relay

DANGEROUS:
  input_rate_threshold = 0      # No disconnect detection!
  stale_timeout_s = 0.0         # No staleness check!
  default_unset_to_zero = False # Channels latch last value!
  command_ts = None             # No timestamp validation!
  watchdog_channel = None       # No hardware failsafe!
```

## Safety Layers Summary

```
Layer 1: command_ts + stale_timeout_s
         └─ Catches: network delays, buffered packets
         └─ Response: reject stale command (500ms)

Layer 2: input_rate_threshold
         └─ Catches: controller disconnect, software pause
         └─ Response: PWM reset after 1 second

Layer 3: watchdog_channel + external relay
         └─ Catches: software crash, kernel panic, I2C lockup
         └─ Response: hardware power cut (~100-200ms)
```
