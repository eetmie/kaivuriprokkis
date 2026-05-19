"""pump_ramp_test.py — Interactive pump ramp-up tool for finding safe pulse_max.

MANUAL HARDWARE TEST — requires real pump hardware. Not a unit test.

Usage:
    python tests/pump_ramp_test.py [--config PATH] [--start US] [--end US]
                                   [--step US] [--interval SEC]

The pump starts at --start (default: pulse_min from config) and increases by
--step every --interval seconds, printing the current pulse width each tick.

Press Ctrl+C when the pump is spinning at the desired maximum speed.
The script will print the last pulse value as a suggested pulse_max.

Defaults:
    --start     pulse_min from config  (pump just started, e.g. 1000)
    --end       pulse_max from config  (hard ceiling, e.g. 1700)
    --step      10  (µs per tick)
    --interval  1.0 (seconds between steps)

Safety:
    - Pump is always stopped (pulse_min) on Ctrl+C or any error.
    - The script never exceeds the pulse_max defined in the config file.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CONFIG = ROOT_DIR / "configuration_files" / "servo_config_200.yaml"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Slowly ramp up the pump while printing pulse widths. "
                    "Press Ctrl+C to stop and record the value."
    )
    p.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="Path to servo config YAML (default: %(default)s)"
    )
    p.add_argument("--start", type=float, default=None,
                   help="Starting pulse width in µs (default: pulse_min from config)")
    p.add_argument("--end", type=float, default=None,
                   help="Maximum pulse width in µs (default: pulse_max from config)")
    p.add_argument("--step", type=float, default=10.0,
                   help="µs increment per tick (default: %(default)s)")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Seconds between steps (default: %(default)s)")
    return p.parse_args()


def _load_pump_limits(config_path: str):
    """Return (pulse_min, pulse_max, output_channel) from the YAML config."""
    import yaml  # type: ignore
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pump = cfg["CHANNEL_CONFIGS"]["pump"]
    return int(pump["pulse_min"]), int(pump["pulse_max"]), int(pump["output_channel"])


def main():
    args = _parse_args()

    # ---- hardware availability check ----
    try:
        import smbus2  # type: ignore
        bus = smbus2.SMBus(1)
        bus.read_byte(0x40)
        bus.close()
    except Exception as exc:
        print(f"ERROR: PCA9685 not reachable on I2C bus 1: {exc}")
        print("Connect hardware and retry.")
        sys.exit(1)

    pulse_min, pulse_max_cfg, output_channel = _load_pump_limits(args.config)

    start_us = args.start if args.start is not None else float(pulse_min)
    end_us = args.end if args.end is not None else float(pulse_max_cfg)

    # Hard-clamp to config bounds
    start_us = max(float(pulse_min), min(end_us, start_us))
    end_us = min(float(pulse_max_cfg), end_us)

    if start_us >= end_us:
        print(f"ERROR: start ({start_us} µs) must be less than end ({end_us} µs)")
        sys.exit(1)

    # ---- low-level PWM writer (bypass full stack for simplicity) ----
    from modules.PCA9685_controller import DirectPWMWriter  # type: ignore

    import smbus2
    i2c = smbus2.SMBus(1)

    # Read actual frequency from config to compute duty cycles correctly
    import yaml
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    pwm_freq = int(raw.get("pwm_frequency", 200))
    pwm_period_us = 1_000_000.0 / pwm_freq

    writer = DirectPWMWriter(
        i2c,
        min_channel=output_channel,
        max_channel=output_channel,
        frequency=pwm_freq,
    )

    last_pulse = start_us

    def _set_pulse(us: float):
        duty = int((us / pwm_period_us) * 65535)
        duty = max(0, min(65535, duty))
        writer.set_channel(output_channel, duty)
        writer.flush()

    def _stop():
        print(f"\n--- Stopping pump (sending pulse_min = {pulse_min} µs) ---")
        try:
            _set_pulse(pulse_min)
        except Exception as e:
            print(f"  Warning: could not send stop pulse: {e}")
        try:
            i2c.close()
        except Exception:
            pass

    # Ctrl+C handler
    interrupted = False

    def _sigint_handler(sig, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _sigint_handler)

    print("=" * 55)
    print("  Pump ramp test")
    print(f"  Config  : {args.config}")
    print(f"  Channel : {output_channel}  |  PWM freq: {pwm_freq} Hz")
    print(f"  Range   : {start_us:.0f} → {end_us:.0f} µs")
    print(f"  Step    : {args.step:.1f} µs  |  Interval: {args.interval:.2f} s")
    print("  Press Ctrl+C when pump reaches desired max speed.")
    print("=" * 55)
    input("  Press Enter to start... ")

    current = start_us
    try:
        while current <= end_us and not interrupted:
            last_pulse = current
            _set_pulse(current)
            print(f"  Pulse: {current:7.1f} µs", end="\r", flush=True)
            elapsed = 0.0
            tick = 0.05
            while elapsed < args.interval and not interrupted:
                time.sleep(tick)
                elapsed += tick
            current += args.step
    finally:
        _stop()

    print()
    if interrupted:
        print(f"  Ctrl+C caught at {last_pulse:.1f} µs")
    else:
        print(f"  Reached end of range: {last_pulse:.1f} µs")

    print()
    print("  Suggested config update:")
    print(f"    pulse_max: {int(round(last_pulse))}")
    print()
    print("  Also consider updating base_pulse_us and activity_gain_us in the YAML.")


if __name__ == "__main__":
    main()
