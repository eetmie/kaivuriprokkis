#!/usr/bin/env python3
"""Compensated / closed-loop excavator driving with data collection.

Split out of simple_drive.py, which is now strictly open-loop. Everything
here is a correction applied between the stick and the valve, and none of it
is validated on hardware yet — hence control_prototype/. Note this is separate
work from the smooth reachability limiter the rest of this directory is about.

The logger, sine generator and input sources are imported from simple_drive so
there is exactly one implementation of each; only the compensation layer and
its tuning knobs live here.

Usage:
    python control_prototype/drive_compensated.py --comp raw
    python control_prototype/drive_compensated.py --comp vel-pid --robot jetson
    python control_prototype/drive_compensated.py --comp universal --ip 0.0.0.0:8080

Compensation modes (--comp, static — not button-toggled):
    none         — raw joystick commands, no correction (same as simple_drive)
    raw          — per-joint valve scaling from linkage-rate table
    universal    — pump-gain modulation from universal shape table
    vel-pid      — closed-loop velocity PI (requires IMU)

Input source and buttons match simple_drive.py: --ip selects a remote UDP
client, otherwise the local gamepad is used. A=log B=sine X=pump Y=reload-config,
D-pad Up/Down steps sine amplitude.
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.board import PROFILES as ROBOT_PROFILES, resolve_profile as _resolve_board_profile
from modules.bringup import wait_for_hardware_ready
from modules.direct_controller import DirectController
from tools.linkage_rate_compensation import (
    DEFAULT_TABLE,
    DEFAULT_UNIVERSAL_TABLE,
    LinkageRateCompensator,
    UniversalShapeCompensator,
)
from simple_drive import (
    JOINT_NAMES,
    LOG_OUTPUT_DIR,
    PRINT_DECIMATION,
    SAMPLING_FREQUENCY,
    STATUS_PRINT_INTERVAL_S,
    RECORD_MINUTES,
    BTN_A, BTN_B, BTN_X, BTN_Y,
    BTN_DPAD_DOWN, BTN_DPAD_UP,
    DataLogger,
    LocalGamepadInput,
    PWMOnlyController,
    SineExcitationGenerator,
    UDPInput,
    _format_imu_line,
    _get_control_joint_names,
    _get_imu_role_order,
)


# ── velocity PID ─────────────────────────────────────────────────────────────

_VEL_CTRL_JOINTS = {'boom': 1, 'arm': 2, 'bucket': 3}


class JointVelocityController:
    """Joystick → desired deg/s → PI → valve command for the three arm joints."""

    def __init__(self, kp: float, ki: float, ki_max: float,
                 deadband_degps: float, max_degps: float):
        self.kp       = kp
        self.ki       = ki
        self.ki_max   = ki_max
        self.deadband = deadband_degps
        self.max_degps = max_degps
        self._integral: dict[str, float] = {n: 0.0 for n in _VEL_CTRL_JOINTS}

    def apply(self, commands: dict, joint_vels, dt: float) -> dict:
        out = dict(commands)
        for name, vel_idx in _VEL_CTRL_JOINTS.items():
            if name not in out or vel_idx >= len(joint_vels):
                continue
            desired = float(out[name]) * self.max_degps
            if abs(desired) < self.deadband:
                self._integral[name] = 0.0
                out[name] = 0.0
                continue
            err = desired - float(joint_vels[vel_idx])
            self._integral[name] = float(
                np.clip(self._integral[name] + err * dt, -self.ki_max, self.ki_max)
            )
            out[name] = float(np.clip(self.kp * err + self.ki * self._integral[name], -1.0, 1.0))
        return out

    def reset(self):
        for k in self._integral:
            self._integral[k] = 0.0


# ── args ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compensated / closed-loop excavator driving + data collection")
    p.add_argument("--robot", choices=[*sorted(ROBOT_PROFILES), "auto"], default="auto",
                   help="Board profile (default: auto-detect)")
    p.add_argument("--ip", default=None, metavar="HOST[:PORT]",
                   help="Listen for a remote UDP client instead of using the local gamepad")
    p.add_argument("--enable-slew",   action="store_true",
                   help="Allow slew in manual commands and sine (default: off)")
    p.add_argument("--enable-tracks", action="store_true",
                   help="Allow track drive from the triggers/paddles (default: off)")
    p.add_argument("--auto-pump", dest="auto_pump", action="store_true",
                   help="Pump follows commanded activity (required by --comp universal)")
    # compensation
    p.add_argument("--comp", choices=["none", "raw", "universal", "vel-pid"], default="none",
                   help="Compensation mode (default: none — raw commands)")
    p.add_argument("--linkage-rate-table",           default=str(DEFAULT_TABLE))
    p.add_argument("--linkage-rate-universal-table", default=str(DEFAULT_UNIVERSAL_TABLE))
    p.add_argument("--linkage-rate-min-factor", type=float, default=0.35)
    p.add_argument("--linkage-rate-max-factor", type=float, default=2.25)
    p.add_argument("--vel-kp",       type=float, default=0.04)
    p.add_argument("--vel-ki",       type=float, default=0.008)
    p.add_argument("--vel-ki-max",   type=float, default=0.4)
    p.add_argument("--vel-deadband", type=float, default=2.0,  help="deg/s")
    p.add_argument("--vel-max-degps",type=float, default=30.0, help="deg/s at full stick")
    return p.parse_args()


def _make_input_source(args):
    """--ip selects the remote client; without it the local pad is used."""
    if not args.ip:
        return LocalGamepadInput()
    host, port = (args.ip.rsplit(":", 1)[0], int(args.ip.rsplit(":", 1)[1])) \
        if ":" in args.ip else (args.ip, 8080)
    return UDPInput(host, port)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = _parse_args()
    profile = _resolve_board_profile(args.robot)
    imu_on  = bool(profile['enable_imu'])

    out_dir = LOG_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── hardware ──────────────────────────────────────────────────────────────
    from modules.hardware_interface import HardwareFaultError, HardwareInterface

    print("Initializing hardware...")
    hardware = HardwareInterface(
        config_file=profile['servo_config_file'],
        control_config_file=profile['control_config_file'],
        pump_auto_mode=args.auto_pump,
        toggle_channels=True,
        stale_timeout_s=0.5,
        enable_pwm=True,
        enable_imu=imu_on,
        enable_adc=False,
        start_imu_reader=imu_on,
        start_adc_reader=False,
        cleanup_disable_osc=False,
        pwm_i2c_bus=profile['pwm_i2c_bus'],
        pwm_i2c_addr=profile['pwm_i2c_addr'],
    )

    try:
        wait_for_hardware_ready(hardware)
    except HardwareFaultError as e:
        print(f"\n*** HARDWARE FAULT ({e.subsystem}): {e.reason} ***")
        hardware.shutdown(); raise SystemExit(1)
    except TimeoutError as e:
        print(f"\n*** {e} ***")
        hardware.shutdown(); raise SystemExit(1)

    # ── controller ────────────────────────────────────────────────────────────
    print("Starting controller...")
    if imu_on:
        from modules.excavator_controller import ExcavatorController
        controller = ExcavatorController(
            hardware, config=None, enable_perf_tracking=False,
            control_config_file=profile['control_config_file'],
        )
    else:
        controller = PWMOnlyController(hardware)

    controller.start()
    if imu_on:
        time.sleep(2.0)      # numba JIT warmup
    direct = DirectController(hardware)
    controller.suspend_ik_output()

    ctrl_joint_names = _get_control_joint_names(controller)
    imu_role_order   = _get_imu_role_order(controller)

    # ── compensation setup ────────────────────────────────────────────────────
    linkage_comp   = None
    universal_comp = None
    vel_ctrl       = None

    if args.comp in ("raw", "universal"):
        try:
            linkage_comp = LinkageRateCompensator(
                args.linkage_rate_table,
                min_factor=args.linkage_rate_min_factor,
                max_factor=args.linkage_rate_max_factor,
            )
        except Exception as e:
            print(f"[WARN] Linkage-rate table unavailable: {e}")

    if args.comp == "universal":
        try:
            universal_comp = UniversalShapeCompensator(
                args.linkage_rate_universal_table,
                min_factor=args.linkage_rate_min_factor,
                max_factor=args.linkage_rate_max_factor,
            )
        except Exception as e:
            print(f"[WARN] Universal shape table unavailable: {e}")

    if args.comp == "vel-pid":
        if not imu_on:
            print("[WARN] --comp vel-pid requires IMU; falling back to none")
            args.comp = "none"
        else:
            vel_ctrl = JointVelocityController(
                kp=args.vel_kp, ki=args.vel_ki, ki_max=args.vel_ki_max,
                deadband_degps=args.vel_deadband, max_degps=args.vel_max_degps,
            )

    pwm = hardware.pwm_controller
    pump_gain_available = pwm is not None and getattr(pwm, 'pump_config', None) and pwm.pump_auto_mode
    pump_gain_base_us   = pwm.get_pump_activity_gain_us() if pump_gain_available else 0.0

    if args.comp == "universal" and not pump_gain_available:
        print("[WARN] --comp universal modulates pump gain, which needs --auto-pump; "
              "the shape table will have no effect")

    print(f"Profile: {profile['profile_name']} | comp: {args.comp}"
          f" | pump: {'auto' if args.auto_pump else 'fixed'}"
          f" | slew: {'on' if args.enable_slew else 'off'}"
          f" | tracks: {'on' if args.enable_tracks else 'off'}")

    # ── RT scheduling (Linux only) ─────────────────────────────────────────────
    try:
        from modules.rt_utils import apply_rt_to_thread, SCHED_FIFO
        apply_rt_to_thread(priority=75, policy=SCHED_FIFO, lock_memory=False)
    except Exception:
        pass

    # ── helpers ───────────────────────────────────────────────────────────────
    sine_gen = SineExcitationGenerator()
    logger   = DataLogger(out_dir)

    # ── input source ──────────────────────────────────────────────────────────
    source = _make_input_source(args)
    if not source.open():
        source.close()
        direct.clear(); controller.resume_ik_output(); controller.stop()
        hardware.shutdown(); raise SystemExit(1)
    print("A=log  B=sine  X=pump  Y=reload-config"
          + ("  Dpad U/D=sine-amp\n" if source.name == "local" else "\n"))

    # ── loop state ────────────────────────────────────────────────────────────
    loop_period     = 1.0 / SAMPLING_FREQUENCY
    next_run_time   = time.perf_counter()
    prev_loop_time  = time.perf_counter()

    right_rl = right_ud = left_rl = left_ud = right_paddle = left_paddle = 0.0
    mask_prev     = 0

    last_status_time = time.time()
    record_start_time = None
    print_imu        = False
    iter_count       = 0

    try:
        while True:
            now      = time.time()
            loop_now = time.perf_counter()
            actual_dt = loop_now - prev_loop_time
            prev_loop_time = loop_now

            # ── 1. receive ────────────────────────────────────────────────────
            axes, mask = source.poll()
            if axes is not None:
                right_rl     = axes['right_rl']
                right_ud     = axes['right_ud']
                left_rl      = axes['left_rl']
                left_ud      = axes['left_ud']
                right_paddle = axes['right_paddle']
                left_paddle  = axes['left_paddle']

                def btn(b):  return bool(mask & (1 << b))
                def prev(b): return bool(mask_prev & (1 << b))

                if btn(BTN_A) and not prev(BTN_A):
                    if not logger.is_logging:
                        sine_gen.reseed()
                        logger.start()
                        record_start_time = now
                    else:
                        logger.stop_and_save(direct)
                        record_start_time = None

                if btn(BTN_B) and not prev(BTN_B):
                    sine_gen.toggle()
                    if sine_gen.enabled:
                        # Params are re-drawn on every enable, so note the seed:
                        # it is what makes a strip's excitation reproducible.
                        print(f"\n[Button B] Sine ON (target={sine_gen.target_name} "
                              f"seed={sine_gen.seed})")
                    else:
                        print("\n[Button B] Sine OFF")

                if btn(BTN_X) and not prev(BTN_X):
                    if pwm is not None:
                        new_state = not pwm.pump_enabled
                        hardware.set_pump_enabled(new_state)
                        print(f"\n[Button X] Pump {'ON' if new_state else 'OFF'}")

                # Y: reload servo config from disk. Outputs are neutralised
                # first — a valve calibration swap while a channel is commanded
                # open would step that valve as the new mapping takes effect.
                if btn(BTN_Y) and not prev(BTN_Y):
                    direct.clear()
                    direct.send_pending()
                    time.sleep(0.1)
                    ok = hardware.reload_config()
                    print(f"\n[Button Y] Config reload {'OK' if ok else 'FAILED'}")

                for bit, step in ((BTN_DPAD_UP, +1), (BTN_DPAD_DOWN, -1)):
                    if btn(bit) and not prev(bit):
                        sine_gen.step_target(step)
                        print(f"\n[D-pad] Sine target → {sine_gen.target_name}")

                mask_prev = mask

            # ── 2. build commands ─────────────────────────────────────────────
            is_logging = logger.is_logging

            manual = {
                'slew':   left_rl if args.enable_slew else 0.0,
                'boom':   right_ud,
                'arm':    left_ud,
                'bucket': right_rl,
            }

            t = time.perf_counter()
            sine = sine_gen.get_all(t)
            if not args.enable_slew:
                sine['slew'] = 0.0

            combined = {n: float(np.clip(manual[n] + sine[n], -1.0, 1.0)) for n in JOINT_NAMES}
            if args.enable_tracks:
                combined['trackR'] = right_paddle
                combined['trackL'] = left_paddle

            # ── 3. compensation ───────────────────────────────────────────────
            joint_angles, _, _ = controller.get_joint_angles()

            if args.comp == "raw" and linkage_comp is not None:
                combined = linkage_comp.apply(combined, joint_angles)
            elif args.comp == "vel-pid" and vel_ctrl is not None:
                vels, vel_age = controller.get_joint_velocities_with_age()
                if vel_age < 0.05:
                    combined = vel_ctrl.apply(combined, vels, actual_dt)
                else:
                    vel_ctrl.reset()

            # ── 4. send ───────────────────────────────────────────────────────
            direct.give_commands(combined)
            direct.send_pending()

            # ── 5. pump gain (universal shape) ────────────────────────────────
            if pump_gain_available:
                factor = (universal_comp.pump_correction_factor(combined, joint_angles)
                          if args.comp == "universal" and universal_comp is not None else 1.0)
                pwm.set_pump_activity_gain_us(pump_gain_base_us * factor)

            # ── 6. log ────────────────────────────────────────────────────────
            if is_logging:
                logger.log_sample(manual, sine, combined, controller, hardware,
                                  sine_gen.enabled,
                                  sine_gen.target_name, sine_gen.seed)

            # ── 7. auto-stop ──────────────────────────────────────────────────
            # Ends the recording rather than rolling into a continuation file,
            # so the machine gets a cooling pause between sets.
            if is_logging and record_start_time is not None \
                    and (now - record_start_time) >= RECORD_MINUTES * 60:
                logger.stop_and_save(direct)
                record_start_time = None
                print(f"[REC] stopped ({RECORD_MINUTES:.0f} min reached). "
                      f"Press A to record the next set.")

            # ── 8. IMU print (decimated) ──────────────────────────────────────
            iter_count += 1
            if print_imu and imu_on and (iter_count % PRINT_DECIMATION == 0):
                print(_format_imu_line(
                    hardware.read_imu_debug_quaternions(),
                    joint_angles, ctrl_joint_names, imu_role_order,
                ), end="\r", flush=True)

            # ── 9. status ─────────────────────────────────────────────────────
            if now - last_status_time >= STATUS_PRINT_INTERVAL_S:
                last_status_time = now
                pump_on  = bool(pwm and pwm.pump_enabled)
                sine_str = (f"ON target={sine_gen.target_name} seed={sine_gen.seed}"
                            if sine_gen.enabled else f"OFF (target={sine_gen.target_name})")
                print(
                    f"[STATUS] comp={args.comp} | pump={'ON' if pump_on else 'OFF'} | "
                    f"sine={sine_str} | "
                    + (f"log=ON {logger.elapsed_min():.1f}/{RECORD_MINUTES:.0f}min "
                       f"{logger.n_samples()} samples"
                       if is_logging else "log=OFF")
                    + ("" if source.is_live() else f" | *** {source.name} INPUT LOST ***")
                )
                print(f"[JOINTS] slew={joint_angles[0]:+.1f} boom={joint_angles[1]:+.1f} "
                      f"arm={joint_angles[2]:+.1f} bucket={joint_angles[3]:+.1f} deg")

            # ── 10. timing ────────────────────────────────────────────────────
            next_run_time += loop_period
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C).")
    finally:
        print("Shutting down...")
        source.close()
        if logger.n_samples() > 0:
            logger.is_logging = False
            direct.clear()
            direct.send_pending()
            time.sleep(0.2)
            logger.save()
        try:
            controller.emergency_stop(reset_pump=True)
        except Exception:
            pass
        try:
            controller.stop()
        except Exception:
            pass
        try:
            hardware.shutdown()
        except Exception:
            pass
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
