#!/usr/bin/env python3
"""Prototype server: relative IK + smooth reachability limiter, no binary gating.

Forks the lightweight pieces of excv_gui.py but:

  * Listens on port 8081 (production server keeps 8080).
  * Decodes incoming pose floats as VELOCITY (vx, vy, vz, vyaw_dps).
  * Runs RelativeCommandProcessor (smooth reachability limiter) per tick.
  * Calls controller.give_pose() with the INTEGRATED target.
  * Disables the controller's pre-flight binary reject:
        controller._reach_enabled = False
    so the smooth limiter is the sole authority on safety.

See control_prototype/README.md for the full design rationale.

-------------------------------------------------------------------------------
STOP BEHAVIOUR  (--zero-snap / lead-window)   *** NOT YET TESTED ON HARDWARE ***
-------------------------------------------------------------------------------

Two mechanisms prevent the integrated target from running ahead of the arm
when hydraulic lag causes the arm to trail the commanded velocity:

1. Lead-window clamp (always active):
   After every step the target is clamped so it cannot be more than
   --max-lead metres ahead of the measured EE position (default 0.05 m).
   If the clamp fires, proc is re-synced to the clamped position so the
   next step integrates from there, not from the original overshot target.
   The current lead distance is printed as "lead=Xmm" every 0.5 s.

2. Zero-velocity snap (--zero-snap, default ON):
   When the incoming velocity command drops to zero on all axes, proc is
   re-synced to the current measured pose BEFORE the step runs. This makes
   the controller hold the arm where it physically is rather than chasing
   whatever lead distance had accumulated before the stick was released.

   *** CAUTION — behaviour untested: ***
   On a real hydraulic machine an instantaneous snap-to-measured may feel
   like an abrupt hard stop or cause a PID transient if the arm was moving
   fast. The intended fix is a short velocity ramp-to-zero applied over
   ~0.1–0.3 s before the snap fires, but that ramp is NOT implemented yet.
   Disable with --no-zero-snap if the behaviour is problematic and test
   the lead-window clamp alone first.

Tuning guidance:
  --max-lead 0.05   Start here. Reduce if arm still coasts on release;
                    increase if the clamp fires during normal motion and
                    causes choppy feel (watch "lead=Xmm" in the log).
  --zero-snap       Toggle the snap. Test without it first to isolate the
                    lead-window behaviour, then enable and compare feel.

-------------------------------------------------------------------------------
FUTURE: PURE RUBBER-BAND MODEL  (not implemented here)
-------------------------------------------------------------------------------

A simpler and potentially more robust alternative to the current
integrator + snap + lead-window stack:

    target = meas_pos + normalize(v_cmd) * lookahead_m

The target is re-derived from the measured EE position every tick.
`lookahead_m` is a fixed tunable distance (e.g. 30-50 mm), not dt-dependent.

Properties:
  - Zero velocity → target = EE, arm holds naturally. No snap needed.
  - Non-zero velocity → target is always exactly `lookahead_m` ahead in the
    commanded direction. Controller pull force is constant and predictable.
  - Windup is impossible by construction. Lead-window clamp not needed.
  - Smooth start/stop falls out for free from server-side slew-rate
    limiting on v_cmd. The client sends raw fast-switching commands; the
    server ramps them to suit the robot. Any client (gamepad, ROS node,
    web UI) then gets safe smooth behaviour for free without knowing
    anything about the machine's dynamics. A simple per-axis rate limiter
    applied before the offset is computed is all that is needed.
  - The reachability limiter (smoothstep scaling of v_cmd) still works
    identically on top — it scales the direction vector before the offset
    is applied, so all that work transfers unchanged.

The only open question is whether a fixed 30-50 mm lead gives the PID
enough authority to drive the hydraulics smoothly across the full speed
range. Worth testing alongside the current integrator approach.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for p in (str(_HERE), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from modules.control_protocol import (
    COMMAND_PACKET_SIZE,
    TELEMETRY_PACKET_SIZE,
    decode_command_message,
    encode_telemetry_message,
)
from modules.excavator_controller import ExcavatorController
from modules.robot_service import RobotService
from modules.rt_utils import SCHED_FIFO, apply_rt_to_thread, reset_to_normal
from modules.udp_socket import UDPSocket

from excavator_fixture import REAL_JOINT_LIMITS_RAD
from reachability_limiter import LimiterConfig
from relative_command_processor import ProcessorConfig, RelativeCommandProcessor


def _cpu_affinity(core):
    return None if core is None else {int(core)}


def _print_banner(logger: logging.Logger) -> None:
    logger.info("=" * 72)
    logger.info("  PROTOTYPE SERVER  --  smooth reachability, no binary gating")
    logger.info("  Pose fields are VELOCITY: (vx, vy, vz, vyaw_dps)")
    logger.info("  Pair this server with control_prototype/gamepad_velocity_client.py")
    logger.info("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smooth-reachability prototype server")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="192.168.0.132")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--fifo-priority", "--rt-priority", dest="fifo_priority",
                        type=int, default=75)
    parser.add_argument("--lock-memory", action="store_true")
    parser.add_argument("--control-core", type=int, default=None)
    parser.add_argument("--io-core", type=int, default=None)
    parser.add_argument("--max-ee-speed", type=float, default=0.15,
                        help="cap on commanded EE speed (m/s) after smoothstep scaling")
    parser.add_argument("--max-yaw-rate", type=float, default=30.0,
                        help="cap on commanded yaw rate (deg/s) after smoothstep scaling")
    parser.add_argument("--joint-slow-zone-deg", type=float, default=8.0)
    parser.add_argument("--cond-ok", type=float, default=20.0)
    parser.add_argument("--cond-warn", type=float, default=40.0)
    parser.add_argument("--yoshikawa-nominal", type=float, default=0.10)
    parser.add_argument("--max-lead", type=float, default=0.05,
                        help="max distance (m) the integrated target may lead the measured pose")
    parser.add_argument("--zero-snap", action="store_true", default=True,
                        help="re-anchor integrator to measured pose on zero velocity (default ON)")
    parser.add_argument("--no-zero-snap", dest="zero_snap", action="store_false",
                        help="disable zero-velocity snap; rely on lead-window clamp only")
    args, _ = parser.parse_known_args()

    logger = logging.getLogger("excv_gui_relative")
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)

    _print_banner(logger)

    # ---- UDP -------------------------------------------------------------
    server = UDPSocket(local_id=2, max_age_seconds=0.5)
    server.setup(args.host, args.port,
                 inputs=f'{COMMAND_PACKET_SIZE}b', outputs=f'{TELEMETRY_PACKET_SIZE}b',
                 is_server=True)
    logger.info(f"Waiting for prototype client on {args.host}:{args.port}...")
    if not server.handshake(timeout=60.0):
        logger.error("Handshake failed.")
        return 1
    info = server.get_handshake_info()
    client_rate_hz = info.get("remote_nominal_rate_hz") or 50.0
    logger.info(f"Connected. Client nominal rate: {client_rate_hz:.1f} Hz")

    # ---- Hardware --------------------------------------------------------
    from modules.hardware_interface import HardwareInterface

    hardware = HardwareInterface(
        config_file="configuration_files/profiles/rpi/servo_config.yaml",
        control_config_file="configuration_files/profiles/rpi/control_config.yaml",
        log_level=args.log_level.upper(),
        pump_auto_mode=False, cleanup_disable_osc=False,
        enable_adc=False, start_adc_reader=False,
        rt_lock_memory=args.lock_memory,
        usb_rt_priority=args.fifo_priority,
        imu_rt_priority=args.fifo_priority,
        adc_rt_priority=args.fifo_priority,
        usb_cpu_core=args.io_core,
        imu_cpu_core=args.io_core,
        adc_cpu_core=args.io_core,
    )
    while not hardware.is_hardware_ready():
        time.sleep(0.1)

    # ---- Controller ------------------------------------------------------
    controller = ExcavatorController(
        hardware, config=None, enable_perf_tracking=False,
        log_level=args.log_level.upper(),
        rt_priority=args.fifo_priority,
        rt_lock_memory=args.lock_memory,
        rt_cpu_core=args.control_core,
        control_config_file="configuration_files/profiles/rpi/control_config.yaml",
    )

    # *** The prototype's defining line ***
    # Disable the controller's pre-flight binary reachability reject. The
    # smooth limiter below is now the sole safety layer. This is a private
    # field on purpose -- when migrating, swap this for a proper
    # config_override kwarg on ExcavatorController.__init__.
    controller._reach_enabled = False
    logger.info("Controller pre-flight reachability reject DISABLED "
                "(smooth limiter is authoritative).")

    service = RobotService(controller, hardware)
    service.start()
    logger.info("Controller started; warming up 5s...")
    time.sleep(5.0)

    if args.fifo_priority > 0 or args.lock_memory or args.control_core is not None:
        apply_rt_to_thread(
            priority=args.fifo_priority, policy=SCHED_FIFO,
            lock_memory=args.lock_memory,
            cpu_affinity=_cpu_affinity(args.control_core),
            quiet=True,
        )

    # ---- Limiter + integrator -------------------------------------------
    limiter_cfg = LimiterConfig(
        joint_slow_zone_deg=args.joint_slow_zone_deg,
        cond_ok=args.cond_ok,
        cond_warn=args.cond_warn,
        yoshikawa_nominal=args.yoshikawa_nominal,
        max_ee_speed_mps=args.max_ee_speed,
        max_yaw_rate_dps=args.max_yaw_rate,
    )
    proc = RelativeCommandProcessor(ProcessorConfig(limiter=limiter_cfg))

    # Seed the integrator with whatever the arm currently reads as.
    init_pos, init_rot = controller.get_pose()
    proc.sync_to_measured(init_pos, init_rot)
    logger.info(f"Integrator anchored at measured pose "
                f"x={init_pos[0]:+.3f} y={init_pos[1]:+.3f} z={init_pos[2]:+.3f} "
                f"rot={init_rot:+.2f} deg")

    server.start_receiving()
    logger.info("READY -- streaming velocity-mode commands.")

    logger.info(
        "Stop behaviour: lead_window=%.0fmm  zero_snap=%s  "
        "(*** untested on hardware — see module docstring ***)",
        args.max_lead * 1000.0,
        "ON" if args.zero_snap else "OFF (--no-zero-snap)",
    )

    # ---- Main loop -------------------------------------------------------
    loop_rate_hz = max(1.0, float(client_rate_hz))
    loop_period = 1.0 / loop_rate_hz
    next_t = time.perf_counter()
    last_step_t = time.perf_counter()
    last_print_t = time.time()
    print_period = 0.5
    packets_received = 0
    last_v_cmd = (0.0, 0.0, 0.0, 0.0)  # vx, vy, vz, vyaw_dps
    paused = False

    try:
        while True:
            now = time.perf_counter()
            dt = now - last_step_t
            last_step_t = now

            data = server.get_latest()
            if data is not None:
                try:
                    cmd = decode_command_message(data)
                    packets_received += 1
                    if cmd.pause and not paused:
                        paused = True
                        logger.info("Pause requested by client.")
                    if cmd.resume and paused:
                        paused = False
                        # On resume, re-sync the integrator to wherever the arm now is.
                        mp, mr = controller.get_pose()
                        proc.sync_to_measured(mp, mr)
                        logger.info("Resume requested; integrator re-anchored.")
                    # Pose fields carry (vx, vy, vz, vyaw_dps) in the prototype protocol.
                    last_v_cmd = (
                        float(cmd.pose.x),
                        float(cmd.pose.y),
                        float(cmd.pose.z),
                        float(cmd.pose.rot_y_deg),
                    )
                except Exception as exc:
                    logger.warning(f"Decode error: {exc!r}")

            if paused:
                v_cmd = (0.0, 0.0, 0.0, 0.0)
            else:
                v_cmd = last_v_cmd

            # Pull live signals from the controller.
            meas_pos, meas_rot = controller.get_pose()
            meas_pos_arr = np.asarray(meas_pos, dtype=np.float32)
            joint_angles_deg, _, _ = controller.get_joint_angles()
            joint_angles_rad = np.radians(np.asarray(joint_angles_deg, dtype=np.float64))
            yoshikawa = controller.get_yoshikawa_index()
            cond = controller.get_condition_number()

            # Zero-velocity snap (toggleable, see module docstring).
            # *** NOT YET TESTED ON HARDWARE — may need a ramp before snap fires ***
            if args.zero_snap:
                v_xyz_mag = math.sqrt(v_cmd[0]**2 + v_cmd[1]**2 + v_cmd[2]**2)
                if v_xyz_mag < 1e-9 and abs(v_cmd[3]) < 1e-9:
                    proc.sync_to_measured(meas_pos_arr, meas_rot)

            target, dbg = proc.step(
                [v_cmd[0], v_cmd[1], v_cmd[2]],
                v_cmd[3],
                dt,
                joint_angles_rad=joint_angles_rad,
                joint_limits_rad=REAL_JOINT_LIMITS_RAD,
                yoshikawa=yoshikawa,
                cond=cond,
            )

            # Lead-window clamp: prevent the integrated target from running more than
            # max_lead ahead of the measured pose (hydraulic lag windup prevention).
            lead = target.as_xyz() - meas_pos_arr
            lead_dist = float(np.linalg.norm(lead))
            if lead_dist > args.max_lead:
                clamped = meas_pos_arr + lead * (args.max_lead / lead_dist)
                proc.sync_to_measured(clamped, target.rot_y_deg)
                target = proc.target

            if not paused:
                controller.give_pose(target.as_xyz(), target.rot_y_deg)

            # Telemetry back to the client (via RobotService for full packet).
            try:
                tlm = service.get_state()
                tlm.target_pose.x = float(target.x)
                tlm.target_pose.y = float(target.y)
                tlm.target_pose.z = float(target.z)
                tlm.target_pose.rot_y_deg = float(target.rot_y_deg)
                tlm.paused = paused
                server.send(encode_telemetry_message(tlm))
            except Exception:
                pass

            now_wall = time.time()
            if now_wall - last_print_t >= print_period:
                last_print_t = now_wall
                v_applied = dbg["applied_v_xyz"]
                logger.info(
                    f"scale={dbg['scale']:.3f} dom={dbg['dominant']:<10} "
                    f"j={dbg['margins']['joint']:.2f} "
                    f"y={dbg['margins']['yoshikawa']:.2f}(raw={yoshikawa:.4f}) "
                    f"c={dbg['margins']['cond']:.2f}(raw={cond:.1f}) | "
                    f"v_in=({v_cmd[0]:+.3f},{v_cmd[1]:+.3f},{v_cmd[2]:+.3f}) "
                    f"v_out=({v_applied[0]:+.3f},{v_applied[1]:+.3f},{v_applied[2]:+.3f}) m/s | "
                    f"tgt=({target.x:+.3f},{target.y:+.3f},{target.z:+.3f}) "
                    f"meas=({meas_pos_arr[0]:+.3f},{meas_pos_arr[1]:+.3f},{meas_pos_arr[2]:+.3f}) "
                    f"lead={lead_dist*1000:.0f}mm | "
                    f"pkts={packets_received}"
                )

            next_t += loop_period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.error(f"Loop error: {exc!r}")
        import traceback
        traceback.print_exc()
    finally:
        reset_to_normal(quiet=True)
        logger.info("Stopping controller...")
        try:
            service.stop()
        except Exception:
            pass
        try:
            final_pos, final_rot = controller.get_pose()
            logger.info(f"Final pose: ({final_pos[0]:+.3f},{final_pos[1]:+.3f},"
                        f"{final_pos[2]:+.3f}) rot={final_rot:+.2f} deg | "
                        f"packets={packets_received}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
