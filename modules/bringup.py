"""Shared control-stack bring-up helpers.

All entry points (``run_hw_v2.py``, ``simple_drive.py``, ``excv_gui.py``, tools)
should block on the same wait-for-hardware logic so they behave identically
during the Pico's ~30 s startup calibration:

* one log line per 5 s with the IMU state and elapsed seconds,
* hard timeout instead of a silent infinite loop,
* clean shutdown + fault propagation on :class:`HardwareFaultError`.

Hall homing and other robot-specific startup steps are intentionally **not**
included here — keep this module to the parts every entry point needs.

Recommended bring-up order::

    hardware = HardwareInterface(...)             # starts IMU thread (calibration begins)
    controller = ExcavatorController(hardware, ...)
    controller.start()                            # JIT/numba warmup overlaps calibration
    wait_for_hardware_ready(hardware, logger=log) # logs progress, raises on fault/timeout
    # ... then open UDP socket / handshake / drive
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from modules.hardware_interface import HardwareFaultError, HardwareInterface


DEFAULT_READY_TIMEOUT_S = 90.0
DEFAULT_POLL_INTERVAL_S = 0.5
DEFAULT_STATUS_INTERVAL_S = 5.0


def wait_for_hardware_ready(
    hardware: HardwareInterface,
    *,
    timeout_s: float = DEFAULT_READY_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    status_interval_s: float = DEFAULT_STATUS_INTERVAL_S,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Block until ``hardware.is_hardware_ready()`` returns True.

    Logs the IMU state every ``status_interval_s`` so the user sees the Pico
    self-calibration progress instead of a silent terminal. Raises
    :class:`HardwareFaultError` immediately on subsystem fault and
    :class:`TimeoutError` if the wait exceeds ``timeout_s``.

    Returns the elapsed wait time in seconds.
    """
    log = logger or logging.getLogger(__name__)
    log.info(
        "Waiting for hardware ready (Pico self-calibrates ~30 s stationary, timeout %.0f s)...",
        timeout_s,
    )

    start = time.time()
    last_status_log = -status_interval_s  # force an immediate first status line
    while True:
        if hardware.is_hardware_ready():
            elapsed = time.time() - start
            log.info("Hardware ready after %.1f s.", elapsed)
            return elapsed

        elapsed = time.time() - start
        if elapsed >= timeout_s:
            raise TimeoutError(
                f"Timed out waiting for hardware readiness after {elapsed:.0f} s. "
                "Check IMU connection and keep the robot stationary during calibration."
            )

        if elapsed - last_status_log >= status_interval_s:
            status = hardware.get_status()
            phase = status.get("imu_startup_phase", "unknown")
            calib_wait_s = status.get("imu_calibration_wait_s") or 0.0
            log.info(
                "  IMU state: %s | phase: %s | %.0f s elapsed (timeout %.0f s, calib %.0f s)",
                status.get("imu_state", "unknown"),
                phase,
                elapsed,
                timeout_s,
                calib_wait_s,
            )
            last_status_log = elapsed

        time.sleep(poll_interval_s)
