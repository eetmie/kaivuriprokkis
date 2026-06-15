"""Unit tests for modules.bringup.wait_for_hardware_ready.

The helper just polls ``hardware.is_hardware_ready()`` until True or
timeout, with optional status logging. Tests use a fake hardware that
advances through ready states by call count, plus a stub clock so we
don't actually sleep.
"""

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.bringup import wait_for_hardware_ready  # noqa: E402
from modules.hardware_interface import HardwareFaultError  # noqa: E402


class _FakeHardware:
    """Becomes ready on the ``ready_after_n`` call to ``is_hardware_ready``."""

    def __init__(self, ready_after_n=1, fault_after_n=None, status=None):
        self._calls = 0
        self._ready_after_n = ready_after_n
        self._fault_after_n = fault_after_n
        self._status = status or {"imu_state": "READY", "imu_startup_phase": "idle"}

    def is_hardware_ready(self):
        self._calls += 1
        if self._fault_after_n is not None and self._calls >= self._fault_after_n:
            raise HardwareFaultError("imu", "fake fault")
        return self._calls >= self._ready_after_n

    def get_status(self):
        return dict(self._status)


class _StubClock:
    """Monotonic clock that advances by a fixed step each call."""

    def __init__(self, step_s=0.5):
        self._t = 0.0
        self._step = step_s

    def now(self):
        t = self._t
        self._t += self._step
        return t


class WaitForHardwareReadyTests(unittest.TestCase):
    def _patch_time(self, clock, sleeper=None):
        sleeper = sleeper or (lambda _s: None)
        return patch.multiple("modules.bringup", time=type("T", (), {
            "time": staticmethod(clock.now),
            "sleep": staticmethod(sleeper),
        }))

    def test_returns_immediately_when_ready(self):
        hw = _FakeHardware(ready_after_n=1)
        clock = _StubClock(step_s=0.1)
        with self._patch_time(clock):
            elapsed = wait_for_hardware_ready(hw, timeout_s=10.0)
        self.assertIsInstance(elapsed, float)

    def test_polls_until_ready(self):
        hw = _FakeHardware(ready_after_n=5)
        clock = _StubClock(step_s=0.5)
        with self._patch_time(clock):
            wait_for_hardware_ready(hw, timeout_s=10.0, poll_interval_s=0.5)
        self.assertEqual(hw._calls, 5)

    def test_timeout_raises(self):
        # Hardware never becomes ready; clock advances 0.5 s per call.
        hw = _FakeHardware(ready_after_n=10_000)
        clock = _StubClock(step_s=0.5)
        with self._patch_time(clock):
            with self.assertRaises(TimeoutError):
                wait_for_hardware_ready(hw, timeout_s=2.0, poll_interval_s=0.5)

    def test_hardware_fault_propagates(self):
        hw = _FakeHardware(ready_after_n=1000, fault_after_n=3)
        clock = _StubClock(step_s=0.1)
        with self._patch_time(clock):
            with self.assertRaises(HardwareFaultError):
                wait_for_hardware_ready(hw, timeout_s=10.0)

    def test_status_logging_includes_imu_phase(self):
        hw = _FakeHardware(
            ready_after_n=4,
            status={"imu_state": "PENDING", "imu_startup_phase": "calibrating",
                    "imu_calibration_wait_s": 7.0},
        )
        clock = _StubClock(step_s=2.5)  # > status_interval_s
        log_records = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record.getMessage())

        log = logging.getLogger("test_bringup")
        log.setLevel(logging.INFO)
        log.handlers.clear()
        log.addHandler(_CaptureHandler())

        with self._patch_time(clock):
            wait_for_hardware_ready(hw, timeout_s=30.0, poll_interval_s=0.5,
                                     status_interval_s=2.0, logger=log)

        joined = "\n".join(log_records)
        self.assertIn("calibrating", joined)
        self.assertIn("PENDING", joined)


if __name__ == "__main__":
    unittest.main()
