"""Direct valve control — bypass IK and PID, push normalized PWM commands.

Owns its own command queue and a thin send method. Lightweight; runs on the
caller's thread (no background loop of its own — the caller drives ticks).
Stale-command timeouts already live in :mod:`modules.PCA9685_controller`'s
PWM layer, so sparse ``send_pending`` calls are still safe.

Typical use::

    direct = DirectController(hardware)
    controller.suspend_ik_output()       # ExcavatorController hands over the bus
    while running:
        commands = read_gamepad()
        direct.give_commands(commands)   # store
        direct.send_pending()            # push to hardware
    direct.clear()
    controller.resume_ik_output()

When commands is empty, ``send_pending`` zeroes outputs idempotently — useful
for emergency stops between strokes without spamming the PWM controller.
"""

from __future__ import annotations

import threading
from typing import Dict


class DirectController:
    """Direct-mode valve driver. Holds the latest commands; sends on demand."""

    def __init__(self, hardware) -> None:
        self.hardware = hardware
        self._lock = threading.Lock()
        self._commands: Dict[str, float] = {}
        self._outputs_zeroed = False

    def give_commands(self, commands: Dict[str, float]) -> None:
        """Store the next set of normalized [-1, 1] valve commands."""
        with self._lock:
            self._commands = dict(commands)
        self._outputs_zeroed = False

    def clear(self) -> None:
        """Clear the command queue. Next ``send_pending`` will zero outputs."""
        with self._lock:
            self._commands = {}

    def send_pending(self) -> None:
        """Push the latest stored commands to hardware. Idempotent zero when empty."""
        with self._lock:
            cmds = dict(self._commands)

        if not cmds:
            if not self._outputs_zeroed:
                self.hardware.reset(reset_pump=False)
                self._outputs_zeroed = True
            return

        self.hardware.send_named_pwm_commands(cmds)
        self._outputs_zeroed = False

    def emergency_stop(self, reset_pump: bool = True) -> None:
        """Clear commands and slam outputs to neutral (optionally stop pump)."""
        with self._lock:
            self._commands = {}
        self._outputs_zeroed = True
        self.hardware.reset(reset_pump=reset_pump)
