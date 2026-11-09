"""Local gamepad input shared by LeRobot recording and inference."""

from __future__ import annotations

import time


BTN_A, BTN_B, BTN_X, BTN_Y = 0, 1, 2, 3
BTN_DPAD_UP, BTN_DPAD_DOWN, BTN_DPAD_LEFT, BTN_DPAD_RIGHT = 4, 5, 6, 7

GAMEPAD_DEADZONE_PCT = 25.0
GAMEPAD_PADDING_PCT = 0.0


class LocalGamepadInput:
    """Axes and button mask from an Xbox controller connected locally.

    Axis signs match the remote client, so teleoperation has the same direction
    in data collection and VLA inference.

    Tracks are on the triggers, which only read 0..1. Direction is set by
    holding the bumper on the same side: released drives that track forward,
    held reverses it. right_paddle/left_paddle already carry that sign, so a
    caller reads one -1..1 value per side rather than a trigger and a bumper
    separately. Matches simple_drive.LocalGamepadInput's convention.
    """

    name = "local"

    def __init__(
        self,
        deadzone: float = GAMEPAD_DEADZONE_PCT,
        padding: float = GAMEPAD_PADDING_PCT,
        connect_timeout_s: float = 10.0,
    ):
        self._deadzone = deadzone
        self._padding = padding
        self._timeout = connect_timeout_s
        self._pad = None

    def open(self) -> bool:
        try:
            from modules.gamepad import XboxController
        except Exception as exc:
            print(f"Gamepad import failed: {exc}")
            return False

        try:
            self._pad = XboxController(
                max_reconnect=None,
                deadzone=self._deadzone,
                padding=self._padding,
            )
        except Exception as exc:
            print(f"Gamepad open failed: {exc}")
            return False

        print("Waiting for gamepad...")
        deadline = time.perf_counter() + self._timeout
        while not self._pad.is_connected():
            if time.perf_counter() >= deadline:
                print(
                    f"No gamepad within {self._timeout:.0f}s. Is it plugged in, "
                    "and is this user in the 'input' group?"
                )
                return False
            time.sleep(0.1)
        print("Gamepad connected.")
        return True

    def poll(self):
        """Return ``(axes, button_mask)`` for the latest controller state."""
        if self._pad is None:
            raise RuntimeError("Gamepad is not open")
        state = self._pad.read()
        right_sign = -1.0 if state.get("RightBumper") else 1.0
        left_sign = -1.0 if state.get("LeftBumper") else 1.0
        axes = {
            "right_rl": -float(state["RightJoystickX"]),
            "right_ud": float(state["RightJoystickY"]),
            "left_rl": -float(state["LeftJoystickX"]),
            "left_ud": -float(state["LeftJoystickY"]),
            "right_paddle": right_sign * float(state["RightTrigger"]),
            "left_paddle": left_sign * float(state["LeftTrigger"]),
        }
        mask = 0
        buttons = (
            (BTN_A, "A"),
            (BTN_B, "B"),
            (BTN_X, "X"),
            (BTN_Y, "Y"),
            (BTN_DPAD_UP, "UpDPad"),
            (BTN_DPAD_DOWN, "DownDPad"),
            (BTN_DPAD_LEFT, "LeftDPad"),
            (BTN_DPAD_RIGHT, "RightDPad"),
        )
        for bit, key in buttons:
            if int(state.get(key, 0)):
                mask |= 1 << bit
        return axes, mask

    def is_live(self) -> bool:
        """Whether the controller is currently connected."""
        return self._pad is not None and self._pad.is_connected()

    def close(self) -> None:
        if self._pad is not None:
            try:
                self._pad.stop_monitoring()
            except Exception:
                pass
