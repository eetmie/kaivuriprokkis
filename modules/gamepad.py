# Base code from: https://github.com/kevinhughes27/TensorKart/blob/master/utils.py

import math
import os
import struct
import threading
import time
import inputs
from inputs import get_gamepad, UnpluggedError

try:
    import fcntl  # Linux only; absent on Windows, where the ioctl probe is skipped.
except ImportError:
    fcntl = None


"""
MAPPING (this is the index read() function returns):
0: 'LeftJoystickY'
1: 'LeftJoystickX'
2: 'RightJoystickY'
3: 'RightJoystickX'
4: 'LeftTrigger'
5: 'RightTrigger'
6: 'LeftBumper'
7: 'RightBumper'
8: 'A'
9: 'X'
10: 'Y'
11: 'B'
12: 'LeftThumb'
13: 'RightThumb'
14: 'Back'
15: 'Start'
16: 'LeftDPad'
17: 'RightDPad'
18: 'UpDPad'
19: 'DownDPad'
"""


class XboxController:
    # Fallbacks only. The real full-scale value per axis is read from the
    # driver at connect time (see _refresh_axis_scales); these are what the
    # XInput backend on Windows reports, where that probe is unavailable.
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    # inputs names buttons after raw evdev codes, where BTN_NORTH is an alias
    # of BTN_X (0x133) and BTN_WEST of BTN_Y (0x134) - a historical Linux
    # naming mismatch. The kernel pad drivers emit those for the buttons
    # physically labelled X and Y, so on Linux north is X. The Windows XInput
    # backend reuses the same names but assigns them the other way round
    # (inputs.XINPUT_MAPPING: X -> 0x134/BTN_WEST, Y -> 0x133/BTN_NORTH), so
    # the mapping has to follow the platform.
    _NORTH_BUTTON = 'Y' if inputs.WIN else 'X'
    _WEST_BUTTON = 'X' if inputs.WIN else 'Y'

    # evdev ABS_* codes, for the EVIOCGABS ioctl.
    _ABS_CODES = {
        'ABS_X': 0x00, 'ABS_Y': 0x01, 'ABS_Z': 0x02,
        'ABS_RX': 0x03, 'ABS_RY': 0x04, 'ABS_RZ': 0x05,
    }

    def __init__(self, max_reconnect=None, deadzone=25.0, padding=0.0):
        self.max_reconnect = max_reconnect
        self.deadzone = deadzone / 100.0
        self.padding = padding / 100.0
        self._axis_scale = self._default_axis_scales()
        self.reset_values()
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._connected = False
        self._reconnect_count = 0
        self.start_monitoring()

    def reset_values(self):
        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

    def start_monitoring(self):
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_controller)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()

    def stop_monitoring(self):
        self._stop_event.set()
        self._connected = False
        self.reset_values()
        if self._monitor_thread:
            # get_gamepad() blocks until the next pad event, so an idle
            # controller would make a plain join() wait forever. The thread is
            # a daemon, so leaving it parked in that read is harmless.
            self._monitor_thread.join(timeout=1.0)

    def _apply_deadzone(self, x):
        ax = abs(x)
        if ax < self.deadzone:
            return 0.0
        scaled = (ax - self.deadzone) / (1.0 - self.deadzone)
        return scaled if x > 0 else -scaled

    def _apply_padding(self, x):
        if self.padding <= 0:
            return x
        x = x / (1.0 - self.padding)
        return max(-1.0, min(1.0, x))

    def _condition(self, x):
        return self._apply_padding(self._apply_deadzone(x))

    def read(self):
        if not self._connected:
            self.reset_values()
        return {
            'LeftJoystickY': self._condition(self.LeftJoystickY),
            'LeftJoystickX': self._condition(self.LeftJoystickX),
            'RightJoystickY': self._condition(self.RightJoystickY),
            'RightJoystickX': self._condition(self.RightJoystickX),
            'LeftTrigger': self._condition(self.LeftTrigger),
            'RightTrigger': self._condition(self.RightTrigger),
            'LeftBumper': self.LeftBumper,
            'RightBumper': self.RightBumper,
            'A': self.A,
            'X': self.X,
            'Y': self.Y,
            'B': self.B,
            'LeftThumb': self.LeftThumb,
            'RightThumb': self.RightThumb,
            'Back': self.Back,
            'Start': self.Start,
            'LeftDPad': self.LeftDPad,
            'RightDPad': self.RightDPad,
            'UpDPad': self.UpDPad,
            'DownDPad': self.DownDPad
        }

    def _default_axis_scales(self):
        return {
            'ABS_X': self.MAX_JOY_VAL, 'ABS_Y': self.MAX_JOY_VAL,
            'ABS_RX': self.MAX_JOY_VAL, 'ABS_RY': self.MAX_JOY_VAL,
            'ABS_Z': self.MAX_TRIG_VAL, 'ABS_RZ': self.MAX_TRIG_VAL,
        }

    def _refresh_axis_scales(self):
        """Ask the driver for each axis' real range.

        Trigger full scale is not universal: the Xbox Series pad on the USB
        kernel driver reports 0..1023, not the 0..255 the XInput backend gives,
        so a hardcoded divisor makes read() return values well above 1.0.
        """
        scales = self._default_axis_scales()
        path = self._device_path()
        if fcntl is None or not path:
            self._axis_scale = scales
            return

        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            self._axis_scale = scales
            return
        try:
            for name, code in self._ABS_CODES.items():
                info = self._read_absinfo(fd, code)
                if info is None:
                    continue
                minimum, maximum = info
                # Signed axes rest at 0 and swing both ways (-32768..32767 ->
                # 32768); one-sided axes like the triggers run 0..max.
                full_scale = max(abs(minimum), abs(maximum) + 1) if minimum < 0 else abs(maximum)
                if full_scale > 0:
                    scales[name] = float(full_scale)
        finally:
            os.close(fd)
        self._axis_scale = scales

    @staticmethod
    def _read_absinfo(fd, abs_code):
        """EVIOCGABS(abs_code) -> (minimum, maximum), or None if unsupported."""
        size = struct.calcsize('6i')  # struct input_absinfo
        request = (2 << 30) | (size << 16) | (ord('E') << 8) | (0x40 + abs_code)
        buf = bytearray(size)
        try:
            fcntl.ioctl(fd, request, buf)
        except (OSError, ValueError):
            return None
        _value, minimum, maximum, _fuzz, _flat, _res = struct.unpack('6i', bytes(buf))
        if minimum == maximum:
            return None
        return minimum, maximum

    @staticmethod
    def _device_path():
        try:
            return inputs.devices.gamepads[0]._character_device_path
        except (IndexError, AttributeError):
            return None

    def _gamepad_present(self):
        return bool(inputs.devices.gamepads)

    def _rescan_devices(self):
        # inputs builds its device list once at import, so a controller plugged
        # in later is never seen unless the manager is rebuilt.
        try:
            inputs.devices = inputs.DeviceManager()
        except Exception:
            return False
        return self._gamepad_present()

    def _monitor_controller(self):
        while not self._stop_event.is_set():
            try:
                # get_gamepad() blocks until the pad is actually moved, so
                # connection state comes from the device list, not from traffic.
                if not self._connected and self._gamepad_present():
                    self._refresh_axis_scales()
                    self._connected = True
                    self._reconnect_count = 0
                events = get_gamepad()
                self._connected = True
                self._reconnect_count = 0  # Reset reconnection counter on successful connection
                for event in events:
                    self._process_event(event)
            except PermissionError:
                print("[ERROR] No read access to /dev/input/event*. Add this user to "
                      "the 'input' group (setup_jetson.sh does this) and log back in.")
                self._connected = False
                self.reset_values()
                self._stop_event.set()
            except UnpluggedError:
                # The pad was already gone when get_gamepad() looked it up.
                self._handle_disconnect()
            except OSError as err:
                # The pad vanished *during* the blocking read: the kernel drops
                # the event node and read() fails with ENODEV/EIO instead of
                # raising UnpluggedError.
                print(f"[JOYSTICK] Read from controller failed ({err}).")
                self._handle_disconnect()
            except Exception as err:  # noqa: BLE001 - the monitor must never die silently
                print(f"[JOYSTICK] Unexpected controller error ({type(err).__name__}: {err}).")
                self._handle_disconnect()

    def _handle_disconnect(self):
        """Drop to a safe state, then block until the pad is back."""
        if self._connected:
            print("[JOYSTICK] Controller disconnected. Attempting to reconnect...")
        self._connected = False
        self.reset_values()
        if not self._attempt_reconnect():
            if not self._stop_event.is_set():
                print("[ERROR] Maximum reconnection attempts reached. "
                      "Stopping controller monitoring.")
            self._stop_event.set()

    def _normalize(self, code, state):
        scale = self._axis_scale.get(code) or 1.0
        return max(-1.0, min(1.0, state / scale))

    def _process_event(self, event):
        if event.code == 'ABS_Y':
            self.LeftJoystickY = self._normalize(event.code, event.state)
        elif event.code == 'ABS_X':
            self.LeftJoystickX = self._normalize(event.code, event.state)
        elif event.code == 'ABS_RY':
            self.RightJoystickY = self._normalize(event.code, event.state)
        elif event.code == 'ABS_RX':
            self.RightJoystickX = self._normalize(event.code, event.state)
        elif event.code == 'ABS_Z':
            self.LeftTrigger = self._normalize(event.code, event.state)
        elif event.code == 'ABS_RZ':
            self.RightTrigger = self._normalize(event.code, event.state)
        elif event.code == 'BTN_TL':
            self.LeftBumper = event.state
        elif event.code == 'BTN_TR':
            self.RightBumper = event.state
        elif event.code == 'BTN_SOUTH':
            self.A = event.state
        elif event.code == 'BTN_NORTH':
            setattr(self, self._NORTH_BUTTON, event.state)  # see _NORTH_BUTTON
        elif event.code == 'BTN_WEST':
            setattr(self, self._WEST_BUTTON, event.state)
        elif event.code == 'BTN_EAST':
            self.B = event.state
        elif event.code == 'BTN_THUMBL':
            self.LeftThumb = event.state
        elif event.code == 'BTN_THUMBR':
            self.RightThumb = event.state
        elif event.code == 'BTN_SELECT':
            self.Back = event.state
        elif event.code == 'BTN_START':
            self.Start = event.state
        elif event.code == 'ABS_HAT0X':
            # Most pads (this Xbox Series controller included) report the D-pad
            # as a hat axis rather than as buttons: -1 left, +1 right, 0
            # released. The Windows XInput backend emulates the same two axes
            # with the same signs, so one branch covers both platforms.
            self.LeftDPad = 1 if event.state < 0 else 0
            self.RightDPad = 1 if event.state > 0 else 0
        elif event.code == 'ABS_HAT0Y':
            self.UpDPad = 1 if event.state < 0 else 0
            self.DownDPad = 1 if event.state > 0 else 0
        elif event.code == 'BTN_TRIGGER_HAPPY1':
            # Kept for pads that do report the D-pad as discrete buttons.
            self.LeftDPad = event.state
        elif event.code == 'BTN_TRIGGER_HAPPY2':
            self.RightDPad = event.state
        elif event.code == 'BTN_TRIGGER_HAPPY3':
            self.UpDPad = event.state
        elif event.code == 'BTN_TRIGGER_HAPPY4':
            self.DownDPad = event.state

    def _attempt_reconnect(self):
        wait_delay = 3

        # Give the kernel a moment to tear the event node down. Probing too
        # early can still list the dead device, which would send us straight
        # back into a failing read and spin this loop.
        if self._stop_event.wait(0.5):
            return False

        while not self._stop_event.is_set():
            if self.max_reconnect is not None and self._reconnect_count >= self.max_reconnect:
                return False
            # Probe the device list rather than get_gamepad(): a present-but-idle
            # pad would block that call until someone moved a stick.
            if self._rescan_devices():
                print("[JOYSTICK] Controller reconnected successfully!")
                # A different pad may have been plugged in; re-read its ranges.
                self._refresh_axis_scales()
                self._connected = True
                self._reconnect_count = 0
                return True

            self._reconnect_count += 1
            if self.max_reconnect is not None:
                remaining = self.max_reconnect - self._reconnect_count
                print(f"[JOYSTICK] Reconnection attempt {self._reconnect_count}/{self.max_reconnect} "
                      f"failed. {remaining} attempts remaining. "
                      f"Retrying in {wait_delay} seconds...")
            else:
                print(f"[JOYSTICK] Reconnection attempt {self._reconnect_count} "
                      f"failed. Retrying in {wait_delay} seconds...")
            # wait() instead of sleep() so stop_monitoring() is not stuck
            # behind a retry delay.
            if self._stop_event.wait(wait_delay):
                break

        return False


    def is_connected(self):
        return self._connected

    def __del__(self):
        # May run on a half-built object if __init__ raised, and during
        # interpreter shutdown; neither is worth a traceback.
        try:
            self.stop_monitoring()
        except Exception:
            pass

if __name__ == "__main__":
    # initialize the controller
    controller = XboxController()

    while True:
        # read values
        print(controller.read())
        # check connection status
        print(f"Connected: {controller.is_connected()}")

        time.sleep(0.1)
