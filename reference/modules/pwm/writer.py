"""Low-level direct PCA9685 I2C writer."""

from __future__ import annotations

import threading
import time

from .constants import PCA9685_I2C_ADDRESS, PCA9685_I2C_BUS, resolve_i2c_bus
from .errors import PWMHardwareIOError


def _open_smbus(bus_number: int | None = None):
    """Open the PCA9685 I2C bus via smbus2 for low-overhead direct writes.

    ``bus_number=None`` resolves from the active board profile.
    """
    import smbus2

    return smbus2.SMBus(resolve_i2c_bus(bus_number))


class DirectPWMWriter:
    """Direct I2C writer for PCA9685 using native 12-bit counts."""

    LED0_ON_L = 0x06
    MODE1 = 0x00
    MODE2 = 0x01
    PRE_SCALE = 0xFE

    SLEEP = 0x10
    RESTART = 0x80
    AI = 0x20

    def __init__(self, i2c_bus, address: int = PCA9685_I2C_ADDRESS,
                 min_channel: int = 0, max_channel: int = 15,
                 frequency: int = 200, bus_number: int | None = None):
        if not (0 <= min_channel <= max_channel < 16):
            raise ValueError(
                f"Invalid PWM channel range [{min_channel}, {max_channel}] - expected 0 <= min <= max < 16"
            )
        self._i2c = i2c_bus
        self._addr = address
        self._bus_number = bus_number  # diagnostics only; None = unknown
        self._min_ch = min_channel
        self._max_ch = max_channel
        self._num_channels = max_channel - min_channel + 1
        self._lock = threading.Lock()
        self._uses_busio = all(hasattr(i2c_bus, name) for name in ("try_lock", "writeto", "unlock"))
        self._uses_smbus = all(hasattr(i2c_bus, name) for name in ("write_byte_data", "write_i2c_block_data"))
        if not (self._uses_busio or self._uses_smbus):
            raise TypeError(
                "i2c_bus must be either busio.I2C-like (try_lock/writeto/unlock) "
                "or SMBus-like (write_byte_data/write_i2c_block_data)"
            )

        self._buf = bytearray(1 + 4 * self._num_channels)
        self._buf[0] = self.LED0_ON_L + (min_channel * 4)
        self._counts = [0] * 16

        self._init_chip(frequency)

    def _target(self) -> str:
        """Human-readable bus/address for error messages."""
        bus = f"i2c-{self._bus_number}" if self._bus_number is not None else "i2c-?"
        return f"{bus} address 0x{self._addr:02X}"

    def _write_reg(self, reg: int, value: int):
        try:
            if self._uses_smbus:
                with self._lock:
                    self._i2c.write_byte_data(self._addr, reg, value)
                return

            while not self._i2c.try_lock():
                time.sleep(0)
            try:
                self._i2c.writeto(self._addr, bytes([reg, value]))
            finally:
                self._i2c.unlock()
        except Exception as exc:
            raise PWMHardwareIOError(
                f"Failed to write PCA9685 register 0x{reg:02X} on {self._target()}: {exc}"
            ) from exc

    def _init_chip(self, frequency: int):
        self._write_reg(self.MODE1, self.SLEEP | self.AI)
        self._write_reg(self.MODE2, 0x04)

        prescale = int(25_000_000 / (4096 * frequency) - 1)
        prescale = max(3, min(255, prescale))
        self._write_reg(self.PRE_SCALE, prescale)

        self._write_reg(self.MODE1, self.AI)
        time.sleep(0.005)
        self._write_reg(self.MODE1, self.AI | self.RESTART)

        self.frequency = frequency

    def sleep(self):
        self._write_reg(self.MODE1, self.SLEEP | self.AI)

    def wake(self):
        self._write_reg(self.MODE1, self.AI)
        time.sleep(0.005)
        self._write_reg(self.MODE1, self.AI | self.RESTART)

    def set_channel(self, channel: int, count: int):
        if not 0 <= channel < 16:
            raise ValueError(f"PWM channel {channel} out of range [0, 15]")
        count = int(count)
        self._counts[channel] = max(0, min(0x0FFF, count))

    def set_channel_range(self, min_channel: int, max_channel: int):
        if not (0 <= min_channel <= max_channel < 16):
            raise ValueError(
                f"Invalid PWM channel range [{min_channel}, {max_channel}] - expected 0 <= min <= max < 16"
            )
        self._min_ch = min_channel
        self._max_ch = max_channel
        self._num_channels = max_channel - min_channel + 1
        self._buf = bytearray(1 + 4 * self._num_channels)
        self._buf[0] = self.LED0_ON_L + (min_channel * 4)

    def flush(self):
        buf = self._buf
        counts = self._counts
        min_ch = self._min_ch

        for i in range(self._num_channels):
            count = counts[min_ch + i]
            if count >= 0x0FFF:
                on_val, off_val = 0x1000, 0
            elif count <= 0:
                on_val, off_val = 0, 0x1000
            else:
                on_val = 0
                off_val = count

            offset = 1 + i * 4
            buf[offset] = on_val & 0xFF
            buf[offset + 1] = (on_val >> 8) & 0xFF
            buf[offset + 2] = off_val & 0xFF
            buf[offset + 3] = (off_val >> 8) & 0xFF

        try:
            if self._uses_smbus:
                with self._lock:
                    data = buf[1:]
                    for offset in range(0, len(data), 32):
                        self._i2c.write_i2c_block_data(
                            self._addr,
                            buf[0] + offset,
                            list(data[offset:offset + 32]),
                        )
                return

            while not self._i2c.try_lock():
                time.sleep(0)
            try:
                self._i2c.writeto(self._addr, buf)
            finally:
                self._i2c.unlock()
        except Exception as exc:
            raise PWMHardwareIOError(
                f"Failed to flush PCA9685 channel buffer on {self._target()}: {exc}"
            ) from exc
