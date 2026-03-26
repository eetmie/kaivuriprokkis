"""Compatibility import for the IMU stream reader module name."""

try:
    from .usb_serial_reader import USBSerialReader
except ImportError:  # pragma: no cover - script mode fallback
    from usb_serial_reader import USBSerialReader
