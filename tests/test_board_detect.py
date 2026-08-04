"""Board auto-detection and PCA9685 I2C bus resolution.

Covers the detect() branches called out as untested in
test_simple_drive_profiles.py, plus the bus-resolution path that keeps the
Pi default (bus 1) from leaking onto the Jetson (bus 7).
"""

import collections
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import platform  # noqa: E402

from modules import board  # noqa: E402
from modules.pwm.constants import resolve_i2c_bus  # noqa: E402

_Uname = collections.namedtuple("U", "system node release version machine")


def _fake_uname(hostname: str):
    return _Uname("Linux", hostname, "", "", "aarch64")


class DetectTests(unittest.TestCase):
    """detect() must identify the board or refuse to guess."""

    def _detect_with(self, hostname: str, model: str | None):
        """Run detect() with a stubbed hostname and device-tree model."""
        with mock.patch.object(platform, "uname", return_value=_fake_uname(hostname)), \
             mock.patch.object(board, "Path") as fake_path:
            if model is None:
                fake_path.return_value.read_text.side_effect = OSError("no device-tree")
            else:
                fake_path.return_value.read_text.return_value = model
            return board.detect()

    def test_hostname_identifies_jetson(self):
        self.assertEqual(self._detect_with("my-jetson-01", None), "jetson")

    def test_hostname_identifies_raspberry_pi(self):
        self.assertEqual(self._detect_with("raspberrypi", None), "rpi")

    def test_device_tree_identifies_jetson(self):
        model = "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super"
        self.assertEqual(self._detect_with("localhost.localdomain", model), "jetson")

    def test_device_tree_identifies_raspberry_pi(self):
        self.assertEqual(self._detect_with("localhost", "Raspberry Pi 4 Model B"), "rpi")

    def test_hostname_wins_over_device_tree(self):
        self.assertEqual(self._detect_with("my-jetson", "Raspberry Pi 4 Model B"), "jetson")

    def test_unknown_board_raises_instead_of_guessing(self):
        # Regression guard: this used to return "rpi", silently selecting I2C
        # bus 1 -- which on an Orin Nano is the kernel-owned ina3221.
        with self.assertRaises(board.BoardDetectionError):
            self._detect_with("some-laptop", None)

    def test_detection_error_is_a_value_error(self):
        """Existing `except ValueError` handlers must keep catching this."""
        with self.assertRaises(ValueError):
            self._detect_with("some-laptop", "QEMU Standard PC")


class BusResolutionTests(unittest.TestCase):
    """The PCA9685 bus must come from the profile, not a hardcoded default."""

    def test_explicit_bus_passes_through(self):
        self.assertEqual(resolve_i2c_bus(3), 3)

    def test_explicit_zero_is_not_treated_as_unset(self):
        self.assertEqual(resolve_i2c_bus(0), 0)

    def test_none_resolves_from_jetson_profile(self):
        with mock.patch.object(board, "detect", return_value="jetson"):
            self.assertEqual(resolve_i2c_bus(None), 7)

    def test_none_resolves_from_rpi_profile(self):
        with mock.patch.object(board, "detect", return_value="rpi"):
            self.assertEqual(resolve_i2c_bus(None), 1)

    def test_profiles_disagree_on_bus(self):
        """Guards against both profiles collapsing to one hardcoded value."""
        self.assertNotEqual(
            board.resolve_profile("rpi")["pwm_i2c_bus"],
            board.resolve_profile("jetson")["pwm_i2c_bus"],
        )


if __name__ == "__main__":
    unittest.main()
