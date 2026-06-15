import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.pwm.writer import DirectPWMWriter  # noqa: E402


class _FakeSmbus:
    def __init__(self):
        self.reg_writes = []
        self.block_writes = []

    def write_byte_data(self, addr, reg, val):
        self.reg_writes.append((addr, reg, val))

    def write_i2c_block_data(self, addr, reg, data):
        self.block_writes.append((addr, reg, list(data)))


class DirectPWMWriterTests(unittest.TestCase):
    def test_set_channel_uses_native_12_bit_counts(self):
        bus = _FakeSmbus()
        writer = DirectPWMWriter(bus, min_channel=0, max_channel=2, frequency=100)

        writer.set_channel(0, 2048)
        writer.set_channel(1, -1)
        writer.set_channel(2, 5000)
        writer.flush()

        _, reg, data = bus.block_writes[-1]
        self.assertEqual(reg, DirectPWMWriter.LED0_ON_L)
        self.assertEqual(data[0:4], [0, 0, 0x00, 0x08])
        self.assertEqual(data[4:8], [0, 0, 0, 0x10])
        self.assertEqual(data[8:12], [0, 0x10, 0, 0])


if __name__ == "__main__":
    unittest.main()
