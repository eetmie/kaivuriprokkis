"""Read a coherent latest Y8 frame through the native shared-memory ABI."""
import argparse
import ctypes as C
from pathlib import Path
import time


class Metadata(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "sequence", "host_monotonic_ns", "sensor_timestamp_us", "frame_timestamp_us",
        "frame_counter", "actual_exposure_us", "gain_level", "valid_metadata")]
    _fields_ += [("sdk_timestamp_ms", C.c_double)]
    _fields_ += [(name, C.c_uint32) for name in (
        "timestamp_domain", "width", "height", "stride", "ir_index", "pixel_bytes")]


class FrameReader:
    def __init__(self, path, library=None, capacity=640 * 480, max_age_s=.2):
        library = library or Path(__file__).resolve().parents[1] / "build/libkaivuri_frames.so"
        self.lib = C.CDLL(str(library))
        self.lib.kaivuri_frame_open.argtypes = [C.c_char_p]
        self.lib.kaivuri_frame_open.restype = C.c_void_p
        self.lib.kaivuri_frame_read.argtypes = [C.c_void_p, C.POINTER(Metadata), C.c_void_p, C.c_size_t]
        self.lib.kaivuri_frame_read.restype = C.c_int
        self.lib.kaivuri_frame_close.argtypes = [C.c_void_p]
        self.lib.kaivuri_frame_close.restype = None
        self.handle = self.lib.kaivuri_frame_open(str(path).encode())
        if not self.handle:
            raise OSError(f"cannot open native frame store: {path}")
        self.buffer = (C.c_uint8 * capacity)()
        self.max_age_s = max_age_s

    def read(self):
        metadata = Metadata()
        result = self.lib.kaivuri_frame_read(self.handle, C.byref(metadata), self.buffer, len(self.buffer))
        if result < 0:
            raise RuntimeError("invalid reader arguments or insufficient pixel capacity")
        if result == 0:
            return None
        age = (time.monotonic_ns() - metadata.host_monotonic_ns) * 1e-9
        if not 0 <= age <= self.max_age_s:
            return None
        # Memoryview avoids another pixel copy. Valid until this reader's next read().
        return metadata, memoryview(self.buffer).cast("B")[:metadata.pixel_bytes]

    def close(self):
        if self.handle:
            self.lib.kaivuri_frame_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--library")
    parser.add_argument("--capacity", type=int, default=640 * 480)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--save-pgm", type=Path)
    args = parser.parse_args()
    with FrameReader(args.path, args.library, args.capacity) as reader:
        end, previous, latest = time.monotonic() + args.seconds, 0, None
        while time.monotonic() < end:
            snapshot = reader.read()
            if snapshot and snapshot[0].sequence != previous:
                metadata, pixels = snapshot
                previous = metadata.sequence
                latest = (metadata, bytes(pixels)) if args.save_pgm else None
                print(" ".join(f"{name}={getattr(metadata, name)}" for name, _ in Metadata._fields_))
            time.sleep(.01)
        if not previous:
            raise SystemExit("No fresh frame received")
        if args.save_pgm and latest:
            m, pixels = latest
            args.save_pgm.write_bytes(f"P5\n{m.width} {m.height}\n255\n".encode() + pixels)


if __name__ == "__main__":
    main()
