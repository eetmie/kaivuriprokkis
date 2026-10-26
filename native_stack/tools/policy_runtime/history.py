"""Bounded raw-frame history on the host monotonic clock, independent of inference."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import struct


class NotReady(RuntimeError):
    """Fresh current observations or real historical frames are not available."""


@dataclass(frozen=True)
class Frame:
    metadata: dict
    pixels: bytes

    @property
    def timestamp(self):
        return self.metadata["host_monotonic_ns"]


class FrameHistory:
    def __init__(self, views: int, offsets_ms=(1000, 0), tolerance_ms=60,
                 max_bytes=128 * 1024 * 1024):
        if views < 1 or max_bytes < 1:
            raise ValueError("positive views and memory budget required")
        values = tuple(float(x) for x in offsets_ms)
        if not values or any(not math.isfinite(x) or not 0 <= x <= 5000 for x in values):
            raise ValueError("history offsets must be finite and between 0 and 5000 ms")
        if not math.isfinite(tolerance_ms) or not 0 < tolerance_ms <= 1000:
            raise ValueError("history tolerance must be in (0, 1000] ms")
        self.offsets = tuple(sorted({int(x * 1e6) for x in values} | {0}, reverse=True))
        self.tolerance = int(tolerance_ms * 1e6)
        self.retention = self.offsets[0] + self.tolerance + 500_000_000
        self.frames = [deque() for _ in range(views)]
        self.max_bytes, self.bytes = max_bytes, 0
        self.evictions = 0

    def push(self, view: int, frame: Frame):
        q = self.frames[view]
        m = frame.metadata
        if (m["width"] <= 0 or m["height"] <= 0 or m["ir_index"] not in (1, 2)
                or m["stride"] != m["width"]
                or m["pixel_bytes"] != m["width"] * m["height"]
                or len(frame.pixels) != m["pixel_bytes"]):
            raise ValueError("packed Y8 IR frames required")
        if m["valid_metadata"] & 3 != 3:
            raise ValueError("camera sensor/readout timestamps required")
        if q:
            previous = q[-1]
            if (m["sequence"] == previous.metadata["sequence"]
                    and frame.timestamp == previous.timestamp):
                return
            if (frame.timestamp <= previous.timestamp
                    or m["sequence"] <= previous.metadata["sequence"]):
                raise ValueError("camera clock/sequence reset; restart the probe")
            if any(m[k] != previous.metadata[k] for k in ("width", "height", "ir_index")):
                raise ValueError("camera stream configuration changed")
        if len(frame.pixels) > self.max_bytes:
            raise ValueError("one frame exceeds history memory budget")
        q.append(frame)
        self.bytes += len(frame.pixels)
        for queue in self.frames:
            while len(queue) > 1 and frame.timestamp - queue[0].timestamp > self.retention:
                self.bytes -= len(queue.popleft().pixels)
        while self.bytes > self.max_bytes:
            oldest = min((queue for queue in self.frames if queue), key=lambda x: x[0].timestamp)
            self.bytes -= len(oldest.popleft().pixels)
            self.evictions += 1

    def select(self, now_ns: int, max_age_ns=200_000_000, max_skew_ns=40_000_000):
        if any(not q for q in self.frames):
            raise NotReady("waiting for all cameras")
        current = [q[-1] for q in self.frames]
        if any(not 0 <= now_ns - f.timestamp <= max_age_ns for f in current):
            raise NotReady("current camera frame stale or in the future")
        selected = []
        for offset in self.offsets:
            slot = []
            for q, latest in zip(self.frames, current):
                target = latest.timestamp - offset
                # Never choose a newer frame to make missing history appear available.
                match = next((f for f in reversed(q) if f.timestamp <= target), None)
                if match is None or target - match.timestamp > self.tolerance:
                    raise NotReady(f"no real frame near requested history offset {offset / 1e6:g} ms")
                slot.append(match)
            times = [f.timestamp for f in slot]
            if max(times) - min(times) > max_skew_ns:
                raise NotReady("camera arrival skew exceeds limit")
            selected.append(slot)
        return selected


def robot_state(status: dict, now_ns: int, max_age_ns=50_000_000) -> dict:
    if not status.get("imu_healthy") or not status.get("fk_valid"):
        raise NotReady("IMU/FK unhealthy")
    timestamp = status["imu_host_monotonic_ns"]
    if not 0 <= now_ns - timestamp <= max_age_ns:
        raise NotReady("IMU observation stale or in the future")
    angles = status.get("angles_rad")
    if not isinstance(angles, list) or len(angles) != 4 or not all(math.isfinite(x) for x in angles):
        raise ValueError("four finite joint angles required")
    # Same float32 rad -> deg boundary as the dataset exporter; keep raw status too.
    def f32(x):
        return struct.unpack("<f", struct.pack("<f", x))[0]
    degrees = [f32(f32(x) * f32(180 / math.pi)) for x in angles]
    return {"names": ["slew", "lift", "tilt", "scoop"], "units": "degrees",
            "values": degrees, "imu_host_monotonic_ns": timestamp}
