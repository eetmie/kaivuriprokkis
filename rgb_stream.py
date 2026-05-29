#!/usr/bin/env python3
"""RealSense D435i stream stress test — color + dummy depth, camera timestamps.

Depth is kept at minimum resolution/fps purely to lock color sensor sync.
Prints per-frame camera timestamps, inter-frame deltas, and host latency.
Ctrl+C to stop.
"""
import time
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)

pipeline.start(config)
print("Streams: color 640x480@30fps  depth 424x240@6fps (sync anchor)")
print(f"{'frame':>7}  {'cam_ts_ms':>14}  {'delta_ms':>8}  {'host_lag_ms':>11}  ts_domain")
print("-" * 65)

prev_color_ts = None
frame_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue

        cam_ts   = color.get_timestamp()                        # ms, camera clock
        domain   = color.get_frame_timestamp_domain()
        host_now = time.time() * 1000.0                         # ms, host clock
        host_lag = host_now - cam_ts

        delta = cam_ts - prev_color_ts if prev_color_ts is not None else 0.0
        prev_color_ts = cam_ts
        frame_count += 1

        print(
            f"{frame_count:>7}  {cam_ts:>14.3f}  {delta:>8.2f}  {host_lag:>11.2f}  {domain}"
        )
except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    print(f"\nStopped after {frame_count} color frames.")
