#!/usr/bin/env python3
"""Publish or measure ROS2 image-topic load entirely inside WSL."""

import argparse
import math
import time
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, Image


def image_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def parse_topics(value: str) -> list[str]:
    topics = [item.strip() for item in value.split(",") if item.strip()]
    if not topics:
        raise argparse.ArgumentTypeError("expected at least one topic")
    return topics


class ImageStressPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("ros2_image_stress_publisher")
        self.args = args
        self.publishers_by_topic = {}
        self.seq = 0
        self.started_at = time.monotonic()
        self.sent_counts = defaultdict(int)
        self.sent_bytes = defaultdict(int)

        if args.msg_type == "image":
            self.payload = self._make_rgb_payload(args.width, args.height)
            self.bytes_per_msg = len(self.payload)
            msg_name = "sensor_msgs/Image rgb8"
        else:
            self.payload = bytes((i * 13) % 256 for i in range(args.compressed_bytes))
            self.bytes_per_msg = len(self.payload)
            msg_name = "sensor_msgs/CompressedImage"

        for topic in args.topics:
            msg_cls = Image if args.msg_type == "image" else CompressedImage
            self.publishers_by_topic[topic] = self.create_publisher(msg_cls, topic, image_qos())

        period_s = 1.0 / args.hz
        self.create_timer(period_s, self._tick)
        self.create_timer(1.0, self._report)
        total_mbps = (self.bytes_per_msg * len(args.topics) * args.hz) / 1_000_000
        self.get_logger().info(
            f"Publishing {msg_name}: topics={len(args.topics)} "
            f"size={self.bytes_per_msg} bytes hz={args.hz:.3f} "
            f"target_payload={total_mbps:.2f} MB/s"
        )

    @staticmethod
    def _make_rgb_payload(width: int, height: int) -> bytes:
        row = bytearray()
        for x in range(width):
            row.extend((x % 256, (255 - x) % 256, 96))
        return bytes(row) * height

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()
        self.seq += 1

        for topic, pub in self.publishers_by_topic.items():
            if self.args.msg_type == "image":
                msg = Image()
                msg.header.stamp = now
                msg.header.frame_id = "wsl_stress"
                msg.height = self.args.height
                msg.width = self.args.width
                msg.encoding = "rgb8"
                msg.is_bigendian = 0
                msg.step = self.args.width * 3
                msg.data = self.payload
            else:
                msg = CompressedImage()
                msg.header.stamp = now
                msg.header.frame_id = "wsl_stress"
                msg.format = self.args.compressed_format
                msg.data = self.payload

            pub.publish(msg)
            self.sent_counts[topic] += 1
            self.sent_bytes[topic] += self.bytes_per_msg

    def _report(self) -> None:
        elapsed = max(1e-9, time.monotonic() - self.started_at)
        total_count = sum(self.sent_counts.values())
        total_bytes = sum(self.sent_bytes.values())
        self.get_logger().info(
            f"sent total_rate={total_count / elapsed:.2f} msg/s "
            f"payload={total_bytes / elapsed / 1_000_000:.2f} MB/s"
        )


class ImageStressMeter(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("ros2_image_stress_meter")
        self.args = args
        self.counts = defaultdict(int)
        self.bytes = defaultdict(int)
        self.times = defaultdict(lambda: deque(maxlen=200))
        self.started_at = time.monotonic()

        for topic in args.topics:
            msg_cls = Image if args.msg_type == "image" else CompressedImage
            self.create_subscription(
                msg_cls,
                topic,
                lambda msg, topic=topic: self._on_msg(topic, msg),
                image_qos(),
            )

        self.create_timer(1.0, self._report)
        self.get_logger().info(
            f"Measuring {args.msg_type} topics: {', '.join(args.topics)}"
        )

    def _on_msg(self, topic: str, msg) -> None:
        now = time.monotonic()
        self.counts[topic] += 1
        self.times[topic].append(now)
        self.bytes[topic] += len(msg.data)

    @staticmethod
    def _window_rate(times: deque[float]) -> float:
        if len(times) < 2:
            return 0.0
        elapsed = times[-1] - times[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(times) - 1) / elapsed

    def _report(self) -> None:
        elapsed = max(1e-9, time.monotonic() - self.started_at)
        total_count = sum(self.counts.values())
        total_bytes = sum(self.bytes.values())
        parts = []
        for topic in self.args.topics:
            rate = self._window_rate(self.times[topic])
            parts.append(f"{topic}={rate:.2f}Hz")
        self.get_logger().info(
            f"recv {' '.join(parts)} total={total_count / elapsed:.2f} msg/s "
            f"payload={total_bytes / elapsed / 1_000_000:.2f} MB/s"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("publish", "measure"),
        required=True,
        help="publish test image load or measure existing test topics",
    )
    parser.add_argument(
        "--topics",
        type=parse_topics,
        default="/stress/camera0/rgb_decoded,/stress/camera1/rgb_decoded",
        help="comma-separated topic list",
    )
    parser.add_argument("--msg-type", choices=("image", "compressed"), default="image")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--compressed-bytes", type=int, default=120_000)
    parser.add_argument("--compressed-format", default="h264")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not math.isfinite(args.hz) or args.hz <= 0.0:
        raise SystemExit("--hz must be positive")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.compressed_bytes <= 0:
        raise SystemExit("--compressed-bytes must be positive")

    rclpy.init()
    node = ImageStressPublisher(args) if args.mode == "publish" else ImageStressMeter(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
