#!/usr/bin/env python3
"""ROS 2 bridge for the Kaivuri native stack: the topics the Isaac sim publishes, so one viewer
(and later one client) works on either. Read-only for now: STATUS polls only, no DRIVE.

  live   [--socket S] [--frames DIR] [--calibration CAL]     the running controller and cameras
  replay SESSION [--episode N] [--calibration CAL]           a recorded take, looped at real time

Published (wall-clock stamps; frames from the shared URDF, sim-scratchpad ros2/urdf):
  /joint_states                      URDF joint positions (rad) for robot_state_publisher
  /excavator/joint_angles            excavator-vla contract v1: [slew, lift, tilt, scoop] in [-1, 1]
  /excavator/feedback_healthy        controller not faulted, IMUs healthy, FK valid, angles in bounds
  /excavator/adapter_info            {interface_id, calibration_id} from the profile, 1 Hz
  /excavator/cameras/<name>/image_raw  IR mono8 (the contract asks rgb8; tripling is left to clients)
  /tf_static                         camera depth frames, with --calibration (sand_height.py calibrate)

Run with the system ROS (source /opt/ros/jazzy/setup.bash); needs no ROS package build.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension, String
from tf2_ros import StaticTransformBroadcaster

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))

AXES = ('slew', 'lift', 'tilt', 'scoop')
URDF_JOINTS = ('revolute_carriage', 'revolute_lift', 'revolute_tilt', 'revolute_tool')
# URDF position = native angle + offset, same signs. From the two geometries (native boom 0.468, arm 0.240,
# tip [-0.009, 0, -0.1335]; URDF stick line 1.6 deg below its x axis, lip 0.2 deg off the native tip).
URDF_OFFSET = np.array([0., 0., -.02813, .00355])
# Native link frames -> URDF link frames, for the camera mounts (same geometry).
CABIN_SHIFT = np.array([0, 0, .014])  # slew-link origin sits 14 mm above upper_carriage's (boom pivot 64.5 vs 78.5 mm)
ARM_TURN = .02813  # tilt_boom is turned this far about +y from the native arm link
CAMERA_KEYS = {'cabin': 'cam2', 'arm': 'cam1'}  # recorder keys
CAMERA_PARENTS = {'cabin': ('upper_carriage', 0), 'arm': ('tilt_boom', 2)}  # URDF link, native link


def digest(value):
    """excavator_stack.contracts.digest: the shared stack checks these IDs."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def matrix_quat(R):
    """Rotation matrix -> (x, y, z, w)."""
    w = math.sqrt(max(0., 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0., 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0., 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0., 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def camera_transforms(calibration, profile):
    """(name, parent, R, t) of each calibrated camera's depth frame in its URDF link."""
    cal = json.loads(Path(calibration).read_text()); out = []
    for name, (parent, link) in CAMERA_PARENTS.items():
        c = cal['cameras'].get(CAMERA_KEYS[name])
        if c is None:
            continue
        if c['link'] != link:
            raise ValueError(f'{name} camera is calibrated on native link {c["link"]}, expected {link}')
        # shortcut: cabin_shift_xy_m (the overlap step's fudge for the wrist camera) is dropped; it rotates with the slew, not the link
        R, t = np.asarray(c['rotation_camera_to_link']), np.asarray(c['translation_camera_to_link_m'])
        if profile['cameras_native'][name].get('rotation_deg') == 180:  # published pixels are turned: turn the frame too
            R = R @ np.diag([-1., -1, 1])
        if parent == 'tilt_boom':
            R, t = ry(ARM_TURN) @ R, ry(ARM_TURN) @ t
        else:
            t = t + CABIN_SHIFT
        out.append((name, parent, R, t))
    return out


class Live:
    def __init__(self, a, profile):
        from client import Client
        from frame_reader import FrameReader
        self.client = Client(a.socket)
        self.readers = {}
        for name, cam in profile['cameras_native'].items():
            path = Path(a.frames) / f'{cam["serial"]}-ir{cam["ir_index"]}.frame'
            if path.exists():
                self.readers[name] = FrameReader(path, a.library, max_age_s=.5)

    def status(self):
        return self.client.status()

    def image(self, name):
        """(age s, width, height, bytes) of the newest frame, or None."""
        got = self.readers[name].read() if name in self.readers else None
        if got is None:
            return None
        m, pixels = got
        return (time.monotonic_ns() - m.host_monotonic_ns) * 1e-9, m.width, m.height, bytes(pixels)


class Replay:
    def __init__(self, a, profile):
        from export_episodes import committed_episodes, samples
        episode = committed_episodes(a.session)[a.episode]
        rows = [r for r in samples(episode) if 'status' in r]
        self.status_rows = [r['status'] for r in rows]
        self.times = np.array([s['host_monotonic_ns'] for s in self.status_rows], np.int64)
        self.frame_at = np.maximum.accumulate([r.get('frame_index', -1) for r in rows])  # newest frame by then
        self.images = {}
        for name, key in CAMERA_KEYS.items():
            info = json.loads((a.session / f'{key}.depth.json').read_text())['ir_intrinsics']
            path = episode / f'{key}.y8'
            if path.exists():
                self.images[name] = np.memmap(path, np.uint8, 'r').reshape(-1, info['height'], info['width'])
        self.start = time.monotonic_ns()

    def index(self):
        span = self.times[-1] - self.times[0]
        return int(np.searchsorted(self.times, self.times[0] + (time.monotonic_ns() - self.start) % span))

    def status(self):
        return self.status_rows[min(self.index(), len(self.status_rows) - 1)]

    def image(self, name):
        i = self.frame_at[min(self.index(), len(self.frame_at) - 1)]
        if name not in self.images or i < 0:
            return None
        frame = self.images[name][min(i, len(self.images[name]) - 1)]
        return 0., frame.shape[1], frame.shape[0], frame.tobytes()


class Bridge(Node):
    def __init__(self, source, profile, a):
        super().__init__('kaivuri_bridge')
        self.source, self.profile = source, profile
        norm = profile['normalization']['joint_angles']
        self.lo, self.hi, self.sign = np.array(norm['min']), np.array(norm['max']), np.array(profile['native_angle_sign'])
        names = ('slew', 'lift', 'tilt', 'scoop')
        self.info = json.dumps({'interface_id': digest({'version': 1, 'names': names, 'cameras': profile['cameras'],
                                                       'state': 'bounded_position_or_track_velocity_minus1_plus1',
                                                       'action': 'signed_actuation_minus1_plus1_zero_neutral'}),
                                'calibration_id': digest(profile)})
        self.pub_joints = self.create_publisher(JointState, 'joint_states', 10)
        self.pub_angles = self.create_publisher(Float32MultiArray, 'excavator/joint_angles', 10)
        self.pub_health = self.create_publisher(Bool, 'excavator/feedback_healthy', 10)
        self.pub_info = self.create_publisher(String, 'excavator/adapter_info', 10)
        self.pub_images = {n: self.create_publisher(Image, f'excavator/cameras/{n}/image_raw', 2) for n in profile['cameras']}
        self.problem = None
        self.create_timer(1 / a.state_hz, self.poll_state)
        self.create_timer(1 / a.camera_hz, self.poll_images)
        self.create_timer(1., lambda: self.pub_info.publish(String(data=self.info)))
        if a.calibration:
            self.static = StaticTransformBroadcaster(self)
            msgs = []
            for name, parent, R, t in camera_transforms(a.calibration, profile):
                m = TransformStamped(); m.header.stamp = self.get_clock().now().to_msg()
                m.header.frame_id, m.child_frame_id = parent, f'{name}_camera'
                m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, t)
                q = matrix_quat(R)
                m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = q
                msgs.append(m)
            self.static.sendTransform(msgs)

    def poll_state(self):
        s = self.source.status()
        stamp = self.get_clock().now().to_msg()
        q = np.array(s['angles_rad'], float)
        q[0] = math.remainder(q[0], 2 * math.pi)
        js = JointState(); js.header.stamp = stamp; js.name = list(URDF_JOINTS)
        js.position = (q + URDF_OFFSET).tolist(); self.pub_joints.publish(js)
        v = 2 * (self.sign * q - self.lo) / (self.hi - self.lo) - 1
        problem = ('controller ' + s.get('reason', '') if s.get('state') == 'fault' else
                   'IMU unhealthy' if not s.get('imu_healthy') else 'FK invalid' if not s.get('fk_valid') else
                   'angles out of bounds: ' + ', '.join(n for n, x in zip(AXES, v) if abs(x) > 1) if (np.abs(v) > 1).any() else None)
        if problem != self.problem:
            (self.get_logger().warning(problem) if problem else self.get_logger().info('feedback healthy'))
            self.problem = problem
        self.pub_health.publish(Bool(data=problem is None))
        if problem is None:  # the contract rejects out-of-range state rather than clipping it
            m = Float32MultiArray(data=v.astype(np.float32).tolist())
            m.layout.dim = [MultiArrayDimension(label=','.join(AXES), size=4, stride=4)]
            self.pub_angles.publish(m)

    def poll_images(self):
        now = self.get_clock().now()
        for name, pub in self.pub_images.items():
            got = self.source.image(name)
            if got is None:
                continue
            age, w, h, data = got
            m = Image(); m.header.stamp = (now - rclpy.duration.Duration(seconds=age)).to_msg()
            m.header.frame_id = f'{name}_camera'; m.height, m.width, m.encoding, m.step, m.data = h, w, 'mono8', w, data
            pub.publish(m)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    live, replay = sub.add_parser('live'), sub.add_parser('replay')
    from client import DEFAULT_SOCKET, RUNTIME_DIR
    live.add_argument('--socket', default=DEFAULT_SOCKET)
    live.add_argument('--frames', default=f'{RUNTIME_DIR}/cameras', help='kaivuri-camera frame store directory')
    live.add_argument('--library', default=None, help='libkaivuri_frames.so (default: native_stack/build)')
    replay.add_argument('session', type=Path); replay.add_argument('--episode', type=int, default=0)
    for q in (live, replay):
        q.add_argument('--profile', type=Path, default=Path(__file__).with_name('kaivuri.profile.json'))
        q.add_argument('--calibration', type=Path, help='sand_height.py calibration: publishes the camera frames')
        q.add_argument('--state-hz', type=float, default=50)
        q.add_argument('--camera-hz', type=float, default=5, help='IR image rate (WiFi viewers: keep it low)')
    a = p.parse_args()
    profile = json.loads(a.profile.read_text())
    source = (Live if a.mode == 'live' else Replay)(a, profile)
    rclpy.init()
    node = Bridge(source, profile, a)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():  # a timer that fires during SIGTERM shutdown publishes into a closed context
            raise
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
