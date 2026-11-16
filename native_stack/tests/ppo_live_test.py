"""ppo_live.py against a simulated controller: the packaged bundle (verified at load) tracks a
circle and stick commands on an ideal plant, and the stops disarm. No devices."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

NATIVE = Path(sys.argv[1])
sys.path.insert(0, str(NATIVE / "tools"))
import client  # noqa: E402
import ppo_live  # noqa: E402
from ppo_policy import Bundle, joint_state  # noqa: E402
from vla_operator import ABS_RZ, ABS_X, BTN_A  # noqa: E402

BUNDLE = NATIVE / "ppo_bundles/controller_pivot"
HOME = np.array([-0.5498, 1.2549, -0.7540, 0.0])


class Plant:
    """kaivuri-control's client interface over joints that turn at valve x full-valve speed."""

    def __init__(self, bundle, response=1.0):
        self.sensors, self.limits, self.response = bundle.sensors, np.array(bundle.contract["joint_speed_limits_rad_s"]), response
        self.q, self.rate, self.valves, self.applied = HOME.copy(), np.zeros(3), np.zeros(3), np.zeros(3)
        self.state, self.ticks, self.armed_for, self.sent = "disarmed", 0, 0, []

    def send(self, op, steps=(), *, flags=0, **_):
        self.sent.append(op)
        if op == client.DISARM:
            self.state, self.valves = "disarmed", np.zeros(3)
        elif op == client.DRIVE:
            self.valves = np.array(steps[0][1:4])
            assert flags & client.DEADMAN and steps[0][0] == steps[0][4] == steps[0][5] == 0
            if flags & client.ARM and self.state != "armed":
                assert not self.valves.any()
                self.armed_for += 1
                self.state = "armed" if self.armed_for >= 25 else "arming"

    def status(self):
        self.ticks += 1
        self.applied = self.valves if self.state == "armed" else np.zeros(3)
        self.rate = self.response * self.applied * np.where(self.applied < 0, self.limits[:, 0], self.limits[:, 1])
        self.q[:3] += 0.01 * self.rate
        pitch = np.cumsum(np.r_[self.q[3] + self.sensors["carriage_pitch_zero_rad"], self.q[:3]]) + self.sensors["mount_pitch_rad"]
        link_rate = np.degrees(np.cumsum(np.r_[0, self.rate]))
        return {"state": self.state, "reason": "test", "control_hz": 100, "live_pwm": False, "imu_healthy": True,
                "ticks": self.ticks, "applied_action": [0, *self.applied, 0, 0],
                "imu_quat_wxyz": [[math.cos(p / 2), 0, math.sin(p / 2), 0] for p in pitch],
                "imu_gyro_dps": [[0, r, 0] for r in link_rate]}

    def close(self):
        pass


class Sticks:
    """Presses A once the run is ready, then holds the given axes; A again after `seconds`."""

    def __init__(self, axes, seconds):
        self.axes, self.polls, self.end = axes, 0, int(seconds * 100)

    def poll(self):
        self.polls += 1
        return [BTN_A] if self.polls in (150, 150 + self.end) else []

    def axis(self, code):
        return self.axes.get(code, 0.0)


def args(**changes):
    values = dict(bundle=str(BUNDLE), mode="circle", radius_mm=50, speed_mm_s=20, direction="ccw", cycles=1,
                  pitch_rate=0.2, max_error_mm=20, gamepad="none", out=None, socket=None, start_control=False)
    return SimpleNamespace(**{**values, **changes})


bundle = Bundle(BUNDLE)
plant = Plant(bundle)
q, v = joint_state(plant.status()["imu_quat_wxyz"], [[0, 1, 0], [0, 3, 0], [0, 3, 0], [0, 7, 0]], bundle.sensors)
assert np.allclose(q, HOME, atol=1e-9) and np.allclose(v, np.radians([2, 0, 4, 1]))

plant = Plant(bundle)
result = ppo_live.run(args(), control=plant)
assert result["outcome"] == "finished" and result["tracking_max_mm"] < 10 and result["angle_max_deg"] < 3, result
assert plant.sent[-1] == client.DISARM and plant.state == "disarmed"

start = bundle.kin.pose_jacobian(HOME[None])[0][0]
plant = Plant(bundle)
result = ppo_live.run(args(mode="gamepad"), control=plant, pad=Sticks({ABS_X: -1.0, ABS_RZ: 1.0}, 3))
moved = bundle.kin.pose_jacobian(plant.q[None])[0][0] - start
assert result["outcome"] == "stopped by the operator", result
turned = math.atan2(math.sin(moved[2]), math.cos(moved[2]))
assert -0.07 < moved[0] < -0.03 and abs(moved[1]) < 0.01 and 0.3 < turned < 0.7, moved   # 20 mm/s in, 0.2 rad/s curl

plant = Plant(bundle, response=0.0)   # valves that do nothing: the tracking-error stop must disarm
result = ppo_live.run(args(speed_mm_s=40, max_error_mm=10), control=plant)
assert result["outcome"].startswith("tracking error") and plant.sent[-1] == client.DISARM, result
print("ppo_live ok")
