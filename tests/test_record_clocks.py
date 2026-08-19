"""Dataset capture-clock tests for lerobot_vla.record_episodes.

The clock.* columns are the only record of when anything in a frame actually
happened -- lerobot's own `timestamp` is `frame_index / fps` and reports a
perfect loop no matter what the loop did. These tests pin the two things that
are easy to break silently: the dtypes lerobot validates every frame against,
and the not-yet-reported sentinels (NaN for ages, -1 for the device clock,
which differ because int64 has no NaN and 0 is a real Pico timestamp).

The controller half of the chain is here too: clock.state_age and clock.imu_us
only carry data because ExcavatorController now keeps the IMU frame's timestamps
alongside the joint angles it derives from them.

Known test gaps
---------------
- The record loop itself is not exercised; it needs a gamepad, a RealSense
  pipeline and the 100 Hz control thread. What it contributes is reading
  perf_counter straight after the observation, verified on hardware instead.
- build_features() is checked for the clock entries only. The state/action/video
  entries predate this module and are covered by every recording run.
- _update_current_state() is not driven here (it needs the IK model and a live
  sensor stream); the tests below cover the two accessors it writes through.
"""

import math
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lerobot_vla.record_episodes import (  # noqa: E402
    CLOCK_CAM1_AGE, CLOCK_CAM2_AGE, CLOCK_IMU_US, CLOCK_LOOP, CLOCK_STATE_AGE,
    build_features, clock_fields,
)

CLOCK_KEYS = (CLOCK_LOOP, CLOCK_CAM1_AGE, CLOCK_CAM2_AGE, CLOCK_STATE_AGE, CLOCK_IMU_US)


class ClockFieldsTests(unittest.TestCase):
    def _obs(self, tick):
        return {
            "img_ts": tick - 0.004,
            "rgb_ts": tick - 0.004,
            "state_ts": tick - 0.0012,
            "imu_device_us": 8_123_456,
        }

    def test_ages_measure_backwards_from_the_tick(self):
        tick = 1000.5
        fields = clock_fields(self._obs(tick), tick, ep_perf0=1000.0)
        self.assertAlmostEqual(float(fields[CLOCK_LOOP][0]), 0.5, places=9)
        self.assertAlmostEqual(float(fields[CLOCK_CAM1_AGE][0]), 0.004, places=6)
        self.assertAlmostEqual(float(fields[CLOCK_CAM2_AGE][0]), 0.004, places=6)
        self.assertAlmostEqual(float(fields[CLOCK_STATE_AGE][0]), 0.0012, places=6)
        self.assertEqual(int(fields[CLOCK_IMU_US][0]), 8_123_456)

    def test_dtypes_and_shapes_match_the_declared_features(self):
        features = build_features(480, 640, state_joints=["lift", "tilt", "scoop"])
        fields = clock_fields(self._obs(1000.0), 1000.0, ep_perf0=1000.0)
        for key in CLOCK_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, features)
                self.assertEqual(fields[key].shape, tuple(features[key]["shape"]))
                self.assertEqual(fields[key].dtype.name, features[key]["dtype"])

    def test_loop_clock_keeps_microseconds_over_a_long_episode(self):
        # float32 would quantise a 120 s episode to ~8 us steps and lose the
        # sub-millisecond jitter these columns exist to show.
        ep_perf0 = 1000.0
        tick = ep_perf0 + 119.999_999
        loop = float(clock_fields(self._obs(tick), tick, ep_perf0)[CLOCK_LOOP][0])
        self.assertAlmostEqual(loop, 119.999_999, places=6)

    def test_missing_sources_use_their_own_sentinels(self):
        # Every source absent: before the camera thread or the control thread
        # has published anything. Ages have no meaningful value; the device
        # clock cannot use NaN because it is an integer column.
        fields = clock_fields({}, 1000.0, ep_perf0=1000.0)
        for key in (CLOCK_CAM1_AGE, CLOCK_CAM2_AGE, CLOCK_STATE_AGE):
            with self.subTest(key=key):
                self.assertTrue(math.isnan(float(fields[key][0])))
        self.assertEqual(int(fields[CLOCK_IMU_US][0]), -1)
        self.assertEqual(float(fields[CLOCK_LOOP][0]), 0.0)

    def test_one_missing_source_does_not_blank_the_others(self):
        tick = 1000.0
        obs = self._obs(tick)
        obs["state_ts"] = None
        obs["imu_device_us"] = None
        fields = clock_fields(obs, tick, ep_perf0=tick)
        self.assertAlmostEqual(float(fields[CLOCK_CAM1_AGE][0]), 0.004, places=6)
        self.assertTrue(math.isnan(float(fields[CLOCK_STATE_AGE][0])))
        self.assertEqual(int(fields[CLOCK_IMU_US][0]), -1)

    def test_zero_is_kept_as_a_real_device_timestamp(self):
        fields = clock_fields({"imu_device_us": 0}, 1000.0, ep_perf0=1000.0)
        self.assertEqual(int(fields[CLOCK_IMU_US][0]), 0)


class SensorTimestampPlumbingTests(unittest.TestCase):
    """ExcavatorController carries the IMU device clock through to the pose."""

    def _make_controller(self, hardware):
        from modules.excavator_controller import ExcavatorController

        # Same approach as ControllerIntegrationTests: skip __init__ and wire
        # only what the accessors under test touch.
        controller = ExcavatorController.__new__(ExcavatorController)
        controller.hardware = hardware
        controller.logger = mock.MagicMock()
        controller._lock = threading.Lock()
        controller._current_joint_angles = None
        controller._current_state_ts = None
        controller._current_state_imu_us = None
        return controller

    def test_timestamp_comes_back_with_the_quaternions(self):
        quats = [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)] * 4
        hw = mock.MagicMock()
        hw.read_all_imu_quaternions.return_value = (quats, 8_123_456)
        controller = self._make_controller(hw)

        got_quats, device_us = controller._get_sensor_quaternions()

        self.assertEqual(device_us, 8_123_456)
        self.assertEqual(got_quats.shape, (4, 4))

    def test_firmware_without_a_timestamp_still_gives_a_pose(self):
        # Losing the drop diagnostic must not cost the joint angles.
        quats = [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)] * 4
        hw = mock.MagicMock()
        hw.read_all_imu_quaternions.return_value = (quats, None)
        controller = self._make_controller(hw)

        got_quats, device_us = controller._get_sensor_quaternions()

        self.assertIsNone(device_us)
        self.assertEqual(got_quats.shape, (4, 4))

    def test_unreadable_sensors_report_no_pose_and_no_clock(self):
        hw = mock.MagicMock()
        hw.read_all_imu_quaternions.side_effect = RuntimeError("IMU not ready")
        controller = self._make_controller(hw)

        self.assertEqual(controller._get_sensor_quaternions(), (None, None))

    def test_a_stub_still_on_the_old_bare_list_contract_yields_no_pose(self):
        # read_all_imu_quaternions() returns (quats, ts) now. An out-of-date
        # duck-typed stub returning a bare list of four quaternions fails the
        # unpack, which must surface as "no pose" rather than as a pose built
        # from a mis-split return.
        hw = mock.MagicMock()
        hw.read_all_imu_quaternions.return_value = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)] * 4
        controller = self._make_controller(hw)

        self.assertEqual(controller._get_sensor_quaternions(), (None, None))
        controller.logger.error.assert_called_once()

    def test_clocks_are_none_until_the_first_state_is_computed(self):
        controller = self._make_controller(mock.MagicMock())

        angles, state_ts, imu_us = controller.get_joint_angles()

        np.testing.assert_array_equal(angles, np.zeros(4, dtype=np.float32))
        self.assertIsNone(state_ts)
        self.assertIsNone(imu_us)

    def test_angles_and_clocks_are_read_together(self):
        controller = self._make_controller(mock.MagicMock())
        with controller._lock:
            controller._current_joint_angles = np.radians(
                np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
            controller._current_state_ts = 1234.5
            controller._current_state_imu_us = 8_123_456

        angles, state_ts, imu_us = controller.get_joint_angles()

        np.testing.assert_allclose(angles, [10.0, 20.0, 30.0, 40.0], atol=1e-4)
        self.assertEqual(state_ts, 1234.5)
        self.assertEqual(imu_us, 8_123_456)


if __name__ == "__main__":
    unittest.main()
