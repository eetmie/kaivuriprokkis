"""Camera identity, frame ownership/freshness and X-VLA ordered multi-view input."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from lerobot_vla.camera import CameraConfig, D435iCamera, _IRReader, check_frames, fresh_observation
from lerobot_vla.excavator_robot import MasiExcavator
from lerobot_vla.runtime.xvla import resolve_cameras, XVLAExcavatorPolicy

KEY1, KEY2 = "observation.images.cam1", "observation.images.cam2"


class DualIRTests(unittest.TestCase):
    def test_devices_pinned_and_different(self):
        cfg = CameraConfig()
        self.assertEqual((cfg.serial, cfg.ir_index, cfg.rotate180),
                         ("233522073919", 2, True))
        self.assertEqual((cfg.cam2_serial, cfg.cam2_ir_index, cfg.cam2_rotate180),
                         ("141722076647", 1, False))
        with self.assertRaises(ValueError):
            CameraConfig(cam2_serial=cfg.serial)

    def test_partial_start_releases_both_devices(self):
        camera = D435iCamera()
        camera._cam1 = Mock()
        camera._cam2 = Mock()
        camera._cam2.start.side_effect = RuntimeError("unplugged")
        with self.assertRaises(RuntimeError):
            camera.start()
        camera._cam1.stop.assert_called_once()
        camera._cam2.stop.assert_called_once()

    def test_frame_rotation_and_buffer_ownership(self):
        cfg = CameraConfig()
        reader = _IRReader(cfg, cfg.serial, 2, True, None, None, "test")
        raw = np.arange(6, dtype=np.uint8).reshape(2, 3)
        frame = Mock()
        frame.get_data.return_value = raw
        frames = Mock()
        frames.get_infrared_frame.return_value = frame
        reader._pipeline = Mock()

        def next_frame(**kwargs):
            reader._stop.set()  # deliver exactly one frame
            return frames

        reader._pipeline.wait_for_frames.side_effect = next_frame
        reader._run()
        image, ts = reader.latest()
        self.assertGreater(ts, 0)
        np.testing.assert_array_equal(image[:, :, 0], raw[::-1, ::-1])
        before = image.copy()
        raw[:] = 200  # SDK may reuse its storage immediately
        np.testing.assert_array_equal(image, before)
        self.assertTrue(image.flags.c_contiguous)

    def test_observation_keeps_independent_timestamps(self):
        robot = MasiExcavator()
        one, two = np.zeros((2, 3, 3), np.uint8), np.ones((2, 3, 3), np.uint8)
        robot.camera = Mock(second_enabled=True)
        robot.camera.get_frames.return_value = (one, 10.0, two, 10.012)
        robot.get_state = Mock(return_value=(np.zeros(3), 10.01, 12345))
        obs = robot.get_observation()
        self.assertIs(obs[KEY1], one)
        self.assertIs(obs[KEY2], two)
        self.assertEqual((obs["img_ts"], obs["cam2_ts"]), (10.0, 10.012))

    def test_stale_missing_and_skewed_views_rejected(self):
        now = time.perf_counter()
        obs = {KEY1: np.zeros((2, 3, 3)), KEY2: np.zeros((2, 3, 3)),
               "img_ts": now, "cam2_ts": now - .012}
        check_frames(obs, [KEY1, KEY2])
        for change in ({"cam2_ts": now - 1}, {KEY2: None}, {"cam2_ts": now - .1}):
            with self.subTest(change=list(change)):
                with self.assertRaises(RuntimeError):
                    check_frames({**obs, **change}, [KEY1, KEY2])

    def test_fresh_observation_waits_for_both_readers_and_times_out(self):
        now = time.perf_counter()
        image = np.zeros((2, 3, 3), np.uint8)
        valid = {KEY1: image, KEY2: image, "img_ts": now, "cam2_ts": now}
        stale = {**valid, "img_ts": now - 1}
        getter = Mock(side_effect=[stale, valid])
        self.assertIs(fresh_observation(getter, [KEY1, KEY2], timeout_s=.05), valid)
        with self.assertRaises(RuntimeError):
            fresh_observation(lambda: stale, [KEY1, KEY2], timeout_s=0)

    def test_checkpoint_view_order_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = {"schema_version": 1, "valid_views": 2,
                      "processor_contract": {"version": 1, "input_features": {
                          KEY2: {"type": "VISUAL"}, KEY1: {"type": "VISUAL"}}}}
            path = Path(tmp) / "bundle.json"
            path.write_text(json.dumps(bundle))
            self.assertEqual(resolve_cameras(tmp), [KEY2, KEY1])
            with self.assertRaises(SystemExit):
                resolve_cameras(tmp, [KEY1, KEY2])
            bundle["valid_views"] = 1
            path.write_text(json.dumps(bundle))
            with self.assertRaises(SystemExit):
                resolve_cameras(tmp)

    def test_xvla_passes_both_views_and_physical_state(self):
        policy = XVLAExcavatorPolicy.__new__(XVLAExcavatorPolicy)
        policy.feasibility_only = False
        policy.policy = Mock(valid_views=2)
        images = [np.zeros((2, 3, 3), np.uint8), np.ones((2, 3, 3), np.uint8)]
        state = np.array([1, 2, 3], dtype=np.float32)
        policy.sample_actions(images, "dig", state)
        args = policy.policy.sample_actions.call_args.args
        self.assertIs(args[0][0], images[0])
        self.assertIs(args[0][1], images[1])
        np.testing.assert_array_equal(args[2], state)
        with self.assertRaises(ValueError):
            policy.sample_actions(images[0], "dig", state)


if __name__ == "__main__":
    unittest.main()
