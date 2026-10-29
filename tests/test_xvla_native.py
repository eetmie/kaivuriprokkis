"""Physical boundaries and input handling around the accelerated X-VLA chain."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lerobot_vla.runtime.xvla_native import physical_state, physical_actions
from lerobot_vla.runtime.vendor.xvla_trt import Bundle, IMAGENET_MEAN, IMAGENET_STD


class PhysicalBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "state": {"dim": 3, "normalization": {"mode": "MEAN_STD", "mean": [10, 20, 30],
                                                    "std": [2, 4, 5], "eps": 1e-8}},
            "action": {"dim": 4, "normalization": {"mode": "MEAN_STD", "mean": [.1, .2, .3, .4],
                                                     "std": [.2, .3, .4, .5]}},
        }

    def test_normalize_only_physical_state_then_zero_pad(self):
        state = physical_state([12, 28, 45], self.contract, 20)
        np.testing.assert_allclose(state[0, :3], [1, 2, 3])
        np.testing.assert_array_equal(state[0, 3:], 0)
        with self.assertRaises(ValueError):
            physical_state([1, 2, 3, 4], self.contract, 20)
        with self.assertRaises(ValueError):
            physical_state([1, float("nan"), 3], self.contract, 20)

    def test_trim_then_unnormalize_every_action_in_chunk(self):
        model = np.ones((1, 50, 20), np.float32)
        actions = physical_actions(model, self.contract)
        self.assertEqual(actions.shape, (1, 50, 4))
        np.testing.assert_allclose(actions[0, 0], [.3, .5, .7, .9])
        np.testing.assert_array_equal(model, 1)  # no in-place mutation of model output

    def test_nonfinite_output_refused(self):
        with self.assertRaises(ValueError):
            physical_actions(np.full((1, 30, 20), np.nan), self.contract)

    def test_auto_action_space_has_no_gripper_sigmoid(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = {"graphs": [{"name": "denoise_0", "file": "denoise_0.onnx"}],
                 "tokens_per_view": 50, "valid_views": 2, "num_image_views": 3,
                 "lang_len": 50, "chunk_size": 50, "max_state_dim": 20,
                 "max_action_dim": 20, "num_denoising_steps": 10, "action_mode": "auto",
                 "tokenizer": {"path": "tokenizer"}}
            (Path(tmp) / "bundle.json").write_text(json.dumps(b))
            bd = Bundle(tmp)
            self.assertEqual(bd.gripper, [])
            self.assertEqual((bd.valid_views, bd.chunk_size), (2, 50))

    def test_lut_is_identical_to_original_normalization(self):
        # Resize-free input isolates the optimization; padding/resize is also
        # gated against the exporter fixture before any model may load.
        raw = np.random.default_rng(7).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        bd = Bundle.__new__(Bundle)
        actual = bd.preprocess(raw)
        expected = raw.transpose(2, 0, 1).astype(np.float32) / 255.0
        expected = (expected - IMAGENET_MEAN) / IMAGENET_STD
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
