"""CPU tests for observation freshness/history, arbitrary prompts and artifact checks."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "tools"))
from policy_runtime.bundle import describe, inside, verify_manifest
from policy_runtime.history import Frame, FrameHistory, NotReady, robot_state
from policy_probe import main, task_text
from types import SimpleNamespace


def frame(timestamp, seq=1, value=0, width=2):
    return Frame({"host_monotonic_ns": timestamp, "sequence": seq,
                  "width": width, "height": 2, "stride": width,
                  "pixel_bytes": width * 2, "ir_index": 1,
                  "valid_metadata": 3}, bytes([value]) * (width * 2))


class HistoryTests(unittest.TestCase):
    def test_real_history_and_fresh_current_have_distinct_times(self):
        h = FrameHistory(2, [1000])
        for view in range(2):
            h.push(view, frame(1_000_000_000 + view * 1_000_000, value=10))
        with self.assertRaises(NotReady):
            h.select(1_010_000_000)
        for view in range(2):
            h.push(view, frame(2_000_000_000 + view * 1_000_000, 2, 20))
        slots = h.select(2_010_000_000)
        self.assertEqual([s[0].pixels[0] for s in slots], [10, 20])
        self.assertEqual(slots[1][0].timestamp - slots[0][0].timestamp, 1_000_000_000)
        self.assertIsNot(slots[0][0], slots[1][0])

    def test_missing_history_is_never_filled_with_current(self):
        h = FrameHistory(1, [1000])
        h.push(0, frame(2_000_000_000))
        with self.assertRaisesRegex(NotReady, "no real frame"):
            h.select(2_000_000_000)

    def test_future_historical_candidate_is_not_used(self):
        h = FrameHistory(1, [1000])
        h.push(0, frame(1_010_000_000))
        h.push(0, frame(2_000_000_000, 2))
        with self.assertRaises(NotReady):
            h.select(2_000_000_000)

    def test_temporal_gap_and_current_staleness(self):
        h = FrameHistory(1, [1000], tolerance_ms=20)
        h.push(0, frame(900_000_000))
        h.push(0, frame(2_000_000_000, 2))
        with self.assertRaises(NotReady):
            h.select(2_000_000_000)
        h = FrameHistory(1, [0])
        h.push(0, frame(2_000_000_000))
        for now in (1_999_999_999, 2_201_000_000):
            with self.assertRaises(NotReady):
                h.select(now)

    def test_camera_skew(self):
        h = FrameHistory(2, [0])
        h.push(0, frame(1_000_000_000))
        h.push(1, frame(1_080_000_000))
        with self.assertRaisesRegex(NotReady, "skew"):
            h.select(1_090_000_000)

    def test_memory_bound_and_reset(self):
        h = FrameHistory(1, [1000], max_bytes=8)
        for seq in range(1, 20):
            h.push(0, frame(1_000_000_000 + seq * 10_000_000, seq))
            self.assertLessEqual(h.bytes, 8)
        self.assertGreater(h.evictions, 0)
        with self.assertRaisesRegex(ValueError, "reset"):
            h.push(0, frame(1_300_000_000, 1))

    def test_duplicate_does_not_grow_history(self):
        h = FrameHistory(1, [0])
        f = frame(1_000_000_000)
        h.push(0, f)
        h.push(0, f)
        self.assertEqual(h.bytes, len(f.pixels))

    def test_offsets_and_limits(self):
        h = FrameHistory(1, [200, 1000, 200])
        self.assertEqual(h.offsets, (1_000_000_000, 200_000_000, 0))
        for bad in ([float("nan")], [-1], [5001]):
            with self.assertRaises(ValueError):
                FrameHistory(1, bad)

    def test_depth_is_not_accepted_as_ir(self):
        f = frame(1_000_000_000)
        with self.assertRaisesRegex(ValueError, "Y8"):
            FrameHistory(1).push(0, Frame(dict(f.metadata, ir_index=0), f.pixels))

    def test_state_freshness_and_units(self):
        status = {"imu_healthy": True, "fk_valid": True,
                  "imu_host_monotonic_ns": 1_000_000_000, "angles_rad": [0, 1, -.5, .1]}
        state = robot_state(status, 1_010_000_000)
        self.assertEqual(state["names"], ["slew", "lift", "tilt", "scoop"])
        self.assertAlmostEqual(state["values"][1], 57.29578, places=4)
        for s, now in ((dict(status, fk_valid=False), 1_010_000_000),
                       (status, 1_051_000_000), (status, 999_999_999)):
            with self.assertRaises(NotReady):
                robot_state(s, now)


class ArtifactTests(unittest.TestCase):
    def make_bundle(self, root):
        (root / "vision_0.onnx").write_bytes(b"test graph")
        info = {"schema": "groot-n1.7-split/2", "model": "test", "views": 2,
                "frames": 2, "video_delta_indices": [-30, 0], "action_horizon": 40,
                "state_dim": 132, "action_dim": 132, "task": "Any instruction 🪣",
                "timesteps": [0, 250, 500, 750],
                "graphs": [{"name": "vision_0", "file": "vision_0.onnx"}]}
        (root / "bundle.json").write_text(json.dumps(info))
        lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
                 for p in sorted(root.iterdir())]
        (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

    def test_verified_bundle_and_prompt_limit_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_bundle(root)
            result = describe(root, verify=True)
            self.assertEqual(result["integrity"]["files_verified"], 2)
            self.assertFalse(result["actuator_output"])
            self.assertFalse(result["prompt"]["arbitrary_runtime_text"])
            (root / "vision_0.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_manifest(root)

    def test_manifest_cannot_reference_outside_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "bundle"
            root.mkdir()
            (parent / "outside").write_text("data")
            with self.assertRaises(ValueError):
                inside(root, "../outside")
            (root / "alias").symlink_to(parent / "outside")
            with self.assertRaises(ValueError):
                inside(root, "alias")

    def test_unlisted_files_fail_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_bundle(root)
            (root / "unlisted").write_text("data")
            with self.assertRaisesRegex(ValueError, "unlisted"):
                verify_manifest(root)

    def test_arbitrary_task_file_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.txt"
            args = SimpleNamespace(task_file=path, task=None)
            for text in ("scoop up the sand", "Dump into that container 🪣"):
                path.write_text(text + "\n")
                self.assertEqual(task_text(args), text)
            path.write_text("")
            with self.assertRaises(ValueError):
                task_text(args)

    def test_cli_rejects_nonfinite_settings_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            self.assertEqual(main(["capture", "--synthetic", "--task", "test", "--out", str(output),
                                   "--rate-hz", "nan"]), 1)
            self.assertFalse(output.exists())

    def test_duplicate_camera_cannot_stand_in_for_another_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            self.assertEqual(main(["capture", "--task", "test", "--out", str(output),
                                   "--camera-frame", "/tmp/same.frame",
                                   "--camera-frame", "/tmp/same.frame"]), 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
