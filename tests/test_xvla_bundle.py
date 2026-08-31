"""X-VLA bundle resolution: architecture detection, task/fps/camera, and the
refusal that keeps a base checkpoint away from the valves.

Every check here is one a wrong answer would pass silently at runtime. An X-VLA
chunk is 20 columns of end-effector motion until a fine-tune's processor contract
says otherwise, and `chunk[:, :4]` is a perfectly well-formed action chunk either
way — so the guard has to be in the loader, and the loader has to be tested
without needing 5 GB of TensorRT engines to say so. All of this runs before the
first engine is built, which is exactly why it can be unit-tested at all.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lerobot_vla import policy as policy_mod  # noqa: E402
from lerobot_vla import xvla_split  # noqa: E402


def _contract(state_dim=3, action_dim=4, complete=True, camera="observation.images.cam1"):
    return {
        "version": 1,
        "physical_boundary_complete": complete,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [state_dim]},
            camera: {"type": "VISUAL", "shape": [3, 480, 640]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [action_dim]}},
        "state": {"feature": "observation.state", "dim": state_dim, "model_dim": 20,
                  "normalization": {"mode": "MEAN_STD"}},
        "action": {"feature": "action", "dim": action_dim, "model_dim": 20,
                   "normalization": {"mode": "MEAN_STD"}},
    }


class BundleTestCase(unittest.TestCase):
    """Writes throwaway bundle directories; nothing here touches a real export."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write_xvla(self, **fields):
        """A schema-v1 bundle by default; pass processor_contract for a v2 one."""
        d = self.root / "xvla"
        d.mkdir(exist_ok=True)
        bundle = {"chunk_size": 30, "num_denoising_steps": 10, "max_state_dim": 20,
                  "action_mode": "ee6d"}
        bundle.update(fields)
        if "processor_contract" in fields:
            bundle.setdefault("schema_version", 1)  # v1 skips verify_bundle's checks
        (d / "bundle.json").write_text(json.dumps(bundle))
        return d

    def write_smolvla(self, **fields):
        d = self.root / "smolvla"
        d.mkdir(exist_ok=True)
        (d / "export_info.json").write_text(json.dumps(fields))
        (d / "smolvlm_vision.onnx").write_bytes(b"")
        return d


class TestDetectArchitecture(BundleTestCase):

    def test_xvla_bundle_detected(self):
        self.assertEqual(policy_mod.detect_architecture(self.write_xvla()), "xvla")

    def test_smolvla_bundle_detected(self):
        self.assertEqual(policy_mod.detect_architecture(self.write_smolvla()), "smolvla")

    def test_both_markers_is_ambiguous_rather_than_first_match(self):
        d = self.write_xvla()
        (d / "export_info.json").write_text("{}")
        with self.assertRaises(SystemExit) as ctx:
            policy_mod.detect_architecture(d)
        self.assertIn("ambiguous", str(ctx.exception))

    def test_directory_with_neither_is_refused(self):
        (self.root / "empty").mkdir()
        with self.assertRaises(SystemExit):
            policy_mod.detect_architecture(self.root / "empty")

    def test_missing_directory_is_refused(self):
        with self.assertRaises(SystemExit):
            policy_mod.detect_architecture(self.root / "nope")


class TestResolveTasks(BundleTestCase):

    def test_reads_the_recorded_task(self):
        d = self.write_xvla(task="move the sand to the container")
        self.assertEqual(xvla_split.resolve_tasks(d, None),
                         ["move the sand to the container"])

    def test_reads_a_recorded_task_list_in_order(self):
        # A multi-task fine-tune: the order is the bundle's, so the run always
        # starts on the same instruction and the D-pad indices mean one thing.
        d = self.write_xvla(tasks=["move sand to container",
                                   "move rock to container"])
        self.assertEqual(xvla_split.resolve_tasks(d, None),
                         ["move sand to container", "move rock to container"])

    def test_explicit_goes_first_and_the_bundle_follows(self):
        # --task picks where the run starts; the bundle's own tasks stay on the
        # D-pad behind it.
        d = self.write_xvla(task="move the sand to the container")
        self.assertEqual(xvla_split.resolve_tasks(d, ["dig a hole"]),
                         ["dig a hole", "move the sand to the container"])

    def test_naming_a_recorded_task_reorders_the_list(self):
        d = self.write_xvla(tasks=["move sand to container",
                                   "move rock to container"])
        self.assertEqual(xvla_split.resolve_tasks(d, ["move rock to container"]),
                         ["move rock to container", "move sand to container"])

    def test_no_task_anywhere_refuses_rather_than_defaulting(self):
        # The failure this prevents is invisible: a phrasing the checkpoint never
        # trained on still yields a well-formed chunk, and the machine just drives
        # somewhere else.
        with self.assertRaises(SystemExit) as ctx:
            xvla_split.resolve_tasks(self.write_xvla(), None)
        self.assertIn("--task", str(ctx.exception))


class TestResolveFps(BundleTestCase):

    def test_reads_the_recorded_rate(self):
        self.assertEqual(xvla_split.resolve_fps(self.write_xvla(fps=10), None), 10.0)

    def test_explicit_fps_contradicting_the_bundle_is_refused(self):
        # Playing 10 fps rate commands at 30 Hz moves the machine at a third speed.
        d = self.write_xvla(fps=10)
        with self.assertRaises(SystemExit) as ctx:
            xvla_split.resolve_fps(d, 30.0)
        self.assertIn("contradicts", str(ctx.exception))

    def test_explicit_fps_agreeing_with_the_bundle_is_fine(self):
        self.assertEqual(xvla_split.resolve_fps(self.write_xvla(fps=30), 30.0), 30.0)

    def test_no_rate_anywhere_refuses(self):
        # Unlike the SmolVLA path there is no 30 fps fallback here: an X-VLA bundle
        # with no fps has never been validated at any rate on this robot.
        with self.assertRaises(SystemExit) as ctx:
            xvla_split.resolve_fps(self.write_xvla(), None)
        self.assertIn("fps", str(ctx.exception))

    def test_explicit_fps_is_accepted_when_the_bundle_records_none(self):
        self.assertEqual(xvla_split.resolve_fps(self.write_xvla(), 10.0), 10.0)


class TestCheckCamera(BundleTestCase):

    def test_trained_camera_passes(self):
        d = self.write_xvla(processor_contract=_contract())
        xvla_split.check_camera(d, "observation.images.cam1")

    def test_untrained_camera_is_refused(self):
        d = self.write_xvla(processor_contract=_contract())
        with self.assertRaises(SystemExit) as ctx:
            xvla_split.check_camera(d, "observation.images.cam2")
        self.assertIn("cam1", str(ctx.exception))

    def test_bundle_without_a_contract_cannot_be_checked(self):
        # A v1 bundle records no input features; the base-bundle gate is what
        # stops it, not this.
        xvla_split.check_camera(self.write_xvla(), "observation.images.cam2")


class TestBaseBundleRefusal(BundleTestCase):
    """The gate runs in __init__ before any engine is built, so it is testable."""

    def _load(self, dir_, **kw):
        from lerobot_vla.xvla_split import XVLAExcavatorPolicy
        return XVLAExcavatorPolicy(dir_, **kw)

    def test_bundle_with_no_contract_refuses_to_load(self):
        with self.assertRaises(SystemExit) as ctx:
            self._load(self.write_xvla())
        message = str(ctx.exception)
        self.assertIn("cannot drive this machine", message)
        self.assertIn("--allow-base-bundle", message)

    def test_incomplete_physical_boundary_refuses_to_load(self):
        # A v2-shaped contract whose checkpoint never saved normalization stats:
        # state and action would pass through unnormalized, silently.
        d = self.write_xvla(processor_contract=_contract(complete=False))
        with self.assertRaises(SystemExit) as ctx:
            self._load(d)
        self.assertIn("physical_boundary_complete", str(ctx.exception))

    def test_the_two_refusals_name_different_fixes(self):
        with self.assertRaises(SystemExit) as no_contract:
            self._load(self.write_xvla())
        d = self.write_xvla(processor_contract=_contract(complete=False))
        with self.assertRaises(SystemExit) as incomplete:
            self._load(d)
        self.assertIn("Fine-tune", str(no_contract.exception))
        self.assertIn("Re-export", str(incomplete.exception))


class TestOnePolicyPerProcess(unittest.TestCase):

    def setUp(self):
        self._prev = policy_mod._LOADED
        self.addCleanup(setattr, policy_mod, "_LOADED", self._prev)

    def test_second_policy_in_one_process_is_refused(self):
        # X-VLA is 5.5 GB resident with the robot stack; SmolVLA's 2.2 GB does not
        # fit in what is left of 7.4 GB. Better a clear error than an OOM mid-run.
        policy_mod._LOADED = "xvla"
        with self.assertRaises(RuntimeError) as ctx:
            policy_mod.make_policy("smolvla", "/nonexistent")
        self.assertIn("restart", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
