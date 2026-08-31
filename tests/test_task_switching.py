"""Multi-task runs: resolving the instruction list, and the pad that cycles it.

A wrong task string is the one setting nothing downstream can detect — the prefix
is well-formed, the chunk is well-shaped, and the machine drives somewhere else —
so the list the run carries, and which entry of it is live, are worth pinning
down without a checkpoint. Everything here runs before the first engine is built.
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lerobot_vla import run_inference  # noqa: E402
from lerobot_vla.policy import bundle_tasks, merge_tasks  # noqa: E402
from simple_drive import (  # noqa: E402
    BTN_A, BTN_B, BTN_DPAD_LEFT, BTN_DPAD_RIGHT,
)

#: Long enough for the 50 Hz button thread to see a mask it has never seen.
POLL_SETTLE_S = 0.2


class FakePad:
    """A pad whose button mask the test sets. Axes are never read here."""

    def __init__(self):
        self.mask = 0
        self.closed = False

    def poll(self):
        return None, self.mask

    def is_live(self):
        return True

    def close(self):
        self.closed = True

    def press(self, *bits):
        """Hold the given buttons, then release — one rising edge each."""
        self.mask = sum(1 << b for b in bits)
        time.sleep(POLL_SETTLE_S)
        self.mask = 0
        time.sleep(POLL_SETTLE_S)


class TestBundleTasks(unittest.TestCase):
    """How the two export formats are allowed to spell their instruction list."""

    def test_single_task_string_is_the_one_element_list(self):
        # Every bundle exported before multi-task existed spells it this way.
        self.assertEqual(bundle_tasks({"task": "move sand to container"}),
                         ["move sand to container"])

    def test_task_list_keeps_its_order(self):
        tasks = ["move sand to container", "move rock to container"]
        self.assertEqual(bundle_tasks({"tasks": tasks}), tasks)

    def test_plural_key_holding_a_bare_string_is_still_one_task(self):
        self.assertEqual(bundle_tasks({"tasks": "move sand to container"}),
                         ["move sand to container"])

    def test_list_wins_over_the_singular_key(self):
        # An exporter writing both means the fine-tune had several; the singular
        # key is then whichever one it happened to name.
        self.assertEqual(bundle_tasks({"task": "a", "tasks": ["a", "b"]}), ["a", "b"])

    def test_nothing_recorded_is_an_empty_list_not_a_guess(self):
        self.assertEqual(bundle_tasks({}), [])


class TestMergeTasks(unittest.TestCase):
    """--task picks where the run starts; it must not narrow what the pad reaches."""

    def test_naming_a_recorded_task_reorders_rather_than_filters(self):
        merged = merge_tasks(["move rock to container"],
                             ["move sand to container", "move rock to container"])
        self.assertEqual(merged, ["move rock to container", "move sand to container"])

    def test_a_task_the_bundle_does_not_record_goes_in_front_of_the_rest(self):
        merged = merge_tasks(["dig a hole"], ["move sand to container"])
        self.assertEqual(merged, ["dig a hole", "move sand to container"])

    def test_repeated_flags_keep_their_order_ahead_of_the_bundle(self):
        merged = merge_tasks(["b", "a"], ["a", "b", "c"])
        self.assertEqual(merged, ["b", "a", "c"])

    def test_a_task_named_twice_is_one_entry(self):
        # Otherwise the D-pad would have two slots that do the same thing.
        self.assertEqual(merge_tasks(["a", "a"], ["a"]), ["a"])

    def test_nothing_recorded_leaves_the_explicit_list_alone(self):
        self.assertEqual(merge_tasks(["a", "b"], []), ["a", "b"])


class TestResolveTasks(unittest.TestCase):
    """The SmolVLA side; xvla_split.resolve_tasks is covered in test_xvla_bundle."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def bundle(self, **fields):
        (self.root / "export_info.json").write_text(json.dumps(fields))
        return str(self.root)

    def test_reads_the_recorded_task(self):
        d = self.bundle(task="move sand to container")
        self.assertEqual(run_inference.resolve_tasks(d, None),
                         ["move sand to container"])

    def test_reads_a_recorded_task_list_in_order(self):
        tasks = ["move sand to container", "move rock to container"]
        self.assertEqual(run_inference.resolve_tasks(self.bundle(tasks=tasks), None),
                         tasks)

    def test_repeated_task_flags_lead_the_list_in_the_order_given(self):
        d = self.bundle(tasks=["move sand to container", "move rock to container"])
        self.assertEqual(
            run_inference.resolve_tasks(d, ["move rock to container",
                                            "move sand to container"]),
            ["move rock to container", "move sand to container"])

    def test_one_task_flag_starts_there_and_keeps_the_rest_on_the_dpad(self):
        # Naming the task to start on must not cost the operator the others: the
        # checkpoint knows both, so both stay selectable.
        d = self.bundle(tasks=["move sand to container", "move rock to container"])
        self.assertEqual(
            run_inference.resolve_tasks(d, ["move rock to container"]),
            ["move rock to container", "move sand to container"])

    def test_task_the_bundle_does_not_record_warns_but_runs(self):
        # Deliberate on purpose: probing how much the phrasing matters is a real
        # thing to want. It just must not happen quietly.
        d = self.bundle(tasks=["move sand to container"])
        # Warned by policy.warn_off_bundle, which both resolvers share. The
        # recorded task stays reachable behind the unrecorded one.
        with self.assertLogs("policy", level="WARNING") as logs:
            self.assertEqual(run_inference.resolve_tasks(d, ["dig a hole"]),
                             ["dig a hole", "move sand to container"])
        self.assertIn("out of distribution", "\n".join(logs.output))

    def test_no_task_anywhere_refuses_rather_than_defaulting(self):
        with self.assertRaises(SystemExit) as ctx:
            run_inference.resolve_tasks(self.bundle(fps=30), None)
        self.assertIn("--task", str(ctx.exception))


class TeleopTestCase(unittest.TestCase):

    def teleop(self, allow_takeover=True):
        pad = FakePad()
        t = run_inference.GamepadTeleop(None, pad, allow_takeover=allow_takeover)
        self.addCleanup(t.close)
        return t, pad


class TestTaskSelector(TeleopTestCase):
    """D-pad right/left, read once per chunk by the inference loop."""

    def test_right_steps_forward_and_left_back(self):
        t, pad = self.teleop()
        pad.press(BTN_DPAD_RIGHT)
        self.assertEqual(t.take_task_step(), 1)
        pad.press(BTN_DPAD_LEFT)
        self.assertEqual(t.take_task_step(), -1)

    def test_steps_are_consumed_once(self):
        t, pad = self.teleop()
        pad.press(BTN_DPAD_RIGHT)
        self.assertEqual(t.take_task_step(), 1)
        self.assertEqual(t.take_task_step(), 0)

    def test_two_clicks_inside_one_inference_move_two_tasks(self):
        # The loop asks once per chunk (~0.5 s), so a second click must not queue
        # behind the next one -- the operator asked for two.
        t, pad = self.teleop()
        pad.press(BTN_DPAD_RIGHT)
        pad.press(BTN_DPAD_RIGHT)
        self.assertEqual(t.take_task_step(), 2)

    def test_a_held_dpad_is_one_step_not_a_repeat(self):
        t, pad = self.teleop()
        pad.mask = 1 << BTN_DPAD_RIGHT
        time.sleep(POLL_SETTLE_S * 3)
        self.assertEqual(t.take_task_step(), 1)

    def test_opposite_directions_cancel(self):
        t, pad = self.teleop()
        pad.press(BTN_DPAD_RIGHT)
        pad.press(BTN_DPAD_LEFT)
        self.assertEqual(t.take_task_step(), 0)

    def test_other_buttons_do_not_move_the_task(self):
        t, pad = self.teleop()
        pad.press(BTN_B)
        self.assertEqual(t.take_task_step(), 0)

    def test_the_selector_works_without_takeover(self):
        # This is what makes a two-task bundle checkable on the bench: picking the
        # next inference's instruction writes no valve, so it needs no --live.
        t, pad = self.teleop(allow_takeover=False)
        pad.press(BTN_DPAD_RIGHT)
        self.assertEqual(t.take_task_step(), 1)


class TestTakeoverGate(TeleopTestCase):

    def test_a_hands_over_when_takeover_is_allowed(self):
        t, pad = self.teleop()
        pad.press(BTN_A)
        self.assertTrue(t.pressed())
        self.assertFalse(t.pressed())

    def test_a_is_inert_without_takeover(self):
        # The sticks write setpoints, so the pad must not be able to drive a
        # machine the safety ladder has not put in --live.
        t, pad = self.teleop(allow_takeover=False)
        pad.press(BTN_A)
        self.assertFalse(t.pressed())


class TestSwitchArithmetic(unittest.TestCase):
    """The wrap the loop applies to the selector's net step."""

    @staticmethod
    def cycle(index, step, n):
        return (index + step) % n

    def test_forward_wraps_at_the_end(self):
        self.assertEqual(self.cycle(1, 1, 2), 0)

    def test_backward_wraps_at_the_start(self):
        self.assertEqual(self.cycle(0, -1, 3), 2)

    def test_a_step_larger_than_the_list_still_lands_in_it(self):
        self.assertEqual(self.cycle(0, 5, 2), 1)


if __name__ == "__main__":
    unittest.main()
