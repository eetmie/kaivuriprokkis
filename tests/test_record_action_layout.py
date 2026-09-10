"""Action-layout tests for lerobot_vla: tracks live inside `action`.

The track commands used to be a sibling `action.tracks` column. lerobot's
tooling reads one flat action per frame -- hw_to_dataset_features puts every
actuator, grippers included, into `action` and lists them in `names` -- so a
sibling column was invisible to it: dataset_to_policy_features prefix-matches it
into FeatureType.ACTION and it lands in the policy's output_features, but the
pipeline only ever transforms the literal key `action`, and the dataset viewer
plots only the keys it hardcodes. Two datasets were recorded that way before it
was noticed, which is what these tests exist to stop repeating.

What is pinned here: the recorded width follows enable_tracks, the gamepad
mapping fills those columns in the dataset's own order, and the robot refuses an
action of the wrong width rather than padding it. A wrong-width action is the
failure that matters -- silently zero-padding a 4-wide action onto a track-
enabled machine would record frames that claim the tracks were commanded to stop.

Known test gaps
---------------
- The --resume schema guard is inline in main() and needs a dataset on disk; it
  is exercised by recording, not here. It compares shapes as well as names,
  because folding made both track settings share one key set.
- send_action's control-thread hand-off is faked. The real give_direct_commands
  is covered by modules/ tests.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lerobot_vla.excavator_robot import (  # noqa: E402
    ACTION_KEY, JOINT_NAMES, TRACK_NAMES, MasiExcavator, _CONTROL_CHANNELS,
    _TRACK_CHANNELS, action_names,
)
from lerobot_vla.record_episodes import (  # noqa: E402
    build_features, manual_action_from_axes,
)

AXES = {"left_rl": 0.1, "right_ud": 0.2, "left_ud": 0.3, "right_rl": 0.4,
        "left_paddle": -0.5, "right_paddle": 0.6}


class FakeController:
    def __init__(self):
        self.commands = None
        self.chunk = None

    def give_direct_commands(self, cmds):
        self.commands = cmds

    def give_direct_chunk(self, chunk, fps, joint_names, t0=None):
        self.chunk = (chunk, joint_names)


def make_robot(enable_tracks):
    """A MasiExcavator with the action layout wired up but no hardware."""
    robot = MasiExcavator.__new__(MasiExcavator)
    robot.enable_slew = True
    robot.enable_tracks = enable_tracks
    robot.use_control_thread = True
    robot.action_names = action_names(enable_tracks)
    robot._action_channels = (_CONTROL_CHANNELS + _TRACK_CHANNELS
                              if enable_tracks else list(_CONTROL_CHANNELS))
    robot.controller = FakeController()
    return robot


class ActionFeatureTests(unittest.TestCase):
    def test_width_follows_enable_tracks(self):
        for enable_tracks, expected in ((False, JOINT_NAMES),
                                        (True, JOINT_NAMES + TRACK_NAMES)):
            with self.subTest(enable_tracks=enable_tracks):
                action = build_features(480, 640,
                                        enable_tracks=enable_tracks)[ACTION_KEY]
                self.assertEqual(action["names"], expected)
                self.assertEqual(action["shape"], (len(expected),))

    def test_tracks_never_get_a_column_of_their_own(self):
        # The whole point of the fold: lerobot reads `action` and nothing else.
        for enable_tracks in (False, True):
            with self.subTest(enable_tracks=enable_tracks):
                features = build_features(480, 640, enable_tracks=enable_tracks)
                self.assertNotIn("action.tracks", features)
                self.assertEqual(
                    [k for k in features if k.startswith("action")], [ACTION_KEY])


class GamepadMappingTests(unittest.TestCase):
    def test_axes_land_in_dataset_order(self):
        np.testing.assert_allclose(manual_action_from_axes(AXES),
                                   [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(manual_action_from_axes(AXES, True),
                                   [0.1, 0.2, 0.3, 0.4, -0.5, 0.6])

    def test_track_columns_are_signed(self):
        # The paddles carry the trigger+bumper sign already, so reverse has to
        # survive into the dataset; clipping it to [0, 1] would erase it.
        action = manual_action_from_axes(AXES, True)
        self.assertLess(action[4], 0.0)
        self.assertEqual(action.dtype, np.float32)

    def test_width_matches_the_recorded_feature(self):
        for enable_tracks in (False, True):
            with self.subTest(enable_tracks=enable_tracks):
                features = build_features(480, 640, enable_tracks=enable_tracks)
                self.assertEqual(
                    manual_action_from_axes(AXES, enable_tracks).shape,
                    tuple(features[ACTION_KEY]["shape"]))


class SendActionTests(unittest.TestCase):
    def test_action_maps_onto_control_channels(self):
        # Dataset names (lift/tilt/scoop) and control names (boom/arm/bucket)
        # differ; the columns have to stay aligned across that rename.
        robot = make_robot(True)
        robot.send_action(np.array([0.1, 0.2, 0.3, 0.4, -0.5, 0.6]))
        self.assertEqual(
            {k: round(v, 3) for k, v in robot.controller.commands.items()},
            {"slew": 0.1, "boom": 0.2, "arm": 0.3, "bucket": 0.4,
             "trackL": -0.5, "trackR": 0.6})

    def test_tracks_are_not_driven_when_disabled(self):
        robot = make_robot(False)
        robot.send_action(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(set(robot.controller.commands), set(_CONTROL_CHANNELS))

    def test_wrong_width_raises_instead_of_padding(self):
        for enable_tracks, bad in ((True, 4), (False, 6)):
            with self.subTest(enable_tracks=enable_tracks, width=bad):
                robot = make_robot(enable_tracks)
                with self.assertRaises(ValueError):
                    robot.send_action(np.zeros(bad, dtype=np.float32))
                self.assertIsNone(robot.controller.commands)

    def test_chunk_width_follows_the_same_rule(self):
        robot = make_robot(True)
        robot.send_action_chunk(np.zeros((5, 6), dtype=np.float32), fps=30.0)
        self.assertEqual(robot.controller.chunk[1],
                         _CONTROL_CHANNELS + _TRACK_CHANNELS)
        with self.assertRaises(ValueError):
            robot.send_action_chunk(np.zeros((5, 4), dtype=np.float32), fps=30.0)


if __name__ == "__main__":
    unittest.main()
