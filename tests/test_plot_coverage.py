"""Hardware-free regression checks for coverage geometry and time accounting."""

import csv
import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_collection.plot_coverage import (
    COMMANDS, POSITIONS, forward_kinematics, load_profile, pitch_edges,
    read_log, resolve_logs, usable_intervals,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configuration_files/profiles/jetson_bucket/control_config.yaml"


def recording(times):
    n = len(times)
    data = {"timestamp": np.asarray(times, dtype=float)}
    data.update({name: np.zeros(n) for name in (*POSITIONS, *COMMANDS)})
    data.update({name: np.zeros(n) for name in ("cmd_stale", "cmd_age_s", "state_age_s")})
    data["sample_idx"] = np.arange(n, dtype=float)
    return data


class CoverageTimeTests(unittest.TestCase):
    def test_actual_time_gaps_and_dropped_samples_do_not_bridge(self):
        data = recording([0, 0.011, 0.021, 0.200, 0.211, 0.222])
        data["sample_idx"][-1] = 8
        ids, seconds, report = usable_intervals(data, 0.03, 0.03)
        np.testing.assert_array_equal(ids, [0, 1, 3])
        self.assertAlmostEqual(seconds.sum(), 0.032)
        self.assertEqual(report["broken_intervals"], 2)

    def test_invalid_stale_and_nan_ages_split_both_adjacent_intervals(self):
        for name, value in (("cmd_stale", 1), ("cmd_age_s", 0.04), ("state_age_s", np.nan), (POSITIONS[1], np.nan)):
            with self.subTest(column=name):
                data = recording(np.arange(6) * 0.01)
                data[name][2] = value
                ids, seconds, _ = usable_intervals(data, 0.03, 0.03)
                np.testing.assert_array_equal(ids, [0, 3, 4])
                self.assertAlmostEqual(seconds.sum(), 0.03)

    def test_clock_reversal_and_duplicate_timestamps_are_not_counted(self):
        data = recording([0, 0.01, 0.01, 0.005, 0.015])
        ids, seconds, _ = usable_intervals(data, 0.03, 0.03)
        np.testing.assert_array_equal(ids, [0, 3])
        self.assertAlmostEqual(seconds.sum(), 0.02)

    def test_motion_only_requires_fresh_measured_motion(self):
        data = recording(np.arange(6) * 0.01)
        for joint in ("boom", "arm", "bucket"):
            data[f"joint_vel_{joint}"] = np.zeros(6)
        data["joint_vel_boom"][1:5] = np.radians(2)
        data["vel_age_s"] = np.zeros(6)
        data["vel_age_s"][3] = 0.04
        ids, _, _ = usable_intervals(data, 0.03, 0.03, motion_only=True, min_speed=1)
        np.testing.assert_array_equal(ids, [1])

    def test_legacy_missing_quality_is_explicit_and_not_invented(self):
        data = recording([0, 0.01])
        for name in ("cmd_stale", "cmd_age_s", "state_age_s", "sample_idx"):
            del data[name]
        _, seconds, report = usable_intervals(data, 0.03, 0.03)
        self.assertAlmostEqual(seconds.sum(), 0.01)
        self.assertEqual(len(report["unavailable_quality_columns"]), 4)

    def test_directory_and_glob_deduplicate_paths_and_ignore_raw_strips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "drive_log_a.csv"
            log.touch()
            (root / "imu_raw_a.csv").touch()
            self.assertEqual(resolve_logs([str(root), str(root / "drive_log_*.csv"), str(log)]), [log.resolve()])

    def test_gzip_and_blank_numeric_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drive_log_a.csv.gz"
            names = ["timestamp", *POSITIONS, *COMMANDS]
            with gzip.open(path, "wt", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(names)
                writer.writerow([0, 0, "", 0, 0, 0, 0])
                writer.writerow([0.01, 0, 0, 0, 0, 0, 0])
            data = read_log(path)
            self.assertTrue(np.isnan(data[POSITIONS[1]][0]))
            _, seconds, _ = usable_intervals(data, 0.03, 0.03)
            self.assertEqual(len(seconds), 0)


class CoverageGeometryTests(unittest.TestCase):
    def test_pitch_bins_preserve_exact_periodic_endpoints(self):
        edges = pitch_edges(17)
        self.assertEqual(edges[0], -180)
        self.assertEqual(edges[-1], 180)
        self.assertTrue((np.diff(edges) > 0).all())

    @unittest.skipUnless(importlib.util.find_spec("numba"), "robot FK comparison needs numba")
    def test_tip_positions_match_existing_robot_fk_at_random_and_limit_poses(self):
        from modules.ik import get_state, load_excavator_model

        geometry = load_profile(PROFILE)
        model = load_excavator_model(PROFILE)
        rng = np.random.default_rng(43)
        limits = np.radians(geometry["limits_deg"])
        q = np.vstack([rng.uniform(limits[:, 0], limits[:, 1], (128, 3)), limits[:, 0], limits[:, 1], [0, 0, 0]])
        points, pitch = forward_kinematics(q, geometry)
        states = [get_state(np.r_[0, pose], model, include_jacobian=False) for pose in q]
        expected = np.array([state.ee_position for state in states])
        np.testing.assert_allclose(points, expected, atol=3e-7)
        # Derive pitch from the existing FK quaternion, independently of the
        # plotter's accumulated joint angles; compare through the wrap boundary.
        quaternions = np.array([state.ee_orientation for state in states], dtype=float)
        expected_pitch = np.degrees(2 * np.arctan2(quaternions[:, 2], quaternions[:, 0]) + geometry["pitch_offset"])
        difference = (pitch - expected_pitch + 180) % 360 - 180
        # The existing robot FK uses float32, unlike the plotter's float64.
        np.testing.assert_allclose(difference, 0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
