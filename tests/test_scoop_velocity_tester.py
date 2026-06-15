import math
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tools.scoop_velocity_tester import Sample, recommend_speed_bins, summarize_sweeps


class ScoopVelocityTesterAnalysisTests(unittest.TestCase):
    def _sample(self, t, phase, pass_index, amplitude, direction, vel):
        return Sample(
            t_s=t,
            phase=phase,
            pass_index=pass_index,
            amplitude=amplitude,
            direction=direction,
            command=direction * amplitude,
            bucket_deg=t * direction * vel,
            bucket_vel_deg_s=vel * direction,
            pressure_age_s=math.inf,
            pressure_json="{}",
        )

    def test_summarize_marks_smooth_command_usable(self):
        rows = [self._sample(i * 0.1, "sweep", 0, 0.2, 1, 4.0) for i in range(20)]

        summaries = summarize_sweeps(
            rows,
            settle_ignore_s=0.0,
            min_vel_deg_s=1.0,
            max_ripple_ratio=0.2,
        )

        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0].usable)
        self.assertAlmostEqual(summaries[0].median_progress_vel_deg_s, 4.0)

    def test_summarize_rejects_high_ripple(self):
        rows = []
        for i in range(20):
            vel = 2.0 if i % 2 == 0 else 8.0
            rows.append(self._sample(i * 0.1, "sweep", 0, 0.2, 1, vel))

        summaries = summarize_sweeps(
            rows,
            settle_ignore_s=0.0,
            min_vel_deg_s=1.0,
            max_ripple_ratio=0.2,
        )

        self.assertFalse(summaries[0].usable)

    def test_recommend_speed_bins_keeps_each_direction(self):
        rows = []
        for direction in (1, -1):
            for idx, vel in enumerate((2.0, 4.0, 6.0, 8.0)):
                rows.extend(
                    self._sample(10 * idx + i * 0.1, "sweep", idx + (10 if direction < 0 else 0), 0.1 * (idx + 1), direction, vel)
                    for i in range(12)
                )
        summaries = summarize_sweeps(
            rows,
            settle_ignore_s=0.0,
            min_vel_deg_s=1.0,
            max_ripple_ratio=0.2,
        )

        bins = recommend_speed_bins(summaries, bins_per_direction=3)

        self.assertEqual(len(bins["positive"]), 3)
        self.assertEqual(len(bins["negative"]), 3)
        self.assertGreater(bins["positive"][-1]["target_deg_s"], bins["positive"][0]["target_deg_s"])
        self.assertLess(bins["negative"][0]["command"], 0.0)


if __name__ == "__main__":
    unittest.main()
