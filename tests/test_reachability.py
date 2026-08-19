"""Pre-flight reachability check tests.

Exercises modules.reachability.check_reachability against the new IK
surface: a reachable nudge converges, a wildly out-of-reach pose returns
the closest envelope point, a singular pose is rejected via condition
number, joystick-style streaming hits the memoization cache, and the
overall budget stays comfortably inside the Pi 5 control-loop window.
"""

from __future__ import annotations

import math
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from modules.ik import get_state  # noqa: E402
from modules.reachability import check_reachability  # noqa: E402

from tests.common_kinematics import (  # noqa: E402
    fk_from_angles,
    load_robot_config,
    make_ik_config,
)


def _nominal_start() -> np.ndarray:
    return np.array([0.0, math.radians(-45.0), math.radians(30.0), 0.0], dtype=np.float32)


def _build_cfg(**overrides):
    return make_ik_config(
        method="dls",
        ik_params={"k_val": 0.5, "min_singular_value": 1e-4, "lambda_val": 0.02},
        enable_velocity_limiting=False,
        **overrides,
    )


class ReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rc = load_robot_config()

    def test_in_workspace_is_reachable(self):
        cfg = _build_cfg()
        start = _nominal_start()
        start_pos, _ = fk_from_angles(start, self.rc)
        target = start_pos + np.array([0.02, 0.01, -0.01], dtype=np.float32)

        result = check_reachability(
            self.rc, cfg, start, target,
            pos_tol=5e-3, max_iters=200, dt=0.005, position_only=True,
        )
        self.assertTrue(result.reachable,
                        msg=f"err={result.pos_error_m:.4f}m closest={result.closest_position}")
        self.assertLess(result.pos_error_m, 5e-3)

    def test_pose_mode_centerline_target_can_change_slew(self):
        cfg = _build_cfg(enable_joint_limit_avoidance=True)
        angles = np.radians([9.6, -24.0, 72.0, -28.0]).astype(np.float32)
        target = np.array([0.6, 0.0, 0.0], dtype=np.float32)

        result = check_reachability(
            self.rc, cfg, current_joint_angles=angles, target_pos=target,
            target_rot_y_deg=0.0, pos_tol=5e-3, max_iters=80,
            cond_threshold=100.0, dt=0.01,
        )

        self.assertTrue(
            result.reachable,
            msg=f"err={result.pos_error_m:.4f}m closest={result.closest_position}",
        )
        self.assertLess(abs(float(result.closest_position[1])), 5e-3)

    def test_out_of_reach_returns_closest_envelope(self):
        cfg = _build_cfg()
        start = _nominal_start()
        # 5m straight up is way outside any ground-mounted excavator workspace.
        target = np.array([0.0, 0.0, 5.0], dtype=np.float32)

        result = check_reachability(
            self.rc, cfg, start, target,
            pos_tol=5e-3, max_iters=80, dt=0.005, position_only=True,
        )
        self.assertFalse(result.reachable)
        self.assertGreater(result.pos_error_m, 0.5)
        self.assertLess(np.linalg.norm(result.closest_position), 10.0)

    def test_condition_threshold_can_reject_converged_pose(self):
        cfg = _build_cfg()
        extended = np.zeros(self.rc.num_joints, dtype=np.float32)
        ee_pos, _ = fk_from_angles(extended, self.rc)
        target = np.asarray(ee_pos, dtype=np.float32) + np.array([0.02, 0.0, 0.0], dtype=np.float32)

        result = check_reachability(
            self.rc, cfg, extended, target,
            pos_tol=5e-3, max_iters=80, cond_threshold=1.0, dt=0.005, position_only=True,
        )
        self.assertFalse(result.reachable)
        self.assertGreater(result.final_cond_number, 1.0)

    def test_joint_limit_blocks_target(self):
        joint_limits = [
            (-0.05, 0.05),                      # slew clamped to ~3 deg
            (math.radians(-90), math.radians(0)),
            (math.radians(0), math.radians(120)),
            (math.radians(-60), math.radians(60)),
        ]
        cfg = _build_cfg(joint_limits=joint_limits, enable_joint_limit_avoidance=True)
        start = _nominal_start()
        start_pos, _ = fk_from_angles(start, self.rc)
        # Demand a big lateral swing the slew limit forbids.
        target = np.array([-start_pos[0], -start_pos[1], start_pos[2]], dtype=np.float32)

        result = check_reachability(
            self.rc, cfg, start, target,
            pos_tol=5e-3, max_iters=120, dt=0.005, position_only=True,
        )
        self.assertFalse(result.reachable)
        self.assertGreater(result.pos_error_m, 0.05)

    def test_perf_smoke(self):
        """100 reachable checks complete well under the 4s budget."""
        cfg = _build_cfg()
        start = _nominal_start()
        start_pos, _ = fk_from_angles(start, self.rc)

        rng = np.random.default_rng(0xCAFE)
        offsets = rng.uniform(-0.02, 0.02, size=(100, 3)).astype(np.float32)

        t0 = time.perf_counter()
        for off in offsets:
            target = start_pos + off
            check_reachability(
                self.rc, cfg, start, target,
                pos_tol=5e-3, max_iters=80, dt=0.005,
            )
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 4.0,
                        msg=f"reachability check too slow: {elapsed:.3f}s for 100 calls")


class ControllerIntegrationTests(unittest.TestCase):
    """give_pose() honours the reachability gate and the memoization cache."""

    def _make_controller(self):
        from modules.excavator_controller import ExcavatorController

        # Stub hardware with a duck-typed object.
        hw = mock.MagicMock()
        hw.is_hardware_ready.return_value = True
        controller = ExcavatorController.__new__(ExcavatorController)
        # Avoid running real __init__ — wire only what give_pose needs.
        rc = load_robot_config()
        controller.model = rc
        controller.ik_cfg = _build_cfg()
        controller.logger = mock.MagicMock()
        controller._lock = __import__("threading").Lock()
        controller._raw_target_position = None
        controller._raw_target_rotation_deg = None
        controller._outputs_zeroed = False
        controller._reach_enabled = True
        controller._reach_pos_tol = 5e-3
        # __init__ sets this alongside the other telemetry counters; this fixture
        # skips __init__, so the reject path has nothing to increment without it.
        controller._reach_reject_count = 0
        # Pose-mode rollout (live controller default) needs more iters than
        # the historical position-only path; bump for the cache-flow test.
        controller._reach_max_iters = 400
        controller._reach_min_target_delta = 0.002
        controller._cond_threshold = 200.0
        controller._ik_default_dt = 0.005
        controller._last_validated_target = None
        controller._last_validated_rot_deg = None
        controller._last_reachability_result = None
        # Seed a "current" joint state.
        controller._current_joint_angles = _nominal_start()
        from types import SimpleNamespace
        controller.config = SimpleNamespace(control_frequency=200.0)
        return controller

    def test_give_pose_rejects_unreachable_target(self):
        ctrl = self._make_controller()
        result = ctrl.give_pose(np.array([0.0, 0.0, 5.0], dtype=np.float32))
        self.assertIsNotNone(result)
        self.assertFalse(result.reachable)
        self.assertIsNone(ctrl._raw_target_position)

    def test_give_pose_accepts_and_caches_reachable_target(self):
        """Verify the controller's reachability cache (sub-threshold delta hits cache).

        We stub check_reachability to a fixed ``reachable=True`` result so this
        test stays focused on the controller's cache logic rather than the
        rollout's convergence behavior.
        """
        from modules.reachability import ReachabilityResult

        ctrl = self._make_controller()
        start_pos, _ = fk_from_angles(_nominal_start(), ctrl.model)
        target = start_pos + np.array([0.02, 0.01, -0.01], dtype=np.float32)

        stub_result = ReachabilityResult(
            reachable=True,
            closest_position=target.copy(),
            pos_error_m=1e-3,
            iters=10,
            final_cond_number=20.0,
        )
        with mock.patch(
            "modules.excavator_controller.reachability_module.check_reachability",
            return_value=stub_result,
        ) as spy:
            r1 = ctrl.give_pose(target)
            self.assertTrue(r1.reachable)
            self.assertEqual(spy.call_count, 1)

            # Nudge target by 1mm — under min_target_delta_m, should hit cache.
            r2 = ctrl.give_pose(target + np.array([0.001, 0.0, 0.0], dtype=np.float32))
            self.assertTrue(r2.reachable)
            self.assertEqual(spy.call_count, 1,
                             "cache miss: rollout ran for sub-threshold delta")

            # Move 5mm — above threshold, should run again.
            ctrl.give_pose(target + np.array([0.005, 0.0, 0.0], dtype=np.float32))
            self.assertEqual(spy.call_count, 2)

    def test_reachability_warns_when_current_state_missing(self):
        ctrl = self._make_controller()
        ctrl._current_joint_angles = None

        result = ctrl._evaluate_reachability(np.array([0.5, 0.0, -0.1], dtype=np.float32), 0.0)

        self.assertTrue(result.reachable)
        ctrl.logger.warning.assert_called()


if __name__ == "__main__":
    unittest.main()
