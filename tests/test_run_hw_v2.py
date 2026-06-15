import inspect
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import run_hw_v2  # noqa: E402
from configuration_files.pathing_config import (  # noqa: E402
    DEFAULT_ENV_CONFIG,
    PathExecutionConfig,
)
from modules.excavator_controller import ExcavatorController  # noqa: E402
from modules.hardware_interface import HardwareInterface  # noqa: E402


class _FakeHardware:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.shutdown_calls = 0
        _FakeHardware.instances.append(self)

    def is_hardware_ready(self):
        return True

    def get_status(self):
        return {"imu_state": "ready"}

    def shutdown(self):
        self.shutdown_calls += 1


class _FakeController:
    instances = []

    def __init__(self, hardware_interface, **kwargs):
        self.hardware = hardware_interface
        self.kwargs = dict(kwargs)
        self.start_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0
        self.stop_calls = 0
        self.pose_log = []
        self.ik_config = type("IKConfig", (), {"command_type": "pose"})()
        _FakeController.instances.append(self)

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1

    def pause(self):
        self.pause_calls += 1

    def resume(self):
        self.resume_calls += 1

    def give_pose(self, position, rotation_y_deg=0.0):
        self.pose_log.append((tuple(float(v) for v in position), float(rotation_y_deg)))

    def get_pose(self):
        return np.asarray([0.55, 0.25, -0.035], dtype=np.float32), 0.0


class RunHwV2Tests(unittest.TestCase):
    def setUp(self):
        self._logging_disabled = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(self._logging_disabled)

    def test_cli_supports_radial_planar_algorithm_name(self):
        args = run_hw_v2._parse_args([
            "--algorithm", "a_star",
            "-r",
            "-p",
            "--task", "in-and-out",
        ])

        self.assertEqual(args.algorithm, "a_star")
        self.assertTrue(args.radial)
        self.assertTrue(args.planar)
        self.assertEqual(args.algorithm_name, "radial_a_star_plane")

    def test_checked_in_obstacles_json_loads(self):
        obstacles = run_hw_v2._load_obstacles_json(str(ROOT_DIR / "obstacles.json"))

        self.assertEqual(len(obstacles), 9)
        for obstacle in obstacles:
            self.assertEqual(set(obstacle), {"size", "pos", "rot"})
            self.assertEqual(len(obstacle["size"]), 3)
            self.assertEqual(len(obstacle["pos"]), 3)
            self.assertEqual(len(obstacle["rot"]), 4)

    def test_obstacle_loader_accepts_bare_list_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obstacles.json"
            path.write_text(
                '[{"scale": [0.1, 0.2, 0.3], "position": [0.4, 0.5, 0.6]}]',
                encoding="utf-8",
            )

            obstacles = run_hw_v2._load_obstacles_json(str(path))

        self.assertEqual(obstacles, [{
            "size": [0.1, 0.2, 0.3],
            "pos": [0.4, 0.5, 0.6],
            "rot": [1, 0, 0, 0],
        }])

    def test_default_obstacles_follow_task_and_environment_config(self):
        in_out = run_hw_v2.make_task_preset("in-and-out")
        wall = run_hw_v2._build_obstacles_for_task(in_out, DEFAULT_ENV_CONFIG)

        self.assertEqual(len(wall), 1)
        np.testing.assert_allclose(wall[0]["size"], DEFAULT_ENV_CONFIG.wall_size)
        np.testing.assert_allclose(wall[0]["pos"], DEFAULT_ENV_CONFIG.wall_pos)
        np.testing.assert_allclose(wall[0]["rot"], DEFAULT_ENV_CONFIG.wall_rot)

        empty = run_hw_v2.make_task_preset("empty")
        self.assertEqual(
            run_hw_v2._build_obstacles_for_task(empty, DEFAULT_ENV_CONFIG),
            [],
        )

    def test_plan_trajectory_dispatches_radial_planar_to_current_planner_api(self):
        planner_result = {
            "exec_positions": np.asarray([
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ], dtype=np.float32),
            "total_length_m": 0.1,
            "chosen_variant": "radial_a_star_plane",
        }

        with patch.object(run_hw_v2, "plan_radial_to_target", return_value=planner_result) as plan_mock, \
                patch.object(run_hw_v2, "print_path_info"):
            trajectory = run_hw_v2._plan_trajectory(
                base_algorithm="a_star",
                radial=True,
                planar=True,
                radial_mode="raw",
                start_pos_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                target_pos_world=np.asarray([0.2, 0.0, 0.0], dtype=np.float32),
                obstacles=[],
                path_config=PathExecutionConfig(speed_mps=0.03, dt=0.02),
                enable_verbose=True,
            )

        kwargs = plan_mock.call_args.kwargs
        self.assertEqual(kwargs["fallback_algorithm"], "a_star_plane")
        self.assertFalse(kwargs["use_3d"])
        self.assertTrue(kwargs["execute_raw"])
        self.assertTrue(kwargs["verbose"])
        self.assertEqual(trajectory.chosen_variant, "radial_a_star_plane")
        self.assertEqual(trajectory.positions.shape, (3, 3))
        np.testing.assert_allclose(trajectory.positions[-1], [0.2, 0.0, 0.0])

    def test_run_hw_v2_uses_current_stack_constructor_and_controller_api(self):
        hw_sig = inspect.signature(HardwareInterface)
        for name in (
            "perf_enabled",
            "cleanup_disable_osc",
            "enable_adc",
            "start_adc_reader",
            "imu_rt_priority",
        ):
            self.assertIn(name, hw_sig.parameters)

        controller_sig = inspect.signature(ExcavatorController)
        self.assertIn("hardware_interface", controller_sig.parameters)
        self.assertIn("enable_perf_tracking", controller_sig.parameters)
        self.assertIn("log_level", controller_sig.parameters)

        for method in (
            "start",
            "pause",
            "resume",
            "give_pose",
            "get_pose",
            "get_joint_angles",
            "get_joint_velocities_degps",
            "get_condition_number",
            "get_performance_stats",
            "stop",
        ):
            self.assertTrue(hasattr(ExcavatorController, method), method)

    def test_read_controller_step_matches_current_controller_methods(self):
        class Controller:
            def get_pose(self):
                return np.asarray([1.0, 2.0, 3.0], dtype=np.float32), 12.0

            def get_joint_angles(self):
                return [0.0, 90.0, -90.0, 180.0]

            def get_joint_velocities_degps(self):
                return [0.0, 10.0, -10.0, 20.0]

            def get_condition_number(self):
                return 42.0

            def get_yoshikawa_index(self):
                return 0.25

            def get_singular_values(self):
                return [4.0, 3.0, 2.0, 1.0]

        pos, rot, joint_pos, joint_vel, cond, yoshikawa, singular_values = run_hw_v2._read_controller_step(Controller())

        np.testing.assert_allclose(pos, [1.0, 2.0, 3.0])
        self.assertEqual(rot, 12.0)
        np.testing.assert_allclose(joint_pos, np.radians([0.0, 90.0, -90.0, 180.0]))
        np.testing.assert_allclose(joint_vel, np.radians([0.0, 10.0, -10.0, 20.0]))
        self.assertEqual(cond, 42.0)
        self.assertEqual(yoshikawa, 0.25)
        np.testing.assert_allclose(singular_values, [4.0, 3.0, 2.0, 1.0])

    def test_main_dry_run_wires_hardware_controller_and_planning_stack(self):
        _FakeHardware.instances = []
        _FakeController.instances = []
        trajectory = run_hw_v2.PlannedTrajectory(
            positions=np.asarray([[0.55, 0.25, -0.035], [0.55, -0.25, -0.06]], dtype=np.float32),
            total_distance_m=0.5,
            calculation_time_s=0.01,
            chosen_variant="a_star",
        )
        fake_time = iter(float(i) for i in range(100))

        with patch.object(run_hw_v2, "HardwareInterface", _FakeHardware), \
                patch.object(run_hw_v2, "ExcavatorController", _FakeController), \
                patch.object(run_hw_v2, "_blind_move_to_start") as blind_mock, \
                patch.object(run_hw_v2, "_plan_trajectory", return_value=trajectory) as plan_mock, \
                patch.object(run_hw_v2, "_execute_on_hardware") as execute_mock, \
                patch.object(run_hw_v2.time, "time", side_effect=lambda: next(fake_time)), \
                patch.object(run_hw_v2.time, "sleep"):
            rc = run_hw_v2.main([
                "--algorithm", "a_star",
                "--robot", "rpi",
                "-r",
                "-p",
                "--task", "in-and-out",
                "--obstacles-json", str(ROOT_DIR / "obstacles.json"),
                "--once",
                "--rt-priority", "0",
                "--imu-priority", "0",
                "--adc-priority", "0",
            ])

        self.assertEqual(rc, 0)
        self.assertEqual(len(_FakeHardware.instances), 1)
        self.assertEqual(len(_FakeController.instances), 1)
        self.assertEqual(_FakeHardware.instances[0].kwargs["config_file"], "configuration_files/profiles/rpi/servo_config.yaml")
        self.assertEqual(_FakeHardware.instances[0].kwargs["control_config_file"], "configuration_files/profiles/rpi/control_config.yaml")
        self.assertEqual(_FakeHardware.instances[0].kwargs["pwm_i2c_bus"], 1)
        self.assertFalse(_FakeHardware.instances[0].kwargs["enable_adc"])
        self.assertFalse(_FakeHardware.instances[0].kwargs["start_adc_reader"])
        self.assertNotIn("adc_channels", _FakeHardware.instances[0].kwargs)
        self.assertNotIn("adc_rt_priority", _FakeHardware.instances[0].kwargs)
        self.assertEqual(_FakeHardware.instances[0].kwargs["imu_rt_priority"], 0)
        self.assertEqual(_FakeController.instances[0].kwargs["control_config_file"], "configuration_files/profiles/rpi/control_config.yaml")
        self.assertEqual(_FakeController.instances[0].kwargs["enable_perf_tracking"], True)
        blind_mock.assert_called_once()
        plan_kwargs = plan_mock.call_args.kwargs
        self.assertTrue(plan_kwargs["radial"])
        self.assertTrue(plan_kwargs["planar"])
        self.assertEqual(plan_kwargs["radial_mode"], "reconstructed")
        self.assertEqual(len(plan_kwargs["obstacles"]), 9)
        execute_mock.assert_called_once()
        self.assertGreaterEqual(_FakeController.instances[0].pause_calls, 1)
        self.assertEqual(_FakeController.instances[0].stop_calls, 1)
        self.assertGreaterEqual(_FakeHardware.instances[0].shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
