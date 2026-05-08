"""Quick smoke test for RRT* relative-improvement convergence.

Runs plan_radial_to_target with rrt_star_plane on the default in-and-out and rotation
task geometries, with a few values of (relative_improvement_tol, relative_improvement_window)
to see how iteration count and final path cost compare.
"""
from __future__ import annotations
import sys
import time
import logging
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

# Quiet down the planner logs so the table is readable.
logging.basicConfig(level=logging.WARNING, format="%(message)s")
# Silence the per-iteration RRT* logs; we only want the result-row output.
logging.getLogger("pathing.path_planning_algorithms").setLevel(logging.WARNING)

from configuration_files.pathing_config import DEFAULT_CONFIG, DEFAULT_ENV_CONFIG
from pathing.normalized_planners import NormalizerParams, plan_radial_to_target
from pathing.path_planning_algorithms import RRTStarParams


def make_obstacles():
    return [{
        "size": np.asarray(DEFAULT_ENV_CONFIG.wall_size, dtype=np.float32),
        "pos":  np.asarray(DEFAULT_ENV_CONFIG.wall_pos,  dtype=np.float32),
        "rot":  np.asarray(DEFAULT_ENV_CONFIG.wall_rot,  dtype=np.float32),
    }]


# Override workspace bounds the same way run_hw_v2.py does so negative-z is allowed.
from pathing import path_planning_algorithms as planner_algos
from pathing import path_utils as path_utils_mod

def _wide_workspace_bounds(obstacle_data, start_pos, goal_pos,
                           padding=0.1,
                           min_bounds=(-1.0, -1.0, -0.3),
                           max_bounds=(1.0, 1.0, 0.6)):
    points = [list(start_pos), list(goal_pos)]
    if obstacle_data:
        for obs in obstacle_data:
            points.append(list(obs.get("pos", obs.get("position", [0.0, 0.0, 0.0]))))
    pts = np.asarray(points, dtype=np.float32)
    bounds_min_calc = np.min(pts, axis=0) - padding
    bounds_max_calc = np.max(pts, axis=0) + padding
    bounds_min_calc = np.minimum(bounds_min_calc, np.asarray(min_bounds, dtype=np.float32))
    bounds_max_calc = np.maximum(bounds_max_calc, np.asarray(max_bounds, dtype=np.float32))
    return tuple(bounds_min_calc.tolist()), tuple(bounds_max_calc.tolist())

planner_algos.calculate_workspace_bounds = _wide_workspace_bounds
path_utils_mod.calculate_workspace_bounds = _wide_workspace_bounds


def run_one(label, start, goal, params: RRTStarParams):
    obstacles = make_obstacles()
    normalizer = NormalizerParams(
        speed_mps=DEFAULT_CONFIG.speed_mps,
        dt=DEFAULT_CONFIG.dt,
        return_poses=DEFAULT_CONFIG.normalizer_return_poses,
        force_goal=DEFAULT_CONFIG.normalizer_force_goal,
    )
    t0 = time.perf_counter()
    result = plan_radial_to_target(
        start_pos_world=tuple(start),
        target_pos_world=tuple(goal),
        obstacle_data=obstacles,
        fallback_algorithm="rrt_star_plane",
        grid_resolution=DEFAULT_CONFIG.grid_resolution,
        safety_margin=DEFAULT_CONFIG.safety_margin,
        normalizer_params=normalizer,
        rrt_star_params=params,
        use_3d=False,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    total_length = float(np.asarray(result.get("total_length_m", 0.0)).flatten()[0])
    return elapsed, total_length, result.get("chosen_variant", "")


def fmt_params(p: RRTStarParams):
    tol = "off" if p.relative_improvement_tol is None else f"{p.relative_improvement_tol*100:.1f}%"
    return f"tol={tol:>5} w={p.relative_improvement_window:<4} patience={p.cost_improvement_patience}"


def main():
    tasks = [
        ("in-and-out A->B", (0.55, 0.25, -0.15),  (0.55, -0.25, -0.20)),
        ("in-and-out B->A", (0.55, -0.25, -0.20), (0.55, 0.25, -0.15)),
        ("rotation   A->B", (-0.55, 0.25, -0.035),(-0.55, -0.25, -0.06)),
    ]

    # Param sweeps. Note: the OLD behaviour was tol=None, patience=5000,
    # plus the unreachable max_acceptable_cost=0.768 magic number.
    sweeps = [
        ("NEW default 1%/500", RRTStarParams()),
        ("looser  2%/300   ", RRTStarParams(relative_improvement_tol=0.02, relative_improvement_window=300)),
        ("tighter 0.5%/800 ", RRTStarParams(relative_improvement_tol=0.005, relative_improvement_window=800)),
    ]

    print()
    print(f"{'task':<18} {'config':<55} {'time_s':>7} {'cost_mm':>9} {'variant':<22}")
    print("-" * 115)
    for task_name, start, goal in tasks:
        for cfg_label, params in sweeps:
            try:
                elapsed, total_length, variant = run_one(task_name, start, goal, params)
                print(f"{task_name:<18} {cfg_label:<55} {elapsed:>7.2f} {total_length*1000:>9.1f} {variant:<22}")
            except Exception as e:
                print(f"{task_name:<18} {cfg_label:<55} ERROR: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
