"""
Assignment 3: visual servoing and image-space path planning with a UR5.

    python main.py --task1             # IBVS towards a static target
    python main.py --task2             # IBVS tracking a moving target
    python main.py --task3             # why IBVS alone fails near an obstacle, then image-space RRT
    python main.py --task4             # image-space RRT*, compared with RRT
    python main.py --all               # all four tasks
    python main.py --task3 --preview   # only show the scene

Each task writes its outputs to results/<task folder>/.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (DATASET_SIZE_FULL, GOAL_FEATURE, IBVS_STEPS, MOVING_STEPS,
                    OBSTACLE_ENVIRONMENTS, OBSTACLE_IMAGE_MARGIN_PX, RRT_MAX_ITER,
                    RRT_STAR_ITER, TARGET_CIRCLE, TASKS, CONTROL_DT)
from controller import IBVSTrajectoryController
from ibvs import IBVSServo
from kinematics import UR5Kinematics
from planner import ImageBasedRRTPlanner, WorkspaceDatasetGenerator
from plotting import (image_clearance, plot_planning, plot_rrt_vs_star, plot_task1,
                      plot_task2, save_outputs, update_summary_table)
from simulation import PyBulletUR5Env
from visualizer import (LiveCameraVisualizer, RRTTreeView, motion_sequence_frames,
                        save_frame_grid, setup_pybullet_gui)

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


# --------------------- helpers -------------------------------------------

def make_env(num, gui):
    """Scene of a task (Tasks 3-4 use one of the obstacle environments)."""
    cfg = TASKS[num]
    scene, walls = cfg["scene"], []
    if "environment" in cfg:
        envcfg = OBSTACLE_ENVIRONMENTS[cfg["environment"]]
        scene = dict(scene, q0=envcfg["q0"], view=envcfg["view"])
        walls = envcfg["walls"]
    env = PyBulletUR5Env(render_mode=p.GUI if gui else p.DIRECT, scene=scene, obstacles=walls)
    if gui:
        setup_pybullet_gui(env, cfg["title"])
    return env


def load_or_generate_dataset(env, n=DATASET_SIZE_FULL, seed=7):
    """Load the scene's workspace dataset from data/, or generate it."""
    path = DATA_DIR / f"dataset_{env.scene['name']}_{n}.npz"
    gen = WorkspaceDatasetGenerator(env)
    if path.is_file():
        ds = gen.load(path)
        if gen.is_compatible(ds, n):
            print(f"[Dataset] Loaded {n} workspace samples from {path}")
            return ds
    print(f"[Dataset] Generating a new {n}-sample workspace dataset (about a minute)...")
    return gen.generate(n, save_path=path, seed=seed)


def circle_motion(center, radius=TARGET_CIRCLE["radius"], period=TARGET_CIRCLE["period"]):
    """Target position at time t: a circle on the table that starts at ``center``."""
    center = np.asarray(center, dtype=np.float64)
    w = 2 * np.pi / period
    return lambda t: center + [radius * np.sin(w * t), radius * (1 - np.cos(w * t)), 0.0]


def _banner(title):
    print("\n" + "=" * 74 + f"\n {title.upper()}\n" + "=" * 74)


def _finish_gui(gui, vis, result, hold):
    if gui:
        vis.show_completion(result, hold_seconds=hold)
        cv2.destroyAllWindows()


# --------------------- Task 1: IBVS, static target -----------------------

def run_task1(out_dir, gui, hold, quick):
    cfg = TASKS[1]
    _banner(cfg["title"])
    env = make_env(1, gui)
    try:
        kin = UR5Kinematics(env.robot_id, env.ee_link_index)
        vis = LiveCameraVisualizer(env, cfg["title"], is_gui=gui, extra_status="Classic IBVS")
        res = IBVSServo(env, kin, GOAL_FEATURE).run(IBVS_STEPS, step_callback=vis)
        err = res["errors"]
        below = np.flatnonzero(err < 10.0)
        metrics = {
            "status": res["status"], "steps": int(len(err)),
            "initial_error_px": float(err[0]), "final_error_px": float(err[-1]),
            "time_to_10px_s": float(res["t"][below[0]]) if len(below) else None,
            "initial_feature": res["features"][0].tolist(),
            "final_feature": res["features"][-1].tolist(),
        }
        metrics["headline"] = (f"error {err[0]:.0f} px -> {err[-1]:.1f} px in {len(err) * CONTROL_DT:.0f} s"
                               f" ({res['status']})")
        save_outputs("task1", cfg["title"], out_dir, metrics,
                     {k: res[k] for k in ("t", "features", "errors", "camera_velocities",
                                          "joint_velocities", "joints")})
        plot_task1(res, GOAL_FEATURE, out_dir / "task1_results.png")
        if vis.last_frame is not None:
            cv2.imwrite(str(out_dir / "task1_camera_final.png"), vis.last_frame)
        _finish_gui(gui, vis, {"initial_error": err[0], "final_error": err[-1]}, hold)
        frames = motion_sequence_frames(env, res["joints"], 6)
        save_frame_grid([frames[:3], frames[3:]], out_dir / "task1_motion_sequence.png")
    finally:
        env.close()


# --------------------- Task 2: IBVS, moving target -----------------------

def run_task2(out_dir, gui, hold, quick):
    cfg = TASKS[2]
    _banner(cfg["title"])
    env = make_env(2, gui)
    try:
        kin = UR5Kinematics(env.robot_id, env.ee_link_index)
        start = np.array(env.target_position)
        motion = circle_motion(start)
        runs = {}
        for name, comp in (("classic", False), ("compensated", True)):
            print(f"\n[Task 2] {name} IBVS, target moving on a circle...")
            env.set_target_position(start)
            vis = LiveCameraVisualizer(env, cfg["title"], is_gui=gui,
                                       extra_status="With compensation" if comp else "Classic IBVS")
            runs[name] = IBVSServo(env, kin, GOAL_FEATURE, compensate=comp).run(
                MOVING_STEPS, target_motion=motion, step_callback=vis)
            if name == "classic" and gui:
                cv2.destroyAllWindows()
        plain, comp = runs["classic"], runs["compensated"]
        half = len(comp["t"]) // 2                       # steady state: second half of the run
        lag_plain = float(np.mean(plain["xy_errors"][half:]))
        lag_comp = float(np.mean(comp["xy_errors"][half:]))
        metrics = {
            "status_classic": plain["status"], "status_compensated": comp["status"],
            "steady_error_classic_px": lag_plain, "steady_error_compensated_px": lag_comp,
            "max_steady_error_classic_px": float(np.max(plain["xy_errors"][half:])),
            "max_steady_error_compensated_px": float(np.max(comp["xy_errors"][half:])),
            "target_circle": TARGET_CIRCLE,
        }
        metrics["headline"] = f"steady tracking error {lag_plain:.0f} px (classic) -> {lag_comp:.1f} px (compensated)"
        save_outputs("task2", cfg["title"], out_dir, metrics,
                     {f"{n}_{k}": runs[n][k] for n in runs
                      for k in ("t", "features", "xy_errors", "camera_velocities", "motion_estimates")})
        plot_task2(plain, comp, GOAL_FEATURE, out_dir / "task2_results.png")
        if vis.last_frame is not None:
            cv2.imwrite(str(out_dir / "task2_camera_final.png"), vis.last_frame)
        _finish_gui(gui, vis, {"initial_error": comp["xy_errors"][0], "final_error": comp["xy_errors"][-1]}, hold)
        frames = motion_sequence_frames(env, comp["joints"], 6,
                                        before_frame=lambda i: env.set_target_position(motion(i * CONTROL_DT)))
        save_frame_grid([frames[:3], frames[3:]], out_dir / "task2_motion_sequence.png")
    finally:
        env.close()


# --------------------- Tasks 3-4: image-space planning -------------------

def _execute(env, kin, planned_s, goal, vis):
    ctrl = IBVSTrajectoryController(env, kin, secondary_task="fixed_orientation")
    print("\n[Execution] Servoing along the planned feature trajectory...")
    result = ctrl.execute_trajectory(planned_s, env.q0, evaluation_goal=goal, step_callback=vis)
    img_clear = image_clearance(env, result["executed_joints"], result["executed_features"],
                                result["depths"])
    return result, img_clear


def _execution_metrics(result, img_clear):
    return {
        "execution_status": result["termination_reason"],
        "control_steps": int(len(result["tracking_errors"])),
        "motion_time_s": float(len(result["tracking_errors"]) * CONTROL_DT),
        "initial_error_px": result["initial_error"],
        "final_error_px": result["final_error"],
        "mean_tracking_error_px": float(np.mean(result["tracking_errors"])),
        "min_clearance_cm": float(100 * result["obstacle_clearance"].min()),
        "collisions": int(np.sum(result["obstacle_clearance"] <= 0.0)),
        # None = the obstacle never appeared in front of the target in the image
        "min_image_clearance_px": (None if np.all(np.isnan(img_clear))
                                   else float(np.nanmin(img_clear))),
    }


def _planner(env, kin, ds, num):
    seed = OBSTACLE_ENVIRONMENTS[TASKS[num]["environment"]]["rrt_seed"]
    return ImageBasedRRTPlanner(env, kin, ds, secondary_task="fixed_orientation", random_seed=seed)


def run_task3(out_dir, gui, hold, quick):
    cfg = TASKS[3]
    _banner(cfg["title"])
    env = make_env(3, gui)
    try:
        kin = UR5Kinematics(env.robot_id, env.ee_link_index)
        regions = env.project_obstacles(OBSTACLE_IMAGE_MARGIN_PX)

        # 1. Classic IBVS (Task 1) goes straight for the goal and hits the obstacle
        print("\n[Task 3] First try classic IBVS from Task 1 ...")
        vis = LiveCameraVisualizer(env, cfg["title"], is_gui=gui, extra_status="Classic IBVS (no planning)")
        direct = IBVSServo(env, kin, GOAL_FEATURE).run(IBVS_STEPS, step_callback=vis)
        print(f"[Task 3] Classic IBVS result: {direct['status']} after {len(direct['t'])} steps "
              f"-> the straight route is blocked, so we plan a path in image space.")
        if gui:
            time.sleep(1.5)
            cv2.destroyAllWindows()
        direct_joints = list(direct["joints"]) + ([direct["collision_q"]]
                                                  if direct["collision_q"] is not None else [])

        # 2. Plan with RRT, watching the tree grow
        ds = load_or_generate_dataset(env)
        planner = _planner(env, kin, ds, 3)
        view = RRTTreeView(env, GOAL_FEATURE, cfg["title"], live=gui, regions=regions, name="RRT")
        t0 = time.time()
        planned_s, _, tree_f, tree_c = planner.plan(env.q0, GOAL_FEATURE,
                                                    max_iter=500 if quick else RRT_MAX_ITER,
                                                    allow_partial=quick, on_new_node=view.on_new_node)
        plan_time = time.time() - t0
        view.finish(tree_f, tree_c, planner.last_plan_result["goal_id"], out_dir / "task3_rrt_tree.png")

        # 3. Execute the plan with trajectory-tracking IBVS
        vis = LiveCameraVisualizer(env, cfg["title"], is_gui=gui, extra_status="Following the RRT path")
        result, img_clear = _execute(env, kin, planned_s, GOAL_FEATURE, vis)
        info = planner.last_plan_result
        metrics = {"classic_ibvs_status": direct["status"],
                   "classic_ibvs_steps": int(len(direct["t"])),
                   "rrt_nodes": int(info["explored_nodes"]), "rrt_iterations": int(info["iterations"]),
                   "path_cost": float(info["path_cost"]), "planning_time_s": float(plan_time),
                   "waypoints": int(len(planned_s))}
        metrics.update(_execution_metrics(result, img_clear))
        metrics["headline"] = (f"classic IBVS: {direct['status']}; RRT: {metrics['collisions']} collisions, "
                               f"final error {result['final_error']:.0f} px, "
                               f"min clearance {metrics['min_clearance_cm']:.1f} cm")
        save_outputs("task3", cfg["title"], out_dir, metrics,
                     {"planned_features": planned_s, "executed_features": result["executed_features"],
                      "executed_joints": result["executed_joints"],
                      "obstacle_clearance": result["obstacle_clearance"],
                      "classic_ibvs_joints": np.asarray(direct_joints)})
        plot_planning(planned_s, result, GOAL_FEATURE, regions, img_clear, out_dir / "task3_results.png")
        if vis.last_frame is not None:
            cv2.imwrite(str(out_dir / "task3_camera_final.png"), vis.last_frame)
        _finish_gui(gui, vis, result, hold)

        crash = motion_sequence_frames(env, direct_joints, min(4, len(direct_joints)))
        if direct["status"] == "collision":
            cv2.putText(crash[-1], "COLLISION", (150, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 3)
        plan = (planned_s, result["executed_features"], regions)
        save_frame_grid([crash, motion_sequence_frames(env, result["executed_joints"], 6, plan=plan)],
                        out_dir / "task3_motion_sequence.png",
                        ["classic IBVS (no planning)", "RRT path"])
    finally:
        env.close()


def run_task4(out_dir, gui, hold, quick):
    cfg = TASKS[4]
    _banner(cfg["title"])
    env = make_env(4, gui)
    try:
        kin = UR5Kinematics(env.robot_id, env.ee_link_index)
        regions = env.project_obstacles(OBSTACLE_IMAGE_MARGIN_PX)
        ds = load_or_generate_dataset(env)

        # 1. RRT (Task 3) on the same problem, for comparison
        print("\n[Task 4] RRT for comparison ...")
        rrt = _planner(env, kin, ds, 4)
        t0 = time.time()
        rrt_s, _, rrt_f, _ = rrt.plan(env.q0, GOAL_FEATURE, max_iter=500 if quick else RRT_MAX_ITER,
                                      allow_partial=quick)
        rrt_time = time.time() - t0

        # 2. RRT*, watching the tree grow and rewire
        print("\n[Task 4] RRT* ...")
        star = _planner(env, kin, ds, 4)
        view = RRTTreeView(env, GOAL_FEATURE, cfg["title"], live=gui, regions=regions, name="RRT*")
        t0 = time.time()
        planned_s, _, tree_f, tree_c = star.plan_star(env.q0, GOAL_FEATURE,
                                                      max_iter=800 if quick else RRT_STAR_ITER,
                                                      allow_partial=quick, on_new_node=view.on_new_node)
        star_time = time.time() - t0
        view.finish(tree_f, tree_c, star.last_plan_result["goal_id"], out_dir / "task4_rrt_star_tree.png")
        plot_rrt_vs_star([("RRT", rrt_f, rrt.last_plan_result["goal_id"], rrt.last_plan_result["path_cost"], rrt_time),
                          ("RRT*", tree_f, star.last_plan_result["goal_id"], star.last_plan_result["path_cost"], star_time)],
                         GOAL_FEATURE, regions, out_dir / "task4_rrt_vs_rrt_star.png")

        # 3. Execute the RRT* path
        vis = LiveCameraVisualizer(env, cfg["title"], is_gui=gui, extra_status="Following the RRT* path")
        result, img_clear = _execute(env, kin, planned_s, GOAL_FEATURE, vis)
        metrics = {"rrt_path_cost": float(rrt.last_plan_result["path_cost"]),
                   "rrt_star_path_cost": float(star.last_plan_result["path_cost"]),
                   "rrt_nodes": int(len(rrt_f)), "rrt_star_nodes": int(len(tree_f)),
                   "rrt_planning_time_s": float(rrt_time), "rrt_star_planning_time_s": float(star_time),
                   "rrt_waypoints": int(len(rrt_s)), "rrt_star_waypoints": int(len(planned_s))}
        metrics.update(_execution_metrics(result, img_clear))
        metrics["headline"] = (f"path cost RRT {metrics['rrt_path_cost']:.0f} -> RRT* {metrics['rrt_star_path_cost']:.0f}; "
                               f"{metrics['collisions']} collisions, final error {result['final_error']:.0f} px, "
                               f"motion {metrics['motion_time_s']:.0f} s")
        save_outputs("task4", cfg["title"], out_dir, metrics,
                     {"planned_features": planned_s, "rrt_planned_features": rrt_s,
                      "executed_features": result["executed_features"],
                      "executed_joints": result["executed_joints"],
                      "obstacle_clearance": result["obstacle_clearance"]})
        plot_planning(planned_s, result, GOAL_FEATURE, regions, img_clear, out_dir / "task4_results.png")
        if vis.last_frame is not None:
            cv2.imwrite(str(out_dir / "task4_camera_final.png"), vis.last_frame)
        _finish_gui(gui, vis, result, hold)
        plan = (planned_s, result["executed_features"], regions)
        save_frame_grid([motion_sequence_frames(env, result["executed_joints"], 6, plan=plan)],
                        out_dir / "task4_motion_sequence.png", ["RRT* path"])
    finally:
        env.close()


RUNNERS = {1: run_task1, 2: run_task2, 3: run_task3, 4: run_task4}


def preview_task(num, hold_seconds=10.0):
    """Show a task's scene and the camera view at the start (no TODOs needed)."""
    env = make_env(num, gui=True)
    try:
        feat, z, rgb, mask = env.get_current_feature(return_mask=True)
        print(f"[Preview] {TASKS[num]['title']}: s0 = {np.round(feat, 1)}, goal s* = {GOAL_FEATURE}")
        vis = LiveCameraVisualizer(env, TASKS[num]["title"], is_gui=True, extra_status="Preview")
        vis({"rgb": rgb, "mask": mask, "current_feature": feat, "target_waypoint": GOAL_FEATURE,
             "step": 0, "waypoint_idx": 0, "total_waypoints": 1, "status": "Preview", "depth": z,
             "goal_error": float(np.linalg.norm(feat - GOAL_FEATURE))})
        t0 = time.time()
        while time.time() - t0 < hold_seconds and p.isConnected(env.physics_client):
            if (cv2.waitKey(50) & 0xFF) in (27, ord("q")):
                break
        cv2.destroyAllWindows()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Assignment 3: visual servoing and image-space path planning (UR5)")
    for n in TASKS:
        parser.add_argument(f"--task{n}", action="store_true", help=TASKS[n]["title"])
    parser.add_argument("--all", action="store_true", help="Run all four tasks")
    parser.add_argument("--headless", action="store_true", help="Run without GUI windows")
    parser.add_argument("--hold", type=float, default=5.0, help="Seconds to keep the GUI open after a run")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test with fewer planner iterations")
    parser.add_argument("--preview", action="store_true", help="Only show the scene of the selected task(s)")
    args = parser.parse_args()

    selected = list(TASKS) if args.all else [n for n in TASKS if getattr(args, f"task{n}")]
    if not selected:
        parser.print_help()
        return
    for n in selected:
        if args.preview:
            preview_task(n, hold_seconds=max(args.hold, 10.0))
            continue
        out_dir = RESULTS_DIR / TASKS[n]["folder"]
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            RUNNERS[n](out_dir, not args.headless, args.hold, args.quick)
        except NotImplementedError as e:
            print(f"\n[Assignment] Task {n} stopped: {e} is not implemented yet.")
            print("[Assignment] See README.md -> 'Tasks' and run the unit tests:")
            print(f"             python -m unittest tests.test_task{n} -v")
            sys.exit(1)
        update_summary_table(RESULTS_DIR, TASKS)
        print(f"\n[Output] Task {n} results are in {out_dir}")


if __name__ == "__main__":
    main()
