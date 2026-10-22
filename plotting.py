"""
Result figures and result files for the four tasks.
"""

import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from config import (CONTROL_DT, FOV_X_MAX, FOV_X_MIN, FOV_Y_MAX, FOV_Y_MIN,
                    OBSTACLE_IMAGE_MARGIN_PX, OBSTACLE_SAFETY_MARGIN)
from visualizer import draw_rrt_tree, tree_path

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "grid.linestyle": "--", "grid.alpha": 0.6,
})
_CAM_LABELS = ["$v_x$", "$v_y$", "$v_z$", "$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]


def _image_axes(ax, title):
    ax.add_patch(Rectangle((FOV_X_MIN, FOV_Y_MIN), FOV_X_MAX - FOV_X_MIN, FOV_Y_MAX - FOV_Y_MIN,
                           fill=False, ec="0.4", lw=1.2))
    ax.set(xlim=(FOV_X_MIN - 20, FOV_X_MAX + 20), ylim=(FOV_Y_MAX + 20, FOV_Y_MIN - 20),
           xlabel="x (px)", ylabel="y (px)", title=title)       # image y points down
    ax.set_aspect("equal")
    ax.grid(True)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {path}")


# --------------------- Task 1 --------------------------------------------

def plot_task1(res, goal, save_path):
    """Image path of the target, error, camera and joint velocities."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    f, t = res["features"], res["t"]
    ax = axs[0, 0]
    ax.plot(f[:, 0], f[:, 1], "b.-", ms=3, label="target centroid")
    ax.plot(*f[0, :2], "rs", label="start")
    ax.plot(*goal[:2], "g*", ms=14, label="goal")
    _image_axes(ax, "(a) Target in the image")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    ax.semilogy(t, np.maximum(res["errors"], 1e-3), "b-")
    ax.set(xlabel="Time (s)", ylabel="$||s - s^*||$ (px)", title="(b) Feature error (log scale)")
    ax.grid(True)
    ax = axs[1, 0]
    for i in range(6):
        ax.plot(t, res["camera_velocities"][:, i], label=_CAM_LABELS[i])
    ax.set(xlabel="Time (s)", ylabel="m/s, rad/s", title="(c) Camera velocity $v = -\\lambda L^+ e$")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True)
    ax = axs[1, 1]
    for i in range(6):
        ax.plot(t, res["joint_velocities"][:, i], label=f"$\\dot q_{i + 1}$")
    ax.set(xlabel="Time (s)", ylabel="rad/s", title="(d) Joint velocities")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True)
    _save(fig, save_path)


# --------------------- Task 2 --------------------------------------------

def plot_task2(plain, comp, goal, save_path):
    """Classic IBVS vs IBVS with target-motion compensation."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax = axs[0, 0]
    ax.plot(plain["t"], plain["xy_errors"], "r-", label="classic IBVS (TODO 2)")
    ax.plot(comp["t"], comp["xy_errors"], "b-", label="with compensation (TODOs 3-4)")
    ax.set(xlabel="Time (s)", ylabel="Centroid error (px)", title="(a) Tracking error of the moving target")
    ax.legend()
    ax.grid(True)
    ax = axs[0, 1]
    ax.plot(plain["features"][:, 0], plain["features"][:, 1], "r-", lw=1, label="classic IBVS")
    ax.plot(comp["features"][:, 0], comp["features"][:, 1], "b-", lw=1, label="with compensation")
    ax.plot(*goal[:2], "g*", ms=14, label="goal")
    _image_axes(ax, "(b) Target centroid in the image")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    for i, name in enumerate(("x", "y")):
        ax.plot(comp["t"], comp["motion_estimates"][:, i], label=f"estimated $\\partial e_{name}/\\partial t$")
    ax.set(xlabel="Time (s)", ylabel="px/s", title="(c) Estimated feature rate caused by the target")
    ax.legend()
    ax.grid(True)
    ax = axs[1, 1]
    for i in range(6):
        ax.plot(comp["t"], comp["camera_velocities"][:, i], label=_CAM_LABELS[i])
    ax.set(xlabel="Time (s)", ylabel="m/s, rad/s", title="(d) Camera velocity (with compensation)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True)
    _save(fig, save_path)


# --------------------- Tasks 3-4 -----------------------------------------

def image_clearance(env, executed_joints, executed_features, depths):
    """Pixel distance from the target centroid to the nearest obstacle region
    (with offset) in the current camera view; negative = target behind it."""
    out = []
    for q, s, z in zip(executed_joints, executed_features, depths):
        env.reset_joint_state(q)
        d = [cv2.pointPolygonTest(h.astype(np.float32).reshape(-1, 1, 2),
                                  (float(s[0]), float(s[1])), True)
             for h, depth in env.project_obstacles(OBSTACLE_IMAGE_MARGIN_PX) if depth < z]
        out.append(-max(d) if d else np.nan)
    return np.asarray(out)


def plot_planning(planned, result, goal, regions, img_clear, save_path):
    """Execution of a planned path: image path, clearances and errors."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    t = np.arange(len(result["tracking_errors"])) * CONTROL_DT
    ex = result["executed_features"]
    ax = axs[0, 0]
    for k, (hull, _) in enumerate(regions):
        ax.add_patch(Polygon(hull, closed=True, fc=(0.9, 0.1, 0.1, 0.2), ec="r",
                             label="obstacle + offset (start view)" if k == 0 else None))
    ax.plot(planned[:, 0], planned[:, 1], "b-", lw=2, label="planned")
    ax.plot(ex[:, 0], ex[:, 1], "m--", lw=2, label="executed")
    ax.plot(*planned[0, :2], "rs", label="start")
    ax.plot(*goal[:2], "g*", ms=14, label="goal")
    _image_axes(ax, "(a) Path in the image")
    ax.legend(fontsize=7, loc="lower left")
    ax = axs[0, 1]
    ax.plot(t, 100 * result["obstacle_clearance"], "k-", lw=1.8, label="robot-obstacle distance")
    ax.axhline(100 * OBSTACLE_SAFETY_MARGIN, color="r", ls="--", label="safety offset")
    ax.set(xlabel="Time (s)", ylabel="cm", title="(b) Clearance to the obstacle")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True)
    ax = axs[1, 0]
    if np.all(np.isnan(img_clear)):
        ax.text(0.5, 0.5, "The obstacle never appears\nin front of the target", ha="center",
                va="center", transform=ax.transAxes, fontsize=12)
    else:
        ax.plot(t, img_clear, "k-", lw=1.5)
        ax.axhline(0, color="r", ls="--", label="obstacle region boundary")
        ax.legend()
    ax.set(xlabel="Time (s)", ylabel="px", title="(c) Target never hidden by the obstacle")
    ax.grid(True)
    ax = axs[1, 1]
    ax.plot(t, result["tracking_errors"], "b-", label="error to the planned feature")
    ax.plot(t, result["goal_errors"], "g-", label="error to the goal")
    ax.set(xlabel="Time (s)", ylabel="px", title="(d) Tracking and goal error")
    ax.legend()
    ax.grid(True)
    _save(fig, save_path)


def plot_rrt_vs_star(runs, goal, regions, save_path):
    """Image-plane trees and paths of RRT and RRT*, and their path costs.

    runs: list of (name, tree_f, goal_id, path_cost, plan_time).
    """
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.7]})
    for ax, (name, tree_f, goal_id, cost, secs) in zip(axs[:2], runs):
        for hull, _ in regions:
            ax.add_patch(Polygon(hull, closed=True, fc=(0.9, 0.1, 0.1, 0.15), ec="r"))
        pts = np.array([n.s for n in tree_f])
        draw_rrt_tree(ax, pts[:, :2], [n.parent_id for n in tree_f], tree_path(tree_f, goal_id))
        ax.plot(*pts[0, :2], "ks", ms=8)
        ax.plot(*goal[:2], "g*", ms=14)
        _image_axes(ax, f"{name}: {len(pts)} nodes, path cost {cost:.0f}")
    ax = axs[2]
    names = [r[0] for r in runs]
    bars = ax.bar(names, [r[3] for r in runs], color=["tab:orange", "tab:blue"])
    ax.bar_label(bars, fmt="%.0f")
    ax.set(ylabel="Path length in feature space", title="Path cost (lower is better)")
    ax.grid(True, axis="y")
    _save(fig, save_path)


# --------------------- Result files --------------------------------------

def save_outputs(tag, title, out_dir, metrics, arrays):
    """Write the metrics (json), a readable summary (txt) and the data (npz)."""
    np.savez(out_dir / f"{tag}_data.npz", **arrays)
    with open(out_dir / f"{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    lines = [title, "=" * len(title)]
    for key, value in metrics.items():
        if isinstance(value, float):
            value = f"{value:.4g}"
        elif isinstance(value, list):
            value = np.round(value, 2).tolist()
        lines.append(f"{key.replace('_', ' '):28s}: {value}")
    text = "\n".join(lines)
    (out_dir / f"{tag}_summary.txt").write_text(text + "\n")
    print("\n" + text)


def update_summary_table(results_dir, tasks):
    """Rebuild results/summary.md from the saved metrics of every task."""
    rows = []
    for num, cfg in tasks.items():
        path = results_dir / cfg["folder"] / f"task{num}_metrics.json"
        if path.is_file():
            m = json.loads(path.read_text())
            rows.append(f"| {num} | {cfg['title'].split(': ', 1)[1]} | {m['headline']} |")
    (results_dir / "summary.md").write_text(
        "\n".join(["# Results", "", "| Task | | Result |", "|:---:|:---|:---|"] + rows) + "\n")
