"""
Visualisation: PyBullet GUI, live camera view, live RRT tree and the saved
motion-sequence and RRT figures.

Student TODO (Task 3):
  - draw_rrt_tree : TODO 6  draw the RRT and the path from start to goal
"""

import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from matplotlib.patches import Polygon

from config import (
    CX, CY, FOV_X_MAX, FOV_X_MIN, FOV_Y_MAX, FOV_Y_MIN,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    GOAL_FEATURE,
)


def setup_pybullet_gui(env, title="Visual Servoing"):
    """Configure PyBullet GUI camera and clean up visualizer elements."""
    if env.render_mode != p.GUI:
        return
    pc = env.physics_client
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=pc)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=pc)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=pc)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=pc)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=pc)
    # No mouse picking: dragging the arm in the GUI would disturb the servoing.
    # Rotate the view with Ctrl + left drag, zoom with the mouse wheel.
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=pc)

    # Position external camera for a clear 3D perspective of robot, table, and target
    view = env.scene["view"]
    p.resetDebugVisualizerCamera(
        cameraDistance=view["distance"],
        cameraYaw=view["yaw"],
        cameraPitch=view["pitch"],
        cameraTargetPosition=list(view["target"]),
        physicsClientId=pc,
    )

    # 3D title text floating above workspace
    p.addUserDebugText(
        f"UR5 Eye-in-Hand IBVS: {title}",
        [0.30, 0.30, 1.25],
        textColorRGB=[0.05, 0.1, 0.3],
        textSize=1.3,
        lifeTime=0,
        physicsClientId=pc,
    )


class LiveCameraVisualizer:
    """Manages real-time OpenCV camera preview with HUD and PyBullet 3D path."""

    def __init__(self, env, title, is_gui=True, extra_status=""):
        self.env = env
        self.title = title
        self.is_gui = is_gui and (env.render_mode == p.GUI)
        self.extra_status = extra_status
        self.window_name = f"Eye-in-Hand Camera (Robot View) - {title}"
        self.prev_ee_pos = None
        self.line_color = [0.0, 0.75, 1.0]  # Cyan 3D path line
        self.step_delay_ms = 1
        self.last_frame = None
        self.user_quit = False

        if self.is_gui:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(self.window_name, 60, 60)
            except Exception as e:
                print(f"[Visualizer] OpenCV window init note: {e}")
                self.is_gui = False

    def draw_hud(self, rgb, mask, cur_feat, des_feat, goal, step_info):
        """Render target outline, reticles, error arrow, telemetry, and area meter."""
        if rgb is None:
            return None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        # 1. Target Contour and Bounding Box
        if mask is not None and np.count_nonzero(mask) > 10:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(bgr, contours, -1, (0, 255, 0), 2)
                bx, by, bw, bh = cv2.boundingRect(contours[0])
                cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), (0, 200, 100), 1)

        # 2. Desired Target Reticle at Image Center (ud=320, vd=240)
        ud, vd = int(CX), int(CY)
        cv2.circle(bgr, (ud, vd), 16, (255, 215, 0), 2)  # Gold circle
        cv2.circle(bgr, (ud, vd), 3, (255, 215, 0), -1)
        cv2.line(bgr, (ud - 22, vd), (ud - 7, vd), (255, 215, 0), 1)
        cv2.line(bgr, (ud + 7, vd), (ud + 22, vd), (255, 215, 0), 1)
        cv2.line(bgr, (ud, vd - 22), (ud, vd - 7), (255, 215, 0), 1)
        cv2.line(bgr, (ud, vd + 7), (ud, vd + 22), (255, 215, 0), 1)
        cv2.putText(bgr, "s* Goal [0, 0]", (ud + 18, vd - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 215, 0), 1)

        # 3. Current Target Centroid & Error Arrow
        if cur_feat is not None and np.all(np.isfinite(cur_feat)):
            u = int(round(cur_feat[0] + CX))
            v = int(round(cur_feat[1] + CY))
            u = max(0, min(w - 1, u))
            v = max(0, min(h - 1, v))

            # Current centroid crosshair
            cv2.circle(bgr, (u, v), 6, (0, 255, 0), -1)
            cv2.circle(bgr, (u, v), 12, (0, 255, 0), 2)
            cv2.line(bgr, (u - 15, v), (u + 15, v), (0, 255, 0), 1)
            cv2.line(bgr, (u, v - 15), (u, v + 15), (0, 255, 0), 1)

            # Error vector (current centroid -> desired image center)
            dist_2d = float(np.hypot(ud - u, vd - v))
            if dist_2d > 6:
                tip = max(0.12, min(0.35, 12.0 / dist_2d))
                cv2.arrowedLine(bgr, (u, v), (ud, vd), (0, 0, 255), 2, tipLength=tip)

        # 4. Semi-transparent Top HUD Banner
        overlay = bgr.copy()
        cv2.rectangle(overlay, (0, 0), (w, 68), (18, 18, 18), -1)
        cv2.rectangle(overlay, (0, h - 85), (w, h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.78, bgr, 0.22, 0, bgr)

        # Top Banner Text
        cv2.putText(bgr, self.title.upper(),
                    (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        wi = step_info.get("waypoint_idx", 0) + 1
        tw = step_info.get("total_waypoints", 1)
        step = step_info.get("step", 0)
        status = step_info.get("status", "Servoing")
        progress_pct = int(100.0 * wi / max(1, tw))
        extra = self.extra_status
        if step_info.get("clearance") is not None:
            extra = f"Clearance: {100 * step_info['clearance']:.1f} cm"

        cv2.putText(
            bgr,
            f"{status} | Waypoint {wi}/{tw} ({progress_pct}%) | Step {step} | {extra}",
            (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 255), 1,
        )

        # 5. Bottom Banner Telemetry
        xc = cur_feat[0] if cur_feat is not None else 0.0
        yc = cur_feat[1] if cur_feat is not None else 0.0
        area = cur_feat[2] if cur_feat is not None else 0.0
        goal_err = step_info.get("goal_error", 0.0)
        trk_err = step_info.get("tracking_error", 0.0)
        q_vel = step_info.get("joint_velocities", np.zeros(6))
        spd_norm = float(np.linalg.norm(q_vel))

        line1 = f"Current  s = [xc: {xc:+6.1f} px, yc: {yc:+6.1f} px, area: {int(area):4d} px]"
        line2 = f"Desired s* = [xc:   +0.0 px, yc:   +0.0 px, area: 3000 px] | Depth: {step_info.get('depth', 0.5):.3f} m"
        line3 = f"Error ||s - s*||: {goal_err:6.1f} px | Wp Err: {trk_err:5.1f} px | Speed: {spd_norm:.4f} rad/s"

        cv2.putText(bgr, line1, (14, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 255, 200), 1)
        cv2.putText(bgr, line2, (14, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 220, 150), 1)
        cv2.putText(bgr, line3, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 255), 1)

        # 6. Right-hand Vertical Area Meter
        gauge_x = w - 32
        gauge_y0, gauge_y1 = 80, h - 95
        gauge_h = gauge_y1 - gauge_y0
        cv2.rectangle(bgr, (gauge_x, gauge_y0), (gauge_x + 16, gauge_y1), (60, 60, 60), -1)
        fill_ratio = max(0.0, min(1.0, float(area) / 3000.0))
        fill_h = int(fill_ratio * gauge_h)
        bar_color = (0, 255, 0) if fill_ratio >= 0.9 else (0, 200, 255)
        cv2.rectangle(bgr, (gauge_x + 2, gauge_y1 - fill_h), (gauge_x + 14, gauge_y1 - 2), bar_color, -1)
        cv2.rectangle(bgr, (gauge_x, gauge_y0), (gauge_x + 16, gauge_y1), (180, 180, 180), 1)
        cv2.putText(bgr, "Area", (gauge_x - 10, gauge_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        return bgr

    def __call__(self, step_info):
        """Step callback interface invoked by the IBVS controller."""
        rgb = step_info.get("rgb")
        mask = step_info.get("mask")
        cur_feat = step_info.get("current_feature")
        des_feat = step_info.get("target_waypoint")
        goal = step_info.get("final_goal", GOAL_FEATURE)

        # 1. Update PyBullet 3D GUI trajectory lines
        if self.env.render_mode == p.GUI:
            pc = self.env.physics_client
            cur_ee_pos = self.env._get_camera_pose()[0].tolist()   # trace the camera tip
            if self.prev_ee_pos is not None:
                p.addUserDebugLine(
                    self.prev_ee_pos,
                    cur_ee_pos,
                    self.line_color,
                    lineWidth=3.5,
                    lifeTime=0,
                    physicsClientId=pc,
                )
            self.prev_ee_pos = cur_ee_pos

        # 2. Render HUD frame in OpenCV
        frame = self.draw_hud(rgb, mask, cur_feat, des_feat, goal, step_info)
        self.last_frame = frame

        if self.is_gui and frame is not None:
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(self.step_delay_ms) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q'
                self.user_quit = True

    def show_completion(self, result, hold_seconds=5.0):
        """Display final completion banner and keep GUI open for inspection."""
        if not self.is_gui or self.last_frame is None:
            return

        frame = self.last_frame.copy()
        h, w = frame.shape[:2]

        # Centered celebration / completion overlay
        box_w, box_h = 500, 130
        x0 = (w - box_w) // 2
        y0 = (h - box_h) // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 255, 120), 2)

        init_err = result.get("initial_error", 1573.0)
        final_err = result.get("final_error", 0.0)
        reduction = max(0.0, (1.0 - final_err / max(1.0, init_err)) * 100.0)
        speed = result.get("final_joint_vel_norm", 0.0)

        cv2.putText(frame, "VISUAL SERVOING COMPLETED",
                    (x0 + 45, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 120), 2)
        cv2.putText(frame, f"Initial Error: {init_err:.1f} px -> Final Error: {final_err:.1f} px",
                    (x0 + 30, y0 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(frame, f"Error Reduced: {reduction:.1f}% | Final Speed: {speed:.4f} rad/s",
                    (x0 + 30, y0 + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 240, 255), 1)
        cv2.putText(frame, "Press 'q' or ESC to exit GUI...",
                    (x0 + 130, y0 + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        try:
            cv2.imshow(self.window_name, frame)
            t_start = time.time()
            while time.time() - t_start < float(hold_seconds):
                k = cv2.waitKey(50) & 0xFF
                if k in (27, ord('q'), ord(' ')):
                    break
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


def render_external_view(env, width=480, height=360):
    """Third-person view of the scene from the scene's viewpoint (GUI or DIRECT)."""
    v = env.scene["view"]
    view = p.computeViewMatrixFromYawPitchRoll(v["target"], v["distance"], v["yaw"],
                                               v["pitch"], 0, 2)
    proj = p.computeProjectionMatrixFOV(50, width / height, 0.05, 5.0)
    img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER,
                           physicsClientId=env.physics_client)
    return cv2.cvtColor(np.asarray(img[2], dtype=np.uint8)[:, :, :3], cv2.COLOR_RGB2BGR)


def _camera_inset(env, size=(176, 132)):
    """Eye-in-hand image with crosshair."""
    cam = cv2.resize(cv2.cvtColor(env.get_camera_image()[0], cv2.COLOR_RGB2BGR), size)
    w, h = size
    cv2.line(cam, (w // 2, 0), (w // 2, h), (20, 20, 20), 1)
    cv2.line(cam, (0, h // 2), (w, h // 2), (20, 20, 20), 1)
    cv2.rectangle(cam, (0, 0), (w - 1, h - 1), (90, 90, 90), 2)
    return cam


def _image_plane_inset(planned, executed, regions, size=(176, 132)):
    """Mini x-y image-plane plot: obstacle regions, planned and executed paths."""
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    to_px = lambda xy: np.column_stack(((xy[:, 0] + CX) * w / IMAGE_WIDTH,
                                        (xy[:, 1] + CY) * h / IMAGE_HEIGHT)).astype(np.int32)
    for hull, _ in regions:
        cv2.polylines(img, [to_px(np.asarray(hull))], True, (40, 40, 220), 1)
    if len(executed):
        cv2.polylines(img, [to_px(executed[:, :2])], False, (200, 0, 200), 1)
    cv2.polylines(img, [to_px(planned[:, :2])], False, (200, 60, 0), 1)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (60, 60, 60), 1)
    return img


def motion_sequence_frames(env, executed_joints, n_frames=7, plan=None, before_frame=None):
    """Frames of the executed motion with a camera-view inset.

    ``plan`` = (planned_s, executed_s, obstacle_regions) adds the image-plane
    plot inset to the first and last frames.  before_frame(i) is called before
    frame i is rendered (e.g. to put a moving target where it was at step i).
    """
    joints = np.asarray(executed_joints)
    frames = []
    idx = np.linspace(0, len(joints) - 1, n_frames).astype(int)
    for k, i in enumerate(idx):
        if before_frame is not None:
            before_frame(i)
        env.reset_joint_state(joints[i])
        frame = render_external_view(env)
        cam = _camera_inset(env)
        frame[6:6 + cam.shape[0], -cam.shape[1] - 6:-6] = cam
        if plan is not None and k in (0, n_frames - 1):
            ins = _image_plane_inset(*plan)
            frame[-ins.shape[0] - 6:-6, -ins.shape[1] - 6:-6] = ins
        cv2.putText(frame, f"step {i}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (20, 20, 20), 2)
        frames.append(frame)
    return frames


def save_frame_grid(rows, save_path, labels=None):
    """Stack rows of frames into one image, optionally labelling each row."""
    out = []
    width = max(sum(f.shape[1] for f in frames) for frames in rows)
    for r, frames in enumerate(rows):
        row = np.hstack(frames)
        if row.shape[1] < width:                     # pad shorter rows with white
            row = np.hstack([row, np.full((row.shape[0], width - row.shape[1], 3), 255, np.uint8)])
        if labels:
            bar = np.full((34, row.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, labels[r], (row.shape[1] // 2 - 8 * len(labels[r]), 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            row = np.vstack([row, bar])
        out.append(row)
    cv2.imwrite(str(save_path), np.vstack(out))
    print(f"[Plot] Saved motion sequence to {save_path}")


# --------------------- RRT tree --------------------------------------------

def draw_rrt_tree(ax, points, parents, path=None):
    """Draw an RRT on a matplotlib axis (2-D or 3-D).

    points  : (N, d) array, node i is at points[i] (d = 2 for the image plane,
              d = 3 for the feature space [x, y, area])
    parents : list of N parent indices, parents[i] is None for the root
    path    : list of node indices from the root to the goal node, or None
    """
    # ------------------------------------------------------------------
    # TODO 6 (Task 3): draw the tree and the path.
    #   1. Every edge: for each node i whose parent j = parents[i] is not None,
    #      draw a thin grey line from points[j] to points[i]:
    #        ax.plot(*np.column_stack((points[j], points[i])), color="0.6", lw=0.8)
    #      (np.column_stack gives one row per coordinate, which works for
    #       both 2-D and 3-D axes)
    #   2. The nodes: ax.scatter(*points.T, s=8, color="tab:blue")
    #   3. If path is given (at least 2 nodes): draw it on top as a thick red line,
    #        ax.plot(*points[path].T, color="red", lw=2.5)
    # ------------------------------------------------------------------
    raise NotImplementedError("TODO 6: draw_rrt_tree (visualizer.py)")


def tree_path(tree_f, goal_id):
    """Node indices from the root to goal_id (follow the parent links)."""
    path = []
    while goal_id is not None:
        path.append(goal_id)
        goal_id = tree_f[goal_id].parent_id
    return path[::-1]


class RRTTreeView:
    """Shows the RRT while it grows and saves the final tree figure.

    With the GUI, every new node adds a green line between the camera
    positions of the node and its parent in the PyBullet window, and an
    OpenCV window shows the tree in feature space and in the image plane.
    """

    def __init__(self, env, goal, title, live=True, every=10, regions=(), name="RRT"):
        self.env = env
        self.name = name
        self.goal = np.asarray(goal, dtype=np.float64)
        self.title = title
        self.live = live and env.render_mode == p.GUI
        self.every = every
        self.regions = regions
        self.window = f"RRT tree - {title}"
        self._cam = {}

    def _camera_position(self, tree_c, nid):
        if nid not in self._cam:
            self.env.reset_joint_state(tree_c[nid].q)
            self._cam[nid] = self.env._get_camera_pose()[0].tolist()
        return self._cam[nid]

    def on_new_node(self, tree_f, tree_c, nid):
        """Planner callback: node ``nid`` was just added to the trees."""
        if not self.live:
            return
        p.addUserDebugLine(self._camera_position(tree_c, tree_c[nid].parent_id),
                           self._camera_position(tree_c, nid), [0.1, 0.65, 0.2],
                           lineWidth=1.5, physicsClientId=self.env.physics_client)
        step = self.every if nid < 10 * self.every else 5 * self.every   # redraw less often later
        if nid % step == 0:
            self._show(self.figure(tree_f))

    def figure(self, tree_f, path=None):
        pts = np.array([n.s for n in tree_f])
        parents = [n.parent_id for n in tree_f]
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        draw_rrt_tree(ax, pts, parents, path)
        ax.scatter(*pts[0], color="k", s=60, marker="s", label="start")
        ax.scatter(*self.goal, color="g", s=120, marker="*", label="goal")
        ax.set(xlabel="x (px)", ylabel="y (px)", zlabel="area (px)",
               title=f"{self.name} in feature space ({len(pts)} nodes)")
        ax.legend(loc="upper left", fontsize=8)

        ax = fig.add_subplot(1, 2, 2)
        for hull, _ in self.regions:
            ax.add_patch(Polygon(hull, closed=True, fc=(0.9, 0.1, 0.1, 0.15), ec="r"))
        draw_rrt_tree(ax, pts[:, :2], parents, path)
        ax.plot(*pts[0, :2], "ks", ms=8)
        ax.plot(*self.goal[:2], "g*", ms=14)
        ax.set(xlim=(FOV_X_MIN, FOV_X_MAX), ylim=(FOV_Y_MAX, FOV_Y_MIN),
               xlabel="x (px)", ylabel="y (px)", title=f"{self.name} in the image plane")
        ax.set_aspect("equal")
        ax.grid(True, ls="--", alpha=0.5)
        fig.suptitle(self.title)
        fig.tight_layout()
        return fig

    def _show(self, fig):
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        cv2.imshow(self.window, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def finish(self, tree_f, tree_c, goal_id, save_path):
        """Redraw the final tree (RRT* rewires edges while it grows), highlight
        the path from start to goal and save the tree figure."""
        path = tree_path(tree_f, goal_id)
        if self.live:
            pc = self.env.physics_client
            p.removeAllUserDebugItems(physicsClientId=pc)
            for nid in range(1, len(tree_f)):
                p.addUserDebugLine(self._camera_position(tree_c, tree_c[nid].parent_id),
                                   self._camera_position(tree_c, nid), [0.1, 0.65, 0.2],
                                   lineWidth=1.0, physicsClientId=pc)
            for a, b in zip(path[:-1], path[1:]):
                p.addUserDebugLine(self._camera_position(tree_c, a), self._camera_position(tree_c, b),
                                   [0.9, 0.1, 0.1], lineWidth=4,
                                   physicsClientId=self.env.physics_client)
        fig = self.figure(tree_f, path)
        fig.savefig(save_path, dpi=150)
        print(f"[Plot] Saved RRT tree to {save_path}")
        if self.live:
            self._show(fig)
        else:
            plt.close(fig)
