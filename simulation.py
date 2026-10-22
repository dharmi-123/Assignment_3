"""
PyBullet environment and camera feature extraction for UR5 eye-in-hand
visual servoing.

Two scenes are available (config.WORKBENCH_SCENE for Tasks 1-2 and
config.OBSTACLE_SCENE for Tasks 3-4).  Both use primitive collision geometry and
the six-joint UR5 URDF in models/.
"""

import os

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from config import (
    FOCAL_LENGTH_X, FOCAL_LENGTH_Y, CX, CY,
    IMAGE_HEIGHT, IMAGE_WIDTH,
    OBSTACLE_COLOR_RGB, OBSTACLE_SAFETY_MARGIN, WORKBENCH_SCENE,
)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def _rot_to_quat(R):
    """3x3 rotation matrix -> PyBullet (x, y, z, w) quaternion."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x = 0.25 * S, (R[2, 1] - R[1, 2]) / S
        y, z = (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / S, 0.25 * S
        y, z = (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S
        y, z = 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S
        y, z = (R[1, 2] + R[2, 1]) / S, 0.25 * S
    return [x, y, z, w]


# --------------------- Camera Feature Extractor --------------------------

class CameraFeatureExtractor:
    """Extracts s = [xc, yc, area]T from an RGB image via HSV segmentation."""

    def __init__(self, hsv_lower=(35, 50, 50), hsv_upper=(85, 255, 255)):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)

    def extract_features(self, rgb_image):
        """Return (feature_vector, mask) or (None, mask) if target not found."""
        if rgb_image is None or rgb_image.size == 0:
            return None, None
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        moments = cv2.moments(mask, binaryImage=True)
        m00 = moments['m00']
        if m00 < 10.0:
            return None, mask
        u = moments['m10'] / m00
        v = moments['m01'] / m00
        h, w = mask.shape[:2]
        xc = u - w / 2.0
        yc = v - h / 2.0
        return np.array([xc, yc, m00], dtype=np.float64), mask


# --------------------- PyBullet UR5 Environment --------------------------

class PyBulletUR5Env:
    """UR5, table, target, obstacles and a rigid eye-in-hand camera.

    ``scene`` is a scene dict from config (table type, robot base, camera
    mount, start configuration q0, target); ``obstacles`` is a list of
    (centre, half_extents) boxes.
    """

    def __init__(self, render_mode=p.DIRECT, scene=WORKBENCH_SCENE, obstacles=()):
        self.render_mode = render_mode
        self.scene = scene
        self.obstacles = list(obstacles)
        self.q0 = np.asarray(scene["q0"], dtype=np.float64)
        self.camera_offset_pos = np.asarray(scene["camera_pos"], dtype=np.float64)
        self.camera_offset_orn = np.asarray(scene["camera_orn"], dtype=np.float64)
        self.target_size = np.asarray(scene["target_size"], dtype=np.float64)
        self.target_position = tuple(scene["target_pos"])
        self.dataset_q_center = np.asarray(scene["dataset_q_center"], dtype=np.float64)
        self.dataset_joint_std = np.asarray(scene["dataset_joint_std"], dtype=np.float64)
        self.physics_client = None
        self.robot_id = self.table_id = self.target_id = self.plane_id = None
        self.table_leg_ids = []
        self.obstacle_ids = []
        self.camera_body_ids = []
        self.ee_link_index = 5
        self.num_joints = 6
        self.revolute_joint_indices = []
        self.feature_extractor = CameraFeatureExtractor(*scene["target_hsv"])

        self.fx, self.fy = float(FOCAL_LENGTH_X), float(FOCAL_LENGTH_Y)
        self.cx, self.cy = float(CX), float(CY)
        self.near_val, self.far_val = 0.01, 5.0

        fov_y = 2.0 * np.arctan((IMAGE_HEIGHT / 2.0) / self.fy) * 180.0 / np.pi
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov_y, IMAGE_WIDTH / IMAGE_HEIGHT, self.near_val, self.far_val,
        )
        self._camera_mount_rotation = np.asarray(
            p.getMatrixFromQuaternion(self.camera_offset_orn), dtype=np.float64,
        ).reshape(3, 3)
        self._setup_simulation()

    # -- Scene construction ------------------------------------------------

    def _setup_simulation(self):
        if "XAUTHORITY" not in os.environ:
            for auth_path in ["/run/user/1000/gdm/Xauthority", os.path.expanduser("~/.Xauthority")]:
                if os.path.isfile(auth_path):
                    os.environ["XAUTHORITY"] = auth_path
                    break
        if "DISPLAY" not in os.environ:
            os.environ["DISPLAY"] = ":1"

        if self.physics_client is None or not p.isConnected(self.physics_client):
            self.physics_client = p.connect(self.render_mode)
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.physics_client)
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)

        self._create_floor()
        self._create_table()
        self.robot_id = self._create_ur5()
        self._discover_joints()
        self._create_camera_body()
        self.reset_joint_state(self.q0)
        self._create_target()
        self._load_obstacles()
        p.performCollisionDetection(physicsClientId=self.physics_client)

    def _create_floor(self):
        pc = self.physics_client
        col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=pc)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[4.0, 4.0, 0.001],
                                  rgbaColor=[0.86, 0.87, 0.88, 1.0], physicsClientId=pc)
        self.plane_id = p.createMultiBody(0, col, vis, [0, 0, -0.001], physicsClientId=pc)

    def _create_table(self):
        """Table top at the robot base height, with four legs."""
        pc = self.physics_client
        top_z = self.scene["robot_base"][2]
        (cx, cy), (hx, hy) = self.scene["table"]["center"], self.scene["table"]["half"]
        top_he = [hx, hy, 0.02]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=top_he, physicsClientId=pc)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=top_he,
                                  rgbaColor=[0.68, 0.70, 0.74, 1.0], physicsClientId=pc)
        self.table_id = p.createMultiBody(0, col, vis, [cx, cy, top_z - top_he[2]],
                                          physicsClientId=pc)

        leg_he = [0.035, 0.035, (top_z - 0.04) / 2.0]
        self.table_leg_ids = []
        for x in (cx - hx + 0.10, cx + hx - 0.10):
            for y in (cy - hy + 0.08, cy + hy - 0.08):
                c = p.createCollisionShape(p.GEOM_BOX, halfExtents=leg_he, physicsClientId=pc)
                v = p.createVisualShape(p.GEOM_BOX, halfExtents=leg_he,
                                        rgbaColor=[0.32, 0.34, 0.38, 1.0], physicsClientId=pc)
                self.table_leg_ids.append(
                    p.createMultiBody(0, c, v, [x, y, leg_he[2]], physicsClientId=pc))

    def _create_ur5(self):
        urdf = os.path.join(PROJECT_ROOT, "models", "ur5.urdf")
        if not os.path.isfile(urdf):
            raise FileNotFoundError(f"UR5 URDF not found: {urdf}")
        robot = p.loadURDF(urdf, self.scene["robot_base"], [0, 0, 0, 1],
                           useFixedBase=True,
                           flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                           physicsClientId=self.physics_client)
        if self.scene["robot_rgba"] is not None:
            for link in range(-1, p.getNumJoints(robot, physicsClientId=self.physics_client)):
                p.changeVisualShape(robot, link, rgbaColor=list(self.scene["robot_rgba"]),
                                    physicsClientId=self.physics_client)
        return robot

    def _discover_joints(self):
        pc = self.physics_client
        self.revolute_joint_indices = []
        ee_map = {}
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=pc)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=pc)
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.revolute_joint_indices.append(i)
            ee_map[info[12].decode("utf-8")] = i
        for name in ("ee_link", "tool0", "wrist_3_link", "ee_tip"):
            if name in ee_map:
                self.ee_link_index = ee_map[name]
                break
        else:
            self.ee_link_index = p.getNumJoints(self.robot_id, physicsClientId=pc) - 1
        self.num_joints = len(self.revolute_joint_indices)
        if self.num_joints != 6:
            raise RuntimeError(f"Expected 6 UR5 joints, found {self.num_joints}")

    def _create_target(self):
        pc = self.physics_client
        he = (self.target_size / 2.0).tolist()
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he, physicsClientId=pc)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he,
                                  rgbaColor=list(self.scene["target_rgba"]), physicsClientId=pc)
        self.target_id = p.createMultiBody(0, col, vis, self.target_position,
                                           physicsClientId=pc)

    def set_target_position(self, position):
        """Move the target (used for the moving-target task)."""
        self.target_position = tuple(float(v) for v in position)
        p.resetBasePositionAndOrientation(self.target_id, list(self.target_position), [0, 0, 0, 1],
                                          physicsClientId=self.physics_client)

    def _load_obstacles(self):
        pc = self.physics_client
        self.obstacle_ids = []
        for centre, he in self.obstacles:
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(he), physicsClientId=pc)
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=list(he),
                                    rgbaColor=OBSTACLE_COLOR_RGB, physicsClientId=pc)
            self.obstacle_ids.append(
                p.createMultiBody(0, c, v, list(centre), physicsClientId=pc))

    def _create_camera_body(self):
        """Rigid camera rod (ee_link -> optical centre) plus camera housing.

        Both parts are kinematic bodies re-posed with the arm, so they collide
        with obstacles like real hardware.  The housing sits behind the lens
        and is never seen in the eye-in-hand image.
        """
        pc = self.physics_client
        self._boom_radius = 0.012
        self._boom_length = float(np.linalg.norm(self.camera_offset_pos))
        self._housing_he = [0.025, 0.02, 0.02]
        boom_c = p.createCollisionShape(p.GEOM_CYLINDER, radius=self._boom_radius,
                                        height=self._boom_length, physicsClientId=pc)
        boom_v = p.createVisualShape(p.GEOM_CYLINDER, radius=self._boom_radius,
                                     length=self._boom_length,
                                     rgbaColor=[0.12, 0.12, 0.14, 1.0], physicsClientId=pc)
        house_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=self._housing_he,
                                         physicsClientId=pc)
        house_v = p.createVisualShape(p.GEOM_BOX, halfExtents=self._housing_he,
                                      rgbaColor=[0.05, 0.25, 0.75, 1.0], physicsClientId=pc)
        self.camera_body_ids = [
            p.createMultiBody(0, boom_c, boom_v, [0, 0, 0], physicsClientId=pc),
            p.createMultiBody(0, house_c, house_v, [0, 0, 0], physicsClientId=pc),
        ]
        for bid in self.camera_body_ids:
            for lid in range(-1, p.getNumJoints(self.robot_id, physicsClientId=pc)):
                p.setCollisionFilterPair(bid, self.robot_id, -1, lid, 0, physicsClientId=pc)

    def _camera_body_poses(self):
        """World poses of the boom and housing for the current joint state."""
        ee = p.getLinkState(self.robot_id, self.ee_link_index,
                            computeForwardKinematics=True,
                            physicsClientId=self.physics_client)
        ee_pos = np.asarray(ee[4], dtype=np.float64)
        cam_pos, cam_rot = self._get_camera_pose()
        axis = cam_pos - ee_pos
        axis /= max(np.linalg.norm(axis), 1e-9)
        # Cylinder z-axis -> boom axis
        z = np.array([0.0, 0.0, 1.0])
        c = float(np.clip(z @ axis, -1.0, 1.0))
        rot_axis = np.cross(z, axis)
        if np.linalg.norm(rot_axis) < 1e-9:
            boom_q = [0, 0, 0, 1] if c > 0 else [1, 0, 0, 0]
        else:
            boom_q = p.getQuaternionFromAxisAngle(
                (rot_axis / np.linalg.norm(rot_axis)).tolist(), float(np.arccos(c)))
        boom_pos = (ee_pos + cam_pos) / 2.0 - axis * (self._housing_he[2] + 0.005) / 2.0
        house_pos = cam_pos - cam_rot[:, 2] * (self._housing_he[2] + 0.006)
        house_q = _rot_to_quat(cam_rot)
        return [(boom_pos, boom_q), (house_pos, house_q)]

    def _sync_camera_body(self):
        if not self.camera_body_ids:
            return
        for bid, (pos, orn) in zip(self.camera_body_ids, self._camera_body_poses()):
            p.resetBasePositionAndOrientation(bid, list(pos), list(orn),
                                              physicsClientId=self.physics_client)

    # -- Joint state and camera --------------------------------------------

    def reset_joint_state(self, q):
        q = np.asarray(q, dtype=float).ravel()
        for val, idx in zip(q, self.revolute_joint_indices):
            p.resetJointState(self.robot_id, idx, float(val),
                              physicsClientId=self.physics_client)
        self._sync_camera_body()
        p.performCollisionDetection(physicsClientId=self.physics_client)

    def get_joint_state(self):
        states = p.getJointStates(self.robot_id, self.revolute_joint_indices,
                                  physicsClientId=self.physics_client)
        return (np.array([s[0] for s in states], dtype=np.float64),
                np.array([s[1] for s in states], dtype=np.float64))

    def _get_camera_pose(self):
        ee = p.getLinkState(self.robot_id, self.ee_link_index,
                            computeForwardKinematics=True,
                            physicsClientId=self.physics_client)
        ee_pos = np.asarray(ee[4], dtype=np.float64)
        ee_rot = np.asarray(p.getMatrixFromQuaternion(ee[5]),
                            dtype=np.float64).reshape(3, 3)
        cam_pos = ee_pos + ee_rot @ self.camera_offset_pos
        cam_rot = ee_rot @ self._camera_mount_rotation
        return cam_pos, cam_rot

    def project_obstacles(self, margin_px=0.0):
        """Project each obstacle into the image as (hull_xy, min_depth).

        hull_xy are centred pixel coordinates (same convention as s), grown
        outward by ``margin_px``.  Obstacles behind the camera are skipped.
        """
        cam_pos, cam_rot = self._get_camera_pose()
        regions = []
        for oid in self.obstacle_ids:
            pos, orn = p.getBasePositionAndOrientation(oid, physicsClientId=self.physics_client)
            he = p.getCollisionShapeData(oid, -1, physicsClientId=self.physics_client)[0][3]
            he = np.asarray(he, dtype=np.float64) / 2.0
            R = np.asarray(p.getMatrixFromQuaternion(orn), dtype=np.float64).reshape(3, 3)
            corners = np.array([cam_rot.T @ (np.asarray(pos) + R @ np.array([dx, dy, dz]) - cam_pos)
                                for dx in (-he[0], he[0]) for dy in (-he[1], he[1])
                                for dz in (-he[2], he[2])])
            # Visible part of the box: corners in front of the camera plus the
            # points where box edges cross the clipping plane z = clip
            clip = 0.02
            front = list(corners[corners[:, 2] > clip])
            for i in range(8):
                for j in range(i + 1, 8):
                    a, b = corners[i], corners[j]
                    if bin(i ^ j).count("1") == 1 and (a[2] - clip) * (b[2] - clip) < 0:
                        front.append(a + (b - a) * (clip - a[2]) / (b[2] - a[2]))
            pts = [[self.fx * c[0] / c[2], self.fy * c[1] / c[2]] for c in front]
            depths = [c[2] for c in front]
            if len(pts) < 3:
                continue
            hull = cv2.convexHull(np.asarray(pts, dtype=np.float32)).reshape(-1, 2)
            if margin_px > 0:
                ctr = hull.mean(axis=0)
                d = hull - ctr
                n = np.linalg.norm(d, axis=1, keepdims=True)
                hull = hull + d / np.maximum(n, 1e-6) * margin_px
            regions.append((hull.astype(np.float64), float(min(depths))))
        return regions

    def get_camera_image(self, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
        """Render RGB and metric depth from the eye-in-hand camera."""
        cam_pos, cam_rot = self._get_camera_pose()
        cam_fwd = cam_rot[:, 2]
        cam_up = -cam_rot[:, 1]
        view = p.computeViewMatrix(cam_pos, cam_pos + cam_fwd, cam_up)
        img = p.getCameraImage(int(width), int(height), view, self.proj_matrix,
                               renderer=p.ER_TINY_RENDERER,
                               physicsClientId=self.physics_client)
        rgb = np.asarray(img[2], dtype=np.uint8)[:, :, :3]
        dbuf = np.asarray(img[3], dtype=np.float64)
        depth = (self.far_val * self.near_val) / (
            self.far_val - (self.far_val - self.near_val) * dbuf)
        return rgb, depth, cam_pos

    def get_current_feature(self, return_mask=False):
        """Extract [xc, yc, area] from the rendered camera image (feedback)."""
        rgb, depth_img, _ = self.get_camera_image()
        feat, mask = self.feature_extractor.extract_features(rgb)
        if feat is None:
            if return_mask:
                return None, 0.5, rgb, mask
            return None, 0.5, rgb
        td = depth_img[mask > 0]
        td = td[np.isfinite(td)]
        z = float(np.median(td)) if td.size else 0.5
        if return_mask:
            return feat, z, rgb, mask
        return feat, z, rgb

    def get_dataset_feature(self, width=160, height=120):
        """Low-res rendered feature for fast dataset collection."""
        rgb, depth_img, _ = self.get_camera_image(width, height)
        feat, mask = self.feature_extractor.extract_features(rgb)
        if feat is None:
            return None, 0.5, rgb
        sx, sy = IMAGE_WIDTH / float(width), IMAGE_HEIGHT / float(height)
        feat = np.asarray(feat, dtype=np.float64)
        feat[0] *= sx;  feat[1] *= sy;  feat[2] *= sx * sy
        td = depth_img[mask > 0]
        td = td[np.isfinite(td)]
        z = float(np.median(td)) if td.size else 0.5
        return feat, z, rgb

    def get_analytic_feature(self):
        """Fast analytic projection model for planning diagnostics."""
        cam_pos, cam_rot = self._get_camera_pose()
        tgt_pos, tgt_orn = p.getBasePositionAndOrientation(
            self.target_id, physicsClientId=self.physics_client)
        tgt_pos = np.asarray(tgt_pos, dtype=np.float64)
        tc = cam_rot.T @ (tgt_pos - cam_pos)
        if tc[2] <= self.near_val:
            return None, float(tc[2]), None
        xc = self.fx * tc[0] / tc[2]
        yc = self.fy * tc[1] / tc[2]
        # Project box corners for convex-hull area
        half = self.target_size / 2.0
        obj_rot = np.asarray(p.getMatrixFromQuaternion(tgt_orn),
                             dtype=np.float64).reshape(3, 3)
        pts = []
        for dx in (-half[0], half[0]):
            for dy in (-half[1], half[1]):
                for dz in (-half[2], half[2]):
                    cw = tgt_pos + obj_rot @ np.array([dx, dy, dz])
                    cc = cam_rot.T @ (cw - cam_pos)
                    if cc[2] > self.near_val:
                        pts.append([self.fx * cc[0] / cc[2],
                                    self.fy * cc[1] / cc[2]])
        if len(pts) < 3:
            return None, float(tc[2]), None
        hull = cv2.convexHull(np.asarray(pts, dtype=np.float32))
        area = float(cv2.contourArea(hull))
        return np.array([xc, yc, area], dtype=np.float64), float(tc[2]), None

    # -- Collision checking ------------------------------------------------

    def _robot_bodies(self):
        """Bodies that move with the arm: the UR5 plus camera boom/housing."""
        return [self.robot_id] + list(self.camera_body_ids)

    def _environment_bodies(self):
        return ([(self.table_id, "table")]
                + [(lid, "table_leg") for lid in self.table_leg_ids]
                + [(self.plane_id, "plane")]
                + [(oid, "obstacle") for oid in self.obstacle_ids])

    def check_collision(self, q=None, include_self=True, return_details=False,
                        obstacle_margin=None):
        """Check collisions without leaving the robot in a changed state.

        Obstacles are inflated by ``obstacle_margin`` (default
        OBSTACLE_SAFETY_MARGIN): any robot/camera part closer than that to an
        obstacle counts as a collision (safety offset around obstacles).
        """
        margin = OBSTACLE_SAFETY_MARGIN if obstacle_margin is None else float(obstacle_margin)
        orig = None
        if q is not None:
            orig, _ = self.get_joint_state()
            self.reset_joint_state(q)
        details = []
        try:
            pc = self.physics_client
            p.performCollisionDetection(physicsClientId=pc)
            for bid, label in self._environment_bodies():
                dist = margin if label == "obstacle" else 0.0
                for rid in self._robot_bodies():
                    for c in p.getClosestPoints(rid, bid, dist, physicsClientId=pc):
                        if c[8] > dist:
                            continue
                        rl = int(c[3]) if rid == self.robot_id else "camera"
                        if rl == -1:
                            continue          # fixed robot base resting on its support
                        details.append({"kind": label, "robot_link": rl,
                                        "distance": float(c[8])})
            if include_self:
                for c in p.getContactPoints(self.robot_id, self.robot_id, physicsClientId=pc):
                    details.append({"kind": "self",
                                    "robot_link": (int(c[3]), int(c[4])),
                                    "distance": float(c[8])})
        finally:
            if orig is not None:
                self.reset_joint_state(orig)
        return (bool(details), details) if return_details else bool(details)

    def min_obstacle_distance(self):
        """Smallest robot/camera-to-obstacle distance (inf if no obstacles)."""
        pc = self.physics_client
        d = np.inf
        for oid in self.obstacle_ids:
            for rid in self._robot_bodies():
                for c in p.getClosestPoints(rid, oid, 1.0, physicsClientId=pc):
                    d = min(d, float(c[8]))
        return d

    def close(self):
        if self.physics_client is not None and p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
            self.physics_client = None
