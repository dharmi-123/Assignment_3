"""
Image-space RRT and RRT* planners and the workspace dataset generator
(Tasks 3-4).

The planners grow two coupled trees: one of image features s = [xc, yc, a]
and one of robot configurations q.  Each feature node has the configuration
that produces it, so the planned feature path is known to be feasible.

Provided:
  - WorkspaceDatasetGenerator   : random workspace images used as RRT samples
  - ImageBasedRRTPlanner.plan      : the RRT loop               (Task 3)
  - ImageBasedRRTPlanner.plan_star : the RRT* loop              (Task 4)
  - constraint checks (field of view, occlusion, collision), tree bookkeeping

Student TODOs (methods of ImageBasedRRTPlanner):
  - _extend         : TODO 5  one RRT extension step                 Task 3
  - _choose_parent  : TODO 7  cheapest parent among the neighbours   Task 4
  - _rewire         : TODO 8  re-route neighbours through a new node Task 4
"""

import json
import os
import warnings

import cv2
import numpy as np

from config import (
    CAMERA_OFFSET_POS,
    FOV_X_MIN, FOV_X_MAX, FOV_Y_MIN, FOV_Y_MAX,
    JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER, JOINT_VELOCITY_LIMITS,
    OBSTACLE_IMAGE_MARGIN_PX, DATASET_RANDOM_SEED,
    RRT_DT, RRT_GOAL_BIAS_K, RRT_LAMBDA, RRT_MAX_WAYPOINT_DISTANCE,
    RRT_MAX_ITER, RRT_RADIUS_R, RRT_RANDOM_SEED,
    RRT_STAR_ITER, RRT_STAR_MAX_JOINT_STEP, RRT_STAR_NEAR_RADIUS,
)
from kinematics import InteractionMatrix, TaskPrioritySolver, secondary_task_jacobian


def _in_fov(s, min_area):
    return FOV_X_MIN <= s[0] <= FOV_X_MAX and FOV_Y_MIN <= s[1] <= FOV_Y_MAX and s[2] >= min_area


# --------------------- Workspace dataset ---------------------------------

class WorkspaceDatasetGenerator:
    """Random robot configurations around the scene's working posture and the
    target feature seen from each of them."""

    def __init__(self, env):
        self.env = env

    def expected_metadata(self):
        """Camera mount and target size the dataset was recorded with."""
        return {"camera": [float(v) for v in np.concatenate((self.env.camera_offset_pos,
                                                             self.env.camera_offset_orn))],
                "target_size_m": [float(v) for v in self.env.target_size]}

    def is_compatible(self, dataset, expected_n=None):
        s = np.asarray(dataset.get("s_workspace", []), dtype=np.float64)
        q = np.asarray(dataset.get("q_workspace", []), dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != 3 or q.shape != (len(s), 6) or len(s) == 0:
            return False
        if expected_n is not None and len(s) != int(expected_n):
            return False
        meta, ref = dataset.get("metadata"), self.expected_metadata()
        return (meta is not None and meta.keys() == ref.keys()
                and all(np.allclose(meta[k], ref[k], atol=1e-6) for k in ref))

    @staticmethod
    def save(dataset, path):
        """Save a dataset as .npz (metadata stored as a JSON string)."""
        os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
        np.savez_compressed(path, q_workspace=dataset["q_workspace"],
                            s_workspace=dataset["s_workspace"],
                            z_workspace=dataset["z_workspace"],
                            metadata=json.dumps(dataset["metadata"]))

    @staticmethod
    def load(path):
        with np.load(path, allow_pickle=False) as d:
            return {"q_workspace": d["q_workspace"], "s_workspace": d["s_workspace"],
                    "z_workspace": d["z_workspace"],
                    "metadata": json.loads(str(d["metadata"]))}

    def generate(self, num_samples=2000, save_path=None, seed=DATASET_RANDOM_SEED,
                 max_attempts=None):
        """Sample configurations, keep those where the target is in view."""
        print(f"[Dataset] Generating {num_samples} workspace samples...")
        rng = np.random.default_rng(seed)
        q_centre = np.asarray(self.env.dataset_q_center, dtype=np.float64)
        joint_std = np.asarray(self.env.dataset_joint_std, dtype=np.float64)
        ds_q, ds_s, ds_z = [], [], []
        attempts, max_attempts = 0, max_attempts or num_samples * 20

        while len(ds_s) < num_samples and attempts < max_attempts:
            attempts += 1
            q = np.clip(q_centre + rng.normal(0.0, joint_std),
                        JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
            if self.env.check_collision(q):
                continue
            self.env.reset_joint_state(q)
            pred, pred_z, _ = self.env.get_analytic_feature()      # cheap pre-filter
            if pred is None or pred_z <= 0 or not _in_fov(pred, 50):
                continue
            feat, z, _ = self.env.get_dataset_feature()             # rendered feature
            if feat is not None and z > 0 and _in_fov(feat, 50):
                ds_q.append(q)
                ds_s.append(np.asarray(feat, dtype=np.float64))
                ds_z.append(float(z))
                if len(ds_s) % 500 == 0 and len(ds_s) != num_samples:
                    print(f"[Dataset] Collected {len(ds_s)} / {num_samples}")

        if len(ds_s) != num_samples:
            raise RuntimeError(f"Collected only {len(ds_s)}/{num_samples} after {attempts} attempts.")
        dataset = {"q_workspace": np.array(ds_q), "s_workspace": np.array(ds_s),
                   "z_workspace": np.array(ds_z), "metadata": self.expected_metadata()}
        if save_path:
            self.save(dataset, save_path)
            print(f"[Dataset] Saved {num_samples} samples to {save_path}")
        return dataset


# --------------------- RRT nodes -----------------------------------------

class _FeatureNode:
    __slots__ = ("s", "s_dot", "node_id", "parent_id")

    def __init__(self, s, s_dot, node_id, parent_id=None):
        self.s = np.asarray(s, dtype=np.float64).copy()
        self.s_dot = np.asarray(s_dot, dtype=np.float64).copy()
        self.node_id = int(node_id)
        self.parent_id = parent_id


class _ConfigNode:
    __slots__ = ("q", "q_dot", "node_id", "parent_id")

    def __init__(self, q, q_dot, node_id, parent_id=None):
        self.q = np.asarray(q, dtype=np.float64).copy()
        self.q_dot = np.asarray(q_dot, dtype=np.float64).copy()
        self.node_id = int(node_id)
        self.parent_id = parent_id


# --------------------- Image-space RRT / RRT* ----------------------------

class ImageBasedRRTPlanner:
    """RRT and RRT* with coupled feature and configuration trees."""

    def __init__(self, env, kinematics, dataset, secondary_task="fixed_orientation",
                 random_seed=RRT_RANDOM_SEED):
        self.env = env
        self.kin = kinematics
        self.dataset_s = np.asarray(dataset["s_workspace"], dtype=np.float64)
        if len(self.dataset_s) == 0:
            raise ValueError("Empty dataset.")
        self.secondary = secondary_task         # keeps the camera orientation fixed
        self.rng = np.random.default_rng(random_seed)
        self.Lm = InteractionMatrix()
        self.tp = TaskPrioritySolver()
        self.has_obstacles = bool(getattr(env, "obstacle_ids", None))
        self.last_plan_result = None

    # -- constraint checks -------------------------------------------------

    def _collision(self, q):
        return bool(self.env.check_collision(q))

    @staticmethod
    def _valid_feature(s):
        """Field of view: centroid inside the image and area > 10 px."""
        s = np.asarray(s, dtype=np.float64)
        return s.shape == (3,) and bool(np.all(np.isfinite(s))) and _in_fov(s, 10)

    def _occluded(self, s, z):
        """True if the target (feature s, depth z) is hidden behind an obstacle:
        its centroid lies in an obstacle's image region (grown by a safety
        offset) and that obstacle is nearer to the camera than the target."""
        if not self.has_obstacles:
            return False
        pt = (float(s[0]), float(s[1]))
        for hull, depth in self.env.project_obstacles(OBSTACLE_IMAGE_MARGIN_PX):
            if depth < z and cv2.pointPolygonTest(
                    hull.astype(np.float32).reshape(-1, 1, 2), pt, False) >= 0:
                return True
        return False

    def _check(self, s, q, z=None):
        """Constraints of a node: field of view, joint limits, target not
        hidden by an obstacle, no collision."""
        if not self._valid_feature(s):
            return False
        q = np.asarray(q)
        if np.any(q < JOINT_LIMITS_LOWER) or np.any(q > JOINT_LIMITS_UPPER):
            return False
        if z is not None:
            self.env.reset_joint_state(q)
            if self._occluded(s, z):
                return False
        return not self._collision(q)

    def _edge_free(self, qa, qb, max_step=0.05):
        """True if every configuration on the straight joint-space edge
        qa -> qb keeps the target in view and unhidden and does not collide."""
        if not self.has_obstacles:
            return True
        n = int(np.ceil(np.max(np.abs(qb - qa)) / max_step))
        for i in range(1, n):
            q = qa + (qb - qa) * i / n
            s, z = self._feature_depth(q)
            if (s is None or not self._valid_feature(s) or self._occluded(s, z)
                    or self._collision(q)):
                return False
        return True

    def _connectable(self, qa, qb):
        """RRT*: can two existing nodes be joined by a direct edge?"""
        return (float(np.max(np.abs(qb - qa))) <= RRT_STAR_MAX_JOINT_STEP
                and self._edge_free(qa, qb))

    # -- helpers -------------------------------------------------------------

    def _feature_depth(self, q):
        """Camera model: feature s and target depth z at configuration q."""
        self.env.reset_joint_state(q)
        return self.env.get_analytic_feature()[:2]

    def _camera_jacobian(self, q):
        return self.kin.compute_camera_jacobian(
            q, physics_client=self.env.physics_client,
            camera_offset_pos=getattr(self.env, "camera_offset_pos", CAMERA_OFFSET_POS),
            camera_mount_rotation=getattr(self.env, "_camera_mount_rotation", None))

    def _secondary_jacobian(self, Jc):
        return None if not self.secondary else secondary_task_jacobian(self.secondary, Jc)

    @staticmethod
    def _limit_vel(q_dot):
        """Scale q_dot down uniformly so no joint exceeds its velocity limit."""
        ratio = float(np.max(np.abs(q_dot) / JOINT_VELOCITY_LIMITS))
        return q_dot / ratio if ratio > 1 else q_dot

    def _sample(self, count, s_desired, k_bias=RRT_GOAL_BIAS_K):
        """Random feature s_r: every k-th iteration the goal, otherwise a
        random workspace feature from the dataset."""
        if count % k_bias == 0:
            return np.asarray(s_desired, dtype=np.float64).copy()
        return self.dataset_s[int(self.rng.integers(len(self.dataset_s)))].copy()

    @staticmethod
    def _nearest(tree_s, s_rand):
        """Index of the tree feature nearest to s_rand (Euclidean distance)."""
        return int(np.argmin(np.linalg.norm(np.asarray(tree_s) - s_rand, axis=1)))

    @staticmethod
    def _near(tree_s, s_new, radius=RRT_STAR_NEAR_RADIUS, max_count=12):
        """RRT*: indices of the (at most max_count) nodes within radius of s_new."""
        d = np.linalg.norm(np.asarray(tree_s) - s_new, axis=1)
        idx = np.flatnonzero(d <= radius)
        return [int(i) for i in idx[np.argsort(d[idx])][:max_count]]

    @staticmethod
    def _update_costs(node_id, delta, tree_f, cost):
        """RRT*: add ``delta`` to the cost of node_id and all its descendants."""
        stack = [node_id]
        while stack:
            i = stack.pop()
            cost[i] += delta
            stack.extend(j for j, n in enumerate(tree_f) if n.parent_id == i)

    @staticmethod
    def _densify(features, joints, max_step):
        """Resample the path so consecutive features are <= max_step px apart."""
        df, dj = [features[0].copy()], [joints[0].copy()]
        for sf, se, jf, je in zip(features[:-1], features[1:], joints[:-1], joints[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(se - sf) / max_step)))
            for i in range(1, n + 1):
                a = i / n
                df.append((1 - a) * sf + a * se)
                dj.append((1 - a) * jf + a * je)
        return df, dj

    # -- student TODO: Task 3 ----------------------------------------------

    def _extend(self, s_parent, q_parent, z_parent, s_rand, dt=RRT_DT):
        """One extension of the trees from the parent node towards s_rand.

        Returns (q_new, q_dot, J1): new configuration, joint velocity used and
        the image Jacobian at the parent.
        """
        Jc = self._camera_jacobian(q_parent)     # 6x6 camera Jacobian at q_parent
        J2 = self._secondary_jacobian(Jc)       # keeps the camera orientation fixed
        # ------------------------------------------------------------------
        # TODO 5 (Task 3): extend the trees one step towards s_rand.
        #   1. Feature velocity towards the sample:
        #        s_dot = RRT_LAMBDA * (s_rand - s_parent)
        #   2. Image Jacobian: Lm = self.Lm.compute(s_parent, z_parent)   (your TODO 1),
        #        J1 = self.Lm.image_jacobian(Lm, Jc)
        #   3. Joint velocity with the camera orientation held fixed:
        #        q_dot = self.tp.solve(J1, s_dot, J2), then q_dot = self._limit_vel(q_dot)
        #   4. Integrate: q_new = q_parent + q_dot * dt
        # return q_new, q_dot, J1
        # ------------------------------------------------------------------
        raise NotImplementedError("TODO 5: ImageBasedRRTPlanner._extend (planner.py)")

    # -- student TODOs: Task 4 (RRT*) --------------------------------------

    def _choose_parent(self, near_ids, s_new, q_new, parent_id, tree_f, tree_c, cost):
        """Cheapest valid parent for the new node.

        near_ids  : indices of the tree nodes near s_new
        parent_id : the node the extension started from (always a valid parent)
        cost[i]   : path length from the root to node i in feature space
        Returns (best_parent_id, best_cost) where best_cost is the new node's cost.
        """
        # ------------------------------------------------------------------
        # TODO 7 (Task 4): choose the parent that gives the shortest path.
        #   best = parent_id,  best_cost = cost[parent_id] + ||s_new - tree_f[parent_id].s||
        #   for i in near_ids:
        #       c = cost[i] + ||s_new - tree_f[i].s||
        #       if c < best_cost and self._connectable(tree_c[i].q, q_new):
        #           best, best_cost = i, c
        #   return best, best_cost
        # (check the cheap cost test first: _connectable runs collision checks)
        # ------------------------------------------------------------------
        raise NotImplementedError("TODO 7: ImageBasedRRTPlanner._choose_parent (planner.py)")

    def _rewire(self, new_id, near_ids, tree_f, tree_c, cost):
        """Re-route neighbours through the new node when that is shorter."""
        # ------------------------------------------------------------------
        # TODO 8 (Task 4): rewire the tree around the new node.
        #   for j in near_ids (skip the new node's own parent):
        #       c = cost[new_id] + ||tree_f[j].s - tree_f[new_id].s||
        #       if c < cost[j] and self._connectable(tree_c[new_id].q, tree_c[j].q):
        #           make new_id the parent of j in BOTH trees:
        #               tree_f[j].parent_id = tree_c[j].parent_id = new_id
        #           and update the costs of j and its whole subtree:
        #               self._update_costs(j, c - cost[j], tree_f, cost)
        # ------------------------------------------------------------------
        raise NotImplementedError("TODO 8: ImageBasedRRTPlanner._rewire (planner.py)")

    # -- planning loops (provided) -----------------------------------------

    def _start(self, q_init, s_desired):
        s_init, z_init = self._feature_depth(q_init)
        if s_init is None or not np.isfinite(z_init) or z_init <= 0:
            raise ValueError("Target not visible from q_init.")
        s_init = np.asarray(s_init, dtype=np.float64)
        if not self._check(s_init, q_init, z_init):
            raise ValueError(f"Initial state violates constraints: s={s_init}")
        print(f"[RRT] Starting... initial error: {np.linalg.norm(s_desired - s_init):.2f}")
        return [_FeatureNode(s_init, np.zeros(3), 0)], [_ConfigNode(q_init, np.zeros(6), 0)]

    def _new_node(self, tree_f, tree_c, count, s_desired, k_bias, dt):
        """Sample, pick the nearest node, extend and check the constraints.
        Returns (parent_id, s_new, q_new, q_dot, J1) or None."""
        s_rand = self._sample(count, s_desired, k_bias)
        all_s = np.vstack([n.s for n in tree_f])
        pid = self._nearest(all_s, s_rand)
        sp, qp = tree_f[pid].s, tree_c[pid].q
        s_at_qp, zp = self._feature_depth(qp)
        if s_at_qp is None or not np.isfinite(zp) or zp <= 0:
            return None
        q_new, q_dot, J1 = self._extend(sp, qp, zp, s_rand, dt)
        if not np.all(np.isfinite(q_dot)):
            return None
        s_new, z_new = self._feature_depth(q_new)
        if s_new is None or not self._valid_feature(s_new):
            return None
        s_new = np.asarray(s_new, dtype=np.float64)
        if np.min(np.linalg.norm(all_s - s_new, axis=1)) < 1e-6:
            return None                                   # duplicate node
        if not self._check(s_new, q_new, z_new) or not self._edge_free(qp, q_new):
            return None
        return pid, s_new, q_new, q_dot, J1

    def _path(self, tree_f, tree_c, goal_id):
        path_f, path_j = [], []
        cid = goal_id
        while cid is not None:
            path_f.append(tree_f[cid].s)
            path_j.append(tree_c[cid].q)
            cid = tree_f[cid].parent_id
        return self._densify(path_f[::-1], path_j[::-1], RRT_MAX_WAYPOINT_DISTANCE)

    @staticmethod
    def path_cost(tree_f, goal_id):
        """Length of the tree path from the root to goal_id in feature space."""
        c, i = 0.0, goal_id
        while tree_f[i].parent_id is not None:
            c += float(np.linalg.norm(tree_f[i].s - tree_f[tree_f[i].parent_id].s))
            i = tree_f[i].parent_id
        return c

    def plan(self, q_init, s_desired, k_bias=RRT_GOAL_BIAS_K, radius=RRT_RADIUS_R,
             max_iter=RRT_MAX_ITER, dt=RRT_DT, allow_partial=False, on_new_node=None):
        """RRT: stop at the first node within ``radius`` of s_desired.

        on_new_node(tree_f, tree_c, node_id) is called after every new node.
        Returns (features, joints, tree_f, tree_c).
        """
        q_init = np.asarray(q_init, dtype=np.float64)
        s_desired = np.asarray(s_desired, dtype=np.float64)
        tree_f, tree_c = self._start(q_init, s_desired)
        goal_id = 0 if np.linalg.norm(s_desired - tree_f[0].s) <= radius else None

        count = 0
        for count in range(1, int(max_iter) + 1):
            if goal_id is not None:
                break
            new = self._new_node(tree_f, tree_c, count, s_desired, k_bias, dt)
            if new is None:
                continue
            pid, s_new, q_new, q_dot, J1 = new
            nid = len(tree_f)
            tree_f.append(_FeatureNode(s_new, J1 @ q_dot, nid, pid))
            tree_c.append(_ConfigNode(q_new, q_dot, nid, pid))
            if on_new_node is not None:
                on_new_node(tree_f, tree_c, nid)
            if np.linalg.norm(s_desired - s_new) <= radius:
                goal_id = nid
                print(f"[RRT] Goal reached at iteration {count}")

        return self._finish(tree_f, tree_c, goal_id, s_desired, count, max_iter, allow_partial)

    def plan_star(self, q_init, s_desired, k_bias=RRT_GOAL_BIAS_K, radius=RRT_RADIUS_R,
                  max_iter=RRT_STAR_ITER, dt=RRT_DT, allow_partial=False, on_new_node=None):
        """RRT*: every new node takes the cheapest parent among its neighbours
        and the neighbours are rewired through it.  Runs all ``max_iter``
        iterations, then returns the cheapest path that reaches the goal."""
        q_init = np.asarray(q_init, dtype=np.float64)
        s_desired = np.asarray(s_desired, dtype=np.float64)
        tree_f, tree_c = self._start(q_init, s_desired)
        cost = [0.0]

        for count in range(1, int(max_iter) + 1):
            new = self._new_node(tree_f, tree_c, count, s_desired, k_bias, dt)
            if new is None:
                continue
            pid, s_new, q_new, q_dot, J1 = new
            near = self._near(np.vstack([n.s for n in tree_f]), s_new)
            best, best_cost = self._choose_parent(near, s_new, q_new, pid, tree_f, tree_c, cost)
            nid = len(tree_f)
            tree_f.append(_FeatureNode(s_new, J1 @ q_dot, nid, best))
            tree_c.append(_ConfigNode(q_new, q_dot, nid, best))
            cost.append(best_cost)
            self._rewire(nid, near, tree_f, tree_c, cost)
            if on_new_node is not None:
                on_new_node(tree_f, tree_c, nid)

        reached = [i for i, n in enumerate(tree_f) if np.linalg.norm(s_desired - n.s) <= radius]
        goal_id = min(reached, key=lambda i: cost[i]) if reached else None
        if goal_id is not None:
            print(f"[RRT*] {len(reached)} nodes reach the goal; best path cost {cost[goal_id]:.0f}")
        return self._finish(tree_f, tree_c, goal_id, s_desired, max_iter, max_iter, allow_partial)

    def _finish(self, tree_f, tree_c, goal_id, s_desired, count, max_iter, allow_partial):
        success = goal_id is not None
        if not success:
            dists = np.array([np.linalg.norm(s_desired - n.s) for n in tree_f])
            goal_id = int(np.argmin(dists))
            msg = f"Planner failed in {max_iter} iterations; closest={dists[goal_id]:.2f}"
            if not allow_partial:
                self.last_plan_result = {"success": False, "iterations": int(max_iter),
                                         "explored_nodes": len(tree_f), "goal_id": goal_id}
                raise RuntimeError(msg)
            warnings.warn(msg + " Returning partial path.")
        path_f, path_j = self._path(tree_f, tree_c, goal_id)
        self.last_plan_result = {"success": success, "iterations": int(count),
                                 "explored_nodes": len(tree_f), "goal_id": goal_id,
                                 "path_cost": self.path_cost(tree_f, goal_id)}
        print(f"[RRT] {len(path_f)} waypoints from {len(tree_f)} explored nodes, "
              f"path cost {self.last_plan_result['path_cost']:.0f} (success={success})")
        return np.asarray(path_f), np.asarray(path_j), tree_f, tree_c
