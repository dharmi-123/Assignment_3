"""Task 4 (image-space RRT*): TODO 7 (choose parent) and TODO 8 (rewire).

    python -m unittest tests.test_task4 -v
"""

import unittest
from pathlib import Path

import numpy as np
import pybullet as p

from config import GOAL_FEATURE, OBSTACLE_ENVIRONMENTS, OBSTACLE_SCENE
from planner import ImageBasedRRTPlanner, WorkspaceDatasetGenerator, _ConfigNode, _FeatureNode
from tests.fakes import FakeEnv, FakeKinematics, tiny_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def make_tree(nodes):
    """nodes: list of (s, q1, parent); q = [q1, 0, 0, 0, 0, 0]."""
    tree_f, tree_c = [], []
    for i, (s, q1, parent) in enumerate(nodes):
        tree_f.append(_FeatureNode(np.array(s, dtype=float), np.zeros(3), i, parent))
        tree_c.append(_ConfigNode(np.array([q1, 0, 0, 0, 0, 0.0]), np.zeros(6), i, parent))
    return tree_f, tree_c


def planner():
    return ImageBasedRRTPlanner(FakeEnv(), FakeKinematics(), tiny_dataset([0, 0, 1000]))


class TestTodo7ChooseParent(unittest.TestCase):
    def setUp(self):
        #           s                    q1    parent
        self.tree_f, self.tree_c = make_tree([
            ([0, 0, 1000], 0.0, None),       # 0 root
            ([100, 0, 1000], 0.1, 0),        # 1  cost 100
            ([50, 50, 1000], 0.2, 1),        # 2  cost 300 (long detour)
            ([100, 40, 1000], 2.0, 0),       # 3  cheap but far away in joint space
        ])
        self.cost = [0.0, 100.0, 300.0, 108.0]

    def test_picks_the_cheapest_connectable_parent(self):
        s_new, q_new = np.array([100.0, 50.0, 1000.0]), np.array([0.25, 0, 0, 0, 0, 0])
        best, c = planner()._choose_parent([1, 2, 3], s_new, q_new, 2, self.tree_f, self.tree_c, self.cost)
        self.assertEqual(best, 1)                      # node 3 is cheaper but not connectable
        self.assertAlmostEqual(c, 150.0)

    def test_keeps_the_extension_parent_when_it_is_best(self):
        s_new, q_new = np.array([60.0, 55.0, 1000.0]), np.array([0.25, 0, 0, 0, 0, 0])
        cost = [0.0, 400.0, 10.0, 400.0]
        best, c = planner()._choose_parent([1, 3], s_new, q_new, 2, self.tree_f, self.tree_c, cost)
        self.assertEqual(best, 2)
        self.assertAlmostEqual(c, 10.0 + np.hypot(10, 5))


class TestTodo8Rewire(unittest.TestCase):
    def test_reroutes_cheaper_neighbours_and_their_subtrees(self):
        tree_f, tree_c = make_tree([
            ([0, 0, 1000], 0.0, None),       # 0 root
            ([200, 0, 1000], 0.1, 0),        # 1  cost 200
            ([200, 100, 1000], 0.2, 1),      # 2  cost 300
            ([250, 100, 1000], 0.25, 2),     # 3  cost 350 (child of 2)
            ([100, 100, 1000], 0.15, 0),     # 4  new node, cost 141.42
        ])
        cost = [0.0, 200.0, 300.0, 350.0, float(np.hypot(100, 100))]
        planner()._rewire(4, [1, 2], tree_f, tree_c, cost)
        self.assertEqual(tree_f[2].parent_id, 4)        # 141.4 + 100 < 300
        self.assertEqual(tree_c[2].parent_id, 4)
        self.assertEqual(tree_f[1].parent_id, 0)        # 141.4 + 141.4 > 200: unchanged
        self.assertAlmostEqual(cost[2], np.hypot(100, 100) + 100)
        self.assertAlmostEqual(cost[3], np.hypot(100, 100) + 150)   # subtree updated

    def test_does_not_rewire_unconnectable_nodes(self):
        tree_f, tree_c = make_tree([
            ([0, 0, 1000], 0.0, None), ([200, 0, 1000], 0.1, 0), ([200, 100, 1000], 1.5, 1),
            ([100, 100, 1000], 0.15, 0)])
        cost = [0.0, 200.0, 300.0, float(np.hypot(100, 100))]
        planner()._rewire(3, [2], tree_f, tree_c, cost)
        self.assertEqual(tree_f[2].parent_id, 1)
        self.assertEqual(cost[2], 300.0)


class TestTask4EndToEnd(unittest.TestCase):
    def test_rrt_star_finds_a_shorter_path_than_rrt(self):
        from kinematics import UR5Kinematics
        from simulation import PyBulletUR5Env
        cfg = OBSTACLE_ENVIRONMENTS["b"]
        env = PyBulletUR5Env(p.DIRECT, scene=dict(OBSTACLE_SCENE, q0=cfg["q0"], view=cfg["view"]),
                             obstacles=cfg["walls"])
        try:
            kin = UR5Kinematics(env.robot_id, env.ee_link_index)
            ds = WorkspaceDatasetGenerator.load(DATA_DIR / "dataset_obstacles_2000.npz")
            costs = []
            for star in (False, True):
                pl = ImageBasedRRTPlanner(env, kin, ds, random_seed=cfg["rrt_seed"])
                _, planned_q, _, _ = (pl.plan_star if star else pl.plan)(cfg["q0"], GOAL_FEATURE)
                costs.append(pl.last_plan_result["path_cost"])
                for q in planned_q[::5]:
                    self.assertFalse(env.check_collision(q, obstacle_margin=0.0))
            self.assertLess(costs[1], 0.7 * costs[0])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
