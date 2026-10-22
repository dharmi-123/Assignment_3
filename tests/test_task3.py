"""Task 3 (image-space RRT): TODO 5 (RRT extension) and TODO 6 (drawing the tree).

    python -m unittest tests.test_task3 -v
"""

import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from config import GOAL_FEATURE, OBSTACLE_ENVIRONMENTS, OBSTACLE_SCENE, Q0_DEFAULT, RRT_LAMBDA
from kinematics import InteractionMatrix
from planner import ImageBasedRRTPlanner, WorkspaceDatasetGenerator
from tests.fakes import FakeEnv, FakeKinematics, tiny_dataset
from visualizer import draw_rrt_tree

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class TestTodo5Extend(unittest.TestCase):
    def test_extension_moves_feature_towards_sample(self):
        planner = ImageBasedRRTPlanner(FakeEnv(), FakeKinematics(), tiny_dataset([0, 0, 1000]))
        sp, s_rand, zp, dt = np.array([10.0, -5.0, 1000.0]), np.array([0.0, 0.0, 1020.0]), 0.5, 1.0
        q_new, q_dot, J1 = planner._extend(sp, Q0_DEFAULT, zp, s_rand, dt)
        np.testing.assert_allclose(J1, InteractionMatrix().compute(sp, zp), atol=1e-9)   # Jc = I here
        np.testing.assert_allclose(J1 @ q_dot, RRT_LAMBDA * (s_rand - sp), atol=1e-3)
        np.testing.assert_allclose(q_dot[3:], 0.0, atol=1e-6)        # camera does not rotate
        np.testing.assert_allclose(q_new, Q0_DEFAULT + q_dot * dt, atol=1e-12)


class TestTodo6TreeDrawing(unittest.TestCase):
    points = np.array([[0.0, 0.0, 1000.0], [10.0, 5.0, 1200.0], [-8.0, 3.0, 1100.0],
                       [20.0, 9.0, 1500.0]])
    parents = [None, 0, 0, 1]

    def _check(self, ax, pts):
        draw_rrt_tree(ax, pts, self.parents, path=[0, 1, 3])
        lines = ax.get_lines()
        self.assertEqual(len(lines), 4)                    # 3 edges + the path
        red = [ln for ln in lines if matplotlib.colors.same_color(ln.get_color(), "red")]
        self.assertEqual(len(red), 1)
        np.testing.assert_allclose(red[0].get_xdata(), pts[[0, 1, 3], 0])
        self.assertEqual(len(ax.collections), 1)           # the node scatter
        plt.close(ax.figure)

    def test_image_plane(self):
        self._check(plt.figure().add_subplot(), self.points[:, :2])

    def test_feature_space_3d(self):
        self._check(plt.figure().add_subplot(projection="3d"), self.points)

    def test_without_path(self):
        ax = plt.figure().add_subplot()
        draw_rrt_tree(ax, self.points[:, :2], self.parents)
        self.assertEqual(len(ax.get_lines()), 3)
        plt.close(ax.figure)


class TestTask3EndToEnd(unittest.TestCase):
    def test_rrt_path_avoids_the_pillar(self):
        from kinematics import UR5Kinematics
        from simulation import PyBulletUR5Env
        cfg = OBSTACLE_ENVIRONMENTS["a"]
        env = PyBulletUR5Env(p.DIRECT, scene=dict(OBSTACLE_SCENE, q0=cfg["q0"], view=cfg["view"]),
                             obstacles=cfg["walls"])
        try:
            kin = UR5Kinematics(env.robot_id, env.ee_link_index)
            ds = WorkspaceDatasetGenerator.load(DATA_DIR / "dataset_obstacles_2000.npz")
            planner = ImageBasedRRTPlanner(env, kin, ds, random_seed=cfg["rrt_seed"])
            planned_s, planned_q, _, _ = planner.plan(cfg["q0"], GOAL_FEATURE)
            self.assertLessEqual(np.linalg.norm(planned_s[-1] - GOAL_FEATURE), 75.0)
            for q in planned_q[::5]:
                self.assertFalse(env.check_collision(q, obstacle_margin=0.0))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
