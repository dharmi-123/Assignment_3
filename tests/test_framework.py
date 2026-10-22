"""Checks of the provided framework.  These pass before any TODO is done.

    python -m unittest tests.test_framework -v
"""

import unittest
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

from config import (GOAL_FEATURE, OBSTACLE_ENVIRONMENTS, OBSTACLE_IMAGE_MARGIN_PX,
                    OBSTACLE_SCENE, Q0_DEFAULT, RRT_GOAL_BIAS_K, WORKBENCH_S0)
from kinematics import (TaskPrioritySolver, damped_pinv,
                        secondary_task_jacobian)
from planner import ImageBasedRRTPlanner, WorkspaceDatasetGenerator
from tests.fakes import FakeEnv, FakeKinematics, tiny_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def obstacle_env(key, q0=None, with_walls=True):
    from simulation import PyBulletUR5Env
    cfg = OBSTACLE_ENVIRONMENTS[key]
    scene = dict(OBSTACLE_SCENE, q0=cfg["q0"] if q0 is None else q0, view=cfg["view"])
    return PyBulletUR5Env(p.DIRECT, scene=scene, obstacles=cfg["walls"] if with_walls else [])


class TestDampedPinv(unittest.TestCase):
    def test_matches_pinv_without_damping(self):
        J = np.random.default_rng(0).normal(size=(3, 6))
        np.testing.assert_allclose(damped_pinv(J, 0.0), np.linalg.pinv(J), atol=1e-10)


class TestTaskPriority(unittest.TestCase):
    def test_primary_task_minimum_norm(self):
        J1 = np.random.default_rng(1).normal(size=(3, 6))
        s_dot = np.array([1.0, -2.0, 3.0])
        q_dot = TaskPrioritySolver(damping=0.0).solve(J1, s_dot)
        np.testing.assert_allclose(q_dot, np.linalg.pinv(J1) @ s_dot, atol=1e-9)

    def test_secondary_task_in_null_space(self):
        rng = np.random.default_rng(2)
        J1, J2 = rng.normal(size=(3, 6)), rng.normal(size=(3, 6))
        s_dot = np.array([0.3, -0.1, 0.2])
        q_dot = TaskPrioritySolver(damping=0.0).solve(J1, s_dot, J2)
        np.testing.assert_allclose(J1 @ q_dot, s_dot, atol=1e-9)
        np.testing.assert_allclose(J2 @ q_dot, 0.0, atol=1e-9)


class TestSamplingAndNearest(unittest.TestCase):
    def test_goal_bias(self):
        ds = tiny_dataset([1, 2, 300], [4, 5, 600], [7, 8, 900])
        planner = ImageBasedRRTPlanner(FakeEnv(), FakeKinematics(), ds, random_seed=0)
        for count in range(1, 3 * RRT_GOAL_BIAS_K + 1):
            s = planner._sample(count, GOAL_FEATURE)
            if count % RRT_GOAL_BIAS_K == 0:
                np.testing.assert_allclose(s, GOAL_FEATURE)
            else:
                self.assertTrue(any(np.allclose(s, row) for row in ds["s_workspace"]))

    def test_nearest(self):
        tree = np.array([[0, 0, 1000], [100, 0, 1500], [-50, 20, 2900.0]])
        self.assertEqual(ImageBasedRRTPlanner._nearest(tree, np.array([0, 0, 3000.0])), 2)

    def test_near_sorted_within_radius(self):
        tree = np.array([[0, 0, 1000], [100, 0, 1000], [30, 0, 1000.0]])
        self.assertEqual(ImageBasedRRTPlanner._near(tree, np.array([0, 0, 1000.0]), radius=50), [0, 2])

    def test_update_costs_reaches_descendants(self):
        from planner import _FeatureNode
        tree = [_FeatureNode(np.zeros(3), np.zeros(3), 0), _FeatureNode(np.ones(3), np.zeros(3), 1, 0),
                _FeatureNode(np.ones(3), np.zeros(3), 2, 1), _FeatureNode(np.ones(3), np.zeros(3), 3, 0)]
        cost = [0.0, 10.0, 20.0, 5.0]
        ImageBasedRRTPlanner._update_costs(1, -4.0, tree, cost)
        self.assertEqual(cost, [0.0, 6.0, 16.0, 5.0])


class TestFixedOrientationJacobian(unittest.TestCase):
    def test_is_angular_block_of_camera_jacobian(self):
        Jc = np.random.default_rng(0).normal(size=(6, 6))
        np.testing.assert_allclose(secondary_task_jacobian("fixed_orientation", Jc), Jc[3:])


class TestDataset(unittest.TestCase):
    def test_generator_is_seeded_and_validated(self):
        gen = WorkspaceDatasetGenerator(FakeEnv(feature=(1.0, 2.0, 1000.0)))
        d1 = gen.generate(3, seed=4, max_attempts=3)
        d2 = gen.generate(3, seed=4, max_attempts=3)
        self.assertTrue(gen.is_compatible(d1, 3))
        np.testing.assert_allclose(d1["q_workspace"], d2["q_workspace"])
        self.assertFalse(gen.is_compatible(dict(d1, metadata=None), 3))

    def test_shipped_datasets_match_the_scenes(self):
        from simulation import PyBulletUR5Env
        for env in (PyBulletUR5Env(p.DIRECT), obstacle_env("a")):
            try:
                gen = WorkspaceDatasetGenerator(env)
                ds = gen.load(DATA_DIR / f"dataset_{env.scene['name']}_2000.npz")
                self.assertTrue(gen.is_compatible(ds, 2000))
            finally:
                env.close()


class TestPlannerLoop(unittest.TestCase):
    def test_start_inside_goal_radius_needs_no_extension(self):
        planner = ImageBasedRRTPlanner(FakeEnv(), FakeKinematics(), tiny_dataset([0, 0, 2950]))
        f, _, _, _ = planner.plan(Q0_DEFAULT, GOAL_FEATURE, radius=60)
        self.assertTrue(planner.last_plan_result["success"])
        self.assertEqual(f.shape, (1, 3))


class TestScenes(unittest.TestCase):
    def test_workbench_initial_view(self):
        from simulation import PyBulletUR5Env
        env = PyBulletUR5Env(p.DIRECT)
        try:
            np.testing.assert_allclose(env.get_current_feature()[0], WORKBENCH_S0, atol=2.0)
        finally:
            env.close()

    def test_obstacle_starts_are_valid(self):
        for key in OBSTACLE_ENVIRONMENTS:
            env = obstacle_env(key)
            try:
                s, _, _ = env.get_current_feature()
                self.assertIsNotNone(s)
                self.assertFalse(env.check_collision())
                for hull, _ in env.project_obstacles(OBSTACLE_IMAGE_MARGIN_PX):
                    inside = cv2.pointPolygonTest(hull.astype(np.float32).reshape(-1, 1, 2),
                                                  (float(s[0]), float(s[1])), False)
                    self.assertLess(inside, 0)          # target not behind an obstacle
            finally:
                env.close()

    def test_obstacle_goal_posture_sees_the_goal_feature(self):
        env = obstacle_env("a", q0=OBSTACLE_SCENE["dataset_q_center"], with_walls=False)
        try:
            np.testing.assert_allclose(env.get_current_feature()[0], GOAL_FEATURE, atol=150)
            self.assertFalse(env.check_collision())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
