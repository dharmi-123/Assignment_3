"""Task 1 (IBVS, static target): TODO 1 (interaction matrix) and TODO 2
(classic IBVS control law).

    python -m unittest tests.test_task1 -v
"""

import unittest

import numpy as np
import pybullet as p

from config import GOAL_FEATURE
from ibvs import IBVSServo, ibvs_camera_velocity
from kinematics import InteractionMatrix

# A fixed interaction matrix, so TODOs 2-4 can be tested without TODO 1
L_FIXED = np.array([[-1000.0, 0, 200, -10, -520, -50],
                    [0, -1000, -100, 505, 10, -100],
                    [0, 0, 4000, -300, -600, 0]])


class TestTodo1InteractionMatrix(unittest.TestCase):
    def test_values(self):
        Lm = InteractionMatrix().compute(np.array([100.0, -50.0, 1000.0]), 0.5)
        self.assertEqual(Lm.shape, (3, 6))
        np.testing.assert_allclose(Lm, [[-1000, 0, 200, -10, -520, -50],
                                        [0, -1000, -100, 505, 10, -100],
                                        [0, 0, 4000, -300, -600, 0]], atol=1e-9)

    def test_values_second_point(self):
        Lm = InteractionMatrix().compute(np.array([-20.0, 30.0, 2500.0]), 0.25)
        np.testing.assert_allclose(Lm, [[-2000, 0, -80, -1.2, -500.8, 30],
                                        [0, -2000, 120, 501.8, 1.2, 20],
                                        [0, 0, 20000, 450, 300, 0]], atol=1e-9)

    def test_centroid_rows_match_projection(self):
        """Rows 1-2 predict how a 3-D point's pixel moves under a camera twist."""
        fx = fy = 500.0
        P = np.array([0.05, -0.03, 0.4])                    # point in camera frame
        twist = np.array([0.01, -0.02, 0.03, 0.05, -0.04, 0.02])
        proj = lambda X: np.array([fx * X[0] / X[2], fy * X[1] / X[2]])
        P_dot = -twist[:3] - np.cross(twist[3:], P)          # static point, moving camera
        h = 1e-6
        numeric = (proj(P + h * P_dot) - proj(P)) / h
        Lm = InteractionMatrix(fx, fy).compute(np.r_[proj(P), 1000.0], P[2])
        np.testing.assert_allclose(Lm[:2] @ twist, numeric, rtol=1e-4)


class TestTodo2IBVSLaw(unittest.TestCase):
    def test_error_decays_exponentially(self):
        e = np.array([40.0, -30.0, -1500.0])
        v = ibvs_camera_velocity(L_FIXED, e, 1.5)
        self.assertEqual(np.shape(v), (6,))
        np.testing.assert_allclose(L_FIXED @ v, -1.5 * e, rtol=1e-4)  # e_dot = -lambda e

    def test_minimum_norm_twist(self):
        e = np.array([5.0, 2.0, -10.0])
        np.testing.assert_allclose(ibvs_camera_velocity(L_FIXED, e, 1.0),
                                   -np.linalg.pinv(L_FIXED) @ e, rtol=1e-4)

    def test_servo_reaches_the_goal(self):
        from kinematics import UR5Kinematics
        from simulation import PyBulletUR5Env
        env = PyBulletUR5Env(p.DIRECT)
        try:
            kin = UR5Kinematics(env.robot_id, env.ee_link_index)
            res = IBVSServo(env, kin, GOAL_FEATURE).run(100)
            self.assertEqual(res["status"], "completed")
            self.assertLess(res["errors"][-1], 10.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
