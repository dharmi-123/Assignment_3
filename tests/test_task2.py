"""Task 2 (IBVS, moving target): TODO 3 (target-motion estimate) and TODO 4
(compensated control law).

    python -m unittest tests.test_task2 -v
"""

import unittest

import numpy as np
import pybullet as p

from config import GOAL_FEATURE
from ibvs import IBVSServo, compensated_camera_velocity, estimate_target_motion
# A fixed interaction matrix, so TODOs 2-4 can be tested without TODO 1
L_FIXED = np.array([[-1000.0, 0, 200, -10, -520, -50],
                    [0, -1000, -100, 505, 10, -100],
                    [0, 0, 4000, -300, -600, 0]])


class TestTodo3MotionEstimate(unittest.TestCase):
    def test_recovers_the_target_part_of_the_feature_rate(self):
        L = L_FIXED
        v = np.array([0.01, -0.02, 0.005, 0.0, 0.01, -0.02])
        target_rate = np.array([12.0, -7.0, 30.0])
        dt = 0.1
        s_prev = np.array([10.0, 20.0, 2000.0])
        s = s_prev + dt * (L @ v + target_rate)          # camera part + target part
        np.testing.assert_allclose(estimate_target_motion(s, s_prev, L, v, dt), target_rate, atol=1e-9)

    def test_static_target_gives_zero(self):
        L = L_FIXED
        v = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        s_prev = np.array([10.0, 20.0, 2000.0])
        np.testing.assert_allclose(estimate_target_motion(s_prev + 0.1 * L @ v, s_prev, L, v, 0.1),
                                   0.0, atol=1e-9)


class TestTodo4CompensatedLaw(unittest.TestCase):
    def test_cancels_target_motion(self):
        L = L_FIXED
        e = np.array([10.0, 20.0, -1000.0])
        de_hat = np.array([12.0, -7.0, 30.0])
        v = compensated_camera_velocity(L, e, 1.0, de_hat)
        np.testing.assert_allclose(L @ v + de_hat, -1.0 * e, rtol=1e-4, atol=1e-3)

    def test_reduces_tracking_error(self):
        from kinematics import UR5Kinematics
        from main import circle_motion
        from simulation import PyBulletUR5Env
        env = PyBulletUR5Env(p.DIRECT)
        try:
            kin = UR5Kinematics(env.robot_id, env.ee_link_index)
            motion = circle_motion(env.target_position)
            errors = {}
            for comp in (False, True):
                env.set_target_position(motion(0.0))
                res = IBVSServo(env, kin, GOAL_FEATURE, compensate=comp).run(160, target_motion=motion)
                self.assertEqual(res["status"], "completed")
                errors[comp] = np.mean(res["xy_errors"][80:])
            self.assertLess(errors[True], errors[False] / 3)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
