"""Small stand-ins for the simulator, so single TODOs can be tested quickly."""

import numpy as np

from config import Q0_DEFAULT


class FakeKinematics:
    """Camera Jacobian = identity: joint rates equal the camera twist."""

    @staticmethod
    def compute_camera_jacobian(*args, **kwargs):
        return np.eye(6)


class FakeEnv:
    """Constant target feature, no collisions, optional obstacle regions."""

    physics_client = 0
    _camera_mount_rotation = np.eye(3)
    camera_offset_pos = np.zeros(3)
    camera_offset_orn = np.array([0.0, 0.0, 0.0, 1.0])
    target_size = np.array([0.03, 0.02, 0.01])
    dataset_q_center = Q0_DEFAULT
    dataset_joint_std = np.full(6, 0.2)

    def __init__(self, feature=(0.0, 0.0, 2950.0), depth=0.5, obstacle_regions=None):
        self.feature = np.asarray(feature, dtype=np.float64)
        self.depth = depth
        self.regions = obstacle_regions or []
        self.obstacle_ids = [1] * len(self.regions)
        self.q = Q0_DEFAULT.copy()

    def reset_joint_state(self, q):
        self.q = np.asarray(q, dtype=np.float64).copy()

    def get_current_feature(self, return_mask=False):
        out = (self.feature.copy(), self.depth, None)
        return out + (None,) if return_mask else out

    def get_analytic_feature(self):
        return self.get_current_feature()

    def get_dataset_feature(self):
        return self.get_current_feature()

    def check_collision(self, q=None, obstacle_margin=None):
        return False

    def min_obstacle_distance(self):
        return np.inf

    def project_obstacles(self, margin_px=0.0):
        return self.regions


def tiny_dataset(*features):
    """Dataset dict with the given features (all at configuration Q0_DEFAULT)."""
    s = np.asarray(features, dtype=np.float64).reshape(-1, 3)
    return {"s_workspace": s, "q_workspace": np.tile(Q0_DEFAULT, (len(s), 1))}
