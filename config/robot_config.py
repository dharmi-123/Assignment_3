"""
Robot and Camera Configuration Parameters for UR5 Eye-in-Hand Visual Servoing.
"""

import numpy as np

# Camera Parameters
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CX = IMAGE_WIDTH / 2.0   # 320.0
CY = IMAGE_HEIGHT / 2.0  # 240.0
FOCAL_LENGTH_X = 500.0   # fx in pixels
FOCAL_LENGTH_Y = 500.0   # fy in pixels
FOV_X_MIN, FOV_X_MAX = -320.0, 320.0
FOV_Y_MIN, FOV_Y_MAX = -240.0, 240.0

# Rigid eye-in-hand transform relative to ``ee_link``. The quaternion uses
# PyBullet's (x, y, z, w) convention and the translation is expressed in the
# end-effector frame. Calibrated so that Q0_DEFAULT sees the target at WORKBENCH_S0.
CAMERA_OFFSET_POS = np.array([0.23467505, 0.13161050, 0.19693721])
CAMERA_OFFSET_ORN = np.array(
    [-0.21886277, -0.01365966, 0.40306311, 0.88851147]
)

# UR5 Joint Limits (radians)
JOINT_LIMITS_LOWER = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
JOINT_LIMITS_UPPER = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
# The local URDF uses the nominal UR5 velocity limit for all six axes.
JOINT_VELOCITY_LIMITS = np.full(6, 3.15)

# Start configuration of Tasks 1 and 2
Q0_DEFAULT = np.array([-0.08, -1.73, 1.04, -0.09, -1.57, 0.63])

# Target Object Parameters (Green rectangle)
TARGET_COLOR_RGB = [0.0, 0.8, 0.2, 1.0]
TARGET_SIZE_METERS = [0.03002, 0.01998, 0.008]  # calibrated to ~1450 px at q0

# Obstacle colour (Tasks 3-4)
OBSTACLE_COLOR_RGB = [0.9, 0.1, 0.1, 1.0]
