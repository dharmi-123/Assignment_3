"""
Experiment configuration for the four tasks of Assignment 3.
"""

import numpy as np
from config.robot_config import (
    CAMERA_OFFSET_ORN, CAMERA_OFFSET_POS, Q0_DEFAULT,
    TARGET_COLOR_RGB, TARGET_SIZE_METERS,
)

# Desired feature for every case: target centred in the image, area 3000 px
GOAL_FEATURE = np.array([0.0, 0.0, 3000.0])
# Feature seen by the workbench camera at Q0_DEFAULT (Tasks 1-2)
WORKBENCH_S0 = np.array([156.64, -122.903, 1450.23])

# ---------------------------------------------------------------------------
# Scenes.  ``table`` is the table top (centre x, y and half extents); its
# height is the robot base height, so the robot stands on the table.
# ---------------------------------------------------------------------------
WORKBENCH_SCENE = {           # Tasks 1-2: green target on a workbench
    "name": "workbench",
    "table": {"center": (0.30, 0.0), "half": (0.55, 0.50)},
    "robot_base": (0.0, 0.0, 0.45),
    "robot_rgba": None,                       # keep the UR5 mesh colours
    "camera_pos": CAMERA_OFFSET_POS,          # in ee_link frame
    "camera_orn": CAMERA_OFFSET_ORN,
    "q0": Q0_DEFAULT,
    "target_pos": (0.45, 0.0, 0.45 + TARGET_SIZE_METERS[2] / 2.0),
    "target_size": TARGET_SIZE_METERS,
    "target_rgba": TARGET_COLOR_RGB,
    "target_hsv": ((35, 50, 50), (85, 255, 255)),     # green
    "dataset_q_center": Q0_DEFAULT,
    "dataset_joint_std": (0.25, 0.25, 0.25, 0.35, 0.25, 0.60),
    # third-person view used by the GUI and the saved motion sequences
    "view": {"yaw": 300, "pitch": -25, "distance": 1.15, "target": (0.32, 0.0, 0.68)},
}

# Tasks 3-4: the table is at the height of the arm (the robot stands on it).
# The camera sits at the tip of a 0.25 m rod along the tool axis and looks
# forward, 20 deg down, at a blue cube.  A red obstacle stands between the
# start pose and the goal pose, so the direct motion would hit it.
TABLE_Z = 0.60
OBSTACLE_SCENE = {
    "name": "obstacles",
    "table": {"center": (0.55, 0.0), "half": (0.70, 0.60)},
    "robot_base": (0.0, 0.0, TABLE_Z),
    "robot_rgba": (0.12, 0.12, 0.13, 1.0),
    "camera_pos": np.array([0.25, 0.0, 0.0]),       # optical axis = ee_link x
    "camera_orn": np.array([0.5, 0.5, 0.5, 0.5]),   # image x = ee y, image y = ee z
    "q0": None,                                     # per environment, see below
    "target_pos": (0.98, 0.0, TABLE_Z + 0.015),
    "target_size": (0.03, 0.03, 0.03),
    "target_rgba": (0.05, 0.15, 0.95, 1.0),
    "target_hsv": ((100, 120, 60), (130, 255, 255)),  # blue
    # workspace images are taken around the goal posture (s = GOAL_FEATURE)
    "dataset_q_center": np.array([-0.302, -1.480, 1.973, -0.129, 1.288, -0.106]),
    "dataset_joint_std": (0.30, 0.20, 0.25, 0.25, 0.20, 0.25),
}


def _box(x, y, size_x, size_y, height=0.25):
    """Red box standing on the obstacle table: (centre, half_extents)."""
    return ([x, y, TABLE_Z + height / 2], [size_x / 2, size_y / 2, height / 2])


_VIEW = {"yaw": 90, "pitch": -40, "distance": 1.5, "target": (0.5, 0.0, 0.62)}

# Two obstacle layouts.  q0 puts the camera to the right of the obstacle with
# the cube in view; the planner has to take the arm around the obstacle.
OBSTACLE_ENVIRONMENTS = {
    "a": {   # a pillar
        "walls": [_box(0.45, -0.08, 0.06, 0.06)],
        "q0": np.array([-1.0228, -1.6071, 2.0914, 0.1255, 0.6400, -0.5108]),
        "rrt_seed": 13,
        "view": _VIEW,
    },
    "b": {   # a wall across the way
        "walls": [_box(0.45, -0.10, 0.02, 0.14)],
        "q0": np.array([-1.2513, -1.5678, 2.0989, 0.3278, 0.4687, -0.8021]),
        "rrt_seed": 11,
        "view": _VIEW,
    },
}

# Safety offsets around obstacles
OBSTACLE_SAFETY_MARGIN = 0.02     # m, minimum allowed robot-obstacle distance
OBSTACLE_IMAGE_MARGIN_PX = 25.0   # px, offset around projected obstacle regions

# Control period of every task (s)
CONTROL_DT = 0.1

# Tasks 1-2: IBVS
IBVS_GAIN = 1.0             # lambda of v = -lambda L+ e (1/s)
MOTION_FILTER = 0.5         # low-pass weight of the new target-motion estimate (Task 2)
IBVS_STEPS = 100            # Task 1: 10 s of servoing
TARGET_CIRCLE = {"radius": 0.05, "period": 8.0}   # Task 2: target moves on this circle
MOVING_STEPS = 250          # Task 2: 25 s of tracking

# Tasks 3-4: image-space RRT / RRT* and trajectory tracking
RRT_GOAL_BIAS_K = 10        # every k-th sample is the goal
RRT_RADIUS_R = 75.0         # goal radius in feature space (px)
RRT_MAX_ITER = 3000
RRT_DT = 1.0                # integration interval of one extension (s)
RRT_LAMBDA = 0.5            # feature-velocity gain of the extension step
RRT_STAR_ITER = 3000        # RRT* keeps improving the path for all iterations
RRT_STAR_NEAR_RADIUS = 150.0  # neighbourhood radius for choose-parent and rewire (px)
RRT_STAR_MAX_JOINT_STEP = 0.5 # longest joint-space edge RRT* may create (rad)
RRT_MAX_WAYPOINT_DISTANCE = 15.0  # planned path resampled to <= this many px per node
TRACKING_GAIN = 8.0         # feedback gain of the trajectory-tracking IBVS (1/s)
MAX_CONTROL_STEPS = 2000

# Dataset
DATASET_SIZE_FULL = 2000
DATASET_RANDOM_SEED = 7
RRT_RANDOM_SEED = 11

# ---------------------------------------------------------------------------
# The four tasks
# ---------------------------------------------------------------------------
TASKS = {
    1: {"folder": "task1_static_ibvs", "title": "Task 1: IBVS - Static Target",
        "scene": WORKBENCH_SCENE},
    2: {"folder": "task2_moving_target", "title": "Task 2: IBVS - Moving Target",
        "scene": WORKBENCH_SCENE},
    3: {"folder": "task3_rrt", "title": "Task 3: Image-Space RRT",
        "scene": OBSTACLE_SCENE, "environment": "a"},
    4: {"folder": "task4_rrt_star", "title": "Task 4: Image-Space RRT*",
        "scene": OBSTACLE_SCENE, "environment": "b"},
}
