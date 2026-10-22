"""
Robot kinematics, image interaction matrix and task-priority redundancy
resolution.

Provided:
  - damped_pinv             : damped least-squares pseudo-inverse
  - UR5Kinematics           : camera-frame Jacobian Jc of the UR5
  - InteractionMatrix.image_jacobian : J1 = Lm Jc
  - TaskPrioritySolver      : primary task plus a secondary task in the null space
  - secondary_task_jacobian : keep the camera orientation fixed (Tasks 3-4)

Student TODO (Task 1):
  - InteractionMatrix.compute : TODO 1  3x6 interaction matrix of s = [xc, yc, a]
"""

import numpy as np
import pybullet as p

from config import (
    CAMERA_OFFSET_POS,
    CAMERA_OFFSET_ORN,
    FOCAL_LENGTH_X,
    FOCAL_LENGTH_Y,
)


# --------------------- Damped pseudo-inverse -----------------------------

def damped_pinv(J, damping=1e-3):
    """Damped least-squares pseudo-inverse J+ = V S^-1 U^T, s/(s^2 + d^2)."""
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2 or not np.all(np.isfinite(J)):
        raise ValueError("J must be a finite 2-D matrix.")
    U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    if damping == 0:
        tol = np.finfo(float).eps * max(J.shape) * (sv[0] if sv.size else 0.0)
        inv_sv = np.where(sv > tol, 1.0 / sv, 0.0)
    else:
        inv_sv = sv / (sv ** 2 + damping ** 2)
    return (Vt.T * inv_sv) @ U.T


# --------------------- UR5 kinematics ------------------------------------

class UR5Kinematics:
    """Camera-frame Jacobian of the UR5 (PyBullet geometric Jacobian at the camera)."""

    def __init__(self, robot_id, ee_link_index=6):
        self.robot_id = robot_id
        self.ee_link_index = ee_link_index

    def _movable_joints(self, pc=0):
        return [
            i for i in range(p.getNumJoints(self.robot_id, physicsClientId=pc))
            if p.getJointInfo(self.robot_id, i, physicsClientId=pc)[2] != p.JOINT_FIXED
        ]

    def compute_camera_jacobian(self, q, physics_client=0,
                                camera_offset_pos=CAMERA_OFFSET_POS,
                                camera_offset_orn=CAMERA_OFFSET_ORN,
                                camera_mount_rotation=None):
        """6x6 Jacobian Jc mapping q_dot -> camera-frame twist [v_c, w_c].

        PyBullet's Jacobian is in the world frame; the interaction matrix
        expects a camera-frame twist.  This evaluates J at the optical centre
        and rotates both linear and angular blocks into camera axes.
        The robot must already be at configuration q.
        """
        q = np.asarray(q, dtype=np.float64)
        n = len(self._movable_joints(physics_client))
        if q.shape != (n,) or not np.all(np.isfinite(q)):
            raise ValueError(f"Expected a finite q of shape ({n},), got {q.shape}")
        if camera_mount_rotation is None:
            ee_R_cam = np.asarray(p.getMatrixFromQuaternion(list(camera_offset_orn)),
                                  dtype=np.float64).reshape(3, 3)
        else:
            ee_R_cam = np.asarray(camera_mount_rotation, dtype=np.float64)

        zeros = [0.0] * n
        J_lin_w, J_ang_w = p.calculateJacobian(
            self.robot_id, self.ee_link_index, list(np.asarray(camera_offset_pos, dtype=float)),
            q.tolist(), zeros, zeros, physicsClientId=physics_client,
        )
        link = p.getLinkState(self.robot_id, self.ee_link_index,
                              computeForwardKinematics=True,
                              physicsClientId=physics_client)
        w_R_ee = np.asarray(p.getMatrixFromQuaternion(link[5]), dtype=np.float64).reshape(3, 3)
        cam_R_w = (w_R_ee @ ee_R_cam).T          # world -> camera rotation
        return np.vstack([cam_R_w @ np.asarray(J_lin_w), cam_R_w @ np.asarray(J_ang_w)])


# --------------------- Interaction matrix --------------------------------

class InteractionMatrix:
    """3x6 image interaction matrix Lm for s = [xc, yc, area].

    Lm maps the camera twist [vx, vy, vz, wx, wy, wz] (camera frame) to the
    feature rate s_dot.  xc, yc are pixel offsets from the image centre and
    area is in pixels^2.
    """

    def __init__(self, fx=FOCAL_LENGTH_X, fy=FOCAL_LENGTH_Y):
        self.fx, self.fy = float(fx), float(fy)

    def compute(self, feature, Z):
        """Return Lm(s, Z), a 3x6 numpy array (Z = target depth in m)."""
        s = np.asarray(feature, dtype=np.float64)
        if s.shape != (3,) or not np.all(np.isfinite(s)):
            raise ValueError("feature must be a finite [xc, yc, area] vector.")
        if not np.isfinite(Z) or Z <= 0:
            raise ValueError("Z must be a positive finite depth.")
        xc, yc, a = s
        if a <= 0:
            raise ValueError("Feature area must be positive.")
        Z = max(float(Z), 0.01)
        fx, fy = self.fx, self.fy
        # ------------------------------------------------------------------
        # TODO 1 (Task 1): build the 3x6 interaction matrix Lm.
        #
        # With normalised coordinates x = xc / fx and y = yc / fy, the
        # point-feature rows from the lecture are
        #   L_x = [-1/Z,    0,  x/Z,    x*y, -(1 + x^2),  y]
        #   L_y = [   0, -1/Z,  y/Z, 1 + y^2,      -x*y, -x]
        # and for the area a of a planar object parallel to the image
        #   L_a = [   0,    0, 2a/Z,  3a*y,      -3a*x,   0]
        # Our xc, yc are in pixels, so scale row 1 by fx and row 2 by fy
        # (xc_dot = fx * x_dot).  Row 3 stays as written (a is already in px^2).
        # Return Lm as np.array of shape (3, 6).
        # ------------------------------------------------------------------
        raise NotImplementedError("TODO 1: InteractionMatrix.compute (kinematics.py)")

    def image_jacobian(self, Lm, J_camera):
        """J1 = Lm @ Jc: maps joint velocities to the feature rate (3 x n)."""
        Lm = np.asarray(Lm, dtype=np.float64)
        Jc = np.asarray(J_camera, dtype=np.float64)
        if Lm.shape != (3, 6) or Jc.shape[0] != 6:
            raise ValueError(f"Shape mismatch: Lm{Lm.shape}, Jc{Jc.shape}")
        return Lm @ Jc


# --------------------- Task-priority solver ------------------------------

class TaskPrioritySolver:
    """Null-space task-priority redundancy resolution.

    q_dot = J1+ s_dot + P1 (J2 P1)+ (q_e_dot - J2 J1+ s_dot),  P1 = I - J1+ J1
    """

    def __init__(self, damping=1e-3):
        self.damping = float(damping)

    def solve(self, J1, s_dot, J2=None, q_e_dot=None):
        """Joint velocity for the primary task J1 q_dot = s_dot and, if J2 is
        given, the secondary task J2 q_dot = q_e_dot (default zero)."""
        J1 = np.asarray(J1, dtype=np.float64)
        s_dot = np.asarray(s_dot, dtype=np.float64)
        n = J1.shape[1]

        # Primary task: q_dot = J1+ s_dot
        J1_pinv = damped_pinv(J1, self.damping)
        q_dot = J1_pinv @ s_dot

        if J2 is None:
            return q_dot
        J2 = np.asarray(J2, dtype=np.float64)
        q_e_dot = (np.zeros(J2.shape[0]) if q_e_dot is None
                   else np.asarray(q_e_dot, dtype=np.float64))

        # Secondary task in the null space of J1 (P1 = null-space projector)
        P1 = np.eye(n) - J1_pinv @ J1
        J2_P1_pinv = damped_pinv(J2 @ P1, self.damping)
        return q_dot + P1 @ J2_P1_pinv @ (q_e_dot - J2 @ q_dot)


# --------------------- Secondary task ------------------------------------

def secondary_task_jacobian(kind, J_camera):
    """J2 (3 x 6) of the secondary task J2 q_dot = 0.

    kind = 'fixed_orientation': the camera does not rotate (Tasks 3-4).
    J_camera is the 6x6 camera Jacobian Jc (rows 0-2 linear, rows 3-5 angular).
    """
    if kind == "fixed_orientation":
        return np.asarray(J_camera, dtype=np.float64)[3:, :]       # w_c = Jc[3:] q_dot
    raise ValueError(f"Unknown secondary task: {kind}")
