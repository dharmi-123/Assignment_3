"""
Trajectory-tracking IBVS: follow a planned feature path (Tasks 3-4).

The control law is the moving-target law of Task 2 with the planned feature
velocity as the known "target motion":
    T = -lambda Lm+ (s_k - s*_k) + Lm+ (s*_k - s*_{k-1}) / dt
"""

import numpy as np

from config import (
    CAMERA_OFFSET_POS,
    CONTROL_DT, TRACKING_GAIN,
    JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER, JOINT_VELOCITY_LIMITS,
    MAX_CONTROL_STEPS,
)
from kinematics import InteractionMatrix, TaskPrioritySolver, damped_pinv, secondary_task_jacobian


class IBVSTrajectoryController:
    """Track a planned image-feature path, one planned node per control step."""

    def __init__(self, env, kinematics, lambda_gain=TRACKING_GAIN, dt=CONTROL_DT,
                 secondary_task=None):
        self.env = env
        self.kin = kinematics
        self.Lm = InteractionMatrix()
        self.tp = TaskPrioritySolver()
        self.lam = float(lambda_gain)
        self.dt = float(dt)
        self.secondary = secondary_task         # None or 'fixed_orientation'
        self.has_obstacles = bool(getattr(env, "obstacle_ids", None))

    @staticmethod
    def _limit_vel(q_dot):
        """Scale q_dot down uniformly so no joint exceeds its velocity limit."""
        ratio = float(np.max(np.abs(q_dot) / JOINT_VELOCITY_LIMITS))
        return q_dot / ratio if ratio > 1 else q_dot

    def _cam_jac(self, q):
        return self.kin.compute_camera_jacobian(
            q, physics_client=self.env.physics_client,
            camera_offset_pos=getattr(self.env, "camera_offset_pos", CAMERA_OFFSET_POS),
            camera_mount_rotation=getattr(self.env, "_camera_mount_rotation", None))

    def _control_step(self, q, s_cur, z, s_des, s_prev):
        """One step of the trajectory-tracking controller.

        q      : current joint configuration (6,)
        s_cur  : current measured feature s_k (3,)
        z      : current target depth (m)
        s_des  : planned feature s*_k of this step (3,)
        s_prev : planned feature s*_{k-1} of the previous step (equal to
                 s_des at the first step, so the feed-forward term is zero)

        Returns (q_dot, err, Jc): joint velocity (6,), tracking error
        s_k - s*_k (3,) and the camera Jacobian used (6x6).
        """
        Jc = self._cam_jac(q)
        J2 = None if not self.secondary else secondary_task_jacobian(self.secondary, Jc)
        # Feedback on the tracking error plus feed-forward of the planned feature velocity
        err = s_cur - s_des
        s_rate = (s_des - s_prev) / self.dt
        Lm = self.Lm.compute(s_des, z)
        if J2 is None:
            Lm_pinv = damped_pinv(Lm)
            T = -self.lam * (Lm_pinv @ err) + Lm_pinv @ s_rate
            q_dot = damped_pinv(Jc) @ T
        else:
            J1 = self.Lm.image_jacobian(Lm, Jc)
            q_dot = self.tp.solve(J1, -self.lam * err + s_rate, J2)
        return q_dot, err, Jc

    # -- execution loop -----------------------------------------------------

    def execute_trajectory(self, planned_s, q_init, max_control_steps=MAX_CONTROL_STEPS,
                           evaluation_goal=None, step_callback=None):
        """Track the planned features: one control step per planned node,
        then the robot moves for one control period dt."""
        traj = np.asarray(planned_s, dtype=np.float64)
        q_init = np.asarray(q_init, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != 3 or len(traj) == 0:
            raise ValueError("planned_s must have shape (N, 3).")

        self.env.reset_joint_state(q_init)
        feat, depth, _ = self.env.get_current_feature()
        if feat is None:
            raise RuntimeError("Target not visible at start.")
        feat = np.asarray(feat, dtype=np.float64)
        goal = np.asarray(evaluation_goal if evaluation_goal is not None
                          else traj[-1], dtype=np.float64)

        ex_feat, ex_jnt, ex_vel = [], [], []
        trk_err, cam_vel, goal_err, clearance, depths = [], [], [], [], []
        q_cur = q_init.copy()
        cur_feat, cur_z = feat.copy(), float(depth)
        done, reason = True, "trajectory_complete"
        n_steps = min(len(traj), int(max_control_steps))

        print(f"[IBVS] Executing {len(traj)} feature nodes (dt = {self.dt} s)...")

        for k in range(n_steps):
            s_des = traj[k]
            s_prev = traj[k - 1] if k > 0 else s_des
            qd, err, Jc = self._control_step(q_cur, cur_feat, cur_z, s_des, s_prev)
            qd = self._limit_vel(np.asarray(qd, dtype=np.float64))
            q_next = np.clip(q_cur + qd * self.dt, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

            # Safety stop on actual contact (the planner keeps a safety offset)
            if self.has_obstacles and self.env.check_collision(q_next, obstacle_margin=0.0):
                done, reason = False, "safety_stop_contact"
                break

            qd = (q_next - q_cur) / self.dt
            q_cur = q_next
            self.env.reset_joint_state(q_cur)          # move the robot
            feat_info = self.env.get_current_feature(return_mask=True)
            if feat_info[0] is None:
                done, reason = False, "target_lost"
                break

            cur_feat = np.asarray(feat_info[0], dtype=np.float64)
            cur_z = float(feat_info[1])
            ex_feat.append(cur_feat.copy())
            ex_jnt.append(q_cur.copy())
            ex_vel.append(qd.copy())
            trk_err.append(float(np.linalg.norm(err)))
            cam_vel.append(Jc @ qd)
            goal_err.append(float(np.linalg.norm(cur_feat - goal)))
            depths.append(cur_z)
            if self.has_obstacles:
                clearance.append(self.env.min_obstacle_distance())

            if callable(step_callback):
                step_callback({
                    "step": k + 1,
                    "waypoint_idx": k,
                    "total_waypoints": len(traj),
                    "target_waypoint": s_des,
                    "current_feature": cur_feat,
                    "depth": cur_z,
                    "tracking_error": trk_err[-1],
                    "goal_error": goal_err[-1],
                    "joint_positions": q_cur,
                    "joint_velocities": qd,
                    "rgb": feat_info[2],
                    "mask": feat_info[3],
                    "status": "Servoing...",
                    "clearance": clearance[-1] if clearance else None,
                })
        else:
            if n_steps < len(traj):
                done, reason = False, "max_control_steps"

        fe = float(np.linalg.norm(cur_feat - goal))
        fv = float(np.linalg.norm(ex_vel[-1])) if ex_vel else 0.0
        print(f"[IBVS] {reason}; goal error={fe:.2f}, joint-speed={fv:.6f} rad/s")

        return {
            "executed_features": np.asarray(ex_feat).reshape(-1, 3),
            "executed_joints": np.asarray(ex_jnt).reshape(-1, 6),
            "executed_joint_vels": np.asarray(ex_vel).reshape(-1, 6),
            "tracking_errors": np.asarray(trk_err),
            "camera_velocities": np.asarray(cam_vel).reshape(-1, 6),
            "goal_errors": np.asarray(goal_err),
            "obstacle_clearance": np.asarray(clearance),
            "depths": np.asarray(depths),
            "initial_error": float(np.linalg.norm(feat - goal)),
            "final_error": fe,
            "final_joint_vel_norm": fv,
            "completed": done,
            "termination_reason": reason,
        }
