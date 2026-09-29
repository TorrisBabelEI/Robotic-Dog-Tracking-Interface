#!/usr/bin/env python3
import asyncio
import csv
import json
import os
import sys
import time

import numpy as np
from go1_highlevel_runtime import run_pose_loop

sys.path.append(os.getcwd() + '/externals/unitree_legged_sdk/lib/python/amd64')


def angle_error(target, current):
    return float(np.arctan2(np.sin(target - current), np.cos(target - current)))


class DenseTrajectoryTracker:
    def __init__(self, trajectory, dt=0.02, use_yaw=False, save_flag=False,
                 config_file_name='experiment/config/config_dog.json'):
        self.trajectory = np.asarray(trajectory, dtype=float).copy()
        if (self.trajectory.ndim != 2 or len(self.trajectory) < 2 or
                self.trajectory.shape[1] not in (2, 3) or not np.isfinite(self.trajectory).all()):
            raise ValueError('trajectory must be a finite N x 2 or N x 3 array, N >= 2')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        if use_yaw and self.trajectory.shape[1] != 3:
            raise ValueError('use_yaw=True requires x/y/yaw samples')
        self.dt = float(dt)
        self.use_yaw = use_yaw
        if use_yaw:
            self.trajectory[:, 2] = np.unwrap(self.trajectory[:, 2])
        self.save_flag = save_flag
        self.n_points = len(self.trajectory)
        self.current_idx = 0
        self.at_start = False
        self.t_start_tracking = 0.0
        self.start_position_tolerance = 0.1
        self.position_tolerance = 0.05
        self.yaw_tolerance = 0.1
        self.prev_yaw = None
        self.prev_command_time = None
        self.vx_max, self.vy_max, self.wz_max = 0.5, 0.3, 1.5
        self.kp_pos, self.kp_yaw, self.kd_yaw = 2.0, 1.5, 0.3
        self.time_traj, self.state_traj, self.cmd_traj = [], [], []
        with open(config_file_name) as config_file:
            self.config_data = json.load(config_file)
        self.IP_server = self.config_data['QUALISYS']['IP_MOCAP_SERVER']
        import robot_interface as sdk
        self.udp = sdk.UDP(0xee, 8080, '192.168.123.161', 8082)
        self.cmd = sdk.HighCmd()
        self.udp.InitCmdData(self.cmd)

    def compute_velocity_command(self, current_pos, current_yaw, t_elapsed):
        current_pos = np.asarray(current_pos, dtype=float)
        if current_pos.shape != (2,) or not np.isfinite(current_pos).all() or not np.isfinite([current_yaw, t_elapsed]).all():
            raise ValueError('pose and time must be finite')
        if self.prev_command_time is not None and t_elapsed < self.prev_command_time:
            raise ValueError('control time must be monotonic')
        if not self.at_start:
            target_pos = self.trajectory[0, :2]
            error = target_pos - current_pos
            distance = np.linalg.norm(error)
            yaw_ready = (not self.use_yaw or
                         abs(angle_error(self.trajectory[0, 2], current_yaw)) <= self.yaw_tolerance)
            if distance < self.start_position_tolerance and yaw_ready:
                self.at_start = True
                self.t_start_tracking = t_elapsed
                self.prev_yaw = current_yaw
                self.prev_command_time = t_elapsed
                print('Reached start pose, beginning trajectory tracking')
                return 0.0, 0.0, 0.0
            # No discontinuous three-second callback sleep or early path clock.
            world = error * min(self.kp_pos, 0.3 / max(distance, 1e-12))
            yaw_target = self.trajectory[0, 2] if self.use_yaw else current_yaw
        else:
            elapsed = max(0.0, t_elapsed - self.t_start_tracking)
            index = min(elapsed / self.dt, self.n_points - 1)
            lower = min(int(index), self.n_points - 2)
            fraction = index - lower
            target = ((1 - fraction) * self.trajectory[lower] +
                      fraction * self.trajectory[lower + 1])
            world = self.kp_pos * (target[:2] - current_pos)
            yaw_target = target[2] if self.use_yaw else current_yaw
        vx = world[0] * np.cos(current_yaw) + world[1] * np.sin(current_yaw)
        vy = -world[0] * np.sin(current_yaw) + world[1] * np.cos(current_yaw)
        yaw_rate = 0.0
        if (self.prev_yaw is not None and self.prev_command_time is not None
                and t_elapsed > self.prev_command_time):
            yaw_rate = angle_error(current_yaw, self.prev_yaw) / (t_elapsed - self.prev_command_time)
        # Damping opposes measured rotation; differentiating target error with a
        # minus sign previously reversed the initial response to a heading step.
        wz = self.kp_yaw * angle_error(yaw_target, current_yaw) - self.kd_yaw * yaw_rate if self.use_yaw else 0.0
        self.prev_yaw = current_yaw
        self.prev_command_time = t_elapsed
        return (float(np.clip(vx, -self.vx_max, self.vx_max)),
                float(np.clip(vy, -self.vy_max, self.vy_max)),
                float(np.clip(wz, -self.wz_max, self.wz_max)))

    def update_progress(self, current_pos, t_elapsed):
        if self.at_start:
            self.current_idx = min(self.n_points - 1, int(max(0, t_elapsed - self.t_start_tracking) / self.dt))

    def is_complete(self, pose, t_elapsed):
        return (self.at_start and t_elapsed - self.t_start_tracking >= (self.n_points - 1) * self.dt
                and np.linalg.norm(np.asarray(pose[:2]) - self.trajectory[-1, :2]) <= self.position_tolerance
                and (not self.use_yaw or abs(angle_error(self.trajectory[-1, 2], pose[2])) <= self.yaw_tolerance))

    def save_data(self):
        if not self.save_flag or not self.time_traj:
            return
        filename = 'experiment/traj/dense_tracking_' + time.strftime('%Y%m%d%H%M%S') + '.csv'
        with open(filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(self.time_traj)
            writer.writerows(np.asarray(self.state_traj).T)
            writer.writerows(np.asarray(self.cmd_traj).T)
        print('Data saved to', filename)

    async def run(self, timeout=60.0):
        self.at_start = False
        self.current_idx = 0
        self.prev_yaw = self.prev_command_time = None
        self.time_traj, self.state_traj, self.cmd_traj = [], [], []
        async def step(pose, elapsed):
            self.update_progress(pose[:2], elapsed)
            if self.is_complete(pose, elapsed):
                return None
            command = self.compute_velocity_command(pose[:2], pose[2], elapsed)
            self.time_traj.append(elapsed)
            self.state_traj.append(pose.tolist())
            self.cmd_traj.append(command)
            return command
        try:
            return await run_pose_loop(self.udp, self.cmd, self.IP_server,
                self.config_data['QUALISYS']['NAME_SINGLE_BODY'], step,
                period=self.dt, timeout=timeout)
        finally:
            self.save_data()
