#!/usr/bin/env python3
import numpy as np
import asyncio
import time
import copy
import json
import csv
import xml.etree.ElementTree as ET
import transforms3d
import sys
import os
sys.path.append(os.getcwd()+'/externals/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

try:
    import qtm
except:
    print("Warning: qtm module not found. Motion capture features will not work.")


class DenseTrajectoryTracker:
    def __init__(self, trajectory, dt=0.02, use_yaw=False, save_flag=False, config_file_name='experiment/config/config_dog.json'):
        """
        Args:
            trajectory: numpy array of shape (n_points, 2) for [x, y] or (n_points, 3) for [x, y, yaw]
            dt: control loop time step in seconds
            use_yaw: whether to track yaw (requires trajectory with 3 columns)
            save_flag: whether to save trajectory data to CSV
        """
        self.trajectory = np.array(trajectory)
        self.dt = dt
        self.use_yaw = use_yaw
        self.save_flag = save_flag
        
        if self.use_yaw and self.trajectory.shape[1] < 3:
            raise ValueError("use_yaw=True requires trajectory with 3 columns [x, y, yaw]")
        
        self.n_points = len(self.trajectory)
        self.current_idx = 0
        self.at_start = False
        
        # Read config
        with open(config_file_name) as f:
            self.config_data = json.load(f)
        self.IP_server = self.config_data["QUALISYS"]["IP_MOCAP_SERVER"]
        
        # Initialize dog
        HIGHLEVEL = 0xee
        self.udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
        self.cmd = sdk.HighCmd()
        self.udp.InitCmdData(self.cmd)
        
        # Velocity limits
        self.vx_max = 0.5
        self.vy_max = 0.3
        self.wz_max = 1.5
        
        # Lookahead for velocity computation
        self.lookahead_steps = max(1, int(0.5 / dt))
        
        # Data logging
        self.time_traj = []
        self.state_traj = []  # [x, y, yaw]
        self.cmd_traj = []    # [vx, vy, wz]

    def create_body_index(self, xml_string):
        xml = ET.fromstring(xml_string)
        body_to_index = {}
        for index, body in enumerate(xml.findall("*/Body/Name")):
            body_to_index[body.text.strip()] = index
        return body_to_index

    def compute_velocity_command(self, current_pos, current_yaw):
        """Compute velocity command based on current position and trajectory lookahead"""
        # Go to start position first
        if not self.at_start:
            target_pos = self.trajectory[0, :2]
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < 0.1:
                self.at_start = True
                print("Reached start position, beginning trajectory tracking")
                return 0.0, 0.0, 0.0
            
            # Move to start with moderate speed
            vx_world = dx / dist * 0.3
            vy_world = dy / dist * 0.3
            
            vx_body = vx_world * np.cos(current_yaw) + vy_world * np.sin(current_yaw)
            vy_body = -vx_world * np.sin(current_yaw) + vy_world * np.cos(current_yaw)
            
            wz = 0.0
            if self.use_yaw:
                target_yaw = self.trajectory[0, 2]
                yaw_error = target_yaw - current_yaw
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
                wz = np.clip(yaw_error * 2.0, -self.wz_max, self.wz_max)
            
            return float(vx_body), float(vy_body), float(wz)
        
        if self.current_idx >= self.n_points:
            return 0.0, 0.0, 0.0
        
        # Find target point with lookahead
        target_idx = min(self.current_idx + self.lookahead_steps, self.n_points - 1)
        target_pos = self.trajectory[target_idx, :2]
        
        # Compute position error in world frame
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Transform to body frame
        vx_world = dx / (self.lookahead_steps * self.dt)
        vy_world = dy / (self.lookahead_steps * self.dt)
        
        vx_body = vx_world * np.cos(current_yaw) + vy_world * np.sin(current_yaw)
        vy_body = -vx_world * np.sin(current_yaw) + vy_world * np.cos(current_yaw)
        
        # Compute yaw rate if enabled
        wz = 0.0
        if self.use_yaw and target_idx < self.n_points:
            target_yaw = self.trajectory[target_idx, 2]
            yaw_error = target_yaw - current_yaw
            # Normalize to [-pi, pi]
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            wz = yaw_error / (self.lookahead_steps * self.dt)
        
        # Clip to limits
        vx_body = np.clip(vx_body, -self.vx_max, self.vx_max)
        vy_body = np.clip(vy_body, -self.vy_max, self.vy_max)
        wz = np.clip(wz, -self.wz_max, self.wz_max)
        
        return float(vx_body), float(vy_body), float(wz)

    def update_progress(self, current_pos):
        """Update trajectory index based on proximity to waypoints"""
        if not self.at_start or self.current_idx >= self.n_points:
            return
        
        # Check if close to current target
        target_pos = self.trajectory[self.current_idx, :2]
        dist = np.linalg.norm(current_pos - target_pos)
        
        if dist < 0.1:
            self.current_idx += 1

    def save_data(self):
        """Save trajectory data to CSV file"""
        if not self.save_flag or len(self.time_traj) == 0:
            return
        
        time_name = time.strftime("%Y%m%d%H%M%S")
        filename = f'experiment/traj/dense_tracking_{time_name}.csv'
        
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.time_traj)  # time
            writer.writerow([s[0] for s in self.state_traj])  # x
            writer.writerow([s[1] for s in self.state_traj])  # y
            writer.writerow([s[2] for s in self.state_traj])  # yaw
            writer.writerow([c[0] for c in self.cmd_traj])    # vx
            writer.writerow([c[1] for c in self.cmd_traj])    # vy
            writer.writerow([c[2] for c in self.cmd_traj])    # wz
        
        print(f"Data saved to {filename}")

    async def run(self, timeout=60.0):
        """Run trajectory tracking with motion capture feedback"""
        connection = await qtm.connect(self.IP_server)
        if connection is None:
            print("Failed to connect to motion capture")
            return
        
        async with qtm.TakeControl(connection, "password"):
            pass
        
        xml_string = await connection.get_parameters(parameters=["6d"])
        body_index = self.create_body_index(xml_string)
        wanted_body = self.config_data["QUALISYS"]["NAME_SINGLE_BODY"]
        
        t_start = time.time()
        
        def on_packet(packet):
            t_now = time.time() - t_start
            
            if t_now > timeout or (self.at_start and self.current_idx >= self.n_points):
                self.cmd.mode = 0
                self.cmd.velocity = [0, 0]
                self.cmd.yawSpeed = 0.0
                self.udp.SetSend(self.cmd)
                self.udp.Send()
                self.save_data()
                print(f"Tracking complete. Points tracked: {self.current_idx}/{self.n_points}")
                raise Exception("Tracking complete")
            
            bodies = packet.get_6d()[1]
            wanted_index = body_index[wanted_body]
            position, rotation = bodies[wanted_index]
            
            rotation_np = np.asarray(rotation.matrix, dtype=float).reshape(3, 3)
            quat = transforms3d.quaternions.mat2quat(rotation_np)
            _, _, yaw_now = transforms3d.euler.quat2euler(quat, axes='sxyz')
            yaw_now = -yaw_now
            
            current_pos = np.array([position.x/1000.0, position.y/1000.0])
            
            # Update progress
            self.update_progress(current_pos)
            
            # Compute and send velocity command
            vx, vy, wz = self.compute_velocity_command(current_pos, yaw_now)
            
            self.cmd.mode = 2
            self.cmd.gaitType = 1
            self.cmd.velocity = [vx, vy]
            self.cmd.yawSpeed = wz
            self.cmd.bodyHeight = 0
            
            self.udp.SetSend(self.cmd)
            self.udp.Send()
            
            # Log data
            self.time_traj.append(t_now)
            self.state_traj.append([current_pos[0], current_pos[1], yaw_now])
            self.cmd_traj.append([vx, vy, wz])
        
        await connection.stream_frames(components=["6d"], on_packet=on_packet)
