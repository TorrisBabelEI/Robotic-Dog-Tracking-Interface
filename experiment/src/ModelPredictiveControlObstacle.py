#!/usr/bin/env python3
"""
MPC with rectangular obstacle avoidance + bi-directional trust blending.
Dog input space: [vx, vy, wz] (body frame).
Trust dynamics ported from Bi-directional_Trust project.
"""
import time
import copy
import numpy as np
import casadi as ca
import asyncio
import xml.etree.ElementTree as ET
import json
import csv
import transforms3d
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd() + '/src')
sys.path.append('externals/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk
import qtm

# ── Trust constants (from utils_trust_update.py) ─────────────────────────────
_B_HR = 0.2
_B_RH = 0.5   # faster recovery: T_rh was recovering too slowly vs T_hr
_W_SAFETY = 1.0
_W_COST = 0.05
_COST_BASELINE = 10.0

def _update_T_rh(T_rh, min_clearance, obstacle_margin, mpc_cost):
    P_h = 0.0
    if min_clearance < 0:
        P_h += _W_SAFETY * (-0.5)
    elif min_clearance < obstacle_margin:
        P_h += _W_SAFETY * (-0.2)
    elif min_clearance < 2 * obstacle_margin:
        P_h += _W_SAFETY * (-0.05)
    else:
        P_h += _W_SAFETY * 0.01
    if mpc_cost is not None:
        r = mpc_cost / (_COST_BASELINE + 1e-6)
        P_h += _W_COST * (-0.1 if r > 2.0 else (-0.02 * (r - 1.0) if r > 1.0 else 0.005 * (1.0 - r)))
    return float(np.clip(T_rh + _B_RH * P_h, 0.0, 1.0))

def _update_T_hr(T_hr, human_override):
    return float(np.clip(T_hr + _B_HR * (-0.01 if human_override else 0.005), 0.0, 1.0))

def _point_to_rect_dist(pos, rect):
    """Returns signed distance (negative = inside)."""
    x_min, x_max, y_min, y_max = rect
    cx = np.clip(pos[0], x_min, x_max)
    cy = np.clip(pos[1], y_min, y_max)
    d = np.linalg.norm(pos - np.array([cx, cy]))
    if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
        d = -min(pos[0]-x_min, x_max-pos[0], pos[1]-y_min, y_max-pos[1])
    return d

# ── MPC with obstacle avoidance ───────────────────────────────────────────────
class MPCObstacle:
    """
    Unicycle MPC for dog: state [x, y, theta], input [vx, vy, wz] (body frame).
    Rectangular obstacle avoidance via smooth log-sum-exp constraints.
    """
    def __init__(self, dt=0.2, N=10,
                 vx_max=0.25, vy_max=0.15, wz_max=1.0,
                 obstacle_margin=0.35,
                 obstacles=None, box_limit=None):
        self.dt = dt
        self.N = N
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.wz_max = wz_max
        self.obstacle_margin = obstacle_margin
        self.obstacles = obstacles or []
        self.box_limit = box_limit  # [x_min, x_max, y_min, y_max]
        self.solver = None
        self.warm_start = None
        self._build()

    def _build(self):
        N, dt = self.N, self.dt
        # Decision vars: states X(3 x N+1), inputs U(3 x N)
        X = ca.MX.sym('X', 3, N + 1)
        U = ca.MX.sym('U', 3, N)
        P = ca.MX.sym('P', 5)  # [x0, y0, th0, xg, yg]

        cost = 0.0
        Q, Qt, Rv, Rw, Rdu = 5.0, 50.0, 0.5, 0.1, 1.0
        for k in range(N):
            dx = X[0, k] - P[3]; dy = X[1, k] - P[4]
            cost += Q * (dx**2 + dy**2)
            cost += Rv * (U[0, k]**2 + U[1, k]**2) + Rw * U[2, k]**2
            if k > 0:
                # penalise control rate: discourages sign-flipping between steps
                cost += Rdu * ((U[0,k]-U[0,k-1])**2 + (U[1,k]-U[1,k-1])**2
                               + Rdu * (U[2,k]-U[2,k-1])**2)
        dx = X[0, N] - P[3]; dy = X[1, N] - P[4]
        cost += Qt * (dx**2 + dy**2)

        g, lbg, ubg = [], [], []

        # Initial state
        g.append(X[:, 0] - P[:3]); lbg += [0,0,0]; ubg += [0,0,0]

        # Dynamics: unicycle with body-frame [vx, vy, wz]
        for k in range(N):
            th = X[2, k]
            xn = X[0,k] + (U[0,k]*ca.cos(th) - U[1,k]*ca.sin(th)) * dt
            yn = X[1,k] + (U[0,k]*ca.sin(th) + U[1,k]*ca.cos(th)) * dt
            tn = X[2,k] + U[2,k] * dt
            g += [X[0,k+1]-xn, X[1,k+1]-yn, X[2,k+1]-tn]
            lbg += [0,0,0]; ubg += [0,0,0]

        # Obstacle avoidance (smooth log-sum-exp, one constraint per obstacle per step)
        alpha = 10.0
        for k in range(N + 1):
            for obs in self.obstacles:
                x_min, x_max, y_min, y_max = obs
                m = self.obstacle_margin
                ox0, ox1 = x_min - m, x_max + m
                oy0, oy1 = y_min - m, y_max + m
                px, py = X[0, k], X[1, k]
                # smooth_min of distances to each edge (must be >= 0 to be outside)
                d_l = ox0 - px; d_r = px - ox1
                d_b = oy0 - py; d_t = py - oy1
                smooth = -(1.0/alpha) * ca.log(
                    ca.exp(-alpha*(-d_l)) + ca.exp(-alpha*(-d_r)) +
                    ca.exp(-alpha*(-d_b)) + ca.exp(-alpha*(-d_t))
                )
                g.append(smooth); lbg.append(-ca.inf); ubg.append(0)

        # Variable bounds
        lbx, ubx = [], []
        bl = self.box_limit if self.box_limit else [-ca.inf]*4
        for _ in range(N + 1):
            lbx += [bl[0], bl[2], -4*np.pi]
            ubx += [bl[1], bl[3],  4*np.pi]
        for _ in range(N):
            lbx += [-self.vx_max, -self.vy_max, -self.wz_max]
            ubx += [ self.vx_max,  self.vy_max,  self.wz_max]

        z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {'x': z, 'f': cost, 'g': ca.vertcat(*g), 'p': P}
        opts = {
            'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0,
            'ipopt.max_iter': 200, 'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4, 'ipopt.acceptable_tol': 1e-3,
        }
        self.solver = ca.nlpsol('mpc_obs', 'ipopt', nlp, opts)
        self.lbx, self.ubx = lbx, ubx
        self.lbg, self.ubg = lbg, ubg
        self.n_states = 3 * (N + 1)
        self.n_controls = 3 * N

    def solve(self, state, goal):
        """
        Returns (u [vx,vy,wz], solve_time, cost) or zeros on failure.
        """
        p = np.array([state[0], state[1], state[2], goal[0], goal[1]])
        if self.warm_start is not None:
            x0 = self.warm_start
        else:
            x0 = np.zeros(self.n_states + self.n_controls)
            for k in range(self.N + 1):
                f = k / self.N
                x0[3*k]   = state[0] + f*(goal[0]-state[0])
                x0[3*k+1] = state[1] + f*(goal[1]-state[1])
                x0[3*k+2] = state[2]
        t0 = time.time()
        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx,
                              lbg=self.lbg, ubg=self.ubg, p=p)
            xopt = sol['x'].full().flatten()
            cost = float(sol['f'])
            # warm-start shift
            ws = np.zeros_like(x0)
            for k in range(self.N):
                ws[3*k:3*k+3] = xopt[3*(k+1):3*(k+1)+3]
            ws[3*self.N:3*self.N+3] = xopt[3*self.N:3*self.N+3]
            cs = self.n_states
            for k in range(self.N - 1):
                ws[cs+3*k:cs+3*k+3] = xopt[cs+3*(k+1):cs+3*(k+1)+3]
            ws[cs+3*(self.N-1):cs+3*self.N] = xopt[cs+3*(self.N-1):cs+3*self.N]
            self.warm_start = ws
            u = xopt[self.n_states:self.n_states+3]
            return u, time.time()-t0, cost
        except Exception as e:
            print(f"MPC solve failed: {e}")
            return np.zeros(3), time.time()-t0, None


# ── Main controller class ─────────────────────────────────────────────────────
class ModelPredictiveControlObstacle:
    """
    Online MPC + bi-directional trust shared control for Unitree Go1.
    Advances through sequential goal states; human can interrupt via joystick.
    """

    def __init__(self, mpc_config, waypoints, obstacles, box_limit,
                 joystick_handler=None,
                 save_flag=False,
                 config_file='experiment/config/config_dog.json'):
        """
        Args:
            mpc_config: dict with keys dt, N, vx_max, vy_max, wz_max, obstacle_margin
            waypoints:  list of [x, y] goal states
            obstacles:  list of [x_min, x_max, y_min, y_max]
            box_limit:  [x_min, x_max, y_min, y_max]
            joystick_handler: JoystickHandler instance (or None for MPC-only)
            save_flag:  save trajectory CSV
            config_file: path to config_dog.json
        """
        self.waypoints = waypoints
        self.obstacles = obstacles
        self.save_flag = save_flag
        self.joystick = joystick_handler
        self.offset = 0.15  # waypoint reach radius [m]

        with open(config_file) as f:
            cfg = json.load(f)
        self.IP_server = cfg['QUALISYS']['IP_MOCAP_SERVER']
        self.body_name = cfg['QUALISYS']['NAME_SINGLE_BODY']

        HIGHLEVEL = 0xee
        self.udp = sdk.UDP(HIGHLEVEL, 8080, '192.168.123.161', 8082)
        self.cmd = sdk.HighCmd()
        self.udp.InitCmdData(self.cmd)

        self.mpc = MPCObstacle(
            dt=mpc_config.get('dt', 0.2),
            N=mpc_config.get('N', 10),
            vx_max=mpc_config.get('vx_max', 0.25),
            vy_max=mpc_config.get('vy_max', 0.15),
            wz_max=mpc_config.get('wz_max', 1.0),
            obstacle_margin=mpc_config.get('obstacle_margin', 0.35),
            obstacles=obstacles,
            box_limit=box_limit,
        )

        # Trust state
        self.T_hr = 0.5
        self.T_rh = 0.5
        self.alpha = 1.0  # start robot-dominant

        # Logging
        self.time_traj, self.x_traj, self.u_traj = [], [], []
        self.alpha_traj, self.T_hr_traj, self.T_rh_traj = [], [], []

    @staticmethod
    def _parse_body_index(xml_string):
        xml = ET.fromstring(xml_string)
        return {b.text.strip(): i for i, b in enumerate(xml.findall('*/Body/Name'))}

    def _send(self, vx, vy, wz):
        self.cmd.mode = 2
        self.cmd.gaitType = 1
        self.cmd.velocity = [float(vx), float(vy)]
        self.cmd.yawSpeed = float(wz)
        self.cmd.bodyHeight = 0
        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def _stop(self):
        self.cmd.mode = 0
        self.cmd.velocity = [0, 0]
        self.cmd.yawSpeed = 0.0
        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def _min_clearance(self, pos):
        return min(_point_to_rect_dist(pos, obs) for obs in self.obstacles)

    def _save(self):
        if not self.save_flag or not self.time_traj:
            return
        fname = f"experiment/traj/joint_control_{time.strftime('%Y%m%d%H%M%S')}.csv"
        with open(fname, 'w') as f:
            w = csv.writer(f)
            w.writerow(self.time_traj)
            w.writerow([s[0] for s in self.x_traj])
            w.writerow([s[1] for s in self.x_traj])
            w.writerow([s[2] for s in self.x_traj])
            w.writerow([u[0] for u in self.u_traj])
            w.writerow([u[1] for u in self.u_traj])
            w.writerow([u[2] for u in self.u_traj])
            w.writerow(self.alpha_traj)
            w.writerow(self.T_hr_traj)
            w.writerow(self.T_rh_traj)
        print(f"Saved to {fname}")
        self._plot()

    def _plot(self):
        if not self.time_traj:
            return
        t = np.array(self.time_traj)
        alpha = np.array(self.alpha_traj)
        T_hr  = np.array(self.T_hr_traj)
        T_rh  = np.array(self.T_rh_traj)
        x_traj = np.array(self.x_traj)

        # Trust / alpha over time
        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        axes[0].plot(t, alpha, label='alpha (robot share)', linewidth=2)
        axes[0].plot(t, T_hr,  label='T_hr (human→robot)',  linewidth=1.5)
        axes[0].plot(t, T_rh,  label='T_rh (robot→human)',  linewidth=1.5)
        axes[0].set_ylabel('Trust / Alpha')
        axes[0].set_title('Bi-directional Trust Dynamics')
        axes[0].legend(); axes[0].grid(True)

        axes[1].set_ylabel('Alpha (robot share)')
        axes[1].set_xlabel('Time (s)')
        axes[1].plot(t, alpha, color='steelblue', linewidth=2)
        axes[1].set_ylim(-0.05, 1.05); axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(f"experiment/traj/joint_control_trust_{time.strftime('%Y%m%d%H%M%S')}.png",
                    dpi=120)
        plt.show()

        # Trajectory
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(x_traj[:, 0], x_traj[:, 1], 'b-', linewidth=2, label='Robot path')
        ax2.scatter(x_traj[0, 0],  x_traj[0, 1],  c='green', s=100, marker='o', label='Start')
        ax2.scatter(x_traj[-1, 0], x_traj[-1, 1], c='red',   s=100, marker='x', label='End')
        for i, wp in enumerate(self.waypoints):
            ax2.scatter(wp[0], wp[1], c='lime', s=80, marker='*', zorder=5,
                        label='Waypoints' if i == 0 else None)
            ax2.annotate(str(i + 1), xy=(wp[0], wp[1]),
                         xytext=(6, 6), textcoords='offset points', fontsize=10, fontweight='bold')
        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = obs
            ax2.add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                        color='gray', alpha=0.7))
        ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
        ax2.set_title('Robot Trajectory')
        ax2.legend(); ax2.axis('equal'); ax2.grid(True)
        plt.tight_layout()
        plt.savefig(f"experiment/traj/joint_control_traj_{time.strftime('%Y%m%d%H%M%S')}.png",
                    dpi=120)
        plt.show()

    async def run(self, timeout=120.0):
        connection = await qtm.connect(self.IP_server)
        if connection is None:
            print("Failed to connect to QTM")
            return
        async with qtm.TakeControl(connection, 'password'):
            pass
        xml_string = await connection.get_parameters(parameters=['6d'])
        body_index = self._parse_body_index(xml_string)

        t_start = time.time()
        reached = 0
        goal = self.waypoints[0]
        at_start = False   # True once dog has reached and aligned at WAYPOINTS[0]
        t_arrived = None   # timestamp when position+yaw both satisfied

        # Target yaw at start: face toward first loop waypoint
        wp0 = np.array(self.waypoints[0])
        wp1 = np.array(self.waypoints[1])
        start_yaw = float(np.arctan2(wp1[1] - wp0[1], wp1[0] - wp0[0]))

        def on_packet(packet):
            nonlocal reached, goal, at_start, t_arrived

            t_now = time.time() - t_start
            joystick_stopped = (self.joystick is not None and
                                not self.joystick.poll_start_stop())
            if t_now > timeout or joystick_stopped:
                self._stop()
                self._save()
                raise Exception("Done")

            # ── State from mocap ──────────────────────────────────────────
            bodies = packet.get_6d()[1]
            pos, rot = bodies[body_index[self.body_name]]
            rot_np = np.asarray(rot.matrix, dtype=float).reshape(3, 3)
            quat = transforms3d.quaternions.mat2quat(rot_np)
            _, _, yaw = transforms3d.euler.quat2euler(quat, axes='sxyz')
            yaw = -yaw
            state = np.array([pos.x/1000.0, pos.y/1000.0, yaw])

            # ── Phase 1: navigate to start (MPC), align yaw, wait 3 s ──────
            if not at_start:
                pos_ok  = np.linalg.norm(state[:2] - np.array(self.waypoints[0])) < self.offset
                yaw_err = float(np.arctan2(np.sin(start_yaw - yaw), np.cos(start_yaw - yaw)))
                yaw_ok  = abs(yaw_err) < 0.15  # ~8 deg

                if pos_ok and yaw_ok:
                    if t_arrived is None:
                        t_arrived = time.time()
                        self._stop()
                        print("At start — waiting 3 s...")
                    elif time.time() - t_arrived >= 3.0:
                        at_start = True
                        print("Starting MPC+trust loop.")
                    return

                # MPC drives to start position; add yaw correction on top
                u_init, _, _ = self.mpc.solve(state, self.waypoints[0])
                u_init[2] = float(np.clip(1.5 * yaw_err, -self.mpc.wz_max, self.mpc.wz_max))
                self._send(*u_init)
                return

            # ── Phase 2: MPC + trust loop ─────────────────────────────────
            u_mpc, _, mpc_cost = self.mpc.solve(state, goal)

            u_h = np.zeros(3)
            if self.joystick is not None:
                vx_h, vy_h, wz_h = self.joystick.get_velocities()
                u_h = np.array([vx_h, vy_h, wz_h])
            human_override = np.linalg.norm(u_h) > 1e-3

            # T_rh updates every step (safety context is always relevant)
            clearance = self._min_clearance(state[:2])
            self.T_rh = _update_T_rh(self.T_rh, clearance,
                                      self.mpc.obstacle_margin, mpc_cost)
            self.T_hr = _update_T_hr(self.T_hr, human_override)

            if not human_override:
                self.alpha = 1.0
                u_final = u_mpc
            else:
                # alpha = robot share; human share = 1 - alpha
                # When T_hr is low (human distrusted) alpha→1; when T_rh is low alpha→0
                raw_alpha = self.T_hr / (self.T_hr + self.T_rh + 1e-8)
                self.alpha = 0.05 * self.alpha + 0.95 * raw_alpha
                u_final = self.alpha * u_mpc + (1 - self.alpha) * u_h

            self._send(*u_final)

            dist = np.linalg.norm(state[:2] - np.array(goal))
            if dist < self.offset:
                reached += 1
                print(f"Reached waypoint {reached} → next: {self.waypoints[reached % len(self.waypoints)]}")
                goal = self.waypoints[reached % len(self.waypoints)]

            self.time_traj.append(t_now)
            self.x_traj.append(state.tolist())
            self.u_traj.append(u_final.tolist())
            self.alpha_traj.append(self.alpha)
            self.T_hr_traj.append(self.T_hr)
            self.T_rh_traj.append(self.T_rh)

        await connection.stream_frames(components=['6d'], on_packet=on_packet)
