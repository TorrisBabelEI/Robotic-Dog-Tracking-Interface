#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd()+'/src')
sys.path.append('externals/unitree_legged_sdk/lib/python/amd64')
from DogSys import DogSys
from OptimalControl import OptimalControl
import json
import csv

import asyncio
from go1_highlevel_runtime import run_pose_loop, TrackingStopped


class ModelPredictiveControl:
    configDict: dict  # a dictionary for parameters

    def __init__(self, configDict: dict, buildFlag=True, waypoints=[[0,0]], saveFlag=False, config_file_name='experiment/config/config_dog.json'):
        self.configDict = configDict
        self.dt = float(self.configDict['dt'])
        points = np.asarray(waypoints, dtype=float)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('dt must be finite and positive')
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
            raise ValueError('waypoints must be a nonempty finite N x 2 array')
        if configDict.get('method', 'MPC') != 'MPC':
            raise ValueError('only MPC is supported')
        self.saveFlag = saveFlag

        # CasADi generates libraries into this local build directory.
        if buildFlag:
            os.makedirs('build', exist_ok=True)

        # initialize DogSys
        self.MyDogSys = DogSys(configDict, buildFlag)
        self.waypoints = points.tolist()
        self.offset = 0.1
        # initialize OptimalControlProblem
        self.MyDogOc = OptimalControl(configDict, self.MyDogSys, buildFlag)
        
        self.numWaypoints = len(self.waypoints)

        # Read the configuration from the json file
        with open(config_file_name) as config_file:
            self.config_data = json.load(config_file)
        self.IP_server = self.config_data["QUALISYS"]["IP_MOCAP_SERVER"]

        # Initialize dog
        HIGHLEVEL = 0xee

        import robot_interface as sdk
        self.udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

        self.cmd = sdk.HighCmd()
        self.udp.InitCmdData(self.cmd)

    async def run(self, iniState: np.array, timeTotal: float):
        """Use measured pose for every solve; iniState is a legacy API argument."""
        self.reached = 0
        self.time_name = time.strftime("%Y%m%d%H%M%S")
        self.timeTraj = []
        self.xTraj = []
        self.uTraj = []
        self.ipoptTimeTraj = []
        self.algTimeTraj = []
        self.logTimeTraj = []
        self.logStrTraj = []

        async def step(pose, elapsed):
            self.stateNow = pose.copy()
            self.timeNow = elapsed
            while (self.reached < self.numWaypoints and
                   np.linalg.norm(pose[:2] - self.waypoints[self.reached]) <= self.offset):
                self.reached += 1
                print('Reached waypoint', self.reached)
            if self.reached == self.numWaypoints:
                return None
            self.waypoint = self.waypoints[self.reached]
            # The solver never sends commands. Its deadline is enforced by the
            # runtime while the event loop continues to receive/validate poses.
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._runOC, pose.copy(), elapsed, self.waypoint)
            ipopt_time, status, success, algo_time, _, command = result
            if not success:
                self.logTimeTraj.append(elapsed)
                self.logStrTraj.append(status)
                raise TrackingStopped('MPC solve failed: ' + status)
            command = np.asarray(command, dtype=float)
            lower = np.array([self.MyDogOc.lin_vel_x_lb, self.MyDogOc.lin_vel_y_lb,
                              self.MyDogOc.ang_vel_lb])
            upper = np.array([self.MyDogOc.lin_vel_x_ub, self.MyDogOc.lin_vel_y_ub,
                              self.MyDogOc.ang_vel_ub])
            if (command.shape != (3,) or not np.isfinite(command).all() or
                    np.any(command < lower - 1e-6) or np.any(command > upper + 1e-6)):
                raise TrackingStopped('MPC returned an invalid or out-of-bounds command')
            command = np.clip(command, lower, upper)
            self.timeTraj.append(elapsed)
            self.xTraj.append(pose.tolist())
            self.uTraj.append(command.tolist())
            self.ipoptTimeTraj.append(ipopt_time)
            self.algTimeTraj.append(algo_time)
            return command

        try:
            return await run_pose_loop(
                self.udp, self.cmd, self.IP_server,
                self.config_data['QUALISYS']['NAME_SINGLE_BODY'], step,
                period=self.dt, timeout=timeTotal)
        finally:
            # run_pose_loop transmits the stop before saving any data.
            self.timeTraj = np.asarray(self.timeTraj)
            self.xTraj = np.asarray(self.xTraj).reshape(-1, 3)
            self.uTraj = np.asarray(self.uTraj).reshape(-1, 3)
            if self.saveFlag and len(self.timeTraj):
                with open('experiment/traj/' + self.time_name + '.csv', 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.timeTraj)
                    writer.writerows(self.xTraj.T)
                    writer.writerows(self.uTraj.T)

    def _runOC(self, stateNow, timeNow, waypoint):
        t0 = time.perf_counter()
        xTrajNow, uTrajNow, timeTrajNow, ipoptTime, returnStatus, successFlag = self.MyDogOc.solve(stateNow, timeNow, waypoint)
        t1 = time.perf_counter()
        algoTime = t1 - t0
        print_str = "Sim time [sec]: " + str(round(timeNow, 1)) + "   Comp. time [sec]: " + str(round(algoTime, 3))
        # apply control
        uNow = uTrajNow[0, :]
        return ipoptTime, returnStatus, successFlag, algoTime, print_str, uNow


    def visualize(self, result: dict, matFlag=False, legendFlag=True, titleFlag=True, blockFlag=True):
        """
        If result is loaded from a .mat file, matFlag = True
        If result is from a seld-defined dict variable, matFlag = False
        """
        timeTraj = result["timeTraj"]
        xTraj = result["xTraj"]
        uTraj = result["uTraj"]

        if matFlag:
            timeTraj = timeTraj[0, 0].flatten()
            xTraj = xTraj[0, 0]
            uTraj = uTraj[0, 0]

        # trajectories for states
        fig1, ax1 = plt.subplots(2, 1)
        if titleFlag:
            fig1.suptitle("State Trajectory")
        ax1[0].plot(xTraj[:,0], xTraj[:,1], color="blue", linewidth=2)

        ax1[0].set_xlabel("x [m]")
        ax1[0].set_ylabel("y [m]")

        ax1[1].plot(timeTraj, xTraj[:,0], color="blue", linewidth=2)
        ax1[1].plot(timeTraj, xTraj[:,1], color="red", linewidth=2)
        ax1[1].set_xlabel("time [sec]")
        ax1[1].set_ylabel("x,y [m]")

        # trajectories for inputs
        fig2, ax2 = plt.subplots(2, 1)
        if titleFlag:
            fig2.suptitle("Input Trajectory")
        # trajectory for current
        ax2[0].plot(timeTraj, uTraj[:,0], color="blue", linewidth=2)
        # for input bounds
        ax2[0].plot(timeTraj,
            self.MyDogOc.lin_vel_x_lb*np.ones(timeTraj.size),
            color="black", linewidth=2, linestyle="dashed")
        ax2[0].plot(timeTraj,
            self.MyDogOc.lin_vel_x_ub*np.ones(timeTraj.size),
            color="black", linewidth=2, linestyle="dashed")
        # ax2[0].set_xlabel("time [sec]")
        ax2[0].set_ylabel(r'$v \  [\mathrm{m/s}]$')

        ax2[1].plot(timeTraj, uTraj[:,2], color="blue", linewidth=2)
        # for input bounds
        ax2[1].plot(timeTraj,
            self.MyDogOc.ang_vel_lb*np.ones(timeTraj.size),
            color="black", linewidth=2, linestyle="dashed")
        ax2[1].plot(timeTraj,
            self.MyDogOc.ang_vel_ub*np.ones(timeTraj.size),
            color="black", linewidth=2, linestyle="dashed")
        ax2[1].set_xlabel("time [sec]")
        ax2[1].set_ylabel(r'$w\  [\mathrm{rad/s}]$')
        plt.tight_layout()
        plt.show(block=blockFlag)


if __name__ == '__main__':
    # dictionary for configuration
    # dt for Euler integration
    configDict = {"dt": 0.1, "stepNumHorizon": 5, "startPointMethod": "zeroInput"}
    config_file_name = 'experiment/config/config_dog.json'

    buildFlag = True
    saveFlag = False

    

    x0 = np.array([0, 0, 0])
    u0 = np.array([0, 0, 0])
    T = 20
    waypoints = [[-0.,0.], [1,-0.5], [2,0]]

    # initialize MPC
    MyMPC = ModelPredictiveControl(configDict, buildFlag, waypoints, saveFlag, config_file_name)

    # Run our asynchronous main function forever
    asyncio.run(MyMPC.run(x0, T))

