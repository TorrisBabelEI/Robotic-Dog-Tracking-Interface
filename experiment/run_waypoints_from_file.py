#!/usr/bin/env python3
import numpy as np
import asyncio
import os
import sys
import joblib
import pandas as pd
sys.path.append(os.getcwd()+'/experiment/src')
sys.path.append(os.getcwd()+'/src')
from ModelPredictiveControl import ModelPredictiveControl
from DenseTrajectoryTracker import DenseTrajectoryTracker

if __name__ == '__main__':
    config_file_name = 'experiment/config/config_dog.json'
    traj_file_name = 'experiment/traj/mppi_wenjian/test_global_pos.pkl'
    
    # Control mode: 'mpc' for sparse waypoints, 'dense' for dense trajectory tracking
    mode = 'dense'
    use_yaw = True  # Set to True to track yaw (requires 3rd row in data)
    
    _, suffix = os.path.splitext(traj_file_name)

    if suffix == '.pkl':
        data = joblib.load(traj_file_name)
        if use_yaw and data.shape[0] >= 3:
            trajectory = data[:3, :].T  # [x, y, yaw]
        else:
            trajectory = data[:2, :].T  # [x, y]
            use_yaw = False
    else:
        data = pd.read_csv(traj_file_name, header=None).values
        if use_yaw and data.shape[0] >= 3:
            trajectory = data[:3, :].T
        else:
            trajectory = data[:2, :].T
            use_yaw = False

    if mode == 'dense':
        # Dense trajectory tracking
        saveFlag = True
        tracker = DenseTrajectoryTracker(trajectory, dt=0.02, use_yaw=use_yaw, 
                                        save_flag=saveFlag, config_file_name=config_file_name)
        asyncio.ensure_future(tracker.run(timeout=60))
        asyncio.get_event_loop().run_forever()
    else:
        # MPC for sparse waypoints
        configDict = {"dt": 0.2, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}
        buildFlag = True
        saveFlag = True
        x0 = np.array([0, 0, 0])
        T = 60
        waypoints = trajectory[:, :2].tolist()
        
        MyMPC = ModelPredictiveControl(configDict, buildFlag, waypoints, saveFlag, config_file_name)
        asyncio.ensure_future(MyMPC.run(x0, T))
        asyncio.get_event_loop().run_forever()