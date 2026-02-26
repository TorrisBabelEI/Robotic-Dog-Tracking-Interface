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
    
    # Time configuration: set EITHER total_time OR dt
    total_time = 15  # Total seconds to complete trajectory (set to None to use dt instead)
    dt = 0.1          # Time step per waypoint (only used if total_time is None)
    
    # Downsampling: auto-downsample if sending frequency too high (>10Hz), or specify target waypoints
    target_waypoints = None  # Set to desired number to resample trajectory to exact count
    max_frequency = 20.0     # Auto-downsample if frequency exceeds this (Hz)
    
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

    print(f"Loaded trajectory with {len(trajectory)} points")
    
    # Unified downsampling: respect both target_waypoints and max_frequency
    effective_dt = dt if total_time is None else (total_time / len(trajectory))
    current_freq = 1.0 / effective_dt
    
    # Determine target based on constraints
    n_target = len(trajectory)
    
    if target_waypoints is not None and target_waypoints < n_target:
        n_target = target_waypoints
    if current_freq > max_frequency:
        n_max_freq = int(len(trajectory) * max_frequency / current_freq)
        n_target = min(n_target, n_max_freq)

    if n_target < 2:
        raise ValueError("Trajectory must have at least 2 waypoints.")
    
    # Resample with first and last waypoints always included
    if n_target < len(trajectory):
        indices = np.round(np.linspace(0, len(trajectory)-1, n_target)).astype(int)
        trajectory = trajectory[indices]

        # Print downsampling info
        print(f"Downsampled to {len(trajectory)} points.")
    
    # Calculate dt from total_time if specified
    if total_time is not None:
        dt = total_time / len(trajectory)
        print(f"Total time: {total_time:.1f}s → dt={dt:.3f}s ({1/dt:.1f}Hz)")
    else:
        print(f"Using dt={dt}s → Total time: {len(trajectory) * dt:.1f}s ({1/dt:.1f}Hz)")

    if mode == 'dense':
        # Dense trajectory tracking
        saveFlag = True
        tracker = DenseTrajectoryTracker(trajectory, dt=dt, use_yaw=use_yaw, 
                                        save_flag=saveFlag, config_file_name=config_file_name)
        asyncio.ensure_future(tracker.run(timeout=120))
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