#!/usr/bin/env python3
import sys
import time
import numpy as np
import joblib
import os

sys.path.append(os.getcwd()+'/externals/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

if __name__ == '__main__':
    HIGHLEVEL = 0xee
    
    traj_data = joblib.load('/Users/yunru/Documents/Projects/Project Wenjian/MPPI_DK_dog_exp/SavedResults/data/test_traj.pkl')
    actions = traj_data[8:11, :]
    
    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)
    
    step_idx = 0
    num_steps = actions.shape[1]
    dt = 0.02
    
    while step_idx < num_steps:
        a_fwd = float(np.clip(actions[0, step_idx], -0.5, 0.5))
        a_lat = float(np.clip(actions[1, step_idx], -0.5, 0.5))
        a_wz = float(np.clip(actions[2, step_idx], -0.5, 0.5))
        
        cmd.mode = 2
        cmd.gaitType = 1
        cmd.velocity = [a_fwd, a_lat]
        cmd.yawSpeed = a_wz
        cmd.bodyHeight = 0
        
        udp.SetSend(cmd)
        udp.Send()
        
        time.sleep(dt)
        step_idx += 1
