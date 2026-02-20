#!/usr/bin/env python3
import sys
import time
import math
import numpy as np
import joblib
import os

sys.path.append(os.getcwd()+'/externals/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

if __name__ == '__main__':
    d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11}
    
    PosStopF = math.pow(10, 9)
    VelStopF = 16000.0
    LOWLEVEL = 0xff
    
    torques = joblib.load('torque.pkl')
    
    udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)
    
    motiontime = 0
    step_idx = 0
    num_steps = torques.shape[1]
    
    while step_idx < num_steps:
        time.sleep(0.002)
        motiontime += 1
        
        udp.Recv()
        udp.GetRecv(state)
        
        if motiontime >= 500:
            for joint_idx in range(12):
                torque = np.clip(torques[joint_idx, step_idx], -5.0, 5.0)
                cmd.motorCmd[joint_idx].q = PosStopF
                cmd.motorCmd[joint_idx].dq = VelStopF
                cmd.motorCmd[joint_idx].Kp = 0
                cmd.motorCmd[joint_idx].Kd = 0
                cmd.motorCmd[joint_idx].tau = torque
            
            if motiontime % 100 == 0:
                step_idx += 1
        
        if motiontime > 10:
            safe.PowerProtect(cmd, state, 1)
        
        udp.SetSend(cmd)
        udp.Send()
