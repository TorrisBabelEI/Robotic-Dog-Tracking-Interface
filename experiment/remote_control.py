#!/usr/bin/env python3
import asyncio
import json
import sys
import os
sys.path.append(os.getcwd()+'/experiment/src')

from keyboard_handler import KeyboardHandler
import externals.unitree_legged_sdk as sdk
import qtm

class RemoteController:
    def __init__(self, config_file_name='experiment/config/config_dog.json'):
        # Load configuration
        with open(config_file_name) as json_file:
            self.config_data = json.load(json_file)
        self.IP_server = self.config_data["QUALISYS"]["IP_MOCAP_SERVER"]
        
        # Initialize dog communication
        HIGHLEVEL = 0xee
        self.udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
        self.cmd = sdk.HighCmd()
        self.udp.InitCmdData(self.cmd)
        
        # Control parameters
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.5  # rad/s
        
        # Initialize keyboard handler
        self.keyboard = KeyboardHandler()
        self.keyboard.start()
    
    def send_command(self, vx, vy, vyaw):
        self.cmd.mode = 2  # Walk mode
        self.cmd.gaitType = 1
        self.cmd.velocity = [vx, vy]
        self.cmd.yawSpeed = vyaw
        self.cmd.bodyHeight = 0
        
        self.udp.SetSend(self.cmd)
        self.udp.Send()
    
    def stop_robot(self):
        self.cmd.mode = 0  # Idle mode
        self.cmd.velocity = [0, 0]
        self.cmd.yawSpeed = 0
        self.udp.SetSend(self.cmd)
        self.udp.Send()
    
    async def run(self):
        print("Remote Control Started")
        print("Controls:")
        print("  Arrow Up/Down: Forward/Backward")
        print("  Arrow Left/Right: Turn Left/Right")
        print("  A/D: Strafe Left/Right")
        print("  ESC: Stop and Exit")
        
        connection = await qtm.connect(self.IP_server)
        if connection is None:
            print("Failed to connect to motion capture system")
            return
        
        async with qtm.TakeControl(connection, "password"):
            pass
        
        while self.keyboard.is_active():
            vx, vy, vyaw = self.keyboard.get_velocities(self.linear_speed, self.angular_speed)
            self.send_command(vx, vy, vyaw)
            await asyncio.sleep(0.05)  # 20Hz control loop
        
        self.stop_robot()
        self.keyboard.stop()
        print("Remote control stopped")

if __name__ == '__main__':
    controller = RemoteController()
    asyncio.run(controller.run())
