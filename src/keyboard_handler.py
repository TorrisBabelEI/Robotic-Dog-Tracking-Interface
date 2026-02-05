"""
Simple keyboard input handler for robot remote control
"""
import pynput
from pynput.keyboard import Key

class KeyboardHandler:
    def __init__(self):
        self.pressed_keys = set()
        self.active = True
        self.listener = None
    
    def start(self):
        def on_press(key):
            if key == Key.esc:
                self.active = False
            else:
                self.pressed_keys.add(key)
        
        def on_release(key):
            self.pressed_keys.discard(key)
        
        self.listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
    
    def stop(self):
        if self.listener:
            self.listener.stop()
    
    def get_velocities(self, linear_speed=0.3, angular_speed=0.5):
        """Return (vx, vy, vyaw) based on current key states"""
        vx = vy = vyaw = 0.0
        
        if Key.up in self.pressed_keys:
            vx = linear_speed
        if Key.down in self.pressed_keys:
            vx = -linear_speed
        if Key.left in self.pressed_keys:
            vyaw = angular_speed
        if Key.right in self.pressed_keys:
            vyaw = -angular_speed
        
        # Strafe with A/D
        try:
            if any(hasattr(k, 'char') and k.char == 'a' for k in self.pressed_keys):
                vy = linear_speed
            if any(hasattr(k, 'char') and k.char == 'd' for k in self.pressed_keys):
                vy = -linear_speed
        except:
            pass
        
        return vx, vy, vyaw
    
    def is_active(self):
        return self.active
