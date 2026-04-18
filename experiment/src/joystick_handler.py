#!/usr/bin/env python3
"""
Xbox joystick handler with keyboard fallback.
Left stick  → (vx, vy),  Right stick X → wz.
Falls back to KeyboardHandler if no joystick is detected.
"""
import sys
import os
sys.path.append(os.getcwd() + '/src')

try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

from keyboard_handler import KeyboardHandler

# Axis indices for a standard Xbox controller (pygame mapping)
_AXIS_LX = 0   # left stick horizontal  → vy (strafe)
_AXIS_LY = 1   # left stick vertical    → vx (forward, inverted)
_AXIS_RX = 3   # right stick horizontal → wz (yaw)

_DEADZONE = 0.10  # ignore stick deflections below this

# Xbox button indices
_BTN_A    = 0   # start control loop
_BTN_B    = 1   # stop / emergency stop
_BTN_START = 7  # also stop


class JoystickHandler:
    def __init__(self, vx_max=0.25, vy_max=0.15, wz_max=1.0):
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.wz_max = wz_max
        self._joystick = None
        self._kb = None
        self.active = True
        self.started = False   # must press A / Space before dog moves

        if _PYGAME_OK:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._joystick = pygame.joystick.Joystick(0)
                self._joystick.init()
                print(f"Joystick: {self._joystick.get_name()}")
            else:
                print("No joystick found — falling back to keyboard.")
        else:
            print("pygame not available — falling back to keyboard.")

        if self._joystick is None:
            self._kb = KeyboardHandler()
            self._kb.start()

    def poll_start_stop(self):
        """
        Call once per control loop tick.
        Returns True if running, False if stopped.
        Joystick: A = start,  B or Start-button = stop.
        Keyboard: Space = start,  ESC = stop (ESC already handled by KeyboardHandler).
        """
        if self._joystick is not None:
            pygame.event.pump()
            if self._joystick.get_button(_BTN_A):
                self.started = True
            if self._joystick.get_button(_BTN_B) or self._joystick.get_button(_BTN_START):
                self.active = False
        else:
            # Space bar starts; ESC is already caught by KeyboardHandler
            try:
                from pynput.keyboard import Key
                if any(hasattr(k, 'char') and k.char == ' '
                       for k in self._kb.pressed_keys):
                    self.started = True
            except Exception:
                pass
        return self.active

    def get_velocities(self):
        """Returns (vx, vy, wz) scaled to dog velocity limits. Zero if not started."""
        if not self.started:
            return 0.0, 0.0, 0.0

        if self._joystick is not None:
            pygame.event.pump()
            raw_lx = self._joystick.get_axis(_AXIS_LX)
            raw_ly = self._joystick.get_axis(_AXIS_LY)
            raw_rx = self._joystick.get_axis(_AXIS_RX)

            def _apply(v, limit):
                return 0.0 if abs(v) < _DEADZONE else float(v) * limit

            vx = _apply(-raw_ly, self.vx_max)   # forward = stick up = negative Y
            vy = _apply(-raw_lx, self.vy_max)   # strafe left = positive vy
            wz = _apply(-raw_rx, self.wz_max)   # turn left = positive wz
            return vx, vy, wz
        else:
            return self._kb.get_velocities(self.vx_max, self.wz_max)

    def is_active(self):
        if self._kb is not None:
            return self._kb.is_active() and self.active
        return self.active

    def stop(self):
        if self._kb is not None:
            self._kb.stop()
        if _PYGAME_OK:
            pygame.quit()
