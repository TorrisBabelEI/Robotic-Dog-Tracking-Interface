#!/usr/bin/env python3
"""
Joint MPC + bi-directional trust shared control — figure-8 path.
Run from project root:  python experiment/run_joint_control.py

Scenario:
  The dog autonomously loops a figure-8 around two known obstacles.
  The human can override with the joystick at any time — e.g. when the
  crossing near the centre is unexpectedly blocked.

  Joystick : A = start,  B or Start-button = stop
  Keyboard : Space = start,  ESC = stop

Environment:
  Lab bounds : x ∈ (-2.4, 2.4) m,  y ∈ (-1.8, 1.6) m
  Obstacle 1 and 2

Waypoints — only the two lobe tips; MPC routes between them online:
  Left tip  → crossing → Right tip → crossing → (repeat)
"""
import asyncio
import numpy as np
import os
import sys
sys.path.append(os.getcwd() + '/experiment/src')
sys.path.append(os.getcwd() + '/src')

from ModelPredictiveControlObstacle import ModelPredictiveControlObstacle
from joystick_handler import JoystickHandler

# ── Environment ───────────────────────────────────────────────────────────────
BOX_LIMIT = [-2.2, 2.2, -1.6, 1.4]    # [x_min, x_max, y_min, y_max] (inset from walls)
OBSTACLES = [
    [-0.90, -0.38, -0.52, -0.26],      # Obs 1: lower-centre
    [ 0.26,  0.78,  0.26,  0.52],      # Obs 2: upper-right
]

# ── Waypoints ─────────────────────────────────────────────────────────────────
# MPC navigates between them with online obstacle avoidance.
WAYPOINTS = [
    [-2.0,  -1.0],
    [ 0.2,  -0.8],
    [-0.5,   0.5],
    [ 0.8,   1.2],
    [ 1.5,   0.0],
    [-0.5,   0.5],
    [-1.5,   0.3]
    # list repeats → cycles back to index 0
]

# ── MPC configuration ─────────────────────────────────────────────────────────
MPC_CONFIG = {
    'dt':              0.2,    # [s]
    'N':               20,     # horizon steps
    'vx_max':          0.25,   # [m/s] — slow enough for human reaction
    'vy_max':          0.25,   # [m/s]
    'wz_max':          1.0,    # [rad/s]
    'obstacle_margin': 0.25,   # [m] — Go1 half-width ~0.15 m + 0.10 m buffer
}

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    config_file = 'experiment/config/config_dog.json'
    save_flag   = True
    timeout     = 200.0   # [s] max run time

    print("Initialising joystick / keyboard...")
    joystick = JoystickHandler(
        vx_max=MPC_CONFIG['vx_max'],
        vy_max=MPC_CONFIG['vy_max'],
        wz_max=MPC_CONFIG['wz_max'],
    )

    print("Building MPC solver (first run compiles CasADi NLP — ~10 s)...")
    controller = ModelPredictiveControlObstacle(
        mpc_config=MPC_CONFIG,
        waypoints=WAYPOINTS,
        obstacles=OBSTACLES,
        box_limit=BOX_LIMIT,
        joystick_handler=joystick,
        save_flag=save_flag,
        config_file=config_file,
    )

    print("\nReady.  Press A (joystick) or Space (keyboard) to start.")
    print("        Press B / Start (joystick) or ESC (keyboard) to stop.\n")

    # Block until start signal, then launch async loop
    import time
    while not joystick.started:
        joystick.poll_start_stop()
        if not joystick.is_active():
            print("Stopped before start.")
            joystick.stop()
            sys.exit(0)
        time.sleep(0.05)

    print("Starting control loop...")
    asyncio.ensure_future(controller.run(timeout=timeout))
    asyncio.get_event_loop().run_forever()
