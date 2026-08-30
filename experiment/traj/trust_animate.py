import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FFMpegWriter

# --- Configuration ---
def get_synced_data(csv_path):
    # Load row-based data
    # Using low_memory=False to ensure large rows are handled
    raw_df = pd.read_csv(csv_path, header=None)
    raw_data = raw_df.values 
    
    # --- CORRECT ROW MAPPING ---
    t_orig        = raw_data[0] # Row 1
    alpha_orig    = raw_data[7] # Row 2
    thr_orig      = raw_data[8] # Row 3
    trh_orig      = raw_data[9] # Row 4
    
    # Target time steps for interpolation (from 0 to max simulation time)
    t_orig -= t_orig[0]  # Normalize time to start at 0
    t_sim = t_orig[-1]
    TOTAL_FRAMES = int(t_sim * FPS)
    t_steps = np.linspace(0, t_sim, TOTAL_FRAMES)
    
    # Interpolation functions
    # Linear for physical/continuous values
    f_thr = interp1d(t_orig, thr_orig, kind='linear', fill_value="extrapolate")
    f_trh = interp1d(t_orig, trh_orig, kind='linear', fill_value="extrapolate")

    # Previous (Zero-order hold) for logical/state values
    f_alpha = interp1d(t_orig, alpha_orig, kind='previous', fill_value="extrapolate")
    
    return {
        'thr': f_thr(t_steps),
        'trh': f_trh(t_steps),
        'alpha': f_alpha(t_steps),
        't_sim': t_sim,
        'total_frames': TOTAL_FRAMES
    }

# --- Execution ---
FPS = 30
data = get_synced_data('/Users/yunru/Documents/GitHub/Robotic-Dog-Tracking-Interface/experiment/traj/joint_control_20260422111243.csv') 
t_display = np.linspace(0, data['t_sim'], data['total_frames'])

# Figure setup
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.subplots_adjust(hspace=0.35)

line_thr,   = ax.plot([], [], 'tab:orange', label='$T_{hr}$', lw=2)
line_trh,   = ax.plot([], [], 'tab:green', label='$T_{rh}$', lw=2)
line_alpha, = ax.plot([], [], 'tab:blue', label='$\\alpha$', lw=1.5, alpha=0.7)

# Axis Decor
ax.set_ylim(-0.05, 1.15)
ax.set_xlabel('Time (s)', fontsize=24)
ax.set_ylabel('Trust / Alpha', fontsize=24)
ax.legend(loc='upper left', ncol=3, fontsize=20)

ax.set_xlim(0, data['t_sim'])
ax.grid(True, linestyle=':', alpha=0.5)

ax.tick_params(axis='both', labelsize=14)

# Save Video
writer = FFMpegWriter(fps=FPS)
with writer.saving(fig, "/Users/yunru/Documents/GitHub/Robotic-Dog-Tracking-Interface/experiment/traj/MPC_with_Correct_Overrides.mp4", dpi=150):
    for i in range(1, data['total_frames'] + 1):
        line_thr.set_data(t_display[:i], data['thr'][:i])
        line_trh.set_data(t_display[:i], data['trh'][:i])
        line_alpha.set_data(t_display[:i], data['alpha'][:i])
        writer.grab_frame()

print("Animation generated successfully.")