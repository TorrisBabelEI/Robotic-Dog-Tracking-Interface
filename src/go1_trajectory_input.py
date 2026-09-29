"""Validate and resample timed body paths without changing their duration."""
import math
import numpy as np


def prepare_trajectory(trajectory, dt=0.08, total_time=None,
                       target_waypoints=None, max_frequency=20.0):
    path = np.asarray(trajectory, dtype=float)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] not in (2, 3):
        raise ValueError('trajectory must contain at least two x/y or x/y/yaw samples')
    if not np.isfinite(path).all():
        raise ValueError('trajectory must be finite')
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(max_frequency) or max_frequency <= 0:
        raise ValueError('dt and max_frequency must be finite and positive')
    duration = (len(path) - 1) * dt if total_time is None else float(total_time)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('total_time must be finite and positive')
    count = len(path)
    if target_waypoints is not None:
        if isinstance(target_waypoints, bool) or int(target_waypoints) != target_waypoints or target_waypoints < 2:
            raise ValueError('target_waypoints must be an integer of at least two')
        count = min(count, int(target_waypoints))
    # N samples span N-1 intervals, including both endpoints.
    count = min(count, int(math.floor(duration * max_frequency + 1e-9)) + 1)
    if count < 2:
        raise ValueError('trajectory duration is too short for max_frequency')
    old_times = np.linspace(0, duration, len(path))
    new_times = np.linspace(0, duration, count)
    values = path.copy()
    if values.shape[1] == 3:
        values[:, 2] = np.unwrap(values[:, 2])
    result = np.column_stack([np.interp(new_times, old_times, values[:, i])
                              for i in range(values.shape[1])])
    return result, duration / (count - 1)
