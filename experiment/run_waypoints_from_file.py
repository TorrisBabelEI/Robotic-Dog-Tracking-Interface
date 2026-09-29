#!/usr/bin/env python3
"""Launch a hardware body-path controller with an explicitly selected input."""
import argparse
import asyncio
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiment/src'), str(ROOT / 'src')]
from go1_trajectory_input import prepare_trajectory


def load_trajectory(path, use_yaw=True):
    path = Path(path)
    if path.suffix.lower() == '.pkl':
        import joblib
        data = np.asarray(joblib.load(path), dtype=float)
    elif path.suffix.lower() == '.csv':
        data = np.loadtxt(path, delimiter=',', ndmin=2)
    else:
        raise ValueError('trajectory must be a row-oriented .csv or trusted .pkl file')
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError('expected x/y rows, optional yaw row, and at least two samples')
    use_yaw = use_yaw and data.shape[0] >= 3
    return data[:3 if use_yaw else 2].T, use_yaw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', required=True, type=Path,
                        help='body x/y[/yaw] rows; loading .pkl requires a trusted file')
    parser.add_argument('--mode', choices=('dense', 'mpc'), default='dense')
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument('--dt', type=float, default=0.08, help='seconds between input samples')
    timing.add_argument('--total-time', type=float, help='seconds from first to last reference sample')
    parser.add_argument('--target-waypoints', type=int)
    parser.add_argument('--max-frequency', type=float, default=20.0)
    parser.add_argument('--no-yaw', action='store_true')
    parser.add_argument('--timeout', type=float, help='whole run limit; default dense=120s, MPC=60s')
    parser.add_argument('--config', type=Path, default=ROOT / 'experiment/config/config_dog.json')
    args = parser.parse_args(argv)
    trajectory, use_yaw = load_trajectory(args.trajectory, not args.no_yaw)
    trajectory, dt = prepare_trajectory(trajectory, dt=args.dt, total_time=args.total_time,
                                        target_waypoints=args.target_waypoints,
                                        max_frequency=args.max_frequency)
    print(f'Prepared {len(trajectory)} points over {(len(trajectory)-1)*dt:.3f}s; dt={dt:.4f}s')
    print('Initial pose:', trajectory[0], 'Final pose:', trajectory[-1])
    timeout = args.timeout if args.timeout is not None else (120.0 if args.mode == 'dense' else 60.0)
    if not np.isfinite(timeout) or timeout <= 0:
        parser.error('--timeout must be finite and positive')
    if args.mode == 'dense':
        from DenseTrajectoryTracker import DenseTrajectoryTracker
        tracker = DenseTrajectoryTracker(trajectory, dt=dt, use_yaw=use_yaw,
                                          save_flag=True, config_file_name=str(args.config))
        asyncio.run(tracker.run(timeout=timeout))
    else:
        from ModelPredictiveControl import ModelPredictiveControl
        config = {'dt': 0.2, 'stepNumHorizon': 10, 'startPointMethod': 'zeroInput'}
        tracker = ModelPredictiveControl(config, True, trajectory[:, :2].tolist(),
                                         True, str(args.config))
        asyncio.run(tracker.run(np.zeros(3), timeout))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
