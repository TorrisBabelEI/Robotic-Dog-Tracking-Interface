"""Offline trajectory regression tests. All SDK and QTM connections are fakes."""
import asyncio
import copy
import importlib.util
from pathlib import Path
import sys
import types
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from DenseTrajectoryTracker import DenseTrajectoryTracker
from go1_highlevel_runtime import run_pose_loop, TrackingStopped
from go1_trajectory_input import prepare_trajectory
spec = importlib.util.spec_from_file_location('hardware_mpc_under_test', ROOT / 'experiment/src/ModelPredictiveControl.py')
mpc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mpc_module)


class FakeUDP:
    def __init__(self, *args):
        self.sent = []
    def InitCmdData(self, cmd):
        pass
    def SetSend(self, cmd):
        self.staged = copy.deepcopy(vars(cmd))
    def Send(self):
        self.sent.append(self.staged)
        if hasattr(self, 'on_send'):
            self.on_send(self.staged)


def frame(number=1, x=1.0, rotation=None):
    return types.SimpleNamespace(framenumber=number, get_6d=lambda: (None, [(
        types.SimpleNamespace(x=x*1000, y=0, z=0),
        types.SimpleNamespace(matrix=tuple(np.eye(3).ravel()) if rotation is None else rotation))]))


class FakeQTM:
    def __init__(self, initial=True):
        self.initial = initial
        self.stopped = False
        self.closed = False
        self.callback = None
        self.disconnected = None
    async def connect(self, host, on_disconnect):
        self.disconnected = on_disconnect
        return self
    async def get_parameters(self, **kwargs):
        return '<QTM><Bodies><Body><Name>body</Name></Body></Bodies></QTM>'
    async def stream_frames(self, on_packet, **kwargs):
        self.callback = on_packet
        if self.initial:
            on_packet(frame())
        return 'Ok'  # Real qtm returns an acknowledgement, not the stream lifetime.
    async def stream_frames_stop(self):
        self.stopped = True
    def disconnect(self):
        self.closed = True


def dense(path, **kwargs):
    with patch.dict(sys.modules, {'robot_interface': types.SimpleNamespace(UDP=FakeUDP, HighCmd=types.SimpleNamespace)}):
        return DenseTrajectoryTracker(path, config_file_name=str(ROOT / 'experiment/config/config_dog.json'), **kwargs)


class DenseTests(unittest.TestCase):
    def test_start_does_not_sleep_or_skip_path(self):
        tracker = dense([[0, 0], [0.1, 0], [0.2, 0]], dt=0.1)
        with patch('time.sleep', side_effect=AssertionError('blocking sleep')):
            self.assertEqual(tracker.compute_velocity_command([0, 0], 0, 5), (0, 0, 0))
            self.assertAlmostEqual(tracker.compute_velocity_command([0, 0], 0, 5.1)[0], 0.2)
        self.assertEqual(tracker.t_start_tracking, 5)

    def test_start_waits_for_heading(self):
        tracker = dense([[0, 0, 0.5], [0.1, 0, 0.5]], use_yaw=True)
        command = tracker.compute_velocity_command([0, 0], 0, 0)
        self.assertFalse(tracker.at_start)
        self.assertGreater(command[2], 0)

    def test_yaw_uses_short_arc(self):
        tracker = dense([[0, 0, np.deg2rad(179)], [0, 0, np.deg2rad(-179)]], dt=1, use_yaw=True)
        tracker.at_start = True
        self.assertAlmostEqual(tracker.compute_velocity_command([0, 0], np.pi, 0.5)[2], 0, places=10)

    def test_yaw_damping_opposes_measured_rotation(self):
        tracker = dense([[0, 0, 0.5], [0, 0, 0.5]], dt=1, use_yaw=True)
        tracker.at_start = True
        self.assertGreater(tracker.compute_velocity_command([0, 0], 0, 0)[2], 0)
        command = tracker.compute_velocity_command([0, 0], 0.1, 0.1)
        self.assertAlmostEqual(command[2], 1.5*0.4 - 0.3*1.0)

    def test_body_frame_and_bounds(self):
        tracker = dense([[1, 0], [1, 0]], dt=1)
        tracker.at_start = True
        vx, vy, _ = tracker.compute_velocity_command([0, 0], np.pi/2, 0.1)
        self.assertAlmostEqual(vx, 0)
        self.assertEqual(vy, -0.3)

    def test_endpoint_requires_actual_arrival(self):
        tracker = dense([[0, 0], [1, 0]], dt=1)
        tracker.at_start = True
        self.assertFalse(tracker.is_complete([0.5, 0, 0], 3))
        self.assertGreater(tracker.compute_velocity_command([0.5, 0], 0, 3)[0], 0)
        self.assertTrue(tracker.is_complete([1, 0, 0], 3))

    def test_invalid_input_rejected_before_sdk(self):
        for path, dt in [([[0, 0]], 1), ([[0, 0], [np.nan, 0]], 1), ([[0, 0], [1, 0]], 0)]:
            with self.subTest(path=path, dt=dt), self.assertRaises(ValueError):
                dense(path, dt=dt)

    def test_ideal_planar_plant_reaches_dense_endpoint(self):
        tracker = dense(np.column_stack((np.linspace(0, 1, 251), np.zeros(251))), dt=0.02)
        pose = np.zeros(3)
        for i in range(401):
            elapsed = i*0.02
            if tracker.is_complete(pose, elapsed):
                break
            vx, vy, wz = tracker.compute_velocity_command(pose[:2], pose[2], elapsed)
            pose += 0.02*np.array([vx*np.cos(pose[2])-vy*np.sin(pose[2]),
                                   vx*np.sin(pose[2])+vy*np.cos(pose[2]), wz])
        self.assertTrue(tracker.is_complete(pose, elapsed))
        self.assertGreaterEqual(elapsed, 5)
        self.assertLess(np.linalg.norm(pose[:2]-[1, 0]), 0.05)

    def test_resampling_preserves_duration_and_frequency(self):
        path = np.column_stack((np.linspace(0, 1, 500), np.zeros(500)))
        result, dt = prepare_trajectory(path, dt=0.01, max_frequency=20)
        self.assertAlmostEqual((len(result)-1)*dt, 4.99)
        self.assertLessEqual(1/dt, 20)
        np.testing.assert_allclose(result[[0, -1]], path[[0, -1]])
        np.testing.assert_allclose(result[:, 0], np.linspace(0, 1, len(result)))

    def test_explicit_duration_counts_intervals(self):
        result, dt = prepare_trajectory([[0, 0], [1, 0], [2, 0]], total_time=10)
        self.assertEqual(dt, 5)
        self.assertEqual((len(result)-1)*dt, 10)

    def test_resampling_heading_wrap(self):
        path = [[0, 0, np.deg2rad(179)], [1, 0, np.deg2rad(-179)], [2, 0, np.deg2rad(-177)]]
        result, _ = prepare_trajectory(path, dt=1, target_waypoints=2)
        self.assertAlmostEqual(np.rad2deg(result[-1, 2] - result[0, 2]), 4)


class InputEntryTests(unittest.TestCase):
    def test_csv_rows_become_pose_samples(self):
        spec = importlib.util.spec_from_file_location('file_entry_under_test', ROOT / 'experiment/run_waypoints_from_file.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'path.csv'
            path.write_text('0,1,2\n0,0,0\n0,0.1,0.2\n')
            result, yaw = module.load_trajectory(path)
            self.assertTrue(yaw)
            np.testing.assert_allclose(result[-1], [2, 0, 0.2])
            with self.assertRaises(SystemExit) as error:
                module.main([])
            self.assertEqual(error.exception.code, 2)


class OptimizerCostTests(unittest.TestCase):
    def test_horizon_includes_terminal_state_and_every_input(self):
        import casadi as ca
        from OptimalControl import OptimalControl
        solver = OptimalControl.__new__(OptimalControl)
        solver.dimStates = solver.dimInputs = 3
        solver.stepNumHorizon = 2
        solver.dimDecision = 12
        solver.w1, solver.w2, solver.w3, solver.w4 = 1, 2, 3, 4
        decision = ca.DM([1, 2, 0, 3, 4, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        parameters = ca.DM.zeros(5)
        self.assertAlmostEqual(float(solver._costFun(decision, parameters)), 33.01)
        decision[3] = 4
        self.assertAlmostEqual(float(solver._costFun(decision, parameters)), 40.01)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.udp, self.cmd, self.qtm = FakeUDP(), types.SimpleNamespace(), FakeQTM()

    def run_loop(self, step, **kwargs):
        options = dict(period=0.02, timeout=0.3, pose_timeout=0.08, startup_timeout=0.08)
        options.update(kwargs)
        return run_pose_loop(self.udp, self.cmd, 'MOCK', 'body', step, qtm_client=self.qtm, **options)

    def assert_stopped(self):
        self.assertEqual(self.udp.sent[-1]['mode'], 0)
        self.assertEqual(self.udp.sent[-1]['velocity'], [0, 0])
        self.assertEqual(self.udp.sent[-1]['yawSpeed'], 0)
        self.assertTrue(self.qtm.stopped and self.qtm.closed)

    async def test_normal_completion_stops_and_closes(self):
        calls = []
        async def step(pose, elapsed):
            calls.append(pose.copy())
            return [0.1, 0, 0] if len(calls) == 1 else None
        self.assertEqual(await self.run_loop(step), 'complete')
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0], [1, 0, 0])
        self.assert_stopped()

    async def test_pose_loss_without_more_callbacks_stops(self):
        async def step(*args):
            return [0.1, 0, 0]
        with self.assertRaisesRegex(TrackingStopped, 'feedback expired'):
            await self.run_loop(step)
        self.assert_stopped()

    async def test_repeated_frame_does_not_renew_deadline(self):
        async def step(*args):
            self.qtm.callback(frame(1))
            return [0.1, 0, 0]
        with self.assertRaisesRegex(TrackingStopped, 'feedback expired'):
            await self.run_loop(step)
        self.assert_stopped()

    async def test_no_frames_stops(self):
        self.qtm.initial = False
        async def step(*args):
            self.fail('control cannot run without a pose')
        with self.assertRaisesRegex(TrackingStopped, 'no motion-capture frames'):
            await self.run_loop(step)
        self.assert_stopped()

    async def test_timeout_with_fresh_frames_stops(self):
        count = 1
        async def step(*args):
            nonlocal count
            count += 1
            self.qtm.callback(frame(count))
            return [0.1, 0, 0]
        with self.assertRaisesRegex(TrackingStopped, 'run timeout'):
            await self.run_loop(step, timeout=0.09)
        self.assert_stopped()

    async def test_invalid_pose_latches_stop(self):
        async def step(*args):
            self.qtm.callback(frame(2, x=np.nan))
            self.qtm.callback(frame(3))  # A valid later pose must not clear a fault.
            return [0.1, 0, 0]
        with self.assertRaisesRegex(TrackingStopped, 'invalid motion-capture frame'):
            await self.run_loop(step)
        self.assertFalse(any(c['mode'] == 2 for c in self.udp.sent))
        self.assert_stopped()

    async def test_invalid_rotation_stops(self):
        async def step(*args):
            self.qtm.callback(frame(2, rotation=np.zeros(9)))
            return [0.1, 0, 0]
        with self.assertRaisesRegex(TrackingStopped, 'invalid rotation'):
            await self.run_loop(step)
        self.assert_stopped()

    async def test_disconnect_stops(self):
        async def step(*args):
            self.qtm.disconnected(None)
            return [0.1, 0, 0]
        with self.assertRaisesRegex(TrackingStopped, 'connection lost'):
            await self.run_loop(step)
        self.assert_stopped()

    async def test_slow_computation_never_sends_late_result(self):
        async def step(*args):
            await asyncio.sleep(0.1)
            return [0.1, 0, 0]
        with self.assertRaisesRegex(TrackingStopped, 'computation deadline'):
            await self.run_loop(step)
        self.assertFalse(any(c['mode'] == 2 for c in self.udp.sent))
        self.assert_stopped()

    async def test_cancellation_stops(self):
        async def step(*args):
            return [0.1, 0, 0]
        task = asyncio.create_task(self.run_loop(step))
        while not any(c['mode'] == 2 for c in self.udp.sent):
            await asyncio.sleep(0.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_stopped()

    async def test_step_exception_stops(self):
        async def step(*args):
            raise ValueError('solver error')
        with self.assertRaisesRegex(ValueError, 'solver error'):
            await self.run_loop(step)
        self.assert_stopped()

    async def test_nonfinite_command_stops(self):
        async def step(*args):
            return [np.nan, 0, 0]
        with self.assertRaisesRegex(TrackingStopped, 'nonfinite'):
            await self.run_loop(step)
        self.assert_stopped()


class MPCTests(unittest.IsolatedAsyncioTestCase):
    def controller(self, success=True, target=(2, 0), command=(0.1, 0, 0)):
        controller = mpc_module.ModelPredictiveControl.__new__(mpc_module.ModelPredictiveControl)
        controller.dt = 0.02
        controller.waypoints = [list(target)]
        controller.numWaypoints, controller.offset, controller.saveFlag = 1, 0.1, False
        controller.udp, controller.cmd = FakeUDP(), types.SimpleNamespace()
        controller.IP_server = 'MOCK'
        controller.config_data = {'QUALISYS': {'NAME_SINGLE_BODY': 'body'}}
        observed = []
        def solve(pose, elapsed, target):
            observed.append(pose.tolist())
            return None, np.array([command]), None, 0.001, 'ok' if success else 'failed', success
        controller.MyDogOc = types.SimpleNamespace(solve=solve,
            lin_vel_x_lb=0, lin_vel_x_ub=0.25, lin_vel_y_lb=-0.1, lin_vel_y_ub=0.1,
            ang_vel_lb=-1.57, ang_vel_ub=1.57)
        return controller, observed

    async def test_failed_solve_uses_current_pose_and_sends_no_motion(self):
        controller, observed = self.controller(success=False)
        qtm = FakeQTM()
        with patch.dict(sys.modules, {'qtm': qtm}), self.assertRaisesRegex(TrackingStopped, 'MPC solve failed'):
            await controller.run(np.array([-1.5, 0, 0]), 0.3)
        self.assertEqual(observed, [[1, 0, 0]])
        self.assertTrue(all(c['mode'] == 0 for c in controller.udp.sent))
        self.assertTrue(qtm.closed)

    async def test_final_waypoint_needs_no_extra_solve(self):
        controller, observed = self.controller(target=(1, 0))
        with patch.dict(sys.modules, {'qtm': FakeQTM()}):
            self.assertEqual(await controller.run(np.zeros(3), 0.3), 'complete')
        self.assertEqual(observed, [])
        self.assertTrue(all(c['mode'] == 0 for c in controller.udp.sent))

    async def test_successful_command_then_arrival_stops(self):
        controller, observed = self.controller()
        qtm = FakeQTM()
        controller.udp.on_send = lambda cmd: qtm.callback(frame(2, x=2)) if cmd['mode'] == 2 else None
        with patch.dict(sys.modules, {'qtm': qtm}):
            self.assertEqual(await controller.run(np.zeros(3), 0.3), 'complete')
        self.assertEqual(observed, [[1, 0, 0]])
        self.assertEqual([c['mode'] for c in controller.udp.sent], [0, 2, 0])
        self.assertEqual(controller.udp.sent[1]['velocity'], [0.1, 0])
        self.assertEqual(controller.uTraj.shape, (1, 3))

    async def test_dense_run_returns_and_stops_after_final_pose(self):
        tracker = dense([[1, 0], [1.1, 0]], dt=0.02)
        tracker.config_data['QUALISYS']['NAME_SINGLE_BODY'] = 'body'
        qtm = FakeQTM()
        tracker.udp.on_send = lambda cmd: qtm.callback(frame(2, x=1.1)) if cmd['velocity'][0] > 0 else None
        with patch.dict(sys.modules, {'qtm': qtm}):
            self.assertEqual(await tracker.run(timeout=0.3), 'complete')
        self.assertGreater(len(tracker.cmd_traj), 1)
        self.assertEqual(tracker.udp.sent[-1]['mode'], 0)
        self.assertTrue(qtm.closed)

    async def test_slow_solver_cannot_send_after_stop(self):
        controller, _ = self.controller()
        original_solve = controller.MyDogOc.solve
        def slow_solve(*args):
            time.sleep(0.08)
            return original_solve(*args)
        controller.MyDogOc.solve = slow_solve
        with patch.dict(sys.modules, {'qtm': FakeQTM()}), self.assertRaisesRegex(TrackingStopped, 'computation deadline'):
            await controller.run(np.zeros(3), 0.3)
        self.assertEqual(controller.udp.sent[-1]['mode'], 0)
        await asyncio.sleep(0.1)  # The background solve finishes after the stop.
        self.assertTrue(all(c['mode'] == 0 for c in controller.udp.sent))

    async def test_out_of_bounds_solver_command_rejected(self):
        controller, _ = self.controller(command=(4, 0, 0))
        with patch.dict(sys.modules, {'qtm': FakeQTM()}), self.assertRaisesRegex(TrackingStopped, 'out-of-bounds'):
            await controller.run(np.zeros(3), 0.3)
        self.assertTrue(all(c['mode'] == 0 for c in controller.udp.sent))


if __name__ == '__main__':
    unittest.main()
