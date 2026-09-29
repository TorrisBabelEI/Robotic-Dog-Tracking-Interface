"""Shared high-level pose loop. SDK send calls stay on the event-loop thread."""
import asyncio
from contextlib import suppress
import math
import xml.etree.ElementTree as ET

import numpy as np
import transforms3d


class TrackingStopped(RuntimeError):
    """The run could not continue with fresh, valid feedback and commands."""


def send_velocity(udp, cmd, velocity=None):
    stop = velocity is None
    values = np.zeros(3) if stop else np.asarray(velocity, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise TrackingStopped('nonfinite or malformed velocity command')
    cmd.mode = 0 if stop else 2
    cmd.gaitType = 0 if stop else 1
    cmd.speedLevel = 0
    cmd.footRaiseHeight = 0
    cmd.bodyHeight = 0
    cmd.euler = [0, 0, 0]
    cmd.velocity = values[:2].tolist()
    cmd.yawSpeed = float(values[2])
    cmd.reserve = 0
    staged = udp.SetSend(cmd)
    if isinstance(staged, (int, float)) and staged < 0:
        raise TrackingStopped('SDK could not stage command')
    sent = udp.Send()
    if isinstance(sent, (int, float)) and sent < 0:
        raise TrackingStopped('SDK could not send command')


async def run_pose_loop(udp, cmd, host, body_name, step, *, period, timeout,
                        pose_timeout=0.25, startup_timeout=3.0, qtm_client=None):
    """Await fresh poses and an async step(pose, elapsed); None means complete.

    Packet callbacks only validate/store pose. Control work can await an executor
    without blocking packet reception. Deadlines also apply while no frames arrive.
    """
    for name, value in [('period', period), ('timeout', timeout),
                        ('pose_timeout', pose_timeout), ('startup_timeout', startup_timeout)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    if qtm_client is None:
        import qtm as qtm_client
    loop = asyncio.get_running_loop()
    connection = None
    latest = None
    received_at = None
    frame_number = None
    fault = None
    wanted_index = None

    def on_disconnect(*args):
        nonlocal fault
        fault = TrackingStopped('motion-capture connection lost')

    def on_packet(packet):
        nonlocal latest, received_at, frame_number, fault
        if fault is not None:
            return
        try:
            number = packet.framenumber
            if frame_number is not None and number <= frame_number:
                return  # Replayed frames cannot renew the feedback deadline.
            position, rotation = packet.get_6d()[1][wanted_index]
            xyz = np.array([position.x, position.y, position.z], dtype=float)
            matrix = np.asarray(rotation.matrix, dtype=float).reshape(3, 3)
            if not np.isfinite(xyz).all() or not np.isfinite(matrix).all():
                raise ValueError('nonfinite pose')
            if (not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3)
                    or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-3)):
                raise ValueError('invalid rotation matrix')
            quat = transforms3d.quaternions.mat2quat(matrix)
            yaw = -transforms3d.euler.quat2euler(quat, axes='sxyz')[2]
            latest = np.array([xyz[0] / 1000, xyz[1] / 1000, yaw])
            received_at = loop.time()
            frame_number = number
        except Exception as error:
            fault = TrackingStopped('invalid motion-capture frame: ' + str(error))

    try:
        send_velocity(udp, cmd)
        connection = await asyncio.wait_for(
            qtm_client.connect(host, on_disconnect=on_disconnect), startup_timeout)
        if connection is None:
            raise TrackingStopped('could not connect to motion capture')
        xml = await asyncio.wait_for(connection.get_parameters(parameters=['6d']), startup_timeout)
        names = [body.text.strip() for body in ET.fromstring(xml).findall('*/Body/Name')]
        if names.count(body_name) != 1:
            raise TrackingStopped('motion-capture body missing or ambiguous: ' + body_name)
        wanted_index = names.index(body_name)
        await asyncio.wait_for(connection.stream_frames(components=['6d'], on_packet=on_packet),
                               startup_timeout)
        started = loop.time()
        next_control = started
        while True:
            now = loop.time()
            if fault is not None:
                raise fault
            if now - started >= timeout:
                raise TrackingStopped('run timeout')
            if received_at is None:
                if now - started >= startup_timeout:
                    raise TrackingStopped('no motion-capture frames received')
            elif now - received_at >= pose_timeout:
                raise TrackingStopped('motion-capture feedback expired')
            elif now >= next_control:
                pose = latest.copy()
                pose_at = received_at
                budget = min(period, pose_at + pose_timeout - now, started + timeout - now)
                try:
                    command = await asyncio.wait_for(step(pose, now - started), budget)
                except asyncio.TimeoutError as error:
                    raise TrackingStopped('control computation deadline exceeded') from error
                if fault is not None:
                    raise fault
                if loop.time() >= started + timeout or loop.time() >= pose_at + pose_timeout:
                    raise TrackingStopped('command expired before transmission')
                if command is None:
                    return 'complete'
                send_velocity(udp, cmd, command)
                next_control = now + period
            await asyncio.sleep(min(0.01, max(0.001, next_control - loop.time())))
    finally:
        # Stop before any awaited cleanup or disk I/O, including cancellation.
        try:
            send_velocity(udp, cmd)
        finally:
            if connection is not None:
                with suppress(Exception):
                    await asyncio.wait_for(connection.stream_frames_stop(), 0.5)
                connection.disconnect()
