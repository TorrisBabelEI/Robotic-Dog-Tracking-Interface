"""Synthetic sensors and localhost-only capture tests; never connects to hardware."""
import asyncio
from collections import Counter
import json
import math
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiment'))
from calibration_protocols import (BridgeDecoder, body_names, decode_go1, decode_qtm,
                                   quaternion_xyzw, read_qtm_packet, sdk_crc)
from export_calibration_reference import join_robot, reference
from record_estimator_calibration import MocapReceiver, parse_args, run


def go1_packet(tick=100):
    b = bytearray(820)
    b[:3] = b'\xfe\xef\xff'
    struct.pack_into('<4f', b, 22, 1, 0, 0, 0)
    struct.pack_into('<3f', b, 38, .1, .2, .3)
    struct.pack_into('<3f', b, 50, 0, 0, 9.81)
    for i in range(12):
        struct.pack_into('<Bffhh', b, 75+32*i, 10, i*.1, i*.2, i, -256*i)
        struct.pack_into('<b', b, 98+32*i, 40+i)
    struct.pack_into('<4h', b, 739, 100, 200, 300, 400)
    struct.pack_into('<4h', b, 747, 10, 20, 30, 40)
    struct.pack_into('<I', b, 755, tick)
    b[759:761] = b'\x55\x51'
    struct.pack_into('<I', b, 803, sdk_crc(b[:803]))
    return bytes(b)


def qtm_packet(frame=1, stamp=10000, angle=math.pi/2, p=(1000, 2000, 3000), residual=None):
    c, s = math.cos(angle), math.sin(angle)
    # A positive 90 degree body-to-world yaw, flattened by COLUMNS on the wire.
    flat = [c, s, 0, -s, c, 0, 0, 0, 1]
    fields = list(p)+flat+([] if residual is None else [residual])
    component = struct.pack('<IIIHH', 16+4*len(fields), 5 if residual is None else 11, 1, 2, 3)
    component += struct.pack('<%df' % len(fields), *fields)
    return struct.pack('<IIQII', 24+len(component), 3, stamp, frame, 1)+component


def reply():
    return dict(schema=1, nonce='a'*32, session='b'*32, source='live', reply_ns=1000000000,
                frames=[dict(sequence=1, sample_ns=990000000, payload_hex=go1_packet().hex())],
                dropped=0, error=None)


class ProtocolTests(unittest.TestCase):
    def test_qtm_units_and_rotation_direction(self):
        row = decode_qtm(qtm_packet(), 0)
        self.assertTrue(row['valid'])
        np.testing.assert_allclose(row['position_world_m'], [1, 2, 3])
        np.testing.assert_allclose(row['R_world_marker'], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-7)
        np.testing.assert_allclose(row['quaternion_xyzw'], [0, 0, math.sqrt(.5), math.sqrt(.5)])
        self.assertEqual(row['qtm_timestamp_us'], 10000)

    def test_180_degree_quaternion(self):
        for i in range(3):
            r = -np.eye(3)
            r[i, i] = 1
            expected = np.zeros(4)
            expected[i] = 1
            np.testing.assert_allclose(quaternion_xyzw(r), expected)

    def test_qtm_residual_and_occlusion_keep_raw_missing(self):
        row = decode_qtm(qtm_packet(residual=.4), 0)
        self.assertAlmostEqual(row['residual_mm'], .4)
        self.assertTrue(row['valid'])
        for packet in (qtm_packet(p=(float('nan'), 0, 0)), qtm_packet(residual=-1)):
            row = decode_qtm(packet, 0)
            self.assertFalse(row['valid'])
            json.dumps(row, allow_nan=False)
        self.assertFalse(decode_qtm(qtm_packet(), 1)['valid'])

    def test_qtm_bad_sizes_and_counts(self):
        packet = qtm_packet()
        for b in (b'', packet[:-1], packet+b'0'):
            with self.assertRaises(ValueError):
                decode_qtm(b, 0)
        b = bytearray(packet)
        struct.pack_into('<I', b, 32, 1000)
        with self.assertRaises(ValueError):
            decode_qtm(b, 0)

    def test_body_name_selection(self):
        xml = '<QTM_Parameters><The_6D><Body><Name>other</Name></Body><Body><Name>dog</Name></Body></The_6D></QTM_Parameters>'
        self.assertEqual(body_names(xml), ['other', 'dog'])
        with self.assertRaises(ValueError):
            body_names(xml.replace('other', 'dog'))
        with self.assertRaises(ValueError):
            body_names(xml.replace('<Name>other</Name>', '<Name/>'))

    def test_native_sensor_decode_and_crc(self):
        row = decode_go1(go1_packet())
        self.assertEqual(row['tick_ms'], 100)
        np.testing.assert_allclose(row['q_rad'], np.arange(12)*.1, atol=1e-7)
        self.assertEqual(row['tau_est_nm'], [-i for i in range(12)])
        self.assertEqual(row['motor_temperature_c'], list(range(40, 52)))
        self.assertEqual(row['foot_force_raw'], (100, 200, 300, 400))
        b = bytearray(go1_packet())
        b[40] ^= 1
        with self.assertRaises(ValueError):
            decode_go1(b)
        with self.assertRaises(ValueError):
            decode_go1(b[:800])

    def test_clock_bounds_across_unrelated_host_epochs(self):
        row = BridgeDecoder().decode(reply(), 'a'*32, 8000000000000, 8000010000000)[0]
        self.assertEqual(row['pc_sample_lower_ns'], 7999990000000)
        self.assertEqual(row['pc_sample_upper_ns'], 8000000000000)
        self.assertEqual(row['bridge_age_upper_ms'], 20)

    def test_bridge_rejects_identity_restart_replay_and_reorder(self):
        for key, value in [('nonce', 'c'*32), ('source', 'replay'), ('schema', 2)]:
            r = reply()
            r[key] = value
            with self.assertRaises(ValueError):
                BridgeDecoder().decode(r, 'a'*32, 0, 1)
        decoder = BridgeDecoder()
        decoder.decode(reply(), 'a'*32, 0, 1)
        with self.assertRaises(ValueError):
            decoder.decode(reply(), 'a'*32, 0, 1)
        r = reply()
        r['session'] = 'd'*32
        with self.assertRaises(ValueError):
            decoder.decode(r, 'a'*32, 0, 1)

    def test_invalid_robot_payload_retained_and_marked(self):
        r = reply()
        r['frames'][0]['payload_hex'] = bytes(820).hex()
        row = BridgeDecoder().decode(r, 'a'*32, 0, 1)[0]
        self.assertFalse(row['valid'])
        self.assertEqual(row['payload_hex'], bytes(820).hex())

    def test_future_frame_rejected_and_gap_counted(self):
        r = reply()
        r['frames'][0]['sample_ns'] = r['reply_ns']+1
        with self.assertRaises(ValueError):
            BridgeDecoder().decode(r, 'a'*32, 0, 1)
        decoder = BridgeDecoder()
        decoder.decode(reply(), 'a'*32, 0, 1)
        r = reply()
        r['frames'][0].update(sequence=4, sample_ns=995000000)
        self.assertEqual(decoder.decode(r, 'a'*32, 0, 1)[0]['sequence_gap'], 2)


class ReceiverTests(unittest.TestCase):
    def recorder(self):
        rec = Mock()
        rec.counts = Counter()
        rec.label = 'turn_left'
        rec.async_error = None
        return rec

    def test_wrong_sender_gaps_duplicates_and_occlusion(self):
        rec = self.recorder()
        receiver = MocapReceiver(rec, '127.0.0.1', 0)
        receiver.datagram_received(qtm_packet(), ('127.0.0.2', 15100))
        rec.write.assert_not_called()
        self.assertEqual(rec.counts['mocap_wrong_source'], 1)
        receiver.datagram_received(qtm_packet(1, 10000), ('127.0.0.1', 15100))
        receiver.datagram_received(qtm_packet(3, 30000), ('127.0.0.1', 15100))
        self.assertEqual(rec.counts['mocap_frame_gaps'], 1)
        receiver.datagram_received(qtm_packet(3, 30000), ('127.0.0.1', 15100))
        self.assertFalse(rec.write.call_args[0][1]['valid'])
        receiver.datagram_received(qtm_packet(4, 40000, p=(float('nan'), 0, 0)), ('127.0.0.1', 15100))
        self.assertFalse(rec.write.call_args[0][1]['valid'])
        self.assertEqual(rec.counts['mocap_valid'], 2)
        self.assertIsNone(rec.async_error)

    def test_udp_write_error_reaches_supervisor(self):
        rec = self.recorder()
        rec.write.side_effect = OSError('disk full')
        receiver = MocapReceiver(rec, '127.0.0.1', 0)
        receiver.datagram_received(qtm_packet(), ('127.0.0.1', 15100))
        self.assertIsInstance(rec.async_error, OSError)


class ReferenceTests(unittest.TestCase):
    def fixture(self):
        mount = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        offset = np.array([.2, .1, .05])
        rows = []
        for i in range(101):
            t = i*.01
            c, s = math.cos(t), math.sin(t)
            r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            marker = r @ mount.T
            rows.append(dict(qtm_timestamp_us=i*10000, frame_number=i, valid=True, ordered=True,
                             R_world_marker=marker.tolist(), quaternion_xyzw=quaternion_xyzw(marker),
                             position_world_m=(np.array([.4*t, 0, 0])+r@offset).tolist()))
        cfg = dict(R_marker_body=mount.tolist(), body_to_marker_position_body_m=offset.tolist(),
                   clock=dict(qtm_origin_s=0, pi_origin_s=100, scale=1, uncertainty_s=.001,
                              provenance='synthetic common clock'), max_reference_gap_s=.03,
                   derivative_window_samples=5)
        return rows, cfg

    def test_turning_lever_arm_mount_and_clock(self):
        rows, cfg = self.fixture()
        ref = reference(rows, cfg)
        self.assertEqual(sum(r['valid'] for r in ref), 97)
        for row in ref:
            if row['valid']:
                t = row['t_s']-100
                np.testing.assert_allclose(row['velocity_body_m_s'],
                                           [.4*math.cos(t), -.4*math.sin(t), 0], atol=1e-11)

    def test_clock_drift_and_nonuniform_times(self):
        rows, cfg = self.fixture()
        cfg['clock']['scale'] = 2
        ref = reference(rows, cfg)
        self.assertAlmostEqual(ref[50]['velocity_world_m_s'][0], .2)
        rows, cfg = self.fixture()
        del rows[50]
        ref = reference(rows, cfg)
        self.assertAlmostEqual(ref[49]['velocity_world_m_s'][0], .4)

    def test_occlusion_and_gap_are_not_differentiated(self):
        rows, cfg = self.fixture()
        rows[50]['valid'] = False
        ref = reference(rows, cfg)
        self.assertEqual(sum(r['valid'] for r in ref), 92)
        self.assertTrue(all(not r['valid'] for r in ref[48:53]))
        rows, cfg = self.fixture()
        del rows[50:60]
        self.assertTrue(any(r['reason'] == 'reference_gap' for r in reference(rows, cfg)))

    def test_join_does_not_extrapolate_or_fill_invalid(self):
        rows, cfg = self.fixture()
        ref = reference(rows, cfg)
        robot = [dict(pi_sample_monotonic_ns=int(t*1e9), valid=True) for t in (99, 100.5, 102)]
        joined = list(join_robot(robot, ref, .03))
        self.assertEqual([r['reference']['valid'] for r in joined], [False, True, False])
        ref[50]['valid'] = False
        self.assertFalse(list(join_robot(robot, ref, .03))[1]['reference']['valid'])

    def test_missing_calibration_and_reset_rejected(self):
        rows, cfg = self.fixture()
        cfg['clock']['pi_origin_s'] = None
        with self.assertRaises(ValueError):
            reference(rows, cfg)
        rows, cfg = self.fixture()
        rows[50]['qtm_timestamp_us'] = 1
        with self.assertRaises(ValueError):
            reference(rows, cfg)


class NetworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_qtm_tcp(self):
        reader = asyncio.StreamReader()
        message = b'Version set to 1.24\0'
        packet = struct.pack('<II', len(message)+8, 1)+message
        task = asyncio.create_task(read_qtm_packet(reader))
        for byte in packet:
            reader.feed_data(bytes([byte]))
            await asyncio.sleep(0)
        self.assertEqual(await task, (1, message))

    async def capture(self, folder, broken=False):
        commands = []
        bridge_requests = []
        handlers = []
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        async def bridge(reader, writer):
            handlers.append(asyncio.current_task())
            seq = 0
            try:
                while line := await reader.readline():
                    request = json.loads(line)
                    bridge_requests.append(request)
                    seq += 1
                    now = time.monotonic_ns()-3000000000
                    response = dict(schema=1, nonce=request['nonce'], session='b'*32, source='live',
                                    reply_ns=now, frames=[dict(sequence=seq, sample_ns=now-1000000,
                                    payload_hex=go1_packet(seq).hex())], dropped=0, error=None)
                    writer.write(json.dumps(response).encode()+b'\n')
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        async def qtm(reader, writer):
            handlers.append(asyncio.current_task())
            def respond(kind, text):
                data = text.encode()+b'\0'
                writer.write(struct.pack('<II', len(data)+8, kind)+data)
            try:
                respond(1, 'QTM RT Interface connected.')
                while True:
                    kind, data = await read_qtm_packet(reader)
                    command = data.rstrip(b'\0').decode()
                    commands.append(command)
                    if command.startswith('Version'):
                        respond(1, 'Version set to 1.24')
                    elif command == 'GetParameters 6D':
                        respond(2, '<QTM_Parameters><The_6D><Body><Name>dog</Name></Body></The_6D></QTM_Parameters>')
                    else:
                        address = command.split()[2].split(':')
                        target = (address[1], int(address[2]))
                        for i in range(1, 100):
                            udp.sendto(qtm_packet(i, i*10000), target)
                            if broken and i == 3:
                                respond(0, 'synthetic stream failure')
                                await writer.drain()
                                return
                            await asyncio.sleep(.01)
                        return
                    await writer.drain()
            except asyncio.IncompleteReadError:
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        bserver = await asyncio.start_server(bridge, '127.0.0.1', 0)
        qserver = await asyncio.start_server(qtm, '127.0.0.1', 0)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as port_probe:
            port_probe.bind(('127.0.0.1', 0))
            udp_port = port_probe.getsockname()[1]
        args = parse_args(['--qtm-host', '127.0.0.1', '--body', 'dog', '--no-stdin',
                           '--seconds', '.3', '--out', str(folder), '--qtm-port',
                           str(qserver.sockets[0].getsockname()[1]), '--bridge-port',
                           str(bserver.sockets[0].getsockname()[1]), '--mocap-port', str(udp_port)])
        try:
            with patch('builtins.print'):
                if broken:
                    with self.assertRaisesRegex(RuntimeError, 'synthetic stream failure'):
                        await run(args)
                else:
                    await run(args)
        finally:
            bserver.close()
            qserver.close()
            await bserver.wait_closed()
            await qserver.wait_closed()
            for task in handlers:
                task.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)
            udp.close()
        self.assertEqual(commands, ['Version 1.24', 'GetParameters 6D',
                                    'StreamFrames AllFrames UDP:127.0.0.1:%d 6D' % udp_port])
        self.assertTrue(all(set(r) == {'op', 'nonce'} and r['op'] == 'sample' for r in bridge_requests))

    async def test_full_recording_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'capture'
            await self.capture(folder)
            summary = json.loads((folder/'summary.json').read_text())
            self.assertIsNone(summary['failure'])
            self.assertGreater(summary['counts']['robot_valid'], 5)
            self.assertGreater(summary['counts']['mocap_valid'], 5)
            self.assertEqual(summary['motor_packets_sent'], 0)
            self.assertIn('qtm_6d_settings.xml', summary['sha256'])
            metadata = json.loads((folder/'metadata.json').read_text())
            self.assertFalse(metadata['clocks_synchronized'])
            self.assertEqual(metadata['arguments']['out'], '.')
            self.assertNotIn(str(folder.parent), json.dumps(metadata))
            self.assertFalse(Path(metadata['arguments']['config']).is_absolute())
            with self.assertRaises(FileExistsError):
                await self.capture(folder)

    async def test_failure_preserves_partial_capture_and_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'capture'
            await self.capture(folder, broken=True)
            summary = json.loads((folder/'summary.json').read_text())
            self.assertIn('synthetic stream failure', summary['failure'])
            self.assertGreater(summary['counts']['mocap_valid'], 0)


if __name__ == '__main__':
    unittest.main()
