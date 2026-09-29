#!/usr/bin/env python3
"""Record factory Go1 telemetry and Qualisys 6DOF UDP on one PC (Python 3.8+).

Uses the existing passive Pi bridge over a loopback SSH tunnel. Sends only
bridge sample requests and QTM stream requests; has no Unitree command socket.
See docs/ESTIMATOR_CALIBRATION_CAPTURE.md for setup, clocks, and file schemas.
"""
import argparse
import asyncio
from collections import Counter
from contextlib import ExitStack
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import secrets
import signal
import socket
import sys
import time

from calibration_protocols import (BridgeDecoder, QTM_VERSION, body_names,
                                   decode_qtm, read_qtm_packet, send_qtm_command)

ROOT = Path(__file__).resolve().parents[1]


class Recording:
    def __init__(self, out, metadata):
        out.mkdir(parents=True, exist_ok=False)
        self.out = out
        self.stack = ExitStack()
        self.files = {name: self.stack.enter_context((out/(name+'.jsonl')).open('x'))
                      for name in ('robot', 'mocap', 'transport', 'events')}
        self.counts = Counter()
        self.started_ns = time.monotonic_ns()
        self.last_robot_ns = self.last_mocap_ns = self.started_ns
        self.last_valid_mocap_ns = None
        self.label = metadata['initial_label']
        self.failure = None
        self.async_error = None
        self.stopped_by = 'duration'
        (out/'metadata.json').write_text(json.dumps(dict(
            schema=1, start_unix_ns=time.time_ns(), start_pc_monotonic_ns=self.started_ns,
            motor_packets_sent=0, **metadata), indent=2)+'\n')
        self.event('label', label=self.label, source='initial_label')

    def write(self, name, row):
        self.files[name].write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
        self.counts[name] += 1

    def event(self, kind, **kwargs):
        self.write('events', dict(event=kind, pc_monotonic_ns=time.monotonic_ns(), **kwargs))

    def flush(self):
        for stream in self.files.values():
            stream.flush()

    def close(self):
        self.event('stop', stopped_by=self.stopped_by, failure=self.failure)
        self.stack.close()
        summary = dict(counts=dict(self.counts), failure=self.failure,
                       stopped_by=self.stopped_by, motor_packets_sent=0,
                       duration_s=(time.monotonic_ns()-self.started_ns)/1e9)
        # Hash incrementally: captures can be much larger than memory.
        hashes = {}
        for path in sorted(self.out.iterdir()):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        digest.update(chunk)
                hashes[path.name] = digest.hexdigest()
        summary['sha256'] = hashes
        (self.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2), '\nRecording:', self.out.resolve(), flush=True)


async def robot_loop(a, rec):
    decoder = BridgeDecoder(a.allow_replay)
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection('127.0.0.1', a.bridge_port, limit=262145), 3)
    writer.get_extra_info('socket').setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            nonce = secrets.token_hex(16)
            sent = time.monotonic_ns()
            writer.write(json.dumps(dict(op='sample', nonce=nonce)).encode()+b'\n')
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), a.source_timeout)
            received = time.monotonic_ns()
            if not line or len(line) > 262144 or not line.endswith(b'\n'):
                raise ValueError('closed or oversized bridge reply')
            reply = json.loads(line)
            # Keep timing/sequence metadata without duplicating sensor bytes.
            transport_reply = {k: v for k, v in reply.items() if k != 'frames'}
            transport_reply['frames'] = [{k: v for k, v in frame.items() if k != 'payload_hex'}
                                         for frame in reply.get('frames', [])]
            rec.write('transport', dict(pc_sent_monotonic_ns=sent,
                                        pc_received_monotonic_ns=received, reply=transport_reply))
            try:
                rows = decoder.decode(reply, nonce, sent, received)
            except (ValueError, RuntimeError, KeyError, TypeError):
                rec.write('transport', dict(rejected_reply=reply))
                raise
            for row in rows:
                row['label_at_receive'] = rec.label
                rec.write('robot', row)
                if row['valid']:
                    rec.counts['robot_valid'] += 1
                    rec.last_robot_ns = received
                else:
                    rec.counts['robot_invalid'] += 1
                rec.counts['robot_sequence_gaps'] += row['sequence_gap']
            await asyncio.sleep(max(0, .01-(time.monotonic_ns()-sent)/1e9))
    finally:
        writer.close()
        await writer.wait_closed()


class MocapReceiver(asyncio.DatagramProtocol):
    def __init__(self, rec, server, index):
        self.rec, self.server, self.index = rec, server, index
        self.previous_frame = self.previous_stamp = None

    def datagram_received(self, data, address):
        # Promote callback exceptions to the supervisor instead of letting
        # asyncio only log them and continue an incomplete recording.
        try:
            self.record_datagram(data, address)
        except Exception as exc:
            self.rec.async_error = exc

    def record_datagram(self, data, address):
        received = time.monotonic_ns()
        if address[0] != self.server:
            self.rec.counts['mocap_wrong_source'] += 1
            return
        row = dict(pc_received_monotonic_ns=received, sender=list(address),
                   raw_packet_hex=data.hex(), label_at_receive=self.rec.label)
        try:
            row.update(decode_qtm(data, self.index))
            frame, stamp = row['frame_number'], row['qtm_timestamp_us']
            delta = None if self.previous_frame is None else (frame-self.previous_frame) % 2**32
            ordered = (delta is None or 0 < delta < 2**31) and (
                self.previous_stamp is None or stamp > self.previous_stamp)
            row['ordered'] = ordered
            row['frame_gap'] = max(0, delta-1) if ordered and delta is not None else 0
            self.rec.counts['mocap_frame_gaps'] += row['frame_gap']
            if ordered:
                self.previous_frame, self.previous_stamp = frame, stamp
            else:
                row.update(valid=False, reason='duplicate_reordered_or_reset_clock')
            self.rec.last_mocap_ns = received
            if row['valid']:
                self.rec.last_valid_mocap_ns = received
                self.rec.counts['mocap_valid'] += 1
            else:
                self.rec.counts['mocap_invalid'] += 1
        except (ValueError, OverflowError) as exc:
            row.update(valid=False, ordered=False, reason=str(exc))
            self.rec.counts['mocap_malformed'] += 1
        self.rec.write('mocap', row)

    def error_received(self, exc):
        self.rec.async_error = exc


async def qtm_response(reader, expected, rec):
    while True:
        kind, data = await asyncio.wait_for(read_qtm_packet(reader), 5)
        if kind == 0:
            raise RuntimeError('QTM: ' + data.rstrip(b'\0').decode(errors='replace'))
        if kind == expected:
            return data.rstrip(b'\0').decode('utf-8')
        rec.event('qtm_control_packet', packet_type=kind, payload_hex=data.hex())


async def mocap_loop(a, rec):
    reader, writer = await asyncio.wait_for(asyncio.open_connection(a.qtm_host, a.qtm_port), 5)
    transport = None
    try:
        welcome = await qtm_response(reader, 1, rec)
        rec.event('qtm_welcome', message=welcome)
        await send_qtm_command(writer, 'Version '+QTM_VERSION)
        version = await qtm_response(reader, 1, rec)
        if version != 'Version set to '+QTM_VERSION:
            raise RuntimeError('Unexpected QTM version response: '+version)
        await send_qtm_command(writer, 'GetParameters 6D')
        xml = await qtm_response(reader, 2, rec)
        (rec.out/'qtm_6d_settings.xml').write_text(xml)
        names = body_names(xml)
        if a.body not in names:
            raise ValueError('QTM body %r not found; available: %s' % (a.body, names))
        index = names.index(a.body)
        # Bind a real local Ethernet address; QTM sends UDP to this TCP client's IP.
        local_ip = writer.get_extra_info('sockname')[0]
        server_ip = writer.get_extra_info('peername')[0]
        transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
            lambda: MocapReceiver(rec, server_ip, index),
            local_addr=(a.mocap_bind or local_ip, a.mocap_port), family=socket.AF_INET)
        rec.event('qtm_stream', body=a.body, body_index=index, names=names,
                  udp_address=transport.get_extra_info('sockname'), protocol_version=QTM_VERSION)
        await send_qtm_command(writer, 'StreamFrames AllFrames UDP:%s:%d %s' % (
            a.mocap_bind or local_ip, a.mocap_port, a.component))
        # Streaming has no mandatory command acknowledgement. Consume errors/events
        # while the independent UDP callback receives measurements.
        while True:
            kind, data = await read_qtm_packet(reader)
            rec.event('qtm_control_packet', packet_type=kind, payload_hex=data.hex())
            if kind == 0:
                raise RuntimeError('QTM stream error: '+data.rstrip(b'\0').decode(errors='replace'))
            # Settings changes invalidate the body-index mapping for this capture.
            if kind == 6 and data and data[0] in (11, 12, 14):
                raise RuntimeError('QTM camera/settings changed; start a new recording')
    finally:
        # Closing this client's connection stops its stream, not QTM acquisition.
        if transport:
            transport.close()
        writer.close()
        await writer.wait_closed()


async def monitor(a, rec, stop):
    while not stop.is_set():
        await asyncio.sleep(1)
        if rec.async_error is not None:
            raise rec.async_error
        now = time.monotonic_ns()
        ages = [(now-rec.last_robot_ns)/1e9, (now-rec.last_mocap_ns)/1e9]
        rec.flush()
        tracking = 'tracked' if (rec.last_valid_mocap_ns is not None and
                                 now-rec.last_valid_mocap_ns < 500000000) else 'TRACKING LOST'
        print('robot=%d mocap=%d valid_poses=%d gaps(robot/mocap)=%d/%d %s label=%s' % (
            rec.counts['robot'], rec.counts['mocap'], rec.counts['mocap_valid'],
            rec.counts['robot_sequence_gaps'], rec.counts['mocap_frame_gaps'], tracking, rec.label), flush=True)
        if max(ages) > a.source_timeout:
            raise RuntimeError('source timeout: robot %.2fs, QTM %.2fs since valid packet' % tuple(ages))


async def run(a):
    source = Path(__file__).resolve()
    arguments = {k: v.name if isinstance(v, Path) else v for k, v in vars(a).items()}
    arguments['out'] = '.'  # metadata.json is stored inside this recording.
    metadata = dict(arguments=arguments, path_references='file_names; out is recording directory',
                    config_sha256=(hashlib.sha256(a.config.read_bytes()).hexdigest()
                                   if a.config.is_file() else None),
                    initial_label=a.label, notes=a.notes, qtm_protocol=QTM_VERSION,
                    robot_profile='native_820_byte_Go1_factory_feedback',
                    sdk_joint_order='FR,FL,RR,RL; hip,thigh,calf; raw SDK signs',
                    clocks_synchronized=False, mounting_transform_measured=False,
                    qtm_pc_timestamp_kind='userspace UDP callback monotonic time',
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (source, source.with_name('calibration_protocols.py'),
                                             source.with_name('decode_native_go1_pcap.py'))})
    rec = Recording(a.out, metadata)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop(name):
        rec.stopped_by = name
        stop.set()

    def label_input():
        line = sys.stdin.readline()
        if not line:
            loop.remove_reader(sys.stdin.fileno())
        elif line.strip():
            rec.label = line.strip()
            rec.event('label', label=rec.label, source='operator_stdin')

    handlers = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, request_stop, sig.name)
        handlers.append(sig)
    interactive = sys.stdin.isatty() and not a.no_stdin
    if interactive:
        loop.add_reader(sys.stdin.fileno(), label_input)
        print('Type a motion label and press Enter to mark it (e.g. stationary, forward, turn_left).')
    tasks = [asyncio.create_task(robot_loop(a, rec)), asyncio.create_task(mocap_loop(a, rec)),
             asyncio.create_task(monitor(a, rec, stop))]
    stop_task = asyncio.create_task(stop.wait())
    timer = loop.call_later(a.seconds, stop.set)
    try:
        print('Recording factory feedback + QTM UDP; no motor commands. Output:', a.out, flush=True)
        done, _ = await asyncio.wait(tasks+[stop_task], return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is not stop_task:
                task.result()
        if rec.async_error is not None:
            raise rec.async_error
        if not rec.counts['robot_valid'] or not rec.counts['mocap_valid']:
            raise RuntimeError('recording lacks valid samples from both sources')
    except Exception as exc:
        rec.failure = '%s: %s' % (type(exc).__name__, exc)
        raise
    finally:
        timer.cancel()
        for task in tasks+[stop_task]:
            task.cancel()
        await asyncio.gather(*tasks, stop_task, return_exceptions=True)
        if interactive:
            loop.remove_reader(sys.stdin.fileno())
        for sig in handlers:
            loop.remove_signal_handler(sig)
        rec.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'experiment/config/config_dog.json',
                   help='existing QUALISYS config; host/body flags override it')
    p.add_argument('--qtm-host', help='QTM Ethernet IPv4 address')
    p.add_argument('--qtm-port', type=int, default=22223, help='QTM little-endian TCP control port')
    p.add_argument('--body', help='case-sensitive QTM rigid-body name')
    p.add_argument('--mocap-bind', help='local PC Ethernet IPv4; default: route to QTM')
    p.add_argument('--mocap-port', type=int, default=15100, help='local UDP data port')
    p.add_argument('--component', choices=('6D', '6DRes'), default='6D')
    p.add_argument('--bridge-port', type=int, default=15002, help='loopback SSH tunnel port')
    p.add_argument('--seconds', type=float, default=120)
    p.add_argument('--source-timeout', type=float, default=10)
    p.add_argument('--out', type=Path, default=ROOT/'logs'/('calibration_'+time.strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--label', default='unlabeled', help='initial motion/trial label')
    p.add_argument('--notes', default='', help='operator role or ID, surface, hardware versions, mounting notes')
    p.add_argument('--no-stdin', action='store_true')
    p.add_argument('--allow-replay', action='store_true', help='offline bridge testing only')
    a = p.parse_args(argv)
    if not a.qtm_host or not a.body:
        c = json.loads(a.config.read_text())['QUALISYS']
        a.qtm_host = a.qtm_host or c['IP_MOCAP_SERVER']
        a.body = a.body or c['NAME_SINGLE_BODY']
    for name in ('qtm_host', 'mocap_bind'):
        if getattr(a, name):
            ipaddress.IPv4Address(getattr(a, name))
    if a.mocap_bind == '0.0.0.0':
        p.error('--mocap-bind must be a concrete local Ethernet address')
    for name in ('qtm_port', 'mocap_port', 'bridge_port'):
        if not 1024 <= getattr(a, name) <= 65535:
            p.error(name+' must be 1024..65535')
    for name in ('seconds', 'source_timeout'):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(name+' must be finite and positive')
    return a


def main():
    try:
        asyncio.run(run(parse_args()))
    except (OSError, ValueError, RuntimeError, asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
