"""Receive/decode helpers for factory Go1 + QTM calibration recordings.

QTM RT 1.24, little endian, 6D/6DRes only. No robot command interface.
Native Go1 offsets/CRC match decode_native_go1_pcap.py, not SDK LowState memory.
"""
import asyncio
import math
import struct
import xml.etree.ElementTree as ET

from decode_native_go1_pcap import sdk_crc

QTM_VERSION = '1.24'
MAX_QTM_PACKET = 4 * 1024 * 1024


def finite_json(value):
    """Retain missing measurements as null; raw packet hex preserves NaN bits."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def valid_rotation(r, tolerance=.01):
    if not all(math.isfinite(v) for row in r for v in row):
        return False
    for i in range(3):
        for j in range(3):
            if abs(sum(r[k][i] * r[k][j] for k in range(3)) - (i == j)) > tolerance:
                return False
    det = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
           - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
           + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    return abs(det - 1) <= tolerance


def quaternion_xyzw(r):
    """Active rotation matrix to unit quaternion, including 180 degree poses."""
    if not valid_rotation(r):
        raise ValueError('invalid rotation')
    trace = sum(r[i][i] for i in range(3))
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = [(r[2][1]-r[1][2])/s, (r[0][2]-r[2][0])/s,
             (r[1][0]-r[0][1])/s, s/4]
    else:
        i = max(range(3), key=lambda k: r[k][k])
        j, k = (i+1) % 3, (i+2) % 3
        s = 2 * math.sqrt(1 + r[i][i] - r[j][j] - r[k][k])
        q = [0., 0., 0., (r[k][j]-r[j][k])/s]
        q[i], q[j], q[k] = s/4, (r[j][i]+r[i][j])/s, (r[k][i]+r[i][k])/s
    norm = math.sqrt(sum(v*v for v in q))
    return [v/norm for v in q]


def body_names(xml):
    bodies = ET.fromstring(xml).findall('.//The_6D/Body')
    names = [(body.findtext('Name') or '').strip() for body in bodies]
    if not names or not all(names) or len(set(names)) != len(names):
        raise ValueError('QTM must have nonempty, unique 6DOF body names')
    return names


def decode_qtm(data, body_index):
    if len(data) < 24 or struct.unpack_from('<II', data) != (len(data), 3):
        raise ValueError('not a complete little-endian QTM data packet')
    timestamp, frame, count = struct.unpack_from('<QII', data, 8)
    offset = 24
    selected = None
    for _ in range(count):
        if offset + 8 > len(data):
            raise ValueError('truncated QTM component header')
        size, kind = struct.unpack_from('<II', data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError('invalid QTM component size')
        if kind in (5, 11):
            if selected is not None or size < 16:
                raise ValueError('duplicate/short 6DOF component')
            bodies, drop, sync = struct.unpack_from('<IHH', data, offset+8)
            width = 52 if kind == 11 else 48
            if size != 16 + bodies * width:
                raise ValueError('invalid 6DOF body count/size')
            selected = dict(body_count=bodies, camera_drop_per_mille=drop,
                            camera_out_of_sync_per_mille=sync, valid=False,
                            reason='body_missing')
            if 0 <= body_index < bodies:
                values = struct.unpack_from('<' + ('13f' if kind == 11 else '12f'),
                                            data, offset+16+body_index*width)
                p, flat = values[:3], values[3:12]
                # QTM wire array is column-major: r0,r1,r2 form the first column.
                r = [[flat[i+3*j] for j in range(3)] for i in range(3)]
                residual = values[12] if kind == 11 else None
                good = all(math.isfinite(v) for v in p) and valid_rotation(r)
                good = good and (residual is None or (math.isfinite(residual) and residual >= 0))
                selected.update(position_world_m=[v*.001 for v in p],
                                rotation_wire_column_major=flat, R_world_marker=r,
                                quaternion_xyzw=quaternion_xyzw(r) if good else None,
                                residual_mm=residual, valid=good,
                                reason='tracked' if good else 'untracked_or_invalid_pose')
        offset += size
    if offset != len(data) or selected is None:
        raise ValueError('missing 6DOF component or trailing QTM data')
    return finite_json(dict(qtm_timestamp_us=timestamp, frame_number=frame, **selected))


def decode_go1(payload):
    if len(payload) != 820 or payload[:3] != b'\xfe\xef\xff':
        raise ValueError('unsupported native Go1 packet profile')
    if sdk_crc(payload[:803]) != struct.unpack_from('<I', payload, 803)[0]:
        raise ValueError('Go1 CRC mismatch')
    motors = [struct.unpack_from('<Bffhh', payload, 75+32*i) for i in range(12)]
    imu = dict(quaternion_wxyz=struct.unpack_from('<4f', payload, 22),
               gyroscope_rad_s=struct.unpack_from('<3f', payload, 38),
               accelerometer_m_s2=struct.unpack_from('<3f', payload, 50),
               rpy_rad=struct.unpack_from('<3f', payload, 62))
    values = [v for m in motors for v in m] + [v for a in imu.values() for v in a]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite Go1 sensor data')
    if not .95 <= math.sqrt(sum(v*v for v in imu['quaternion_wxyz'])) <= 1.05:
        raise ValueError('invalid Go1 IMU quaternion norm')
    return dict(tick_ms=struct.unpack_from('<I', payload, 755)[0], imu=imu,
                motor_mode=[m[0] for m in motors], q_rad=[m[1] for m in motors],
                dq_rad_s=[m[2] for m in motors], ddq_native_raw=[m[3] for m in motors],
                tau_est_nm=[m[4]/256 for m in motors],
                motor_temperature_c=[struct.unpack_from('<b', payload, 98+32*i)[0] for i in range(12)],
                foot_force_raw=struct.unpack_from('<4h', payload, 739),
                foot_force_est_raw=struct.unpack_from('<4h', payload, 747),
                wireless_remote_hex=payload[759:799].hex())


class BridgeDecoder:
    """Preserve every frame and each request's PC-minus-Pi clock interval."""
    def __init__(self, allow_replay=False):
        self.allow_replay = allow_replay
        self.session = None
        self.sequence = 0
        self.stamp = None

    def decode(self, reply, nonce, sent_ns, received_ns):
        if reply.get('schema') != 1 or reply.get('nonce') != nonce or received_ns < sent_ns:
            raise ValueError('bridge identity/schema/timing mismatch')
        if reply.get('error'):
            raise RuntimeError('bridge capture failed: ' + str(reply['error']))
        if reply.get('source') not in (('live', 'replay') if self.allow_replay else ('live',)):
            raise ValueError('replay source requires --allow-replay')
        session = reply.get('session')
        if not isinstance(session, str) or len(session) != 32:
            raise ValueError('invalid bridge session')
        if self.session is not None and self.session != session:
            raise ValueError('bridge restarted; start a new recording')
        self.session = session
        now, frames = reply.get('reply_ns'), reply.get('frames')
        if type(now) is not int or not isinstance(frames, list) or len(frames) > 64:
            raise ValueError('invalid bridge batch')
        rows = []
        for frame in frames:
            seq, stamp = frame['sequence'], frame['sample_ns']
            if (type(seq) is not int or type(stamp) is not int or seq <= self.sequence
                    or stamp > now or (self.stamp is not None and stamp <= self.stamp)):
                raise ValueError('duplicate/reordered/future bridge frame')
            payload = bytes.fromhex(frame['payload_hex'])
            row = dict(sequence=seq, sequence_gap=seq-self.sequence-1 if self.sequence else 0,
                       pi_sample_monotonic_ns=stamp, pc_received_monotonic_ns=received_ns,
                       pc_sample_lower_ns=stamp+sent_ns-now,
                       pc_sample_upper_ns=stamp+received_ns-now,
                       bridge_age_upper_ms=(now-stamp+received_ns-sent_ns)/1e6,
                       payload_hex=frame['payload_hex'])
            try:
                row.update(valid=True, sensors=decode_go1(payload))
            except ValueError as exc:
                row.update(valid=False, reason=str(exc), sensors=None)
            rows.append(row)
            self.sequence, self.stamp = seq, stamp
        return rows


async def read_qtm_packet(reader):
    header = await reader.readexactly(8)
    size, kind = struct.unpack('<II', header)
    if not 8 <= size <= MAX_QTM_PACKET:
        raise ValueError('invalid QTM TCP packet size')
    return kind, await reader.readexactly(size-8)


async def send_qtm_command(writer, command):
    data = command.encode('ascii') + b'\0'
    writer.write(struct.pack('<II', len(data)+8, 1) + data)
    await writer.drain()
