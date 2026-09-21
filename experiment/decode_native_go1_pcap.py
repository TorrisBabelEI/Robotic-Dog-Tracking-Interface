#!/usr/bin/env python3
"""Offline decoder for the observed Go1 SDK 3.8.6 native 820/614-byte profile.
Never opens sockets. Not a general decoder or a command-replay tool.
Offsets/scales are checked against bundled SDK refineState/refineCmd machine code.
CRC checked with SDK native command transform; 13-byte state extension unparsed.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

JOINTS = [f'{leg}_{j}' for leg in ('FR', 'FL', 'RR', 'RL') for j in range(3)]
TABLE = []
for index in range(256):
    value = index << 24
    for _ in range(8):
        value = ((value << 1) ^ (0x04c11db7 if value & 0x80000000 else 0)) & 0xffffffff
    TABLE.append(value)


def sdk_crc(data):
    value = 0xffffffff
    for start in range(0, len(data) // 4 * 4, 4):
        for byte in data[start:start + 4][::-1]:
            value = ((value << 8) & 0xffffffff) ^ TABLE[(value >> 24) ^ byte]
    return value


def command_crc(data):
    value = bytearray(struct.pack("<I", sdk_crc(data) ^ 0xedcab9de))
    for a, b in ((1, 2), (0, 3), (0, 2)):
        value[a], value[b] = value[b], value[a]
    return struct.unpack("<I", value)[0]


def packets(path):
    with path.open('rb') as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] != b'\xd4\xc3\xb2\xa1':
            raise ValueError('requires little-endian microsecond classic PCAP')
        linktype = struct.unpack_from('<I', header, 20)[0]
        if linktype not in (1, 113):
            raise ValueError('requires Ethernet (DLT 1) or Linux cooked v1 (DLT 113) capture')
        while record := stream.read(16):
            if len(record) != 16:
                raise ValueError('truncated record header')
            sec, us, size, original = struct.unpack('<IIII', record)
            if size > 1048576:
                raise ValueError('unreasonable packet size')
            frame = stream.read(size)
            if len(frame) != size or size != original:
                raise ValueError('truncated captured packet')
            offset = 14 if linktype == 1 else 16
            if len(frame) < offset + 28 or frame[offset-2:offset] != b'\x08\x00':
                raise ValueError('unexpected link protocol or short packet')
            ip = frame[offset:]
            ihl = (ip[0] & 15) * 4
            if ip[0] >> 4 != 4 or ihl < 20 or ip[9] != 17:
                raise ValueError('expected IPv4 UDP')
            if struct.unpack_from('!H', ip, 6)[0] & 0x3fff:
                raise ValueError('fragmented IPv4 not supported')
            total = struct.unpack_from('!H', ip, 2)[0]
            if len(ip) < total or total < ihl + 8:
                raise ValueError('invalid IPv4 length')
            src, dst, length, _ = struct.unpack_from('!HHHH', ip, ihl)
            if length < 8 or ihl + length != total:
                raise ValueError('invalid UDP length')
            yield sec * 1000000 + us, '.'.join(map(str, ip[12:16])), src, '.'.join(map(str, ip[16:20])), dst, ip[ihl+8:total]


def decode(path, out):
    if out.exists():
        raise ValueError('output directory already exists; choose a new directory')
    out.mkdir(parents=True)
    rows = {'state': [], 'command': []}
    flows = collections.Counter()
    for timestamp, source, sport, dest, dport, payload in packets(path):
        flows[f'{source}:{sport} -> {dest}:{dport} length={len(payload)}'] += 1
        if (source, sport, dest, dport, len(payload)) == ('192.168.123.10',8007,'192.168.123.161',8008,820):
            kind = 'state'
        elif (source, sport, dest, dport, len(payload)) == ('192.168.123.161',8008,'192.168.123.10',8007,614):
            kind = 'command'
        else:
            raise ValueError('unrecognized flow/profile; do not infer offsets')
        if payload[:3] != b'\xfe\xef\xff':
            raise ValueError('unexpected Go1 header')
        crc_offset = 803 if kind == 'state' else 610
        crc_fn = sdk_crc if kind == 'state' else command_crc
        crc_ok = crc_fn(payload[:crc_offset]) == struct.unpack_from('<I', payload, crc_offset)[0]
        if not crc_ok:
            raise ValueError(f'{kind} SDK CRC mismatch at {timestamp}')
        row = {'pcap_time_us': timestamp, 'sdk_crc_match': int(crc_ok)}
        if kind == 'state':
            row['tick_ms'] = struct.unpack_from('<I', payload, 755)[0]
            for i, name in enumerate(('roll','pitch','yaw')):
                row[name] = struct.unpack_from('<f',payload,62 + i*4)[0]
            row['extension_hex'] = payload[807:].hex()
            row['remote_head0'], row['remote_head1'], row['remote_buttons'] = struct.unpack_from('<BBH', payload, 759)
            row['remote_nonzero'] = int(any(payload[759:799]))
            row['remote_l2_b'] = int((row['remote_buttons'] & 0x220) == 0x220)
        for index, joint in enumerate(JOINTS):
            if kind == 'state':
                offset = 75 + index*32
                mode,q,dq,ddq,tau = struct.unpack_from('<Bffhh',payload,offset)
                fields = dict(mode=mode,q=q,dq=dq,ddq=ddq,tau_est=tau/256,temperature=struct.unpack_from('<b',payload,offset+23)[0])
            else:
                offset = 22 + index*27
                mode,q,dq,tau,kp,kd = struct.unpack_from('<BffhHH',payload,offset)
                fields = dict(mode=mode,q=q,dq=dq,tau_ff=tau/256,kp=kp/32,kd=kd/16)
            if not all(math.isfinite(v) for v in fields.values()):
                raise ValueError('nonfinite joint data')
            row.update({f'{joint}_{key}':value for key,value in fields.items()})
        rows[kind].append(row)
    summary = {'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'flows':dict(flows),
               'limitations':['Passive observation only; posture requires operator context; never replay as commands.',
                              'CRC is an integrity check, not sender authentication; command uses SDK XOR/byte permutation.',
                              'State CRC covers bytes 0..799; command CRC covers 0..607. Partial words and state extension are excluded.',
                              'PCAP times are host capture times, not control-loop guarantees.']}
    for kind, data in rows.items():
        if len(data) < 2:
            raise ValueError('requires at least two packets in each direction')
        with (out / f'{kind}.csv').open('w', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        gaps=[(b['pcap_time_us']-a['pcap_time_us'])/1000 for a,b in zip(data,data[1:])]
        summary[kind]={'count':len(data),'span_s':(data[-1]['pcap_time_us']-data[0]['pcap_time_us'])/1e6,
                       'median_gap_ms':statistics.median(gaps),'max_gap_ms':max(gaps),
                       'crc_matches':sum(r['sdk_crc_match'] for r in data)}
    states=rows['state']; commands=rows['command']
    ticks=[(b['tick_ms']-a['tick_ms'])%2**32 for a,b in zip(states,states[1:])]
    summary['remote'] = {
        'nonzero_samples': sum(r['remote_nonzero'] for r in states),
        'header_counts': dict(collections.Counter(f"{r['remote_head0']:02x}{r['remote_head1']:02x}" for r in states)),
        'button_counts': dict(collections.Counter(f"{r['remote_buttons']:04x}" for r in states)),
        'l2_b_samples': sum(r['remote_l2_b'] for r in states),
        'button_transitions': [
            {'time_s': (r['pcap_time_us']-states[0]['pcap_time_us'])/1e6,
             'buttons_hex': f"{r['remote_buttons']:04x}",
             'header_hex': f"{r['remote_head0']:02x}{r['remote_head1']:02x}"}
            for i, r in enumerate(states)
            if i == 0 or r['remote_buttons'] != states[i-1]['remote_buttons']
               or (r['remote_head0'], r['remote_head1']) != (states[i-1]['remote_head0'], states[i-1]['remote_head1'])],
        'limitation': 'Factory stream observation only; does not validate experiment-port delivery or stop response.'}
    summary['state']['tick_gap_counts']=dict(collections.Counter(ticks))
    summary['state']['max_abs_dq']=max(abs(r[j+'_dq']) for r in states for j in JOINTS)
    summary['state']['max_temperature']=max(r[j+'_temperature'] for r in states for j in JOINTS)
    summary['imu']={key:{'median':statistics.median(r[key] for r in states),'range':max(r[key] for r in states)-min(r[key] for r in states)} for key in ('roll','pitch','yaw')}
    summary['joints']={j:{'q_median':statistics.median(r[j+'_q'] for r in states),
                          'q_range':max(r[j+'_q'] for r in states)-min(r[j+'_q'] for r in states),
                          'tau_est_median':statistics.median(r[j+'_tau_est'] for r in states),
                          'cmd_tau_median':statistics.median(r[j+'_tau_ff'] for r in commands),
                          'kp_values':sorted(set(r[j+'_kp'] for r in commands)),
                          'kd_values':sorted(set(r[j+'_kd'] for r in commands))} for j in JOINTS}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pcap',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=decode(args.pcap,args.out)
        print(json.dumps({k:v for k,v in result.items() if k!='joints'},indent=2))
        print('Output:',args.out.resolve())
    except (ValueError,OSError,struct.error) as error:
        parser.exit(2,f'ERROR: {error}\n')
