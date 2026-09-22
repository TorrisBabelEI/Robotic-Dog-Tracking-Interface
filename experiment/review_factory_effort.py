"""Offline command/state pairing. Reports commanded PD+FF effort, not measured torque.
No sockets or replay. Cannot authorize takeover from historical PCAP data.
"""
import argparse,collections,hashlib,json,math,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment.decode_native_go1_pcap import packets,sdk_crc,command_crc
# Adapted from Wenjian_test_walking_policy/deployment/review_factory_effort.py.

def total_effort(command,state,profile="factory"):
    if profile not in ("factory", "sdk"):raise ValueError("Unknown command CRC profile")
    if len(command)!=614 or len(state)!=820:raise ValueError('Unsupported native packet layout')
    if command[:3]!=b'\xfe\xef\xff' or state[:3]!=b'\xfe\xef\xff':raise ValueError('Invalid native header')
    if (command_crc if profile=="factory" else sdk_crc)(command[:610])!=struct.unpack_from('<I',command,610)[0]:raise ValueError('Command CRC')
    if sdk_crc(state[:803])!=struct.unpack_from('<I',state,803)[0]:raise ValueError('State CRC')
    effort=[]
    for i in range(12):
        mode,q,dq,ff,kp,kd=struct.unpack_from('<BffhHH',command,22+27*i)
        smode,sq,sdq=struct.unpack_from('<Bff',state,75+32*i)
        kp/=32.;kd/=16.;ff/=256.
        if mode!=10 or smode!=10:raise ValueError('Not confirmed servo command/state')
        if not all(math.isfinite(v) for v in (q,dq,sq,sdq)):raise ValueError('Nonfinite state or target')
        # Do not multiply SDK position/velocity stop sentinels by gains.
        if (abs(q)>1e8 and kp!=0) or (abs(dq)>1000 and kd!=0):raise ValueError('Active gain with stop sentinel')
        effort.append((kp*(q-sq) if kp else 0)+(kd*(dq-sdq) if kd else 0)+ff)
    return effort

def review(path, profile='factory', source_port=8008):
    if profile not in ('factory', 'sdk') or not 1 <= source_port <= 65535:
        raise ValueError('Invalid command profile or port')
    latest=None; rows=[]; rejected=collections.Counter(); flows=collections.Counter()
    last_stamp=None; command_frames=state_frames=0; reordered=0
    command_flow=('192.168.123.161',source_port,'192.168.123.10',8007)
    state_flow=('192.168.123.10',8007,'192.168.123.161',source_port)
    for stamp,src,sp,dst,dp,b in packets(path):
        if last_stamp is not None and stamp < last_stamp:
            reordered+=1  # Multi-interface captures can be recorded out of timestamp order.
        last_stamp=stamp
        flow=(src,sp,dst,dp)
        flows[f'{src}:{sp} -> {dst}:{dp} len={len(b)}']+=1
        if flow==state_flow:
            state_frames+=1; latest=(stamp,b)
        elif flow==command_flow:
            command_frames+=1
            try:
                if latest is None or not 0<=stamp-latest[0]<=4000:
                    raise ValueError('No past feedback within 4ms')
                tau=total_effort(b,latest[1],profile)
                rows.append(dict(stamp_us=stamp,state_stamp_us=latest[0],
                                 state_age_us=stamp-latest[0],effort_nm=tau))
            except ValueError as error:
                rejected[str(error)]+=1
    summary=dict(source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        command_profile=profile,source_port=source_port,flows=dict(flows),
        command_frames=command_frames,state_frames=state_frames,paired_frames=len(rows),
        out_of_order_packets=reordered,
        rejected=dict(rejected),max_pair_age_us=max((r['state_age_us'] for r in rows),default=None),
        max_abs_commanded_effort_nm=max((abs(v) for r in rows for v in r['effort_nm']),default=None),
        scope='Offline past-state pairing; commanded effort estimate only, no hardware calibration or takeover authority',
        limitations=['Joint order FR, FL, RR, RL; hip, thigh, calf.',
          'Only observed 614/820-byte servo profile decoded; other modes rejected.',
          'Capture timestamps bound pairing age, not true sensor delay.',
          'Factory and SDK CRC profiles are selected explicitly; no fallback.',
          'Concurrent flows are counted but never cross-paired.',
          'Out-of-order packets are counted; a future or >4ms-old latest state is never paired.'],motor_packets_sent=0)
    return summary,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pcap',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--command-profile',choices=('factory','sdk'),default='factory')
    parser.add_argument('--source-port',type=int,default=8008)
    args=parser.parse_args()
    try:
        if args.out.exists():raise ValueError('Output exists; choose a new directory')
        summary,rows=review(args.pcap,args.command_profile,args.source_port)
        args.out.mkdir(parents=True)
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        with (args.out/'pairs.jsonl').open('w') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
        print(json.dumps(summary,indent=2))
        return 0 if rows else 2
    except (OSError,ValueError,struct.error) as error:
        parser.exit(2,f'ERROR: {error}\n')

if __name__=='__main__':raise SystemExit(main())
