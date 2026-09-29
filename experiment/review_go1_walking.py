#!/usr/bin/env python3
"""Review local walking packages/recordings. No SSH, subprocesses, sockets or arming.

Adapts the policy project’s kp100_slew_20260925/hardware/quick_decode.py and
tracking_report.py to the parent decoder, explicit endpoints and fresh pairing.
Runtime packages remain independently versioned; never execute their imports.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import numpy as np

try:
    from .decode_native_go1_pcap import packets, sdk_crc
except ImportError:
    from decode_native_go1_pcap import packets, sdk_crc

JOINTS = [leg + '_' + joint for leg in ('FR', 'FL', 'RR', 'RL')
          for joint in ('hip', 'thigh', 'calf')]
LIMITATION = 'Offline evidence only; not motor-torque measurement, current Pi verification or permission to arm.'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def package_review(root):
    root = root.resolve()
    manifest = read_json(root / 'click_release.json')
    files = manifest['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing release file inventory')
    checked, problems = {}, []
    for name, expected in files.items():
        path = root / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or
                not path.resolve().is_relative_to(root) or path.is_symlink()):
            raise ValueError('Release pathname escapes package: ' + name)
        if not path.is_file():
            problems.append('missing: ' + name)
            continue
        checked[name] = sha256(path)
        if checked[name] != expected:
            problems.append('hash mismatch: ' + name)
    tuning = read_json(root / 'tuning.json')
    if manifest.get('active_tuning') != tuning:
        problems.append('tuning.json differs from active_tuning')
    # Bind the data-format representation without importing package Python.
    lines = ['GO1_WALKING_TUNING_V1']
    for name in ('kp', 'kd', 'command'):
        lines.append(name + ' ' + ' '.join(format(float(v), '.17g') for v in tuning[name]))
    for name in ('duration_s', 'gain_ramp_s'):
        lines.append(name + ' ' + format(float(tuning[name]), '.17g'))
    if (root / 'walking_gains.conf').read_text() != '\n'.join(lines) + '\n':
        problems.append('native gain config differs from tuning.json')
    policy = read_json(root / 'policy/manifest.json')
    if policy.get('weights_sha256') != checked.get('policy/weights.npz'):
        problems.append('policy metadata weights hash mismatch')
    return dict(kind='local_package_integrity', source=root.name,
                source_path_kind='artifact_directory_name',
                integrity_passed=not problems, problems=problems,
                checked_files=len(checked), file_sha256=checked,
                manifest_sha256=sha256(root / 'click_release.json'),
                tuning=tuning, recorded_boot_id=manifest.get('boot_id'),
                release_hardware_validated=manifest.get('hardware_validated'),
                release_production=manifest.get('production_release'),
                policy_metadata=policy, hardware_ready=False, limitation=LIMITATION)


def decode_frame(payload, state=False):
    size, crc_offset = (820, 803) if state else (614, 610)
    if len(payload) != size or payload[:3] != b'\xfe\xef\xff':
        raise ValueError('native header/length')
    if sdk_crc(payload[:crc_offset]) != struct.unpack_from('<I', payload, crc_offset)[0]:
        raise ValueError('SDK CRC')
    rows = []
    for i in range(12):
        if state:
            mode, q, dq = struct.unpack_from('<Bff', payload, 75 + 32 * i)
            row = [q, dq]
        else:
            mode, q, dq, ff, kp, kd = struct.unpack_from('<BffhHH', payload, 22 + 27 * i)
            row = [q, dq, ff / 256, kp / 32, kd / 16]
            if (abs(q) > 1e8 and kp) or (abs(dq) > 1000 and kd):
                raise ValueError('active stop sentinel')
        if mode != 10 or not all(math.isfinite(v) for v in row):
            raise ValueError('non-servo/nonfinite frame')
        rows.append(row)
    return np.asarray(rows)


def tagged(text, name):
    matches = re.findall(r'^' + re.escape(name) + r':(\{.*\})$', text, re.M)
    if len(matches) != 1:
        raise ValueError('Require one ' + name + ' record')
    return json.loads(matches[0])


def exit_review(events):
    def first(name):
        return next((e['monotonic_ns'] for e in events if e['event'] == name), None)
    resume = first('actors_resume_begin')
    names = ('guardian_exit_proved', 'sender_exit_proved', 'workers_exit_proved')
    exits = [first(name) for name in names]
    proved = resume is not None and all(t is not None and t < resume for t in exits)
    resumed = first('actors_resumed')
    return dict(exit_before_restore=proved,
                actors_resumed=resumed is not None and resume is not None and resumed >= resume,
                stop_reasons=[e['event'][len('sender_fault:'):] for e in events
                              if e['event'].startswith('sender_fault:')],
                scope='Recorded process evidence; physical factory response is a separate observation.')


def tracking_metrics(state_times, states, command_times, commands, start, end, max_age_us=4000):
    st, ct = np.asarray(state_times), np.asarray(command_times)
    # Never use a future command or extend the last command across a capture gap.
    index = np.searchsorted(ct, st, side='right') - 1
    in_window = (st >= start) & (st <= end)
    valid = in_window & (index >= 0)
    valid &= st - ct[np.maximum(index, 0)] <= max_age_us
    if not valid.any():
        raise ValueError('No fresh command/state pairs in requested policy window')
    actual, cmd = states[valid], commands[index[valid]]
    if np.any(np.abs(cmd[:, :, 0]) > 1e8) or np.any(cmd[:, :, 3] <= 0):
        raise ValueError('Tracking window is not a position-target interval')
    error = cmd[:, :, 0] - actual[:, :, 0]
    effort = cmd[:, :, 3] * error + cmd[:, :, 4] * (cmd[:, :, 1] - actual[:, :, 1]) + cmd[:, :, 2]
    return dict(samples=int(valid.sum()), rejected_stale_or_missing=int(in_window.sum()-valid.sum()),
                max_pair_age_us=int((st[valid]-ct[index[valid]]).max()),
                pooled_q_rmse_rad=float(np.sqrt(np.mean(error**2))),
                calf_q_rmse_rad=float(np.sqrt(np.mean(error[:, 2::3]**2))),
                max_abs_joint_speed_rad_s=float(np.max(np.abs(actual[:, :, 1]))),
                max_abs_predicted_effort_nm=float(np.max(np.abs(effort))),
                q_rmse_rad=dict(zip(JOINTS, np.sqrt(np.mean(error**2, axis=0)).tolist())),
                target_minus_measured_bias_rad=dict(zip(JOINTS, error.mean(axis=0).tolist())),
                q_error_p95_rad=dict(zip(JOINTS, np.percentile(np.abs(error), 95, axis=0).tolist())))


def run_review(root, port=8080, window=(0.35, 4.8)):
    root = root.resolve()
    manifest = read_json(root / 'release_manifest.json')
    tuning = read_json(root / 'tuning.json')
    terminal = (root / 'test/run/terminal.stdout').read_text()
    events = [json.loads(line) for line in (root / 'test/run/watchdog.log').read_text().splitlines() if line.strip()]
    result = dict(kind='archived_walking_run', source=root.name,
                  source_path_kind='artifact_directory_name', tuning=tuning,
                  run_identity={k: manifest['files'].get(k) for k in
                                ('policy_sender', 'policy/weights.npz', 'worker.py', 'estimator.json')},
                  exit=exit_review(events), hardware_ready=False, limitation=LIMITATION)
    result['evidence_sha256'] = {name: sha256(root/name) for name in
                                ('traffic.pcap','release_manifest.json','tuning.json',
                                 'test/run/terminal.stdout','test/run/watchdog.log','capture.log')}
    span = tagged(terminal, 'POLICY_SEND_SPAN')
    audit = tagged(terminal, 'SDK_AUDIT')
    result['policy_send_span'] = span
    result['sdk_audit'] = audit
    result['declared_duration_complete'] = 'policy_duration_complete_revoke' in result['exit']['stop_reasons']
    drops = re.findall(r'(\d+) packets dropped by kernel', (root/'capture.log').read_text())
    result['capture_drops'] = int(drops[-1]) if drops else None
    ct, commands, st, states = [], [], [], []
    flow_counts, rejected = Counter(), Counter()
    duplicate = 0
    last_tick = None
    for stamp, src, sp, dst, dp, payload in packets(root/'traffic.pcap'):
        flow = (src, sp, dst, dp)
        flow_counts[f'{src}:{sp} -> {dst}:{dp} len={len(payload)}'] += 1
        is_command = flow == ('192.168.123.161', port, '192.168.123.10', 8007)
        is_state = flow == ('192.168.123.10', 8007, '192.168.123.161', port)
        if not is_command and not is_state:
            continue
        try:
            values = decode_frame(payload, state=is_state)
        except ValueError as error:
            rejected[('state: ' if is_state else 'command: ') + str(error)] += 1
            continue
        if is_command:
            ct.append(stamp); commands.append(values)
        else:
            tick = struct.unpack_from('<I', payload, 755)[0]
            if tick == last_tick:
                duplicate += 1
                continue
            last_tick = tick
            st.append(stamp); states.append(values)
    result.update(flows=dict(flow_counts), rejected_frames=dict(rejected), duplicate_state_ticks=duplicate,
                  captured_custom_commands=len(ct), fresh_custom_states=len(st))
    # Exact captured-send count is required before using send ordinal for phase alignment.
    count = span['count']
    if len(ct) != audit['sent'] or any(k.startswith('command:') for k in rejected):
        raise ValueError('Captured SDK commands do not match sender audit; cannot anchor policy interval')
    if not 0 <= count <= len(ct):
        raise ValueError('Invalid policy send count')
    if count == 0:
        result['tracking'] = None
        return result
    if np.any(np.diff(ct) <= 0) or np.any(np.diff(st) < 0):
        raise ValueError('Nonmonotonic same-flow capture timestamps')
    policy_start = ct[-count]
    capture_span = ct[-1] - policy_start
    if abs(capture_span - span['span_ns']/1000) > 4000:
        raise ValueError('Capture and reported policy spans disagree by more than 4ms')
    result['policy_capture_alignment'] = dict(first_us=policy_start, last_us=ct[-1],
        method='Final policy-send-count packets; exact SDK-audit count and span checked. Host capture timestamps, not sensor delay.')
    end = min(ct[-1], policy_start + window[1]*1e6)
    result['tracking_window_s'] = [window[0], (end-policy_start)/1e6]
    if end <= policy_start + window[0]*1e6:
        result['tracking'] = None
    else:
        result['tracking'] = tracking_metrics(st, np.asarray(states), ct, np.asarray(commands),
            policy_start+window[0]*1e6, end)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--package', type=Path)
    group.add_argument('--run', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise ValueError('Output exists; use a new file')
        if not 1 <= args.port <= 65535:
            raise ValueError('Invalid port')
        result = package_review(args.package) if args.package else run_review(args.run, args.port)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        print('report=' + str(args.out.resolve()))
        if args.package:
            print('local_package_integrity=' + ('PASS' if result['integrity_passed'] else 'FAIL'))
            return 0 if result['integrity_passed'] else 2
        print('recorded_exit_before_restore=' + str(result['exit']['exit_before_restore']))
        print('declared_duration_complete=' + str(result['declared_duration_complete']))
        if result['tracking']:
            print('position_rmse_rad=' + str(result['tracking']['pooled_q_rmse_rad']))
        return 0
    except (OSError, ValueError, KeyError, struct.error) as error:
        parser.exit(2, 'STOP: ' + str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
