#!/usr/bin/env python3
"""Offline MoCap trunk velocity labels with explicit mounting and clock mapping.

No fitting of estimator parameters. Python 3.8+ and NumPy required. All invalid
poses/gaps remain visible; centered derivatives are offline reference labels.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration_protocols import quaternion_xyzw, valid_rotation


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def reference(mocap, config):
    """Convert marker poses to body-origin velocities at the middle of each window."""
    mount = np.asarray(config['R_marker_body'], dtype=float)
    offset = np.asarray(config['body_to_marker_position_body_m'], dtype=float)
    if (mount.shape != (3, 3) or not valid_rotation(mount, 1e-6)
            or offset.shape != (3,) or not np.isfinite(offset).all()):
        raise ValueError('supply measured R_marker_body and body-to-marker offset')
    clock = config['clock']
    numbers = [clock.get(k) for k in ('qtm_origin_s', 'pi_origin_s', 'scale', 'uncertainty_s')]
    if (any(v is None for v in numbers) or not np.isfinite(numbers).all()
            or numbers[2] <= 0 or numbers[3] < 0 or not clock.get('provenance')):
        raise ValueError('supply clock anchors, positive scale, uncertainty and provenance')
    qo, po, scale, uncertainty = numbers
    gap = float(config['max_reference_gap_s'])
    window = config.get('derivative_window_samples', 5)
    if not np.isfinite(gap) or gap <= 0 or type(window) is not int or window < 3 or window % 2 != 1:
        raise ValueError('need positive gap and odd derivative window >= 3')
    radius = window // 2
    rows = []
    previous = None
    for source in mocap:
        if 'qtm_timestamp_us' not in source:
            continue
        t = (source['qtm_timestamp_us']/1e6-qo)*scale+po
        if previous is not None and t <= previous:
            raise ValueError('QTM timestamp reset/reorder: split or repair the recording before export')
        previous = t
        good = source.get('valid') is True and source.get('ordered') is True
        row = dict(t_s=t, frame_number=source['frame_number'], valid=False,
                   pose_valid=False, velocity_body_m_s=None, reason='invalid_pose',
                   clock_uncertainty_s=uncertainty,
                   label_at_receive=source.get('label_at_receive'))
        if good:
            r = np.asarray(source['R_world_marker'], dtype=float) @ mount
            p = np.asarray(source['position_world_m'], dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all() or not valid_rotation(r):
                raise ValueError('invalid recorded pose')
            row.update(pose_valid=True, R_world_body=r.tolist(),
                       position_body_origin_world_m=(p-r@offset).tolist(),
                       position_world_m=p.tolist(), quaternion_xyzw=source['quaternion_xyzw'],
                       quaternion_body_xyzw=quaternion_xyzw(r), reason='derivative_endpoint')
        rows.append(row)
    if len(rows) < window:
        raise ValueError('not enough MoCap frames for derivative window')
    for i in range(radius, len(rows)-radius):
        selected = rows[i-radius:i+radius+1]
        row = rows[i]
        if not all(r['pose_valid'] for r in selected):
            row['reason'] = 'invalid_pose_in_derivative_window'
            continue
        t = np.array([r['t_s']-row['t_s'] for r in selected])
        if np.max(np.diff(t)) > gap:
            row['reason'] = 'reference_gap'
            continue
        # Correct marker-origin position before differentiating rotating offsets.
        span = np.max(np.abs(t))
        u = t/span
        design = np.stack([np.ones(len(t)), u, u*u], axis=1)
        positions = np.array([r['position_body_origin_world_m'] for r in selected])
        coefficients = np.linalg.lstsq(design, positions-positions[radius], rcond=None)[0]
        world_v = coefficients[1]/span
        row.update(valid=True, reason='ok', velocity_world_m_s=world_v.tolist(),
                   velocity_body_m_s=(np.asarray(row['R_world_body']).T@world_v).tolist(),
                   derivative_span_s=float(t[-1]-t[0]))
    return rows


def join_robot(robot, ref, max_gap_s):
    times = [r['t_s'] for r in ref]
    for row in robot:
        t = row['pi_sample_monotonic_ns']/1e9
        label = dict(valid=False, reason='no_bracketing_reference', velocity_body_m_s=None)
        j = bisect.bisect_left(times, t)
        indices = [j] if j < len(times) and abs(times[j]-t) < 1e-9 else [j-1, j]
        if all(0 <= k < len(ref) for k in indices):
            selected = [ref[k] for k in indices]
            if all(r['valid'] for r in selected) and times[indices[-1]]-times[indices[0]] <= max_gap_s:
                if len(indices) == 1:
                    velocity = selected[0]['velocity_body_m_s']
                else:
                    alpha = (t-selected[0]['t_s'])/(selected[1]['t_s']-selected[0]['t_s'])
                    velocity = ((1-alpha)*np.array(selected[0]['velocity_body_m_s'])
                                + alpha*np.array(selected[1]['velocity_body_m_s'])).tolist()
                label = dict(valid=True, reason='ok', velocity_body_m_s=velocity,
                             clock_uncertainty_s=max(r['clock_uncertainty_s'] for r in selected))
            else:
                label['reason'] = 'invalid_or_gapped_reference'
        if not row['valid']:
            label.update(valid=False, reason='invalid_robot_sample', velocity_body_m_s=None)
        yield dict(t_s=t, robot=row, reference=label)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.calibration.read_text())
    ref = reference(read_rows(a.recording/'mocap.jsonl'), config)
    if not any(r['valid'] for r in ref):
        raise ValueError('no valid reference velocities; inspect tracking/gaps/window')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'calibration.json').write_text(json.dumps(config, indent=2)+'\n')
    with (a.out/'reference_velocity.jsonl').open('x') as stream:
        for row in ref:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    # Pose schema accepted by the policy project’s compare_mocap_velocity.py. Its clock
    # config must be identity because t_s here is already mapped to Pi time.
    with (a.out/'mocap_poses.jsonl').open('x') as stream:
        for row in ref:
            stream.write(json.dumps(dict(t_s=row['t_s'], valid=row['pose_valid'],
                                         position_world_m=row.get('position_world_m'),
                                         quaternion_xyzw=row.get('quaternion_xyzw')), allow_nan=False)+'\n')
    count = valid = 0
    with (a.out/'aligned_samples.jsonl').open('x') as stream, (a.recording/'robot.jsonl').open() as robot_file:
        robot_rows = (json.loads(line) for line in robot_file if line.strip())
        for row in join_robot(robot_rows, ref, config['max_reference_gap_s']):
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            count += 1
            valid += row['reference']['valid']
    summary = dict(robot_samples=count, matched_velocity_labels=valid,
                   reference_frames=len(ref), valid_reference_velocities=sum(r['valid'] for r in ref),
                   calibration_sha256=hashlib.sha256(a.calibration.read_bytes()).hexdigest(),
                   recording=a.recording.resolve().name,
                   recording_path_kind='artifact_directory_name',
                   recording_sha256={name: sha256_file(a.recording/name)
                                     for name in ('robot.jsonl', 'mocap.jsonl')},
                   method='centered local quadratic derivative; measured mount and supplied clock map',
                   estimator_parameters_fitted=False)
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
