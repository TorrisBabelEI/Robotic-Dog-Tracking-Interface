#!/usr/bin/env python3
"""Build and verify the torque command/analysis chain; simulated data only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [f'{leg}_{axis}' for leg in ('FR', 'FL', 'RR', 'RL') for axis in range(3)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(rows, joint, amplitude, frequency, duration, fault=None):
    """Check signal generation and complete phase/fault behavior, not just exit 0."""
    require(bool(rows), 'empty log')
    phases = [r['phase'] for i, r in enumerate(rows)
              if i == 0 or r['phase'] != rows[i-1]['phase']]
    expected = ['PRECHECK', 'CAPTURE_POSE', 'HOLD', 'TORQUE_EXCITE',
                'RETURN', 'SAFE_HOLD', 'COMPLETE']
    reasons = {r['abort_reason'] for r in rows if r['abort_reason']}
    if fault in ('remote_l2_b', 'double_ctrl_c', 'command_watchdog_over_20ms'):
        require(reasons == {fault}, f'unexpected fault reasons: {reasons}')
        require('TORQUE_EXCITE' in phases and 'PANIC_DAMPING' in phases,
                f'missing excitation or panic: {phases}')
        panic = [r for r in rows if r['phase'] == 'PANIC_DAMPING']
        for r in panic:
            for j in JOINTS:
                require(float(r[j+'_cmd_kp']) == 0 and
                        float(r[j+'_cmd_kd']) == 1 and
                        float(r[j+'_tau_ff']) == 0 and
                        float(r[j+'_cmd_q']) > 1e8,
                        'panic failed to publish position-free damping')
        require(phases[-1] == 'COMPLETE', 'panic did not complete')
        latency = (int(panic[0]['host_monotonic_ns']) -
                   int(rows[0]['host_monotonic_ns'])) / 1e9 - 4.0
        require(0 <= latency <= 0.02, f'late injected fault response: {latency}s')
    else:
        require(not reasons, f'unexpected abort: {reasons}')
        require(phases == expected, f'unexpected phase sequence: {phases}')
        if fault == 'ctrl_c':
            require(any(r['stop_source'] == 'ctrl_c' for r in rows),
                    'cancellation not recorded')
            returned = next(r for r in rows if r['phase'] == 'RETURN')
            latency = (int(returned['host_monotonic_ns']) -
                       int(rows[0]['host_monotonic_ns'])) / 1e9 - 4.0
            require(0 <= latency <= 0.02, 'late injected cancellation response')
    action = [r for r in rows if r['phase'] == 'TORQUE_EXCITE']
    require(len(action) > 100, 'missing excitation data')
    # Phase labels change on the final HOLD row. Omit it; elapsed time starts
    # at that row. The boundary row entering RETURN contains the final command.
    start = int(action[0]['host_monotonic_ns'])
    q0 = float(next(r for r in rows if r['phase'] == 'HOLD')[joint+'_cmd_q'])
    errors = []
    for r in action[1:]:
        t = (int(r['host_monotonic_ns']) - start) / 1e9
        ramp = min(0.05, duration / 2)
        x = min(1.0, max(0.0, min(t, duration-t) / ramp))
        envelope = x*x*x*(10+x*(-15+6*x))
        sine = amplitude * envelope * math.sin(2*math.pi*frequency*t)
        expected_command = sine + 2*(q0-float(r[joint+'_state_q'])) - 0.2*float(r[joint+'_state_dq'])
        require(float(r[joint+'_cmd_kp']) == 0 and float(r[joint+'_cmd_kd']) == 0,
                'selected joint is not in torque command mode')
        errors.append(abs(float(r[joint+'_tau_ff']) - expected_command))
        for other in JOINTS:
            if other != joint:
                require(float(r[other+'_tau_ff']) == 0,
                        'torque excitation leaked to another joint')
    require(max(errors) < 2e-5, f'torque waveform mismatch: {max(errors)} Nm')
    if fault is None:
        require(abs(len(action)*0.002-duration) < 0.01, 'truncated excitation')
    return {'phase_sequence': ' -> '.join(phases),
            'waveform_max_error_nm': max(errors)}


def main():
    parent = ROOT / 'logs' / 'torque-software'
    parent.mkdir(parents=True, exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix='review-', dir=parent))
    print(f'archive={archive}', flush=True)
    records = []
    def run(args, name):
        with (archive / (name+'.txt')).open('w') as out:
            result = subprocess.run(args, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        require(result.returncode == 0, f'{name} failed ({result.returncode}); see {archive / (name+".txt")}')
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        sources = ['CMakeLists.txt', 'src/go1_lowlevel_experiment.cpp',
                   'src/go1_kinematics.cpp', 'src/go1_kinematics.hpp',
                   'src/go1_log_file.hpp', 'experiment/run_go1_torque_cluster.py',
                   'experiment/analyze_lowlevel_log.py']
        metadata = {'data_kind': 'SIMULATED', 'revision': revision,
                    'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
        (archive/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
        with tempfile.TemporaryDirectory(prefix='go1-torque-build-') as build:
            run(['cmake', '-S', str(ROOT), '-B', build, '-DBUILD_TESTING=OFF',
                 '-DBUILD_SDK_EXAMPLES=OFF', '-DPYTHON_BUILD=OFF',
                 '-DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE'], 'configure')
            run(['cmake', '--build', build, '--target', 'go1_lowlevel_simulator', '-j2'], 'build')
            binary = str(Path(build)/'go1_lowlevel_simulator')
            cases = [(f'a{a}_f{f}', a, f, 6/f, None, [])
                     for a in (0.10, 0.20) for f in (0.5, 1.0, 2.0)]
            cases += [(name, .10, 1.0, 6.0, reason, [flag, '4']) for name, reason, flag in (
                ('cancel', 'ctrl_c', '--inject-soft-stop-s'),
                ('remote_stop', 'remote_l2_b', '--inject-panic-s'),
                ('double_ctrl_c', 'double_ctrl_c', '--inject-double-ctrl-c-s'),
                ('watchdog', 'command_watchdog_over_20ms', '--inject-watchdog-s'))]
            for name, amplitude, frequency, duration, fault, injection in cases:
                print(f'Running {name} (simulation)', flush=True)
                log = archive/(name+'.csv')
                command = [binary, '--dry-run', '--mode', 'torque-sine', '--joint', 'FR_1',
                           '--amplitude-nm', str(amplitude), '--frequency-hz', str(frequency),
                           '--duration-s', str(duration), '--log', str(log)] + injection
                (archive/(name+'.command.json')).write_text(json.dumps(command)+'\n')
                run(command, name+'.run')
                with log.open() as stream:
                    rows = list(csv.DictReader(stream))
                checks = validate(rows, 'FR_1', amplitude, frequency, duration, fault)
                run([sys.executable, '-B', str(ROOT/'experiment/analyze_lowlevel_log.py'),
                     str(log), '--joint', 'FR_1', '--no-plots'], name+'.analysis')
                with Path(str(log)+'.summary.csv').open() as stream:
                    metric = next(r for r in csv.DictReader(stream) if r['joint'] == 'FR_1')
                record = dict(case=name, data_kind='SIMULATED', amplitude_nm=amplitude,
                              frequency_hz=frequency, duration_s=duration, result='PASS', **checks)
                for field in ('tau_rmse_nm', 'tau_bias_nm', 'tau_corr_total', 'tau_gain_total',
                              'lag_s', 'dominant_frequency_hz', 'frequency_gain', 'frequency_phase_deg'):
                    require(math.isfinite(float(metric[field])), f'{name}: nonfinite {field}')
                    record[field] = metric[field]
                records.append(record)
                (archive/(name+'.checks.json')).write_text(json.dumps(record, indent=2)+'\n')
                print(f'PASS {name}: command waveform and phase/fault checks', flush=True)
        with (archive/'torque_matrix.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (archive/'result.txt').write_text('torque_software_cluster=PASS\n10/10 cases\nSIMULATED: no hardware torque-tracking acceptance\n')
        print(f'torque_software_cluster=PASS (10/10)\narchive={archive}', flush=True)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        (archive/'result.txt').write_text(f'torque_software_cluster=FAIL\n{error}\n')
        print(f'STOP: {error}\narchive={archive}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
