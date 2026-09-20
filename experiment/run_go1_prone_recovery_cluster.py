#!/usr/bin/env python3
"""Offline acceptance of the revised prone return/recovery chain; no SDK."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parent = ROOT/'logs'/'prone-recovery-software'
    parent.mkdir(parents=True, exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix='review-', dir=parent))
    print(f'archive={archive}', flush=True)
    def run(command, name):
        with (archive/(name+'.txt')).open('w') as stream:
            status = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if status.returncode:
            raise ValueError(f'{name} exited {status.returncode}; see {archive/(name+".txt")}')
    try:
        files = ['src/go1_lowlevel_experiment.cpp', 'src/go1_kinematics.cpp',
                 'src/go1_kinematics.hpp', 'src/go1_log_file.hpp',
                 'src/go1_operator_support.hpp',
                 'test/go1_prone_low_rise_test.cpp', 'CMakeLists.txt',
                 'experiment/run_go1_prone_recovery_cluster.py',
                 'experiment/analyze_lowlevel_log.py']
        manifest = {'data_kind': 'SIMULATED', 'revision': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
        (archive/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        with tempfile.TemporaryDirectory(prefix='go1-prone-recovery-') as build:
            run(['cmake', '-S', str(ROOT), '-B', build, '-DBUILD_TESTING=ON',
                 '-DBUILD_SDK_EXAMPLES=OFF', '-DPYTHON_BUILD=OFF',
                 '-DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE'], 'configure')
            run(['cmake', '--build', build, '--target', 'go1_lowlevel_simulator',
                 'go1_prone_low_rise_test', '-j2'], 'build')
            run([str(Path(build)/'go1_prone_low_rise_test')], 'recovery_core')
            print('PASS revised recovery, release, cancellation and fault core checks', flush=True)
            for name, injection in [('normal', []),
                                    ('cancel', ['--inject-soft-stop-s', '8']),
                                    ('remote_stop', ['--inject-panic-s', '8']),
                                    ('watchdog', ['--inject-watchdog-s', '8'])]:
                log = archive/(name+'.csv')
                command = [str(Path(build)/'go1_lowlevel_simulator'), '--dry-run',
                           '--mode', 'prone-low-rise', '--log', str(log)] + injection
                (archive/(name+'.command.json')).write_text(json.dumps(command)+'\n')
                run(command, name+'.run')
                with log.open() as stream:
                    rows = list(csv.DictReader(stream))
                phases = [r['phase'] for i,r in enumerate(rows)
                          if i == 0 or r['phase'] != rows[i-1]['phase']]
                reasons = {r['abort_reason'] for r in rows if r['abort_reason']}
                expected = {'remote_stop': {'remote_l2_b'},
                            'watchdog': {'command_watchdog_over_20ms'}}.get(name, set())
                if not phases or phases[-1] != 'COMPLETE' or reasons != expected:
                    raise ValueError(f'{name}: incomplete or unexpected fault: {phases}, {reasons}')
                required = (['PANIC_DAMPING'] if expected else
                            ['PRONE_RETURN', 'PRONE_SETTLE', 'PRONE_RELEASE', 'PRONE_FINAL_DAMPING'])
                if any(p not in phases for p in required) or 'PRONE_SUPPORT_HOLD' in phases:
                    raise ValueError(f'{name}: unexpected recovery sequence: {phases}')
                if name == 'cancel' and not any(r['stop_source'] == 'ctrl_c' for r in rows):
                    raise ValueError('cancellation was not recorded')
                for leg in ('FR', 'FL', 'RR', 'RL'):
                    for axis in range(3):
                        j = f'{leg}_{axis}'
                        if (float(rows[-1][j+'_cmd_kp']) != 0 or
                            float(rows[-1][j+'_cmd_kd']) != 1 or
                            float(rows[-1][j+'_tau_ff']) != 0 or
                            float(rows[-1][j+'_cmd_q']) <= 1e8):
                            raise ValueError(f'{name}: final command is not damping')
                (archive/(name+'.phases.txt')).write_text(' -> '.join(phases)+'\n')
                run([sys.executable, '-B', str(ROOT/'experiment/analyze_lowlevel_log.py'),
                     str(log), '--no-plots'], name+'.analysis')
                print(f'PASS {name}: complete sequence and final damping', flush=True)
        (archive/'result.txt').write_text('prone_recovery_cluster=PASS\nSIMULATED; hardware remains locked\n')
        print(f'prone_recovery_cluster=PASS\narchive={archive}', flush=True)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        (archive/'result.txt').write_text(f'prone_recovery_cluster=FAIL\n{error}\n')
        print(f'STOP: {error}\narchive={archive}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
