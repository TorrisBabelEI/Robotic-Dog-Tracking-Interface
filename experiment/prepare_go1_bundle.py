#!/usr/bin/env python3
"""Create/verify an immutable selected-scope source bundle; no SSH or motor I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'CMakeLists.txt', 'LICENSE',
    'src/go1_lowlevel_experiment.cpp', 'src/go1_kinematics.cpp',
    'src/go1_kinematics.hpp', 'src/go1_log_file.hpp',
    'src/go1_operator_support.hpp', 'src/go1_operator_support_probe.cpp',
    'src/go1_sdk_receive.hpp', 'src/go1_sdk_transport.hpp',
    'experiment/decode_native_go1_pcap.py', 'experiment/review_factory_effort.py',
    'experiment/analyze_lowlevel_log.py',
]
TESTS = ['go1_sdk_command_adapter_test', 'go1_sdk_transport_test',
    'go1_standing_capture_test', 'go1_ground_entry_test', 'go1_ground_exit_test',
    'go1_prone_low_rise_test', 'go1_operator_support_test',
    'go1_operator_support_server_test', 'go1_log_file_test', 'go1_kinematics_test']
FILES += ['test/'+name+'.cpp' for name in TESTS]
FILES += ['test/test_factory_effort_review.py', 'test/test_decode_native_go1_pcap.py']
SCOPE = ['sdk_transport_checks', 'offline_factory_effort_analysis', 'deployment_evidence']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out, root=ROOT):
    if out.exists():raise ValueError('Bundle output already exists; use a new directory')
    files = FILES + sorted(str(p.relative_to(root)) for p in
        (root/'externals/unitree_legged_sdk/include').rglob('*.h'))
    files += ['externals/unitree_legged_sdk/lib/cpp/'+arch+'/libunitree_legged_sdk.a'
              for arch in ('amd64','arm64')]
    # Read each approved source once; copy those exact bytes and verify stability.
    payload = {}
    for name in files:
        path=root/name
        if path.is_symlink() or not path.is_file():raise ValueError('Missing/nonregular source: '+name)
        payload[name]=path.read_bytes()
    out.mkdir(parents=True)
    hashes={}
    for name,data in payload.items():
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
        if digest(root/name)!=hashes[name]:raise ValueError('Source changed during snapshot: '+name)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    manifest={'schema':1,'scope':SCOPE,'git_revision':revision,
        'source_sha256':hashes,'policy_development':False,'hardware_qualified':False,
        'build_flags':['-DGO1_ENABLE_POLICY_DEVELOPMENT=OFF','-DPYTHON_BUILD=OFF',
                       '-DBUILD_SDK_EXAMPLES=OFF','-DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE'],
        'note':'Per-file hashes identify working-tree bytes, including uncommitted changes. Prior binary acceptance does not qualify this binary.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Verify the manifest itself on target alongside the selected payload.
    checks={**hashes,'manifest.json':digest(out/'manifest.json')}
    (out/'source.sha256').write_text(''.join(f'{sha}  {name}\n' for name,sha in sorted(checks.items())))
    return verify(out)


def verify(out):
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['schema']!=1 or manifest['scope']!=SCOPE or manifest['policy_development'] is not False:
        raise ValueError('Unexpected bundle scope')
    for name,expected in manifest['source_sha256'].items():
        path=out/name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(out.resolve()):
            raise ValueError('Invalid bundle pathname')
        if path.is_symlink() or not path.is_file() or digest(path)!=expected:
            raise ValueError('Bundle checksum mismatch: '+name)
    expected_lines={**manifest['source_sha256'],'manifest.json':digest(out/'manifest.json')}
    if (out/'source.sha256').read_text()!=''.join(f'{sha}  {name}\n' for name,sha in sorted(expected_lines.items())):
        raise ValueError('Bundle checksum inventory mismatch')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    try:
        result=verify(args.out) if args.verify else build(args.out)
        print('source_bundle=VERIFIED; files='+str(len(result['source_sha256'])))
        print('bundle='+str(args.out.resolve()))
        return 0
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
        parser.exit(2,'STOP: '+str(error)+'\n')

if __name__=='__main__':raise SystemExit(main())
