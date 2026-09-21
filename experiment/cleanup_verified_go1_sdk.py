#!/usr/bin/env python3
"""Remove only the known archived SDK scratch tree after fresh verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.prepare_go1_cleanup import SDK, ROOT, REMOTE, verify_archive


def ensure_unused():
    # Refuse use by a process owned by pi (including active compiler cwd).
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for link in ('cwd', 'exe'):
                try:
                    target = os.readlink(str(proc/link))
                except PermissionError:
                    # Read-only privilege fallback; deletion remains unprivileged.
                    result = subprocess.run(
                        ['sudo', '-n', 'readlink', '--', str(proc/link)],
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if result.returncode:
                        if not proc.exists():
                            break  # Process exited during inspection.
                        raise ValueError('cannot inspect PID '+proc.name+' '+link+
                                         '; read-only sudo inspection failed; nothing deleted')
                    target = result.stdout.rstrip('\n')
                if target == SDK or target.startswith(SDK+'/'):
                    raise ValueError('SDK directory in use by PID '+proc.name)
        except (FileNotFoundError, ProcessLookupError):
            continue

def remove_verified_files(root, expected, execute=False):
    # This function is also sent to the Pi with the manifest() implementation.
    root = Path(root)
    if str(root) != SDK or root.is_symlink() or str(root.resolve()) != SDK:
        raise ValueError('refusing non-approved SDK root')
    if not root.exists():
        return 'SDK scratch directory already absent; nothing deleted'
    if not expected or any(v['kind'] != 'file' for v in expected.values()):
        raise ValueError('expected a nonempty regular-file-only manifest')
    if manifest(str(root)) != expected:
        raise ValueError('Pi files changed since backup; nothing deleted')
    ensure_unused()
    if not execute:
        return 'PREVIEW: verified '+str(len(expected))+' files under '+SDK
    # No recursive force-delete: remove only the verified files, then empty dirs.
    for name, meta in expected.items():
        path = root/name
        if not path.is_file() or path.is_symlink() or not str(path.resolve()).startswith(SDK+'/'):
            raise ValueError('file changed before removal: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('hash changed before removal: '+name)
        path.unlink()
    for base, dirs, files in os.walk(str(root), topdown=False):
        Path(base).rmdir()  # Fails if any unexpected new file remains.
    return 'Deleted verified SDK scratch directory: '+SDK


# Reuse the same manifest function that produced the archive inventory.
MANIFEST_SOURCE = REMOTE[:REMOTE.index("\nresult={'disk'")]
exec(MANIFEST_SOURCE, globals())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    try:
        archive=args.archive.resolve(strict=True)
        if not archive.is_relative_to((ROOT/'logs/pi-cleanup').resolve()):
            raise ValueError('archive must be beneath repo logs/pi-cleanup')
        before=json.loads((archive/'inventory-before.json').read_text())['paths'][SDK]['manifest']
        after=json.loads((archive/'inventory-after.json').read_text())['paths'][SDK]['manifest']
        if before != after: raise ValueError('backup manifests disagree')
        backup=archive/'sdk-staging.tar.gz'
        checksum=(archive/'backup.sha256').read_text().split()[0]
        if hashlib.sha256(backup.read_bytes()).hexdigest()!=checksum:
            raise ValueError('backup checksum mismatch')
        verify_archive(backup,after)
        import inspect
        script=(MANIFEST_SOURCE+'\nSDK='+repr(SDK)+'\nimport sys\n'+
                inspect.getsource(ensure_unused)+
                inspect.getsource(remove_verified_files)+
                '\ntry:\n    expected=json.load(sys.stdin)\n    print(remove_verified_files(SDK,expected,'+
                repr(args.execute)+'))\nexcept (OSError, ValueError, KeyError) as error:\n'+
                '    sys.exit(\"STOP: \"+str(error))\n')
        result=subprocess.run(['ssh','-o','ConnectTimeout=10','pi@192.168.12.1',
                               'python3 -c '+shlex.quote(script)],
                              input=json.dumps(after),text=True,stdout=subprocess.PIPE,check=True)
        print(result.stdout.strip());print('Ubuntu backup retained: '+str(backup))
        with (archive/'sdk-cleanup-history.txt').open('a') as stream:
            stream.write(('EXECUTE' if args.execute else 'PREVIEW')+'\n'+result.stdout)
        return 0
    except subprocess.CalledProcessError as error:
        parser.exit(2, 'STOP: Pi cleanup failed (exit '+str(error.returncode)+
                    '); see the message above. Ubuntu backup retained.\n')
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2,'STOP: '+str(error)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
