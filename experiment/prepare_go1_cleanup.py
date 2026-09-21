#!/usr/bin/env python3
"""Operator-run read-only Pi inventory and verified SDK scratch backup. Never deletes."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SDK = '/home/pi/go1-sdk-adapter-review-68mDJRKk'
REMOTE = r'''
import os, json, hashlib, stat, shutil, subprocess
from pathlib import Path
roots=['/home/pi/go1-sdk-adapter-review-68mDJRKk', '/home/pi/go1-prone-engagement', '/home/pi/Robotic-Dog-Tracking-Interface']
def manifest(root):
    root=Path(root)
    if root.is_symlink() or str(root.resolve())!=str(root):
        raise ValueError('refusing symlink root: '+str(root))
    result={}
    for base, dirs, files in os.walk(str(root), followlinks=False):
        for name in dirs+files:
            p=Path(base)/name; info=p.lstat(); key=str(p.relative_to(root))
            if stat.S_ISLNK(info.st_mode): result[key]={'kind':'symlink','target':os.readlink(str(p))}
            elif stat.S_ISREG(info.st_mode):
                digest=hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
                result[key]={'kind':'file','size':info.st_size,'sha256':digest.hexdigest()}
            elif not stat.S_ISDIR(info.st_mode): raise ValueError('special file: '+str(p))
    return result
result={'disk':dict(zip(['total','used','free'],shutil.disk_usage('/'))),'paths':{}}
for root in roots:
    p=Path(root)
    item={'exists':p.exists(),'symlink':p.is_symlink()}
    if p.exists() and not p.is_symlink():
        item['top_level']=sorted(x.name for x in p.iterdir())
        item['du']=subprocess.check_output(['du','-sk','--',root],universal_newlines=True).strip()
        if root==roots[0]: item['manifest']=manifest(root)
        logs=p/'logs'
        if logs.is_dir() and not logs.is_symlink():
            item['logs']=[{'path':str(x),'bytes':x.lstat().st_size} for x in logs.rglob('*') if x.is_file() and not x.is_symlink()]
    result['paths'][root]=item
print(json.dumps(result,sort_keys=True))
'''


def ssh(command, **kwargs):
    return subprocess.run(['ssh','-o','ConnectTimeout=10','pi@192.168.12.1', command],
                          check=True, **kwargs)


def inventory():
    return json.loads(ssh('python3 -c '+shlex.quote(REMOTE), stdout=subprocess.PIPE, text=True).stdout)


def verify_archive(path, expected):
    actual = {}
    with tarfile.open(path, 'r:gz') as tar:
        for entry in tar:
            parts = Path(entry.name).parts
            if not parts or parts[0] != Path(SDK).name or '..' in parts or entry.name.startswith('/'):
                raise ValueError('unexpected archive path')
            key = str(Path(*parts[1:])) if len(parts)>1 else '.'
            if entry.isdir(): continue
            if entry.isfile():
                digest=hashlib.sha256()
                with tar.extractfile(entry) as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
                value={'kind':'file','size':entry.size,'sha256':digest.hexdigest()}
            elif entry.issym(): value={'kind':'symlink','target':entry.linkname}
            else: raise ValueError('unsupported archive member')
            if key in actual: raise ValueError('duplicate archive entry')
            actual[key]=value
    if actual != expected: raise ValueError('backup does not match Pi manifest')


def main():
    parent=ROOT/'logs/pi-cleanup';parent.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='review-',dir=parent))
    print('archive='+str(out),flush=True)
    try:
        before=inventory();(out/'inventory-before.json').write_text(json.dumps(before,indent=2))
        entry=before['paths'][SDK]
        if entry['exists']:
            if entry['symlink'] or 'manifest' not in entry: raise ValueError('SDK candidate is not a regular directory')
            command='tar -C /home/pi -czf - -- '+shlex.quote(Path(SDK).name)
            (out/'backup-command.txt').write_text(command+'\n')
            partial=out/'sdk-staging.tar.gz.partial'
            with partial.open('wb') as f: ssh(command,stdout=f)
            verify_archive(partial,entry['manifest'])
            after=inventory();(out/'inventory-after.json').write_text(json.dumps(after,indent=2))
            if after['paths'][SDK].get('manifest')!=entry['manifest']:
                raise ValueError('Pi staging changed during backup')
            backup=out/'sdk-staging.tar.gz';partial.rename(backup)
            (out/'backup.sha256').write_text(hashlib.sha256(backup.read_bytes()).hexdigest()+'  sdk-staging.tar.gz\n')
            print('SDK staging backup verified file-by-file',flush=True)
        plan={'status':'PREPARATION ONLY; nothing deleted',
              'code_candidate':SDK if entry['exists'] else None,
              'keep':['/home/pi/Unitree','/home/pi/Robotic-Dog-Tracking-Interface','/home/pi/go1-prone-engagement'],
              'next':'Review inventory and verified backup before preparing exact code-removal commands. Use the single-file verified cleanup tool for already archived logs.'}
        (out/'plan.json').write_text(json.dumps(plan,indent=2))
        print(json.dumps(plan,indent=2)); print('archive='+str(out))
        return 0
    except (OSError,ValueError,subprocess.SubprocessError,tarfile.TarError) as error:
        (out/'error.txt').write_text(str(error)+'\n');print('STOP: '+str(error));return 1


if __name__=='__main__':
    raise SystemExit(main())
