import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from experiment.prepare_go1_cleanup import SDK, verify_archive


class BackupTests(unittest.TestCase):
    def archive(self, root, name, content=b'known code'):
        path=root/'backup.tar.gz'
        with tarfile.open(path,'w:gz') as tar:
            member=tarfile.TarInfo(name);member.size=len(content)
            tar.addfile(member,io.BytesIO(content))
        return path

    def test_exact_backup_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.archive(Path(folder),Path(SDK).name+'/src/a.cpp')
            expected={'src/a.cpp':{'kind':'file','size':10,'sha256':hashlib.sha256(b'known code').hexdigest()}}
            verify_archive(p,expected)
            expected['src/a.cpp']['sha256']='0'*64
            with self.assertRaises(ValueError):verify_archive(p,expected)

    def test_outside_paths_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('/etc/passwd',Path(SDK).name+'/../escape','different-root/a'):
                p=self.archive(Path(folder),name)
                with self.assertRaises(ValueError):verify_archive(p,{})

    def test_missing_files_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.archive(Path(folder),Path(SDK).name+'/unexpected')
            with self.assertRaises(ValueError):verify_archive(p,{})


if __name__=='__main__':unittest.main()
