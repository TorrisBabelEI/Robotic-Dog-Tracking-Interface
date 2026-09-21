import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from experiment import cleanup_verified_go1_sdk as cleanup


class SDKCleanupTests(unittest.TestCase):
    def test_preview_then_exact_removal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'sdk';root.mkdir();(root/'a').write_bytes(b'code')
            expected=cleanup.manifest(str(root))
            with mock.patch.object(cleanup,'SDK',str(root)), mock.patch.object(cleanup,'ensure_unused'):
                cleanup.remove_verified_files(root,expected)
                self.assertTrue((root/'a').exists())
                cleanup.remove_verified_files(root,expected,True)
                self.assertFalse(root.exists())

    def test_changed_or_added_file_stops_before_deletion(self):
        for added in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'sdk';root.mkdir();(root/'a').write_bytes(b'code')
                expected=cleanup.manifest(str(root))
                (root/('extra' if added else 'a')).write_bytes(b'changed')
                with mock.patch.object(cleanup,'SDK',str(root)), mock.patch.object(cleanup,'ensure_unused'):
                    with self.assertRaises(ValueError):cleanup.remove_verified_files(root,expected,True)
                self.assertTrue((root/'a').exists())

    def test_active_or_uninspectable_process_blocks_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'sdk';root.mkdir();(root/'a').write_bytes(b'code')
            expected=cleanup.manifest(str(root))
            with mock.patch.object(cleanup,'SDK',str(root)), mock.patch.object(
                    cleanup,'ensure_unused',side_effect=PermissionError('cannot inspect active process')):
                with self.assertRaises(PermissionError):cleanup.remove_verified_files(root,expected,True)
            self.assertTrue((root/'a').exists())

    def test_protected_process_readonly_fallback(self):
        for target, status, fails in [('/usr/bin/service', 0, False),
                                      (cleanup.SDK+'/build/test', 0, True),
                                      ('', 1, True)]:
            proc = mock.MagicMock()
            proc.name = '686'
            proc.stat.return_value.st_uid = cleanup.os.getuid()
            proc.__truediv__.side_effect = lambda link: Path('/proc/686')/link
            result = mock.Mock(returncode=status, stdout=target+'\n')
            with mock.patch.object(Path, 'iterdir', return_value=[proc]), \
                 mock.patch.object(cleanup.os, 'readlink', side_effect=PermissionError()), \
                 mock.patch.object(cleanup.subprocess, 'run', return_value=result) as run:
                if fails:
                    with self.assertRaises(ValueError): cleanup.ensure_unused()
                else:
                    cleanup.ensure_unused()
                self.assertEqual(run.call_args.args[0][:4], ['sudo', '-n', 'readlink', '--'])

    def test_wrong_root_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'sdk';root.mkdir();link=Path(folder)/'link';link.symlink_to(root)
            with mock.patch.object(cleanup,'SDK',str(link)):
                with self.assertRaises(ValueError):cleanup.remove_verified_files(link,{},True)
            with self.assertRaises(ValueError):cleanup.remove_verified_files(root,{},True)


if __name__=='__main__':unittest.main()
