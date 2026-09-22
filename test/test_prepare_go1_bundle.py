import json
from pathlib import Path
import tempfile
import unittest
from experiment.prepare_go1_bundle import build, verify

class BundleTests(unittest.TestCase):
    def test_snapshot_scope_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'bundle'
            manifest=build(root)
            self.assertFalse(manifest['policy_development'])
            self.assertFalse(any('policy_' in n or 'command_owner' in n or 'Wenjian' in n
                                 for n in manifest['source_sha256']))
            with self.assertRaises(ValueError):build(root)
            self.assertEqual(verify(root),manifest)
            (root/'src/go1_sdk_receive.hpp').write_text('modified')
            with self.assertRaises(ValueError):verify(root)

    def test_manifest_escape_and_scope_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'bundle';manifest=build(root)
            manifest['source_sha256']['../escaped']='0'*64
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify(root)
            manifest['policy_development']=True
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify(root)

if __name__=='__main__':unittest.main()
