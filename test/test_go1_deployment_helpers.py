"""Exercise local staging/SSH orchestration with fake network commands only."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

class DeploymentHelpers(unittest.TestCase):
    def test_snapshot_transfer_and_explicit_selected_build(self):
        for helper in ('prepare_go1_prone_engagement.sh','run_go1_pi_adapter_check.sh'):
            with self.subTest(helper=helper), tempfile.TemporaryDirectory() as folder:
                root=Path(folder);(root/'experiment').mkdir();bins=root/'bin';bins.mkdir()
                (root/'experiment'/helper).write_text((ROOT/'experiment'/helper).read_text())
                # The snapshot utility is separately tested with real repository inputs.
                (root/'experiment/prepare_go1_bundle.py').write_text(
                    'import sys\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index("--out")+1]);p.mkdir(parents=True)\n(p/"source.sha256").write_text("fixture")\n')
                for name in ('ssh','rsync'):
                    path=bins/name
                    path.write_text('#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["CALL_LOG"],"a") as f:f.write(json.dumps([sys.argv[0]]+sys.argv[1:])+"\\n")\n')
                    path.chmod(0o755)
                log=root/'calls.jsonl'
                result=subprocess.run(['bash',str(root/'experiment'/helper)],
                    env={**os.environ,'PATH':str(bins)+':'+os.environ['PATH'],'CALL_LOG':str(log)},
                    capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,0,result.stderr)
                calls=log.read_text()
                self.assertIn('/source/',calls)
                self.assertIn('sha256sum -c source.sha256',calls)
                self.assertIn('-DGO1_ENABLE_POLICY_DEVELOPMENT=OFF',calls)
                self.assertIn('./build/go1_sdk_transport_test',calls)
                self.assertNotIn('./build/go1_lowlevel_experiment --mode',calls)
                self.assertNotIn('kill -',calls)
                self.assertIn('PASS',result.stdout)

if __name__=='__main__':unittest.main()
