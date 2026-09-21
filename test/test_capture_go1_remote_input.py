"""Exercise the actual remote Bash block without SSH, sudo or robot access."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'experiment/capture_go1_remote_input.sh'


class CaptureScriptTests(unittest.TestCase):
    def run_block(self, fail=False):
        source = SCRIPT.read_text().split("script=$(cat <<'PI'\n", 1)[1].split('\nPI\n)', 1)[0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bins = root/'bin'
            bins.mkdir()
            fixtures = {
                'sudo': '[ "$1" = -v ] && exit 0\nexec "$@"',
                'ss': 'exit 0',
                'timeout': 'shift 3\nexec "$@"',
                'sleep': '/bin/sleep 0.05',
                'tcpdump': ('echo "capture failed" >&2; exit 7' if fail else
                            'echo "tcpdump: listening" >&2\nprintf fixture > remote_input.pcap\n/bin/sleep 0.3'),
            }
            for name, body in fixtures.items():
                path = bins/name
                path.write_text('#!/bin/bash\n'+body+'\n')
                path.chmod(0o755)
            # Reproduce EPERM semantics for kill -0 on a sudo-owned child.
            source = 'kill() { return 1; }\n'+source
            result = subprocess.run(['bash', '-c', source, '--', str(root)],
                                    env={**os.environ, 'PATH': str(bins)+':'+os.environ['PATH']},
                                    capture_output=True, text=True, timeout=5)
            files = {p.name: p.read_text() for p in root.iterdir() if p.is_file()}
            return result, files

    def test_running_capture_with_no_signal_permission_reaches_prompts(self):
        result, files = self.run_block()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('NOW:', result.stdout)
        self.assertIn('RELEASE', result.stdout)
        self.assertIn('press_prompt=', files['events.txt'])
        self.assertIn('remote_input.pcap', files['capture.sha256'])

    def test_early_capture_failure_stops_before_prompts(self):
        result, files = self.run_block(fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('NOW:', result.stdout)
        self.assertIn('status=7', result.stderr)
        self.assertNotIn('events.txt', files)


if __name__ == '__main__':
    unittest.main()
