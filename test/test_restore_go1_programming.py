"""Local tests only; no SSH, robot commands, or vendor files are touched."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

SOURCE = (Path(__file__).resolve().parents[1]/'experiment/restore_go1_programming.sh').read_text()
VERIFY = SOURCE.split("<<'VERIFY'\n", 1)[1].split('\nVERIFY\n)', 1)[0]
START = SOURCE.split("<<'PI'\n", 1)[1].split('\nPI\n)', 1)[0]


class RestorationTests(unittest.TestCase):
    def verify(self, pids='321', expected='321', socket_pid='321', peer='192.168.123.161:8082'):
        script = '''set -euo pipefail
pgrep() { if [[ "$1" == -af ]]; then echo '321 python3 programming.py'; else printf '%s\\n' "$FIXTURE_PIDS"; fi; }
ss() { printf 'ESTAB 0 0 192.168.123.161:8090 %s users:(("python3",pid=%s,fd=3))\\n' "$FIXTURE_PEER" "$FIXTURE_SOCKET_PID"; }
'''+VERIFY
        return subprocess.run(['bash', '-c', script, '--', expected], capture_output=True, text=True,
                              env={**os.environ, 'FIXTURE_PIDS': pids, 'FIXTURE_PEER': peer,
                                   'FIXTURE_SOCKET_PID': socket_pid}, timeout=5)

    def test_same_pid_and_socket_accepted(self):
        self.assertEqual(self.verify().returncode, 0)

    def test_pid_disappearance_or_change_rejected(self):
        self.assertNotEqual(self.verify(pids='').returncode, 0)
        self.assertNotEqual(self.verify(pids='322').returncode, 0)

    def test_duplicate_or_wrong_socket_rejected(self):
        self.assertNotEqual(self.verify(pids='321\n322').returncode, 0)
        self.assertNotEqual(self.verify(socket_pid='999').returncode, 0)
        self.assertNotEqual(self.verify(peer='192.168.123.10:8007').returncode, 0)

    def test_background_child_survives_shell_exit_and_hangup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); vendor = root/'vendor'; vendor.mkdir()
            (vendor/'programming.sh').write_text('python3 programming.py &\n')
            (vendor/'programming.py').write_text(
                "import os,time\nfrom pathlib import Path\nPath('child.pid').write_text(str(os.getpid()))\ntime.sleep(30)\n")
            script = START.replace('/home/pi/Unitree/autostart/programming', str(vendor)).replace(
                '/home/pi/go1-prone-engagement', str(root/'go1'))
            script = 'id() { echo pi; }; pgrep() { return 1; };\n'+script
            pid = None
            try:
                result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=6)
                self.assertEqual(result.returncode, 0, result.stderr)
                pid = int((vendor/'child.pid').read_text())
                os.kill(pid, signal.SIGHUP)
                time.sleep(.05)
                state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
                self.assertNotEqual(state, 'Z', 'child died when SSH-style hangup arrived')
            finally:
                if pid is None and (vendor/'child.pid').exists():
                    pid = int((vendor/'child.pid').read_text())
                if pid is not None:
                    try: os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError: pass


if __name__ == '__main__':
    unittest.main()
