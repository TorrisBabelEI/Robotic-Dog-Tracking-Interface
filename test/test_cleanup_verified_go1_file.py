import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest import mock

from experiment import cleanup_verified_go1_file as cleanup


PI_FILE = "/home/pi/Robotic-Dog-Tracking-Interface/logs/dry-run/go1.csv"
UBUNTU_COPY = "/home/aims/Yuxuan/Robotic-Dog-Tracking-Interface/logs/archive/go1.csv"
TEST_HASH = "a" * 64


class CleanupPathTests(unittest.TestCase):
    def test_exact_file_beneath_pi_logs_is_allowed(self):
        path = "/home/pi/Robotic-Dog-Tracking-Interface/logs/dry-run/go1.csv"
        self.assertEqual(cleanup.validate_pi_path(path), path)

    def test_broad_or_escaped_targets_are_rejected(self):
        for path in (
            "/home/pi/Robotic-Dog-Tracking-Interface/logs/",
            "/home/pi/Robotic-Dog-Tracking-Interface/logs/../src/file.csv",
            "/home/pi/Robotic-Dog-Tracking-Interface/logs/a;rm -rf b",
            "/home/pi/Robotic-Dog-Tracking-Interface/logs/a/*.csv",
            "/home/pi/Robotic-Dog-Tracking-Interface/src/file.csv",
            str(Path.home()),
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                cleanup.validate_pi_path(path)

    def test_preview_never_deletes_and_execute_rechecks(self):
        for execute in (False, True):
            argv = ["cleanup", "--pi-file", PI_FILE,
                    "--ubuntu-copy", UBUNTU_COPY]
            if execute:
                argv.append("--execute")
            with mock.patch("sys.argv", argv), \
                 mock.patch.object(cleanup, "validate_ubuntu_copy",
                                   return_value=Path(UBUNTU_COPY)), \
                 mock.patch.object(cleanup, "sha256_file", return_value=TEST_HASH), \
                 mock.patch.object(cleanup, "remote_sha256", return_value=TEST_HASH), \
                 mock.patch.object(cleanup, "remove_if_still_matching") as remove:
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(cleanup.main(), 0)
                self.assertEqual(remove.call_count, int(execute))
                if execute:
                    remove.assert_called_once_with(PI_FILE, TEST_HASH)
                    self.assertIn("Deleted verified Pi file", output.getvalue())
                else:
                    self.assertIn("PREVIEW ONLY", output.getvalue())

    def test_mismatch_never_deletes(self):
        argv = ["cleanup", "--pi-file", PI_FILE,
                "--ubuntu-copy", UBUNTU_COPY, "--execute"]
        with mock.patch("sys.argv", argv), \
             mock.patch.object(cleanup, "validate_ubuntu_copy",
                               return_value=Path(UBUNTU_COPY)), \
             mock.patch.object(cleanup, "sha256_file", return_value=TEST_HASH), \
             mock.patch.object(cleanup, "remote_sha256", return_value="b" * 64), \
             mock.patch.object(cleanup, "remove_if_still_matching") as remove:
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    cleanup.main()
            self.assertEqual(result.exception.code, 2)
            remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
