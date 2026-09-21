#!/usr/bin/env python3
"""Verify one Pi log against its Ubuntu archive before optional Pi deletion.

Only a single explicit regular file under the Pi project logs directory can be
removed. The Ubuntu archive copy is never deleted. Without --execute this is
read-only and prints the exact deletion that would be performed.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shlex
import subprocess
from pathlib import Path


PI_LOG_ROOTS = ("/home/pi/Robotic-Dog-Tracking-Interface/logs/",
                "/home/pi/go1-prone-engagement/logs/")
PI_PATH_PATTERN = re.compile(r"[A-Za-z0-9_./-]+\Z")
HASH_PATTERN = re.compile(r"([0-9a-f]{64})  (.+)\Z")


def validate_pi_path(path: str) -> str:
    root = next((r for r in PI_LOG_ROOTS if path.startswith(r)), None)
    if (
        root is None
        or not PI_PATH_PATTERN.fullmatch(path)
        or ".." in Path(path).parts
        or path.endswith("/")
        or len(Path(path).parts) <= len(Path(root).parts)
    ):
        raise ValueError("Pi path must name one file beneath the exact project logs directory")
    return path


def validate_ubuntu_copy(path: Path) -> Path:
    archive_root = (Path(__file__).resolve().parents[1] / "logs").resolve()
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(archive_root) or not path.is_file() or path.is_symlink():
        raise ValueError("Ubuntu copy must be a regular, non-symlink file under repo logs")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ssh(command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5",
         "pi@192.168.12.1", command],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def remote_sha256(pi_path: str) -> str:
    quoted = shlex.quote(pi_path)
    output = ssh(
        f"test -f {quoted} && test ! -L {quoted} && "
        f"test \"$(realpath -e -- {quoted})\" = {quoted} && "
        f"sha256sum -- {quoted}"
    )
    match = HASH_PATTERN.fullmatch(output)
    if match is None or match.group(2) != pi_path:
        raise ValueError("unexpected Pi checksum output; nothing deleted")
    return match.group(1)


def remove_if_still_matching(pi_path: str, expected: str) -> None:
    quoted = shlex.quote(pi_path)
    # Recheck on the Pi immediately before deletion. The fixed path grammar
    # prevents shell metacharacters; no wildcard or directory is accepted.
    command = (
        f"test -f {quoted} && test ! -L {quoted} && "
        f"test \"$(realpath -e -- {quoted})\" = {quoted} && "
        f"test \"$(sha256sum -- {quoted} | cut -d ' ' -f 1)\" = {expected} && "
        f"rm -- {quoted}"
    )
    ssh(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-file", required=True,
                        help="Exact Pi file under its project logs directory")
    parser.add_argument("--ubuntu-copy", required=True, type=Path,
                        help="Archived copy under this Ubuntu repository's logs directory")
    parser.add_argument("--execute", action="store_true",
                        help="Delete the exact Pi file after both hashes match")
    args = parser.parse_args()
    try:
        pi_path = validate_pi_path(args.pi_file)
        ubuntu_copy = validate_ubuntu_copy(args.ubuntu_copy)
        ubuntu_hash = sha256_file(ubuntu_copy)
        pi_hash = remote_sha256(pi_path)
        if ubuntu_hash != pi_hash:
            raise ValueError("SHA-256 mismatch; nothing deleted")
        print(f"verified_sha256={ubuntu_hash}")
        print(f"Ubuntu archive retained: {ubuntu_copy}")
        if args.execute:
            remove_if_still_matching(pi_path, ubuntu_hash)
            print(f"Deleted verified Pi file: {pi_path}")
        else:
            print(f"PREVIEW ONLY: would delete Pi file: {pi_path}")
            print("Pass --execute only after reviewing this exact pathname.")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"STOP: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
