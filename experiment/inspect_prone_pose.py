#!/usr/bin/env python3
"""Offline observation of an archived prone preflight; never produces commands."""

import argparse
import csv
import hashlib
import math
import statistics
from pathlib import Path


JOINTS = [f"{leg}_{axis}" for leg in ("FR", "FL", "RR", "RL") for axis in range(3)]
# Same command bounds as src/go1_lowlevel_experiment.cpp. Measurements may
# legitimately exceed them; do not clamp measurements into command targets.
BOUNDS = [(-1.047, 1.047), (-0.663, 2.966), (-2.721, -0.837)]


def inspect(path):
    window = []
    required = ["host_monotonic_ns", "state_tick_ms", "phase", "recv_ok",
                "level_flag", "watchdog_active", "abort_reason", "imu_roll", "imu_pitch"]
    required += [j + suffix for j in JOINTS for suffix in ("_state_q", "_state_dq")]
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            try:
                values = {k: float(row[k]) for k in required if k not in ("phase", "abort_reason")}
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError("nonfinite sample")
                valid = (row["phase"] == "REMOTE_PREFLIGHT" and values["recv_ok"] == 1
                         and values["level_flag"] == 255 and not row["abort_reason"]
                         and values["watchdog_active"] == 0
                         and all(abs(values[j + "_state_dq"]) <= 0.05 for j in JOINTS))
            except (ValueError, TypeError):
                valid = False
            if not valid:
                window = []
                continue
            if window:
                gap = (values["host_monotonic_ns"] - window[-1]["host_monotonic_ns"]) / 1e9
                tick_gap = (values["state_tick_ms"] - window[-1]["state_tick_ms"]) % (2**32)
                if not (0 < gap <= 0.02 and 0 < tick_gap <= 20):
                    window = []
            window.append(values)
            while len(window) > 1 and (values["host_monotonic_ns"] - window[1]["host_monotonic_ns"]) / 1e9 >= 2:
                window.pop(0)
            duration = (values["host_monotonic_ns"] - window[0]["host_monotonic_ns"]) / 1e9
            if duration < 2 or len(window) < 900:
                continue
            columns = [j + "_state_q" for j in JOINTS] + ["imu_roll", "imu_pitch"]
            if any(max(r[k] for r in window) - min(r[k] for r in window) > 0.03 for k in columns):
                continue
            return window
    raise ValueError("no continuous 2-second quiet preflight window; retain this result for review, do not rerun hardware automatically")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    try:
        window = inspect(args.log)
        digest = hashlib.sha256()
        with args.log.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        print(f"source: {args.log.resolve()}\nsha256: {digest.hexdigest()}")
        print(f"[PASS] quiet observation window: {len(window)} fresh samples; "
              f"host_ns={int(window[0]['host_monotonic_ns'])}..{int(window[-1]['host_monotonic_ns'])}")
        print("joint  median_rad  min_rad  max_rad  command_bound_check")
        for index, joint in enumerate(JOINTS):
            q = [r[joint + "_state_q"] for r in window]
            low, high = BOUNDS[index % 3]
            status = "WITHIN" if min(q) >= low and max(q) <= high else "OUTSIDE_DO_NOT_REPLAY"
            print(f"{joint:4s} {statistics.median(q): .6f} {min(q): .6f} {max(q): .6f} {status}")
        for key in ("imu_roll", "imu_pitch"):
            print(f"{key}_median_rad: {statistics.median(r[key] for r in window):.6f}")
        print("OBSERVATION ONLY: neither belly contact nor hardware lie-down target is validated.")
        return 0
    except (OSError, ValueError, csv.Error) as error:
        print(f"[STOP] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
