#!/usr/bin/env python3
"""Legacy guard for the removed Python low-level sender.

The former implementation replayed an unvalidated pickle at an effective
reference rate of about 5 Hz while placing all twelve joints in pure torque
mode. Keeping a hard failure at the old entry point prevents accidental use
on hardware; Git history retains the original source.
"""

import sys


MESSAGE = """\
This legacy Python torque sender is intentionally disabled.

Use the C++ safety-state-machine instead:
  ./build/go1_lowlevel_experiment --mode remote-preflight \\
      --duration-s 30 --log remote_preflight.csv

For an offline test on any platform:
  ./build/go1_lowlevel_experiment --dry-run --mode leg-lift \\
      --leg auto --lift-height-m 0.02 --tau-overlay-nm 0.10 \\
      --tau-overlay-hz 0.5 --log /tmp/go1_leg_lift_dry.csv

Analyze a resulting log with:
  python3 experiment/analyze_lowlevel_log.py /tmp/go1_leg_lift_dry.csv

Follow docs/GO1_LOWLEVEL_EXPERIMENT.md for the staged ground procedure. The
single-joint pure-torque mode still requires a load-bearing support stand.
"""


if __name__ == "__main__":
    print(MESSAGE, file=sys.stderr)
    raise SystemExit(2)
