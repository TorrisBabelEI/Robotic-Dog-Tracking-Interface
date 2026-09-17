#!/usr/bin/env bash
# Robot-off software cluster for the Ubuntu/Pi operator-support channel.
# This script never opens a Go1 UDP socket or starts the hardware controller.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${GO1_SUPPORT_BUILD:-/tmp/go1-operator-support-build}"
cd "$repo_dir"

python3 -m unittest discover -s test -p test_operator_support_gate.py -v
python3 -m unittest discover -s test -p test_cleanup_verified_go1_file.py -v
python3 -c 'import tkinter; print("tkinter=available")'
cmake -S . -B "$build_dir" -DBUILD_TESTING=ON -DBUILD_SDK_EXAMPLES=OFF
cmake --build "$build_dir" \
  --target go1_operator_support_test go1_operator_support_server_test \
           go1_operator_support_probe go1_prone_low_rise_test \
           go1_lowlevel_experiment -j2
ctest --test-dir "$build_dir" \
  -R '^(go1_operator_support_|go1_prone_low_rise_)' --output-on-failure
printf 'support_cluster=PASS\n'
