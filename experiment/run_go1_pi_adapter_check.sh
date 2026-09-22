#!/usr/bin/env bash
# Operator-run ARM SDK check. No UDP/controller is instantiated by the test.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p logs/pi-sdk-adapter
archive_dir="$(mktemp -d "$PWD/logs/pi-sdk-adapter/review-XXXXXXXX")"
remote_dir="/home/pi/go1-sdk-adapter-${archive_dir##*/}"
printf 'archive=%s\nPi staging directory=%s\n' "$archive_dir" "$remote_dir"
trap 'printf "STOP: retain archive %s and inspect the last transcript.\n" "$archive_dir" >&2' ERR
python3 -B experiment/prepare_go1_bundle.py --out "$archive_dir/source"
printf '%s\n' "$remote_dir" > "$archive_dir/pi-directory.txt"
cp experiment/run_go1_pi_adapter_check.sh "$archive_dir/invoked-script.sh"
ssh pi@192.168.12.1 "mkdir '$remote_dir'" 2>&1 | tee "$archive_dir/staging.txt"
# Use Pi arrival times in this fresh staging directory to avoid clock skew.
rsync -av --no-times "$archive_dir/source/" "pi@192.168.12.1:$remote_dir/" \
  2>&1 | tee "$archive_dir/transfer.txt"
ssh pi@192.168.12.1 "cd '$remote_dir' &&
  sha256sum -c source.sha256 &&
  cmake -S . -B build -DBUILD_TESTING=ON -DBUILD_SDK_EXAMPLES=OFF \
    -DPYTHON_BUILD=OFF -DGO1_ENABLE_POLICY_DEVELOPMENT=OFF -DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE &&
  cmake --build build --target go1_sdk_command_adapter_test go1_sdk_transport_test -j2 &&
  ./build/go1_sdk_transport_test &&
  ./build/go1_sdk_command_adapter_test" 2>&1 | tee "$archive_dir/pi-check.txt"
printf 'pi_sdk_adapter=PASS\narchive=%s\n' "$archive_dir" | tee "$archive_dir/result.txt"
printf 'Keep this archive; no controller or Programming Module was started/stopped.\n'
