#!/usr/bin/env bash
# Operator launches on Ubuntu; snapshots/transfers/builds only, never starts motion.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p logs/engagement-deployment
archive_dir="$(mktemp -d "$PWD/logs/engagement-deployment/review-XXXXXXXX")"
printf 'deployment_archive=%s\n' "$archive_dir"
trap 'printf "STOP: inspect %s; no trial was launched.\n" "$archive_dir" >&2' ERR
cp experiment/prepare_go1_prone_engagement.sh experiment/prepare_go1_bundle.py "$archive_dir/"
python3 -B experiment/prepare_go1_bundle.py --out "$archive_dir/source"
# Copy the verified snapshot, not mutable files from a mixed development tree.
ssh pi@192.168.12.1 'mkdir -p /home/pi/go1-prone-engagement' \
  2>&1 | tee "$archive_dir/staging.txt"
rsync -av --no-times "$archive_dir/source/" pi@192.168.12.1:/home/pi/go1-prone-engagement/ \
  2>&1 | tee "$archive_dir/transfer.txt"
ssh pi@192.168.12.1 'cd /home/pi/go1-prone-engagement &&
  sha256sum -c source.sha256 &&
  cmake -S . -B build -DBUILD_TESTING=ON -DBUILD_SDK_EXAMPLES=OFF \
    -DPYTHON_BUILD=OFF -DGO1_ENABLE_POLICY_DEVELOPMENT=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE &&
  cmake --build build --target go1_sdk_transport_test go1_sdk_command_adapter_test go1_lowlevel_experiment -j2 &&
  ./build/go1_sdk_transport_test &&
  ./build/go1_sdk_command_adapter_test &&
  mkdir -p logs &&
  sha256sum build/go1_lowlevel_experiment > build/go1_lowlevel_experiment.sha256 &&
  cat build/go1_lowlevel_experiment.sha256' 2>&1 | tee "$archive_dir/build.txt"
printf 'engagement_deployment=PASS; no controller started\n' | tee "$archive_dir/result.txt"
