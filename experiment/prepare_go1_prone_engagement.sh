#!/usr/bin/env bash
# Operator launches on Ubuntu; only transfers/builds on Pi, never starts motion.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p logs/engagement-deployment
archive_dir="$(mktemp -d "$PWD/logs/engagement-deployment/review-XXXXXXXX")"
printf 'deployment_archive=%s\n' "$archive_dir"
trap 'printf "STOP: inspect %s; no trial was launched.\n" "$archive_dir" >&2' ERR
cp experiment/prepare_go1_prone_engagement.sh "$archive_dir/invoked-script.sh"
git rev-parse HEAD > "$archive_dir/revision.txt"
sha256sum CMakeLists.txt src/go1_lowlevel_experiment.cpp src/go1_kinematics.cpp \
  src/go1_kinematics.hpp src/go1_log_file.hpp src/go1_operator_support.hpp \
  src/go1_operator_support_probe.cpp test/go1_sdk_command_adapter_test.cpp \
  externals/unitree_legged_sdk/lib/cpp/arm64/libunitree_legged_sdk.a \
  > "$archive_dir/source.sha256"
ssh pi@192.168.12.1 'mkdir -p /home/pi/go1-prone-engagement' \
  2>&1 | tee "$archive_dir/staging.txt"
rsync -avR --no-times \
  ./CMakeLists.txt ./src/go1_lowlevel_experiment.cpp \
  ./src/go1_kinematics.cpp ./src/go1_kinematics.hpp ./src/go1_log_file.hpp \
  ./src/go1_operator_support.hpp ./src/go1_operator_support_probe.cpp \
  ./test/go1_sdk_command_adapter_test.cpp \
  ./externals/unitree_legged_sdk/include/ \
  ./externals/unitree_legged_sdk/lib/cpp/arm64/ \
  pi@192.168.12.1:/home/pi/go1-prone-engagement/ \
  2>&1 | tee "$archive_dir/transfer.txt"
ssh pi@192.168.12.1 'cd /home/pi/go1-prone-engagement &&
  cmake -S . -B build -DBUILD_TESTING=OFF -DBUILD_SDK_EXAMPLES=OFF \
    -DPYTHON_BUILD=OFF -DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE &&
  cmake --build build --target go1_lowlevel_experiment -j2 &&
  mkdir -p logs &&
  sha256sum build/go1_lowlevel_experiment' 2>&1 | tee "$archive_dir/build.txt"
printf 'engagement_deployment=PASS; no controller started\n' | tee "$archive_dir/result.txt"
