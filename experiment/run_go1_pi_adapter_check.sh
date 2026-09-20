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
inputs=(
  ./CMakeLists.txt
  ./src/go1_lowlevel_experiment.cpp
  ./src/go1_kinematics.cpp
  ./src/go1_kinematics.hpp
  ./src/go1_log_file.hpp
  ./src/go1_operator_support.hpp
  ./src/go1_operator_support_probe.cpp
  ./test/go1_sdk_command_adapter_test.cpp
  ./externals/unitree_legged_sdk/include/
  ./externals/unitree_legged_sdk/lib/cpp/arm64/
)
git rev-parse HEAD > "$archive_dir/revision.txt"
sha256sum CMakeLists.txt src/go1_lowlevel_experiment.cpp \
  src/go1_kinematics.cpp src/go1_kinematics.hpp src/go1_log_file.hpp \
  src/go1_operator_support.hpp src/go1_operator_support_probe.cpp \
  test/go1_sdk_command_adapter_test.cpp \
  externals/unitree_legged_sdk/lib/cpp/arm64/libunitree_legged_sdk.a \
  > "$archive_dir/source.sha256"
printf '%s\n' "$remote_dir" > "$archive_dir/pi-directory.txt"
cp experiment/run_go1_pi_adapter_check.sh "$archive_dir/invoked-script.sh"
ssh pi@192.168.12.1 "mkdir '$remote_dir'" 2>&1 | tee "$archive_dir/staging.txt"
# Use Pi arrival times in this fresh staging directory to avoid clock skew.
rsync -avR --no-times "${inputs[@]}" "pi@192.168.12.1:$remote_dir/" \
  2>&1 | tee "$archive_dir/transfer.txt"
ssh pi@192.168.12.1 "cd '$remote_dir' &&
  cmake -S . -B build -DBUILD_TESTING=OFF -DBUILD_SDK_EXAMPLES=OFF \
    -DPYTHON_BUILD=OFF -DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE &&
  cmake --build build --target go1_sdk_command_adapter_test -j2 &&
  ./build/go1_sdk_command_adapter_test" 2>&1 | tee "$archive_dir/pi-check.txt"
printf 'pi_sdk_adapter=PASS\narchive=%s\n' "$archive_dir" | tee "$archive_dir/result.txt"
printf 'Keep this archive; no controller or Programming Module was started/stopped.\n'
