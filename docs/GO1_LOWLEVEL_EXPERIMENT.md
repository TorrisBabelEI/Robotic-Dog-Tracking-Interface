# Go1 low-level experiment: current step-by-step runbook

This is an operating checklist, not a design note. Follow the numbered steps
in order and stop whenever a stated gate fails.

## Read this first

The current hardware task is exactly one 60-second, prone
`remote-preflight` run using the receive-snapshot panic fix.

Do **not** run `ground-handover`, squat, `leg-lift`, or
`leg-lift-sequence`. Those standing modes remain blocked until a smooth,
tested lie-down-and-exit path exists.

The file `remote_preflight_fix_01.csv` does **not** exist in the repository and
was not generated during software development. It will be created on the Go1
Raspberry Pi only when Step 7 runs successfully. Step 8 copies it to Ubuntu.

### Machine roles

| Label in this document | Machine | Purpose |
| --- | --- | --- |
| **Development computer** | Computer used to edit and push Git | Source changes only |
| **Ubuntu** | `aims-Precision-7780` | Pull source, deploy, SSH, download and analyze CSV |
| **Pi** | Go1 onboard Raspberry Pi, `192.168.12.1` | Build and run the 500 Hz hardware process |

Qualisys/MOCAP is not required for this preflight.

The machine changes are fixed and happen only at these points:

1. Step 1 runs on the development computer and pushes the revision.
2. Steps 2–3 run on Ubuntu and copy only the required build inputs.
3. Steps 4–7 run in one Pi SSH session and create one hardware CSV.
4. Steps 8–9 return to Ubuntu for analysis and exact-file cleanup.
5. Step 10 shuts down the floor-supported robot and ends the experiment.

### Preflight panic behavior

- `remote-preflight` continuously sends zero-torque damping. Hearing the motors
  engage without visible motion is expected.
- Joystick motion is logged but does not command robot motion.
- `L2+B` is logged during this mode and deliberately does not cause another
  state transition.
- A preflight fault enters damping and closes automatically after a final
  0.5-second damping window.
- If `PANIC DAMPING ACTIVE` appears, do not immediately run the test again.
  Continue to Step 8, copy the CSV, analyze the first abort, and stop.
- The abort reason is latched. Hundreds of later panic rows with the same reason
  describe one panic episode, not hundreds of independent faults.

## Step 1 — Push the current revision from the development computer

Skip this step only if the current source revision is already on GitHub.

The revision must include these files:

```text
CMakeLists.txt
src/go1_lowlevel_experiment.cpp
src/go1_kinematics.cpp
src/go1_kinematics.hpp
test/go1_kinematics_test.cpp
experiment/analyze_lowlevel_log.py
docs/GO1_LOWLEVEL_EXPERIMENT.md
```

From the repository root on the **development computer**, run:

```bash
git status --short
git add .gitignore CMakeLists.txt docs/GO1_LOWLEVEL_EXPERIMENT.md
git diff --cached --check
git commit -m "Clarify onboard low-level preflight workflow"
git push origin main
```

Do not add raw experiment CSVs, generated plots, or build directories. The
repository ignores the local `logs/` directory.

## Step 2 — Update the Ubuntu checkout

Run on **Ubuntu**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git status --short
git pull --ff-only
git rev-parse --short HEAD
```

`git status --short` must be empty before pulling. If it is not empty, preserve
or resolve the Ubuntu changes; do not discard them to force the update.

## Step 3 — Copy only the hardware build inputs to the Pi

Run on **Ubuntu**, from the repository root:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
ssh pi@192.168.12.1 'mkdir -p ~/Robotic-Dog-Tracking-Interface'
rsync -avR \
  ./CMakeLists.txt \
  ./src/go1_lowlevel_experiment.cpp \
  ./src/go1_kinematics.cpp \
  ./src/go1_kinematics.hpp \
  ./test/go1_kinematics_test.cpp \
  ./externals/unitree_legged_sdk/include/ \
  ./externals/unitree_legged_sdk/lib/cpp/arm64/ \
  pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/
```

This command intentionally does **not** copy:

- `experiment/` analysis code;
- historical trajectory CSVs;
- PNG/JPG plots;
- Git history;
- Ubuntu/macOS build directories.

The analyzer stays on Ubuntu. `BUILD_SDK_EXAMPLES=OFF` allows the Pi build to
omit Unitree's example source files as well.

## Step 4 — Inspect Pi storage and old logs

Run on **Ubuntu**:

```bash
ssh pi@192.168.12.1
```

The remaining commands in Steps 4–7 run on the **Pi**. Confirm the prompt starts
with `pi@raspberrypi`.

```bash
cd ~/Robotic-Dog-Tracking-Interface
df -h /
find . -maxdepth 2 -type f -name '*.csv' \
  -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n'
```

Do not delete anything during this inspection step. If `/` is at or above 90%
use, stop here and archive old files before scheduling the hardware run.

Keep the repository, `build-arm64`, SDK headers/libraries, and current
executable on the Pi.

## Step 5 — Build and run software tests on the Pi

Run on the **Pi**:

```bash
cd ~/Robotic-Dog-Tracking-Interface
cmake -S . -B build-arm64 \
  -DPYTHON_BUILD=OFF -DBUILD_SDK_EXAMPLES=OFF
cmake --build build-arm64 --target \
  go1_lowlevel_experiment go1_kinematics_test -j2
ctest --test-dir build-arm64 --output-on-failure
mkdir -p logs
```

All tests must pass. Do not continue after a compiler or test failure.

CTest creates dry-run CSVs inside `build-arm64`. They are not hardware data.
Remove only those generated test CSVs after the tests pass:

```bash
find build-arm64 -maxdepth 1 -type f -name 'go1_dry_*.csv' \
  -print -delete
```

## Step 6 — Check route, processes, and UDP ports

Run on the **Pi**:

```bash
ip route get 192.168.123.10
pgrep -af 'go1_lowlevel_experiment|example_|run_torque_tracking'
sudo ss -lunp | grep -E ':(8090|8091)\b'
```

Required results:

- The route resembles
  `192.168.123.10 dev eth0 src 192.168.123.161`.
- `pgrep` lists no active experiment or Unitree SDK example process.
- `ss` prints nothing for ports `8090` and `8091`.

Do not continue if a controller or socket is active. Identify its owner first;
do not work around it by choosing a random port. The current executable also
checks its port before showing the arming prompt.

## Step 7 — Run one prone preflight on the Pi

### 7.1 Prepare the robot

1. Put Go1 on a flat, non-slip floor with its abdomen fully supported.
2. Fold all legs into the normal prone position; no leg may be trapped beneath
   the body.
3. Turn on the original remote.
4. Power on Go1 normally and wait for startup.
5. Use the factory remote to make Go1 lie down fully and enter prone damping
   with `L2+B`.
6. Keep everyone clear of the legs.

Do not lift the powered robot.

### 7.2 Start the program

Run on the **Pi**:

```bash
cd ~/Robotic-Dog-Tracking-Interface
if [ -e logs/remote_preflight_fix_01.csv ]; then
  echo 'STOP: this log filename already exists'
else
  echo 'OK: this log filename is unused'
  ./build-arm64/go1_lowlevel_experiment --mode remote-preflight \
    --prone-confirmed --duration-s 60 \
    --log logs/remote_preflight_fix_01.csv
fi
```

If the check prints `STOP`, the program will not start. First preserve the
existing file, or use a new unique filename consistently in Steps 7–9.

At the prompt type exactly:

```text
ARM DAMPING
```

During the 60 seconds:

1. move each joystick;
2. press `L2+B` at least once;
3. confirm joystick values change and `L2+B=1` appears;
4. do not expect the robot to follow joystick motion.

On success, the program exits by itself and prints the log path. Only now should
this file exist on the Pi:

```text
~/Robotic-Dog-Tracking-Interface/logs/remote_preflight_fix_01.csv
```

If panic occurs, wait for the prone preflight to close automatically. Do not
start another run.

## Step 8 — Copy and analyze the CSV on Ubuntu

Return to **Ubuntu**:

```bash
exit
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
mkdir -p logs/downloaded
ssh pi@192.168.12.1 \
  'sha256sum ~/Robotic-Dog-Tracking-Interface/logs/remote_preflight_fix_01.csv'
scp pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/logs/remote_preflight_fix_01.csv \
  logs/downloaded/
sha256sum logs/downloaded/remote_preflight_fix_01.csv
python3 experiment/analyze_lowlevel_log.py \
  logs/downloaded/remote_preflight_fix_01.csv --no-plots
```

The remote and local SHA-256 values must match.

The run passes only if all of these conditions hold:

```text
feedback rate             >= 450 Hz
p99 feedback gap          <= 10 ms
maximum feedback gap      <= 20 ms
remote_valid_fresh_ratio  close to 1.0
L2+B_seen                 1
lowlevel_fresh_ratio      close to 1.0
duplicate_fresh           0
gap_over_20ms             0
watchdog_cycles           0
abort reasons             absent
```

The analyzer now prints the first abort sample separately. In the earlier
failed log, only the first duplicate `tick=332367` triggered
`remote_preflight_tick_invalid`; subsequent panic rows retained that one
latched reason. The fixed receive path publishes the state packet and sequence
under the same lock.

If any gate fails, save the Ubuntu CSV and terminal output, stop the experiment,
and diagnose that one run offline. Do not keep repeating preflight.

## Step 9 — Clean the verified hardware CSV from the Pi

Perform this only after Step 8 succeeds and the Ubuntu copy opens correctly.
Run on **Ubuntu**:

```bash
ssh pi@192.168.12.1 \
  'rm -- ~/Robotic-Dog-Tracking-Interface/logs/remote_preflight_fix_01.csv'
ssh pi@192.168.12.1 \
  'find ~/Robotic-Dog-Tracking-Interface/logs -maxdepth 1 \
   -type f -name "*.csv" -printf "%10s %p\n"'
```

The first command deletes only the exact verified Pi copy. The archived Ubuntu
copy remains in `logs/downloaded/`.

For an older CSV stored in the Pi repository root, use the same sequence:
copy it, compare checksums, analyze it on Ubuntu, and delete only its exact
filename. Never use `rm *.csv`, and never delete `build-arm64` or the SDK.

## Step 10 — Shut down and stop

Go1 should still be fully prone. Shut it down using the normal battery shutdown
procedure while it remains floor-supported.

After one successful fixed preflight, mark the preflight gate complete. Do not
repeat it again unless code affecting UDP reception, state freshness, remote
decoding, damping, watchdog, or panic handling changes.

There is no standing hardware step in the current runbook. The next task is
software-only implementation and dry-run review of a smooth lie-down-and-exit
path. This document must be extended with a new numbered procedure before any
standing experiment is authorized.

## Reference: why control runs onboard

The onboard run measured approximately `469.63 Hz`, with a `4.096 ms` p99 gap
and an `8.542 ms` maximum gap. A previous Ubuntu-direct Wi-Fi run measured only
`339.42 Hz` and a `142.002 ms` maximum gap. Therefore the Pi owns the 500 Hz
motor loop, watchdog, and safety state machine. Ubuntu owns analysis and future
MPPI/Qualisys integration.

For future work, the intended split is:

```text
Qualisys -> Ubuntu MPPI -> lower-rate references -> Pi 500 Hz safety loop -> Go1
```

MOCAP is optional ground truth and must not be required by the fast motor loop.
