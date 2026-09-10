# Go1 low-level experiments: staged operating manual

Each experiment has its own chapter. Complete the previous chapter's hardware
acceptance before starting the next chapter's hardware test. Simulation results
do not count as hardware acceptance.

| Chapter | Experiment | Current status |
| --- | --- | --- |
| [1](#chapter-1--remote-preflight) | Communication and remote preflight | Reported passing run: `remote_preflight_fix_02.csv` |
| [2](#chapter-2--ground-handover) | Standing takeover and 10-second hold | Original dry-run completed and archived twice; hardware locked |
| [3](#chapter-3--squat-and-return) | Four-leg half-squat and return | Original dry-run completed and archived; hardware pending Chapter 2 |
| [4](#chapter-4--single-leg-lift) | Weight transfer and one leg lift | Original dry-run completed and archived; hardware pending Chapter 3 |
| [5](#chapter-5--four-leg-sequence) | Four sequential leg lifts | Original dry-run completed and archived; hardware pending Chapter 4 |

**Current next step: endpoint code development and validation in 2.1.1.**
The operator has confirmed that all four original dry-run CSV files and their
summaries are fully archived on Ubuntu. `handover-SB8m9xb7` and
`handover-Z6dkjhnB` are two independent archives; retain both. They are not
evidence of an overwritten download. No routine dry-run or preflight repetition
is requested. The old rehearsal commands remain below as reference only.

The new normal-exit state machine is a development fixture with synthetic
support confirmation. Its target has not been calibrated to actual belly
contact. Ground hardware modes now fail before ARM or UDP initialization;
there is no command-line override. The hardware references in Chapters 2–5
remain blocked pending endpoint and standing-takeover review.

## Common setup and operating rules

The reported `remote_preflight_fix_02.csv` run passed the preflight metrics:
497.90 Hz feedback, 2.193 ms p99 gap, 10.762 ms maximum gap, valid remote and
low-level ratios of 1.000, L2+B observed, zero duplicate fresh ticks, zero
gaps over 20 ms, zero watchdog cycles, and no reported abort. Preserve that
CSV on Ubuntu and compare its SHA-256 with the Pi original before deleting
the Pi copy. This completes the communication/remote preflight gate; it does
not validate standing actions. No new hardware run is needed solely to test
the log-overwrite prompt.

The numbered procedure below is retained for a future required 60-second,
prone `remote-preflight` run. It uses the C++ example's local UDP port:

```text
Go1 onboard Pi local UDP 8090 -> 192.168.123.10:8007
```

Do not run `ground-handover`, `squat`, `leg-lift`, or
`leg-lift-sequence` yet. Those standing modes remain blocked until the program
has a smooth, tested lie-down-and-exit path.

The proposed file `remote_preflight_fix_01.csv` is not stored in Git. It will
be created on the Pi only when Step 10 runs. Step 12 copies and analyzes it on
Ubuntu.

### Machine roles

| Name used below | Machine | Responsibility |
| --- | --- | --- |
| **Development computer** | Computer used to edit this repository | Review, commit, and push source changes |
| **Ubuntu** | `aims-Precision-7780` | Pull, deploy, SSH, download, and analyze logs |
| **Pi** | Go1 onboard Raspberry Pi at `192.168.12.1` | Build and run the 500 Hz hardware process |

The analyzer runs on Ubuntu, not on the Pi. Qualisys/MOCAP is not needed for
this preflight.

### Confirmed UDP 8090 conflict

This Go1 starts Unitree's optional Programming Module in the Pi desktop
session:

```text
python3 /home/pi/Unitree/autostart/programming/programming.py
192.168.123.161:8090 -> 192.168.123.161:8082
```

The module supplies Unitree's GUI/Blockly/MQTT programming interface. It is not
a core leg-control process, but importing its high-level robot interface claims
local UDP 8090 even while no Blockly program is running.

An earlier preflight from local 8092 returned only about `0.54 Hz`; that
observation alone does not prove that the firmware requires source port 8090.
The reported passing run used the current 8090 procedure. We therefore retain
that configuration for reproducibility. The procedure
temporarily stops only `programming.py`, runs one prone preflight on 8090, and
then immediately restores the module with its own Unitree wrapper:

```text
/home/pi/Unitree/autostart/programming/programming.sh
```

Never reuse an old PID. The observed PID has already changed across boots.

### Processes that must not be stopped

Do not stop, kill, or restart any of these processes:

- `startup_manager.py`;
- `Legged_sport`;
- `appTransit`;
- `hostapd`;
- ROS obstacle or ultrasonic processes.

Restarting the whole `startup_manager.py` may duplicate other Unitree modules.
Only its `programming.py` child is in scope.

### What remote-preflight does

- It actively sends `q=PosStop`, `Kp=0`, `Kd=1`, and zero feed-forward torque
  to all 12 joints.
- A motor-engagement sound without visible motion is expected while the robot
  is already fully prone.
- Joystick motion is decoded and logged but never commands robot motion.
- `L2+B` is decoded and logged. In `remote-preflight` it deliberately does not
  request a second transition because the program is already sending damping.
- A fault sends a final 0.5-second damping window, writes the CSV, and closes
  automatically.
- If `PANIC DAMPING ACTIVE` appears, do not immediately repeat the test. Keep
  the robot clear, let the process close, preserve that CSV, and continue to
  the recovery and analysis steps.

### Shared rules for Chapters 2–5

- Build and deploy using Chapter 1, Steps 1, 3–6. Do not rerun the completed
  preflight solely because you are preparing another experiment. Update Ubuntu
  from GitHub while Internet access is available, before connecting to Go1 Wi-Fi
  if the Wi-Fi does not provide Internet access.
- The software rehearsal commands run on the **Pi** after SSH, and all include
  `--dry-run`; they do not open robot UDP sockets. Create `logs/dry-run` first.
- Download every dry-run or hardware CSV to **Ubuntu**. Run the Python analyzer
  only there. Keep simulated and hardware files in separate directories.
- For hardware, run only one experiment at a time. Archive and evaluate its
  result before the next repetition. Reusing a Pi filename is allowed after
  archiving; the hardware executable asks before replacing an existing file.
- A successful analyzer process exit is not a pass certificate. Review the
  metrics, abort reasons, and actual phase sequence against the chapter's gates.
- Restore the Programming Module only after the experiment has closed and the
  robot is floor-supported. Use Chapter 1, Step 11; do not restore it while the
  experiment still owns UDP 8090.
- `SAFE_HOLD` means continued standing control. It does not mean prone, motor
  power off, process exit, or control returned to the factory controller.
- In hardware ground modes, single Ctrl-C requests return to the captured
  pose and continued hold. `L2+B` or two Ctrl-C presses within one second
  requests damping. A further Ctrl-C in panic requests exit after the damping
  window. These are fault responses, not a normal standing shutdown procedure.

### Download and analyze each Chapter 2–5 rehearsal

Each chapter gives a **Pi command** and a corresponding **Ubuntu block**. Open
a second Ubuntu terminal for download/analysis, keeping the Pi terminal clear.
The download block creates a unique local directory; it never overwrites an
earlier archive. It does not remove the Pi file. After checking the remote and
local hashes and opening the archive, you may remove that exact Pi file using
the same copy/verify/delete rule as Chapter 1, Step 13.

For eventual hardware runs, the same workflow applies with `logs/<mode>.csv`
as the remote path and `logs/downloaded` as the Ubuntu archive parent. Do not
label a `logs/dry-run` file as a hardware result.

## Chapter 1 — Remote preflight

Status: passed based on the supplied `remote_preflight_fix_02.csv` summary.
The full original procedure is retained below for reproducibility. Proceed to
Chapter 2 for the next experiment; do not restart this chapter by default.

### Step 1 — Push the revision from the development computer

Skip this step only if the revision containing this document and the current
experiment source is already on GitHub.

Run from the repository root on the **development computer**:

```bash
git status --short
git add CMakeLists.txt \
  src/go1_lowlevel_experiment.cpp \
  src/go1_kinematics.cpp \
  src/go1_kinematics.hpp \
  src/go1_log_file.hpp \
  test/go1_log_file_test.cpp \
  test/go1_kinematics_test.cpp \
  docs/GO1_LOWLEVEL_EXPERIMENT.md
git diff --cached --check
git diff --cached --stat
git commit -m "Document standard-port onboard preflight"
git push origin main
```

Review the staged files before committing. Do not add raw CSVs, plots, or build
directories.

### Step 2 — Power Go1, put it prone, and connect Ubuntu

Complete these physical and network steps in order:

1. Install a charged, switched-off battery.
2. Place Go1 on a flat, non-slip, open floor with its abdomen down and all legs
   folded normally. No leg may be trapped beneath the body.
3. Turn on the original remote by pressing its power button once and then
   holding it for more than two seconds.
4. With everyone clear of the legs, turn on the Go1 battery by pressing its
   button once and then holding it for more than two seconds.
5. Wait for startup and the robot's normal automatic stand-up. If startup is
   abnormal, stop the experiment.
6. Use the factory remote and the lab's normal procedure to make Go1 lie down
   fully.
7. Press `L2+B` so the fully prone robot enters the factory damping state.
8. Do not lift the powered robot.
9. Connect Ubuntu to the Go1 Wi-Fi while leaving the wired lab/MOCAP network
   connected if needed.

Run on **Ubuntu**:

```bash
ip -br addr
ip route get 192.168.12.1
ping -c 5 192.168.12.1
```

The Go1 Wi-Fi interface should own an address in `192.168.12.0/24`, the route
to `192.168.12.1` should use that Wi-Fi interface, and the ping must succeed.
Do not replace Ubuntu's default route for this onboard workflow. The 500 Hz
process will run on the Pi, which reaches `192.168.123.10` directly over its
internal Ethernet.

### Step 3 — Update the Ubuntu checkout

Run on **Ubuntu**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git status --short
git pull --ff-only
git rev-parse --short HEAD
```

`git status --short` must be empty before pulling. If it is not empty, preserve
or resolve those changes; do not discard them just to force the pull.

### Step 4 — Copy only build inputs from Ubuntu to the Pi

Run on **Ubuntu**, from the repository root:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
ssh pi@192.168.12.1 'mkdir -p ~/Robotic-Dog-Tracking-Interface'
rsync -avR \
  ./CMakeLists.txt \
  ./src/go1_lowlevel_experiment.cpp \
  ./src/go1_kinematics.cpp \
  ./src/go1_kinematics.hpp \
  ./src/go1_log_file.hpp \
  ./test/go1_log_file_test.cpp \
  ./test/go1_kinematics_test.cpp \
  ./externals/unitree_legged_sdk/include/ \
  ./externals/unitree_legged_sdk/lib/cpp/arm64/ \
  pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/
```

This intentionally does not copy `experiment/`, analysis summaries, plots,
historical CSVs, Git history, or workstation build directories. The Python
analyzer remains on Ubuntu.

### Step 5 — Open one Pi SSH session and inspect storage

Run on **Ubuntu**:

```bash
ssh pi@192.168.12.1
```

Confirm the prompt begins with `pi@raspberrypi`. Steps 5 through 11 run in this
same Pi SSH session.

Run on the **Pi**:

```bash
cd ~/Robotic-Dog-Tracking-Interface
df -h /
find . -maxdepth 2 -type f -name '*.csv' \
  -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n'
```

Do not delete anything during inspection. If `/` is at or above 90% use, stop
and archive specific old logs before continuing. Keep the repository,
`build-arm64`, SDK headers/libraries, and experiment executable.

### Step 6 — Build and run all software tests on the Pi

Run on the **Pi**:

```bash
cd ~/Robotic-Dog-Tracking-Interface
cmake -S . -B build-arm64 \
  -DPYTHON_BUILD=OFF -DBUILD_SDK_EXAMPLES=OFF
cmake --build build-arm64 --target \
  go1_lowlevel_experiment go1_kinematics_test go1_log_file_test -j2
(cd build-arm64 && ctest --output-on-failure)
mkdir -p logs
find build-arm64 -maxdepth 1 -type f -name 'go1_dry_*.csv' \
  -print -delete
```

All 18 tests must pass. The parenthesized `cd` form is intentional: the older
CTest on the Pi does not support `ctest --test-dir` and can otherwise report
`No tests were found`. The final command deletes only test-generated dry-run
CSVs.

### Step 7 — Check the low-level route and controller processes

Run on the **Pi**:

```bash
ip route get 192.168.123.10
pgrep -af 'go1_lowlevel_experiment|example_|run_torque_tracking' \
  || echo 'OK: no known experiment controller is running'
pgrep -af '^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
sudo ss -Huanp | awk '$4 ~ /:8090$/ { print }'
sudo fuser -v 8090/udp
```

Required results:

- The route resembles
  `192.168.123.10 dev eth0 src 192.168.123.161`.
- No old experiment, SDK example, or legacy torque sender is running.
- Exactly one `programming.py` process is present.
- `ss` shows `192.168.123.161:8090` connected to
  `192.168.123.161:8082`.
- `fuser` identifies the same current `programming.py` owner.

Use `ss -Huanp`, not `ss -lunp`. The `-l` form can hide the connected UDP
socket that caused the earlier false conclusion that 8090 was free.

If the process and UDP owner do not agree, stop here. Do not kill anything.

### Step 8 — Reconfirm the prone damping state

The robot should have remained prone throughout deployment and building.
Immediately before changing the port owner, visually confirm all of the
following again:

1. Go1's abdomen is fully floor-supported.
2. All four legs are folded normally and unobstructed.
3. The factory remote is on.
4. `L2+B` has placed the prone robot in damping.
5. Everyone is clear of the legs.

Do not proceed if Go1 is standing or its state is uncertain.

### Step 9 — Temporarily stop only Unitree programming.py

Run the following block on the **Pi**. Its exact process pattern and one-PID
gate prevent reuse of a stale PID:

```bash
PROGRAMMING_PATTERN='^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
mapfile -t PROGRAMMING_PIDS < <(pgrep -f "$PROGRAMMING_PATTERN")

if [ "${#PROGRAMMING_PIDS[@]}" -ne 1 ]; then
  printf 'STOP: expected exactly one programming.py PID, found %s\n' \
    "${#PROGRAMMING_PIDS[@]}"
  pgrep -af "$PROGRAMMING_PATTERN"
else
  PROGRAMMING_PID="${PROGRAMMING_PIDS[0]}"
  ps -fp "$PROGRAMMING_PID"
  kill -TERM "$PROGRAMMING_PID"
  sleep 2
fi
```

Now verify on the **Pi**:

```bash
pgrep -af "$PROGRAMMING_PATTERN" \
  || echo 'OK: programming.py is temporarily stopped'
sudo ss -Huanp | awk '$4 ~ /:8090$/ { print }'
sudo fuser -v 8090/udp 2>&1 \
  || echo 'OK: fuser found no owner for UDP 8090'
python3 - <<'PY'
import socket

probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    probe.bind(("0.0.0.0", 8090))
except OSError as error:
    print(f"BUSY: UDP 8090: {error}")
else:
    print("FREE: UDP 8090")
finally:
    probe.close()
PY
```

The required final line is:

```text
FREE: UDP 8090
```

If `programming.py` reappears or the bind probe reports `BUSY`, do not kill it
again and do not run preflight. Restore/check the Unitree desktop session and
stop the experiment.

### Step 10 — Run exactly one prone preflight on standard port 8090

Run on the **Pi**:

```bash
cd ~/Robotic-Dog-Tracking-Interface
if pgrep -f "$PROGRAMMING_PATTERN" >/dev/null; then
  echo 'STOP: programming.py has reclaimed UDP 8090'
else
  ./build-arm64/go1_lowlevel_experiment --mode remote-preflight \
    --local-port 8090 \
    --prone-confirmed --duration-s 60 \
    --log logs/remote_preflight_fix_01.csv
fi
```

You can reuse the same log filename. If it exists, the executable asks before
arming or starting UDP:

```text
Log already exists: logs/remote_preflight_fix_01.csv
Replace it with this run's log when the run finishes? [y/N]:
```

Enter `y` or `yes` to approve replacement. Enter `n`, press Enter, or send EOF
to cancel: the old file stays intact and no motor commands are sent. After
approval, the usual `ARM DAMPING` confirmation still follows. The old file is
not truncated by either prompt; it is replaced when this run writes its log,
including a failed run. Archive any result you want to keep before approving.
Hardware confirmation does not change automated `--dry-run` output behavior.

If you cancel after stopping the Programming Module, proceed directly to
Step 11 to restore it. Do not analyze the old CSV as though it were a new run.

At the prompt type exactly:

```text
ARM DAMPING
```

During the 60 seconds:

1. move both joysticks through several directions;
2. press `L2+B` at least once;
3. confirm joystick numbers change and `L2+B=1` appears;
4. do not expect joystick motion to move the robot.

On success, the program exits by itself. If panic appears, wait for the final
damping window and automatic close. In either case, do not start a second run.

Confirm that the log exists:

```bash
ls -lh logs/remote_preflight_fix_01.csv
```

### Step 11 — Restore the Unitree Programming Module immediately

Keep Go1 fully prone. Run on the **Pi**:

```bash
if pgrep -f "$PROGRAMMING_PATTERN" >/dev/null; then
  echo 'Programming Module already running; not starting a duplicate'
else
  (cd /home/pi/Unitree/autostart/programming && bash ./programming.sh)
fi
sleep 2
pgrep -af "$PROGRAMMING_PATTERN"
sudo ss -Huanp | awk '$4 ~ /:8090$/ { print }'
```

This uses the vendor module's own confirmed wrapper, from the working directory
expected by its relative command `python3 programming.py &`. Do not start or
restart `startup_manager.py`.

Required results:

- exactly one `programming.py` process exists;
- UDP 8090 again shows the Programming Module's high-level connection to
  `192.168.123.161:8082`.

If either result is missing, keep the robot prone and stop. Do not run another
hardware experiment.

Exit the Pi session:

```bash
exit
```

Then verify from **Ubuntu** that the restored process survived SSH logout:

```bash
ssh pi@192.168.12.1 \
  "pgrep -af '^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'; \
   sudo ss -Huanp | awk '\$4 ~ /:8090\$/ { print }'"
```

If it did not survive logout, reconnect, keep Go1 prone, rerun the vendor
wrapper once, and diagnose that restoration before doing anything else.

### Step 12 — Copy and analyze the raw CSV on Ubuntu

Run on **Ubuntu**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
mkdir -p logs/downloaded
GO1_ARCHIVE_DIR=$(mktemp -d logs/downloaded/preflight-XXXXXXXX)
ssh pi@192.168.12.1 \
  'sha256sum ~/Robotic-Dog-Tracking-Interface/logs/remote_preflight_fix_01.csv'
scp pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/logs/remote_preflight_fix_01.csv \
  "$GO1_ARCHIVE_DIR/"
sha256sum "$GO1_ARCHIVE_DIR/remote_preflight_fix_01.csv"
python3 experiment/analyze_lowlevel_log.py \
  "$GO1_ARCHIVE_DIR/remote_preflight_fix_01.csv" --no-plots
```

The Pi and Ubuntu SHA-256 values must match. The analyzer runs only on Ubuntu.
The automatically named Ubuntu archive directory preserves earlier downloads
even when you reuse the same Pi filename. Keep the printed archive path.

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

The earlier `remote_preflight_tick_invalid` log contained one duplicated fresh
snapshot at `tick=332367`; later panic rows merely retained that latched reason.
The current receive path copies the packet and sequence under the same lock.

If any gate fails, keep the Ubuntu CSV and complete terminal output, then stop.
Diagnose that one run offline instead of repeating preflight.

### Step 13 — Delete only the verified Pi copy

Perform this only after the checksum matches, the Ubuntu file opens, and the
analysis command completes. Run on **Ubuntu**:

```bash
ssh pi@192.168.12.1 \
  'rm -- ~/Robotic-Dog-Tracking-Interface/logs/remote_preflight_fix_01.csv'
ssh pi@192.168.12.1 \
  'find ~/Robotic-Dog-Tracking-Interface/logs -maxdepth 1 \
   -type f -name "*.csv" -printf "%10s %p\n"'
```

This deletes only the exact verified Pi copy. The Ubuntu copy remains under
`logs/downloaded/`. Never use `rm *.csv`, and never delete `build-arm64` or the
SDK.

For an older Pi CSV, first copy that exact file, compare checksums, and inspect
the Ubuntu copy. Only then delete that exact Pi pathname.

### Step 14 — Shut down and end the experiment

Confirm Go1 remains fully prone and floor-supported. Shut it down with the
normal battery shutdown procedure. Do not power it off while standing.

After one passing run, the prone preflight gate is complete. Do not run it
again unless code affecting UDP reception, state freshness, remote decoding,
damping, watchdog, or panic handling changes.

Continue to Chapter 2's software rehearsal. Standing hardware operation remains
pending the entry and exit requirements below.

## Chapter 2 — Ground handover

Purpose: verify that taking over a stable standing robot does not introduce a
position jump, then hold its captured pose for 10 seconds. No squat or leg lift
is requested. This is the first intended hardware experiment after preflight.

### 2.1 Entry and exit requirements before hardware

The current program first requests a stable high-level standing pose through
Pi local UDP 8091 to `192.168.123.161:8082`, then starts low-level commands on
8090 to `192.168.123.10:8007`. This transition has not been validated on this
robot. A passing prone low-level preflight does not validate it.

Before floor execution, the implementation and operating procedure must provide:

1. A verified way to reach and capture the standing start state while releasing
   8090 from the Programming Module. Do not simply repeat the prone preflight's
   mode-switch sequence and then expect this command to stand the robot up.
2. Continuous support during high-level to low-level takeover, including the
   first low-level feedback interval. If high-level capture fails, investigate
   that entry path; do not bypass the capture check.
3. A normal endpoint that lowers the robot to a verified floor-supported prone
   pose, enters damping, writes the log, and exits. The state machine now exists
   as a **development-only fixture**, described in 2.1.1. A calibrated target
   and a real, independent floor-support confirmation input are still missing.
4. Software tests for normal completion, operator cancellation, and loss of
   feedback during that endpoint, followed by a reviewed first-hardware procedure.

Until these are complete, continue endpoint development and targeted tests.
The four original software rehearsals are already complete. Do not improvise an exit with
double Ctrl-C, a factory remote command during takeover, or battery removal.
No reliable support rig has been established for this setup.

### 2.1.1 Normal-exit development and test status

The implemented development path is:

```text
GROUND_HANDOVER -> RETURN -> EXIT_LOWER (8 s)
  -> EXIT_VERIFY_SUPPORT (at least 1 s stable, at most 5 s waiting)
  -> EXIT_DAMPING (1 s) -> COMPLETE -> write CSV -> process exit
```

It is selected only by `--dry-run --mode ground-handover --dry-run-normal-exit`.
The original ordinary dry-run behavior remains reproducible, so the existing
archives are still valid records of that earlier action rehearsal. They do not
test the new endpoint.

The descent uses quintic interpolation, zero feed-forward torque, and the
existing impedance gains. Its simulation-only target, in FR/FL/RR/RL order,
is `(-0.28, 1.25, -2.70)`, `(0.28, 1.25, -2.70)`,
`(-0.28, 1.25, -2.70)`, `(0.28, 1.25, -2.70)` rad.
These values are **not an approved hardware lie-down pose**. In particular,
the observed factory calf positions around -2.80 rad are outside the SDK
command bounds and are not replayed as targets. Only the development endpoint
can exceed the ordinary +/-0.3 rad displacement envelope; SDK joint bounds
remain enforced, with a 0.10 rad tracking-error limit, 0.3 rad/s speed limit,
and 0.10 rad relative roll/pitch limit during the endpoint.

Support verification requires new valid feedback, all joint errors below
0.05 rad, all joint speeds below 0.05 rad/s, and relative roll/pitch below
0.10 rad for at least one second, together with an independent, current
floor-support observation. Foot-force unloading, elapsed time, or reaching
the target alone cannot prove that the belly is supported. An observation
before this stage is not latched as permission. The development simulator
explicitly supplies a synthetic observation; there is currently no hardware
input for it. CSV adds `exit_support_confirmed` and `exit_stable_s`; a true
flag in this fixture records simulated evidence only.

Cancellation and faults have distinct outcomes:

| Event | Result |
| --- | --- |
| Single Ctrl-C in handover or return | Return to captured standing hold; do not begin descent |
| Single Ctrl-C during descent/verification | `EXIT_HOLD`: retain last commanded position, clear velocity reference and torque; do not automatically rise or exit |
| Support not confirmed within 5 s | Failed `EXIT_HOLD`, retaining impedance; no automatic damping or exit |
| Feedback older than 20 ms, watchdog, invalid feedback, L2+B, double Ctrl-C | Immediate `PANIC_DAMPING`; hardware panic semantics remain latched |
| Failed send reported during normal damping | Panic; do not report successful normal exit |
| Verified support and completed damping dwell | Write CSV, check write success, then exit |

Holding a target after cancellation is not a dynamically validated braking
trajectory. The low-speed cancellation and actual support-confirmation method
still require engineering review before hardware. A successful SDK Send result
also does not acknowledge physical damping at the motors.

Developer checks run on the **development computer**, with no SSH or robot UDP.
They cover normal descent/exit, missing and premature support confirmation,
cancellation in handover/return/descent/verification, feedback loss during
descent/verification/damping, remote stop, double Ctrl-C, watchdog, send
failure, invalid IMU, joint limits, tracking error, and hardware CLI rejection.
The state-machine fixture uses ideal tracking and independently controlled
support evidence; the integration fixture uses the existing simplified plant.
Neither models belly contact or proves physical stability.

#### Test the new endpoint now — one Ubuntu terminal

Run this new code check on **Ubuntu (`aims-Precision-7780`)**, in the local
terminal. Do not SSH into the Pi. Go1 can remain powered off. This is the
targeted test of the newly added endpoint, separate from the four completed
ordinary rehearsals. Nothing needs to be copied onto the Pi.

**1. Synchronize the source.** On the development computer, commit and push
the current changes through the existing GitHub workflow first. Then on Ubuntu:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git status --short
git pull --ff-only
git submodule update --init --recursive
```

If Git reports conflicts or a failed pull, stop and retain the output; do not
reset or discard local changes. The following file must now exist:

```bash
ls test/go1_ground_exit_test.cpp
```

**2. Build the two targets in a separate temporary build directory.** Continue
in that same Ubuntu terminal, from the repository directory:

```bash
cmake -S . -B /tmp/go1-exit-build -DBUILD_TESTING=ON -DBUILD_SDK_EXAMPLES=OFF
cmake --build /tmp/go1-exit-build --target go1_ground_exit_test go1_lowlevel_experiment -j2
```

Both commands must succeed before continuing. This leaves the existing Pi
`build-arm64` installation untouched.

**3. Run only the new endpoint tests.** The subshell form below also works with
older CTest versions that do not support `--test-dir`:

```bash
(cd /tmp/go1-exit-build && ctest -R '^go1_ground_exit_' -V)
```

The first test prints five `[PASS]` lines, covering normal completion, missing
support, cancellation, fault handling, and hardware lockout. The second test
runs the normal-exit integration fixture and writes its simulated CSV.
The final required result is:

```text
100% tests passed, 0 tests failed out of 2
```

`No tests were found` is not a pass. A `[FAIL]`, build error, or fewer than two
tests means stop here and retain the complete output. Send the output from
this step for review; no additional robot preflight is needed.

**4. Locate the result and optionally inspect it on Ubuntu.** The only generated
CSV is `/tmp/go1-exit-build/go1_dry_normal_exit.csv` on Ubuntu. It is a disposable
developer artifact, not a new hardware archive. For the usual summary:

```bash
conda activate dog_ctrl
python3 experiment/analyze_lowlevel_log.py \
  /tmp/go1-exit-build/go1_dry_normal_exit.csv --no-plots
```

The analyzer writes `/tmp/go1-exit-build/go1_dry_normal_exit.csv.summary.csv`.
The two passing tests are the endpoint acceptance criterion; the analyzer's
network and motion numbers describe simulated inputs, not robot performance.
The old Ubuntu archives, including both handover archives, are unchanged.

**5. Optional cleanup after a passing result has been reviewed.** There are no
new Pi files to clean. On Ubuntu, remove only the generated simulation log and,
if Step 4 was run, its summary; `-i` asks before each removal:

```bash
rm -i -- /tmp/go1-exit-build/go1_dry_normal_exit.csv
rm -i -- /tmp/go1-exit-build/go1_dry_normal_exit.csv.summary.csv
```

Keep failed-test output until diagnosed. No deletion is needed to continue.
Passing this check completes the new software endpoint test only; keep the
hardware gate below in place.

Before hardware can be enabled, review/calibrate the final pose and descent,
provide an independent support-confirmation input with a tested failure path,
and review the standing capture/takeover sequence. Then revise the first-run
procedure and the code lock together. Do not remove `--dry-run` to proceed.

### 2.2 Original Pi rehearsal — completed; reference only

**Completed on 2026-09-08; do not repeat this normal rehearsal just to proceed.**
The reported run completed with 8,251 samples (about 16.5 simulated seconds).
The Pi and Ubuntu SHA-256 both matched:

```text
f460d30a2c2c23570e501a47ebd0c4982b66a70f9ac920bf5f235fc820202944
```

The Ubuntu archive is
`~/Yuxuan/Robotic-Dog-Tracking-Interface/logs/dry-run/handover-Z6dkjhnB/`.
It contains the CSV, `.summary.csv`, and `ground_handover_plots/`.
The operator also confirmed the separate `handover-SB8m9xb7` archive.
Keep both; optional deduplication needs checksum and analysis comparison first.
Acceptance here is based on the supplied terminal output; the raw CSV and
plots have not been independently inspected on the development computer.
The commands below remain available for a future required rerun.

In the **Pi SSH terminal**:

```bash
cd ~/Robotic-Dog-Tracking-Interface
mkdir -p logs/dry-run
./build-arm64/go1_lowlevel_experiment --dry-run --mode ground-handover \
  --log logs/dry-run/ground_handover.csv
sha256sum logs/dry-run/ground_handover.csv
```

The simulated phase sequence is `PRECHECK -> CAPTURE_POSE -> HOLD ->
GROUND_HANDOVER -> RETURN -> SAFE_HOLD -> COMPLETE`. The dedicated hold lasts
10 seconds, in addition to precheck, initial hold, and return. `--duration-s`
does not change this mode's fixed hold time. Dry-run automatically leaves
`SAFE_HOLD` after two seconds; hardware does not.

### 2.3 Download and inspect on Ubuntu

In an **Ubuntu terminal**, not the Pi SSH terminal:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
mkdir -p logs/dry-run
GO1_REVIEW_DIR=$(mktemp -d logs/dry-run/handover-XXXXXXXX)
scp pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/logs/dry-run/ground_handover.csv \
  "$GO1_REVIEW_DIR/"
sha256sum "$GO1_REVIEW_DIR/ground_handover.csv"
python3 experiment/analyze_lowlevel_log.py \
  "$GO1_REVIEW_DIR/ground_handover.csv"
```

Compare this hash with 2.2. Inspect the summary and plots printed by the analyzer.
Check that commanded positions stay at the captured pose during
`GROUND_HANDOVER` and feed-forward torque stays zero. Synthetic timing and
tracking do not predict actual takeover behavior or validate normal shutdown.

For the reported normal handover dry-run, the following are expected:

- `rate=500.00 Hz`, 2 ms gaps: the simulator supplies these timestamps; this is
  not a measurement of the Pi's real scheduler or network.
- Zero roll/pitch excursion and joint speed: the simulated robot starts at its
  hold target and there is no motion reference in this mode.
- `L2+B_seen=0`: no stop was injected. A normal handover must not require an
  emergency-stop event to pass.
- Support and remote-stop latency `nan`: this run has no lift/contact-verification
  phases and no remote-stop transition to measure. These are not failures here.
- No abort, duplicate fresh tick, or watchdog event: the normal dry-run passed.

The repeated `scp` shown in the terminal copied the same file to the same local
pathname; it did not create a second CSV in that archive directory. The final
matching checksum confirms the downloaded file. There is no extra copy there
to delete merely because `scp` was run twice.

### 2.4 Clean the archived handover log from the Pi

Run in **Ubuntu window 2**, after reviewing the summary. This block uses the
known archive path from the passing run so it also works after reopening the
terminal. For a future rerun, use that run's archive directory instead.

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
GO1_REVIEW_DIR=logs/dry-run/handover-Z6dkjhnB
if [ -s "$GO1_REVIEW_DIR/ground_handover.csv.summary.csv" ] && \
   GO1_CHECKSUM=$(sha256sum "$GO1_REVIEW_DIR/ground_handover.csv"); then
  GO1_CHECKSUM=${GO1_CHECKSUM%% *}
  ssh pi@192.168.12.1 \
    "cd /home/pi/Robotic-Dog-Tracking-Interface/logs/dry-run && \
     printf '%s\n' '$GO1_CHECKSUM  ground_handover.csv' | sha256sum -c - && \
     rm -- /home/pi/Robotic-Dog-Tracking-Interface/logs/dry-run/ground_handover.csv"
else
  echo 'STOP: local raw log or summary missing; nothing deleted'
fi
```

Only a matching Pi copy is deleted. The Ubuntu CSV, summary, and plots remain.
If the Pi log has been regenerated since download, its hash will differ and
deletion will not run. Do not start a new run while checking/removing its log.
No need to stop or restore Programming Module for a dry-run: it opens no robot
UDP socket. All four original dry-runs have now been archived; proceed to
endpoint development in 2.1.1, not another routine rehearsal.

### 2.5 Hardware test reference — pending 2.1

Existing Pi CLI, **not an executable instruction for the current floor setup**:

```text
./build-arm64/go1_lowlevel_experiment --mode ground-handover \
  --ground-confirmed --local-port 8090 --high-local-port 8091 \
  --log logs/ground_handover.csv
```

Once 2.1 is implemented and this procedure updated, each run must include the
complete chain: verified standing entry, takeover, 10-second hold, normal
lie-down, process exit, Programming Module restoration, Ubuntu download and
analysis. Perform one run at a time; require three passing runs before Chapter 3.

Acceptance targets: no abrupt leg or body movement, no unexpected protection,
no watchdog event, all-joint `position_rms_rad < 0.08`,
`return_error_rad < 0.05`, and roll/pitch excursions below 0.15 rad. Retain the
preflight communication thresholds (>=450 Hz, p99 <=10 ms, maximum gap <=20 ms).
The return error currently measures return to standing, not success of a future
lie-down endpoint. That endpoint needs its own final-pose/contact acceptance.

Dedicated stop tests are separate from normal passing runs: a commanded
`remote_l2_b` panic is expected in a remote-stop test, but is a failed normal
action run. Receiving `L2+B` during preflight did not measure an actual switch
from position hold to damping. Verify that response with appropriate support
before relying on it during a leg lift.

## Chapter 3 — Squat and return

Purpose: exercise all four legs with slow impedance-controlled motion while
keeping four-foot contact. Hardware prerequisite: Chapter 2's complete entry,
hold, and normal exit passed three times.

### 3.1 Original Pi rehearsal — completed; reference only

The operator confirmed the raw CSV and summary are archived on Ubuntu.
No routine rerun is required.

In the **Pi SSH terminal**:

```bash
cd ~/Robotic-Dog-Tracking-Interface
mkdir -p logs/dry-run
./build-arm64/go1_lowlevel_experiment --dry-run --mode squat \
  --log logs/dry-run/squat.csv
sha256sum logs/dry-run/squat.csv
```

After capture and initial hold, the action takes eight seconds: three seconds
down, two seconds held, three seconds back. All thighs move +0.12 rad and calves
-0.24 rad relative to capture; hips stay fixed. A two-second return phase then
leads to `SAFE_HOLD`. This is joint-space motion, not a calibrated body-height
command, and `--duration-s` does not change its timing.

### 3.2 Download and inspect on Ubuntu

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
mkdir -p logs/dry-run
GO1_REVIEW_DIR=$(mktemp -d logs/dry-run/squat-XXXXXXXX)
scp pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/logs/dry-run/squat.csv \
  "$GO1_REVIEW_DIR/"
sha256sum "$GO1_REVIEW_DIR/squat.csv"
python3 experiment/analyze_lowlevel_log.py "$GO1_REVIEW_DIR/squat.csv"
```

Compare hashes. Check smooth symmetric thigh/calf targets, position tracking,
return error, IMU excursions, and foot forces. This action has no intended
feed-forward torque excitation, so a torque-correlation threshold is not its
acceptance criterion.

### 3.3 Clean the archived squat log from the Pi, then continue

Run in the **same Ubuntu window 2** used for 3.2, after reviewing that result:

```bash
if [ -s "$GO1_REVIEW_DIR/squat.csv.summary.csv" ] && \
   GO1_CHECKSUM=$(sha256sum "$GO1_REVIEW_DIR/squat.csv"); then
  GO1_CHECKSUM=${GO1_CHECKSUM%% *}
  ssh pi@192.168.12.1 \
    "cd /home/pi/Robotic-Dog-Tracking-Interface/logs/dry-run && \
     printf '%s\n' '$GO1_CHECKSUM  squat.csv' | sha256sum -c - && \
     rm -- /home/pi/Robotic-Dog-Tracking-Interface/logs/dry-run/squat.csv"
else
  echo 'STOP: local raw log or summary missing; nothing deleted'
fi
```

Keep the Ubuntu archive. The single-leg dry-run in 4.1 is also complete.
For any future changed-code rehearsal with an abort or missing phase, preserve
the result for diagnosis instead of moving on. To reuse this block after
opening a new terminal, first set `GO1_REVIEW_DIR` to the archive path printed
in 3.2; do not create an empty replacement directory.

### 3.4 Hardware test reference — pending Chapter 2

```text
./build-arm64/go1_lowlevel_experiment --mode squat \
  --ground-confirmed --local-port 8090 --high-local-port 8091 \
  --log logs/squat.csv
```

The current executable rejects this hardware command before opening UDP.
The ordinary squat simulation ends at standing `SAFE_HOLD`; Chapter 2's
development exit is not yet integrated or validated for this mode.
Use the verified entry/exit and archive cycle for each repetition. Require three
normal runs with all four feet maintaining contact, no protection or watchdog,
all-joint `position_rms_rad < 0.08`, `return_error_rad < 0.05`, and roll/pitch
excursions <0.15 rad. Communication must meet Chapter 2's thresholds. Review
each run before repeating. Only then proceed to Chapter 4 hardware testing.

## Chapter 4 — Single-leg lift

Purpose: test weight transfer, one small leg lift, and confirmed touchdown.
Hardware prerequisites: three passing squats, a verified stop response, and
adequate fall protection with another person present for the initial test.
Three contacting feet alone do not guarantee static stability. The estimated
CoP margin is a load-distribution check, not a complete stability guarantee.

### 4.1 Original Pi rehearsal — completed; reference only

The operator confirmed the raw CSV and summary are archived on Ubuntu.
No routine rerun is required.

```bash
cd ~/Robotic-Dog-Tracking-Interface
mkdir -p logs/dry-run
./build-arm64/go1_lowlevel_experiment --dry-run --mode leg-lift \
  --leg auto --lift-height-m 0.02 \
  --tau-overlay-nm 0.10 --tau-overlay-hz 0.5 \
  --log logs/dry-run/leg_lift.csv
sha256sum logs/dry-run/leg_lift.csv
```

Expected action sequence:

1. Collect two seconds of foot-force baseline and select the candidate leg with
   the largest estimated support margin; initially hold for one second.
2. Shift the body target over two seconds, limited to 30 mm.
3. Permit lift only after margin >=15 mm and target-foot load <=30% of baseline
   have held for 0.5 seconds.
4. Lift the target foot 20 mm over 1.5 seconds, hold for one second, then lower
   over 1.5 seconds. During air hold, add 0.10 Nm at 0.5 Hz to its thigh.
5. Confirm target-foot force >=60% of baseline for 0.5 seconds (verification
   times out after 1.5 seconds), then recenter over two seconds.
6. Return to the captured pose and enter `SAFE_HOLD`.

The overlay's one-second window at 0.5 Hz covers only half a sine cycle. It is
an action-chain/torque-channel smoke test, not a zero-mean periodic experiment
or a torque frequency-response measurement.

### 4.2 Download and inspect on Ubuntu

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
mkdir -p logs/dry-run
GO1_REVIEW_DIR=$(mktemp -d logs/dry-run/leg-lift-XXXXXXXX)
scp pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/logs/dry-run/leg_lift.csv \
  "$GO1_REVIEW_DIR/"
sha256sum "$GO1_REVIEW_DIR/leg_lift.csv"
python3 experiment/analyze_lowlevel_log.py "$GO1_REVIEW_DIR/leg_lift.csv"
```

Compare hashes and inspect `active_leg`, foot targets, force baselines, CoP,
support margins, and the full phase sequence. Check torque against
`tau_cmd_total = tau_ff + Kp*(qd-q) + Kd*(dqd-dq)`; `tauEst` includes the
impedance contribution and loading effects.

### 4.3 Clean the archived single-leg log from the Pi, then continue

Run in the **same Ubuntu window 2** used for 4.2, after reviewing that result:

```bash
if [ -s "$GO1_REVIEW_DIR/leg_lift.csv.summary.csv" ] && \
   GO1_CHECKSUM=$(sha256sum "$GO1_REVIEW_DIR/leg_lift.csv"); then
  GO1_CHECKSUM=${GO1_CHECKSUM%% *}
  ssh pi@192.168.12.1 \
    "cd /home/pi/Robotic-Dog-Tracking-Interface/logs/dry-run && \
     printf '%s\n' '$GO1_CHECKSUM  leg_lift.csv' | sha256sum -c - && \
     rm -- /home/pi/Robotic-Dog-Tracking-Interface/logs/dry-run/leg_lift.csv"
else
  echo 'STOP: local raw log or summary missing; nothing deleted'
fi
```

Keep the Ubuntu archive. After the single-leg rehearsal completes all required
phases without abort, retain that acceptance. The sequence dry-run in 5.1
is also already complete. In a new Ubuntu
terminal, restore `GO1_REVIEW_DIR` to the actual 4.2 archive path first.

### 4.4 Hardware test reference — pending Chapters 2–3

```text
./build-arm64/go1_lowlevel_experiment --mode leg-lift \
  --leg auto --lift-height-m 0.02 \
  --tau-overlay-nm 0.10 --tau-overlay-hz 0.5 \
  --ground-confirmed --remote-confirmed \
  --local-port 8090 --high-local-port 8091 --log logs/leg_lift.csv
```

Use the verified standing entry and normal exit, once available for this mode.
Require three complete passing single-leg trials before Chapter 5. Record the
actual selected leg each time; `auto` may choose a different leg when loading
changes. Never relax a load-transfer threshold merely to make the robot lift.

Each trial must pass the baseline noise checks, unloading gate, and touchdown
gate. During air hold, force must remain below 20% of baseline and support
margin >=10 mm; total force must stay within 70–130% of baseline. Inspect the
whole interval: a low `min_airborne_force_ratio` alone cannot prove that force
stayed below the limit throughout. Require no protection/watchdog event and
the same position/return thresholds as Chapter 3. Keep leg-mode roll/pitch
excursions <=0.10 rad and speed <=0.8 rad/s.

## Chapter 5 — Four-leg sequence

Purpose: repeat the validated single-leg action across all four legs with
verified contact between lifts. Hardware prerequisite: Chapter 4 has passed
three times and the same entry, exit, and fall-protection arrangements apply.

### 5.1 Original Pi rehearsal — completed; reference only

The operator confirmed the raw CSV and summary are archived on Ubuntu.
No routine rerun is required.

```bash
cd ~/Robotic-Dog-Tracking-Interface
mkdir -p logs/dry-run
./build-arm64/go1_lowlevel_experiment --dry-run --mode leg-lift-sequence \
  --leg auto --lift-height-m 0.02 \
  --tau-overlay-nm 0.10 --tau-overlay-hz 0.5 \
  --log logs/dry-run/leg_sequence.csv
sha256sum logs/dry-run/leg_sequence.csv
```

The program completes touchdown and recentering, holds for three seconds,
collects a new two-second baseline, and chooses the next untested leg. The
order is determined by current estimated margin, not fixed FR/FL/RR/RL order.
A failed leg gate stops progression to later legs. `--leg auto` is required.

### 5.2 Download and inspect on Ubuntu

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
mkdir -p logs/dry-run
GO1_REVIEW_DIR=$(mktemp -d logs/dry-run/leg-sequence-XXXXXXXX)
scp pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/logs/dry-run/leg_sequence.csv \
  "$GO1_REVIEW_DIR/"
sha256sum "$GO1_REVIEW_DIR/leg_sequence.csv"
python3 experiment/analyze_lowlevel_log.py "$GO1_REVIEW_DIR/leg_sequence.csv"
```

Compare hashes. Confirm all four distinct legs complete the lift, air-hold,
lower, contact-verification, and recenter phases. Examine each leg separately;
the analyzer's aggregate minimum margin and final contact ratio cannot certify
all four touchdowns. Keep the per-leg phase and force traces with the summary.

### 5.3 Clean the archived sequence log and finish software testing

Run in the **same Ubuntu window 2** used for 5.2, after reviewing that result:

```bash
if [ -s "$GO1_REVIEW_DIR/leg_sequence.csv.summary.csv" ] && \
   GO1_CHECKSUM=$(sha256sum "$GO1_REVIEW_DIR/leg_sequence.csv"); then
  GO1_CHECKSUM=${GO1_CHECKSUM%% *}
  ssh pi@192.168.12.1 \
    "cd /home/pi/Robotic-Dog-Tracking-Interface/logs/dry-run && \
     printf '%s\n' '$GO1_CHECKSUM  leg_sequence.csv' | sha256sum -c - && \
     rm -- /home/pi/Robotic-Dog-Tracking-Interface/logs/dry-run/leg_sequence.csv"
else
  echo 'STOP: local raw log or summary missing; nothing deleted'
fi
ssh pi@192.168.12.1 \
  'df -h /; find /home/pi/Robotic-Dog-Tracking-Interface/logs -maxdepth 2 \
   -type f -name "*.csv" -printf "%10s %p\n"'
```

Keep all four Ubuntu raw logs, summaries, and plots as the software test record.
Remaining Pi filenames are an inventory, not permission for a bulk delete.
For an older file, archive and hash-check its exact pathname first. Never
remove all of `logs/`, the SDK, or `build-arm64` to reclaim log storage.

CTest-generated files, if left by a previous build, can be removed on the **Pi**
after CTest has finished:

```bash
cd ~/Robotic-Dog-Tracking-Interface
find build-arm64 -maxdepth 1 -type f -name 'go1_dry_*.csv' -print -delete
```

This targets simulated test outputs only. Downloaded Ubuntu plots and summaries
are useful review artifacts, so this procedure keeps them. No raw hardware
records are deleted by these dry-run cleanup blocks.

All four original dry-runs are confirmed complete and archived. The software
rehearsal sequence is complete; no routine repetition is requested.
Do not convert these commands to hardware commands by removing `--dry-run`.
The next development gate is Chapter 2.1.1's endpoint validation and calibrated
support confirmation, plus verified standing entry, followed by hardware acceptance. If finishing the
session, leave SSH with `exit`; if Go1 is still powered, use the established
shutdown procedure only once it is fully prone and floor-supported.

### 5.4 Hardware test reference — pending Chapters 2–4

```text
./build-arm64/go1_lowlevel_experiment --mode leg-lift-sequence \
  --leg auto --lift-height-m 0.02 \
  --tau-overlay-nm 0.10 --tau-overlay-hz 0.5 \
  --ground-confirmed --remote-confirmed \
  --local-port 8090 --high-local-port 8091 --log logs/leg_sequence.csv
```

Once released for hardware, require three complete sequences. Each of the four
legs must independently meet Chapter 4's gates, without aborts, watchdog events,
or support-margin violations. Archive and review one sequence before running
the next. These experiments establish the hybrid-impedance action chain; they
do not establish pure-torque tracking bandwidth or readiness for rear-leg-only
standing.

## Reference — Why the 500 Hz loop runs onboard

An onboard run measured about `469.63 Hz`, with a `4.096 ms` p99 gap and an
`8.542 ms` maximum gap. An Ubuntu-direct Wi-Fi run measured only `339.42 Hz`
and a `142.002 ms` maximum gap. Therefore:

```text
Pi:      500 Hz motor loop, watchdog, and safety state machine
Ubuntu:  SSH/deployment, CSV analysis, future Qualisys and MPPI
```

The intended later architecture is:

```text
Qualisys -> Ubuntu MPPI -> lower-rate references -> Pi 500 Hz safety loop -> Go1
```

MOCAP may provide optional ground truth or low-rate supervision. It is not
required for the preflight and must not enter the fast motor loop.

## Reference — Narrow startup diagnostics

The Programming Module startup evidence is:

```text
/home/pi/Unitree/autostart/.startlist.sh: programming
/home/pi/Unitree/autostart/programming/programming.sh:
  python3 programming.py &
```

If this must be rechecked later, use narrowly scoped commands:

```bash
sed -n '1,260p' /home/pi/Unitree/autostart/startup_manager.py
sed -n '1,40p' /home/pi/Unitree/autostart/programming/programming.sh
grep -nE 'programming|startup_manager' \
  /home/pi/Unitree/autostart/.startlist.sh \
  /home/pi/.config/lxsession/LXDE-pi/autostart \
  /etc/xdg/lxsession/LXDE-pi/autostart 2>/dev/null
```

Do not recursively grep all of `/home/pi/Unitree/autostart`. That tree contains
large JavaScript source maps and pybind11 documentation; the earlier recursive
command produced hundreds of kilobytes of irrelevant output.
