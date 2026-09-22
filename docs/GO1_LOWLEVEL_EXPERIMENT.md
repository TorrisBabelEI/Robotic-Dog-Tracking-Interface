# Go1 low-level experiments: staged operating manual

Each experiment has its own chapter. Complete the previous chapter's hardware
acceptance before starting the next chapter's hardware test. Simulation results
do not count as hardware acceptance.

| Chapter | Experiment | Current status |
| --- | --- | --- |
| [1](#chapter-1--remote-preflight) | Communication and remote preflight | Reported passing run: `remote_preflight_fix_02.csv` |
| [2](#chapter-2--ground-handover) | Standing takeover and 10-second hold | Bounded prone entry/release accepted; active exit tests next; standing still locked |
| [3](#chapter-3--squat-and-return) | Four-leg half-squat and return | Original dry-run completed and archived; hardware pending Chapter 2 |
| [4](#chapter-4--single-leg-lift) | Weight transfer and one leg lift | Original dry-run completed and archived; hardware pending Chapter 3 |
| [5](#chapter-5--four-leg-sequence) | Four sequential leg lifts | Original dry-run completed and archived; hardware pending Chapter 4 |

**Current status (2026-09-22): bounded prone engagement and release accepted;
selected Wenjian integrations pass Ubuntu verification.**
Section 2.1.30 records the successful `prone_engagement_03.csv` hardware result
and operator confirmation. The settling defect is corrected; the initial
watchdog flag occurred during damping-only observation, with none during
engagement/release. Factory module restoration survived SSH logout.
Next is the combined Pi build/check in 2.1.32, followed by the two-case hardware
exit block in 2.1.31: single-Ctrl-C cancellation and L2+B stop at the same small
gain. Normal engagement, old support probes, and completed simulation blocks
do not need another operator run. No rise, standing or torque waveform is
released yet. Earlier aborted trials remain documented as historical evidence.

Earlier, the operator confirmed no support equipment and requested continued
development without external lifting. Section 2.1.6 is closed; lack of a rig is not a
project-wide stop condition. The old standing-handover executable remains
locked because its transition and endpoint are still unvalidated. The new
route is specified in 2.1.7; its first implementation gate is in 2.1.9.
Wenjian's subsequent work describes belly support with feet airborne. That is
a different posture: it does not satisfy this manual's grounded prone trial
conditions or qualify the separate supported-hold controller. If that is the
current setup, complete only the non-actuating build/check in 2.1.32 and defer
2.1.31 until its grounded posture is established by the operator.
Section 2.1.5 passed twice on Ubuntu: four checks, 1/1 test, and
`test_exit=0` in both runs. Reports are `logs/capture-review-VrsaobOT/` and
`logs/capture-review-l6xWpjfG/`. Retain either or both; no repeat is required.
Section 2.1.4 passed twice on Ubuntu; the last result was 1/1 with
`test_exit=0`, archived at `logs/entry-review-YNGYdJWV/`. Both runs are valid;
no additional repetition or cleanup is needed.
Section 2.1.3 passed on Ubuntu: all five support-confirmation checks passed,
CTest completed 1/1 with `test_exit=0`; report: `logs/support-review-q5unojt4/`.
Do not repeat that completed check to proceed.
Section 2.1.1 passed on Ubuntu: both endpoint tests and the log analyzer
completed successfully. Do not repeat that completed check to proceed.
The operator has confirmed that all four original dry-run CSV files and their
summaries are fully archived on Ubuntu. `handover-SB8m9xb7` and
`handover-Z6dkjhnB` are two independent archives; retain both. They are not
evidence of an overwritten download. No routine dry-run or preflight repetition
is requested. The old rehearsal commands remain below as reference only.

The prone-engagement normal exit has now been demonstrated with operator
contact confirmation. Standing takeover and low-rise remain separate,
unvalidated hardware transitions. Ground hardware modes now fail before ARM or UDP initialization;
there is no command-line override. The hardware references in Chapters 2–5
remain blocked pending endpoint and standing-takeover review.

## Common setup and operating rules

### Execution ownership — operator preference recorded 2026-09-20

Codex runs local Ubuntu simulations, offline analysis, builds, and software
verification directly. After a passing local block, continue to the next
authorized step without waiting for the operator to say proceed. Stop when
Pi execution, physical observation, or a material operating decision is needed. Do not ask the operator to rerun a simulation that Codex
can execute in this workspace merely to advance the conversation. For each
work block, document the objective, exact commands/source revision or hashes,
archive path, results, failures and corrections, and remaining acceptance gaps
in this manual or an explicitly linked report. Preserve completed evidence;
rerun only when changed code or an unresolved result warrants it.

When a block requires execution on the Pi, involve the operator with one
complete, reviewable command sequence and expected outcomes. The operator
participates in Pi execution and physical robot observations; do not launch
Pi processes or robot motion autonomously. Complete available local preparation
and verification first. This preference does not itself release a hardware lock.

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
floor-support observation throughout that same one-second interval (strengthened
in 2.1.3). Withdrawal resets the interval, including between fresh robot
packets. Foot-force unloading, elapsed time, or reaching
the target alone cannot prove that the belly is supported. An observation
before this stage is not latched as permission. The development simulator
explicitly supplies a synthetic observation; there is currently no hardware
input for it. CSV adds `exit_support_confirmed` and `exit_stable_s`; the latter now measures
the overlapping stable-feedback/support interval. A true
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

#### Completed Ubuntu endpoint test — reference commands

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

### 2.1.2 Inspect the archived prone pose — completed

Purpose: extract a quiet measured pose from the already passing
`remote_preflight_fix_02.csv`, and compare all 12 measured joint ranges with
our command bounds. This supplies evidence for endpoint design; it does not
calibrate a lie-down target or prove belly contact. Do not copy or clamp these
measurements into motor commands. A calf outside the command bounds is a
finding to review, not a reason to widen the bounds.

Run all steps below in **one Ubuntu terminal (`aims-Precision-7780`)**.
Go1 can remain powered off. No Pi deployment, SSH, MOCAP, build, or new robot
run is needed. Stop after this section and send the results before proceeding.

**1. Obtain the new offline script.** After the development changes have been
committed and pushed through the existing GitHub workflow, run on Ubuntu:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git status --short
git pull --ff-only
ls experiment/inspect_prone_pose.py
conda activate dog_ctrl
```

Stop on a failed pull or missing script; retain the output. Do not discard
local changes. The script uses only the Python standard library.

**2. Locate and select the archived passing CSV.** This searches Ubuntu logs
only and asks you to select a numbered path if multiple archives exist:

```bash
mapfile -d '' GO1_POSE_LOGS < <(find "$PWD/logs" -type f -name 'remote_preflight_fix_02.csv' -print0)
GO1_POSE_LOG=''
if [ "${#GO1_POSE_LOGS[@]}" -eq 0 ]; then
  echo 'STOP: archived remote_preflight_fix_02.csv not found; send this output'
elif [ "${#GO1_POSE_LOGS[@]}" -eq 1 ]; then
  GO1_POSE_LOG="${GO1_POSE_LOGS[0]}"
else
  PS3='Select the passing preflight archive number: '
  select GO1_POSE_LOG in "${GO1_POSE_LOGS[@]}"; do
    [ -n "$GO1_POSE_LOG" ] && break
  done
fi
printf 'Selected CSV: %s\n' "$GO1_POSE_LOG"
```

Use Ubuntu's normal Bash terminal for these commands. If no file is found,
stop and send the output; do not generate a replacement hardware run. If you
previously stored the archive outside `logs`, send its location for the next
instruction. Keep the selected original CSV unchanged.

**3. Run the existing summary and the new pose inspection.** Continue only
with a selected existing CSV. Each invocation creates a new report directory:

```bash
if [ -n "$GO1_POSE_LOG" ] && [ -f "$GO1_POSE_LOG" ]; then
  GO1_POSE_REVIEW=$(mktemp -d "$PWD/logs/prone-review-XXXXXXXX")
  sha256sum "$GO1_POSE_LOG" > "$GO1_POSE_REVIEW/source.sha256"
  python3 experiment/analyze_lowlevel_log.py "$GO1_POSE_LOG" --no-plots \
    --summary "$GO1_POSE_REVIEW/preflight.summary.csv" \
    > "$GO1_POSE_REVIEW/preflight.txt" 2>&1
  GO1_SUMMARY_STATUS=$?
  python3 experiment/inspect_prone_pose.py "$GO1_POSE_LOG" \
    > "$GO1_POSE_REVIEW/prone-pose.txt" 2>&1
  GO1_POSE_STATUS=$?
  cat "$GO1_POSE_REVIEW/source.sha256"
  cat "$GO1_POSE_REVIEW/preflight.txt"
  cat "$GO1_POSE_REVIEW/prone-pose.txt"
  printf 'summary_exit=%s pose_exit=%s\nReports: %s\n' \
    "$GO1_SUMMARY_STATUS" "$GO1_POSE_STATUS" "$GO1_POSE_REVIEW"
else
  echo 'STOP: no selected CSV; complete Step 2 first'
fi
```

The inspector selects the first continuous quiet window lasting at least two
seconds, with at least 900 fresh low-level samples, host/tick gaps no greater
than 20 ms, no abort/watchdog, all joint speeds at most 0.05 rad/s, and joint
position/roll/pitch ranges at most 0.03 rad. These are offline screening
criteria, not physical stability or contact acceptance criteria. A `[STOP]`
means the data needs review; it does not automatically invalidate the earlier
communication preflight. Both exit codes should be `0` for a completed report.

The table must contain FR/FL/RR/RL joints 0/1/2 (hip/thigh/calf), in radians.
`OUTSIDE_DO_NOT_REPLAY` explicitly flags measured values outside the current
command bounds. Even `WITHIN` does not approve a command target. The script
cannot distinguish a synthetic CSV from hardware data, so use the exact
archived preflight file and retain its hash and provenance.

**4. Send the result and the physical observation.** Send all Step 3 terminal
output (including the table, exit codes, and report path). Also state whether,
during the original preflight, the belly/body was visibly supported by the
floor or a support pad, whether anyone held the robot, and whether it moved.
If you cannot recall, say so; do not infer contact from joint angles or foot
forces. No new measurement is requested in this step.

**Stop here for review.** Retain the original CSV and the report directory;
there are no new Pi files to delete. The next increment will use these results
to resolve the final-pose/support evidence, then test the real support input
and standing takeover before writing the first hardware handover procedure.
The existing ground hardware lock remains in effect.

### 2.1.3 Continuous support confirmation — completed

**Accepted evidence from 2.1.2.** The reported Ubuntu extraction completed with
`summary_exit=0 pose_exit=0`, selecting 1,001 fresh samples over approximately
two seconds. The original file is
`~/Yuxuan/Robotic-Dog-Tracking-Interface/logs/downloaded/remote_preflight_fix_02.csv`;
the report directory is `logs/prone-review-idhuW79a/`. Reported SHA-256:

```text
b59dd4240125e147056c98399c6f8307a5599420bcb29a535600e438a9fec527
```

| Joint | FR median | FL median | RR median | RL median |
| --- | ---: | ---: | ---: | ---: |
| Hip (rad) | -0.314886 | 0.279764 | -0.296538 | 0.303199 |
| Thigh (rad) | 1.289278 | 1.304901 | 1.304114 | 1.274745 |
| Calf (rad) | -2.794414 | -2.797079 | -2.799137 | -2.767608 |

The operator reports entering the factory prone posture with L2+A, then
switching to damping with L2+B. There was no visible additional movement,
the trunk remained approximately level, and nobody touched the robot. The
operator believes the body rested directly on the floor; independent contact
measurement was not performed. This supports use as a natural prone reference,
not proof of contact at our different simulation target. All four measured
calves are below the command lower bound of -2.721 rad. Do not replay the
measurements, clamp them into a target, or widen the bound to match them.

**What changes now.** Previously, one true support sample after one second of
quiet joints could authorize normal damping. The development controller now
requires support to remain true while feedback remains stable for a full
second. Withdrawal restarts the dwell even if no new robot packet arrives in
that cycle; joint motion also restarts it. Intermittent confirmation still
reaches the existing five-second timeout and latched impedance hold. Early
confirmation during descent does not carry into verification.

This tests the receiving control logic using injected support observations.
It does not implement a physical contact sensor or an operator input device.
The actual input, target calibration, and takeover review remain subsequent
work. L2+B retains emergency-stop semantics; it is not a normal support-confirmation
button. No new hardware run or repeat of 2.1.2 is requested.

**1. Synchronize on Ubuntu.** After committing and pushing the development
changes through the existing GitHub workflow, use one local Ubuntu terminal:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git status --short
git pull --ff-only
```

Stop if the pull fails. Do not discard local changes. Keep Go1 powered off;
there is no SSH or Pi deployment in this test.

**2. Build the offline control-core test.** Continue in the same terminal:

```bash
cmake -S . -B /tmp/go1-support-build -DBUILD_TESTING=ON -DBUILD_SDK_EXAMPLES=OFF
cmake --build /tmp/go1-support-build --target go1_ground_exit_test -j2
```

Both commands must finish successfully before Step 3. The test target does
not link the Unitree SDK or open UDP. The existing Pi binary is unchanged.

**3. Run the new support-confirmation test and save its output.**

```bash
mkdir -p logs
GO1_SUPPORT_REVIEW=$(mktemp -d "$PWD/logs/support-review-XXXXXXXX")
(cd /tmp/go1-support-build && ctest -R '^go1_ground_support_confirmation$' -V) \
  > "$GO1_SUPPORT_REVIEW/support-test.txt" 2>&1
GO1_SUPPORT_STATUS=$?
cat "$GO1_SUPPORT_REVIEW/support-test.txt"
printf 'test_exit=%s\nReports: %s\n' "$GO1_SUPPORT_STATUS" "$GO1_SUPPORT_REVIEW"
```

Required output includes these five lines:

```text
[PASS] brief confirmation after quiet hold cannot exit
[PASS] withdrawal between feedback packets restarts dwell
[PASS] motion during confirmation restarts dwell
[PASS] continuous support and quiet feedback authorize damping
[PASS] intermittent confirmation times out into hold
```

CTest must report `100% tests passed, 0 tests failed out of 1`, with
`test_exit=0`. `No tests were found` is not a pass, even if the exit code is
zero. Any missing pass line, build failure, or test failure means stop and
retain the output. No CSV is generated by this fixture.

**4. Send the complete Step 3 output.** Stop here for review. Keep the report
folder and prior hardware archives; there is nothing to remove from the Pi.
Passing this step verifies the continuous-confirmation gate only. It does not
approve the synthetic target or unlock ground hardware modes.

### 2.1.4 Seeded entry continuity and feedback faults — completed

Purpose: check the low-level controller's entry logic after a standing seed
has been supplied. This is the next bounded software check after the passing
support-confirmation test. It does not request another prone preflight.

The entry code previously kept issuing the seeded pose during PRECHECK even
when low-level feedback was absent, up to the general five-second precheck
timeout. It also replaced that command target with a later measured pose at
CAPTURE_POSE, allowing a position-command step. The revised code:

- Rejects nonfinite or out-of-command-bounds seeds before using them.
- Checks fresh entry feedback against the seed: at most 0.05 rad position
  difference and 0.05 rad/s joint speed, valid low-level state and joint limits,
  valid motor temperatures, finite IMU, and absolute roll/pitch at most 0.5 rad.
- Latches damping on invalid entry feedback, duplicate fresh ticks, send
  failure, or watchdog. Missing/stale feedback beyond 20 ms from the first
  seeded entry control step also latches damping, instead of waiting five seconds.
- Requires fresh feedback to finish CAPTURE_POSE and retains the already
  commanded seed through PRECHECK, CAPTURE_POSE, and HOLD. An accepted small
  measured offset does not create a new position target.

These are software screening limits, not measured robot stability limits.
The first command still comes from the high-level seed before low-level
feedback is available. This test cannot prove that initial command matches the
physical robot, that high-level mode 0 reaches standing, or that support stays
continuous during switching. The high-level capture/transport and actual mode
transition remain unvalidated. Ground hardware modes remain locked; the fault
response in this synthetic test is not an approved way to lower a standing robot.

**1. Synchronize in one Ubuntu terminal.** After committing and pushing the
new development changes through the existing GitHub workflow, run locally:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git status --short
git pull --ff-only
ls test/go1_ground_entry_test.cpp
```

Stop on a failed pull or missing file; do not discard local changes. Go1 can
remain powered off. Do not SSH into the Pi or replace its current binary.

**2. Build only the new offline test target.** Continue in the same terminal:

```bash
cmake -S . -B /tmp/go1-entry-build -DBUILD_TESTING=ON -DBUILD_SDK_EXAMPLES=OFF
cmake --build /tmp/go1-entry-build --target go1_ground_entry_test -j2
```

Both commands must succeed. This test compiles the real control core with
synthetic feedback, without linking the Unitree SDK or opening UDP sockets.

**3. Run the entry test and retain its output.**

```bash
mkdir -p logs
GO1_ENTRY_REVIEW=$(mktemp -d "$PWD/logs/entry-review-XXXXXXXX")
(cd /tmp/go1-entry-build && ctest -R '^go1_ground_entry_core$' -V) \
  > "$GO1_ENTRY_REVIEW/entry-test.txt" 2>&1
GO1_ENTRY_STATUS=$?
cat "$GO1_ENTRY_REVIEW/entry-test.txt"
printf 'test_exit=%s\nReports: %s\n' "$GO1_ENTRY_STATUS" "$GO1_ENTRY_REVIEW"
```

Expected checks:

```text
[PASS] seed target remains continuous through fresh capture and hold
[PASS] mismatched, moving, invalid and duplicate feedback reject entry
[PASS] feedback loss, send failure, watchdog and remote stop latch damping
[PASS] nonfinite and out-of-bounds seed poses rejected
Ground entry core tests passed (synthetic feedback only).
```

CTest must finish with `100% tests passed, 0 tests failed out of 1` and
`test_exit=0`. `No tests were found` is not a pass. Missing pass lines or any
build/test error means stop and retain the complete output.

**4. Send the complete Step 3 output, then stop for review.** No CSV is
produced and no Pi cleanup is needed. Preserve the report directory. Actual
support-confirmation input, a calibrated supported endpoint, and validation of
the high-level capture and mode-switch procedure remain necessary before the
first hardware handover. Do not remove `--dry-run` from other commands or
try the locked standing modes after this software test.

### 2.1.5 High-level standing-capture accumulator — completed

Purpose: test the capture logic that produces the standing seed used by
2.1.4. Previously this logic existed only in the SDK hardware path, accepted
joint speeds up to 1.0 rad/s, and had no independent test against repeated
reads of the same receive buffer. The real hardware path now calls the same
SDK-independent accumulator exercised here.

The accumulator requires at least 100 new receive events spanning at least
200 ms, with gaps at most 20 ms, joint speeds at most 0.05 rad/s, and joint
positions within 0.02 rad of the window's first sample. It also checks high-level
flag, finite joint/IMU values, command bounds, temperature, and absolute
roll/pitch at most 0.5 rad. Invalid feedback, excessive motion, counter rollback,
or a gap resets the window. A 500 Hz stream needs 101 samples to span 200 ms.
Only a completed window writes the averaged seed; a partial window leaves the
output unchanged. The surrounding hardware attempt retains its three-second
deadline.

In the hardware adapter, a usable event requires a successful SDK receive,
an increased `udpState.RecvCount`, and no increase in flag/CRC errors during
that call. A send failure or an observed L2+B chord aborts capture before the
low-level loops start. This test exercises the accumulator with synthetic
transport counters; it does not exercise those SDK transport/remote branches.

**Limit of this step:** the SDK HighState structure has no low-level-style
source `tick`. A changed receive counter distinguishes a new receive event
from a cached buffer read; it does not prove that the upstream controller
updated the underlying measurement. That transport behavior still needs
verification on the robot. The existing mode-0 request remains unchanged and
must not be treated as a verified stand-up command. Neither quiet joint data
nor this passing test proves load-bearing standing or continuous support
through the high-level/low-level switch. Ground hardware modes stay locked.

**1. Synchronize on Ubuntu.** After the development changes have been committed
and pushed through the existing GitHub workflow, run in one local terminal:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git status --short
git pull --ff-only
ls test/go1_standing_capture_test.cpp
```

Stop on a failed pull or missing file; do not discard local changes. Keep Go1
powered off. There is no SSH, robot connection, or Pi deployment in this step.

**2. Build the new offline test target.** Continue in the same terminal:

```bash
cmake -S . -B /tmp/go1-capture-build -DBUILD_TESTING=ON -DBUILD_SDK_EXAMPLES=OFF
cmake --build /tmp/go1-capture-build --target go1_standing_capture_test -j2
```

Both commands must succeed before continuing. This target does not link the
SDK or open UDP. It executes the accumulator used by the hardware code.

**3. Run once and save the output.**

```bash
mkdir -p logs
GO1_CAPTURE_REVIEW=$(mktemp -d "$PWD/logs/capture-review-XXXXXXXX")
(cd /tmp/go1-capture-build && ctest -R '^go1_standing_capture$' -V) \
  > "$GO1_CAPTURE_REVIEW/capture-test.txt" 2>&1
GO1_CAPTURE_STATUS=$?
cat "$GO1_CAPTURE_REVIEW/capture-test.txt"
printf 'test_exit=%s\nReports: %s\n' "$GO1_CAPTURE_STATUS" "$GO1_CAPTURE_REVIEW"
```

Required output:

```text
[PASS] quiet capture requires sample count and elapsed time
[PASS] cached data and unsuccessful receives cannot complete capture
[PASS] movement, invalid feedback, gaps and counter rollback reset capture
[PASS] elapsed time alone cannot bypass minimum sample count
Standing capture tests passed (synthetic transport and feedback only).
```

CTest must report `100% tests passed, 0 tests failed out of 1` with
`test_exit=0`. A missing test, missing pass line, or build error means stop and
send the complete output. `No tests were found` is not a passing result.

**4. Send the complete Step 3 output and stop for review.** No CSV or Pi file
is created. Retain the report directory; no deletion is required. After this
software check, the remaining work includes the actual support-confirmation
input, supported endpoint calibration, and a reviewed hardware observation
procedure for the transport/mode transition. Passing the accumulator test
alone does not authorize a hardware handover.

### 2.1.6 Identify the physical support setup — closed: none available

Purpose: determine the available physical arrangement before writing endpoint
calibration and the first live takeover procedure. The completed software
checks do not establish physical support. The observed factory prone calves
(-2.768 to -2.799 rad) exceed our command range; the simulated -2.70 rad target
has never been shown to place the body on the floor. A level trunk and a quiet
pose alone cannot resolve that difference.

This step is an equipment inventory, not a powered robot experiment or an
approval of a particular rig. There is no new test executable, deployment,
SSH command, or robot command to run. Do not repeat 2.1.1–2.1.5.

**1. Leave the robot powered off in its existing floor-supported prone pose.**
Do not stand it up, lift it, put equipment underneath it, or try to force its
joints to the simulated target for this check. The task is to identify available
equipment before specifying how it should be used.

**2. Inspect the equipment already available in the lab.** Determine whether
there is a robot support stand, adjustable body support, or rated overhead
support/harness intended to carry this robot. If available, record its model
or description, rated capacity if known, adjustment range, and intended body
contact/attachment points. Unknown details can be marked unknown. A loose
stack of objects or a person holding the robot is not an established setup.
Do not purchase, assemble, or load-test a rig as part of this step.

**3. Record observation conditions.** State whether an operator can see the
body/support contact from beside the robot without reaching between the legs,
and whether a second operator is available for a future test. These details
will inform the support-confirmation input and stop procedure; they are not
assumed to be satisfied by the earlier hands-off preflight.

**4. Send this short report.** Copy and complete the following text; there is
no terminal command to execute:

```text
Available support equipment: none / description
Model and rated capacity: known values / unknown / not applicable
Height adjustment and attachment/contact points: description / unknown
Can body/support contact be seen from beside the robot?: yes / no / unknown
Second operator available for a future test?: yes / no
```

**Result and revised decision:** the operator reports no support equipment and
accepts some experimental risk while seeking to avoid mechanical damage.
Continue with 2.1.7 rather than requiring equipment purchase. This acceptance
cannot establish a guarantee against damage, nor does it make the old
standing-transition implementation ready for use.

### 2.1.7 No-lift route: passive observation of factory traffic — baseline completed

#### Research conclusion and selected route

A no-lift development route is technically plausible. Use a **prone start,
small four-foot body rise, return close to the floor, and gradual support-torque
release**, all within one low-level controller. Increase height only after the
previous rise/return has been reviewed. This is the selected engineering
proposal, not a validated hardware trajectory. Do not start by switching
controllers while the robot is already standing.

The source review and the proposed sequence are detailed in
[GO1_NO_LIFT_PLAN.md](GO1_NO_LIFT_PLAN.md). Factory high-level modes provide
an alternative for initial motion observation, but do not provide custom joint
torque tracking. We will first observe this robot's own factory trajectory;
then implement the low-height entry/return using that evidence. Existing
synthetic tests remain useful regression tests and need not be repeated.

The next observation is deliberately passive: check whether the Pi can see
factory UDP packets while the robot remains prone. Packet presence/layout is
not yet known. A positive result enables a subsequent single factory
stand-up/lie-down recording. A negative result calls for a different recording
interface, not extra motor commands to provoke traffic. Do not perform that
standing cycle during this initial 15-second capture.

**1. Prepare Ubuntu and the robot.** Read this entire section before starting.
No new robot executable needs compiling or deploying. Use the normal factory
controller; the robot must already be prone on a flat, nonslip floor, in the
factory damping state reached by the familiar L2+A then L2+B procedure. Use
normal startup if powering on is necessary; startup can itself cause motion.
Do not enter developer/basic mode with L1+L2+Start. Do not run our low-level
experiment, stop `Legged_sport`, or stop the Programming Module. No one should
lift the robot. Keep the space around its legs clear.

On **Ubuntu**, synchronize the documentation through the existing GitHub
workflow if needed, then connect to Go1 Wi-Fi and open a Pi terminal:

```bash
ssh pi@192.168.12.1
```

**2. On the Pi, check the capture tools and existing processes.**

```bash
command -v tcpdump
command -v timeout
pgrep -af 'Legged_sport|programming[.]py|go1_lowlevel_experiment|example_position|example_torque|example_velocity|example_walk'
sudo ss -Huanp
```

Keep this output. `timeout` must print a path. If the ordinary-user lookup
for `tcpdump` prints nothing, use Step 2a below before proceeding; it may be
installed outside the user's PATH. Do not run an SDK example in its place. If one of our controllers or an SDK motion example
is active, do not start this capture or kill the process blindly; report its
state first. The factory processes should remain running. Another user program
sending robot commands also needs to be identified before continuing.

**2a. Resolve the missing tcpdump tool — Pi terminal.** The reported check
found `/usr/bin/timeout` but no ordinary-user `tcpdump` path. The reported
socket list shows `Legged_sport` using local 8008 to MCU 8007, which is covered
by the capture filter; a socket listing alone does not prove packets are flowing.
No process needs stopping for the tool check or passive recording.

First check root's PATH, then install only the capture utility if necessary:

```bash
if sudo sh -c 'command -v tcpdump >/dev/null 2>&1'; then
  sudo tcpdump --version
else
  sudo apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=15 \
    -o Acquire::https::Timeout=15 update && \
  sudo apt-get install --no-install-recommends --no-remove -y tcpdump && \
  sudo tcpdump --version
fi
```

Installation requires access to the Pi's configured package repositories. If
DNS, repository access, signatures, or package installation fails, send the
complete error; do not change network interfaces or package sources while
following this step. We can prepare a matching offline package if necessary.
This is not an instruction to upgrade the OS or reboot the robot. A printed
tcpdump version means the utility is available through `sudo`, as used in Step 3.

The output also showed `ros2udp_motion_` connected to high-level port 8082.
Identify it by its current full command and executable path, without stopping it:

```bash
pgrep -af '[r]os2udp_motion_'
for GO1_PROC_PID in $(pgrep -f '[r]os2udp_motion_'); do
  ps -ww -p "$GO1_PROC_PID" -o pid=,ppid=,args=
  sudo readlink -f "/proc/$GO1_PROC_PID/exe"
done
```

Send this output with the tcpdump version or installation error. Keep the
robot prone and do not request factory movement while an additional command
publisher's origin is unresolved. Do not reuse the PID from the earlier output.

**2b. Offline install after the reported DNS failure — completed.**
The operator confirmed that direct `dpkg -i` installation succeeded:
`tcpdump 4.9.3`, `libpcap 1.8.1`, and `OpenSSL 1.1.1d` were printed.
The package SHA-256 matched the Pi APT cache. The previous local-file
`apt-get --no-download` attempt failed; it did not complete installation.
Do not repeat installation or change sources. Continue to Step 3 below.
The remaining commands in Step 2b are retained as installation reference.

The Pi could not resolve either the Tsinghua mirror or `packages.ros.org`.
This is a name-resolution/connectivity failure, not evidence that replacing
one mirror will fix it. APT reused old indexes and selected only
`tcpdump:arm64 4.9.3-1~deb10u2`; the package download failed, so installation
has not completed. Do not repeat the failed online installation or run the
suggested `autoremove` (it lists development libraries used by this system).

The matching package is present in the
[official Debian archive](https://archive.debian.org/debian/pool/main/t/tcpdump/).
Download on Ubuntu, then transfer over the existing robot connection. This
avoids changing the Pi's sources, DNS or network interfaces.

On **Ubuntu with working Internet access**, in one terminal:

```bash
mkdir -p ~/Downloads/go1-offline-tcpdump
cd ~/Downloads/go1-offline-tcpdump
curl --fail --location --connect-timeout 15 --max-time 120 \
  -o tcpdump_4.9.3-1~deb10u2_arm64.deb \
  'https://archive.debian.org/debian/pool/main/t/tcpdump/tcpdump_4.9.3-1~deb10u2_arm64.deb' && \
  dpkg-deb -f tcpdump_4.9.3-1~deb10u2_arm64.deb Package Version Architecture
```

Require a successful download and metadata `tcpdump`, `4.9.3-1~deb10u2`,
`arm64`. Do not install this ARM64 package on Ubuntu. After reconnecting Ubuntu
to Go1 Wi-Fi, use the same terminal:

```bash
scp ~/Downloads/go1-offline-tcpdump/tcpdump_4.9.3-1~deb10u2_arm64.deb pi@192.168.12.1:~/
```

On **Pi**, verify the package against the SHA-256 in its existing APT index
before installing it without downloads:

```bash
cd ~
GO1_TCPDUMP_SHA=$(apt-cache show 'tcpdump=4.9.3-1~deb10u2' | awk '/^SHA256: / {print $2; exit}')
if [ "$(dpkg --print-architecture)" != arm64 ]; then
  echo 'STOP: unexpected Pi architecture'
elif [[ "$GO1_TCPDUMP_SHA" =~ ^[0-9a-fA-F]{64}$ ]]; then
  printf '%s  %s\n' "$GO1_TCPDUMP_SHA" 'tcpdump_4.9.3-1~deb10u2_arm64.deb' | sha256sum -c - && \
  sudo dpkg -i ./tcpdump_4.9.3-1~deb10u2_arm64.deb && \
  sudo tcpdump --version
else
  echo 'STOP: package checksum unavailable in the Pi APT cache; send output'
fi
```

Require checksum `OK` and a printed tcpdump version. If verification or offline
installation fails, retain the error; do not force dependencies or substitute a
newer distribution's package. Once successful, continue directly to Step 3;
there is no reboot or additional software rehearsal. Keep the robot prone.

The identified additional process resolves to
`/home/pi/Unitree/autostart/utrack/catkin_utrack/devel/lib/a2_ros2udp_adv/ros2udp_motion_mode_adv`.
Its location is consistent with the installed Unitree utrack component; it
must remain running for this passive capture. That path alone does not prove
its current command contents, so keep tracking/walking functions inactive and
make no movement requests during this capture.

**3. Capture once for 15 seconds while the robot remains prone.** In that same
**Pi terminal**:

```bash
mkdir -p ~/Robotic-Dog-Tracking-Interface/logs
GO1_NATIVE_DIR=$(mktemp -d "$HOME/Robotic-Dog-Tracking-Interface/logs/native-observe-XXXXXXXX")
sudo timeout --signal=INT --kill-after=3s 15s \
  tcpdump -Z "$(id -un)" -p -n -i any -s 0 -B 4096 -U -c 15000 \
  -w "$GO1_NATIVE_DIR/native-prone.pcap" \
  'udp and ((host 192.168.123.10 and port 8007) or (host 192.168.123.161 and port 8082))' \
  2> "$GO1_NATIVE_DIR/capture.txt"
GO1_NATIVE_STATUS=$?
cat "$GO1_NATIVE_DIR/capture.txt"
printf 'capture_exit=%s\nPi archive: %s\n' "$GO1_NATIVE_STATUS" "$GO1_NATIVE_DIR"
```

Do not press mode-change buttons during the capture. `tcpdump` listens through
a packet-capture socket: it does not bind the SDK's UDP source port or publish
HighCmd/LowCmd. It does add capture/disk load, so this check is short and bounded
by both time and packet count. Exit `124` is expected when `timeout` ends the
15-second capture; `0` is expected if the packet limit is reached first. Other
exit codes, permission errors, or missing files require review. Captured packet
counts and kernel drops are printed in `capture.txt`; zero packets is a useful
negative observation, not permission to send a motor command.

**4. Inspect a short packet summary and make the archive readable.** Still on
**Pi**, after the capture command has ended:

```bash
sudo chown "$(id -u):$(id -g)" "$GO1_NATIVE_DIR/native-prone.pcap"
sudo tcpdump -nn -tt -r "$GO1_NATIVE_DIR/native-prone.pcap" -c 20 \
  > "$GO1_NATIVE_DIR/packet-summary.txt" 2>&1
cat "$GO1_NATIVE_DIR/packet-summary.txt"
sha256sum "$GO1_NATIVE_DIR/native-prone.pcap"
printf 'Copy this Pi archive path: %s\n' "$GO1_NATIVE_DIR"
```

This is offline reading of the capture, not another robot run. Preserve errors
if the file could not be created. Addresses and packet lengths show which
links are observable; they do not yet establish a decoded joint trajectory.
Capturing on `any` can also include duplicate observations of a packet; packet
counts must not be treated as unique feedback frequency.

**5. Copy the archive to Ubuntu.** In a separate **Ubuntu terminal**, run these
commands and paste the exact Pi archive path printed in Step 4 when prompted:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
mkdir -p logs/factory-observation
GO1_NATIVE_LOCAL=$(mktemp -d "$PWD/logs/factory-observation/prone-XXXXXXXX")
read -r -p 'Paste the complete Pi archive path from Step 4: ' GO1_NATIVE_REMOTE
if [[ "$GO1_NATIVE_REMOTE" =~ ^/home/pi/Robotic-Dog-Tracking-Interface/logs/native-observe-[A-Za-z0-9]+$ ]]; then
  scp -r "pi@192.168.12.1:$GO1_NATIVE_REMOTE/." "$GO1_NATIVE_LOCAL/" && \
  sha256sum "$GO1_NATIVE_LOCAL/native-prone.pcap"
  printf 'Ubuntu archive: %s\n' "$GO1_NATIVE_LOCAL"
else
  echo 'STOP: unexpected archive path; retain the Pi output for review'
fi
```

The SHA-256 on Ubuntu must match Step 4. Keep the Pi copy until the archive is
verified; no deletion is required now. No factory module was stopped, so no
module-restoration command is needed. If ending the session, shut down only
with the robot still fully prone using the familiar normal battery procedure.

**6. Send the result.** Send Steps 2–4 terminal output and the Ubuntu checksum
and archive path. The PCAP will be needed for offline decoding; retain it and
attach it if practical. After reviewing which packets are available, the next
instruction will record one factory rise/return cycle or use an alternative
observation interface. No rig purchase or repeated synthetic rehearsal is
required to proceed with this route.

### 2.1.8 Factory return to prone and damping — completed

**Received capture; do not repeat it.** The supplied `return-CxCtzXzf` archive
contains a valid 39.86-second recording with PCAP SHA-256
`91271ff4c5bb79d85082a5c83d4d93e095157c21766c4fdd89b79d4e46991b8e`.
All 39,859 state packets and 19,929 command packets passed CRC checks, and
tcpdump reported zero kernel drops. Independent re-decoding reproduced the
supplied summary exactly.

The packets show quiet standing until approximately 9.79 seconds, a factory
lie-down command through 12.83 seconds, and a switch at 15.17 seconds from
`Kp=50, Kd=3` position control to `Kp=0, Kd=2` damping. Damping caused
additional settling: peak post-switch joint speed was 1.845 rad/s, with the
largest motion concentrated in the next 0.92 seconds. The final feedback is
very quiet, but three calf positions are approximately 0.094–0.100 rad below
the existing -2.70 rad development target and below the SDK command limit.
Do not copy these feedback values into a command target or widen that limit.
See [GO1_NATIVE_RETURN_ANALYSIS.md](GO1_NATIVE_RETURN_ANALYSIS.md).

The archive's `observation.txt` contains the commands following its interactive
prompt rather than a physical observation, but the operator subsequently
supplied the missing facts from memory:

```text
Initial posture: Standing; L2+A used: yes; trunk fully on floor before L2+B: yes; motion after damping: nothing but lie flat on the ground; Sound: None
```

Together with the valid telemetry, this completes 2.1.8. No abnormal motion
beyond settling flat and no sound were reported. Slip and impact were not
listed as separate events. Do not repeat the motion merely to repair the text
file. To repair the accepted Ubuntu archive record without touching the PCAP,
run this optional command on Ubuntu:

```bash
printf '%s\n' 'Initial posture: Standing; L2+A used: yes; trunk fully on floor before L2+B: yes; motion after damping: nothing but lie flat on the ground; Sound: None' \
  > /home/aims/Yuxuan/Robotic-Dog-Tracking-Interface/logs/factory-observation/return-CxCtzXzf/observation.txt
```

The procedure below is retained as reference.

**Reference procedure:** make one passive recording using the steps below, then
download and decode it on Ubuntu. The factory remote controls the robot;
our program does not send motor commands during this observation. Do not
reuse the old `remote-preflight` or `ground-handover` command.

**Before Step 1, prepare the two terminals.** On Ubuntu, from the repository:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git pull --ff-only
ls experiment/decode_native_go1_pcap.py
```

If synchronization fails or the decoder is missing, resolve that before
recording. Connect Ubuntu to the Go1 network. In **terminal 1**, log into Pi:

```bash
ssh pi@192.168.12.1
```

Run Steps 2 and 4 in this Pi session. Keep **terminal 2 on Ubuntu** for Step 5.
Step 3 is performed with the original remote while terminal 1 is capturing.
Read the posture branch in Step 1 before starting the capture. Do not stop
the Programming Module or free port 8090 for this procedure: passive capture
does not claim that SDK port. If a custom controller is still running, do not
begin the factory observation until its state and exit have been resolved.

**Accepted baseline:** the file called `native-prone.pcap` was actually recorded
while standing. Its SHA-256 is
`a1d5d005c33a2aae89e36d5beb906735112cda483e4c1d109cd033f966212ff3`.
Offline decoding found 10,000 state packets and 5,000 command packets over
approximately 10 seconds; every packet passed its SDK CRC check. The 77 reported
kernel drops limit timing conclusions but do not invalidate the standing
baseline. See [GO1_NATIVE_STANDING_ANALYSIS.md](GO1_NATIVE_STANDING_ANALYSIS.md).
Do not repeat standing capture or previous synthetic tests.

**1. Prepare one factory-only observation.** Keep all factory services running.
Do not start our low-level executable or enter developer mode. Use a flat,
nonslip floor with space for the normal leg motion. Read Steps 2–3 first.
In the Pi terminal, run this read-only guard before starting the capture:

```bash
if pgrep -af 'go1_lowlevel_experiment|example_(position|torque|velocity|walk)'; then
  echo 'STOP: a custom low-level controller is still running; send this output'
else
  echo 'PASS: no known custom low-level controller is running'
fi
```

Proceed only when the `PASS` line is printed. `programming.py`, `Legged_sport`,
and the other factory services are intentionally not included in this check;
do not stop them for the passive observation.

If the robot is already standing under its normal factory controller, record
one familiar **L2+A lie-down**, followed by **L2+B damping only after the robot
has finished lying down**. No lifting is required.

If it is already prone, leave it prone: record that state and the familiar
L2+B damping transition, and report that no standing-to-prone motion occurred.
Do not stand it up solely to satisfy this recording. If it is already damping,
record it without pressing any buttons. This still supplies the missing prone
endpoint observation.

**2. Start a 40-second passive capture in the Pi terminal.** This version has
no packet-count cap, so it will not end after about 10 seconds. It captures
only the two already identified native flows. No new binary is deployed.

```bash
mkdir -p ~/Robotic-Dog-Tracking-Interface/logs
GO1_RETURN_DIR=$(mktemp -d "$HOME/Robotic-Dog-Tracking-Interface/logs/native-return-XXXXXXXX")
sudo -v
printf 'Capture starting: follow Step 3 now; recording lasts 40 seconds.\n'
sudo timeout --signal=INT --kill-after=3s 40s \
  tcpdump -Z "$(id -un)" -p -n -i any -s 0 -B 8192 -U \
  -w "$GO1_RETURN_DIR/native-return.pcap" \
  'udp and ((src host 192.168.123.10 and src port 8007 and dst host 192.168.123.161 and dst port 8008) or (src host 192.168.123.161 and src port 8008 and dst host 192.168.123.10 and dst port 8007))' \
  2> "$GO1_RETURN_DIR/capture.txt"
GO1_RETURN_STATUS=$?
cat "$GO1_RETURN_DIR/capture.txt"
printf 'capture_exit=%s\nPi archive: %s\n' "$GO1_RETURN_STATUS" "$GO1_RETURN_DIR"
```

**3. During that capture, use the normal remote.** Leave the initial state
unchanged for approximately five seconds. If initially standing, press the
familiar L2+A combination once and wait until the lie-down has finished and
the body is resting on the floor. Observe for another five seconds, then press
L2+B once. Leave the robot untouched for the remaining recording time.
Do not press L2+B on a timer while the body is still descending. If lie-down
does not complete normally, do not advance to the planned damping step;
report what occurred. Do not force another cycle if the capture ends early.
For an initially prone robot, follow the prone branch in Step 1 instead.

Exit `124` is expected when the 40-second timeout ends capture. Keep the capture
output, including drops or errors. The timeout stops tcpdump only; it does not
change robot mode. Record whether the trunk reached the floor, whether damping
caused additional motion, and any slip, impact, or unusual sound.

**4. Save context and checksum on Pi.** After capture ends, first run only:

```bash
sudo chown "$(id -u):$(id -g)" "$GO1_RETURN_DIR/native-return.pcap"
```

Then paste this single command by itself. When its prompt appears, type the
physical observation and press Enter. Do not paste later shell commands as the
answer:

```bash
read -r -p 'Initial posture; buttons used; final posture; motion after damping; slip/impact/sound: ' GO1_RETURN_NOTE && printf '%s\n' "$GO1_RETURN_NOTE" > "$GO1_RETURN_DIR/observation.txt"
```

After the normal shell prompt returns, run:

```bash
cat "$GO1_RETURN_DIR/observation.txt"
sha256sum "$GO1_RETURN_DIR/native-return.pcap"
printf 'Copy this Pi archive path: %s\n' "$GO1_RETURN_DIR"
```

**5. Copy and decode on Ubuntu.** Ensure the updated repository contains
`experiment/decode_native_go1_pcap.py`. In the Ubuntu terminal:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
mkdir -p logs/factory-observation
GO1_RETURN_LOCAL=$(mktemp -d "$PWD/logs/factory-observation/return-XXXXXXXX")
read -r -p 'Paste the complete Pi archive path from Step 4: ' GO1_RETURN_REMOTE
if [[ "$GO1_RETURN_REMOTE" =~ ^/home/pi/Robotic-Dog-Tracking-Interface/logs/native-return-[A-Za-z0-9]+$ ]]; then
  scp -r "pi@192.168.12.1:$GO1_RETURN_REMOTE/." "$GO1_RETURN_LOCAL/" && \
  sha256sum "$GO1_RETURN_LOCAL/native-return.pcap"
else
  echo 'STOP: unexpected Pi archive path; correct it before continuing'
fi
```

Compare this hash with Step 4. Only when they match, run:

```bash
python3 experiment/decode_native_go1_pcap.py \
  "$GO1_RETURN_LOCAL/native-return.pcap" \
  --out "$GO1_RETURN_LOCAL/decoded"
GO1_RETURN_DECODE_STATUS=$?
printf 'decode_exit=%s\nUbuntu archive: %s\n' \
  "$GO1_RETURN_DECODE_STATUS" "$GO1_RETURN_LOCAL"
cat "$GO1_RETURN_LOCAL/observation.txt"
```

Send the capture output, decoding output, and observation text; attach the new
PCAP for trajectory analysis. A decoder error is evidence to inspect offline,
not a reason to repeat robot motion. The decoder intentionally rejects other
wire layouts and does not overwrite an existing output directory. No custom
low-level movement is authorized by passing the CRC check; the next controller
change depends on reviewing the endpoint and transition data.

Keep both Pi and Ubuntu copies of this new observation until decoding and
trajectory review are complete. No cleanup or repeat capture is requested at
this stage. The earlier preflight and software-test archives remain accepted.

### 2.1.9 Prone low-rise software gate — passed on Ubuntu; reference only

**Accepted operator result.** Ubuntu GNU 9.4.0 built both requested targets.
The dedicated CTest run passed all 3/3 tests with `prone_test_exit=0`:
core state-machine checks, a 11,255-sample integration dry-run, and the
expected hardware-lock rejection (`no UDP opened`). The separately archived
dry-run and offline analyzer both exited 0. The report showed 500.00 Hz,
2.000 ms p99/maximum gap, 0.0000 rad roll/pitch excursion, 0.0655 rad/s
maximum joint speed, 30 degrees C maximum temperature, valid remote and
low-level ratios of 1.000, zero duplicate/gapped ticks, and zero watchdog
cycles. The lift/contact support metrics were `nan` as expected for this mode.

Retain both files in the Ubuntu archive; neither needs to be copied to the Pi:

```text
/home/aims/Yuxuan/Robotic-Dog-Tracking-Interface/logs/prone-low-rise-software/review-nwpAsaCt/
prone_low_rise.csv SHA-256: 0a231979a1076ad88a8d8fb17bd8480a384b62d9bc3f47b9357b3e09f075c309
prone_low_rise.csv.summary.csv SHA-256: b89f065f927b38d8ba21d61568dc58b0b434984defce68fdbb3e1e2a33e9184d
```

This acceptance is based on the supplied terminal transcript; the archived
CSV files were not independently downloaded to this development computer.
The commands below are retained for reproducibility and are not a request to
repeat the passed run.

This is an **Ubuntu-only software test**. Leave the Go1 powered off or
disconnected, do not SSH to the Pi, and do not start any factory/developer
mode. Every executable invocation below includes `--dry-run`; the dedicated
hardware-lock test also proves that omitting it is rejected before UDP setup
or the `ARM` prompt.

The new path starts from the passively measured prone fixture and implements:

- one second and at least 500 fresh samples of position-free damping while the
  pose is checked for quietness;
- a four-second smooth engagement from `Kp=0` to at most `Kp=5`, with `Kd=1`,
  zero feed-forward torque, and a per-joint predicted command-effort cap of
  0.50 N m;
- a one-second stable hold, followed by a four-second 5 mm symmetric trunk
  rise generated by inverse kinematics and another one-second stable hold;
- a four-second return to the captured/clamped engagement target;
- one continuous second in which quiet feedback and an independent floor-
  support observation overlap; missing or intermittent support latches an
  impedance hold instead of releasing;
- a four-second smooth `Kp=5` to `Kp=0` release and a final one-second damping
  window.

The observed calf feedback below the SDK command minimum is accepted only as
feedback. Command positions remain within the existing SDK bounds. A single
Ctrl-C during movement returns to the prone target before release; double
Ctrl-C, L2+B, loss of the valid remote stop channel, feedback loss, watchdog
expiry, excessive speed, tracking error, or envelope violation enters damping
immediately. These are software design
properties, not evidence that the motion is safe on the physical robot.

**1. Synchronize and create a clean build on Ubuntu.** From the repository:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git pull --ff-only
git submodule update --init --recursive
conda activate dog_ctrl
GO1_PRONE_BUILD=/tmp/go1-prone-low-rise-build
cmake -S . -B "$GO1_PRONE_BUILD" \
  -DBUILD_TESTING=ON -DBUILD_SDK_EXAMPLES=OFF
```

Stop and send the output if pull, submodule initialization, or configuration
fails. A CMake deprecation warning is not a failure.

**2. Build only the required software-test targets.**

```bash
cmake --build "$GO1_PRONE_BUILD" \
  --target go1_prone_low_rise_test go1_lowlevel_experiment -j2
```

**3. Run the three-part software gate.**

```bash
ctest --test-dir "$GO1_PRONE_BUILD" \
  -R '^go1_prone_low_rise_' -V
GO1_PRONE_TEST_STATUS=$?
printf 'prone_test_exit=%s\n' "$GO1_PRONE_TEST_STATUS"
```

Expected result: 3/3 tests pass and `prone_test_exit=0`. The core test should
print these four checks:

```text
[PASS] bounded nominal sequence and exact 5 mm kinematics
[PASS] independent continuous support interlock
[PASS] cancel, feedback, watchdog, remote and envelope faults
[PASS] hardware CLI rejected before UDP
```

The hardware-lock test is successful when the non-dry invocation fails. CTest
knows that failure is expected; do not run that command manually and do not
try to bypass the lock.

**4. Produce and analyze one separately archived dry-run.**

```bash
mkdir -p logs/prone-low-rise-software
GO1_PRONE_REVIEW=$(mktemp -d \
  "$PWD/logs/prone-low-rise-software/review-XXXXXXXX")
"$GO1_PRONE_BUILD/go1_lowlevel_experiment" \
  --dry-run --mode prone-low-rise \
  --log "$GO1_PRONE_REVIEW/prone_low_rise.csv"
GO1_PRONE_RUN_STATUS=$?
python3 experiment/analyze_lowlevel_log.py \
  "$GO1_PRONE_REVIEW/prone_low_rise.csv" --no-plots
GO1_PRONE_ANALYZE_STATUS=$?
sha256sum "$GO1_PRONE_REVIEW/prone_low_rise.csv" \
  "$GO1_PRONE_REVIEW/prone_low_rise.csv.summary.csv"
printf 'dry_run_exit=%s analyze_exit=%s\nUbuntu archive: %s\n' \
  "$GO1_PRONE_RUN_STATUS" "$GO1_PRONE_ANALYZE_STATUS" "$GO1_PRONE_REVIEW"
```

Expected dry-run indicators include 500 Hz feedback, 2 ms p99/maximum gaps,
zero roll/pitch excursion, about 0.0655 rad/s maximum joint speed, valid remote
and low-level ratios of 1.000, zero duplicate/gapped ticks, and zero watchdog
cycles. `nan` for lift/contact support-margin metrics is expected because this
mode does not use the standing foot-force support polygon. The synthetic plant
and synthetic independent-support flag do not validate floor contact, real
effort, network timing, or physical stability.

**5. Stop after the result.** The reported run passed; do not repeat it solely
to proceed. Do not deploy to the Pi and do not perform a physical low-rise
run. Hardware remains locked until a separate operator-observation mechanism
for continuous floor support is designed and validated.

### 2.1.10 Ubuntu operator-support input — software prototype

**Hold-button prototype evaluated on Ubuntu.** The unit tests passed 4/4 and
`tkinter=available`. With the local receiver listening on `127.0.0.1:18092`,
the operator held the button for more than two seconds and released it, then
held it again and moved the pointer outside the window. The receiver reported
`support=true`, `support=false` for each action. This validates the two
observed release paths on that Ubuntu laptop. The operator then confirmed that
closing the GUI produced the expected disconnect warning. These tests need no
repetition. The operator also pointed out that continuously holding the screen
button while keeping the L2+B remote ready occupies both hands. The continuous
hold interaction is therefore rejected for the one-person test.

The first simulation supplied `floorSupportObserved` independently. The new
hardware-code path now reads a separate loopback-only TCP receiver, but the
`prone-low-rise` hardware CLI remains locked before constructing the SDK/UDP
runner. Thus this code is prepared for an offline transport test, **not** a
hardware-ready path. Joint positions, foot forces, and IMU quietness do not
independently prove that the trunk is on the floor.

The operator confirmed that one person can watch the Go1 and operate the
Ubuntu laptop. The revised independent input is a **single click after visual
confirmation** of trunk contact, issuing a non-extendable 1.5-second pulse of
100-ms-leased heartbeats. It does not need the operator's hand throughout the
release. It is *not* a contact sensor, and it does not replace the factory
remote or its L2+B emergency chord. A click before the controller's return is
complete does not count toward support verification: the C++ state machine
requires an unconfirmed observation within `PRONE_SETTLE` before a new
assertion can begin the one-second dwell.

An offline prototype is in `experiment/operator_support_gate.py`. The Ubuntu
sender connects only to `127.0.0.1`; an SSH local-forward can deliver that
stream to the Pi's new C++ receiver, which also binds only `127.0.0.1`. The
receiver accepts only current pulse heartbeats, rejects
out-of-order/malformed frames, and revokes confirmation on pulse expiry,
disconnect, window focus loss, or a gap longer than 100 ms. Automated tests
exercise both the short pulse and lease rules. The separate C++ probe does not
link the Go1 SDK or send motor commands. The hardware CLI lock remains active.
SSH authenticates the tunnel endpoint, but a localhost TCP stream is not a
physical contact sensor or a hard real-time safety channel; delayed/buffered
transport and latched-hold recovery remain physical-test review items.

**1. On Ubuntu, run the logic tests and check the GUI dependency.** The robot
may remain powered off. Use a graphical Ubuntu desktop session, not an SSH
terminal on the Pi:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git pull --ff-only
conda activate dog_ctrl
python3 -m unittest discover -s test -p test_operator_support_gate.py -v
python3 -c 'import tkinter; print("tkinter=available")'
```

Expected for the revised version: six tests pass and `tkinter=available`. If Tkinter is unavailable,
stop and send that error; do not install packages or improvise an input tool.

**2. In Ubuntu terminal 1, start only the offline receiver probe.**

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 -u experiment/operator_support_gate.py probe
```

It should print `Offline probe listening on 127.0.0.1:18092; no robot access`.
If that port is busy, stop and send the error; do not kill an unknown process.

**3. In a separate Ubuntu terminal 2, open the local GUI.**

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 experiment/operator_support_gate.py sender
```

Click the on-screen button once after an imagined visual contact confirmation.
The probe should show `support=true`, then automatically `support=false` after
about 1.5 seconds without holding the button. Switching away from the GUI
while the pulse is active cancels it. Closing the GUI should print
`Sender disconnected; support=false`. Stop the probe with Ctrl-C. This test
uses no Go1 connection and no robot motion. **The operator is not being asked
to repeat this GUI test now; these steps document the revised prototype for a
later consolidated software check.**

**4. No further manual micro-test is needed in this section.** Continue to
the consolidated transport test in 2.1.11. Do not remove the C++ hardware
lock. Low gain and a predicted
0.50 N m per-joint limit are not proof of zero contact force or physical safety:
the factory capture showed substantial post-damping settling, and the first
custom physical motion still needs a separate controlled acceptance plan.

### 2.1.11 Consolidated operator-input transport test — robot off

**Accepted operator result, 2026-09-20; commands below are reference only.**
The earlier Ubuntu transcript passed six pulse/lease tests, four cleanup tests,
and five CTest cases, ending in `support_cluster=PASS`. The subsequent Pi
transcript showed the listening banner, seven true/false pairs, and
`Probe stopped; support=false`. The sender opened its GUI and the first run
closed with `sender_exit=0`. The operator reported that the test worked.
The second sender run was interrupted with Ctrl-C inside a Tkinter callback;
that traceback followed a manual interruption and does not invalidate the
first completed run. Close the GUI with its window close button for routine
shutdown.

Evidence limits: the supplied Terminal B text duplicates Terminal A, so it
is not an independent tunnel transcript. The probe output has no timestamps
or click count; it establishes received transitions, not a measured 1.5-second
pulse duration or one pair per click. Acceptance here combines the operator's
reported success with the existing timing tests. No repeat is requested and
no physical contact or motor behavior is inferred.

This entire section is **software and networking only**. The Go1 must be
powered off, with no active custom motor controller. Connecting the Ubuntu
laptop to the Pi over SSH for this test is allowed; do not start
`go1_lowlevel_experiment`, enter developer mode, or press any factory motion
buttons. The Pi program below is `go1_operator_support_probe`, which does not
link the Unitree SDK or open robot UDP sockets.

**1. On Ubuntu, synchronize and run the entire robot-off software cluster.**
Check status first:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git status --short
```

If it shows anything, preserve those changes and stop before pulling.
Otherwise run the cluster in the same Ubuntu terminal:

```bash
git pull --ff-only
conda activate dog_ctrl
bash experiment/run_go1_support_cluster.sh
```

The script stops at its first failure. Expected: six Python pulse/lease tests, four cleanup
guard tests, Tkinter available, five CTest cases including the expected-failure
hardware lock, and `support_cluster=PASS`.

**2. Copy only the offline probe's build inputs from Ubuntu to Pi.** This is
source transfer, not a controller launch. Keep the robot powered off.

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
ssh pi@192.168.12.1 'mkdir -p ~/Robotic-Dog-Tracking-Interface'
rsync -avR \
  ./CMakeLists.txt \
  ./src/go1_lowlevel_experiment.cpp \
  ./src/go1_kinematics.cpp \
  ./src/go1_kinematics.hpp \
  ./src/go1_log_file.hpp \
  ./src/go1_operator_support.hpp \
  ./src/go1_operator_support_probe.cpp \
  ./externals/unitree_legged_sdk/include/ \
  ./externals/unitree_legged_sdk/lib/cpp/arm64/ \
  pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/
ssh pi@192.168.12.1 \
  'cd ~/Robotic-Dog-Tracking-Interface && \
   cmake -S . -B build-support-probe -DBUILD_TESTING=OFF -DBUILD_SDK_EXAMPLES=OFF && \
   cmake --build build-support-probe --target go1_operator_support_probe -j2'
```

The probe build is isolated in `build-support-probe`; it does not replace the
older `build-arm64` controller. If Pi storage is tight or the build fails,
stop and send the output rather than deleting another file.

**3. Start the Pi probe and SSH tunnel in separate Ubuntu terminals.** In
terminal A, leave this command running:

```bash
ssh -t pi@192.168.12.1 \
  'cd ~/Robotic-Dog-Tracking-Interface && \
   ./build-support-probe/go1_operator_support_probe'
```

It should say `C++ offline support probe listening on 127.0.0.1:18092; no Go1
SDK or motor commands`. The `-t` gives the probe a terminal so Ctrl-C
reaches it; the probe also flushes its output when run without a terminal.
Keep A running. In terminal B, leave the tunnel running:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18092:127.0.0.1:18092 pi@192.168.12.1
```

After authentication, terminal B normally stays blank: `ssh -N` runs no
remote command. Leave it running while using terminal C; Ctrl-C closes the
tunnel. A quiet tunnel alone does not prove the Pi probe is listening.

This exposes port 18092 only on Ubuntu loopback and forwards to Pi loopback.
Do not use `-g` or bind `0.0.0.0`. If the local port is already occupied by
the earlier Python probe, close only that known probe; otherwise stop and
inspect the owner rather than killing an unknown process.

**4. Test one click from Ubuntu, then close the transport.** In terminal C:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 experiment/operator_support_gate.py sender
```

The sender prints its connection attempt and `Support GUI opened`; a desktop
window should appear and terminal C should stay occupied until it closes.
An immediate return without a window is not a passing test. If this happens,
retain the output and run `echo "sender_exit=$?"` immediately after the sender
command. Connection or display failures should print an error.

Click once. Terminal A should show `support=true`, then `support=false`
automatically after about 1.5 seconds without continuing to hold the button.
Close the GUI, stop terminal B's tunnel with Ctrl-C, then stop terminal A's
probe with Ctrl-C. Do not run the controller. This is one combined end-to-end
check, not a series of separate robot trials.

**5. Close the cluster.** If all of the above passed, record
`support_cluster=PASS` and the Pi probe's `support=true` then `support=false`
output together. There is no need to ask for the next micro-step: continue to
the goal-based sequence in 2.1.13. If anything fails, stop at that step and
send the error; do not retry by opening robot UDP ports. No hardware movement
is authorized by the transport results.

### 2.1.12 Verified-file cleanup — use only after archiving

The support probe and SSH tunnel above create no Pi log file, so there is
nothing to delete after 2.1.11. Retain the Ubuntu dry-run archive from 2.1.9.
For a future Pi-generated log that has already been copied to this Ubuntu
repository's `logs/` directory, use
`experiment/cleanup_verified_go1_file.py` with the **exact** Pi file and
Ubuntu archived copy. The tool rejects broad or escaped Pi paths, checks that
both files are regular (not symlinks), compares SHA-256, and defaults to a
read-only preview. It deletes only the selected Pi original on `--execute`,
after rechecking its hash on the Pi; the Ubuntu archive remains untouched.

First inspect the specific filenames and run without `--execute`:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
python3 experiment/cleanup_verified_go1_file.py \
  --pi-file '/home/pi/Robotic-Dog-Tracking-Interface/logs/EXACT_FILE' \
  --ubuntu-copy "$PWD/logs/EXACT_ARCHIVE/EXACT_FILE"
```

Only after the preview prints the expected exact pathname and matching hash,
repeat the same command with `--execute` appended. Replace the uppercase
placeholders with the actual known single-file paths from that later run; do
not pass a directory or wildcard. If either file is missing, has changed, or
does not match, the script stops and deletes nothing. Do not use this tool to
erase the only copy of a capture or a file you have not inspected.

### 2.1.13 Goal-based test clusters

Use one result per **objective**, not one conversation per button or unit test.
The operator may run every step within a currently open cluster in one session;
only stop on a failed check, unexpected motion/sound, loss of the known stop
channel, or the explicit hardware lock below.

| Cluster | Objective and existing evidence | Completion / next action |
| --- | --- | --- |
| A — computer-to-Go1 command and feedback | Already completed by the archived `remote_preflight_fix_02.csv` run: the Pi-origin low-level damping stream and Go1 feedback were observed. This is **real computer control**, but not a movement test. | Accepted. Do not repeat solely to prove the link. |
| B — independent operator confirmation, Ubuntu to Pi | Accepted operator report on 2026-09-20: software cluster passed; Pi support transitions and clean probe shutdown observed; GUI first run exited 0. See 2.1.11 for evidence limits. | Complete. No routine repetition. Continue to the whole development block in 2.1.14. |
| C — first simple motion | One 5-mm *nominal* prone body rise, return to the floor, then damping, using the single-click confirmation only after visible floor contact. This is the next **new** physical objective, not a standing or walking test. | **Not executable yet.** The `prone-low-rise` CLI lock must remain until the physical stop/recovery path and command-effort behavior are reviewed against real hardware. A successful B does not itself satisfy that review. No command in this manual bypasses the lock. |
| D — repeatability and larger actions | Only after one C run is archived and its measured motion, feedback, faults, and final damping are accepted, consider a small number of identical prone repetitions. Standing hold, squat, and leg lift remain later, separate objectives. | Do not batch them into the first physical movement session or infer safety from a predicted torque limit. |

The one-person interface for C is a single mouse click, not a continuous
button hold: it leaves a hand free for the factory remote. Keep L2+B as the
known remote damping/stop action, not as the routine confirmation input. The
operator's earlier observation that the robot lay still after factory damping
supports the chosen prone starting pose, but does not establish that a new
low-level position command will exert negligible force. The present software
limits the *predicted* joint effort; it has not measured or bounded the actual
floor contact force, and a missing confirmation enters a position
hold. Section 2.1.16 now permits a fresh confirmation to recover from that
hold, while preserving the timeout in the failed-run record. Those are the specific reasons C remains locked, rather than a demand
for more tiny UI checks.

### 2.1.14 Next whole block — torque command and analysis verification

**Accepted operator result, 2026-09-20.** The supplied Ubuntu transcript and
matrix review show `torque_software_cluster=PASS (10/10)` in
`logs/torque-software/review-hr0uqdbk/`. Retain that archive. The six normal
runs and four stop cases passed; no repetition is needed. At 0.10 Nm the
reported RMSE was 0.02206, 0.00875, and 0.00449 Nm at 0.5, 1, and 2 Hz;
the corresponding gains were 0.897, 0.934, and 0.946. These are simulated
response metrics. Correlation-based negative lag values are alignment results,
not measured negative hardware latency. The commands below remain reference
instructions for a future relevant regression.

**Executable now on Ubuntu; robot off/disconnected.** This is one complete
batch toward torque tracking, replacing the previous non-executable next-step
outline. It exercises the real `torque-sine` command implementation and log
analyzer using simulated feedback. No SSH, Pi probe, GUI click, robot Wi-Fi,
Programming Module change, or support equipment is needed for this block.

**Run the whole block in one Ubuntu terminal:**

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 -B experiment/run_go1_torque_cluster.py
```

The script is in the current local checkout; it has not been committed or
pushed automatically. These commands run the modified local files directly.
It creates a unique `logs/torque-software/review-*` archive, builds
`go1_lowlevel_simulator` in a temporary directory, runs the complete matrix,
validates every CSV, and invokes the existing analyzer for FR_1. The simulator
is compiled without `GO1_WITH_SDK` and does not link Unitree's library. The
script always supplies `--dry-run`. ROS/catkin discovery is disabled for this
standalone build, so this block does not need ROS Python build dependencies.

| Cases | Parameters / objective |
| --- | --- |
| Six normal runs | FR_1; 0.10 and 0.20 Nm excitation amplitudes, each at 0.5, 1.0, and 2.0 Hz; six cycles per run (12, 6, or 3 seconds) |
| Normal cancellation | Inject single Ctrl-C at simulated t=4 s; check return and completion |
| Remote stop | Inject L2+B at simulated t=4 s; check the exact fault reason and damping commands on all joints |
| Double Ctrl-C | Inject at simulated t=4 s; check panic damping and completion |
| Watchdog | Inject at simulated t=4 s; check watchdog reason, damping, and completion |

The validator checks the full phase sequence, excitation duration, reconstructed
command waveform, zero feed-forward excitation on unselected joints, and the
expected stop/fault behavior. Injected stop responses must appear within 20 ms
of the scheduled simulation event. This is a software timing check, not a
measurement of radio, network, or motor latency.

**What torque is actually being tested:** in this implementation,
`torque-sine` sets the selected joint's SDK `Kp=Kd=0` but computes

```text
tau_ff = enveloped_sine + 2 * (captured_q - measured_q) - 0.2 * measured_dq
```

Thus `--amplitude-nm` is the sine component's amplitude, not a bound on the
complete torque command. The restoring terms remain present. This mode sends
a torque command; it does not close a torque-error feedback loop around
`tauEst`. The batch checks that exact command and reports response metrics
against total commanded torque, rather than labeling it pure sine tracking.

**Expected final output:**

```text
torque_software_cluster=PASS (10/10)
archive=/home/aims/Yuxuan/Robotic-Dog-Tracking-Interface/logs/torque-software/review-...
```

The whole batch stops at its first failed command or validation. Read the
reported error file and retain that archive; do not compensate by running a
hardware command. Temporary build files are removed, but all experiment logs,
analysis outputs, command lines, and results remain in the archive.

**Review all cases together after the batch:** this prints the newest archive's
result and torque metrics. Check that its path matches the path from your run.

```bash
python3 - <<'PYREVIEW'
from pathlib import Path
import csv
root = Path('logs/torque-software')
archives = [p for p in root.glob('review-*') if p.is_dir()]
if not archives:
    raise SystemExit('No torque archive found; run the block first.')
archive = max(archives, key=lambda p: p.stat().st_mtime_ns)
print('archive=' + str(archive.resolve()))
print((archive / 'result.txt').read_text())
matrix = archive / 'torque_matrix.csv'
if not matrix.exists():
    raise SystemExit('No completed matrix; inspect the failure above.')
for row in csv.DictReader(matrix.open()):
    print(f"{row['case']:18s} {row['result']} "
          f"RMSE={float(row['tau_rmse_nm']):.5f} Nm "
          f"gain={float(row['tau_gain_total']):.3f} "
          f"lag={float(row['lag_s']):.4f} s")
PYREVIEW
```

`torque_matrix.csv` contains the combined metrics. Each case also has a raw
CSV, analyzer summary, analysis transcript, command record, and validation
record. `manifest.json` records the Git revision and hashes of the local
sources, including uncommitted changes. No copying or deletion on the Pi is
part of this block.

**Acceptance and its meaning:** all ten cases must pass waveform and
phase/fault checks and produce finite response metrics. Do not apply hardware
bandwidth or torque-accuracy thresholds to this simulator: its `tauEst` is
constructed from a simplified plant with a 0.95 scale factor and artificial
joint dynamics. Normal software completion passes through `SAFE_HOLD`; the
dry-run then exits automatically. That is not a verified physical shutdown.
These results validate command generation, fault handling, and the measurement
pipeline; they do not establish real torque tracking or release hardware modes.

**Next physical objective remains one complete prone engagement/rise/return.**
The operator need not repeat this batch after a pass unless relevant code
changes. The remaining controller work below must be completed by development,
not delegated back as manual button checks. After physical entry and recovery
are validated, a torque-overlay trial and measured amplitude/frequency response
can use the same CSV analyzer. Pure torque operation still requires its own
support/operating design; `--support-confirmed` is not a way to bypass the
established no-rig setup.

#### Controller review record before the physical session

The table below records the findings before the recovery change. Section
2.1.16 supersedes its cancellation and late-confirmation behavior and defines
one-shot release authorization explicitly. Physical effort/contact review
remains open; the table must not be interpreted as the current software
behavior where 2.1.16 documents a change.

**1. Retain completed evidence.** Use sections 2.1.8, 2.1.9, and 2.1.11 as the
factory endpoint, software baseline, and operator transport records. Do not
repeat factory motion, the original dry-runs, or the transport session solely
to begin this work. The absence of lifting equipment remains the established
setup, not a request to obtain a rig.

**2. Resolve the following source-review findings.** The 2026-09-20 review of
`src/go1_lowlevel_experiment.cpp` found:

| Condition | Current implementation | Required outcome before a hardware procedure is released |
| --- | --- | --- |
| No eligible click during `PRONE_SETTLE` | After five seconds, enters `PRONE_SUPPORT_HOLD` with position stiffness active. A subsequent click cannot leave that phase. | Define and implement an explicit recovery from a missed/late confirmation, with a visible operator instruction and a bounded outcome; do not describe this state as shutdown. |
| Single Ctrl-C during the rise | Returns toward the engagement target, then latches `PRONE_SUPPORT_HOLD`; the existing test deliberately checks that confirmation cannot release it. | Define a complete cancellation-to-supported-exit path and distinguish it from emergency damping. Update the tests and operating instructions together. |
| Confirmation expires or the tunnel disconnects during `PRONE_RELEASE` | After a fresh one-second dwell has admitted release, the four-second release does not read `floorSupportObserved`. | Specify whether the click authorizes the entire release or must remain valid, and review loss-of-contact/transport behavior explicitly. The current 1.5-second pulse cannot provide continuous confirmation for the dwell plus four-second release. Do not silently lengthen the pulse or require a continuous hand-held button. |
| Initial calf commands are clamped to SDK limits | Measured prone calf positions lie outside those command limits; engagement can exert effort even without a requested rise. | Review the measured pose mismatch, initial PD contribution, gain changes, and release response using the existing factory evidence; establish the hardware observation and abort criteria before sending position commands. |
| Effort and movement limits | The 0.50 Nm limit is predicted PD effort; 5 mm is a nominal kinematic displacement. | Label these as command-model limits. Define how actual response will be observed and accepted; neither is a measured bound on floor force or actual body displacement. |

The first three entries are specific software/operating-contract gaps. The
last two require a concrete physical-test design grounded in the archived
measurements. Passing another existing unit-test suite cannot close them.

**3. Verify the revised complete action chain offline.** Once the control
changes exist, run a focused regression block covering normal completion,
late/missing/early clicks, pulse expiry and disconnect during release, normal
cancellation, L2+B, feedback loss, and command-effort limits. Include the real
sender/receiver protocol connected to the control core with synthetic robot
feedback: a standalone probe cannot establish how the controller consumes
confirmation. Record expected phase sequences and final command states for
each case. These revised checks are development work, not instructions to
inject faults into a powered robot.

**4. Release one concrete operator procedure.** Update the code gate and this
manual together only after reviewing the revised behavior and physical test
conditions. The release must identify the exact revision, build/deployment
commands, entry checks, normal and abnormal exit actions, logging fields,
acceptance criteria, and recovery procedure. Until then, keep the existing
hardware CLI rejection. Do not turn a dry-run into a hardware command by
removing `--dry-run`.

**Block completion:** implemented and reviewed recovery/confirmation behavior,
focused offline results, and a runnable revision-specific version of 2.1.15.
The operator has no additional transport work to do while this is pending.

### 2.1.15 First prone motion — complete session outline, not yet released

**Planning reference only.** This section defines the whole next physical
session so preparation, motion, shutdown, and result review stay together.
It intentionally contains no motor-launch command: 2.1.14 is not complete,
and the current executable rejects `prone-low-rise` hardware operation.
Do not begin this session with the present revision.

Once released, perform the following stages as **one session**, without asking
for a new conversational step between successful stages. Stop at an unexpected
result and preserve its evidence rather than repeating motion immediately.

1. **Prepare on Ubuntu with Internet available.** Record the released revision;
   build and deploy exactly its approved inputs. Create a unique local archive
   for this session and select an unused Pi log filename. Complete only the
   checks required by the changed controller; retain earlier acceptance.
2. **Establish the documented starting state.** Follow the released factory
   startup-to-prone procedure, confirm visible trunk support and unobstructed
   legs, and keep the remote available. Record the initial posture and any
   abnormal sound or movement. Do not infer contact from joint position alone.
3. **Establish command ownership and confirmation input.** Verify current
   controller processes and port ownership; temporarily stop only the known
   Programming Module using the existing exact-process procedure. Start the
   confirmation tunnel and GUI. The actual controller must own the support
   receiver; do not run the offline support probe alongside it on port 18092.
4. **Run exactly one released prone action.** Observe initial damping, gradual
   engagement, the small nominal rise, and return. No torque overlay, standing
   hold, squat, or single-leg action belongs in this session. The released
   procedure must state when to cancel and what that cancellation does.
5. **Confirm contact at the requested phase and finish release.** Click only
   after visible trunk contact and the controller's confirmation request.
   Follow the revised timeout/disconnect handling from 2.1.14. Observe final
   damping and process exit; a position-hold message is not completion. Do not
   use GUI closure or SSH disconnection as a motor shutdown command.
6. **Restore and close.** After the controller has exited and the robot is
   floor-supported, restore the Programming Module with its documented wrapper.
   Close the GUI and tunnel. Follow the established robot shutdown procedure
   if ending the session.
7. **Archive and review on Ubuntu.** Download the exact Pi CSV to the unique
   session archive, compare SHA-256, and run the existing low-level analyzer.
   Retain raw data, summary, plots, source revision, phase/fault console output,
   and a written observation of contact, motion, slip, sound, and final state.
   Retain the Pi original until the archive is verified. No cleanup is required
   just to finish this test.
8. **Decide from that one result.** Check the entire entry-to-exit sequence,
   feedback quality, joint response, effort estimates, faults, and observed
   contact against the released acceptance criteria. Synthetic thresholds alone
   do not certify the physical result. Review an incomplete or abnormal run
   before any repetition. Consider identical low-rise repetitions only after
   the first result is accepted; larger actions remain separate later blocks.

No run count, physical acceptance threshold, or recovery action omitted here
should be improvised at the robot. Those details and exact commands must be
filled in during 2.1.14 before this outline becomes executable.

### 2.1.16 Prone recovery — complete executable software block

**Accepted operator result, 2026-09-20.** The supplied transcript reports
`prone_recovery_cluster=PASS` in
`logs/prone-recovery-software/review-6nblyb66/`: revised recovery/release/core
checks and all four normal, cancellation, remote-stop, and watchdog sequences
passed. Acceptance here is based on that transcript. Retain the archive; no
operator rerun is requested. Commands below remain reproducible reference
instructions; Codex will execute future relevant local regressions directly.

**Purpose:** verify the revised entry/return/release controller as one complete
block before physical torque experiments depend on it. Run on Ubuntu with the
robot off or disconnected. No Pi session, tunnel, GUI, or hardware controller
is required. The script uses an SDK-free simulator and an ephemeral local TCP
port for its receiver-to-core integration check.

**Implemented changes:**

- A single Ctrl-C during the rise returns to the prone engagement target and
  permits a fresh contact confirmation to authorize normal release and damping.
- A missed five-second confirmation deadline still records a failed trial and
  holds position. A new false-to-true confirmation with a full one-second
  settled dwell can now complete shutdown. The failure reason is retained;
  recovered shutdown does not turn the experiment into a passing normal trial.
- An old continuously asserted confirmation cannot release the recovery hold.
  Loss of confirmation before completing the dwell restarts the dwell.
- Once the dwell succeeds, its acknowledgement authorizes the complete
  four-second release. Pulse expiry or sender disconnect does not restore
  stiffness. The input is not continuous contact sensing. Feedback, watchdog,
  and remote-stop checks continue to apply independently.
- Other fault holds are not made recoverable merely by clicking the button.
  If contact is never confirmed, position hold can still persist. This change
  provides an explicit recovery route; it does not promise autonomous physical
  shutdown in the absence of support evidence.

**Run once in an Ubuntu terminal:**

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 -B experiment/run_go1_prone_recovery_cluster.py
```

This builds the changed core tests and simulator, checks nominal control,
late/missing/early/interrupted confirmation, pulse expiry, cancellation,
feedback and other existing faults, command bounds, and the hardware CLI lock.
A real loopback TCP sender then drives the actual support receiver connected
to the core with synthetic robot feedback, including disconnection after
release authorization. It is automated; no manual transport repetition is
needed. This test exercises the wire protocol, not the Tkinter GUI again.

The script also produces and analyzes four full action logs: normal low-rise,
cancellation at simulated t=8 s, remote stop at t=8 s, and watchdog at t=8 s.
Normal/cancelled runs must return, settle, release and reach final damping.
Injected faults must preserve their exact abort reasons and reach damping.
Every run must reach `COMPLETE` with position-free damping commands on all
12 joints. Failed assertions or commands stop the block and preserve output.

**Expected final output:**

```text
prone_recovery_cluster=PASS
archive=/home/aims/Yuxuan/Robotic-Dog-Tracking-Interface/logs/prone-recovery-software/review-...
```

The unique archive contains `result.txt`, `recovery_core.txt`, four raw CSVs,
phase-sequence records, analyzer summaries/transcripts, command records, build
output, and source hashes. No file is overwritten and no Pi file is deleted.
The script performs the acceptance checks; a second manual analyzer pass is
not needed. If it fails, send the reported failure and its referenced text
file rather than running the whole block again immediately.

**Completion:** retain the result as software acceptance of the changed
recovery path. Do not repeat the torque matrix or GUI probe to proceed.
The next engineering decision is the first physical engagement/release test
using the archived prone pose and command-effort calculations, followed by
one low rise/return and eventually a measured torque overlay. This block
resolves software recovery behavior but does not measure trunk contact force,
validate a release trajectory on the robot, or unlock hardware commands.

### 2.1.17 Readiness review and Pi SDK adapter — completed

**Pi acceptance, 2026-09-20:** the operator supplied `pi_sdk_adapter=PASS`
with passing damping, prone-engagement, and torque-channel checks from
`logs/pi-sdk-adapter/review-68mDJRKk/`. Pi staging directory:
`/home/pi/go1-sdk-adapter-review-68mDJRKk`. GNU 8.3.0 compiled and linked
this fresh aarch64 build, and the resulting executable ran successfully.
The initial password rejection was recovered; it is not a failed test.

The approximately 7,723-second future-source timestamp warning is a clock
skew between source timestamps and Pi time. This fresh build visibly compiled
and linked the test; no rerun is needed solely for that warning. Future staging
now uses `rsync -avR --no-times` so the fresh Pi files receive arrival times.
No clock setting was changed and the completed archive is retained.
The commands below are reference only; this block is complete.

**Local work completed by Codex, 2026-09-20.** Reviewing nonideal feedback
exposed two release issues beyond the earlier ideal-tracking tests:

1. The effort limiter could permit Kp to increase when tracking error improved,
   even during the intended stiffness release. Release Kp is now capped at its
   preceding command value, so it cannot increase during that phase.
2. Excess roll/pitch during release called a soft-stop path that did not
   interrupt release. It now enters the existing panic-damping path with
   `prone_release_attitude`. This is an emergency response, not controlled
   lowering or proof against a physical drop.

Added regression cases disturb joint tracking during release, then remove the
error, and inject excessive roll. The complete recovery block was rerun with:

```bash
/home/aims/miniconda3/envs/dog_ctrl/bin/python3 -B experiment/run_go1_prone_recovery_cluster.py
```

**Result:** `prone_recovery_cluster=PASS`, including all four full action logs.
Archive: `logs/prone-recovery-software/review-8m0xlhts/`. The operator's prior
`review-6nblyb66/` remains valid evidence for its earlier revision. Do not rerun
this updated local block on the operator's behalf; Codex has already done it.

**Next local check performed:** added `go1_sdk_command_adapter_test`, which
uses the actual command converters and bundled SDK `Safety` functions on
synthetic states. It tests final/emergency damping, prone engagement, and a
selected torque channel. It never constructs `UDP` or `HardwareRunner`.
The reproducible configure/build/run sequence was:

```bash
# Executed locally by Codex with a fresh temporary build directory.
cmake -S . -B "$GO1_ADAPTER_BUILD" -DBUILD_TESTING=OFF   -DBUILD_SDK_EXAMPLES=OFF -DPYTHON_BUILD=OFF   -DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE
cmake --build "$GO1_ADAPTER_BUILD" --target go1_sdk_command_adapter_test -j2
strace -f -e trace=network -o "$GO1_ADAPTER_ARCHIVE/network.trace"   "$GO1_ADAPTER_BUILD/go1_sdk_command_adapter_test"
```

These are recorded execution steps, not another operator task. Exact expanded
commands and source/library hashes are saved in each archive's `commands.json`
and `sha256.json`.

The initial check in `logs/sdk-adapter-software/review-b61z_n4d/` failed an
assumption that the SDK preserves every position field: `PositionLimit`
clamped the `PosStop` sentinel, including with `Kp=0`. The test was corrected
to permit only this bounded zero-Kp position-field change while still requiring
unchanged mode, velocity target, gains and feed-forward torque. This is a
zero-stiffness command interpretation, not a firmware/contact measurement.
No production SDK adapter behavior was changed to make the check pass.

**Final Ubuntu result:** `sdk_command_adapter=PASS` and
`network_syscalls=NONE` in `logs/sdk-adapter-software/review-h61bhc3y/`.
The trace, build output, adapter output, source hashes and exact commands are
archived. The test uses zero-power synthetic feedback for `PowerProtect`;
it does not characterize protection thresholds or overload behavior.

**Why the Pi is needed now:** Ubuntu links the bundled amd64 SDK library;
the deployed controller uses its separately built ARM library. Verify the
same command-field behavior with that actual library before any motor trial.
This is one platform check, not a repeat of the torque sweeps or GUI tests.

#### Operator command — one complete no-motion Pi block

Use an available Pi SSH connection. Keep the robot's motors unpowered; if the
onboard Pi cannot be reached independently of normal robot startup, stop and
report that constraint instead of powering the robot merely for this check.
Do not stop the Programming Module or start any motor controller.

From **Ubuntu**, run:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
bash experiment/run_go1_pi_adapter_check.sh
```

Enter the Pi password when requested. No conda environment is required.
The script creates an isolated Pi staging directory and a unique Ubuntu
archive, copies only required build inputs, builds the adapter test on the Pi,
runs it, and saves the combined output back on Ubuntu. It does not alter the
existing Pi controller checkout, open robot UDP ports, ask for ARM, or delete
files. It deliberately does not run the full controller binary.

Expected output after the build:

```text
PASS damping
PASS prone engagement
PASS torque channel
NOTE: SDK PositionLimit clamps PosStop q with Kp=0; effort fields preserved
sdk_command_adapter=PASS; no UDP constructed or motor commands sent
pi_sdk_adapter=PASS
archive=/home/aims/Yuxuan/Robotic-Dog-Tracking-Interface/logs/pi-sdk-adapter/review-...
```

The SDK may also print `End Safety.` at destruction. Preserve the archive
and send the final result. A build failure or changed command field stops the
script; send the referenced transcript without retrying by launching a motor
program. The no-network syscall trace was obtained on Ubuntu; this Pi script
does not require `strace` to be installed and does not claim a Pi syscall trace.

**Boundary after this block:** a passing ARM adapter check validates library
compatibility only. Prone calf measurements still differ from valid position
command bounds; engagement can load the robot against the floor before any
nominal 5-mm rise. Actual contact force and settling are not established by
these software tests. The first physical session needs the reviewed contact,
stop, and observation conditions in 2.1.15; do not remove a hardware lock or
use `--support-confirmed` to proceed from this platform check.

### 2.1.18 Engagement geometry — reviewed; starting posture confirmed

**Operator observation received:** belly and all four feet are on the ground,
placed in factory damping using L2+B. This answers the contact-layout question.
It was not necessary for the prior SDK adapter test. Section 2.1.19 uses this
layout as the required starting condition for the new physical trial.

After Pi adapter acceptance, Codex evaluated the recorded factory prone pose
from `GO1_NATIVE_RETURN_ANALYSIS.md` with the repository's forward kinematics.
Only each calf was changed to the current command minimum, -2.721 rad; hips
and thighs stayed at the measured pose. This isolates the change required by
initial position engagement, before the requested 5-mm rise.

Archive: `logs/engagement-review/review-ip3wjp1l/`. It includes the calculation
source, executable, `geometry.csv`, exact compile/run commands in
`commands.json`, and source hashes. The calculation used `g++ -std=c++14`,
`src/go1_kinematics.cpp`, and the archived calculation source; no robot
connection or simulation rerun was required.

| Leg | Calf target change (rad) | Static Kp=5 contribution (Nm) | Modeled foot displacement (mm) | Foot z change (mm) |
| --- | ---: | ---: | ---: | ---: |
| FR | 0.073212 | 0.366060 | 15.59 | -14.78 |
| FL | 0.076724 | 0.383620 | 16.34 | -15.61 |
| RR | 0.078541 | 0.392705 | 16.72 | -15.96 |
| RL | 0.047375 | 0.236875 | 10.09 | -9.61 |

The effort column is `5 * (target_q - recorded_q)` at zero velocity and
unchanged initial pose, before any subsequent physical response. It is not
measured torque or a floor-force bound. The displacement is forward-kinematic
foot movement relative to the trunk, not predicted body rise or measured
contact motion. Floor constraints, friction, compliance and body contact
can change the physical response substantially.

**Decision:** initial engagement is itself a movement/load-transfer experiment;
calling the whole first run a 5-mm motion understates the commanded geometric
change. Plan a complete engagement-and-release trial before adding the nominal
rise or torque excitation. Do not widen command limits to remove this mismatch.
The existing hardware lock stays in place. The current `prone-low-rise` mode
includes a rise; there is no released engagement-only hardware command yet.

**Physical input now needed:** from the operator's existing observations of
normal factory prone damping, establish whether the trunk is floor-supported
with all four feet also touching the floor, or whether some feet are folded
underneath, free, or not clearly visible. Record which feet if known. Do not
power or move the robot solely to answer this question. This information is
needed to define the contact assumptions and what the first engagement trial
must observe. It is not a request to repeat the completed factory capture,
transport check, or SDK test.

### 2.1.19 First powered engagement and release — complete operator block

**HOLD after the first run:** see 2.1.20 before any repeat. The procedure below
is retained for reference; it is not a request to rerun the failed trial.

**This block sends motor commands.** The preceding adapter test did not.
Perform one trial, archive its result, and stop for review before repeating or
adding a rise/torque excitation. It is a first hardware validation, not a
claim that simulation establishes physical safety.

#### Implemented profile and local acceptance

New mode: `prone-engagement`. It starts with one second of quiet prone
feedback in damping, ramps position stiffness over four seconds, checks one
second of quiet feedback, waits for a fresh visual-contact acknowledgement,
releases stiffness over four seconds, sends one second of final damping,
and exits. It does not enter `PRONE_RISE` or apply feed-forward torque.
All position targets remain at the captured pose clamped to existing SDK
bounds; velocity targets stay zero.

- Maximum Kp is 1 (one fifth of the low-rise fixture); Kd remains 1.
- The position-command limiter budgets 0.10 Nm per joint using current
  position/velocity feedback. Static initial calf contributions calculated
  from the archived pose are about 0.047–0.079 Nm. These are predicted command
  efforts, not measured contact forces. Emergency damping is not subject to
  that position-command cap.
- A fresh measured joint speed over 0.08 rad/s triggers emergency damping.
  Roll/pitch excursion limit remains 0.05 rad; loss of feedback, remote validity,
  or watchdog faults retains the existing damping response.
- Quietness checks retain the 0.05 rad/s threshold. For this mode only, the
  target-error gate is 0.10 rad instead of 0.05: stationary calves at the
  recorded floor-supported pose must not be forced to reach their clamped
  targets simply to allow release. Visual contact confirmation is still required.
- A single Ctrl-C returns toward the same fixed target and then waits for
  confirmation; gains cannot increase after cancellation. L2+B or double
  Ctrl-C within one second requests emergency damping and automatic close.
- A missing click after five seconds enters the recoverable support hold and
  marks the trial failed. A new one-second confirmation dwell can finish release,
  but the failure remains in the log. A hold message is not motor shutdown.

**Codex local evidence:** updated recovery cases, including stationary
floor-constrained feedback, cancellation, late confirmation and the tighter
speed guard passed in `logs/prone-recovery-software/review-9jaac3eu/`.
Full Linux build and CTest passed **28/28**, and the engagement-only dry-run
and analyzer passed in `logs/prone-engagement-software/review-gfb8ae3m/`.
The final hardware startup change was compiled and the SDK adapter rerun there.
Exact commands, test outputs, CSV, and final source hashes are archived.
No operator simulation repeat is needed.

#### 1. Deploy from Ubuntu — no motion starts here

Run in an **Ubuntu terminal**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
bash experiment/prepare_go1_prone_engagement.sh
```

This script contains `ssh` and `rsync`. It builds on the Pi in
`/home/pi/go1-prone-engagement`, saves a unique Ubuntu deployment archive,
and ends with `engagement_deployment=PASS; no controller started`.
It does not replace the older `build-arm64` executable. If it fails, stop at
that error. Preparation does not arm or launch the trial.

#### 2. Prepare three terminals and confirm the physical starting state

Use the established factory procedure to have the belly and all four feet
on the flat floor, with L2+B damping selected. Do not lift the powered robot.
Keep hands clear of the legs and the factory remote immediately available.
Do not start if the posture or remote response is uncertain. The previous
calf geometry calculation means even this engagement can move/load the robot;
there is no guarantee of a motionless trial.

**Terminal A on Ubuntu: open the Pi shell.** All commands in steps 3–4 and 6
run inside this Pi shell:

```bash
ssh -t pi@192.168.12.1
cd /home/pi/go1-prone-engagement
```

**Terminal B on Ubuntu: start the confirmation tunnel and leave it running.**
A blank terminal after authentication is normal:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18092:127.0.0.1:18092 pi@192.168.12.1
```

If a port is occupied, identify its owner rather than killing an unknown
process. Do not run an offline support probe: the controller owns that receiver.

**Terminal C on Ubuntu: prepare the environment now, but wait until step 4
to run the sender.**

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
```

#### 3. Terminal A / Pi — verify ownership and temporarily stop programming.py

```bash
ip route get 192.168.123.10
pgrep -af 'go1_lowlevel_experiment|example_|run_torque_tracking' || true
PROGRAMMING_PATTERN='^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
pgrep -af "$PROGRAMMING_PATTERN"
sudo ss -Huanp | awk '$4 ~ /:8090$/ { print }'
sudo fuser -v 8090/udp
```

Proceed only if no experiment controller is running, the route is the known
internal Ethernet route, and exactly one `programming.py` PID matches the UDP
8090 owner. Do not stop startup_manager, Legged_sport, appTransit or other
Unitree services. With the starting posture still confirmed, run:

```bash
mapfile -t PROGRAMMING_PIDS < <(pgrep -f "$PROGRAMMING_PATTERN")
if [ "${#PROGRAMMING_PIDS[@]}" -eq 1 ]; then
  ps -fp "${PROGRAMMING_PIDS[0]}"
  kill -TERM "${PROGRAMMING_PIDS[0]}"
  sleep 2
else
  echo 'STOP: expected exactly one Programming Module PID'
fi
pgrep -af "$PROGRAMMING_PATTERN" || true
sudo ss -Huanp | awk '$4 ~ /:8090$/ { print }'
```

Continue only if the module has stopped and no UDP 8090 owner remains. If it
reappears, do not repeatedly kill it. Stop and use step 6 to restore/check it.
The controller also checks that its UDP port can be bound before ARM.

#### 4. Terminal A / Pi — launch one trial; connect GUI before starting motion

```bash
./build/go1_lowlevel_experiment --mode prone-engagement \
  --prone-confirmed --remote-confirmed --local-port 8090 \
  --log logs/prone_engagement_01.csv
```

If that filename already exists, do not overwrite an unarchived result. The
program asks before replacement. For the first run it should not exist.
Read the profile, then enter `ARM`. The program then prints:

```text
Support receiver ready; no motor packets sent yet. Open the Ubuntu GUI.
With belly/all feet on floor and remote ready, press Enter here to begin:
```

**Leave A at this prompt. In Terminal C on Ubuntu, run:**

```bash
python3 -u experiment/operator_support_gate.py sender
```

The GUI should open. If it does not, do not press Enter in A. Cancel the
waiting program with Ctrl-C then Enter if needed, and restore the module
using step 6. No experiment motor packets are sent while it waits here.

With the GUI visible, remote ready and starting contact reconfirmed, press
Enter in A to begin. Return focus to the GUI without clicking its button yet.
Keep Terminal A visible alongside it so its phase messages can be read.

#### 5. Observe engagement, confirm contact, and let shutdown complete

Expected normal sequence:

```text
PRONE_OBSERVE -> PRONE_ENGAGE -> PRONE_ENGAGE_HOLD -> PRONE_SETTLE
-> PRONE_RELEASE -> PRONE_FINAL_DAMPING -> COMPLETE
```

At `CONFIRM CONTACT NOW`, click once **only if belly and all four feet remain
supported and the trial looks normal**. Keep GUI focus for the 1.5-second
pulse; switching away cancels it. The one-second dwell authorizes the full
four-second release; there is no need to hold the button or click repeatedly.
Watch the robot until the program reports completion and returns to the Pi
shell. Full normal duration is roughly 12–17 seconds depending on confirmation.

If there is unexpected lifting, slipping, leg motion, abnormal sound, lost
contact, or uncertainty, use the known **L2+B** stop and keep clear. Do not
click contact confirmation to make the program progress. Panic sends its
final damping window and closes automatically. Emergency damping can allow
settling; it is not a controlled return or a guarantee against damage.

Single Ctrl-C is a normal cancellation request, **not immediate exit**.
If the robot remains correctly supported, a fresh contact click can finish
the cancellation release. If contact is uncertain, use L2+B instead.
If `PRONE_SUPPORT_HOLD` appears, the program is still applying position
control. A late contact click can recover a confirmation timeout, but does
not recover every other fault. If a fresh click does not begin release after
the one-second dwell, use L2+B and retain the failed run. Do not abandon the
controller or close SSH while it is holding. Do not inject deliberate faults
into this first physical trial; those checks have been done offline.

#### 6. Terminal A / Pi — restore the module after the controller has exited

Keep the robot floor-supported. Whether the run completed, failed, or was
cancelled before starting, restore/check the module once the controller has
closed:

```bash
if pgrep -f "$PROGRAMMING_PATTERN" >/dev/null; then
  echo 'Programming Module already running; not starting a duplicate'
else
  (cd /home/pi/Unitree/autostart/programming && bash ./programming.sh)
fi
sleep 2
pgrep -af "$PROGRAMMING_PATTERN"
sudo ss -Huanp | awk '$4 ~ /:8090$/ { print }'
ls -lh /home/pi/go1-prone-engagement/logs/prone_engagement_01.csv
sha256sum /home/pi/go1-prone-engagement/logs/prone_engagement_01.csv
```

Expect one Programming Module process and its usual UDP 8090 connection.
If restoration is abnormal, keep the robot prone and report it. Close the GUI
with its window close button; stop Terminal B's tunnel with Ctrl-C.

#### 7. Ubuntu — archive the one physical result and send it for review

In Terminal C after the GUI closes (also activate this environment if using
a new Terminal D):

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
mkdir -p logs/prone-engagement-hardware
GO1_ENGAGEMENT_ARCHIVE=$(mktemp -d "$PWD/logs/prone-engagement-hardware/review-XXXXXXXX")
scp pi@192.168.12.1:/home/pi/go1-prone-engagement/logs/prone_engagement_01.csv \
  "$GO1_ENGAGEMENT_ARCHIVE/"
sha256sum "$GO1_ENGAGEMENT_ARCHIVE/prone_engagement_01.csv"
python3 experiment/analyze_lowlevel_log.py \
  "$GO1_ENGAGEMENT_ARCHIVE/prone_engagement_01.csv"
printf 'archive=%s\n' "$GO1_ENGAGEMENT_ARCHIVE"
```

Compare the SHA-256 with Terminal A. Retain the Pi original for now. If the
program was cancelled before it started and no CSV exists, report that; do
not analyze an old file as a new result. Send the console output, archive
path, and observation of belly/four-foot contact throughout, movement/slip,
sound, and final posture. Codex can review the local archive directly.

The analyzer exiting 0 is not physical acceptance. Require the complete
normal phase chain and final damping, no abort/watchdog, feedback >=450 Hz,
p99 gap <=10 ms and maximum gap <=20 ms, observed support throughout, and
no unexpected motion/sound. Review actual gains, total predicted effort,
velocity, attitude, and any target/feedback mismatch. Failure to move toward
the clamped calf target is not itself a failed engagement test. Do not infer
contact force or torque-tracking bandwidth from this run. After review,
decide whether engagement is repeatable before progressing to rise or torque.

### 2.1.20 First hardware result — failed readiness, no engagement

**Reviewed locally on 2026-09-21.** Archive:
`logs/prone-engagement-hardware/review-fTXlbb7z/prone_engagement_01.csv`.
SHA-256 `39dc7ddccab8f5b3b8f40d8fd3c4d824e7c139e6ed0a121c2f8aa338f45c72b6`
matches the supplied Pi checksum and the local recomputation. Deployment was
`logs/engagement-deployment/review-glwGHRK7/`; Pi binary SHA-256 was
`e77ab13da7d5d3d2936a8acb429e6393e118852b317a1596703ebd9853bf6744`.

Codex ran the analyzer in the existing project environment:

```bash
/home/aims/miniconda3/envs/dog_ctrl/bin/python3 -B \
  experiment/analyze_lowlevel_log.py \
  logs/prone-engagement-hardware/review-fTXlbb7z/prone_engagement_01.csv --no-plots
```

The analyzer succeeded and wrote its summary next to the CSV. The earlier
NumPy error was from Terminal D using `base`; no installation or new download
is required. Activating `dog_ctrl` is now explicit in the archive procedure.

| Evidence | Result |
| --- | --- |
| Phase chain | `PRONE_OBSERVE -> PANIC_DAMPING -> COMPLETE`; no engagement phase |
| Abort reason | `prone_observe_remote_not_valid_for_prone_mode`, first abort at sample 2501 |
| Recorded samples / fresh states | 2,766 / 67 |
| Fresh-state rate | 11.87 Hz |
| p99 / maximum fresh gap | 146.356 / 172.155 ms |
| Fresh remote validity | 0%; all logged remote headers, buttons and stick fields zero |
| Low-level flag | All fresh states 255 |
| All recorded joint commands | Kp=0, Kd=1, feed-forward torque=0 |
| Maximum joint speed / temperature | 0.02724 rad/s / 56 C |
| Watchdog cycles | 0; feedback tick gaps over 20 ms: 63 |

The operator confirmed the remote was powered and connected, no movement was
observed, and the sound was the usual one. Do not treat all-zero received
remote fields as proof the remote was switched off. The previous accepted
preflight has valid remote headers 85/81 and approximately 498 Hz feedback;
this run is a materially different communication result. Its root cause is
not established by these logs. A connected GUI does not validate robot/remote
feedback.

Terminal A proves the controller closed its UDP socket and returned to the
shell before archiving. The Programming Module was restarted as PID 8590
and reclaimed its expected UDP 8090 connection. The GUI subsequently wrote
to a closed support receiver (`Broken pipe`); repeated Ctrl-C interrupted
Tkinter. That GUI traceback did not cause the earlier observation abort.
Close the GUI window and Terminal B tunnel now; no experiment remains active
in the supplied transcript. No extra restoration or Pi-file deletion is needed.

**Independent defect fixed locally:** the SDK returns successful send byte
counts (614 for these low commands), while the core expects status 0. The
hardware adapter now normalizes an exact SDK command-length result to 0,
rejects missing/short/failed sends, and keeps raw byte counts in CSV logs.
High-level send handling uses the corresponding SDK length constant too.
This would affect later entry/final-damping checks; it is not an explanation
for zero remote fields or this observation-stage abort. No validity/rate
threshold was relaxed. New send-status/final-damping tests and the entire
recovery block passed in `logs/prone-recovery-software/review-stuioqjd/`.
The sender's startup wording now distinguishes the controller from the offline
probe, and Ctrl-C requests orderly GUI closure instead of interrupting a Tk
callback. Six existing pulse/lease tests passed; real GUI closure has not been
retested on the operator's desktop. These changes have not been deployed to Pi.

#### Next operator block — passive capture, no experiment controller

Keep the restored factory processes running. This diagnostic only records
existing traffic; it does not send control commands, stop services or require
remote button presses. Keep the already-prone robot undisturbed. If the session
has ended, report that rather than powering it solely on these instructions.
Do not rerun `prone-engagement` or the original active preflight yet.

In **Terminal A, the Pi shell**:

```bash
cd /home/pi/go1-prone-engagement
pgrep -af 'Legged_sport|programming[.]py|go1_lowlevel_experiment|example_'
sudo ss -Huanp
```

There must be no custom experiment or SDK example active. Preserve that output.
Then capture a new, non-overwriting file:

```bash
if [ -e logs/feedback_diag_01.pcap ]; then
  echo 'STOP: diagnostic file already exists; preserve it'
else
  sudo timeout -s INT 15 tcpdump -i eth0 -nn -s 0 -U \
    -w /home/pi/go1-prone-engagement/logs/feedback_diag_01.pcap \
    'udp and (port 8007 or port 8008 or port 8082 or port 8090)'
fi
sha256sum /home/pi/go1-prone-engagement/logs/feedback_diag_01.pcap
```

`timeout` normally ends this capture after 15 seconds (exit 124 can be normal).
If tcpdump is unavailable, permission is denied, or capture fails, retain the
error; do not install or launch a control program as a substitute.

In **an Ubuntu terminal**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
mkdir -p logs/feedback-diagnostic
GO1_FEEDBACK_REVIEW=$(mktemp -d "$PWD/logs/feedback-diagnostic/review-XXXXXXXX")
scp pi@192.168.12.1:/home/pi/go1-prone-engagement/logs/feedback_diag_01.pcap \
  "$GO1_FEEDBACK_REVIEW/"
sha256sum "$GO1_FEEDBACK_REVIEW/feedback_diag_01.pcap"
printf 'archive=%s\n' "$GO1_FEEDBACK_REVIEW"
```

Send the process/socket output, capture summary, and archive path. Codex will
inspect the local packet capture and compare it with the prior factory capture;
no manual simulation or analyzer run is requested. This capture observes the
restored factory state, not the vanished experiment state, so it may narrow
rather than fully explain the failure. Preserve the original hardware CSV.

### 2.1.21 Passive diagnostic result and onboard cleanup preparation

**Capture accepted, 2026-09-21.**
`logs/feedback-diagnostic/review-IxwMnFeM/feedback_diag_01.pcap` matches the Pi
SHA-256 `7d04559f8a26da35b5124ad36b0f80e1266a0c4cb7ba12ee1dd4442043cd7f46`.
The reported tcpdump totals were 22,198 captured, 22,373 received by filter,
and zero kernel drops. These are not identical counts; do not describe the
capture as proving that every filtered packet was saved.

Codex extended the decoder to accept Ethernet framing as well as the earlier
Linux cooked framing. Three parser regression tests passed. It decoded all
22,198 saved packets with valid profile CRCs. Commands used:

```bash
/home/aims/miniconda3/envs/dog_ctrl/bin/python3 -B \
  experiment/decode_native_go1_pcap.py \
  logs/feedback-diagnostic/review-IxwMnFeM/feedback_diag_01.pcap \
  --out logs/feedback-diagnostic/review-IxwMnFeM/decoded
/home/aims/miniconda3/envs/dog_ctrl/bin/python3 -B -m unittest discover \
  -s test -p test_decode_native_go1_pcap.py -v
```

| Flow | Saved packets | Measured rate | Median / maximum capture gap |
| --- | ---: | ---: | --- |
| MCU 8007 -> Pi factory 8008 | 14,799 | 999.93 Hz | 1.000 / 2.647 ms |
| Pi factory 8008 -> MCU 8007 | 7,399 | 499.93 Hz | 2.000 / 4.008 ms |

Factory commands had Kp=0, Kd=2 and zero feed-forward torque for all 12 joints;
feedback was quiet. This later restored-factory capture establishes healthy
factory traffic, not healthy experiment-port feedback at the time of failure.
Only these two flows appear in this eth0 capture; same-host traffic need not
traverse that interface.

The SDK `refineState` mapping confirms native bytes 759–798 are the 40-byte
remote payload. All current factory states have that payload zeroed. Offline
comparison found all 10,000 states in the older `native-prone.pcap` likewise
zeroed, while all 39,859 states in `native-return.pcap` had nonzero payloads,
including 85/81 headers and recorded button changes. Detailed counts are in
`remote-review.json` and `previous-remote-review.json` beside this capture.
Zero remote data is therefore context-dependent in the archived native stream;
it does not prove the operator powered the remote off. Neither the remote
availability issue nor the experiment's 12 Hz receive rate is resolved.
Section 2.1.19 remains on hold. Do not stop factory control processes or bypass
remote validation based on this capture.

#### Cleanup scope

Prepare cleanup without mixing it with another powered trial. The robot can
remain off if the Pi is independently accessible; no motor controller or
capture may be writing the target files during deletion. If Pi access is not
available, defer this housekeeping rather than power the robot just to free space.

| Onboard item | Plan |
| --- | --- |
| `/home/pi/go1-sdk-adapter-review-68mDJRKk` | Completed scratch code/build; archive and verify first, then consider exact removal |
| `/home/pi/go1-prone-engagement/logs/prone_engagement_01.csv` | Already archived and hash-verified on Ubuntu; eligible for single-file cleanup after preview |
| `/home/pi/go1-prone-engagement/logs/feedback_diag_01.pcap` | Already archived and hash-verified on Ubuntu; eligible for single-file cleanup after preview |
| `/home/pi/go1-prone-engagement` code/build | Keep for fault investigation; log cleanup is not whole-directory removal |
| `/home/pi/Robotic-Dog-Tracking-Interface` | Keep source, SDK and known controller builds; inventory older logs before selecting anything else |
| `/home/pi/Unitree`, autostart, factory modules | Out of cleanup scope |

No remote deletion has been performed by Codex.

#### A. Ubuntu command — inventory and archive completed SDK scratch code

Run on **Ubuntu**, not inside a Pi SSH shell:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 -B experiment/prepare_go1_cleanup.py
```

This script contains SSH calls and may ask for the Pi password several times.
It inventories the three exact project/scratch paths above and reports storage.
If the known SDK scratch directory exists, it streams a tar backup to a unique
`logs/pi-cleanup/review-*` directory on Ubuntu. It compares regular-file hashes
and symlink metadata with a Pi manifest, then rechecks that the manifest did
not change during backup. It never extracts tar contents and **never deletes
anything**. A missing scratch directory is reported and skipped; a changed or
invalid backup stops preparation. It does not automatically select similarly
named unknown directories for removal.

Send the printed archive path. Codex will review `inventory-before.json`,
`inventory-after.json`, `plan.json`, and the verified backup before preparing
any code-directory removal. This is inspection of the actual cleanup targets,
not a reason to repeat an experiment. Backup verification tests cover matching,
corrupted, missing and outside-path archive contents; three tests passed locally.

#### B. Optional log cleanup — preview exact files from Ubuntu

The single-file cleanup tool now permits the engagement log root as well as
the original project log root. SSH password authentication is supported.
Path, mismatch, preview and recheck guards passed five tests locally.

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 -B experiment/cleanup_verified_go1_file.py \
  --pi-file /home/pi/go1-prone-engagement/logs/prone_engagement_01.csv \
  --ubuntu-copy logs/prone-engagement-hardware/review-fTXlbb7z/prone_engagement_01.csv
python3 -B experiment/cleanup_verified_go1_file.py \
  --pi-file /home/pi/go1-prone-engagement/logs/feedback_diag_01.pcap \
  --ubuntu-copy logs/feedback-diagnostic/review-IxwMnFeM/feedback_diag_01.pcap
```

These commands only preview deletion after computing matching hashes on both
machines. If both previews show the correct paths and the files are no longer
being written, repeat each exact command with `--execute` appended to delete
only its Pi original. The tool rechecks the remote hash immediately before
removal; the Ubuntu copies and analyses are retained. If a path is absent or a
hash differs, stop and keep the evidence rather than using a wildcard or `rm -rf`.
Log cleanup is optional and does not fix the controller communication issue.

### 2.1.22 Reviewed onboard cleanup — exact execution commands

**Preparation reviewed, 2026-09-21.** Archive:
`logs/pi-cleanup/review-oyswg744/`. Codex independently verified all 83 regular
files in `sdk-staging.tar.gz` against both Pi manifests. Backup SHA-256:
`1688ceac0e0cdd1eeee6f9d63c24a8c22b8bbcc2c7aa0e3ebaf7285263baa61e`.
The SDK staging directory was unchanged during backup. Its reported disk use
was 848 KiB. The Pi reported 13,187,010,560 bytes free (about 12.3 GiB), so no
emergency storage cleanup is needed.

Both exact log previews also passed: the engagement CSV (3,482,646 bytes)
and passive PCAP (17,965,674 bytes) match their retained Ubuntu archives.
Nothing has been deleted by those previews or by Codex.

The following block performs the reviewed cleanup. Run it from **Ubuntu**
only after all experiment/capture/build processes have stopped. It does not
stop factory services. It may ask for the Pi password for each command.
Each command is independent: if one reports failure, retain its output and
stop instead of continuing manually or substituting `rm -rf`.

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 -B experiment/cleanup_verified_go1_sdk.py \
  --archive logs/pi-cleanup/review-oyswg744 --execute
```

This command is restricted to
`/home/pi/go1-sdk-adapter-review-68mDJRKk`. It validates the local backup
again, compares a fresh Pi manifest, checks for use by Pi-owned processes,
and removes only individually verified files followed by empty directories.
Changed/added files, a symlink root, active use or an inability to inspect a
process stops cleanup. A failure partway through removal can leave a partially
removed scratch tree; the complete Ubuntu archive is retained.

The first execution stopped before deletion because `/proc/686/cwd` was
permission-protected. The checker now retries protected process links using
read-only `sudo -n readlink`; deletion still runs as `pi`. If noninteractive
sudo is unavailable or inspection fails, cleanup stops without bypassing the
check. Re-run the same Ubuntu command above; no new backup or Pi deployment
is required. Errors now show a short explanation instead of the embedded SSH
program. All 13 local cleanup regression tests passed, including protected
process inspection, active SDK use, failed sudo inspection, backup integrity,
and exact-file deletion safeguards.
No Pi execution has been performed by Codex. An already-absent scratch root
is reported without touching any other directory.

After SDK cleanup succeeds, execute the two already-reviewed log removals:

```bash
python3 -B experiment/cleanup_verified_go1_file.py \
  --pi-file /home/pi/go1-prone-engagement/logs/prone_engagement_01.csv \
  --ubuntu-copy logs/prone-engagement-hardware/review-fTXlbb7z/prone_engagement_01.csv \
  --execute
python3 -B experiment/cleanup_verified_go1_file.py \
  --pi-file /home/pi/go1-prone-engagement/logs/feedback_diag_01.pcap \
  --ubuntu-copy logs/feedback-diagnostic/review-IxwMnFeM/feedback_diag_01.pcap \
  --execute
```

Each file is rehashed on both machines and on the Pi immediately before
removal. Ubuntu raw logs, decoded data and summaries remain. The current
engagement code/build, original project, Unitree SDK in the working projects,
and `/home/pi/Unitree` remain in place. No other adapter folders or older
builds are selected. Send the completion output to record actual deletions;
preparation alone is not cleanup completion. The communication investigation
remains open, and 2.1.19 remains on hold regardless of cleanup outcome.

### 2.1.23 Known remote input under factory control — next operator block

Cleanup completion was reported by the operator. Further housekeeping is not
required to continue. The failed engagement remains blocked by sparse feedback
and unavailable remote input; no additional torque simulation is required now.

**Purpose:** determine whether the factory state stream carries a deliberate
L2+B press and release, rather than inferring remote availability from idle
zeros. This is a 20-second passive capture with one prompted two-second button
press. It does not start a custom controller, stop factory services, or change
the robot's control mode through software. The operator uses the same factory
L2+B command already used to make this robot prone.

Run only with the robot already prone, belly and all four feet supported, remote
powered and connected, sticks centered, and no custom controller running. If it
is standing or its posture is uncertain, do not type READY. Keep clear of the
joints; unexpected movement or sound means stop the test and report it.

**Ubuntu terminal (includes SSH, capture, SCP, hash check and local analysis):**

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
bash experiment/capture_go1_remote_input.sh
```

Type `READY` at the posture prompt. Enter SSH/sudo passwords when requested.
After capture starts, leave the remote untouched for five seconds. At `NOW`,
hold L2+B together; at `RELEASE`, release both buttons. Leave sticks centered
throughout. No support GUI or separate SSH tunnel is needed. Allow capture and
download to finish. If you miss the cues, report that; do not claim a successful
input test. Recorded cue timestamps are prompts, not proof of button presses.

The helper retains new Pi files in a unique `logs/remote-input-*` directory and
archives them on Ubuntu under `logs/remote-input-diagnostic/review-*`. It records
processes, socket owners, capture statistics, UTC cue times and checksums. The
decoder reports remote headers, button counts and timed transitions, plus native
feedback gaps and command fields. `l2_b_samples > 0` alone is insufficient:
review the 5551 header and a press/release transition aligned with the prompts.

**Local preparation verified:** shell syntax and all three packet-parser tests
passed. The updated decoder reproduced zero remote payloads in all 14,799 states
of `feedback_diag_01.pcap`; the prior `native-return.pcap` produced 39,859 valid
5551 headers, 440 L2+B samples (mask 0x0220), and 13 header/button transition
entries. Both full captures passed their existing CRC checks. The helper itself
has not been executed on the Pi by Codex.

Send the archive path, whether you followed the two cues, and any observed
motion or unusual sound. Codex will perform the detailed offline review.
If remote transitions appear, the next diagnostic targets the experiment's
receive path with simultaneous packet/log evidence; factory-stream success
alone does not authorize engagement. If remote remains zero, investigate the
factory remote delivery path before any powered custom trial. Do not repeat
2.1.19 or weaken its remote/rate checks.

#### First attempt and corrected retry

The operator's `review-zGNcnFI3` attempt printed tcpdump's listening message,
then exited without either button prompt or download. Its Pi directory is
`/home/pi/go1-prone-engagement/logs/remote-input-PnjnwYaB`; the local directory
contains only that path record. Do not count this as a remote-input test.
Existing Pi capture data is retained; the retry selects a new directory.

The helper incorrectly used `kill -0` to check the background sudo process.
A root-owned process can deny that signal-permission check while still running.
The script now uses `ps -p` to inspect process existence without signaling it,
and reports the child exit status on an early capture failure. The exit trap
in the old script waited for capture completion, so its lack of prompts does
not establish that tcpdump itself failed. Two local tests execute the actual
remote Bash block with mocked commands: denied signal permission still reaches
both prompts; early capture failure stops before prompting. Shell syntax passes.
No Pi controller command was executed by Codex.

The operator now reports that the remote may have been switched off before
previous tests. Treat its earlier powered/connected status as uncertain. This
could explain missing remote fields; it does not establish why experiment
feedback was approximately 12 Hz. Keep the remote powered and connected for
the entire corrected block. Re-run the same Ubuntu command in this section;
no deployment, cleanup or additional software simulation is needed first.

The shortest remaining path is: resolve communication and stop-input delivery;
repeat the brief prone engagement/release; then prepare a bounded small-torque
trial with its mechanical setup. More complex motion follows hardware evidence,
not completion of another software checklist. A completion time cannot be
promised until the communication fault is isolated.

### 2.1.24 Remote input confirmed; synchronized damping-only feedback diagnostic

**Factory remote input confirmed, 2026-09-21.** Archive
`logs/remote-input-diagnostic/review-ixy7fnv2`, SHA-256
`e194bd7031775e57caa6e641dcee63763a83341bc584f060d993b3653b10b233`.
All 19,931 native states passed CRC and carried 5551 remote headers.
Recorded input was neutral -> L2 -> L2+B -> L2 -> neutral; L2+B appeared
in 1,401 samples, from 7.037 to 8.438 seconds after capture start. Input was
later than the nominal five-second cue, but a complete press/release was
captured. State rate was approximately 1,000 Hz, maximum gap 2.851 ms;
9,966 commands arrived at approximately 500 Hz. All joints retained Kp=0,
Kd=2, zero feed-forward torque. Peak measured speed was 0.03725 rad/s.
There were zero kernel drops (29,897 saved, 29,943 received by filter).
This is evidence of input delivery, not a test of a custom controller's stop
response. The remote-off explanation for previous zero data is plausible;
the earlier experiment-port feedback rate remains unverified.

**Next block:** a single 15-second `remote-preflight` on UDP 8090 with packet
capture spanning factory and experiment traffic. It actively sends Kp=0,
Kd=1, zero feed-forward torque to all joints. This changes damping and can
change joint loading. Keep belly and all feet floor-supported, remote on and
connected, sticks centered and everyone clear. No standing/rise/torque test
is authorized by this diagnostic. In this diagnostic mode L2+B is recorded,
not used as an exit trigger; unexpected motion/sound means press Ctrl-C in
Terminal A and allow final damping/exit. Do not start another trial.

#### A. Ubuntu Terminal A — deploy the already-tested send-status correction

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
bash experiment/prepare_go1_prone_engagement.sh
ssh -t pi@192.168.12.1
```

The deployment must report PASS. Now in the **Pi shell**, inspect the owners:

```bash
cd /home/pi/go1-prone-engagement
ip route get 192.168.123.10
pgrep -af 'go1_lowlevel_experiment|example_|run_torque_tracking' || true
PROGRAMMING_PATTERN='^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
pgrep -af "$PROGRAMMING_PATTERN"
sudo ss -Huanp | awk '$4 ~ /:8090$/ { print }'
sudo fuser -v 8090/udp
```

Require the established eth0 route via 192.168.123.161, no custom controller,
and exactly one Programming Module whose PID owns 8090 connected to 8082.
If these differ, stop and send the output. Leave Legged_sport running.
Reconfirm the supported prone posture and factory L2+B damping before the
following block. It stops only that one Programming Module, verifies the
port is free, and launches the diagnostic to its arm prompt:

```bash
(
  set -e
  test ! -e logs/feedback_path_01.csv
  test ! -e logs/feedback_path_01.pcap
  mapfile -t PIDS < <(pgrep -f "$PROGRAMMING_PATTERN")
  [ "${#PIDS[@]}" -eq 1 ] || { echo 'STOP: unexpected module count'; exit 1; }
  kill -TERM "${PIDS[0]}"
  sleep 2
  if pgrep -f "$PROGRAMMING_PATTERN"; then
    echo 'STOP: module returned; do not kill again'; exit 1
  fi
  python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(('0.0.0.0',8090))
print('FREE: UDP 8090')
PY
  ./build/go1_lowlevel_experiment --mode remote-preflight \
    --local-port 8090 --prone-confirmed --duration-s 15 \
    --log logs/feedback_path_01.csv
)
```

**Wait at `ARM DAMPING`; do not arm until Terminal B says `listening`.**
If this block fails/cancels after stopping the module, restore it using C below.
No support GUI or loopback tunnel is used.

#### B. Ubuntu Terminal B — simultaneous capture, then arm A

```bash
ssh -t pi@192.168.12.1 'bash -c '\''
cd /home/pi/go1-prone-engagement || exit 1
test ! -e logs/feedback_path_01.pcap || { echo "STOP: capture exists"; exit 1; }
sudo timeout -s INT 60 tcpdump -i any -nn -s 0 -U \
  -w logs/feedback_path_01.pcap \
  "ip and udp and (port 8007 or port 8008 or port 8082 or port 8090)"
status=$?
if [ "$status" -ne 0 ] && [ "$status" -ne 124 ]; then exit "$status"; fi
sudo chmod a+r logs/feedback_path_01.pcap
sha256sum logs/feedback_path_01.pcap
'\'''
```

Once tcpdump prints `listening`, return to Terminal A, type `ARM DAMPING`,
and press Enter promptly (within 20 seconds). About five seconds into the run,
hold L2+B together for two seconds, then release; keep sticks centered. The
15-second controller exits automatically, possibly after final damping. Record
its console output and any physical observation. Capture continues for 60
seconds to include restoration; let it finish. Do not use the native factory
only decoder on this multi-interface/multi-port capture.

#### C. Ubuntu — restore immediately after controller exit using explicit SSH

Once the controller has returned to the Pi shell, run this from **Ubuntu**,
even if the trial reported a fault or you cancelled arming:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
bash experiment/restore_go1_programming.sh
```

This helper explicitly connects to the Pi, defines its own nonempty process
pattern, refuses an active experiment or duplicate modules, and starts only
the known vendor wrapper if the module is absent. It does not kill a process.
Require exactly one module and its PID owning the usual UDP 8090 -> 8082
connection in the printed output. Preserve any failure output and keep the
robot prone. Do not retry the controller or start engagement. Both the local
helper and its embedded remote block passed Bash syntax checks; actual Pi
restoration must be confirmed from operator output.

#### D. Ubuntu — archive after A exits and B capture finishes

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
mkdir -p logs/feedback-path
GO1_PATH_REVIEW=$(mktemp -d "$PWD/logs/feedback-path/review-XXXXXXXX")
scp pi@192.168.12.1:/home/pi/go1-prone-engagement/logs/feedback_path_01.csv \
    pi@192.168.12.1:/home/pi/go1-prone-engagement/logs/feedback_path_01.pcap \
    "$GO1_PATH_REVIEW/"
sha256sum "$GO1_PATH_REVIEW"/feedback_path_01.*
python3 -B experiment/analyze_lowlevel_log.py \
  "$GO1_PATH_REVIEW/feedback_path_01.csv" --no-plots
printf 'archive=%s\n' "$GO1_PATH_REVIEW"
```

Compare hashes with the two Pi outputs. Send the archive path, A's controller
and restoration output, B's capture totals, and physical observations. Codex
will compare native 8008 traffic, experiment 8090 traffic and CSV receive
freshness during the same interval. If wire feedback is healthy but CSV remains
slow, investigate the SDK/receive path; if wire feedback to 8090 is sparse,
investigate delivery/competing destinations. Review remote press/release on
8090 as well as state flags, gaps, sends and final damping. Neither an exit
code nor the generic analyzer's PASS alone authorizes the engagement retry.

### 2.1.25 Synchronized diagnostic result — restore Programming Module first

Archive: `logs/feedback-path/review-7X0qmEAc`. The controller completed 7,511
samples, closed UDP, and reported no fault. Deployment executable SHA-256:
`4629d3cd50db3a457113f5ef46f5553846be863d9ac7219c51549fcf178d8ded`.
PCAP SHA-256 matches the Pi output:
`2d1b02e8e02051f731e99ee2e47f053c1fadbba1e49886c159c1e454695e773f`.
Local CSV SHA-256:
`1f8cb27db0d499079d1fae78af67f5a11894967fed002356bf246e9ba81b7c53`;
the submitted transcript contains no Pi CSV hash comparison yet.

Codex ran `analyze_lowlevel_log.py --no-plots` on this archive and saved the
reproducible packet review in `review_capture.py` and `capture-review.json`
inside it. Summary:

| Measure | Result |
| --- | --- |
| CSV fresh states | 5,448; 359.25 Hz; p99 gap 4.942 ms; maximum gap 10.175 ms |
| Remote validity / low-level validity | 100% of fresh samples; L2+B recorded |
| Faults / watchdog events / fresh tick gaps >20 ms | None |
| CSV command fields | All joints Kp=0, Kd=1, feed-forward torque=0 |
| SDK send results | 614 bytes for all logged samples |
| Experiment outgoing packets | Approximately 500 Hz; maximum gap 2.423 ms |
| Wire feedback to experiment 8090 during command window | Approximately 581 Hz; maximum gap 3.226 ms |
| CSV ticks absent from capture | 0 |
| Maximum joint speed / temperature | 0.0371 rad/s / 55 C |

All 92,539 saved packets passed the appropriate CRC check. Factory commands
use the previously identified transformed CRC; experiment SDK commands use
the standard SDK CRC. Applying the factory transform to experiment commands
would falsely flag every packet. The capture reports 92,936 packets received
by filter and zero kernel drops; these counts do not imply all were saved.
Wire and CSV fresh rates measure different stages; their difference alone is
not evidence of packet loss. The previous 12 Hz/170 ms-gap condition did not
recur. Concurrent factory 8008 commands remain visible (about 458 Hz during
this interval); this diagnostic does not prove exclusive motor command
ownership and is not torque-tracking acceptance. Do not stop Legged_sport
based on this observation.

**Restoration was attempted in the wrong shell.** The second transcript shows
`aims@aims-Precision-7780`, not `pi@raspberrypi`. An unset
`PROGRAMMING_PATTERN` made `pgrep` match unrelated Ubuntu processes, producing
the misleading message that the module was already running. The Pi module's
restoration is therefore unconfirmed. Keep the robot prone. Run from Ubuntu:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
bash experiment/restore_go1_programming.sh
```

Send the module PID and socket output. Require exactly one Programming Module
and that PID owning UDP 8090 connected to 192.168.123.161:8082. The helper
refuses to start it while the named custom controller remains active. No new
controller trial is required to correct restoration. No remote restoration
was performed by Codex. Physical motion/sound observations for this run were
not provided; telemetry is not a substitute for those observations.

### 2.1.26 One prone-engagement retry after recovered communication

**Historical failed run; use 2.1.28 for the corrected retry.**

**Restoration confirmed by operator, 2026-09-21:** Programming Module PID
10316 owns 192.168.123.161:8090 -> 192.168.123.161:8082. The operator reports
no movement or unusual sound in the preceding damping diagnostic. No further
restoration or repeat preflight is required before this block.

Offline review of all 25,264 concurrent factory command packets found Kp=0,
feed-forward torque=0 and Kd in {0,2}; saved as
`logs/feedback-path/review-7X0qmEAc/factory-command-review.json`. This establishes
the observed neutral/damping fields, not firmware arbitration or exclusive
ownership. Keep the factory services unchanged. Any later torque acceptance
must account for this concurrent stream.

The earlier blanket hold on repeating 2.1.19 is superseded **only for this
single bounded retry**, using the newly deployed binary and recovered remote
feedback. This does not authorize low-rise, standing or torque-sine modes.
The engagement commands all joints with Kp<=1, Kd=1, zero feed-forward torque
and fixed targets. Calf target clamping can move/load the robot. The controller
must independently pass its unchanged quiet-feedback and remote checks before
engaging; do not bypass a failed check. The 500-fresh-state observation may take
longer than one second at the measured 359 Hz.

Keep belly and all four feet supported, legs unobstructed, remote on and
connected, sticks centered, and everyone clear. Set the already-prone robot
to its established factory L2+B damping state, then release both buttons before
starting. If posture/contact is uncertain, do not arm. No intentional fault
injection in this first completed engagement trial.

#### A. Ubuntu Terminal A — SSH, verify deployed binary and ownership

```bash
ssh -t pi@192.168.12.1
```

Now run in the **Pi shell**:

```bash
cd /home/pi/go1-prone-engagement
printf '%s\n' '4629d3cd50db3a457113f5ef46f5553846be863d9ac7219c51549fcf178d8ded  build/go1_lowlevel_experiment' | sha256sum -c -
ip route get 192.168.123.10
pgrep -af 'go1_lowlevel_experiment|example_|run_torque_tracking' || true
PROGRAMMING_PATTERN='^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
pgrep -af "$PROGRAMMING_PATTERN"
sudo ss -Huanp | awk '$4 ~ /:8090$/ { print }'
sudo fuser -v 8090/udp
```

Require binary `OK`, the known eth0 route, no custom controller, and one
Programming Module whose PID owns the shown 8090 -> 8082 connection. PID may
change; never reuse 10316 as a kill target. If any check differs, stop.

#### B. Ubuntu Terminal B — keep the confirmation tunnel open

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18092:127.0.0.1:18092 pi@192.168.12.1
```

Silence after authentication is normal. If the port is occupied, check the
existing tunnel rather than launching another probe or killing unknown owners.

#### C. Pi Terminal A — release the port and start exactly one trial

With supported posture reconfirmed, run this in the same **Pi shell**:

```bash
(
  set -e
  test ! -e logs/prone_engagement_02.csv
  PROGRAMMING_PATTERN='^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
  mapfile -t PIDS < <(pgrep -f "$PROGRAMMING_PATTERN")
  [ "${#PIDS[@]}" -eq 1 ] || { echo 'STOP: unexpected module count'; exit 1; }
  kill -TERM "${PIDS[0]}"
  sleep 2
  if pgrep -f "$PROGRAMMING_PATTERN"; then
    echo 'STOP: module returned; do not kill again'; exit 1
  fi
  python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(('0.0.0.0',8090))
print('FREE: UDP 8090')
PY
  ./build/go1_lowlevel_experiment --mode prone-engagement \
    --prone-confirmed --remote-confirmed --local-port 8090 \
    --log logs/prone_engagement_02.csv
)
```

Type `ARM`, then **wait at the second prompt** that says the support receiver
is ready and no motor packets have been sent. If the block fails after stopping
the Programming Module, restore it using E; do not automatically retry.

#### D. Ubuntu Terminal C — GUI, then begin and confirm contact

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 -u experiment/operator_support_gate.py sender
```

After the GUI opens, with posture and remote readiness confirmed, press Enter
in A to begin. Return focus to the GUI without clicking yet; keep A visible.
Expected sequence:

`PRONE_OBSERVE -> PRONE_ENGAGE -> PRONE_ENGAGE_HOLD -> PRONE_SETTLE -> PRONE_RELEASE -> PRONE_FINAL_DAMPING -> COMPLETE`.

At **CONFIRM CONTACT NOW**, click once only if belly and all four feet remain
supported and the robot looks/sounds normal. Keep GUI focus for the 1.5-second
pulse. The one-second confirmed dwell authorizes the full release; do not
repeatedly click. Normal duration is roughly 12–17 seconds, depending on
observation and confirmation.

Unexpected motion, lifting, slipping, lost contact or abnormal sound: use
**L2+B** and keep clear; panic should finish its damping window and close.
Do not click confirmation to force progress. Single Ctrl-C requests normal
cancellation, not immediate exit. If `PRONE_SUPPORT_HOLD` appears, position
control is still active. Only confirm if contact is genuinely supported; if
uncertain or a fresh confirmation does not start release after its dwell, use
L2+B. Do not close SSH or abandon a holding controller. Wait for UDP closure
and return to the Pi shell before restoration. No second trial in this block.

#### E. Ubuntu — restore and archive after controller exit

Close the GUI; stop B's tunnel with Ctrl-C. In **Ubuntu Terminal C**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
bash experiment/restore_go1_programming.sh
```

Require one module and its matching 8090 -> 8082 socket. If abnormal, keep the
robot prone and report it. Then archive from **Ubuntu**:

```bash
mkdir -p logs/prone-engagement-hardware
GO1_ENGAGEMENT_REVIEW=$(mktemp -d "$PWD/logs/prone-engagement-hardware/review-XXXXXXXX")
ssh pi@192.168.12.1 'cd /home/pi/go1-prone-engagement/logs && sha256sum prone_engagement_02.csv' \
  > "$GO1_ENGAGEMENT_REVIEW/pi.sha256"
scp pi@192.168.12.1:/home/pi/go1-prone-engagement/logs/prone_engagement_02.csv \
  "$GO1_ENGAGEMENT_REVIEW/"
(cd "$GO1_ENGAGEMENT_REVIEW" && sha256sum -c pi.sha256)
python3 -B experiment/analyze_lowlevel_log.py \
  "$GO1_ENGAGEMENT_REVIEW/prone_engagement_02.csv" --no-plots
printf 'archive=%s\n' "$GO1_ENGAGEMENT_REVIEW"
```

Send A's phase/completion output, restoration output, archive path and actual
movement/sound observations. Review requires the full engagement/release chain,
no faults, bounded effort/motion, valid feedback, and final damping. A general
analyzer summary alone does not establish engagement acceptance. Do not
proceed to a rise or torque waveform until this physical result is reviewed.

### 2.1.27 Engagement retry failed — settling timer corrected offline

Archive `logs/prone-engagement-hardware/review-XkFFPRcI` verified against the
Pi SHA-256 `94e12643128df16538568305fd42d7f5353eb11262cf3d26f9e4f58472ea9555`.
Restoration is confirmed: PID 8228 owns 8090 -> 8082. Operator hesitated about
which control to use, then clicked; no unusual movement/sound was observed.
GUI closure was orderly. The run is **not accepted**.

The controller reached engagement at 1.004 s and engagement hold at 5.012 s.
At 8.016 s it failed with `prone_engagement_not_settled`, returned, and entered
fault support hold at 12.030 s. It never reached release. At 81.654 s a feedback
gap triggered panic damping; completion followed at 82.156 s. These are distinct
failures: the late feedback interruption did not cause the first settling fault.
The long log was mostly fault hold, not a successful long-duration trial.

**Confirmed software defect:** all joint position/speed settling conditions
passed during the 1,501 hold cycles. Five cycles reused a still-live state;
`updateProneStable` incorrectly reset its dwell for each. The longest fresh
streak was 461 cycles, or 0.922 s, insufficient for the required second.
The timer now preserves credit on non-fresh live cycles but adds no credit.
Fresh unsettled samples reset it, and stale feedback still resets/triggers the
existing guard. Position, speed, remote, effort, and 20 ms timeout limits are
unchanged. Overall telemetry: 493.09 Hz fresh rate, 24.008 ms maximum gap, one
watchdog cycle and one fresh tick gap above 20 ms. The isolated late scheduling/
feedback interruption remains a real fault; this timer correction does not
claim to fix it. The reported torque correlations are not tracking acceptance
for a fixed-target engagement.

**Prompt defect:** the old hardware loop printed CONFIRM CONTACT even for an
unrecoverable fault. The new thread-safe status flag permits that prompt only
for contact-recoverable states. Other settle/hold faults print FAULT HOLD and
instruct L2+B for final damping/exit. A late click did not cause the initial
failure and could not clear it. Do not wait in fault hold on the next run.

A regression with intermittent feedback reproduced failure before the fix and
passes afterward, including a fresh speed disturbance that must reset dwell.
All ten core test groups and normal/cancel/remote-stop/watchdog recovery
simulations passed in `logs/prone-recovery-software/review-1igs0jv_/`. Hardware
source compilation with GO1_WITH_SDK also passed. No corrected binary has yet
been deployed to the Pi by Codex. Detailed phase evidence is saved in
`engagement-review.json` beside the hardware CSV.

### 2.1.28 Corrected bounded engagement retry — new deployment required

**Deployment completed; see 2.1.29 to restore the missing module before resuming. Do not redeploy for that guard failure.**

This replaces 2.1.26 for execution; do not rerun the old Pi binary. One retry
only, same Kp<=1, Kd=1, zero feed-forward torque and fixed targets; no rise or
torque waveform. Concurrent factory traffic still limits torque acceptance.
First run on **Ubuntu**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
bash experiment/prepare_go1_prone_engagement.sh
```

Require deployment PASS and retain its binary hash. Then follow the complete
terminal sequence below. No repeat simulation or damping-only preflight is
needed. Log `prone_engagement_03.csv` must be new; preserve existing trials.

Keep belly and all four feet supported, legs unobstructed, remote on and
connected, sticks centered, and everyone clear. Set the already-prone robot
to its established factory L2+B damping state, then release both buttons before
starting. If posture/contact is uncertain, do not arm. No intentional fault
injection in this first completed engagement trial.

#### A. Ubuntu Terminal A — SSH, verify deployed binary and ownership

```bash
ssh -t pi@192.168.12.1
```

Now run in the **Pi shell**:

```bash
cd /home/pi/go1-prone-engagement
sha256sum build/go1_lowlevel_experiment
ip route get 192.168.123.10
pgrep -af 'go1_lowlevel_experiment|example_|run_torque_tracking' || true
PROGRAMMING_PATTERN='^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
pgrep -af "$PROGRAMMING_PATTERN"
sudo ss -Huanp | awk '$4 ~ /:8090$/ { print }'
sudo fuser -v 8090/udp
```

Require the binary hash to match the just-completed deployment output, the known eth0 route, no custom controller, and one
Programming Module whose PID owns the shown 8090 -> 8082 connection. PID may
change; never reuse 10316 as a kill target. If any check differs, stop.

#### B. Ubuntu Terminal B — keep the confirmation tunnel open

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18092:127.0.0.1:18092 pi@192.168.12.1
```

Silence after authentication is normal. If the port is occupied, check the
existing tunnel rather than launching another probe or killing unknown owners.

#### C. Pi Terminal A — release the port and start exactly one trial

With supported posture reconfirmed, run this in the same **Pi shell**:

```bash
(
  set -e
  test ! -e logs/prone_engagement_03.csv
  PROGRAMMING_PATTERN='^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
  mapfile -t PIDS < <(pgrep -f "$PROGRAMMING_PATTERN")
  [ "${#PIDS[@]}" -eq 1 ] || { echo 'STOP: unexpected module count'; exit 1; }
  kill -TERM "${PIDS[0]}"
  sleep 2
  if pgrep -f "$PROGRAMMING_PATTERN"; then
    echo 'STOP: module returned; do not kill again'; exit 1
  fi
  python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(('0.0.0.0',8090))
print('FREE: UDP 8090')
PY
  ./build/go1_lowlevel_experiment --mode prone-engagement \
    --prone-confirmed --remote-confirmed --local-port 8090 \
    --log logs/prone_engagement_03.csv
)
```

Type `ARM`, then **wait at the second prompt** that says the support receiver
is ready and no motor packets have been sent. If the block fails after stopping
the Programming Module, restore it using E; do not automatically retry.

#### D. Ubuntu Terminal C — GUI, then begin and confirm contact

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 -u experiment/operator_support_gate.py sender
```

After the GUI opens, with posture and remote readiness confirmed, press Enter
in A to begin. Return focus to the GUI without clicking yet; keep A visible.
Expected sequence:

`PRONE_OBSERVE -> PRONE_ENGAGE -> PRONE_ENGAGE_HOLD -> PRONE_SETTLE -> PRONE_RELEASE -> PRONE_FINAL_DAMPING -> COMPLETE`.

At **CONFIRM CONTACT NOW**, click the GUI button labelled **Click once after visually confirming belly contact** once. This is a mouse click, not a keyboard key. Click only if belly and all four feet remain
supported and the robot looks/sounds normal. Keep GUI focus for the 1.5-second
pulse. The one-second confirmed dwell authorizes the full release; do not
repeatedly click. Normal duration is roughly 12–17 seconds, depending on
observation and confirmation.

Unexpected motion, lifting, slipping, lost contact or abnormal sound: use
**L2+B** and keep clear; panic should finish its damping window and close.
Do not click confirmation to force progress. Single Ctrl-C requests normal
cancellation, not immediate exit. If **FAULT HOLD** appears, the fault cannot be cleared by clicking: use L2+B and allow final damping/exit. If a recoverable `PRONE_SUPPORT_HOLD` appears, position
control is still active. Only confirm if contact is genuinely supported; if
uncertain or a fresh confirmation does not start release after its dwell, use
L2+B. Do not close SSH or abandon a holding controller. Wait for UDP closure
and return to the Pi shell before restoration. No second trial in this block.

#### E. Ubuntu — restore and archive after controller exit

Close the GUI; stop B's tunnel with Ctrl-C. In **Ubuntu Terminal C**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
bash experiment/restore_go1_programming.sh
```

Require one module and its matching 8090 -> 8082 socket. If abnormal, keep the
robot prone and report it. Then archive from **Ubuntu**:

```bash
mkdir -p logs/prone-engagement-hardware
GO1_ENGAGEMENT_REVIEW=$(mktemp -d "$PWD/logs/prone-engagement-hardware/review-XXXXXXXX")
ssh pi@192.168.12.1 'cd /home/pi/go1-prone-engagement/logs && sha256sum prone_engagement_03.csv' \
  > "$GO1_ENGAGEMENT_REVIEW/pi.sha256"
scp pi@192.168.12.1:/home/pi/go1-prone-engagement/logs/prone_engagement_03.csv \
  "$GO1_ENGAGEMENT_REVIEW/"
(cd "$GO1_ENGAGEMENT_REVIEW" && sha256sum -c pi.sha256)
python3 -B experiment/analyze_lowlevel_log.py \
  "$GO1_ENGAGEMENT_REVIEW/prone_engagement_03.csv" --no-plots
printf 'archive=%s\n' "$GO1_ENGAGEMENT_REVIEW"
```

Send A's phase/completion output, restoration output, archive path and actual
movement/sound observations. Review requires the full engagement/release chain,
no faults, bounded effort/motion, valid feedback, and final damping. A general
analyzer summary alone does not establish engagement acceptance. Do not
proceed to a rise or torque waveform until this physical result is reviewed.

### 2.1.29 Corrected binary deployed; launch blocked before arming

Operator deployment `logs/engagement-deployment/review-YkzqfG0X` succeeded.
The Pi binary hash matches the deployment output:
`4fdb05d3a3fe003368aba54e27afd95f6792c63f5975b74227e3b7b75f9b9c1b`.
The route and absence of custom controllers were checked. However,
`programming.py` was absent and UDP 8090 had no reported owner. Both attempts
stopped at the one-module guard, before TERM, UDP construction or ARM.
No `prone_engagement_03` controller trial occurred. Do not bypass the guard
or count this as a failed motor trial.

The prior restoration proved PID/socket presence only inside the restoring
SSH session. Its subsequent disappearance is consistent with terminal hangup,
but the exact Pi exit cause is not established. The helper now starts the
existing vendor wrapper with `nohup`, stdin from `/dev/null`, and stdout/stderr
redirected to a unique Pi `programming-restore-*.log`. The factory wrapper and
services are unchanged. It never restarts an already-present module or kills
an unknown process.

It verifies the exact module PID and expected 8090 -> 8082 socket, closes
that SSH session, then uses a new read-only SSH session to require the **same
PID and socket**. Failure in that second check stops without silently restarting
anything. Local transcripts are retained under `logs/programming-restoration`.
Four local tests passed: correct PID/socket, missing/changed PID, duplicate/
wrong socket, and a harmless mock background child surviving shell exit plus
SIGHUP. These are helper tests, not a claim of verified Pi persistence yet.

Run from **Ubuntu** (a separate Ubuntu terminal if A remains in the Pi shell):

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
bash experiment/restore_go1_programming.sh
```

It may ask for the Pi password twice. Require the final line:

```text
programming_restoration=PASS; same PID and socket verified after SSH logout
```

If PASS appears, resume 2.1.28 at A's Pi ownership checks, then B–E. Keep the
same new log `prone_engagement_03.csv`; neither blocked attempt started the
controller. Do not rebuild/redeploy or repeat earlier simulations. Keep the
robot supported and remote connected as required there. If either restoration
check fails, send the concise output and restoration archive; do not launch
another trial. This verifies persistence across the SSH session, not permanent
service supervision.

### 2.1.30 Normal prone engagement and release accepted

**Accepted for this bounded profile, 2026-09-21.** The operator confirmed belly
and all four feet remained supported, with no unexpected movement or unusual
sound. The corrected binary completed the normal sequence. Programming Module
PID 26678 and its expected socket were verified after SSH logout; archive
`logs/programming-restoration/review-rdolry2W`.

Hardware archive: `logs/prone-engagement-hardware/review-jl57lTcl`.
The CSV matches the Pi SHA-256
`783804de11170a8f1be771631889b14a80dba5b12c0109952ffa32d9165ded0d`.
Offline checks and results are retained as `review_engagement.py` and
`engagement-review.json` in that directory. No repeat normal trial is needed.

| Check | Observed result |
| --- | --- |
| Sequence | Observe -> engage -> hold -> settle -> release -> final damping -> complete |
| Total time / samples | 13.488 s to COMPLETE; 6,742 samples |
| Engagement hold | Completed after 1.040 s, with intermittent feedback handled correctly |
| Contact dwell | 1.00197 s before release; one authorization event |
| Commands | Kp 0..1, Kd=1, feed-forward torque=0 on all joints |
| Peak absolute predicted total command effort | 0.09394 Nm, below 0.10 Nm bound |
| Release / final damping | Kp non-increasing throughout release, then zero through final damping |
| Feedback | 472.27 fresh Hz; p99 gap 4.033 ms; max gap 9.993 ms; no >20 ms tick gap |
| Remote / low-level validity | 100% of fresh samples |
| Faults / sends | No abort/stop reasons; all recorded sends 614 bytes |
| Motion | Max joint speed 0.0375 rad/s; roll/pitch excursions 0.0014/0.0033 rad |

**The one watchdog flag is a startup event.** It is sample 0 in PRONE_OBSERVE,
with every joint already Kp=0, Kd=1, tau_ff=0. The sender starts after waiting
for the GUI; its initial publication timestamp precedes that wait, so its
startup fallback is damping. There are no watchdog flags during engagement or
release. Keep this fact in the record rather than describing the run as having
zero flags or weakening any active watchdog checks.

This validates the bounded entry/contact/release behavior. Predicted effort is
not independently measured torque. The general analyzer's correlations, gains,
and RMSE here do not establish torque tracking: this was a fixed-target trial
with zero feed-forward waveform. The support geometry NaNs belong to standing/
leg-lift analysis fields unused by this prone protocol; they are not substituted
for the operator's contact assertion. L2+B_seen=0 is expected in a normal run.
Remote stop under active engagement has not yet been tested. Previous packet
captures still show concurrent factory commands, so exclusive authority and
later torque acceptance remain separate questions.

### 2.1.31 Two-case exit verification at the accepted engagement limits

**Next physical block, after the combined deployment/check in 2.1.32:** one
ordinary cancellation, then one deliberate remote stop. The new binary includes
the selected SDK checks; the profile remains Kp<=1 and zero feed-forward torque,
with no rise, gain increase or waveform. Use this same binary for both cases;
no rebuild between them. This tests actual exit commands while position control
is active, beyond the earlier damping-only button decoding.
Do not disconnect communications or inject a watchdog failure on hardware.

Use the same **belly and all four feet on the floor** posture, remote
ON/connected, centered sticks and clear joints. Do not run this block with feet
airborne on Wenjian's belly support. Begin each case in the established factory
L2+B damping state,
then release the buttons. No physical observation is inferred from the CSV.
If contact changes or anything is abnormal, use the known L2+B stop and keep
clear. Do not proceed to the second case if the first does not restore normally.

#### A. Terminal setup — run these on Ubuntu

Terminal A:

```bash
ssh -t pi@192.168.12.1
```

In the resulting **Pi shell**, select the first case and check the binary:

```bash
cd /home/pi/go1-prone-engagement
GO1_EXIT_CASE=cancel
sha256sum -c build/go1_lowlevel_experiment.sha256
cat build/go1_lowlevel_experiment.sha256
ip route get 192.168.123.10
```

Require `build/go1_lowlevel_experiment: OK`; the printed hash must match the
just-completed 2.1.32 deployment transcript on Ubuntu. Require the established
eth0/192.168.123.161 route. The old accepted normal-run hash remains historical
evidence; it is not the identity of this newly built binary.
Terminal B on **Ubuntu** (reuse a healthy existing tunnel; do not duplicate it):

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18092:127.0.0.1:18092 pi@192.168.12.1
```

#### B. Pi Terminal A — launch the selected case

This complete block defines its own process pattern. It refuses missing or
ambiguous module ownership and existing logs before stopping any process.

```bash
(
  set -e
  case "$GO1_EXIT_CASE" in cancel|remote_stop) ;; *) echo 'STOP: select cancel or remote_stop'; exit 1 ;; esac
  GO1_CASE_LOG="logs/prone_${GO1_EXIT_CASE}_01.csv"
  test ! -e "$GO1_CASE_LOG"
  if pgrep -af '^([^ ]*/)?go1_lowlevel_experiment( |$)|^([^ ]*/)?example_[^ ]*( |$)'; then
    echo 'STOP: controller/example already running'; exit 1
  fi
  PATTERN='^[^ ]*python3 ([^ ]*/)?programming[.]py( |$)'
  mapfile -t PIDS < <(pgrep -f "$PATTERN")
  [ "${#PIDS[@]}" -eq 1 ] || { echo 'STOP: expected one Programming Module'; exit 1; }
  SOCKETS=$(sudo ss -Huanp)
  OWNER=$(awk -v pid="${PIDS[0]}" '$4 ~ /:8090$/ && $5 == "192.168.123.161:8082" && index($0,"pid=" pid ",") {print}' <<< "$SOCKETS")
  [ -n "$OWNER" ] || { echo 'STOP: unexpected port owner'; exit 1; }
  printf '%s\n' "$OWNER"
  kill -TERM "${PIDS[0]}"
  sleep 2
  if pgrep -f "$PATTERN"; then echo 'STOP: module returned; do not kill again'; exit 1; fi
  python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(('0.0.0.0',8090))
print('FREE: UDP 8090')
PY
  ./build/go1_lowlevel_experiment --mode prone-engagement \
    --prone-confirmed --remote-confirmed --local-port 8090 --log "$GO1_CASE_LOG"
)
```

Type `ARM`, then leave A at the second Enter prompt. In **Ubuntu Terminal C**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
python3 -u experiment/operator_support_gate.py sender
```

After the GUI opens, press Enter in A to start. Keep A and the GUI visible.
Do the selected case below once; then restore/download using D even if it fails.

#### C. Actions for the two cases

**Case 1 — cancel.** About two seconds after `phase=PRONE_ENGAGE` appears,
press **Ctrl-C once in Terminal A**. Do not double-press: that requests panic.
Expect PRONE_RETURN, then PRONE_SETTLE. At CONFIRM CONTACT NOW, click the GUI
button **Click once after visually confirming belly contact**, only with belly
and all four feet still supported; keep GUI focus for its 1.5-second pulse.
Expect release -> final damping -> complete and return to the shell. During
cancellation the gain must not increase; the offline review will check it.
If FAULT HOLD appears or contact is uncertain, use L2+B instead of clicking
repeatedly. Record that as a failed cancellation case and do not start case 2.

**Case 2 — remote_stop.** Only after case 1 completed/restored normally and was
archived, set `GO1_EXIT_CASE=remote_stop` in the same **Pi Terminal A**, and
repeat B (reopen the GUI in C). About two seconds into PRONE_ENGAGE, press
**L2+B together** on the factory remote. Release once PANIC_DAMPING appears.
Do not click GUI confirmation. Expect reason `remote_l2_b`, final damping and
automatic exit after its roughly half-second window. A nonzero controller exit
status is expected for this deliberately requested panic; unrelated reasons
or failure to exit are not a pass. If no response occurs promptly, use two
Ctrl-C presses within one second in A as the software fallback and report the
remote-stop test as failed. Neither mechanism is a physical emergency stop.

If either input was accidentally sent during observation, retain the log and
report that; it does not test an exit from nonzero position gain. Do not keep
repeating attempts to manufacture a passing result.

#### D. Ubuntu Terminal C — restore and archive each case before continuing

After the controller has closed UDP and returned to the Pi shell, close the
GUI and run on **Ubuntu**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
bash experiment/restore_go1_programming.sh
```

Require the post-logout PASS. Keep B's tunnel open between the two cases.
For case 1 set this in **Ubuntu Terminal C** (the Pi variable does not carry
into this shell):

```bash
GO1_EXIT_CASE=cancel
```

For case 2 use this instead:

```bash
GO1_EXIT_CASE=remote_stop
```

Then run the same archive block in **Ubuntu**:

```bash
(
set -e
case "$GO1_EXIT_CASE" in cancel|remote_stop) ;; *) echo 'Invalid case'; exit 1 ;; esac
GO1_CASE_FILE="prone_${GO1_EXIT_CASE}_01.csv"
mkdir -p logs/prone-exit-hardware
GO1_EXIT_REVIEW=$(mktemp -d "$PWD/logs/prone-exit-hardware/review-XXXXXXXX")
scp pi@192.168.12.1:/home/pi/go1-prone-engagement/manifest.json \
  pi@192.168.12.1:/home/pi/go1-prone-engagement/source.sha256 \
  pi@192.168.12.1:/home/pi/go1-prone-engagement/build/go1_lowlevel_experiment.sha256 \
  "$GO1_EXIT_REVIEW/"
ssh pi@192.168.12.1 "cd /home/pi/go1-prone-engagement/logs && sha256sum $GO1_CASE_FILE" \
  > "$GO1_EXIT_REVIEW/pi.sha256"
scp "pi@192.168.12.1:/home/pi/go1-prone-engagement/logs/$GO1_CASE_FILE" "$GO1_EXIT_REVIEW/"
(cd "$GO1_EXIT_REVIEW" && sha256sum -c pi.sha256)
python3 -B experiment/analyze_lowlevel_log.py "$GO1_EXIT_REVIEW/$GO1_CASE_FILE" --no-plots
printf 'case=%s archive=%s\n' "$GO1_EXIT_CASE" "$GO1_EXIT_REVIEW"
)
```

Keep any failed-transfer log on the Pi and report the failure rather than
launching another case. After both, close the GUI and B's tunnel. Send both
archive paths, console phase/stop output, restoration PASS records and physical
observations. Codex will check cancellation gain monotonicity, the remote chord
on fresh valid feedback, immediate damping publication after detection,
unchanged bounds and completion. CSV detection-to-command latency does not
measure physical-button-to-motor latency. Neither test establishes torque
tracking; the next torque profile and command-ownership review remain separate.

### 2.1.32 Selected integration and reduced next-test sequence

**Ubuntu completed, 2026-09-22.** The operator selected Wenjian items **1 + 3 +
7**: SDK receive/send checks, offline factory commanded-effort analysis, and
deployment/evidence packaging. Implementation, exact verification commands,
results and limitations are in [the selected-integration report](GO1_SELECTED_INTEGRATION.md).
Archive: `logs/integration-review/review-9HRx4VVD`. All 30 CTests and 14 Python
tests passed (including two deployment-helper subcases with mocked SSH/rsync).
The SDK command adapter also passed on Ubuntu without constructing UDP.
No Pi connection or robot command was made for this integration.

The supported-hold controller, factory process handover, timing-policy changes
and walking-policy deployment were not selected. Existing policy development
sources are preserved behind `GO1_ENABLE_POLICY_DEVELOPMENT=ON`; the selected
bundle and both deployment helpers explicitly build with it OFF. Hardware
walking remains locked even in the opt-in development build. These changes do
not increase gains, torque caps or watchdog deadlines.

| Work block | Do it now? |
| --- | --- |
| Old support probe, remote button decoding, torque matrix and recovery simulations | No; retain completed evidence. Changed code was checked locally. |
| Normal prone engagement/release | No additional standalone run; 2.1.30 is accepted. Both pending exit cases exercise entry with the new binary. |
| Another passive factory capture solely for effort analysis | No; both existing archives have been analyzed. Historical samples cannot seed a future live takeover. |
| Standalone `run_go1_pi_adapter_check.sh` | Optional diagnostic only; the combined preparation below includes its checks. |
| Combined Pi SDK checks and controller build | Once for this changed source; command below. |
| Single-Ctrl-C and L2+B exits during active prone engagement | Still required once each, in 2.1.31, with the specified grounded posture. |
| Hardware disconnect/watchdog injection, low-rise, standing, walking, torque waveform | Deferred; not released by these results. |

**Next operator action — Ubuntu, one command block.** A Pi connection and ARM
build require the operator. No GUI or support tunnel is needed for this step.
Run between trials, with no custom controller executing:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
bash experiment/prepare_go1_prone_engagement.sh
```

This freezes 38 selected source/SDK files and a manifest, transfers that
snapshot, verifies checksums on the Pi, builds the two SDK tests and controller,
and executes only the tests. It does not instantiate motor UDP in those tests,
start a controller, or stop/restart the Programming Module. Both SDK library
architectures are retained in the bundle; CMake selects the target architecture.
It records the new binary SHA-256 on the Pi and in the Ubuntu build transcript.

Require the source checksum checks, transport test and SDK adapter checks to
pass, ending with:

```text
engagement_deployment=PASS; no controller started
```

Keep the printed `deployment_archive` under `logs/engagement-deployment/`.
Then use 2.1.31 A–D for the two physical cases, only if its grounded posture
holds. No repeat normal-engagement trial or separate SDK staging directory is
needed. If the robot is currently supported with feet airborne, this preparation
can finish but the prone cases remain deferred; it is not a supported-hold
launch. Record each case's physical observation and restore the factory module
as specified. A successful ARM build is not hardware torque-tracking acceptance.

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
