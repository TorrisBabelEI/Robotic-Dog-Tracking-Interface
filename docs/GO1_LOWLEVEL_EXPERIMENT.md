# Go1 low-level experiment: step-by-step runbook

This is an operating checklist, not a design note. Follow the numbered steps
in order. Do not skip a gate, and stop whenever a required result is missing.

## Read this before starting

The only currently authorized hardware task is one 60-second, prone
`remote-preflight` run using the standard Unitree low-level UDP path:

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

Changing this experiment to an arbitrary free source port is not a valid
workaround. A preflight from local 8092 returned only about `0.54 Hz`, whereas
the SDK's standard low-level path uses local 8090. The procedure therefore
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

## Step 1 — Push the revision from the development computer

Skip this step only if the revision containing this document and the current
experiment source is already on GitHub.

Run from the repository root on the **development computer**:

```bash
git status --short
git add CMakeLists.txt \
  src/go1_lowlevel_experiment.cpp \
  src/go1_kinematics.cpp \
  src/go1_kinematics.hpp \
  test/go1_kinematics_test.cpp \
  docs/GO1_LOWLEVEL_EXPERIMENT.md
git diff --cached --check
git diff --cached --stat
git commit -m "Document standard-port onboard preflight"
git push origin main
```

Review the staged files before committing. Do not add raw CSVs, plots, or build
directories.

## Step 2 — Power Go1, put it prone, and connect Ubuntu

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

## Step 3 — Update the Ubuntu checkout

Run on **Ubuntu**:

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git status --short
git pull --ff-only
git rev-parse --short HEAD
```

`git status --short` must be empty before pulling. If it is not empty, preserve
or resolve those changes; do not discard them just to force the pull.

## Step 4 — Copy only build inputs from Ubuntu to the Pi

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

This intentionally does not copy `experiment/`, analysis summaries, plots,
historical CSVs, Git history, or workstation build directories. The Python
analyzer remains on Ubuntu.

## Step 5 — Open one Pi SSH session and inspect storage

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

## Step 6 — Build and run all software tests on the Pi

Run on the **Pi**:

```bash
cd ~/Robotic-Dog-Tracking-Interface
cmake -S . -B build-arm64 \
  -DPYTHON_BUILD=OFF -DBUILD_SDK_EXAMPLES=OFF
cmake --build build-arm64 --target \
  go1_lowlevel_experiment go1_kinematics_test -j2
(cd build-arm64 && ctest --output-on-failure)
mkdir -p logs
find build-arm64 -maxdepth 1 -type f -name 'go1_dry_*.csv' \
  -print -delete
```

All 17 tests must pass. The parenthesized `cd` form is intentional: the older
CTest on the Pi does not support `ctest --test-dir` and can otherwise report
`No tests were found`. The final command deletes only test-generated dry-run
CSVs.

## Step 7 — Check the low-level route and controller processes

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

## Step 8 — Reconfirm the prone damping state

The robot should have remained prone throughout deployment and building.
Immediately before changing the port owner, visually confirm all of the
following again:

1. Go1's abdomen is fully floor-supported.
2. All four legs are folded normally and unobstructed.
3. The factory remote is on.
4. `L2+B` has placed the prone robot in damping.
5. Everyone is clear of the legs.

Do not proceed if Go1 is standing or its state is uncertain.

## Step 9 — Temporarily stop only Unitree programming.py

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

## Step 10 — Run exactly one prone preflight on standard port 8090

Run on the **Pi**:

```bash
cd ~/Robotic-Dog-Tracking-Interface
if [ -e logs/remote_preflight_fix_01.csv ]; then
  echo 'STOP: logs/remote_preflight_fix_01.csv already exists'
elif pgrep -f "$PROGRAMMING_PATTERN" >/dev/null; then
  echo 'STOP: programming.py has reclaimed UDP 8090'
else
  ./build-arm64/go1_lowlevel_experiment --mode remote-preflight \
    --local-port 8090 \
    --prone-confirmed --duration-s 60 \
    --log logs/remote_preflight_fix_01.csv
fi
```

If the file already exists, preserve it or choose a new unique name and use
that same name in all later steps. Do not overwrite a previous hardware log.

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

## Step 11 — Restore the Unitree Programming Module immediately

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

## Step 12 — Copy and analyze the raw CSV on Ubuntu

Run on **Ubuntu**:

```bash
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

The Pi and Ubuntu SHA-256 values must match. The analyzer runs only on Ubuntu.

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

## Step 13 — Delete only the verified Pi copy

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

## Step 14 — Shut down and end the experiment

Confirm Go1 remains fully prone and floor-supported. Shut it down with the
normal battery shutdown procedure. Do not power it off while standing.

After one passing run, the prone preflight gate is complete. Do not run it
again unless code affecting UDP reception, state freshness, remote decoding,
damping, watchdog, or panic handling changes.

There is no standing hardware step in this runbook yet. The next development
task is software-only implementation and dry-run review of a smooth
lie-down-and-exit path. Add and review that path before enabling
`ground-handover` or later actions.

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
