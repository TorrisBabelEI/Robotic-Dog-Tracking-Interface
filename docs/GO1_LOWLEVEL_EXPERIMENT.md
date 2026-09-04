# Go1 low-level experiment: onboard Raspberry Pi runbook

This document is the single operating procedure for the current torque-channel
and basic-motion experiments. Historical Ubuntu-direct tests are recorded only
in the appendix and are **not** the current deployment method.

## Current decision

All hardware modes of `go1_lowlevel_experiment` run on the Raspberry Pi inside
Go1. The external Ubuntu workstation is only the operator and analysis machine.

| Function | Machine |
| --- | --- |
| `remote-preflight` | Go1 Raspberry Pi |
| `ground-handover` | Go1 Raspberry Pi |
| `squat`, `leg-lift`, `leg-lift-sequence` | Go1 Raspberry Pi |
| 500 Hz motor command, watchdog, and safety checks | Go1 Raspberry Pi |
| SSH, source synchronization, and log download | Ubuntu workstation |
| Python log analysis | Ubuntu workstation |
| Qualisys and future MPPI | Ubuntu workstation |

The present data support this decision:

- The onboard preflight achieved `469.63 Hz`, a `4.096 ms` p99 packet gap, and
  an `8.542 ms` maximum gap. It also received valid low-level state and detected
  `L2+B`.
- An Ubuntu-direct Wi-Fi run achieved only `339.42 Hz` and had a `142.002 ms`
  maximum packet gap. It fails the low-level communication gate.

Do not run a hardware experiment from Ubuntu with `./build/...`. Hardware
commands in this runbook use `./build-arm64/...` after SSHing into the Pi.

Current architecture:

```text
Ubuntu terminal -- SSH --> Go1 Raspberry Pi -- internal Ethernet --> motors
```

Planned MPPI architecture:

```text
Qualisys --> Ubuntu MPPI/high-level controller --> reference messages
                                              --> Pi 500 Hz controller --> motors
```

The Pi will retain the 500 Hz loop, watchdog, state machine, and final safety
authority. It is not intended to become an open-loop relay.

Keep the repository and `build-arm64` directory on the Pi. They do nothing when
the executable is not running and are the intended deployment copy.

## What to do next

An earlier onboard `remote-preflight` reached the required packet rate, but a
later log exposed a false `remote_preflight_tick_invalid` panic caused by an
inconsistent receive-thread snapshot. That race is fixed in the current source.

The only hardware task currently allowed is to rebuild on the Pi and repeat
`remote-preflight` once. Do not proceed to a standing mode.

After the repeated preflight passes:

1. allow `remote-preflight` to close and write its log;
2. keep Go1 powered but fully prone on the floor;
3. exit SSH, then copy, checksum, and analyze the CSV on Ubuntu;
4. remove the verified copy from the Pi if disk space is needed;
5. shut down Go1 normally while it is still prone and floor-supported;
6. do not start `ground-handover`, squat, or a leg-lift mode;
7. continue with software-only work on the smooth lie-down-and-exit path.

`ground-handover` is the next *future experimental stage*, not the next command
to run now. Once a tested normal exit exists, power-cycle Go1, let the factory
controller bring it to a normal stable stand, verify the original remote, and
then proceed to Stage 1.

However, **do not run `ground-handover` yet unless one of these conditions is
met**:

1. the program has a tested smooth lie-down-and-exit path; or
2. a reliable overhead support or person can carry the robot's full weight
   while panic damping is activated.

The current program does not automatically return control to Unitree's factory
controller after a standing run. Normal completion and one `Ctrl-C` leave the
robot in four-foot impedance hold. Panic damping can make a standing robot
collapse. The unstable large Lego block is not an acceptable load-bearing
support.

The immediate next task is therefore software work: implement and dry-test the
smooth lie-down-and-exit path. Once that is available, continue with Stage 1 in
this document. MOCAP is not needed for this work.

Progress checklist:

- [x] Initial onboard prone remote preflight and remote decoding
- [ ] Repeat onboard preflight with the receive-snapshot fix
- [ ] Smooth lie-down-and-exit implementation and dry-run validation
- [ ] Three ground-handover trials
- [ ] Three squat trials
- [ ] Three single-leg trials
- [ ] Three four-leg-sequence trials

## Safety rules and stop behavior

- Use a charged battery, a flat non-slip floor, and a clear leg workspace.
- Do not lift Go1 after power-on; doing so can invalidate contact detection and
  cause uncontrolled leg motion.
- Keep only one Unitree controller process running.
- Do not turn off the battery while Go1 is standing.
- For the first standing runs, have another person present and keep the original
  remote within reach.
- `--ground-confirmed`, `--remote-confirmed`, and interactive arming prompts are
  confirmations, not physical safeguards.

Current stop semantics are:

| Input | Result |
| --- | --- |
| One `Ctrl-C` during a ground action | Cancel overlay/action, return to four-foot captured-pose hold; process does not exit |
| `L2+B` or a second `Ctrl-C` within one second | Clear feed-forward torque and continuously send damping commands |
| One `Ctrl-C` while panic damping is active | Continue damping for at least another 0.5 s, then close UDP, write the log, and exit |
| Command publication stalls for more than 20 ms | Send thread overrides the stale command with damping |

`L2+B` is interpreted by this program during low-level operation. It is not a
request to the factory controller. Software damping still depends on the Pi
process and robot control electronics; battery isolation remains the final
physical stop.

## One-time installation and build

### 1. From the Ubuntu workstation: synchronize the repository

Connect Ubuntu to the normal Go1 Wi-Fi network. Ubuntu only needs to reach the
Pi at `192.168.12.1`; it does not need a route to `192.168.123.10` for this
deployment.

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
git pull --ff-only
rsync -av --exclude .git --exclude build --exclude build-arm64 \
  ./ pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/
```

If the Ubuntu checkout has uncommitted work, resolve or preserve it before
pulling; do not discard it to force the update.

Do not copy the Ubuntu `amd64` executable to the Pi. Build an arm64 executable
on the Pi.

### 2. On the Raspberry Pi: build and test

```bash
ssh pi@192.168.12.1
cd ~/Robotic-Dog-Tracking-Interface
cmake -S . -B build-arm64 -DPYTHON_BUILD=OFF
cmake --build build-arm64 --target \
  go1_lowlevel_experiment go1_kinematics_test -j2
ctest --test-dir build-arm64 --output-on-failure
mkdir -p logs
ip route get 192.168.123.10
```

The final command should report a direct route similar to:

```text
192.168.123.10 dev eth0 src 192.168.123.161
```

If it does not, stop before running any hardware mode.

### 3. Before every run: confirm there is only one controller

Run this on the Pi:

```bash
pgrep -af 'go1_lowlevel_experiment|example_|run_torque_tracking'
sudo ss -lunp | grep -E ':(8090|8091)\b'
```

No experiment process should be listed and ports `8090` and `8091` should not
have an owner; no output from the `ss` command is the expected result. Do not
proceed if another motor-control program or socket is active. The executable
now checks its required ports before showing the arming prompt and exits instead
of continuing after `Address already in use`.

## Common per-run workflow and machine boundary

1. Power and place Go1 as required by the selected stage.
2. From Ubuntu, SSH to the Pi.
3. **On the Pi**, run the arm64 experiment. The Pi writes the raw CSV.
4. Watch the terminal continuously and use the stop behavior above if needed.
5. End the program using the stage-specific safe procedure.
6. Exit SSH and return to the **Ubuntu workstation**.
7. **On Ubuntu**, download, verify, and analyze the CSV. Do not run the Python
   analyzer on the Pi.

The shell prompt is the easiest way to avoid mixing the two machines:

```text
pi@raspberrypi:...$       C++ hardware process and raw CSV creation
aims@aims-Precision-...$  scp, checksum, Python analysis, plots, and archive
```

From Ubuntu, download and analyze a log as follows. Replace `<log>` with one
exact filename; do not paste the angle brackets literally.

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
mkdir -p logs/downloaded
ssh pi@192.168.12.1 \
  'sha256sum ~/Robotic-Dog-Tracking-Interface/logs/<log>.csv'
scp pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/logs/<log>.csv \
  logs/downloaded/
sha256sum logs/downloaded/<log>.csv
python3 experiment/analyze_lowlevel_log.py \
  logs/downloaded/<log>.csv --no-plots
```

The remote and local SHA-256 values must match. Never reuse a log filename for
separate trials.

### Raspberry Pi storage cleanup

Raw 500 Hz CSV files are large and should not accumulate on the Pi. Inspect
storage from Ubuntu without deleting anything:

```bash
ssh pi@192.168.12.1 'df -h / && \
  du -sh ~/Robotic-Dog-Tracking-Interface/logs \
         ~/Robotic-Dog-Tracking-Interface/build-arm64 2>/dev/null && \
  find ~/Robotic-Dog-Tracking-Interface/logs -maxdepth 1 \
       -type f -name "*.csv" -printf "%TY-%Tm-%Td %TH:%TM %10s %p\n"'
```

Delete a Pi log only after its checksum matches the Ubuntu copy and the Ubuntu
analyzer has completed. Delete the exact file, not a wildcard:

```bash
ssh pi@192.168.12.1 \
  'rm -- ~/Robotic-Dog-Tracking-Interface/logs/<verified-log>.csv'
```

Older logs created before the `logs/` convention may be in the repository root.
List them first with `find ... -maxdepth 1 -type f -name "*.csv"`; apply the
same copy, checksum, analysis, and exact-file deletion procedure. Keep the
repository, `build-arm64`, SDK libraries, and current executable on the Pi.

## Stage 0: prone remote preflight — repeat after rebuilding

This stage actively sends low-level damping to all joints. It is not a passive
receiver. Go1 must already be fully prone, with nobody near its legs.

### Physical preparation

1. Place Go1 belly-down with all legs correctly folded and not trapped beneath
   the body.
2. Turn on the original Unitree remote: press the power button once, then hold
   it for more than two seconds.
3. Power on the robot: press the battery button once, then hold it for more than
   two seconds.
4. After startup, use the factory remote to make Go1 lie down and enter the
   prone damping state with `L2+B`.

### Command on the Pi

```bash
cd ~/Robotic-Dog-Tracking-Interface
mkdir -p logs
./build-arm64/go1_lowlevel_experiment --mode remote-preflight \
  --prone-confirmed --duration-s 60 \
  --log logs/remote_preflight_onboard_fix_01.csv
```

At the prompt, type exactly:

```text
ARM DAMPING
```

Operate the joysticks and press `L2+B` while the program runs. The robot is not
expected to execute factory remote motions because the program is continuously
sending low-level damping. Judge the test by terminal fields and the CSV:

- joystick values must change;
- `buttons=0x220` or `L2+B=1` must appear;
- valid state must remain fresh.

The recorded onboard result passed:

```text
network: rate=469.63 Hz, p99_gap=4.096 ms, max_gap=8.542 ms
remote: valid_fresh_ratio=1.000, L2+B_seen=1, lowlevel_fresh_ratio=1.000
watchdog_cycles=0
```

The fixed build adds two analyzer diagnostics. This repeat passes only when it
also reports:

```text
tick: duplicate_fresh=0, gap_over_20ms=0
```

In the failed log, sample 5510 copied the old `tick=332367` while observing a
new receive sequence. It was the single trigger for
`remote_preflight_tick_invalid`. Later `PANIC_DAMPING` rows retained the same
latched abort reason; they were not hundreds of independent tick failures. The
receive thread now publishes the packet and sequence under one lock, and the
control thread copies them under that same lock.

Prone calf angles can sit slightly below the SDK's normal command boundary.
The program accepts a small feedback-only margin in confirmed prone preflight;
it does not enlarge the limits used for commands or standing actions.

### Mandatory stop after preflight

On success, the program closes its UDP socket and leaves the robot physically
prone. Do not launch a standing mode in the same boot. Keep it prone while the
CSV is copied and checked on Ubuntu, optionally clean the verified Pi copy, and
then shut the robot down normally while it remains floor-supported. With the
current build, this completes the hardware session.

## Stage 1: ground handover — future stage, currently blocked

Only start this stage after the exit limitation in **What to do next** has been
resolved. The goal is to verify a quiet transition from the factory standing
controller to low-level captured-pose hold before any commanded motion.

The command in this section is retained for use after that software gate is
cleared. It is not an instruction to run it immediately after Stage 0.

### Preconditions

- Onboard remote preflight has passed.
- A safe normal-exit procedure or full-weight support is available.
- Go1 has started normally and is standing still on a flat, non-slip floor.
- The original remote is valid and within reach.
- No other controller process is running.

### Command on the Pi

```bash
cd ~/Robotic-Dog-Tracking-Interface
./build-arm64/go1_lowlevel_experiment --mode ground-handover \
  --ground-confirmed --log logs/ground_handover_01.csv
```

Type `ARM` only after checking the physical scene again. The program captures a
stable standing pose, then the first low-level packet commands that measured
pose with impedance hold. It does not command a nominal hard-coded pose.

Run three separate handover trials before Stage 2. Each trial must satisfy:

- valid feedback rate at least `450 Hz`;
- p99 packet gap no greater than `10 ms`;
- no packet gap greater than `20 ms`;
- no safety or watchdog trigger;
- no visible step, collapse, or violent motor response during takeover.

### Interpreting the observed failed attempt

The following output means that low-level takeover did not occur:

```text
[Error] Bind client ip&port failed: Address already in use
Ground takeover aborted before low-level transmission:
a stable high-level standing pose was not received.
```

The send, receive, and control loops are started only after the high-level pose
capture succeeds. Therefore this attempt did not send low-level standing-hold
commands.

The first line means another live socket owned local UDP port `8090`; a
completed UDP process does not retain a UDP `TIME_WAIT` reservation. Before any
future retry, inspect the owner on the Pi without killing anything:

```bash
pgrep -af 'go1_lowlevel_experiment|example_|run_torque_tracking'
sudo ss -lunp | grep -E ':(8090|8091)\b'
```

Do not work around this by choosing a random local port until the owner is
identified. It may indicate that another robot controller is still active. The
current build probes both required local ports before arming, so this condition
now exits without constructing an SDK controller or asking for `ARM`.

The high-level capture failure separately means the program did not collect
100 consecutive valid, stable `HIGHLEVEL` state packets from
`192.168.123.161:8082`. Running this directly after prone low-level preflight,
without a full shutdown and normal factory-controlled restart, is not a valid
handover setup. The current program reports only the aggregate failure, so this
message alone cannot distinguish a missing endpoint from rejected state data.

## Stage 2: squat and return

Start only after three successful handover trials and a safe normal-exit method.
The action uses four-leg impedance control: three seconds down, two seconds
hold, and three seconds return. Default offsets are `+0.12 rad` at each thigh
and `-0.24 rad` at each calf.

```bash
cd ~/Robotic-Dog-Tracking-Interface
./build-arm64/go1_lowlevel_experiment --mode squat \
  --ground-confirmed --log logs/squat_01.csv
```

Complete three trials before Stage 3. Acceptance criteria are:

- no protection or watchdog trigger;
- joint tracking RMS below `0.08 rad`;
- final pose error below `0.05 rad`;
- roll and pitch excursions below `0.15 rad`.

## Stage 3: one automatically selected leg

Start only after three successful squat trials. This is a mixed-impedance
motion test, not a pure-torque frequency-response test. The controller shifts
the body toward the three-foot support polygon, unloads the selected foot,
lifts it by 20 mm, and applies a small torque overlay to the airborne thigh.

```bash
cd ~/Robotic-Dog-Tracking-Interface
./build-arm64/go1_lowlevel_experiment --mode leg-lift \
  --leg auto --lift-height-m 0.02 \
  --tau-overlay-nm 0.10 --tau-overlay-hz 0.5 \
  --ground-confirmed --remote-confirmed \
  --log logs/leg_lift_auto_01.csv
```

Stop and return to four-foot hold if the support margin decreases, the target
foot does not unload, the IMU excursion approaches its limit, or any feedback
becomes stale. Complete three successful trials before Stage 4.

## Stage 4: four-leg sequence

This stage repeats the support and contact checks before every leg. It is
allowed only after three successful Stage 3 trials.

```bash
cd ~/Robotic-Dog-Tracking-Interface
./build-arm64/go1_lowlevel_experiment --mode leg-lift-sequence \
  --leg auto --lift-height-m 0.02 \
  --tau-overlay-nm 0.10 --tau-overlay-hz 0.5 \
  --ground-confirmed --remote-confirmed \
  --log logs/leg_lift_sequence_01.csv
```

The sequence passes only if every foot unloads, lifts, and recontacts normally,
the support margin remains valid, and no protection or watchdog trigger occurs.

## Hardware gates enforced by the program

The controller runs a 2 ms control period and checks, among other conditions:

- valid and fresh `LowState` packets and an advancing `state.tick`;
- `levelFlag == LOWLEVEL` after takeover;
- finite joint feedback, motor mode, and temperature;
- command and manufacturer joint limits;
- command torque at or below the configured first-stage limit;
- communication staleness, joint speed, roll/pitch, and foot-force conditions;
- `PositionLimit` and `PowerProtect(..., 1)` on every normal command.

Hard faults clear `tau_ff` and enter persistent damping. Ground-action faults
cancel the action and return to captured four-foot hold when that is the safer
available response.

## Logs and analysis

The Pi creates the raw CSV; Ubuntu downloads, checksums, analyzes, plots, and
archives it. Every 2 ms cycle records host time, state tick, phase,
receive/send status, loop timing, remote input, stop source, watchdog state,
IMU, foot force, support margin, command fields, and joint feedback.

For each joint, interpret torque tracking using the total commanded torque:

```text
tau_cmd_total = tau_ff + Kp * (q_des - q) + Kd * (dq_des - dq)
```

Do not compare `tauEst` only with `tau_ff` during impedance actions. The Python
analyzer reports communication timing, motion limits, remote validity, support
metrics, abort reasons, and torque alignment. Plot generation can be enabled by
omitting `--no-plots`.

`recv_ok=1` means the control loop consumed a newly published state snapshot;
it does not merely mean that `UDP::Recv()` returned zero. A valid log must have
`fresh_tick_duplicate_count=0` and
`fresh_tick_gap_over_20ms_count=0`. The analyzer prints these as
`duplicate_fresh` and `gap_over_20ms`, and prints the first sample that latched
an abort. Because abort state is latched, all later panic rows keep the same
reason; count the first abort as the trigger rather than treating every panic
row as a new fault.

## MOCAP and future MPPI boundary

MOCAP is not an input to `remote-preflight`, `ground-handover`, squat, or the
initial leg-lift experiments. Encoders, IMU, estimated joint torque, foot force,
and the remote are sufficient for these local state machines.

Qualisys may be recorded as independent ground truth. In the future, Qualisys
and MPPI can run on Ubuntu and send lower-rate reference commands to the Pi.
MOCAP delay or loss must not be allowed to bypass the Pi's local 500 Hz safety
loop or become a requirement for fast balance.

## Appendix A: legacy Ubuntu-direct network test

This appendix explains earlier logs; it is not the current operating procedure.

The Ubuntu workstation had both the wired MOCAP network and Go1 Wi-Fi active:

```text
enp0s31f6  192.168.1.167/24   wired MOCAP network
wlp0s20f3  192.168.12.20/24   Go1 Wi-Fi
Go1 Pi     192.168.12.1       Wi-Fi/robot gateway
controller 192.168.123.10     robot internal low-level endpoint
```

A default-route change was initially used, but it is not appropriate for the
split-network workstation. A narrow route was required to test Ubuntu-direct
access without diverting MOCAP traffic:

```bash
sudo ip route replace 192.168.123.0/24 \
  via 192.168.12.1 dev wlp0s20f3 src 192.168.12.20
```

That route made the endpoint reachable, but the measured Wi-Fi low-level rate
and maximum gap failed the gate. Do not restore Ubuntu-direct hardware control
unless a new 60-second qualification produces at least `450 Hz`, p99 at most
`10 ms`, and no single gap over `20 ms`. Even then, onboard execution remains
the preferred deployment.

## Appendix B: offline development

`--dry-run` does not open UDP and may run on a development machine. For example:

```bash
./build/go1_lowlevel_experiment --dry-run --mode leg-lift \
  --leg auto --ground-confirmed --remote-confirmed \
  --log leg_lift_dry.csv
python3 experiment/analyze_lowlevel_log.py leg_lift_dry.csv --no-plots
```

Dry-run validates reference generation, state-machine sequencing, and log
format only. It cannot validate firmware mode switching, robot networking,
remote delivery, motor response, or physical stability.
