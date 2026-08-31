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

The onboard `remote-preflight` has passed. The next experimental stage is
`ground-handover`, followed by squat, one leg, and finally the four-leg
sequence.

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

Therefore the immediate next software task is to implement and dry-test the
smooth lie-down-and-exit path. Once that is available, continue with Stage 1 in
this document. MOCAP is not needed for this work.

Progress checklist:

- [x] Onboard prone remote preflight
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
| One `Ctrl-C` while panic damping is active | Close UDP, write the log, and exit |
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
```

Do not proceed if another motor-control program is active.

## Common per-run workflow

1. Power and place Go1 as required by the selected stage.
2. From Ubuntu, SSH to the Pi.
3. On the Pi, enter `~/Robotic-Dog-Tracking-Interface` and run the stage command.
4. Watch the terminal continuously and use the stop behavior above if needed.
5. End the program using the stage-specific safe procedure.
6. From Ubuntu, download the CSV and run the analyzer.

Download and analyze a log from Ubuntu as follows:

```bash
scp pi@192.168.12.1:~/Robotic-Dog-Tracking-Interface/<log>.csv .
python3 experiment/analyze_lowlevel_log.py <log>.csv --no-plots
```

Never reuse a log filename for separate trials.

## Stage 0: prone remote preflight — completed

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
./build-arm64/go1_lowlevel_experiment --mode remote-preflight \
  --prone-confirmed --duration-s 60 \
  --log remote_preflight_onboard.csv
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

Prone calf angles can sit slightly below the SDK's normal command boundary.
The program accepts a small feedback-only margin in confirmed prone preflight;
it does not enlarge the limits used for commands or standing actions.

## Stage 1: ground handover — next, but currently blocked

Only start this stage after the exit limitation in **What to do next** has been
resolved. The goal is to verify a quiet transition from the factory standing
controller to low-level captured-pose hold before any commanded motion.

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
  --ground-confirmed --log ground_handover_01.csv
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

## Stage 2: squat and return

Start only after three successful handover trials and a safe normal-exit method.
The action uses four-leg impedance control: three seconds down, two seconds
hold, and three seconds return. Default offsets are `+0.12 rad` at each thigh
and `-0.24 rad` at each calf.

```bash
cd ~/Robotic-Dog-Tracking-Interface
./build-arm64/go1_lowlevel_experiment --mode squat \
  --ground-confirmed --log squat_01.csv
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
  --log leg_lift_auto_01.csv
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
  --log leg_lift_sequence_01.csv
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

Every 2 ms cycle records host time, state tick, phase, receive/send status,
loop timing, remote input, stop source, watchdog state, IMU, foot force, support
margin, command fields, and joint feedback.

For each joint, interpret torque tracking using the total commanded torque:

```text
tau_cmd_total = tau_ff + Kp * (q_des - q) + Kd * (dq_des - dq)
```

Do not compare `tauEst` only with `tau_ff` during impedance actions. The Python
analyzer reports communication timing, motion limits, remote validity, support
metrics, abort reasons, and torque alignment. Plot generation can be enabled by
omitting `--no-plots`.

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
