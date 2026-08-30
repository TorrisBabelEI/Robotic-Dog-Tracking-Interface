# Go1 low-level torque tracking and ground experiment

This runner provides a 500 Hz C++ safety state machine for receiving Unitree
`LowState`, taking over a standing Go1, exercising a small hybrid-impedance
torque channel, and writing a per-cycle CSV log. The real-time loop uses only
joint encoders, `tauEst`, IMU, foot force, and the Unitree remote. MOCAP is
optional external ground truth and is never part of the control loop.

## Safety boundary

Low-level control can still make a free-standing robot fall. Software damping
depends on the process, network, and motor controller remaining alive, so it
does not replace battery disconnection. For the first action runs, keep a
second person ready to support the robot; use physical power removal only after
the robot's weight is supported or as a last-resort emergency action. Do not
use the unstable block as a load-bearing stand.

Only one process may send Unitree commands. Stop the trajectory tracker,
native SDK examples, and every other high- or low-level command process before
starting this runner.

The intended ground sequence is deliberately gated:

1. remote receive-only preflight with the robot prone or already in damping;
2. standing low-level handover and hold;
3. symmetric squat and return;
4. one automatically selected leg lift;
5. all four legs, with a new force baseline before each leg.

`--ground-confirmed` acknowledges that the robot is on clear, level ground.
`--remote-confirmed` acknowledges that `L2+B` was seen during the receive-only
preflight. Both flags are required for hardware leg-lift modes. `--dry-run`
opens no UDP socket and needs no confirmation flags.

Single-joint pure torque remains available only for a genuinely load-bearing
stand using `--mode torque-sine --support-confirmed`. The ground leg lift uses
position impedance and only a small thigh torque overlay; it is not a pure
torque frequency-response measurement.

## Build

The SDK submodule contains prebuilt Linux libraries for `amd64` and `arm64`.
Build hardware access on Ubuntu or the Go1 onboard computer:

```bash
git submodule update --init --recursive
cmake -S . -B build
cmake --build build --target go1_lowlevel_experiment go1_kinematics_test
ctest --test-dir build --output-on-failure
```

On macOS, the target is intentionally dry-run-only:

```bash
cmake -S . -B build
cmake --build build --target go1_lowlevel_experiment
./build/go1_lowlevel_experiment --dry-run --mode leg-lift \
  --leg auto --log /tmp/go1_leg_lift_dry.csv
```

Install the offline analysis dependencies in the plotting environment:

```bash
python3 -m pip install -r requirements-analysis.txt
```

## Stop behavior

- One Ctrl-C immediately cancels the torque overlay, lowers/recenters when
  possible, and returns to the captured four-foot pose. The process then
  keeps transmitting that hold command.
- `L2+B`, or a second Ctrl-C within one second, switches all joints on the next
  control cycle to `q=PosStop`, `Kp=0`, `Kd=1`, `tau_ff=0` and keeps sending
  damping.
- After panic damping is active and the robot is physically safe, another
  Ctrl-C ends the process and writes the buffered CSV.
- The send thread alone owns UDP transmission. If the controller fails to
  publish a command for more than 20 ms, the send thread substitutes damping;
  recovery from that event is latched as panic rather than resuming motion.

The remote stop is recognized by this program while it owns low-level mode;
it does not rely on the normal high-level controller handling the buttons.

## Ubuntu operator runbook: battery installation through ground experiments

The placement, startup, remote-control shutdown sequence, and warnings in this
section follow the [Unitree Go1 User Manual](https://static.generation-robots.com/media/user-manual-go1-unitree-robotics.pdf).
Read that manual before operating the robot. Run every shell command below
from the root of the `Robotic-Dog-Tracking-Interface` checkout on the Ubuntu
computer.

### 0. Current exit limitation: read before a ground run

The receive-only remote preflight can be performed immediately with the robot
prone. The current low-level program does **not** provide a normal hand-back to
Unitree's high-level controller or an automatic lie-down-and-exit action.
After any command-sending mode completes, it continues holding the captured
four-foot standing pose.

- One Ctrl-C cancels the active overlay/action, returns to four-foot hold, and
  remains in that hold. It does not exit.
- `L2+B`, or two Ctrl-C presses within one second, commands damping. A standing
  robot may sink or fall when damping is applied.
- After panic damping is active, one more Ctrl-C exits and writes the log.
- Never turn off the battery while the robot is standing. The robot will lose
  support immediately and fall.

Consequently, do not start `ground-handover` or a later ground action without
either a reliable overhead safety rope/frame or personnel who can physically
support the robot throughout the transition to damping. The preferred fix is
to implement and validate a smooth lie-down-and-exit mode before routine ground
experiments. An observer near the battery is useful for fault response but is
not, by itself, a reliable load-bearing support.

### 1. Prepare the battery, robot, remote, and Ubuntu computer

1. Use a sufficiently charged battery while it is switched off. Insert it in
   the correct orientation; do not force it into the battery bay.
2. Put Go1 on open, level, non-slip ground with its abdomen flat on the floor.
   Fully fold the legs into the normal startup configuration. No thigh or calf
   may be trapped beneath the body.
3. Turn on the original Unitree remote: press its power button once, then press
   and hold it for more than two seconds. Wait for its data-link indicator.
4. Connect the Ubuntu computer to the Go1 network normally used in the lab.
5. MOCAP is not required. It may remain disconnected for this entire runbook.
6. Stop the trajectory tracker, Unitree examples, and every other high- or
   low-level command process. Only one process may transmit robot commands.
7. Keep people, cables, and equipment outside the leg workspace. For the first
   command-sending trials, assign another person to watch and, if necessary,
   support the robot. Keep the physical power control accessible, but never use
   it as a normal exit while the robot is standing unsupported.

If Go1 Wi-Fi and wired Qualisys are connected simultaneously, verify that the
two subnets use the intended interfaces:

```bash
ip route get 192.168.123.10
ip route get 192.168.1.122
```

### 2. Power on Go1

With the robot placed as described above and everyone clear of the legs:

1. Press the battery power button once.
2. Press it again and hold for more than two seconds.
3. Wait for the power-on self-test to finish.
4. After a successful self-test, Go1 should stand at its normal initial height.

Do not lift or carry the robot after it is powered on. If it does not stand,
only makes abnormal sounds, kicks its legs, or otherwise fails the self-test,
stop here. Do not compensate by increasing controller gains or torque.

### 3. First run: receive-only remote preflight

Before starting the program, use the original remote to put the robot down.
Starting from quiet static standing, hold `L2` and press `A` three times. Wait
while Go1 squats, stands, and then lies prone. The manual's complete shutdown
sequence subsequently uses `L2` held while `B` is pressed twice to pass through
prone damping and prone undamped states.

Once the robot is stable and prone, run:

```bash
./build/go1_lowlevel_experiment --mode remote-preflight \
  --duration-s 30 --log remote_preflight.csv
```

This mode receives `LowState` and writes a log, but never sends a motor command.
Move the sticks and press buttons while watching the terminal. Confirm that it
prints:

```text
remote valid=1
```

For a clean factory shutdown sequence, keep `L2` held and press `B` once while
watching the terminal. Confirm:

```text
L2+B=1
```

The factory controller still receives the remote even though this program is
receive-only. Keep Go1 flat on the floor, then press `B` a second time while
continuing to hold `L2`; this completes the manual's two `L2+B` transitions and
leaves the robot prone and undamped. Then inspect the log:

```bash
python3 experiment/analyze_lowlevel_log.py \
  remote_preflight.csv --no-plots
```

If `remote valid=1` and `L2+B=1` are not both observed reliably, do not run
`leg-lift` or `leg-lift-sequence`.

### 4. Shut down after preflight, then restart for standing tests

The program exits automatically after the preflight duration. If the two
`L2+B` presses above were completed from the prone state, Go1 should now be
prone and undamped. Do not guess: if the state is uncertain, follow the manual
and verify the prone undamped state before touching the battery. Only then
switch off the battery by pressing the battery button once and then holding it
for more than two seconds. All battery indicators should turn off.

To prepare for a standing experiment, return the legs to the normal startup
configuration and repeat Sections 1 and 2. Let Go1 complete its self-test and
stand normally. Do not pick it up after startup.

### 5. Standing low-level handover

Proceed only if the exit limitation in Section 0 has been addressed. Confirm
that no other process is transmitting commands, then run:

```bash
./build/go1_lowlevel_experiment --mode ground-handover \
  --ground-confirmed --log ground_handover_01.csv
```

The program displays a warning and requires the operator to type exactly:

```text
ARM
```

No command is sent before this confirmation. In a ground mode, the UDP send
loop also stays off until the program has received a complete finite, in-range
joint state and constructed a hold command from it. The first transmitted
packet is therefore the measured standing-pose hold, not the default damping
packet. The program captures and holds that pose for 10 seconds, then remains
in four-foot hold because of the exit limitation above.

Complete three handover trials before enabling an action. Review each log for
an average valid-feedback rate of at least 450 Hz, p99 packet gap below 10 ms,
and no individual gap above 20 ms:

```bash
python3 experiment/analyze_lowlevel_log.py \
  ground_handover_01.csv --no-plots
```

If Wi-Fi misses this gate, run the same executable on the onboard arm64
computer and copy the CSV back to Ubuntu for analysis.

### 6. Squat and return

After three successful handovers, run:

```bash
./build/go1_lowlevel_experiment --mode squat \
  --ground-confirmed --log squat_01.csv
```

The quintic action takes 3 seconds to squat, holds for 2 seconds, and takes 3
seconds to recover. Each thigh moves `+0.12 rad`, each calf moves `-0.24 rad`,
and each hip remains at its captured angle. The program then remains in
four-foot hold.

Run three trials. Require no protection event, per-joint position RMS below
0.08 rad, final position error below 0.05 rad, and roll/pitch excursion below
0.15 rad before attempting a leg lift.

### 7. One automatically selected leg

The first leg trials must use `auto` selection:

```bash
./build/go1_lowlevel_experiment --mode leg-lift \
  --leg auto --lift-height-m 0.02 \
  --tau-overlay-nm 0.10 --tau-overlay-hz 0.5 \
  --ground-confirmed --remote-confirmed \
  --log leg_lift_auto_01.csv
```

The runner records two seconds of foot-force data and rejects a sensor whose
median baseline is below 5 raw units or whose MAD exceeds 20 percent of its
baseline. It estimates CoP, selects the candidate three-foot support triangle
with the largest predicted margin, shifts the body by at most 30 mm, lifts the
selected foot by 20 mm, applies the ramped `0.10 Nm / 0.5 Hz` thigh overlay,
lowers the foot, verifies contact, and recenters.

Complete three guarded single-leg trials without a protection event or support
margin violation before enabling the sequence. Explicit `--leg FR|FL|RR|RL`
selection is intended for later diagnosis, not the initial hardware trial.

### 8. Four-leg sequence

```bash
./build/go1_lowlevel_experiment --mode leg-lift-sequence \
  --leg auto --ground-confirmed --remote-confirmed \
  --lift-height-m 0.02 --tau-overlay-nm 0.10 \
  --tau-overlay-hz 0.5 --log leg_lift_sequence_01.csv
```

After each leg, the controller recenters, waits 3 seconds, and records a new
force baseline. It selects the safest remaining untested leg from the current
support estimate; there is no fixed leg order. A new leg starts only after the
previous foot has passed contact confirmation.

### 9. End-of-run procedure for the current implementation

The normal action completion state is four-foot hold. Do not press the battery
button while Go1 is standing. Until a smooth lie-down-and-exit mode is added,
ending a command-sending run requires a controlled support arrangement:

1. Have the safety rope/frame or support person take the robot's weight.
2. Press `L2+B`, or press Ctrl-C twice within one second, to enter damping.
3. Confirm physically that Go1 is fully supported and no leg is driving.
4. Press Ctrl-C once more to close the program and write the CSV.
5. Place Go1 fully prone and undamped before using the battery's normal
   short-press-then-long-press shutdown sequence.

Treat this as a temporary experimental limitation, not as the desired routine
workflow. Software damping still depends on the Ubuntu process, network link,
and Go1 control electronics remaining operational.

## Runtime gates

Every command passes `PositionLimit` and `PowerProtect(..., 1)`. The runner
also enforces:

- feedforward torque at or below 1 Nm and position targets no farther than
  0.3 rad from the captured pose;
- valid low-level state, finite joint feedback, manufacturer joint bounds,
  temperature below 70 C, and no motor overheat mode;
- feedback age no greater than 20 ms;
- during ground actions, roll/pitch change no greater than 0.10 rad and joint
  speed no greater than 0.8 rad/s;
- total foot force within 70-130 percent of baseline during leg motion;
- at least 10 mm support margin and target-foot force below 20 percent of its
  baseline while airborne.

Failure of a recoverable motion gate cancels the overlay and returns toward
the captured four-foot pose. Invalid feedback, SDK protection rejection,
remote panic, or watchdog expiry enters damping.

## CSV and analysis

Each 2 ms record includes timing, robot tick, phase, receive/send state,
watchdog state, stop source and damping timestamp; decoded remote input; IMU;
foot force baselines and MAD; CoP and every candidate support margin; IK foot
targets; and for all 12 joints:

- commanded `q`, `dq`, `Kp`, `Kd`, and `tau_ff`;
- reconstructed `tau_cmd_total = tau_ff + Kp*(q_des-q) + Kd*(dq_des-dq)`;
- measured `q`, `dq`, `tauEst`, mode, and temperature.

The analyzer writes a summary CSV and plots for torque alignment, position,
IMU, force, packet/loop timing, and `cop_support_margin.png`. Its summary also
reports airborne force ratio, minimum support margin, final contact ratio,
remote-stop-to-damping latency, and watchdog-active cycles.

This ground test is a mixed-impedance torque-path and action-chain check. A
stand-supported pure torque test is still required for quantitative motor
frequency response. MOCAP may be synchronized for body height, drift, and
attitude ground truth, but its latency or loss cannot affect the 500 Hz state
machine.
