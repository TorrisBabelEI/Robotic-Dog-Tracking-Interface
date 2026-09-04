# Go1 low-level experiment: current gate

This document contains only the next permitted operation. Follow it from top to
bottom and stop at the end. It does not currently authorize a hardware
preflight.

## Current objective

Identify the exact GUI-session command that starts and restarts Unitree's
`programming.py`. This step is read-only.

Do **not** stop a process, run `go1_lowlevel_experiment`, or create a hardware
CSV yet. The proposed file `remote_preflight_fix_01.csv` does not exist in the
repository and should not be created during this gate.

## Confirmed port conflict

The onboard Pi currently runs:

```text
python3 /home/pi/Unitree/autostart/programming/programming.py
local UDP 192.168.123.161:8090 -> 192.168.123.161:8082
```

This is Unitree's optional Programming Module for its GUI/Blockly/MQTT
interface. It:

- imports `robot_interface_high_level`;
- connects to the local MQTT broker;
- waits for `programming/action` and `programming/code`;
- starts high-level `robotControl`, `UDPSend`, and `UDPRecv` children only after
  receiving programming code;
- nevertheless claims local UDP 8090 while idle because it imports the
  high-level robot interface.

It is not a core leg-control process, but it conflicts with Unitree's standard
low-level source port. Testing local port 8092 produced only about `0.54 Hz`, so
changing the source port is not an acceptable workaround.

The required low-level path remains:

```text
Pi local UDP 8090 -> Go1 low-level endpoint 192.168.123.10:8007
```

The future high-level pose-capture source port remains 8091, but no standing
mode is currently allowed.

## Processes outside the experiment scope

Do not stop or modify any of these processes:

- `Legged_sport`;
- `appTransit`;
- `hostapd`;
- ROS obstacle or ultrasonic processes.

Only `programming.py` is the confirmed 8090 conflict. Even that process must not
be stopped until its exact restoration command is known.

## Step 1 — SSH from Ubuntu to the onboard Pi

On the **Ubuntu workstation**:

```bash
ssh pi@192.168.12.1
```

Confirm that the prompt begins with `pi@raspberrypi`. All remaining commands in
this runbook run on the **Pi**.

## Step 2 — Record the current process and UDP owner

Run on the **Pi**:

```bash
pgrep -af 'python3 programming.py'
sudo ss -Huanp | awk '$4 ~ /:8090$/ { print }'
sudo fuser -v 8090/udp
```

Expected observations:

- `pgrep` prints the current `programming.py` PID. The PID may change on every
  boot; never reuse an earlier PID such as 1869 or 1848.
- `ss` shows local `192.168.123.161:8090` connected to
  `192.168.123.161:8082`.
- `fuser` identifies the same current process as the port owner.

Use `ss -Huanp`, not `ss -lunp`: the latter only shows listening/unconnected
UDP sockets and previously hid this connected socket.

## Step 3 — Read the GUI startup configuration

Run on the **Pi**:

```bash
sed -n '1,260p' /home/pi/Unitree/autostart/startup_manager.py
sed -n '1,260p' /home/pi/.config/lxsession/LXDE-pi/autostart
grep -Rsn 'programming\|startup_manager' \
  /home/pi/.config /home/pi/Unitree/autostart \
  /etc/xdg 2>/dev/null
```

Keep the complete output, including the command line around every match. The
required result is the exact action that recreates the Programming Module in
the existing GUI user session—not a guessed `systemctl` command. It is already
known that `programming.py` is not an independent systemd service.

## Step 4 — Exit and stop

Run on the **Pi**:

```bash
exit
```

Save the complete Step 2 and Step 3 terminal output on Ubuntu. This is the end
of the current procedure.

Do not run `kill`, do not rebuild or start the experiment, and do not repeat a
preflight. First add the verified Programming Module restart command to this
document.

## Locked next phase — not executable yet

After the restart path is known, the next revision will expand the following
sequence into exact commands:

1. Pull the revised source on Ubuntu and copy only the C++ build inputs, SDK
   headers, and arm64 SDK library to the Pi.
2. Build and run all 17 software tests on the Pi.
3. Put Go1 fully prone on a flat floor and enter remote damping with `L2+B`.
4. Resolve the current `programming.py` PID with `pgrep` and confirm that the
   same process owns local UDP 8090.
5. Send `TERM` to that current PID only, wait two seconds, and verify that 8090
   is free. If a startup manager immediately recreates it, do not kill it
   repeatedly; stop and address the manager first.
6. Run one prone preflight using local 8090 and the unchanged low-level target
   `192.168.123.10:8007`.
7. Restore the Programming Module with the verified GUI-session command and
   confirm that both its process and
   `8090 -> 192.168.123.161:8082` socket return.
8. Copy the raw CSV to Ubuntu, compare checksums, analyze it on Ubuntu, and
   delete only the verified Pi copy.
9. Shut down Go1 normally while it remains prone and floor-supported.

The future preflight command will be:

```bash
./build-arm64/go1_lowlevel_experiment --mode remote-preflight \
  --local-port 8090 \
  --prone-confirmed --duration-s 60 \
  --log logs/remote_preflight_fix_01.csv
```

Do not run it in the current gate. The executable now also defaults to local
port 8090.

## Future preflight behavior

- The program sends zero-torque damping continuously. Motor engagement sounds
  without visible motion are expected.
- Joystick motion is logged but does not command robot motion.
- `L2+B` is recorded and deliberately does not cause a second transition in
  `remote-preflight`.
- A preflight fault sends a final 0.5-second damping window and then closes.
- If panic occurs, preserve and analyze that one CSV; do not immediately repeat
  the test.

## Deployment boundary

The Pi owns the 500 Hz motor loop, watchdog, and safety state machine. Ubuntu
owns SSH operation, CSV analysis, and future MPPI/Qualisys integration. MOCAP
is not required for the preflight and must not enter the fast motor loop.
