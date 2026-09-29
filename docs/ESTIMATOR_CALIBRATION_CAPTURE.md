# Factory walking + Qualisys estimator calibration capture

Scope: reference-data capture under factory control. This is separate from the [active body-path tracker](GO1_TRAJECTORY_TRACKING.md) and [joint torque experiments](GO1_TORQUE_TRACKING.md).

Path notation and workstation setup: [local paths](GO1_LOCAL_PATHS.md).

The walking package already runs its own onboard linear-velocity Kalman
estimator. This guide provides independent Qualisys reference measurements;
it is not an estimator-launch prerequisite for each walking session. For
ordinary hardware operation use [the walking procedure](GO1_WALKING_OPERATIONS.md).
Clock refresh after a Pi reboot does not require this Qualisys experiment.

Run `experiment/record_estimator_calibration.py` on the Linux PC while an operator
walks the Go1 using its **factory remote**. It records the existing factory sensor
stream through `${GO1_POLICY_SOURCE}/deployment/pi_state_bridge.py` and
receives Qualisys **6DOF UDP** over Ethernet. It never creates a Unitree SDK
connection or sends a motor command. The factory controller stays in charge.

The recorder needs Python 3.8+ and only the standard library. The optional offline
reference exporter needs NumPy (`/usr/bin/python3` already has it on this PC).
The existing tracking scripts supplied the QTM host/body configuration; the
recorder uses a small QTM RT 1.24 client so UDP capture does not depend on which
`qtm`/`qtm_rt` package is installed.

## Start recording

Run commands from the repository root unless noted otherwise. Keep QTM running
with real-time 6DOF processing and the `dog` rigid body visible. Use the normal
factory walking mode. This workflow does not require joint-level mode, stopping
`Legged_sport`, or launching a policy controller.

1. Copy the existing passive bridge to the Pi, if it is not already there:

   ```bash
   source experiment/go1_paths.sh
   : "${GO1_POLICY_SOURCE:?Set GO1_POLICY_SOURCE in .go1-paths.env}"
   scp "${GO1_POLICY_SOURCE}/deployment/pi_state_bridge.py" pi@192.168.12.1:~/pi_state_bridge.py
   ssh pi@192.168.12.1
   ```

   In that Pi terminal:

   ```bash
   sudo python3 /home/pi/pi_state_bridge.py --interface eth0
   ```

   This passively reads the verified native feedback flow
   `192.168.123.10:8007 -> 192.168.123.161:8008`, 820-byte packets. Root is needed
   for the packet socket. It leaves the factory controller running. A different
   robot packet profile needs its own verified decoder; it is rejected here.

2. In another PC terminal, keep the SSH tunnel running:

   ```bash
   ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=2 -o ServerAliveCountMax=2 \
     -L 127.0.0.1:15002:127.0.0.1:15002 pi@192.168.12.1
   ```

   Use one bridge client during capture. `live_shadow.py` and other consumers
   drain the same bridge queue and must not share this recording session.

3. On the PC, start the recorder:

   ```bash
   /usr/bin/python3 experiment/record_estimator_calibration.py \
     --seconds 120 --label stationary \
     --notes 'Factory remote; describe surface, marker mounting, SDK/firmware and trial split' \
     --out logs/calibration_trial_01
   ```

   Defaults come from `experiment/config/config_dog.json`: QTM `192.168.1.122`,
   rigid body `dog`. Override them when needed:

   ```bash
   /usr/bin/python3 experiment/record_estimator_calibration.py \
     --qtm-host 192.168.1.122 --body dog \
     --mocap-bind 192.168.1.100 --mocap-port 15100 \
     --seconds 120 --out logs/calibration_trial_02
   ```

   Replace `192.168.1.100` with this PC's **actual Ethernet address**, not the QTM
   server address. Omitting `--mocap-bind` selects the local address of the route
   to QTM. The PC needs connectivity both to QTM Ethernet and to the Pi tunnel.
   QTM TCP port 22223 negotiates the stream; measurements arrive at PC UDP 15100.
   Allow that UDP traffic through the PC firewall. `--component 6DRes` also saves
   the rigid-body residual if your QTM configuration supplies it.

Wait for increasing `robot`, `mocap`, and `valid_poses` counts before collecting
motion. Type labels such as `stationary`, `forward`, `backward`, `lateral_left`,
`turn_left`, and `stationary_end`, pressing Enter at each transition. An assistant
can enter these while the operator drives. These are PC-time annotations, with
human reaction delay, not precise contact labels or calibrated velocity commands.
Choose separate fitting and held-out trials and record their trial IDs in `--notes`.
Use roles or anonymous operator IDs instead of personal names in shared notes.

Ctrl+C or SIGTERM closes the files and writes the summary; it does not stop the
dog. The factory remote remains responsible for motion. Missing input streams
end recording with an error after `--source-timeout` seconds (default 10).
Occluded poses remain recorded as invalid and display `TRACKING LOST`.
Existing output directories are never overwritten. Buffered files flush every
second; a normal stop also closes and hashes them. A power loss/SIGKILL may lose
the final buffered second and will not produce a complete summary.

## Recorded files

| File | Contents |
|---|---|
| `metadata.json` | Arguments with file names instead of absolute paths, notes, clock status, raw joint ordering, source/configuration hashes |
| `robot.jsonl` | Raw native packet; CRC validity; 12 q/dq/torque/temperature channels; raw ddq; IMU quaternion/gyro/accelerometer/RPY; both foot-force arrays; 40 remote bytes; robot tick; Pi capture time; PC arrival and capture-time bounds |
| `mocap.jsonl` | Raw UDP datagram; QTM timestamp and frame number; position in meters; original rotation array; marker-to-world matrix and xyzw quaternion; validity, frame gaps, optional residual |
| `transport.jsonl` | Bridge request/reply timing, source/session identity, queue drops, capture discards, packet sequence information |
| `events.jsonl` | Motion labels, QTM events and stream configuration, stop reason |
| `qtm_6d_settings.xml` | QTM body definitions, including the name-to-index mapping used for this run |
| `summary.json` | Counts, completion/failure status, duration, SHA256 hashes of saved inputs |

Native robot values use the SDK's raw FR, FL, RR, RL order, with hip/thigh/calf
within each leg. The `tau_est_nm` scale matches the repository's native decoder;
it remains a motor torque estimate, not an independently calibrated load sensor.
`ddq_native_raw` deliberately does not assert a physical acceleration scale.
Nonfinite or CRC-invalid robot packets retain their raw bytes and an invalid flag.
The 13-byte native extension is retained in the raw packet but not interpreted.
The existing CRC excludes partial trailing words and that extension.

Qualisys rotation columns are assembled as
`R_world_marker[i][j] = wire_rotation[i + 3*j]`. This differs from reshaping the
flat array row-first in the older trajectory scripts; no empirical yaw negation
is applied. Missing tracking becomes `null`/`valid=false`, never a zero pose.
The raw datagram retains the exact original bytes, including nonfinite values.

## Clock alignment and mounting

The three clocks are kept separate:

- **QTM:** `qtm_timestamp_us` is camera time from the QTM measurement epoch.
- **Pi:** `pi_sample_monotonic_ns` is the passive bridge's network capture time.
- **PC:** `pc_received_monotonic_ns` and event times share the recorder's clock.

For each bridge request with PC send/receive times `s,r` and Pi reply time `p`,
the PC-minus-Pi clock offset lies between `s-p` and `r-p`, assuming negligible
clock-rate difference within that request. Each robot row saves the corresponding
`pc_sample_lower_ns` and `pc_sample_upper_ns`. These are bounds on translation of
the **bridge timestamp**. The existing bridge includes a conservative 1 ms
timestamp margin and rejects old kernel queues. Its capture discards and reasons
are saved; the recorder does not recover packets the bridge discarded.

A QTM UDP arrival gives only an arrival time, with camera processing/network
delay. **Simultaneous receipt is not simultaneous measurement.** No absolute QTM
latency or firmware sensor delay is inferred here. Use an independently measured
clock/latency relationship or a separate alignment experiment; record the method
and uncertainty. Fit offset/drift on calibration data, then hold it fixed for
validation. Preserve original files for improved alignment later. For better
transport timing use a stable wired Pi route if your setup supports one; the
logger still records uncertainty when the bridge is reached through Wi-Fi.

Record the marker mounting transform as well:

```text
R_world_body = R_world_marker @ R_marker_body
p_world_body = p_world_marker - R_world_body @ body_to_marker_position_body_m
v_body = R_world_body.T @ derivative(p_world_body)
```

`R_marker_body` maps body-frame vectors into the marker frame.
`body_to_marker_position_body_m` runs from the estimator's trunk origin to the
QTM rigid-body origin, expressed in body coordinates. Measure these; a marker
offset produces apparent translation during turning if left uncorrected.

## Export reference velocities after alignment

Copy `experiment/config/estimator_reference.template.json` to a run-specific
calibration file. Fill in the measured mounting transform and clock fields:

```text
pi_time_s = (qtm_time_s - qtm_origin_s) * scale + pi_origin_s
```

The anchors describe the same instant in the two clocks; `scale` accounts for
relative clock rate. `uncertainty_s` records your remaining time uncertainty;
`provenance` describes how you obtained it. Unknowns are `null` in the template
so the exporter cannot silently assume a coincident marker origin or synchronized
clocks. A nonzero timing uncertainty remains attached to every velocity label.

```bash
/usr/bin/python3 experiment/export_calibration_reference.py logs/calibration_trial_01 \
  --calibration logs/trial_01_calibration.json \
  --out logs/calibration_trial_01_reference
```

Outputs:

- `reference_velocity.jsonl`: body-origin poses and body/world velocity with
  validity, uncertainty, and derivative-window information.
- `aligned_samples.jsonl`: each recorded robot sample and its interpolated
  reference velocity. Unavailable labels remain invalid; no extrapolation.
- `mocap_poses.jsonl`: marker poses in the schema consumed by the package’s existing
  `compare_mocap_velocity.py`. These timestamps already use the Pi clock; use an
  identity time map in that comparison and retain the same mounting transform.
- `calibration.json` and `summary.json`: the exact configuration and label counts.

The derivative is a centered local quadratic fit (default five frames), after
correcting the marker offset. It uses actual sample times and rejects windows
touching tracking loss or gaps over `max_reference_gap_s`. It is an **offline
reference**, not a causal estimator for motor control. Inspect residual/noise and
choose the window on fitting data. Timestamp resets/reordering stop export for
explicit review rather than silently sorting separate measurement epochs.

## Use with the walking-package estimator

The 48 values are the actor's observation vector, not 48 independent states to
recover from MoCap. The estimator needs IMU, joints, torque/contact information;
MoCap supplies an independent body-velocity target for evaluating/calibrating it.
Factory remote walking does not reveal the policy's previous 12 actions, so this
recorder does not fabricate closed-loop 48D policy observations or policy actions.

The recorded native payloads can be passed directly to the existing decoder:

```python
# In the policy deployment Python environment, with its modules on sys.path:
from replay_lowstate import decode_state

# row is one JSON object from robot.jsonl; c/ec are your hardware/estimator configs.
if row['valid']:
    sensors = decode_state(bytes.fromhex(row['payload_hex']),
                           row['pi_sample_monotonic_ns'] / 1e9, c, ec)
```

Use the estimator's existing reset, yaw alignment, joint mapping and timing
conventions during replay. Do not equate operator motion labels with per-foot
contact truth, remote stick position with measured speed, or camera tracking
success with no foot slip. This change records calibration evidence and prepares
velocity targets; it does not fit/promote estimator parameters automatically.

## Offline checks

```bash
/usr/bin/python3 -m unittest discover -s test -p 'test_estimator_calibration.py' -v
```

Tests use synthetic sensor packets and localhost TCP/UDP servers. They cover
clock bounds, CRC, packet lengths, positive yaw and 180-degree rotations,
occlusion, mounting offsets during turns, drift, missing labels, stream failure,
output preservation and the recorder's exact outgoing request types. No robot
commands or physical experiments are run by the tests.

Protocol references: [QTM RT protocol](https://docs.qualisys.com/qtm-rt-protocol/),
[Qualisys 6DOF rotation convention](https://cdn-content.qualisys.com/2016/06/Webinar-6DOF.pdf).
Local references: `test/run_mocap_qualisys_for_dog.py`,
`src/DenseTrajectoryTracker.py`, `experiment/decode_native_go1_pcap.py`,
`${GO1_POLICY_SOURCE}/deployment/LIVE_SHADOW.md` and
`${GO1_POLICY_SOURCE}/deployment/compare_mocap_velocity.py`.
