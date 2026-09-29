# Joint torque and impedance experiments

Use this workflow to develop **joint effort references, command/feedback
analysis and low-level entry/exit behavior**. For body waypoints and timed x/y/yaw
paths, use the [trajectory guide](GO1_TRAJECTORY_TRACKING.md). The
[project overview](../README.md#choose-your-task) also links the separately
installed walking package.

## What is controlled and what is measured

The parent C++ runner has a 500 Hz joint-control state machine with remote-stop,
feedback freshness and command-watchdog handling. Its motor command contains
position and velocity targets, stiffness/damping gains and feed-forward torque.
An impedance command can contribute all three terms:

```text
predicted command effort = Kp * (q_target - q)
                         + Kd * (dq_target - dq)
                         + tau_feedforward
```

A small feed-forward waveform does not imply equally small total joint effort.
“Pure torque” and a torque overlay on position control are distinct experiments.
The analyzer also records SDK `tauEst`; that estimate and the calculated command
effort are not independent calibrated torque measurements.

## Entry points and current status

| Entry point / mode | Purpose | Current scope |
| --- | --- | --- |
| [`run_go1_torque_cluster.py`](../experiment/run_go1_torque_cluster.py) | Build SDK-free simulator; run six waveform cases and four stop/fault cases | Offline; recorded 10/10 pass |
| `go1_lowlevel_simulator --dry-run` | Develop the same control state machine without SDK hardware access | Offline implementation checks |
| `go1_lowlevel_experiment` | SDK-enabled parent experiment runner on the Pi | Operator-run modes and locks described in the [low-level manual](GO1_LOWLEVEL_EXPERIMENT.md) |
| `prone-engagement` | Bounded joint engagement, contact confirmation and release | Normal grounded trial accepted at Kp≤1, Kd=1 and zero feed-forward torque; active cancellation/remote-stop cases remain deferred |
| `torque-sine` | Single-joint torque reference for a supported setup | Hardware CLI exists with support confirmation; no independent hardware torque-tracking acceptance is recorded, and no support stand is established for this setup |
| Standing/ground torque, low-rise, squat and leg-lift modes | Future supported action/torque experiments | Parent hardware paths remain locked |
| Former `run_torque_tracking.py` | February Python torque prototype; disabled in August | Removed from the working tree; use the current C++ workflow above |

The accepted prone trial validates that bounded normal entry/release. It does
not release a standing takeover or qualify torque bandwidth. The separate
walking package has recorded motion, but its policy/PD joint tracking is a
different result. Neither workflow is an automatic prerequisite for running the
other's already-established procedure.

The 2026-09-29 [workflow review](GO1_WORKFLOW_REVIEW.md) re-ran all ten torque
software cases and the SDK command-adapter/transport checks successfully. The
trajectory repairs did not modify the September C++ torque implementation.

## Software work on Ubuntu

Use the project's analysis Python environment, CMake and a C++ compiler. The
simulator target does not link the SDK hardware library. For new or changed
code, the complete software block is:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 -B experiment/run_go1_torque_cluster.py
```

It tests 0.10/0.20 N·m command amplitudes at 0.5/1/2 Hz and simulated cancellation,
remote stop, double Ctrl-C and watchdog cases. Require
`torque_software_cluster=PASS (10/10)` and retain the printed archive under
`logs/torque-software/`. Results remain explicitly **SIMULATED**; the comparison
with simulated feedback does not accept physical torque tracking.

The cluster already performs log analysis. For a separately obtained parent
joint log, an optional offline analysis command is:

```bash
read -r -p 'Path to the parent low-level CSV: ' GO1_LOWLEVEL_LOG
python3 -B experiment/analyze_lowlevel_log.py "$GO1_LOWLEVEL_LOG" --no-plots
```

Use the analyzer's joint-labelled CSV schema. A high-level body-path CSV from
`experiment/traj/` is a different format. Preserve raw data, configuration,
source/binary identity and operator observations with every physical result.

## Hardware work on the Pi

Follow the [condensed low-level status](GO1_LOWLEVEL_EXPERIMENT.md#21-retained-results-and-remaining-legacy-scope)
for the parent's remaining work. It identifies the accepted trial and deferred
exit/transition questions. Former 2.1.x diagnostics are historical evidence;
new access does not require repeating them all. Removing `--dry-run` from a
software command is not a hardware procedure.

If the goal is to operate the existing five-second walking package now, use its
[operator sequence](GO1_WALKING_OPERATIONS.md). It has a separate deployment,
command owner, estimator, controls and restoration path. The parent controller's
support GUI and Programming Module handling do not belong in that sequence.

A future hardware torque-tracking report must state the command terms, joint,
reference waveform, torque feedback source, timing and physical setup, with
measured results and the experiment's acceptance criteria. This repository's
current software pass and prone/walking records do not supply that acceptance.

## Historical entry points

Git history distinguishes the original prototypes from later verification work:

| File / change | Recorded date | Commit |
| --- | --- | --- |
| `experiment/run_torque_tracking.py` first added | 2026-02-20 | `ca72025` |
| `src/DenseTrajectoryTracker.py` added; file-driven waypoint entry updated | 2026-02-24 | `58552d0` |
| `experiment/run_joint_control.py` first added | 2026-04-18 | `116f15b` |
| Old Python torque sender replaced by its disabled guard | 2026-08-30 | `43f9a0a` |

The torque prototype and dense tracker therefore date from the same week,
four days apart. Both obsolete Python entry points and the failed shared-control
helpers were purged on 2026-09-29; the September C++ torque workflow was preserved.
The [cleanup record](../archive/failed_shared_control/README.md) retains the old
measurement evidence. No file
literally named `run_dense_tracking.py` was found in the available Git history;
the current dense implementation is `DenseTrajectoryTracker.py`, entered through
`run_waypoints_from_file.py`. Dates describe source history, not successful
hardware acceptance.
