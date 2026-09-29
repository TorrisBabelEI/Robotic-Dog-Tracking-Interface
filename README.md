# Robotic-Dog-Tracking-Interface

Tools and research examples for **body trajectory tracking** and **joint torque
experiments** on the Unitree Go1. The project also reviews an independently
installed walking policy and records Qualisys reference data for estimator
calibration. These workflows use different command interfaces and have different
hardware acceptance records.

## Choose your task

| What you want to do | What the robot receives | Start here |
| --- | --- | --- |
| Follow waypoints or a timed body path | Factory high-level forward/lateral velocity and yaw-rate commands, using Qualisys pose feedback | [Trajectory tracking guide](docs/GO1_TRAJECTORY_TRACKING.md) |
| Develop or evaluate joint torque/impedance control | Low-level joint position, velocity, gains and feed-forward torque commands | [Torque experiment guide](docs/GO1_TORQUE_TRACKING.md) |
| Run the existing short walking-policy demonstration | Onboard policy joint targets through its own supervised low-level package | [Walking operator procedure](docs/GO1_WALKING_OPERATIONS.md) |
| Record reference motion for velocity-estimator calibration | Factory remote remains in control; the recorder observes robot and Qualisys data | [Calibration capture guide](docs/ESTIMATOR_CALIBRATION_CAPTURE.md) |

**Body trajectory tracking** asks where the robot should move: position in
metres and heading in radians. **Joint torque tracking** asks how closely a
joint's torque follows a reference in N·m. A robot following a path does not by
itself demonstrate torque accuracy. Likewise, a walking policy tracking joint
positions does not establish body-path accuracy or independently measured torque
tracking.

The existing high-level trajectory scripts are not connected to the separately
installed walking policy as an interchangeable backend. That connection would
require additional implementation and verification.

## When you first get access

1. Choose one workflow above. Each guide identifies the relevant entry points,
   configuration, data format and current operating scope.
2. Start Ubuntu commands from this checkout. Follow [local path setup](docs/GO1_LOCAL_PATHS.md)
   if a guide uses the separate policy source or Pi installation; machine-specific
   locations are held in an ignored local file.
3. Use offline analysis or software verification for code/data work. For an
   actual robot session, use the selected workflow's operator procedure. Do not
   combine the parent prone controller's GUI/process handling with the walking
   package's controls, or launch competing command senders.

The robot's onboard Pi and the Ubuntu workstation have different roles. The
low-level hardware loop and installed walking package run on the **Pi**; offline
analysis runs on **Ubuntu**. The older high-level trajectory examples run on an
**Ubuntu workstation with the SDK Python binding, a working robot UDP route and
Qualisys connectivity**. SSH access alone does not supply that direct UDP route.

## Checkout and software setup

For a new checkout:

```bash
git clone https://github.com/tianyuzhou-sam/Robotic-Dog-Tracking-Interface.git
cd Robotic-Dog-Tracking-Interface
git submodule update --init --recursive
```

The original development setup uses Ubuntu 20.04 and an onboard ARM64 Raspberry
Pi. Use a C++ compiler and CMake for the parent low-level tools. The vendored
[Unitree SDK](https://github.com/unitreerobotics/unitree_legged_sdk/tree/go1)
provides the hardware headers/libraries; installing or building software does
not run a controller.

For offline low-level log analysis, install in your chosen Python environment:

```bash
python3 -m pip install -r requirements-analysis.txt
```

This installs NumPy and Matplotlib; it is **not the complete trajectory-control
environment**. The [trajectory guide](docs/GO1_TRAJECTORY_TRACKING.md#software-and-site-configuration)
lists its additional imports and SDK binding requirements. The walking reviewer
uses Python 3.9+ (`dog_ctrl` on the current workstation); the calibration guide
uses its documented system Python commands.

## Torque software entry point

For a new checkout or changed torque code, this whole block builds an SDK-free
simulator and exercises the waveform and fault cases without connecting to the
robot:

```bash
python3 -B experiment/run_go1_torque_cluster.py
```

Expected completion: `torque_software_cluster=PASS (10/10)`. The printed archive
under `logs/torque-software/` contains the simulated logs, analysis and source
identity. This block is already accepted for the recorded source; it is not a
routine prerequisite for every walking or trajectory session. See the
[torque guide](docs/GO1_TORQUE_TRACKING.md) for what those results establish.

## Current hardware scope

| Workflow | Recorded status and limit |
| --- | --- |
| High-level body trajectory examples | MPC/dense logic repaired on 2026-09-29; 31 offline regressions and ideal-model checks passed. Corrected source still needs physical validation. See the [workflow review](docs/GO1_WORKFLOW_REVIEW.md). |
| Parent low-level controller | Communication preflight and bounded prone engagement/release accepted. Torque software checks passed. Standing, low-rise, squat and leg-lift hardware remain locked; independent hardware torque-tracking acceptance is still pending. |
| Separate supervised walking package | Recorded five-second hardware runs with its onboard velocity estimator. Use the [exact installed-package procedure](docs/GO1_WALKING_OPERATIONS.md) and [profile review](docs/GO1_WALKING_INTEGRATION.md); these runs establish neither general path tracking nor measured torque accuracy. |

The February Python torque sender and the unsuccessful April
`run_joint_control.py` shared-control experiment have been **purged from the
working tree**, including their obsolete entry points. September C++ torque
work remains intact. Historical measurements and the cleanup record are in
[the evidence archive](archive/failed_shared_control/README.md).

Pending external-policy work is listed in the [handover status](docs/GO1_POLICY_HANDOVER.md),
including the September 29 bipedal fault-recovery and training-contract issues.
Raw reference data and logs follow the [local data policy](docs/GO1_DATA_POLICY.md);
source, configuration, manifests and small test fixtures remain versioned.

## Where the work lives

| Location | Purpose |
| --- | --- |
| `experiment/run_waypoints*.py`, `run_many_waypoints.py` | High-level body path and waypoint entry points |
| `src/DenseTrajectoryTracker.py`, `experiment/src/ModelPredictiveControl.py` | Body-pose feedback and high-level velocity control |
| `src/go1_lowlevel_experiment.cpp` | Parent low-level experiment state machine |
| `experiment/run_go1_*_cluster.py`, `analyze_lowlevel_log.py` | Offline experiment verification and analysis |
| `experiment/review_go1_walking.py` | Offline review of the separately versioned walking package and captures |
| `experiment/record_estimator_calibration.py`, `export_calibration_reference.py` | Qualisys/robot recording and reference-velocity export |
| `experiment/config/` | Site configuration, reference metadata and a local-path template |
| `experiment/traj/` | Historical trajectory inputs and outputs; each script has its own schema |
| `logs/` | Local experiment evidence, excluded from version control |
| `archive/failed_shared_control/` | Retained failed-experiment evidence and cleanup record; source purged |

The [low-level manual](docs/GO1_LOWLEVEL_EXPERIMENT.md) retains the current parent
controller status. Its former 2.1.x checks are in a separate
[historical record](docs/GO1_LOWLEVEL_2_1_HISTORY.md), not a test queue for new users.
