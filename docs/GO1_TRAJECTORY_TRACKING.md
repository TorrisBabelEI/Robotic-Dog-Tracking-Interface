# Body trajectory and waypoint tracking

Use this workflow when the objective is a **body path in the laboratory frame**:
waypoints or a timed x/y/yaw reference.
Return to the [project overview](../README.md#choose-your-task) to choose another
workflow. Joint torque experiments have a [separate guide](GO1_TORQUE_TRACKING.md).

The 2026-09-29 [workflow review](GO1_WORKFLOW_REVIEW.md) corrected the original
run-loop and timing defects. Offline regression and ideal-model checks pass;
these changes have not yet been exercised on the physical robot.

## Control interface and entry points

The closed-loop path is:

```text
Waypoints or body trajectory + Qualisys pose
  -> MPC or dense-path controller on Ubuntu
  -> body velocity [vx, vy] and yaw rate
  -> Unitree high-level controller -> robot motion
```

The factory controller supplies the underlying gait/joint actuation. These
scripts do not command a joint torque waveform or use the parent C++ low-level
state machine. The separate supervised walking policy also has its own runtime;
it is not currently selected as a backend by these scripts.

| Entry point | Input / behavior | What to configure |
| --- | --- | --- |
| [`run_waypoints.py`](../experiment/run_waypoints.py) | Sparse x/y waypoints with MPC and Qualisys feedback | Waypoints, run time and MPC settings in the script; initial pose comes from Qualisys |
| [`run_many_waypoints.py`](../experiment/run_many_waypoints.py) | Larger waypoint example using the same MPC family | Its waypoint list and timing |
| [`run_waypoints_from_file.py`](../experiment/run_waypoints_from_file.py) | CSV/joblib body-path input; dense by default, optional MPC branch | Required `--trajectory`; `--mode`, `--dt` or `--total-time`, `--no-yaw` and resampling options |
| [`run_highlevel_tracking.py`](../experiment/run_highlevel_tracking.py) | Open-loop replay of high-level velocity commands from a trusted joblib file | Required `--trajectory` argument; this is not Qualisys path feedback |

The unsuccessful April `run_joint_control.py` experiment and its exclusive
helpers have been removed from the working tree. Its CSV/plot evidence and
cleanup record remain in [the archive](../archive/failed_shared_control/README.md).
Its joystick/trust features are not part of this trajectory workflow.

## Software and site configuration

Run from the repository root. The hardware-facing trajectory modules import
`robot_interface` from `externals/unitree_legged_sdk/lib/python/amd64` and expect
a compatible Linux x86-64 SDK Python binding. Their source imports NumPy,
Matplotlib, CasADi, `transforms3d` and `qtm`; joblib input loading additionally uses joblib. The lightweight
`requirements-analysis.txt` does not provision all of these dependencies.
The existing environment should be preserved when it already works; package
compatibility is not pinned by this guide.

The optional SDK Python wrapper is selected by `-DPYTHON_BUILD=ON` in CMake;
it must be built for the interpreter/architecture that runs the examples.
The default CMake build leaves that wrapper and SDK example executables off.
Building the C++ low-level simulator does not build the Python wrapper.

Read/edit [`experiment/config/config_dog.json`](../experiment/config/config_dog.json)
for the actual `QUALISYS.IP_MOCAP_SERVER` and `QUALISYS.NAME_SINGLE_BODY`.
QTM must provide the selected 6DOF body, and the Ubuntu Ethernet interface must
reach that server. These older trackers use the `qtm` client; the newer passive
[calibration recorder](ESTIMATOR_CALIBRATION_CAPTURE.md) has a different client.

The examples create high-level UDP on local port 8080 addressed to
`192.168.123.161:8082`. Confirm the workstation's configured robot route for this
site. The onboard-only walking procedure uses SSH and its own command owner;
copying its connection steps alone does not route a workstation SDK socket.
Network changes belong to the site setup, not a generic firewall flush/default
route replacement in the README.

## Body-path data format

`run_waypoints_from_file.py` reads a **row-oriented, headerless** CSV or joblib
array. Columns are samples: the first two rows are x and y in metres; an
optional third row is yaw in radians. It transposes those rows into an N×2 or
N×3 trajectory for `DenseTrajectoryTracker`.

For example, this is three x/y/yaw samples, not three rows of time-stamped poses:

```text
0.0,0.1,0.2
0.0,0.0,0.0
0.0,0.0,0.0
```

Time comes from the script's `dt` or `total_time`, not a CSV timestamp column.
The default interval is `--dt 0.08` s; yaw is used when a third row is present
unless `--no-yaw` is supplied. N samples span N−1 intervals. Resampling preserves
that duration and both endpoints, including shortest-arc heading interpolation.
`--total-time` defines the interval from the first to the last reference sample. The MPC branch instead takes x/y waypoints and its own horizon/time
settings. Keep the file's coordinate frame and yaw convention consistent with
the selected tracker's QTM conversion; QTM position is converted from mm to m.

The file-driven entry requires `--trajectory`; it no longer selects the missing
legacy file by default. `experiment/traj/mppi_reference/`
can contain local neutral-named reference files; these raw inputs are ignored by
Git and are not supplied by a new checkout. File presence alone does not establish
that their frame, timing or shape matches your chosen experiment. Load only
trusted joblib/pickle data. Logged CSVs in `experiment/traj/` have different row
layouts and are not automatically valid input trajectories.

For `run_highlevel_tracking.py`, rows 8–10 (zero-based) of the joblib array are
forward velocity, lateral velocity and yaw rate, replayed at a fixed 0.02 s step.
They are commands, not x/y/yaw targets; a body-path CSV cannot be substituted.

## Control-loop behavior and offline verification

Both trackers use the shared [pose runtime](../src/go1_highlevel_runtime.py).
It checks for new QTM frame numbers and valid finite poses, applies a 0.25 s
feedback deadline and the configured whole-run timeout, and sends zero body
velocity/yaw rate in SDK mode 0 on completion, cancellation or failure before
closing the QTM stream and saving data. Initial frame acquisition is limited to
3 s. A repeated frame cannot renew the feedback deadline. These are software
command stops; physical stopping behavior still depends on the high-level robot
controller and UDP delivery.

MPC now solves from the latest measured pose at its configured control period
(0.2 s in the examples). Failed, nonfinite, out-of-bounds or late results cannot
be sent. Solving runs outside the QTM event loop; a late background result cannot
send commands after shutdown. The horizon objective includes every predicted
state and input once. `iniState` remains an API compatibility argument; hardware
initialization no longer uses it as a substitute for the first measured pose.

Dense tracking begins its clock when the starting position/heading is reached,
without the former three-second sleep. Yaw follows the short arc across ±π,
and its damping term uses measured angular velocity and actual elapsed time.
At the reference endpoint it continues correcting position/heading until within
0.05 m / 0.1 rad or the whole-run timeout. This changes the old behavior that
stopped on elapsed time regardless of remaining position error.

The following offline block is already passed for the reviewed source. Re-run
it after trajectory-control changes; it constructs only fake SDK/QTM connections:

```bash
cd "$(git rev-parse --show-toplevel)"
conda activate dog_ctrl
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python3 -B -m unittest discover \
  -s test -p test_trajectory_control.py -v
```

Recorded result: **31 tests passed**. This is separate from the C++ torque
simulation; no old 2.1.x hardware checks were added back to the procedure.

## From access to a physical session

1. Select the entry point and its input schema above. Configure the actual lab
   frame, path, timing, bounds and site addresses in that script/configuration.
2. Use the existing trajectory environment and review its start/stop behavior
   with the operator. These research scripts have no general `--dry-run` switch
   and do not inherit the walking package's guardian/restoration procedure.
3. When the site setup and selected experiment are ready, the operator launches
   only that controller from the repository root. The corrected source still
   needs operator-supervised physical validation. For the sparse-waypoint entry:

   ```bash
   python3 experiment/run_waypoints.py
   ```

   That is a **hardware command**, not an installation check. For a configured
   file-driven dense experiment, the hardware command is:

   ```bash
   read -r -p 'Reviewed body-path CSV or trusted joblib file: ' GO1_TRAJECTORY_FILE
   python3 -B experiment/run_waypoints_from_file.py \
     --trajectory "$GO1_TRAJECTORY_FILE" --mode dense --dt 0.08 --timeout 120
   ```

   Replace `--dt` with `--total-time` when the complete reference span is known.
   `--mode mpc` uses x/y waypoints and its own 0.2 s prediction step; it does not
   promise to arrive at each waypoint on the dense reference timetable.
   `python3 -B experiment/run_waypoints_from_file.py --help` lists options without
   constructing an SDK connection.
4. Retain the input, source/configuration identity, timing and measured output
   from the same run. Review body position/yaw error and any tracking loss or
   intervention. Keep those results distinct from joint torque metrics.

With saving enabled, dense tracking writes `experiment/traj/dense_tracking_*.csv`,
including measured state and commands in its row layout. Keep the input
reference alongside the output; these logs are not the header-based joint logs
expected by `analyze_lowlevel_log.py`.

New MPC logs align the time, measured-state and computed-command rows sample
by sample. Older MPC files can contain an extra initial-state/time sample. Both
trackers record computed commands; these CSVs are not an acknowledgement from
the robot that a UDP command was applied.

No new trajectory hardware run was performed or accepted during this repair.

The dense tracker was added on 2026-02-24 (`58552d0`), four days after the old
Python torque prototype. The file-driven waypoint entry itself predates that
addition; it was updated to use the dense tracker. See the
[source timeline](GO1_TORQUE_TRACKING.md#historical-entry-points) for the distinction
between the February prototypes and the later retired shared-control experiment.
