# Trajectory and torque workflow review — 2026-09-29

The two original trajectory methods were preserved by the obsolete-code cleanup,
but their hardware loops contained defects. Those defects have now been repaired
and checked offline. The current C++ torque workflow and the separately reviewed
walking package passed their applicable software/integrity checks. This review
ran no robot controller, sent no robot packets and made no Pi changes.

## Results

| Workflow | Result | What remains unverified |
| --- | --- | --- |
| Sparse waypoint MPC | Latest-pose, stop, failure/deadline handling and objective indexing repaired; regression checks pass | Corrected loop on real Qualisys/robot, including its 0.2 s command period |
| Dense timed body path | Clock, heading interpolation/damping, duration-preserving resampling and endpoint handling repaired; regression checks pass | Physical path/heading error, site coordinate convention and stopping behavior |
| September C++ torque | 10/10 simulator cases passed; SDK command adapter and transport regressions passed | Independent physical torque-tracking accuracy/bandwidth |
| Separate walking package | Local `normal_walking` mirror: 104/104 listed files match; tuning representations agree | Current Pi installation was not inspected; release metadata still reports `hardware_validated=false`, `production_release=false` |

The walking integrity check establishes consistent local package contents. It
does not replace the [installed-package procedure](GO1_WALKING_OPERATIONS.md) or
change the scope of earlier five-second walking results.

## Corrected behavior

1. **MPC uses the current measured pose.** Previously it solved from the preceding
   pose, including the hard-coded initial state on the first frame. A mock
   measurement at x=1 m reproduced a solve using x=−1.5 m before the repair.
2. **Failed solves and completion stop command output.** A mocked failed solve
   previously transmitted 0.2 m/s. Reaching the last waypoint then raised `Done`
   without transmitting a zero command. Failed/nonfinite/out-of-bounds results
   now stop; final arrival returns normally with a zero-velocity command.
3. **The MPC objective covers the full horizon.** It previously counted the first
   predicted state twice and omitted the last state/input. Every predicted state
   and input is now penalized once, retaining the existing weights and bounds.
4. **Feedback and run deadlines work without callbacks.** Both methods now await
   their full run lifetime. Missing/stale/replayed frames, invalid poses,
   disconnects, computation overruns and cancellation trigger a software stop.
   The optimizer runs in a worker that cannot send commands; its late result
   cannot restart motion. QTM streaming is closed before the controller returns.
5. **Dense tracking no longer skips its opening seconds.** Previously its clock
   began before a blocking three-second sleep. At dt=0.08 s the next callback
   could begin around reference index 38.5. Tracking now starts immediately when
   the starting pose is reached, with no callback sleep.
6. **Heading math is continuous across ±π.** The midpoint between +179° and −179°
   follows the short arc through 180°. Damping opposes measured rotation rather
   than subtracting the derivative of target error and reversing the initial
   response to a heading step.
7. **Resampling preserves duration.** The old dt-based preprocessing reduced a
   500-point, 0.01 s path from 4.99 s to 0.99 s while still claiming a 20 Hz cap.
   Resampling now uses uniform reference times, includes both endpoints, and
   counts N−1 intervals. Endpoint correction continues until tolerance or timeout.
8. **The file input is explicit.** The absent legacy default was replaced with
   required `--trajectory` and documented timing/mode options. `--help` does not
   construct a robot connection.

Implementation: [shared pose runtime](../src/go1_highlevel_runtime.py),
[MPC hardware loop](../experiment/src/ModelPredictiveControl.py),
[MPC objective](../src/OptimalControl.py),
[dense tracker](../src/DenseTrajectoryTracker.py),
[path resampling](../src/go1_trajectory_input.py), and
[file entry](../experiment/run_waypoints_from_file.py).

## Verification performed

- **31 offline regression tests passed**, including mocked successful runs,
  failed and slow MPC solves, missing/stale/duplicate/invalid QTM frames,
  disconnect, cancellation, timeouts, heading wrap, endpoint convergence and
  duration-preserving resampling. Real SDK/QTM connections are replaced by fakes.
- **Real CasADi/IPOPT core:** generated/compiled its C functions in a temporary
  directory; all 40 solves succeeded in an eight-second ideal planar straight-line
  test. Final position error was 0.00002865 m; maximum dynamic-constraint residual
  was 2.22e−16. Commands stayed inside the configured bounds. This is a model
  calculation, not a physical accuracy measurement.
- **Torque simulator:** all six waveform cases and four stop/fault cases passed.
- **SDK adapter:** damping, prone-engagement and torque effort fields preserved
  through SDK conversion/safety processing; no UDP object constructed.
- **SDK transport mock:** receive counters/CRC/errors/reset and guarded send
  handling passed; no transport constructed.
- **Cleanup integrity:** all 44 recorded source hashes matched before repairs.
  The two trajectory files in that cleanup baseline intentionally changed during
  this repair; the remaining 42, including the C++ torque sources, still match.

The regression command is recorded in the [trajectory guide](GO1_TRAJECTORY_TRACKING.md#control-loop-behavior-and-offline-verification).
These are completed checks, not a new sequence of Pi tasks to repeat.

Local evidence:

- `logs/workflow-review/review-AjIKQuZQ/`: original reproductions/source snapshots,
  before/after numerical results, regression output, SDK results, walking-package
  review and post-repair source hashes.
- `logs/torque-software/review-990l3j7o/`: complete 10-case simulation archive.

The standalone C++ audit used `CMAKE_DISABLE_FIND_PACKAGE_catkin=ON`: the local
Conda base interpreter lacks catkin's `empy` dependency. No ROS changes or package
installation were needed for these SDK-only checks.

## Operating scope

Use the [trajectory guide](GO1_TRAJECTORY_TRACKING.md) for input schemas, the
revised runtime behavior and operator commands. The next trajectory step needs
an operator with the real Qualisys/robot setup; software results alone do not
establish physical path tracking. The shared loop issues a high-level
zero-velocity/mode-0 command on exit, not a physical emergency stop.

The dense controller still uses proportional position feedback, so moving-path
lag is expected; the ideal-model checks are bounded examples, not a claim of
exact tracking for arbitrary paths. Torque measurement and supported action
qualification remain separate from these body-path repairs. No historical 2.1.x
checks have been reinstated.
