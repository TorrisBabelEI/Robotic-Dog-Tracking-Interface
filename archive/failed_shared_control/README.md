# Retired shared-control evidence and legacy cleanup

**Status: unsuccessful code purged; recorded evidence retained.** The operator
identified the shared-control work as an unsuccessful older experiment and
authorized removal of the older code while preserving September 2026 torque
work. Cleanup completed on 2026-09-29.

## Removed code

- The old `experiment/run_torque_tracking.py` entry point: originally added on
  2026-02-20 (`ca72025`), replaced by a disabled guard on 2026-08-30 (`43f9a0a`).
  The remaining guard has now been removed; it is not the September C++ workflow.
- The April 2026 `experiment/run_joint_control.py` shared-control runner.
- Its exclusive `ModelPredictiveControlObstacle.py` and `joystick_handler.py` helpers.
- Its `trust_animate.py` plotting script.

The temporary archived source copies were also purged. Earlier committed source
versions remain in Git history. The shared-control implementation sent high-level
body velocity/yaw-rate commands; it did not implement motor-joint torque tracking.
Its historical plots do not establish successful control.

## Retained evidence and current software

Two CSVs and four PNGs remain locally under `data/`, with unchanged bytes.
The payload directory is ignored by Git; the README and manifest remain
shareable. See the [data policy](../../docs/GO1_DATA_POLICY.md). The
[manifest](manifest.json) records their original/archive relative paths and
SHA-256 values, identifies the removed source, and records the current software
hashes checked before and after removal. Removed-snapshot hashes identify the
purged copies; they are not paths to retained source files.

The September C++ torque controller, software clusters, analysis and deployment
tools remain intact. Dense/waypoint tracking is separate and remains intact.
`src/keyboard_handler.py` remains because `experiment/remote_control.py` uses it.
The installed supervised walking package and Pi were not modified.

Choose current work from the [project overview](../../README.md#choose-your-task),
[trajectory guide](../../docs/GO1_TRAJECTORY_TRACKING.md) or
[torque guide](../../docs/GO1_TORQUE_TRACKING.md). Old process-detection patterns
may still mention the legacy sender because an older copy could exist on a Pi;
this local cleanup did not inspect or remove Pi installations.
