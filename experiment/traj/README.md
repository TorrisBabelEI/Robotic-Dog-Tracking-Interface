# Local trajectory inputs and outputs

Place reviewed CSV/joblib body paths here, or supply their existing local path
with `experiment/run_waypoints_from_file.py --trajectory PATH`.

The controllers also write their historical row-oriented CSV outputs here. Raw
arrays, recordings and plots are ignored by Git and are not bundled with a fresh
checkout. This README keeps the output directory present.

See the [trajectory guide](../../docs/GO1_TRAJECTORY_TRACKING.md) for input schemas
and the [data policy](../../docs/GO1_DATA_POLICY.md) for retention rules.
