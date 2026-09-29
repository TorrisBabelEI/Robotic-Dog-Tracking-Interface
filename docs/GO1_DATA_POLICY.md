# Local reference data and recording policy

Raw experiment data stays on the workstation or the separately maintained
recording storage. Git retains source, procedures, configuration, manifests and
small intentional test fixtures.

The root `.gitignore` excludes:

- `logs/` and the existing separate walking-project copies.
- Raw CSV/joblib/NumPy arrays, plots, videos, packet captures and logs under
  `experiment/traj/`.
- `archive/*/data/`, while keeping each archive's README and manifest visible.
- Known top-level experiment outputs (`controlData.csv`, `squat_*.csv`),
  top-level captures/logs, Python bytecode and macOS resource-fork sidecars.

Patterns are scoped to artifact locations; CSV/JSON fixtures under `test/`,
configuration under `experiment/config/`, controller code and SDK submodules are
not broadly ignored by file extension.

On 2026-09-29, **75 raw-artifact paths were removed from Git's index**, so these
ignore rules also apply to previously tracked data. All payloads were retained
and hash-verified: 66 at their existing locations and nine at the locations used
by the earlier archive/person-neutral rename. The removals are staged; no commit
or history rewrite was made. Audit: `logs/policy-handover/git-data-cleanup.json`.

A fresh checkout will not receive these raw files after the cleanup is committed.
Select an available local body-path input with `--trajectory`, and obtain any
historical recording from its data custodian/storage when needed. Archive
manifests describe locally retained evidence; they do not imply that its payload
is bundled with a new checkout. Required small test fixtures should live under
`test/` with their provenance, rather than in the local trajectory-data directory.
