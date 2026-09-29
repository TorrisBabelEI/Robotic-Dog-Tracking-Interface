# Local paths for Go1 procedures

Shared instructions use roles: **operator**, **policy developer**, **walking
package**, **parent controller**, **Ubuntu workstation** and **onboard Pi**.
Names of people, workstation accounts and personal parent folders are not part
of the procedure. Upstream repository URLs and copyright attribution retain
their original identities.

Start an Ubuntu terminal anywhere inside this checkout. The commands below
resolve the repository root and load optional machine-specific settings:

```bash
cd "$(git rev-parse --show-toplevel)"
source experiment/go1_paths.sh
```

This workstation's existing source and Pi installation paths are already mapped
in the ignored `.go1-paths.env`; no setup change is needed here. The file is local
shell configuration: source only a file you control. On another workstation,
copy `experiment/config/go1_paths.env.example` to `.go1-paths.env` and fill in
its actual locations once. Do not rename a deployed package to match an example.

| Variable | Location / meaning |
| --- | --- |
| `GO1_PROJECT_ROOT` | This checkout, determined by the loader |
| `GO1_POLICY_SOURCE` | Separate policy source repository on Ubuntu |
| `GO1_POLICY_REFERENCE` | Supplied operation command sheet on Ubuntu |
| `GO1_PI_POLICY_ROOT` | Existing package installation root on Pi, configured on Ubuntu |
| `GO1_POLICY_ROOT` | The same Pi root passed into an interactive SSH session |

The walking procedure passes only the configured Pi root into its SSH session;
Ubuntu environment variables do not automatically exist on the Pi. Commands
quote path variables so locations containing spaces work. `go1_walking_reference.json`
uses `${GO1_POLICY_SOURCE}` as a documented path placeholder, not a literal
pathname or a configuration automatically consumed by the controller.

Evidence paths beginning with `logs/` are relative to this checkout. Historical
commands and report prose have normalized personal labels and workstation paths;
raw recordings, source snapshots, hashes and actual deployed filenames retain
their original identities. This is a portability edit, not evidence redaction.

The older `experiment/run_highlevel_tracking.py` example now requires an
explicit `--trajectory` file instead of a personal default. Its control loop is
unchanged; it is separate from the supervised walking procedure. The older
failed shared-control code and trust-animation script were subsequently purged;
the [cleanup record and data](../archive/failed_shared_control/README.md) remain.

## File names and generated metadata

The trajectory sample directory is `experiment/traj/mppi_reference/`. Its three
data files retain their original bytes and hashes. Generated Python caches are
excluded from version control because bytecode can embed the compiler's local
source path.

New walking review files identify their input by artifact directory name and
retain manifest/source hashes instead of storing an absolute workstation path.
Calibration recording metadata stores configuration filenames, the configuration
hash, and `out: "."`; exporter summaries identify recordings by directory name
and hash both input streams. The `source` and `recording` fields are labels,
not absolute paths to open on a different machine.

Use an operator role or trial ID in free-text notes when sharing records. Existing
logs, copied package metadata and historical source snapshots may still contain
personal text or original paths. These evidence originals are not rewritten by
this change. The local `.go1-paths.env` holds actual installation locations and
is excluded from version control; keep it out of shared folders or archives.

The current Ubuntu checkout and the independent source/Pi installation are not
relocated by these internal renames. A physical checkout move is a separate
choice because open tools and build caches can refer to its current directory.
Earlier Git commits also retain their original content; working-tree edits do
not rewrite that history. Original copyright and upstream attribution remain.

### Verification of the path cleanup — 2026-09-29

The three renamed trajectory files matched their pre-rename SHA-256 values.
A scan of 187 current project files (including binary contents; excluding
ignored local files, historical logs and third-party dependencies) found no
remaining known personal names or personal workstation-path prefixes in their
names/content. This scope does not include earlier Git commits.

The nine walking-review tests passed in `dog_ctrl`; all 20 calibration tests
passed with the guide's `/usr/bin/python3`. An initial combined run under the
Conda interpreter stalled in the synthetic network tests and was terminated;
no robot connection was involved. A separate exporter check preserved all 97
valid synthetic velocity results and verified both input hashes while confirming
that its summary omitted the private parent path. Python syntax and diff checks
also passed. No Pi execution, installation rename or hardware run was performed.
