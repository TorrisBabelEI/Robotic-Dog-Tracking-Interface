# Existing Go1 walking package: operator sequence

Scope: the independently installed walking policy. For high-level body paths or parent joint torque experiments, choose the workflow in the [project overview](../README.md#choose-your-task).

Path notation and workstation setup: [local paths](GO1_LOCAL_PATHS.md).

Updated 2026-09-29 from `${GO1_POLICY_REFERENCE}` and the local
mirror of the deployed package. The referenced file is a text command sheet,
not a script to run with Bash. Its commands describe the existing hardware
workflow; this update did not execute them or contact the Pi.

**Use the installed five-second `normal_walking` package for this work while
policy development continues.** Compilation in the policy source directories does
not require another simulation cycle here. Do not copy files from `updates/`,
rebuild the sender, change gains, or replace an estimator/model just to start an
ordinary session. Pi installation, clock refresh and physical runs must happen
between other Pi trials/installations, not concurrently with them.

The old parent prone GUI, support probe, Programming Module stop/start script
and Kp<=1 test sequence belong to a different controller. They are not steps in
this workflow. This package owns its own handover and factory restoration.

## What is already integrated

The walking worker uses a **body linear-velocity estimator**. The traced path is:

```text
Native Go1 feedback / device tick
  -> boot-bound clock and common-time alignment
  -> measured foot-contact + IMU/joint-kinematics Kalman filter
  -> body linear velocity + validity/contact checks
  -> 48-element policy observation
  -> frozen actor -> governed joint targets -> SDK sender
```

`worker.py` constructs `VelocityIntegrityGuard(MeasuredContactKalmanEstimator(...))`.
The native filter has 21 states: body position (3), velocity (3), foot positions
(12), and acceleration bias (3). It uses the measured orientation and nonlinear
leg geometry to form its observations. This is not a newly trained neural
velocity estimator, nor does it require a separate Ubuntu estimator process.
Its body-velocity output is scaled into the first three actor-observation
entries. The sender starts the worker with the native actor backend.

The measured-contact implementation uses per-foot raw-force hysteresis from
`estimator.json`. It does not use a motor-torque inverse-dynamics model to infer
contact. The native filter, configuration, alignment library and worker are all
hash-bound in the reviewed local release manifest. The successful five-second
recordings already used this estimator path.

**Clock refresh and estimator calibration are different operations.**
`go1.py refresh` passively records telemetry after a boot change, fits the device
tick period and updates the clock binding, its worker reference and release
hashes. It does not retrain the actor or recalibrate contact thresholds.
The current helper visits every installed package it knows about, including
long walking, bipedal and joint calibration; it can stop on a package that is
being updated. Do not bypass that failure or assume every package was unchanged
if a later package fails after an earlier one refreshed successfully.

The previous review's contact-calibration and export-metadata questions remain
recorded in [the integration review](GO1_WALKING_INTEGRATION.md#current-mirror-is-a-later-configuration).
They are not evidence that the velocity estimator is missing. This procedure
records the exact installed package for a supervised session; it does not
promote new weights or certify general torque/velocity accuracy. If the live
package differs from the operator baseline maintained by the policy developer,
stop after the identity check and review that difference before executing it.

## A. Ubuntu to Pi: status, refresh when needed, verify

Use established floor-based factory startup. Before an active trial, the robot
must be stationary in factory standing, with all four feet bearing weight,
remote ON/connected and controls released, catch rope slack and clear of the
legs, and a clear walking area. Do not start this walking workflow from the
old belly-on-floor prone position or with feet suspended.

On **Ubuntu**:

```bash
(
set -e
cd "$(git rev-parse --show-toplevel)"
source experiment/go1_paths.sh
: "${GO1_PI_POLICY_ROOT:?Set GO1_PI_POLICY_ROOT in .go1-paths.env}"
printf -v GO1_PI_SHELL 'export GO1_POLICY_ROOT=%q; exec bash -l' "$GO1_PI_POLICY_ROOT"
ssh -t pi@192.168.12.1 "$GO1_PI_SHELL"
)
```

The SSH command sets `GO1_POLICY_ROOT` to the configured installation path.
Keep this session for A–C. In the resulting **Pi shell**:

```bash
sudo /usr/bin/python3 "${GO1_POLICY_ROOT:?Use the Ubuntu SSH setup block}/go1.py" status
```

Only after a reboot/power-on, or if status says `NEEDS REFRESH`, run on **Pi**:

```bash
sudo /usr/bin/python3 "${GO1_POLICY_ROOT:?Use the Ubuntu SSH setup block}/go1.py" refresh
sudo /usr/bin/python3 "${GO1_POLICY_ROOT:?Use the Ubuntu SSH setup block}/go1.py" status
```

A refresh is passive but updates boot-binding files. It needs factory telemetry
and an idle deployment, not a new policy compilation. If refresh fails, retain
its output and backup path; do not launch a trial or edit hashes by hand.

Then verify and print the actual short-package identity on **Pi**:

```bash
cd "${GO1_POLICY_ROOT:?Use the Ubuntu SSH setup block}/normal_walking"
cat tuning.json
sha256sum policy_sender policy/weights.npz worker.py estimator.json
sudo /usr/bin/python3 run_trial.py
```

Require `Package verified; NOT ARMED.` The local mirror previously showed Kp80,
front Kd2.0, rear Kd2.5/3.0/2.5, command [0.5,0,0], five seconds; that is a
reference, not a statement about today's Pi. The earlier three-run hardware
baseline had front Kd1.5. Preserve the actual values rather than silently
resetting them to match an old report. `hardware_validated:false` in the scoped
trial manifest is an intentional distinction from a production release, not
an instruction to edit it to true.

## B. One physical five-second run with the maintained package

After A succeeds, with the operator beside the robot and the stated setup,
use the package’s existing command in **Pi Terminal A**:

```bash
sudo /usr/bin/python3 "${GO1_POLICY_ROOT:?Use the Ubuntu SSH setup block}/go1.py" walk --seconds 5 --execute
```

Leave controls released until **READY**, then **tap L1 alone once and release**.
The package runs its startup and configured five-second policy interval, stops
its custom command owner, and restores factory ownership. Total elapsed time
includes preparation/startup, so it exceeds five seconds.

**L2+B stops early.** Releasing L1 is not a stop in this click-start flow. Do not
send the old prone GUI confirmation or manually kill/restart factory processes.
Wait for the terminal to return and record its exact `Logs:` directory. Record
whether the rope loaded, there was a stumble, unexpected movement, turning or
an early stop. Preserve failed runs too; do not automatically retry.

`go1.py walk` selects the existing package and can apply a duration setting
through its configuration tool; even without `--execute` it is not a pure
read-only inspection. Use `run_trial.py` without arguments in A for unarmed
verification. Ordinary changes to compiled policies elsewhere do not require
another parameter-apply command for this run.

The reference sheet's 30/120-second and bipedal commands are **not part of this
block**. Long walking has a separate binary/authority budget and curves;
“bipedal not installed” in that sheet is an old status note, not verified
current state. Inspect actual package status rather than copying that assumption.

## C. Retain and copy this exact recording

In the same **Pi shell**, after the controller has exited and factory restoration
has completed, paste the full directory printed after `Logs:` when prompted.
This archives root-owned diagnostics without changing their permissions:

```bash
(
set -e
: "${GO1_POLICY_ROOT:?Use the Ubuntu SSH setup block}"
read -r -p 'Paste the full Logs directory from this run: ' GO1_RUN_DIR
GO1_RUN_NAME=${GO1_RUN_DIR##*/}
[[ "$GO1_RUN_NAME" =~ ^manual_[0-9]+$ ]]
[ "$GO1_RUN_DIR" = "${GO1_POLICY_ROOT}/normal_walking/logs/$GO1_RUN_NAME" ]
test -d "$GO1_RUN_DIR"
GO1_EXPORT=$(mktemp -d /home/pi/go1-walk-export-XXXXXXXX)
sudo tar -czf "$GO1_EXPORT/run.tgz" \
  -C "${GO1_POLICY_ROOT}/normal_walking/logs" -- "$GO1_RUN_NAME"
sudo chown "$(id -u):$(id -g)" "$GO1_EXPORT/run.tgz"
(cd "$GO1_EXPORT" && sha256sum run.tgz > run.tgz.sha256)
printf 'Pi export directory: %s\n' "$GO1_EXPORT"
)
```

On **Ubuntu**, paste that export directory when prompted:

```bash
(
set -e
cd "$(git rev-parse --show-toplevel)"
conda activate dog_ctrl
read -r -p 'Paste the Pi export directory: ' GO1_PI_EXPORT
[[ "$GO1_PI_EXPORT" =~ ^/home/pi/go1-walk-export-[a-zA-Z0-9]+$ ]]
mkdir -p logs/walking-hardware
GO1_REVIEW=$(mktemp -d "$PWD/logs/walking-hardware/review-XXXXXXXX")
scp "pi@192.168.12.1:$GO1_PI_EXPORT/run.tgz" \
  "pi@192.168.12.1:$GO1_PI_EXPORT/run.tgz.sha256" "$GO1_REVIEW/"
(cd "$GO1_REVIEW" && sha256sum -c run.tgz.sha256)
python3 -B - "$GO1_REVIEW" <<'PY'
import re, sys, tarfile
from pathlib import Path, PurePosixPath
root = Path(sys.argv[1])
with tarfile.open(root/'run.tgz', 'r:gz') as archive:
    members = archive.getmembers()
    names = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (path.is_absolute() or '..' in path.parts or not path.parts
                or not re.fullmatch(r'manual_[0-9]+', path.parts[0])
                or not (member.isdir() or member.isfile())):
            raise SystemExit('STOP: unexpected archive member: '+member.name)
        names.add(path.parts[0])
    if len(names) != 1:
        raise SystemExit('STOP: expected exactly one run')
    destination = root/'recording'
    destination.mkdir()
    archive.extractall(destination, members=members)
(root/'run-path.txt').write_text(str(destination/next(iter(names)))+'\n')
PY
GO1_LOCAL_RUN=$(cat "$GO1_REVIEW/run-path.txt")
printf 'archive=%s\n' "$GO1_REVIEW"
python3 -B experiment/review_go1_walking.py \
  --run "$GO1_LOCAL_RUN" --out "$GO1_REVIEW/review.json"
)
```

The recording already contains the tuning and release manifest captured at run
start, along with packet and supervisor evidence. Send the Ubuntu archive path,
console outcome and physical observations for review. If the analyzer rejects
an early or incomplete run, retain the archive and the error; that is not a
reason to repeat a motor run to create a nicer log. Leave the Pi original until
its Ubuntu copy and contents are verified.

## Scope of the refresh

This update traced the live estimator composition and checked its release-bound
source/library identities. It refreshed the operating documentation and old
source paths. No new simulation gate, controller rebuild, estimator replacement,
Pi deployment or robot execution was performed. The next participation needed
is the operator's Pi session in A, followed by the supervised B when its stated
conditions hold; new-policy compilation can continue independently.
