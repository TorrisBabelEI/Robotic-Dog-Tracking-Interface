# Walking integration review — 2026-09-29

**Later handover review:** see [September 29 pending handovers](GO1_POLICY_HANDOVER.md).
It includes newer bipedal hardware/addendum evidence, and records the previously
unindexed normal-walking front-Kd-2 trial. The table below remains the earlier
independently decoded baseline; the later reported trial was not redecoded here.

Path notation and workstation setup: [local paths](GO1_LOCAL_PATHS.md).

**Operation refresh:** the installed package already includes the body
linear-velocity Kalman estimator. Use [the current hardware operation
sequence](GO1_WALKING_OPERATIONS.md) for status, boot-clock refresh when
needed, a supervised five-second run and recording retrieval. This review
describes evidence and future parent-release questions; it is not a demand
to keep repeating simulations while the policy developer compiles new policies.

**There are now real, recorded five-second forward-walking examples.** They
live in `${GO1_POLICY_SOURCE}`, with a separate supervised runtime
on the Pi at `${GO1_POLICY_ROOT}/normal_walking`. The parent's earlier
prone-only status does not describe that runtime. Its recordings provide a
useful walking baseline, but do not establish measured torque tracking or
closed-loop straight-line/path accuracy.

This review incorporated a reusable offline package/recording reviewer into the
parent project and indexed exact tested profiles. It did not copy walking gains
into the prone controller or unlock the parent's `walking-policy` hardware mode.
The working supervisor, sender, estimator and policy remain a separately
versioned package. No Pi connection, parameter installation or motor execution
was performed in this review.

## What is new and ready to reuse

| Component | Evidence and integration decision |
| --- | --- |
| Five-second walking | Three consecutive archived runs with the September 25 profile independently reanalyzed below. Use as the recorded baseline for further integration. |
| Runtime gains and bounded target handling | Kp 80, front Kd 1.5, rear Kd 2.5/3.0/2.5; desired command [0.5,0,0], 0.3 s gain ramp. Software gain bounds, slew and SDK protection differ substantially from the old prone experiment; retain their package-specific scope. |
| Complete supervised execution | Fresh L1 tap after READY, L2+B stop, private ownership/supervisor channels, process exit before factory restoration, per-run config/manifest/capture. Reuse the complete runtime rather than extracting the sender alone. |
| Reusable offline analysis | Added `experiment/review_go1_walking.py`: explicit command/state endpoints, native CRC checks, fresh past-command pairing, per-joint position error and predicted PD+FF effort, recorded exit ordering, local package checksum verification. It never imports or launches package code. |
| Package provenance | Added `experiment/config/go1_walking_reference.json`; exact profile/model/binary identities are separated from later package changes. Both current local mirrors passed all 104 manifest file hashes. This is local integrity, not a fresh Pi verification. |
| Long-duration package | A separate binary and larger authority/log/time budgets support up to 120 s. Two 110 s connected simulations pass. No matching full-duration hardware evidence was found in the reviewed walking archives; do not promote simulation to hardware acceptance. |

The frozen policy produces **joint-position residuals**, converted to governed
joint targets. The SDK command includes position/velocity targets, Kp/Kd and
feed-forward torque. The corresponding software estimate is
`Kp*(q_des-q) + Kd*(dq_des-dq) + tau_ff`. These runs are position tracking through
that actuation interface. They do not measure independent torque response, use
a torque-error feedback loop, or establish a torque transfer function.

A zero lateral/yaw command is an intended forward direction, not proof of a
straight trajectory. The reports retain heading drift and simulation/body
contact mismatch. No measured path-tracking acceptance is inferred here.

## Independently reviewed evidence

Archive in this project: `logs/walking-integration/review-qxq_oay2`.
`index.json` records 12 available walking-run manifests and exit logs, source
paths and hashes. Five runs were redecoded from their original PCAPs, without
modifying the source recordings. Model/package metadata and two long simulation
results are indexed separately.

| Recorded run | Policy send span | Joint-position RMS, rad | Outcome |
| --- | --- | --- | --- |
| `manual_1790362027630407117` | 4.9980 s | 0.09312 | Normal duration completion |
| `manual_1790362100725253092` | 4.9960 s | 0.09374 | Normal duration completion |
| `manual_1790362174785576551` | 4.9980 s | 0.09154 | Normal duration completion |
| `manual_1790469618501160762` | 2.7260 s | 0.08605, shortened interval | New policy stopped on `policy_state_or_estimate_stale` |
| `manual_1790469987345486410` | 4.9980 s | 0.09383 | New policy completed after worker guard change |

The first three average 0.09280 rad, reproducing the earlier report's rounded
0.093 rad. Their profile uses old policy weights `3f2c9176…`, sender `7cd00988…`,
and front Kd 1.5. The last two use weights `64e97206…`. Both of those runs still
use front Kd 1.5. Full hashes and settings are in the reference JSON and review
outputs. Seven additional old-policy/fault logs are retained in the metadata
index; not every indexed run has a complete packet-level reanalysis.

All five decoded runs have zero reported kernel capture drops, matching SDK
send counts, valid selected-flow frame CRCs, and recorded guardian/sender/worker
exit before factory resume. The analyzer uses state traffic returning to the
custom port 8080, not concurrent factory traffic on 8008; duplicate ticks do not
add samples. Position statistics cover policy elapsed 0.35–4.8 s, truncated at
actual policy end for the fault case, with a maximum 4 ms age for the latest
preceding command. Policy alignment uses captured send ordinal/count and checks
the duration against `POLICY_SEND_SPAN`; it is not sensor-clock calibration.
These details explain small differences from the original all-stream analyzer.

Recomputed peak PD+FF effort estimates reach approximately 20.2 Nm in the older
runs and 21.9 Nm in the completed new-policy run. This uses a subsequent measured
state with the preceding captured command; it is not the controller's own
same-cycle limit calculation or measured motor torque. It cannot prove or
refute exact enforcement of the stated 19.5/20 Nm limits. Keep it as a separate
quantity when comparing command generation, sent commands and actuator response.
The launcher itself records `qualified:false` and asks for packet/operator
review; a guardian return of -9 occurs in the documented shutdown path and is
not by itself proof of an uncontrolled crash or successful recovery.

## Current mirror is a later configuration

The recorded mirror at review time has these values:

| | Short package | Long package |
| --- | --- | --- |
| Sender | `7cd00988…` | `447eec0c…` |
| Policy | `64e97206…` | `64e97206…` |
| Kp | 80, all joints | 80, all joints |
| Front Kd | **2.0** | **1.5** |
| Rear Kd | 2.5 / 3.0 / 2.5 | 2.5 / 3.0 / 2.5 |
| Duration | 5 s | 120 s |
| Velocity-integrity residual threshold | 0.30 m/s, 0.25 s window | Same |

The operations guide's opening profile still says front Kd 1.5. The actual
short-package tuning, native config and manifest agree on 2.0; this is not a
hash corruption. It is a later configuration, and its complete identity is not
covered by the three-run baseline. Worker hashes also change with boot-specific
clock binding, so neither sender hash alone nor model hash alone identifies a
complete run. No claim is made about the currently powered Pi's state.

Before promoting the current package into a new parent hardware release, resolve
these specific evidence gaps:

1. The September 27 `outputs/hardware_calibration_20260927/TEST4_REVIEW.md`
   explicitly calls for re-deriving foot-contact thresholds before another
   estimator-dependent run. FR standing readings fall below the deployed on
   threshold; unloaded RL is only about 5 raw units below its off threshold.
   Incorporate that calibration evidence instead of repeating generic remote,
   support-probe or prone tests.
2. The new policy's `policy/manifest.json` reports
   `export_max_abs_error_vs_torch=1.999980151591566`. Its meaning/export comparison
   needs reconciliation with the source checkpoint. A hash match or successful
   walking run alone does not establish faithful export. This review does not
   diagnose the cause or relabel that field as passing.
3. Match the intended gains, worker/guard, weights, estimator and current boot to
   an exact archived profile. The later 0.18 -> 0.30 m/s integrity threshold is
   a safeguard change, not merely a new walking parameter. Do not silently
   inherit it or revert a working package based only on older documentation.
4. For a longer run, evaluate the changed duration/authority/log budgets as their
   own scope. Existing 110 s simulations and a 5 s physical run are different
   evidence. Bipedal walking is outside this forward-walking integration.

The earlier report's statements that servo dynamics are conclusively correct,
or that extra calf effort proves a specific physical load, are stronger than
these recordings alone support. Predicted PD effort, an approximate simulator
replay and separate closed-loop runs do not identify the unique mismatch cause.

## Shortened procedure for continuing this work

**Ubuntu review — already completed; reusable for new evidence.** No robot,
SSH tunnel, GUI, build, or support-probe repetition is needed to inspect an
existing package or recording:

```bash
cd "$(git rev-parse --show-toplevel)"
source experiment/go1_paths.sh
: "${GO1_POLICY_SOURCE:?Set GO1_POLICY_SOURCE in .go1-paths.env}"
conda activate dog_ctrl
mkdir -p logs/walking-integration
GO1_WALK_REVIEW=$(mktemp -d "$PWD/logs/walking-integration/review-XXXXXXXX")
python3 -B experiment/review_go1_walking.py \
  --package "${GO1_POLICY_SOURCE}/deployment/pi_package_mirror/normal_walking" \
  --out "$GO1_WALK_REVIEW/package.json"
python3 -B experiment/review_go1_walking.py \
  --run "${GO1_POLICY_SOURCE}/deployment/walking_policy_hw_20260926/hardware/manual_1790469987345486410" \
  --out "$GO1_WALK_REVIEW/run.json"
printf 'archive=%s\n' "$GO1_WALK_REVIEW"
```

The commands never overwrite a report. A successful review command means it
produced a report; inspect the stop reason and integrity fields rather than
calling any processed capture a passed hardware test.

**Pi identity/package check — only if the operator wants to resume a physical
session.** These are checks, not walking commands. The current task did not run
them. Start from Ubuntu:

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

Then in the Pi shell:

```bash
sudo /usr/bin/python3 "${GO1_POLICY_ROOT:?Use the Ubuntu SSH setup block}/go1.py" status
cd "${GO1_POLICY_ROOT:?Use the Ubuntu SSH setup block}/normal_walking"
sha256sum policy_sender policy/weights.npz worker.py estimator.json
cat tuning.json
sudo /usr/bin/python3 run_trial.py
```

The last command has no `--execute` and should print
`Package verified; NOT ARMED.` A stale boot binding must be refreshed using the
existing package's documented passive clock procedure; do not edit hashes or
boot IDs manually. Use direct `run_trial.py` for verification: the unified
`go1.py walk --seconds ...` command can apply a duration change even without
`--execute`.

The first review above stopped at package inspection. The subsequent
[operation refresh](GO1_WALKING_OPERATIONS.md) now provides the existing
operator-run hardware sequence, including execution and retrieval commands.
It preserves the maintained installed package rather than promoting policy
builds in progress. Keep the recorded calibration/profile questions visible;
do not interpret them as a missing estimator implementation or silently
change its thresholds. The old prone GUI is unrelated.
The parent's separate Kp<=1 prone controller has deferred exit/deployment work,
summarized in [the condensed legacy status](GO1_LOWLEVEL_EXPERIMENT.md#21-retained-results-and-remaining-legacy-scope).
Former sections 2.1.31–2.1.32 are retained in the historical record, outside the
active procedure. They are not prerequisites for using or analyzing the
already demonstrated walking package.

## Local validation and source references

Nine focused reviewer tests pass: CRC/scales/modes, nonfinite/sentinel rejection,
past-command age bounds, missing pairs, exit/restore ordering, hash/config
mismatch, package path escape, duplicate JSON keys and endpoint isolation/send
count checks. The existing packet decoder and factory effort tests are also
retained. No controller code or gains were changed by this integration.

The source project is reviewed read-only. Important references there:

- `USER_OPERATION_AND_AI_TUNING.md` — operating flow, with the profile staleness noted above.
- `deployment/kp100_slew_20260925/REPORT_FINAL.md` and `PID_AND_SIM_MATCHING_REPORT.md` — old-policy hardware metrics.
- `deployment/pi_package_mirror/SYNC_RECORD.md` — rename, installation and later policy/guard history; historical Pi sync only.
- `deployment/walking_policy_hw_20260926/hardware/` — new-policy failure and completed run.
- `deployment/normal_walking_long_20260926/connected/run_110s_{a,b}/result.json` — simulation scope explicitly recorded.
- `outputs/hardware_calibration_20260927/TEST4_REVIEW.md` — newer contact-calibration evidence.

The top entries of `deployment/DEPLOYMENT_STATUS_CURRENT.md` still describe
September 24 candidates. They must not override the later dated run artifacts.
