# Policy-project handover status — 2026-09-29

**There are pending handovers. The most urgent is bipedal fault/ending recovery.**
The newest trainer addendum reports four hardware runs of the September 29
retrained goal-1 package: three reached the end of the eight-second policy stage,
none started landing, and all reportedly ended in power loss. This is later
than the 10:48 deployment summary that had not yet established a hardware run.

This review reads local source, reports and saved result metadata. It does not
inspect the live Pi, reproduce every raw recording, install a package, or run a
controller. Source paths below are relative to `${GO1_POLICY_SOURCE}` from the
[local path configuration](GO1_LOCAL_PATHS.md). The external project is a working
file tree without Git metadata here; file hashes identify the reviewed snapshot.

## Pending handovers

| Priority / owner | What needs to be handed over | Evidence and current status |
| --- | --- | --- |
| **1 — deployment/controller** | A reviewed fault and ending sequence, with damping and an explicitly qualified posture-dependent return to factory control | The latest addendum lists this as planned work. In the reviewed worker, ABORT/DONE preserves `last_action` for up to 3 s instead of taking the runtime's new abort target. The watchdog/revocation code restores factory processes after process exit; the handover must address robot posture as well as process ownership. This remains unresolved before adopting the bipedal runtime. |
| **2 — trainer and deployment** | One versioned observation/action contract including the heading change, gains and actuator limits | A heading-relative observation patch is reported deployed; training still uses a legacy rotated angular channel and zero-heading resets. The trainer is asked to randomize heading or retrain with body-frame angular velocity. Hold metadata says Kd 3 while supplied tuning uses Kd 4. These need one consistent contract, not another unlabelled patch. |
| **3 — trainer/calibration** | The September 28 evening and September 29 addenda incorporated into the next model and acceptance results | Pending requests include loaded body-response matching, realistic contact readings, velocity noise, latency variation, longer stable hold/brake, and action-rate/power costs. The calf actuator/model interpretation remains disputed; the latest addendum flags the `cal28` range as exceeding the physical calf rating. Measured restraint effects and the electrical cutoff cause are not established. |
| **4 — package owner/operator** | An exact post-patch package snapshot plus matching full-loop evidence | The retrained delivery, 3 s goal-1 settle change, heading patch and mock-only smoothing variants are separate artifacts. Return source/config/model hashes, applied patch set, release verification and matched recordings. Eight seconds of behavior is not full-loop completion. Goal 2 at 120 s remains explicitly unvalidated. |

The source authors propose factory restoration from an unsuitable pose as a
possible power-loss trigger. That is a hypothesis requiring matched controller,
posture and electrical evidence; this review does not establish the electrical
cause or independently endorse their “ruled out” claims. Their draw figures are
computed estimates, not direct high-bandwidth power measurements. The proposed
~400 W training target is not a verified hardware limit.

## Latest delivery and changes

The delivery directory is `updates/full_loop_delivery_8s_retrain_20260928`,
**built September 29 at 09:45**, despite the date in its name:

- Goal 1: hold `hw30_133436`, launch/catch v46, landing v58.
- Goal 2: walker `162011`, launch/catch v45, brake hold `133436`, landing v58,
  with a calm-state gate and transition blend.
- Supplied tuning: Kp 30 / Kd 4; default behavior duration 8 s.
- Preparation/patch scripts are in `deployment/bipedal_8s_retrain_20260929/`.
  Script presence alone does not prove installation, but the later addendum
  explicitly reports testing the retrained package and deploying the settle and
  heading changes. Live state was not independently checked here.

Two patches require distinct treatment:

- `update_settle3.py`: goal-1 minimum post-policy settle 5 → 3 s. One recorded
  gate replay could have benefited; the other failures never became landable.
  Shortening the wait does not resolve those balance failures.
- `heading_patch.py` / `update_heading.py`: observations use heading relative to
  the starting pose. The robot-like harness reached the end of the eight-second
  hold in 3/3 patched +122° trials, but **all three saved full-loop results are
  `passed:false`**, with no DONE. The hold result is not full-loop acceptance.

The latest smoothing sweep is experimental: 12 available `sm_*` result files
all report `passed:false`, and one additional directory lacks a final result.
The unsmoothed baseline also fails full-loop completion. The addendum reports
stronger action smoothing reduced eight-second hold success to 1/3. These are
not changes to copy into the working walking or torque controller.

## Already delivered or answered

These should not be reintroduced as missing-work checks:

- **The linear-velocity estimator exists and is integrated.** A September 27 Pi
  comparison of compiled and NumPy leg filters reports 33,700 steps, zero
  mismatches and maximum velocity difference 1.8e−14 m/s. This checks equivalent
  computation, not moving-robot velocity accuracy.
- **Front Kd 2.0 has a normal-walking hardware record.** Run
  `manual_1790524105070849195` is reported to complete 4.998 s, with RMS joint
  error 0.0937 rad and factory restoration recorded. It was one cooler trial,
  so it does not establish a repeatable performance improvement.
- **Bipedal contact-threshold changes have a handover record.** The September 28
  evening addendum reports deployed FL off/on 120/170 and rear 120/180. Training
  still needs the corresponding in-air sensor distribution. These values are
  specific to that package, not a silent update to normal walking or this parent.
- **The one-invalid-estimate-tick question was acknowledged by the trainer.**
  `ANSWERS_20260927_EVENING.md` states that invalid ticks are counted as failures.
  The older `QUESTIONS_FOR_LINUX_SIDE.md` alone is not the current status.
- **The native-compute handover has a later Pi result.** The end of
  `deployment/NATIVE_PIPELINE_HANDOFF.md` supersedes its initial “after recharge”
  task with the v2 target result. The old v1 task is not pending anew.

One separate normal-walking metadata question remains: its manifest still lists
`export_max_abs_error_vs_torch=1.999980151591566`. The new bipedal hold's reported
export difference, 8.97e−6, concerns a different actor and does not resolve that
older field. Keep this in the normal-walking provenance review.

## Source index and retained evidence

Start with these current records, not the old root status snapshot:

- `external_storage/go1_retraining_addendum_20260929/ADDENDUM_20260929.md`
- `external_storage/go1_retraining_addendum_20260928_evening/ADDENDUM_20260928_evening.md`
- `updates/full_loop_delivery_8s_retrain_20260928/DELIVERY_README.md`
- `deployment/bipedal_8s_retrain_20260929/{update_retrain,update_settle3,update_heading,heading_patch}.py`
- `bipedal_walking/build_8s_20260927/connected/{st3_*,yawrob_*,sm_*}/result.json`
- `outputs/normal_walking_tuning_20260927/FRONTKD2_RESULT.md`
- `deployment/bipedal_8s_20260927/HARDWARE_TRIALS_20260927.md`

Local audit: `logs/policy-handover/review-S7OmZcSz/handover_snapshot.json` records
source hashes and the saved-result counts. No external files were changed.

This parent still uses high-level MPC/dense trajectory control and its separate
September C++ torque experiments. No new bipedal policy, threshold, gain, heading
convention or fault handler was merged into those controllers by this review.
