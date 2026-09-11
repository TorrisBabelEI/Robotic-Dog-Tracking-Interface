# Go1 development without external lifting

Research decision for this setup: proceed with a ground-start route. A hanging
rig is not a prerequisite for all further work. The operator accepts some
experimental risk; mechanical damage cannot be ruled out or guaranteed absent.
This document distinguishes an implementable direction from an already
validated robot procedure. The next executable procedure is
[experiment section 2.1.7](GO1_LOWLEVEL_EXPERIMENT.md#217-no-lift-route-passive-observation-of-factory-traffic--current-step).

## What the sources establish

1. Unitree documents high-level modes 5 (stand down), 6 (stand up), and 7
   (damping). The documented rise sequence is 7 -> 5 -> 6; from a force-stand
   state, the return goes through 6 -> 5 -> 7. Mode 5 is transitional and should
   not be held for a long time. This describes the factory high-level FSM, not
   a handoff from an arbitrary external low-level controller. The page primarily
   describes SDK 3.5.1; this repository uses 3.8.6, so mode/feedback behavior must
   be observed on this particular robot.
   [Unitree Go1 interface and transition documentation](https://github.com/UnitreeSupport/Unitree_Docs/blob/master/docs/get_started/Go1_Edu.md#311--main-control-commands-in-the-highcmd-structure).
2. Unitree's `State_FixedStand` records actual initial joint positions, applies
   stance gains, then interpolates to a stand target. This provides a useful
   control architecture for starting from the current pose within a single
   controller. Its README's ground-start demonstration is in Gazebo; it is not
   evidence that our Go1 can safely execute an untested ground startup.
   [FixedStand source](https://github.com/unitreerobotics/unitree_guide/blob/main/unitree_guide/src/FSM/State_FixedStand.cpp),
   [official guide README](https://github.com/unitreerobotics/unitree_guide/blob/main/README.md).
3. TechShare's own Go1 demonstration implements a gradual folded-pose-to-standing
   transition. Its example uses calf targets around -2.8 rad and substantial
   stiffness, without calling the SDK PositionLimit in the shown control loop.
   It explicitly recommends initial hanging tests. This is evidence that a
   joint-space rise is implementable, not a suitable drop-in program or a
   verified no-lift procedure for our command limits.
   [TechShare implementation and video](https://techshare.co.jp/faq/unitree/go1_low_stand_up.html).
4. The Go1 research deployment in Walk These Ways similarly interpolates from
   measured joints toward a nominal pose. Its own instructions still recommend
   hanging for initial testing. We should borrow the architecture rather than
   present this project as a safety endorsement of an unsupported first test.
   [Deployment calibration source](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym_deploy/utils/deployment_runner.py),
   [deployment recommendations](https://github.com/Improbable-AI/walk-these-ways/blob/master/README.md#deploying-a-model).
5. The official SDK `example_position` excites a single leg and explicitly
   requests a suspended robot. It is not a four-foot ground stand-up routine.
   [SDK example](https://github.com/unitreerobotics/unitree_legged_sdk/blob/go1/example/example_position.cpp).

The checked local SDK revision is
`4539a6c10dfbc9781cea6fcb7d51bc6ddc6f71e1` (README: v3.8.6). Its high-level
header agrees on modes 5/6/7, while the Unitree support page and this header
differ on the reported high-level flag (0x00 versus 0xEE). Observe actual
packets before treating either value as universally applicable. Do not silently
expand the current capture acceptance based on that documentation discrepancy.

## Compare the available routes

| Route | What it can achieve | Main unresolved issue | Decision |
| --- | --- | --- | --- |
| Factory high-level rise, height change, lie-down | Ground motion baseline; possibly an initial squat demonstration | Cannot command independent joint torque; version-specific transitions | Use for reference observation; not a substitute for the torque experiment |
| Factory stand -> external low-level hold -> factory lie-down | Would reuse factory entry and endpoint | Neither command ownership nor continuous support at both handoffs is established | Defer; do not assume that stopping UDP transfers control safely |
| Prone external low-level entry -> very low four-foot rise -> low return -> gradual release | Direct joint control without standing-time handoffs; can later support torque overlays | Initial gain engagement, body contact, friction, and controlled settling require actual calibration | Preferred development direction |
| Replay factory joint commands or run an existing example unchanged | Quick apparent route to motion | Gains, mode semantics, saturation, body contact and fault behavior differ | Do not use as the first trial |

The preferred route is an engineering inference from the control architecture
and our setup, not a manufacturer's validated unsupported-testing procedure.
No researched source establishes a guarantee against hardware damage.

## Resolve the actual control problems

**Entry near the floor.** Enter low-level mode only while already prone, as in
the passing preflight. Preserve the existing position-free initial damping.
Capture current feedback, then engage impedance progressively. The four natural
calf readings lie below the SDK command minimum. Keeping the command bound
means the position target cannot exactly equal those measurements. Therefore
startup must bound the resulting *total* motor effort while gains rise; merely
limiting feed-forward torque or claiming zero position step is insufficient.
Use the measured discrepancy to calculate the initial PD contribution. Do not
widen the command bounds solely because the factory rests beyond them.

**First motion.** All four feet remain in contact. First verify low-effort
engagement and release close to the initial prone configuration; then attempt
a small symmetric body rise and immediate return. The initial lift magnitude,
gains, speed, duration and effort limits must be selected from the recorded
pose and geometry, not copied from the nominal standing fixture. A low height
reduces potential fall energy; it does not rule out joint overload, foot slip,
or contact with underside components. No single-leg lift or torque oscillation
belongs in the first rise trial.

**Normal return.** Return close to the floor with position support still active,
then gradually reduce the supporting effort while retaining velocity damping
and checking actual motion. This is a controlled settling phase, not a sudden
switch from full standing stiffness to damping. Joint/IMU/foot-force observations
and a visible endpoint inform the result; foot unloading or reaching a target
alone does not certify belly support. Full damping and process exit follow the
reviewed supported endpoint. The duration of gain release does not by itself
bound descent speed, and a descent/contact model plus near-floor trials are
needed before assigning a hardware release profile. If settling cannot be
controlled, redesign this phase before increasing height.

**Faults.** Normal cancellation with valid feedback and a major feedback/control
fault need different paths. A valid-feedback cancellation should use the same
reviewed return when feasible. Feedback loss cannot safely be handled by
blindly continuing an interpolation; the existing emergency damping remains
a fault response that can permit a drop. It is not a promise of damage-free
recovery. Keeping initial tests close to the floor limits the consequence of
that unresolved fault. Neither the factory remote nor `PowerProtect` guarantees
against all contact or mechanical damage.

## Concrete progression

1. **Now: 15-second passive capture, robot prone, factory controller unchanged.**
   Establish whether native traffic is visible on the Pi. Section 2.1.7 provides
   exact commands and archiving. No motor command is sent by the recorder.
2. **After packet review: record one factory rise/return.** Obtain the measured
   trajectory, timing, available state and command fields, and visual evidence
   of initial/final contact. Preserve this as a reference; do not replay it as
   an external controller. If native traffic is inaccessible, choose an
   alternative logging interface before requesting that cycle.
3. **Implement prone engagement/release and one low rise/return.** Use the new
   measured baseline and retain the already-tested watchdog, remote stop and
   logging paths. Add tests only for the changed control path, then write the
   first low-height hardware procedure. This route needs a separate entry path;
   removing the `ground-handover` lock would not implement it.
4. **Increase height, then add the experiment.** After reviewing the real low-rise
   result, progress to a brief stand, a small squat, then a small torque overlay.
   Single-leg and sequential lifts remain later, because they change the support
   polygon. MOCAP is not needed for the initial acquisition/engagement trial;
   it may be added for quantitative motion assessment later.

The lack of a rig no longer blocks progress. The unresolved items are now
specific measurements and control implementations, each connected to the next
trial rather than an indefinite requirement for more synthetic tests.
