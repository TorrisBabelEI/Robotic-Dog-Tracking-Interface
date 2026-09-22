# Selected Wenjian integration — 2026-09-22

The operator selected items **1 + 3 + 7**. Their implementation and local
verification are complete. The next operator action is the combined Pi build
and SDK check in [manual section 2.1.32](GO1_LOWLEVEL_EXPERIMENT.md#2132-selected-integration-and-reduced-next-test-sequence),
then only the two pending grounded exit cases in 2.1.31. Completed normal
engagement and simulation blocks are retained without another operator run.

## Scope and provenance

| Selection | Integrated behavior |
| --- | --- |
| 1 — SDK receive/send checks | Receive success requires an advancing SDK receive counter and unchanged flag/CRC error counters. Duplicate ticks do not refresh acquisition age. Packet, sequence and acquisition timestamp are read together; control time is sampled afterward. A failed `SetSend` never calls `Send`; raw send lengths remain available to the existing exact-length check. |
| 3 — Factory effort analysis | `experiment/review_factory_effort.py` pairs archived factory commands with past feedback from the matching endpoint and computes Kp×position error + Kd×velocity error + feed-forward torque. Both native CRC profiles are explicit; invalid frames, non-servo modes, active sentinels and missing/stale/future pairs are rejected. |
| 7 — Deployment evidence | `experiment/prepare_go1_bundle.py` snapshots an explicit source list, SDK headers/libraries, tests, per-file hashes and scope manifest. Both Pi helpers transfer the frozen snapshot and verify it before building. The main preparation includes the SDK tests and retains the built binary hash. |

Sources reviewed include `Wenjian_test_walking_policy/deployment/review_factory_effort.py`,
its tests, `SUPPORTED_HOLD_PROGRESS.md`, `STAGED_HARDWARE_INTEGRATION.md`, and
`COMPUTE_RELEASE_REVIEW.md`, plus SDK receive changes already present in the
parent workspace. The decoder is adapted to the parent's packet reader and
adds endpoint isolation, explicit CRC selection and bounded past-state pairing.
The ignored Wenjian directory remains intact.

Other uncommitted policy integration was present before this selection. It is
preserved behind the OFF-by-default CMake option
`GO1_ENABLE_POLICY_DEVELOPMENT`, with the offline bridge explicitly opting in.
The selected source bundle excludes policy/command-owner headers and policy
targets; both hardware walking entry points remain locked. The existing
opt-in policy session/worker tests still pass. This preservation is not a
promotion of that work into the selected hardware build.

There are no changes to the prone trajectory, Kp/Kd limits, torque caps or
watchdog deadlines. Factory ownership/supervision, the suspended supported-hold
candidate, scheduling-policy changes, estimator and walking deployment were
not selected. The reported afternoon relaxation/movement is retained as an
operator observation; it is not mapped to a qualified torque-tracking run by
the evidence reviewed here.

## Evidence and validation

Archive: `logs/integration-review/review-9HRx4VVD`.
`source-final/manifest.json` identifies the exact 38-file source snapshot,
including uncommitted bytes; a Git revision alone is insufficient.

| Check | Result / transcript |
| --- | --- |
| Isolated selected bundle build and CTest | 30/30 PASS, `configure.txt`, `build.txt`, `ctest.txt` |
| SDK command adapter, Ubuntu data-only execution | PASS damping, prone engagement, torque channel; `sdk-adapter.txt` |
| Final bundled decoder/effort Python tests | 11/11 PASS (3 decoder + 8 effort), `final-python-tests-discovery.txt` |
| Bundle integrity/no-overwrite/path/scope tests | 2/2 PASS, `bundle-tests.txt` |
| Deployment helpers, fake SSH and rsync | 1/1 PASS with 2 helper subcases, `helper-tests.txt` |
| Preserved opt-in policy session/worker | 2/2 PASS, `preserved-policy-tests.txt` |
| Hardware translation unit, policy OFF and ON | Both compiled as objects; neither executed |
| Final snapshot comparison and integrity | C++/CMake identical to tested snapshot; only analyzer and analyzer tests changed afterward, rechecked above; `snapshot-comparison.json`, `final-bundle-check.txt` |

A module-style Python test invocation encountered the environment's unrelated
`test` package and failed import before running any test. The failure is retained
in `final-python-tests.txt`; unittest discovery in the isolated bundle resolved
the invocation and passed all 11 tests. The analyzer initially rejected an
entire multi-interface capture on timestamp inversion. It now reports inversion
counts while rejecting every future/stale candidate pair; that correction is
covered by the final tests.

Reproduction commands for future changed-source review (already completed;
not another operator gate):

```bash
cd ~/Yuxuan/Robotic-Dog-Tracking-Interface
conda activate dog_ctrl
GO1_REVIEW=$(mktemp -d "$PWD/logs/integration-review/review-XXXXXXXX")
python3 -B experiment/prepare_go1_bundle.py --out "$GO1_REVIEW/source"
cmake -S "$GO1_REVIEW/source" -B "$GO1_REVIEW/build" \
  -DBUILD_TESTING=ON -DBUILD_SDK_EXAMPLES=OFF -DPYTHON_BUILD=OFF \
  -DGO1_ENABLE_POLICY_DEVELOPMENT=OFF -DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE
cmake --build "$GO1_REVIEW/build" -j2
ctest --test-dir "$GO1_REVIEW/build" --output-on-failure
"$GO1_REVIEW/build/go1_sdk_command_adapter_test"
(cd "$GO1_REVIEW/source" && python3 -B -m unittest discover -s test -p 'test_*.py')
python3 -B -m unittest discover -s test -p 'test_prepare_go1_bundle.py'
python3 -B -m unittest discover -s test -p 'test_go1_deployment_helpers.py'
```

## Reused factory captures

Both analyses used the factory CRC profile and Pi source port 8008. Other flows
are counted but never paired across endpoints. A pair must use past feedback
no more than 4 ms old by capture timestamps. Results are software commanded
effort estimates, not physical torque measurements or fresh takeover seeds.

| Capture | Accepted pairs | Maximum pair age | Maximum absolute effort among accepted pairs |
| --- | --- | --- | --- |
| Parent `logs/feedback-path/review-7X0qmEAc/feedback_path_01.pcap` | 18,328 / 25,264 commands | 2.268 ms | 0.077763 Nm |
| Wenjian `deployment/logs/supported_hold_boot_6782cf88/paired.pcap` | 2,470 / 2,470 commands | 2.463 ms | 0.903043 Nm |

Parent source SHA-256:
`2d1b02e8e02051f731e99ee2e47f053c1fadbba1e49886c159c1e454695e773f`.
Wenjian source SHA-256:
`fe7c69321bda23fad20a2d0a6c4be876e1f28737370cd5a7a548c389d004b0e7`.
The parent capture excludes 6,536 non-servo pairs, 392 without recent past
feedback, and 8 invalid native headers, and records 3 timestamp inversions.
Its maximum is over accepted pairs, not the entire physical trial. Different
postures and capture conditions preclude interpreting the two maxima as a
tracking comparison. These analyses do not resolve concurrent factory/custom
command ownership or actuator response. Wenjian's capture also had one kernel
capture drop according to its accompanying report.

Results are in `factory-effort/` and `wenjian-effort-final/` within the integration
archive. Each contains a source hash, summary, rejected-frame accounting and
per-pair effort JSONL. For a later capture, run this offline command with a new
output directory:

```bash
python3 -B experiment/review_factory_effort.py PATH_TO_CAPTURE.pcap \
  --command-profile factory --source-port 8008 --out NEW_REVIEW_DIRECTORY
```

This tool reads files only; it cannot send or replay motor commands. No new
capture, cleanup, factory process change, or Pi execution was performed for
this integration. Pi checks and the new binary's physical exit verification
remain operator work; passing software tests does not mark torque tracking as
accepted.
