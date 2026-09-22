# Acceptance matrix

Evidence is linked in [the implementation record](Phase1_Implementation_Status.md). All GPU checks use livestream/taxim; no tactile backend was changed.

| Requirement | Evidence | Status |
|---|---|---|
| Isolated extension; original code/config unchanged | Git diff and output path validation | Passed |
| Typed actions, world rotvec IK, joint limits/rejections | CPU contracts; 14 signed GPU axes at configured amplitudes | Passed |
| Six physical steps and 0.05 simulation seconds | GPU control traces and independent HDF5 audits | Passed |
| force=False; distinct command acceptance/measured response | Saved controls/prefix targets and real readback regression | Passed |
| Deployment whitelist, detached arrays, sensor times/ages | Truth-invariance/freshness tests; six rendered streams and PNGs | Passed |
| Empty/grasp references at correct stages | Prefix reference events and replay validity checks | Passed |
| All outcomes survive; exceptions distinct from task failures | Real zero-step rejection, reset exception, budget failure, dropped object; forced-process-exit test | Passed |
| Complete accepted prefix; history/cache/budget reconstruction | Standard-init smoke; eight history samples; timing/reference match; queue cleared at zero cost | Passed |
| Version/tamper/split isolation | CPU tests; persistent parent registry; actual legacy source splits inherited | Passed |
| Independent dev calibration and frozen match gate | v3 fresh-process dev cohort: 4 parents/8 replays; test 26 matched, test 16 conservatively rejected; both retained | Passed with rejection gate |
| Nominal/lateral physical conditions | Nominal and +2 mm lateral test-12 GPU resets; matching parent/split, no condition in observation; episode audited | Passed |
| Legacy read/convert/compatibility conclusion | 100-source manifest, 6,315 pairs, three conversions; two GPU proxy successes | Passed with provenance limits |
| Original rendering smoke and one-seed collection results | Original smoke PASS; seed 0 Plan=True/Check=False, failure metadata/video saved | Executed; original task failure recorded |
| Interaction costs include priming/replays/failures | Per-attempt and cumulative ledgers audited | Passed |
| Frozen YAML and phase-2 resource handoff | Controller/YAML/v3 tolerances frozen; resources and phase-2 limits documented | Passed |

Phase-1 interface acceptance is complete with the recorded replay limits; this does not claim universal match or task success. Original single-seed task failure is reported explicitly, not replaced with an exit-code success claim.
