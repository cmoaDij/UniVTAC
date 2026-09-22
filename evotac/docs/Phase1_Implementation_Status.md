# EvoTac Phase 1 implementation and verification

Status: phase-1 interface acceptance completed; configuration and v3 tolerances frozen, with explicit replay rejections. All new source, configuration and project outputs are under `evotac/`; original tracked files remain unchanged.

> 本文记录阶段 1 的历史验收，不是当前总状态。后续以 [阶段 2 计划](Phase2_Plan.md)、[阶段 2 结果](Phase2_Results.md) 和 [EvoTac 实现状态](EvoTac_Implementation_Status.md) 为准。

## Running and using the interface

Use `/data/ZED/conda/envs/UniVTAC/bin/python`. The authoritative implementation is the author's Isaac Sim 5.1 `isaac51` branch. Existing acceptance runs use `--livestream 2`, taxim, physical GPU 1 exposed as `cuda:0`, cameras enabled, and the rendering Kit experience. The separate `--performance-profile` path permits headless/livestream=0 for speed A/B only; it keeps camera/tactile sensors and is not a replacement for validation.

```bash
/data/ZED/conda/envs/UniVTAC/bin/python -m pytest evotac/tests -q -o cache_dir=evotac/.pytest_cache
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/data/ZED/conda/envs/UniVTAC/lib /data/ZED/conda/envs/UniVTAC/bin/python -u -m evotac.scripts.smoke_phase1 --livestream 2 --device cuda:0 --seed 12 --split test
```

`UniVTACRLWrapper` owns `reset/step/replay_start/close`. `Action` explicitly separates 8-value joint targets and 7-value normalized recovery increments. Each normal decision advances six 120 Hz physics steps (50 ms simulation time). Recovery uses local damped IK, world translations/rotation vectors at `panda_hand`, physical targets with `force=False` and zero target velocity. Configured amplitudes are 1 mm translation, 0.01 rad rotation and 0.2 mm single-finger target change. Actual actuator motion is recorded separately.

Deployment observations contain camera/tactile RGB, named proprioception, accepted previous command, modality sample times/ages/validity, known budgets and reference/history information. Hidden actor poses, condition and success metadata are saved only in training/scene channels. Empty references are captured before grasp; stable grasp references after lifting and before approaching the hole.

Additional entry points:

- `collect_rollouts --scene-manifest FILE`: JSONL rows with seed, parent_scene_id, split, optional condition, and typed actions; all attempts retained. Infrastructure errors stop by default.
- `smoke_phase1 --lateral-offset 0.002 0`: physically execute a lateral approach offset. Condition variants share the same parent split.
- `check_action_axes`, `smoke_failures`, `check_legacy_compatibility --episodes 0 1`: targeted GPU checks.
- `calibrate_replay --config evotac/configs/calibration_insert_hole.yaml --fresh-processes --dev-seeds 0 18 23 24 --repeats 2`: development parents inherited from the source manifest; non-dev source seeds are rejected before launching Isaac.
- `freeze_replay_tolerances`, `audit_rollouts`: CPU-only freezing of development statistics and read-only episode integrity audits.

Prefix protocol `accepted_targets.v2_standard_init` records complete accepted position/velocity targets, hold lengths and render/reference events. The first reset primes the original saved-frame mechanism without grasp/approach, then uses the same restore path as later resets. Priming costs remain in initialization. Replays resample references, reconstruct eight history samples and original action context, and cancel pending policy commands. A persistent parent registry prevents condition/replay branches from crossing dataset splits across processes. Match reports check max-absolute and RMSE limits, sensor/history/reference validity and time alignment; uncalibrated or out-of-envelope samples remain invalid.

## Verified evidence

Paths below are relative to `evotac/`. Commands, source/config/asset hashes and device details accompany GPU datasets in `datasets/phase1/<run_id>/manifest.json`. Do not infer success from process exit alone: Kit fast shutdown can return zero following an exception; inspect checks and HDF5 outcomes.

| Check | Evidence and result |
|---|---|
| CPU contracts | 32 tests passed. Includes rotated-base IK, invalid actions, finite budget/partial execution, true zero-step rejection, observation truth isolation and freshness, legacy pairing/source splits, replay tampering, queue cancellation, and flushed HDF5 recovery after forced process termination, and interruption-safe collection denominators. |
| Rendered sensors | `livestream_initial_01/checks.json` and PNGs under its `images/`: all six streams valid and nonconstant. Head scene and marker images visually inspected. |
| Signed full-scale controls | `acceptance_standard_v2_01_axes/checks.json`: all 14 signed directions passed; each six steps. X about ±0.991 mm, Z ±0.997 mm, rotations about 0.01 rad with correct sign, both fingers respond. Execution approximately 0.26–0.30 s wall time per decision; this is not wall-clock real-time control. |
| Current initialization/replay chain | `livestream_standard_init_03/checks.json`: 12 controls, replay, eight-history/time/reference match, queue cancellation and recovery-budget failure. First reset cost 598 includes 52 priming steps; decision boundary 546, replay reset 546; source/replay pre-prefix ID both 27. This run is uncalibrated, not accepted as a match. |
| Real failures | `acceptance_standard_v2_01_failures/checks.json`: recovery budget and physical gripper-release/object_lost passed. Both HDF5 episodes independently audited consistent. |
| Legacy controller compatibility | `acceptance_standard_v2_01_legacy/legacy_compatibility.json`: episodes 0 and 1 reached task success after 56 and 50 future-qpos proxy targets. Both episodes audited consistent. Classification remains warm_start_candidate_only. |
| Original rendering smoke | `runs/phase1/launch_logs/legacy_smoke_02.log`: original `scripts/smoke_isaac51.py --backend taxim --livestream 2` explicitly reports PASS for both GelSight Mini sensors, Actor API and one render copy. |
| Original collection | `acceptance_standard_v2_01/original_collection.log`: original seed 0, `--max_seed 0`, one requested episode, isolated save root. Plan=True, Check=False; no infrastructure exception. Relative insertion z≈−0.02013 m did not reach the original −0.04 m criterion. Failure metadata/video retained; this is not a successful demonstration. |

Run directories in the table are under `runs/phase1/` unless a full relative path is shown. HDF5 outcomes distinguish finalized storage (`complete`) from `valid_trial` and `incomplete`; rejected and interrupted attempts are not silently dropped. Per-attempt and cumulative physics ledgers charge initialization, prefix, stabilization, base and recovery separately, including replay and priming.

## Frozen calibration and independent validation

`calibration_standard_v2_dev_01` completed 8 replays on actual development source seeds 0, 18, 23, 24. All 12 episodes audited consistent. Frozen `replay_tolerances_v2.json` did not pass independent `validation_standard_v2_test12_01`: robot/history timing/reference validity matched, but tactile history and prism pose errors exceeded the envelope. The failed report and both episodes are retained and audited; no v2 threshold was altered.

That cohort placed all four parents in one Isaac process, whereas validation began from a fresh application. Its first parent had measurable contact reconstruction error and later parents had essentially zero error. This is evidence of a launch-condition coverage gap, not proof of a specific simulator defect. `calibration_fresh_v3_dev_01` therefore ran each of the same four development parents in its own process, each followed by two replays. The estimator (max over development repeats × 1.2), analytical float32 state floor and image metrics are unchanged. No test measurements enter the estimator. The four fresh processes completed all eight replays; all 12 episodes audited consistent. Their frozen artifact is `configs/replay_tolerances_v3.json` (`standard_init_fresh_dev_precision.v3`).

Independent v3 checks:
- `validation_fresh_v3_test16_01`: rejected. All state/time/reference/RMSE checks met bounds, but two history maximum pixel errors were 28 against 27.6. The frozen bounds were not rounded or enlarged. Both saved episodes audited consistent.
- `validation_fresh_v3_test26_01`: checked, `valid_match=true`, all 65 error metrics covered, eight-history/time/reference validity passed, source queue cancellation and recovery-budget termination passed. Twelve controls consumed 72 physics steps / 0.6 simulated seconds and 4.551 s including logging (2.637 decisions per wall second). Both saved episodes audited consistent.

Thus one of two independent v3 test-parent checks matched. This is a short interface verification sample, not a benchmark success-rate estimate or a claim that every parent reconstructs within tolerance. Out-of-envelope starts remain excluded from valid effect labels while keeping their records/cost. Earlier v2 test seed 12 remains a separate failed validation. The YAML validation-status label was updated after these checks; its physical execution fingerprint is unchanged.

`lateral_standard_v2_test12_01` passed physical reset/approach and six-stream rendering for +2 mm world X. Its observation had no condition label and its parent/split matched the nominal seed 12. Across those two independent runs, measured end-effector displacement was [2.594, 0.013, 0.278] mm; this includes physical tracking/settling differences and is not an exact 2 mm state write. The prefix and one episode were saved and audited.

Earlier attempts remain as diagnostic evidence:

- `livestream_control_01`: real zero-step command rejection caused by float64/float32 target readback comparison. Corrected to compare the exact cast submitted to the controller; regression tested.
- `livestream_calibration_01`: reset exception after three physics steps at 138.45 s, fully saved as invalid/incomplete. Original smoke independently demonstrated 87.38 s startup for its first two steps. EvoTac's reset wall guard is now explicitly 300 s; physics budgets are unchanged.
- `livestream_calibration_02` and tolerance v1 used the old initialization protocol and seeds 1/2, which later source-split verification identified as TRAIN parents. They are exploratory, not formal dev calibration, and are inactive. Explicit non-fast Kit shutdown segfaulted after saved reports; default shutdown was restored.
- `livestream_validation_03`: exceeded the old frozen envelope; retained valid_match=false. Its errors were not used to widen tolerances. The old candidate YAML is archived with that run.

## Legacy resources and phase-2 handoff

`datasets/legacy_insert_hole/future_qpos_images_v1_20260918_000229/` contains the 100-source manifest and three exploratory conversions, including decoded samples of all six JPEG image streams. Remaining images are read-only source references, not duplicated. Across 100 trajectories the reader generates 6,315 i→i+3 candidate pairs; original episode 99 is source seed 761. Original JPEG channel round-trip is preserved. Future measured qpos is not an accepted action command; source rate and joint-order provenance remain unconfirmed, factual horizon_seconds is null, and formal training eligibility is false. Proxy control success does not establish ACT policy performance. New matched successful demonstrations are needed only if phase-2 compatibility/coverage checks show these candidates are insufficient.

The existing audit covers 800 trajectories in eight Isaac51 tasks. Contact data inventory is 14 shapes/638 HDF5 files; content/version quality has not been audited. Public encoder weights have not been downloaded or load-verified. Released policy weights refer to Isaac Sim 4.5 and remain unverified on 5.1. See [existing resource assessment](Isaac51_Existing_Data_Assessment.md). These are explicit phase-2 tasks, not reasons to train/download during phase 1.

Phase 2 must validate encoder loading, adapt ACT inference without granting it simulator stepping ownership, rebuild real ACT history/cache at control handoffs, and perform closed-loop compatibility checks. The current policy-state hook is tested with script state only. Replay acceptance measures observable/recorded consistency within the frozen envelope; it does not establish identical hidden-state counterfactuals.
