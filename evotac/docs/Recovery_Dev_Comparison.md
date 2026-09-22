# Frozen recovery development comparison

This protocol extends the three-parent calibration using the ten existing dev
parents in `configs/phase2_dev_scenes.json`. It is exploratory development
evidence. The headless sensor configuration is shared by all new arms; historical
livestream results are not pooled. It does not replace the strict snapshot gate
or authorize independent test.

Freeze `configs/recovery_dev_comparison.json` before the first attempt:

- Baseline A and baseline B: identical FTP-1, recovery disabled. Their outcome
  disagreements measure fresh-run variation.
- Warmstart: FTP-1 plus the train-only warmstart recovery actor, before online
  SAC updates.
- Trained: FTP-1 plus the same recovery actor after the existing real training.
- All arms share threshold 0.3, maximum 12 recovery actions, three stable cycles
  before handoff, 1200 task physics steps, and the original sensor/controller
  configuration. Baselines compute features through the same evaluation loop.

The ten seeds are fixed before observing new outcomes. A seeded shuffle and
rotating arm order spread ordering effects. Each of the 40 attempts starts a
fresh Isaac process on one selected GPU, sequentially. FTP-1 resets its inference
seed to the scene seed. Recovery is deterministic. No training updates occur.
Task initialization and rendering can still vary; these are matched scene
families, not identical hidden-state counterfactuals.

Finish every planned attempt without success-based early stopping, scene
replacement, threshold changes, or checkpoint selection. Infrastructure and
provenance errors stop the coordinator for inspection. Invalid resets remain in
the scheduled denominator and are separately reported, not relabeled as physical
task failures. Resume accepts only finished/pending rows and unchanged code,
weights, configuration and GPU. Incomplete attempts require explicit inspection.

The primary descriptive comparison is trained versus baseline A. Baseline B
quantifies repeat variability, and trained versus warmstart assesses whether
online SAC improved this single skill. Report valid/scheduled counts, terminal
reasons, recovery engagement, discordant outcomes, paired parent differences,
descriptive bootstrap intervals and exact McNemar p-values. The small dev sample,
previous calibration on three of its parents, and secondary comparisons prevent
confirmatory generalization claims. No isolated favorable seed proves efficacy.
This experiment does not test the full skill-memory/selection/self-evolution
system; an improvement over warmstart would only support single-skill learning.

```bash
PYTHONPATH=. python -u -m evotac.scripts.compare_recovery_dev \
  --run-id recovery_dev_comparison_new_run \
  --checkpoint evotac/runs/phase1/p4_validation_resume_20260922_tail18/recovery_trainer.pt \
  --warmstart evotac/runs/phase1/recovery_warmstart_train_seed1_6_9_10_20260921.pt \
  --gpu <available-GPU-UUID>
```

Use the configured UniVTAC environment from the repository root. The immutable
`protocol.json`, live `status.json`, subprocess logs, checks, HDF5 episodes and
weight/source hashes remain local under `evotac/runs` and `evotac/datasets`.
Do not start a second simulation on the selected GPU. Existing jobs are preserved.
