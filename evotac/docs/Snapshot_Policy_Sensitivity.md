# Snapshot policy sensitivity diagnostic

`diagnose_snapshot_policy` reads completed, aligned **dev** observations from
`run_recovery_paired_eval --probe-controls N`. It runs the frozen FTP-1 and SAC
policies without starting Isaac, updating weights, or executing actions.

```bash
PYTHONPATH=. python -m evotac.scripts.diagnose_snapshot_policy \
  --source-run evotac/runs/phase1/p4_snapshot_aligned_source_dev23_20260923 \
  --output evotac/runs/phase1/policy_sensitivity_new_run \
  --model-gpu <available-GPU-UUID> --inference-seed 23
```

Run from the repository root in the configured UniVTAC environment. The output
directory must be new and inside `evotac/`'s output directories. Model inference
uses the selected GPU; tactile/history encoding and SAC inference use CPU, as
in the paired evaluator. Do not compete with an active simulation or training
job. The script only closes its own inference worker.

At the snapshot boundary, use the sealed scene observation captured after the
source render, not the older episode observation. Later boundaries use the
source's uninterrupted continuation and the restored probe at the same physics
step and remaining budgets. Both episodes must be completed recordings with
matching dev parent, seed, condition and versions. Their diagnostic rollouts
need not be complete task trials. Source scene integrity and the checkpoint's
recorded hashes are checked before model startup.

Each boundary has six queries: source, identical source repeat, restored, and
source with only camera, tactile, or proprioception replaced by restored data.
The modality interventions preserve source references and other context; the
full restored query uses the complete restored observation. These are input
sensitivity interventions, not physically executed branches. Each FTP-1 query
resets the same inference seed. Each SAC query is deterministic and starts a
fresh masked history, representing a hypothetical recovery entry at that
boundary. It does not reproduce an ongoing recovery's accumulated history or
the base policy's queued action execution.

`report.json` records input/code hashes, query IDs, per-boundary differences,
uncalibrated monitor summaries and explicit scope limitations. `actions.npz`
retains decoded joint targets and normalized recovery actions. The worker
retains input arrays, raw output chunks, configuration and RNG reset records.
Joint angles (radians), finger targets (metres), and normalized recovery
components are reported separately. These are requested actions before
controller limits/IK, not measured robot motion.

Compare source/restored sensitivity with the identical-input repeat before
interpreting it. A single seed and short fixed-action prefix cannot establish
long-horizon equivalence or robustness. Small output differences do not certify
hidden state restoration; large differences do not measure success-rate loss.
This diagnostic supplies no pass threshold, does not change the strict audit,
and never enables formal test. All generated evidence stays outside Git.
