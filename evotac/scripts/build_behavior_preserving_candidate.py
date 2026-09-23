"""Build a train-only actor interpolation for retention diagnostics.

The candidate is derived only from the frozen train warm-start and the frozen
train SAC checkpoint.  It is an exploratory retention ablation, not a new
training result: critics, replay, optimizer state and provenance remain from
the trained checkpoint while the deployed actor is interpolated toward the
warm-start actor.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import torch


def blend_actor_state(trained_state, warmstart_state, trained_weight: float):
    """Return a blended actor state without mutating either input mapping."""
    if not 0.0 <= float(trained_weight) <= 1.0:
        raise ValueError("trained_weight must be in [0, 1]")
    if set(trained_state) != set(warmstart_state):
        raise ValueError("trained and warm-start actor keys do not match")
    result = {}
    for key in trained_state:
        trained = trained_state[key]
        warm = warmstart_state[key]
        if trained.shape != warm.shape or trained.dtype != warm.dtype:
            raise ValueError(f"actor tensor mismatch for {key}")
        if not torch.is_floating_point(trained):
            if not torch.equal(trained, warm):
                raise ValueError(f"non-floating actor tensor differs for {key}")
            result[key] = trained.detach().clone()
        else:
            result[key] = ((1.0 - trained_weight) * warm.detach()
                           + trained_weight * trained.detach()).clone()
    return result


def build_candidate(trained_checkpoint: Path, warmstart_checkpoint: Path,
                    output: Path, trained_weight: float):
    trained = torch.load(trained_checkpoint, map_location="cpu", weights_only=False)
    warm = torch.load(warmstart_checkpoint, map_location="cpu", weights_only=False)
    if trained.get("schema") != "evotac.recovery_sac_trainer.v1":
        raise ValueError("trained checkpoint is not a recovery trainer checkpoint")
    if warm.get("schema") != "evotac.control_warmstart.v1":
        raise ValueError("warm-start checkpoint is not a control warm-start")
    actor = trained.get("learner", {}).get("actor")
    skill = trained.get("experiment_contract", {}).get("skill_name")
    if not skill or skill not in warm.get("skill_names", ()):
        raise ValueError("trained checkpoint skill is absent from warm-start contract")
    network_config = trained.get("learner", {}).get("network_config", {})
    if network_config.get("observation_dim") != 128 or network_config.get("action_dim") != 7:
        raise ValueError("candidate requires the frozen 128D/7D recovery actor contract")
    warm_actor = {
        key.removeprefix(f"{skill}."): value
        for key, value in warm.get("actors", {}).items()
        if key.startswith(f"{skill}.")
    }
    if actor is None or not warm_actor:
        raise ValueError("checkpoint actor or contracted warm-start skill is missing")
    candidate = deepcopy(trained)
    candidate["learner"]["actor"] = blend_actor_state(actor, warm_actor, trained_weight)
    candidate["candidate_derivation"] = {
        "schema": "evotac.behavior_preserving_actor_blend.v1",
        "trained_checkpoint": str(trained_checkpoint),
        "warmstart_checkpoint": str(warmstart_checkpoint),
        "trained_checkpoint_sha256": hashlib.sha256(trained_checkpoint.read_bytes()).hexdigest(),
        "warmstart_checkpoint_sha256": hashlib.sha256(warmstart_checkpoint.read_bytes()).hexdigest(),
        "actor_keys": sorted(actor),
        "trained_weight": float(trained_weight),
        "warmstart_weight": float(1.0 - trained_weight),
        "selection_scope": "train_artifacts_only",
        "claim_scope": "exploratory_dev_retention_ablation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(candidate, temporary)
    temporary.replace(output)
    return candidate["candidate_derivation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--warmstart-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trained-weight", type=float, default=0.25)
    args = parser.parse_args()
    print(build_candidate(args.trained_checkpoint, args.warmstart_checkpoint,
                          args.output, args.trained_weight))


if __name__ == "__main__":
    main()
