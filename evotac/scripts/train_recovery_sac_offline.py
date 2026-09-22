"""Train the recovery SAC learner from real, labeled replay transitions.

This is the second stage after behavior-cloning warm-start.  It uses a
standard replay batch (default 32), preserves terminal/bootstrap flags, and
exports the same trainer checkpoint consumed by ``train_recovery.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evotac.learning.recovery_buffer import ReplayBuffer, Transition
from evotac.learning.recovery_sac import RecoverySAC
from evotac.learning.recovery_warmstart import RecoveryWarmStart
from evotac.learning.recovery_experiment import sha256


def train(data_path, warmstart, output, *, updates=1000, batch_size=32, seed=0):
    if type(updates) is not int or updates < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("updates and batch_size must be positive integers")
    with np.load(data_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        required = {"observation", "action", "reward", "next_observation",
                    "terminated", "truncated", "bootstrap_allowed"}
        if not required <= set(data.files):
            raise ValueError(f"replay dataset is missing: {sorted(required - set(data.files))}")
        arrays = {key: np.asarray(data[key]) for key in required}
    if metadata.get("schema") not in {"evotac.recovery_replay_dataset.v1", "evotac.recovery_replay_dataset.v2"} or metadata.get("split") != "train":
        raise ValueError("offline SAC requires a train-split replay dataset")
    if len(arrays["reward"]) < batch_size:
        raise ValueError(f"need at least batch_size={batch_size} transitions, have {len(arrays['reward'])}")
    warm_state = torch.load(warmstart, map_location="cpu", weights_only=False)
    warm = RecoveryWarmStart.from_checkpoint(warm_state, device="cpu")
    skill_names = tuple(metadata.get("skill_names", ()))
    if len(skill_names) != 1 or skill_names[0] not in warm.actors:
        raise ValueError("offline SAC currently requires one matching recovery skill")
    learner = RecoverySAC(128, 7, hidden_dim=128, device="cpu", seed=seed)
    learner.actor.load_state_dict(warm.actors[skill_names[0]].state_dict(), strict=True)
    replay = ReplayBuffer(max(100000, len(arrays["reward"])), seed=seed)
    for index in range(len(arrays["reward"])):
        replay.add(Transition(arrays["observation"][index], arrays["action"][index],
                              arrays["reward"][index], arrays["next_observation"][index],
                              bool(arrays["terminated"][index]), bool(arrays["truncated"][index]),
                              bool(arrays["bootstrap_allowed"][index])))
    losses = []
    for _ in range(updates):
        losses.append(learner.update(replay.sample(batch_size)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema": "evotac.recovery_sac_trainer.v1", "batch_size": batch_size,
             "updates_per_interaction": 1, "learning_starts": batch_size,
             "learner": learner.state_dict(), "buffer": replay.state_dict(),
             "experiment_contract": {"warmstart_sha256": sha256(warmstart),
                                     "skill_name": skill_names[0],
                                     "monitor_object_lost_risk": 0.3,
                                     "monitor_contact_blocked": 0.8,
                                     "stable_cycles": 3},
             "training_evidence": {"split": "train", "dataset": str(data_path),
                                    "transitions": len(replay), "updates": updates,
                                    "positive_rewards": int(np.count_nonzero(arrays["reward"] > 0))}}
    torch.save(state, output)
    report = {"status": "written", "schema": "evotac.recovery_sac_trainer.v1",
              "checkpoint": str(output), "dataset": str(data_path),
              "warmstart": str(warmstart), "transitions": len(replay),
              "updates": updates, "batch_size": batch_size,
              "positive_rewards": int(np.count_nonzero(arrays["reward"] > 0)),
              "mean_critic_loss": float(np.mean([x.critic_loss for x in losses])),
              "mean_actor_loss": float(np.mean([x.actor_loss for x in losses])),
              "final_alpha": float(losses[-1].alpha), "split": metadata["split"]}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.warmstart, args.output,
                           updates=args.updates, batch_size=args.batch_size, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
