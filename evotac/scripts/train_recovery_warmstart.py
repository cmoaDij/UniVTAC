"""Warm-start independent recovery actors from train-split HDF5 windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evotac.learning.history_encoder import HistoryEncoder
from evotac.learning.recovery_observation import RECOVERY_OBSERVATION_DIM
from evotac.learning.recovery_warmstart import RecoveryWarmStart


def train(data_path, output, *, epochs=10, batch_size=32, seed=0):
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers")
    with np.load(data_path, allow_pickle=False) as data:
        required = {"history", "mask", "action", "skill_indices", "metadata"}
        if not required <= set(data.files):
            raise ValueError(f"warm-start dataset is missing: {sorted(required - set(data.files))}")
        history, mask = np.asarray(data["history"], np.float32), np.asarray(data["mask"], bool)
        action = np.asarray(data["action"], np.float32)
        skill_indices = np.asarray(data["skill_indices"], np.int64)
        metadata = json.loads(str(data["metadata"]))
    if metadata.get("schema") != "evotac.recovery_history_dataset.v1" or metadata.get("split") != "train":
        raise ValueError("warm-start requires a dataset explicitly marked split=train")
    if history.ndim != 3 or history.shape[1:] != (8, RECOVERY_OBSERVATION_DIM):
        raise ValueError("history shape does not match the frozen recovery frame contract")
    if mask.shape != history.shape[:2] or action.shape != (len(history), 7) or skill_indices.shape != (len(history),):
        raise ValueError("warm-start arrays have inconsistent shapes")
    skills = tuple(metadata.get("skill_names", ()))
    provenance = metadata.get("provenance", [])
    if not skills or not provenance or any(row.get("split") != "train" for row in provenance):
        raise ValueError("training provenance must contain train-split parents")
    if set(row.get("skill") for row in provenance) != set(skills):
        raise ValueError("each skill needs at least one train parent in provenance")
    torch.manual_seed(seed)
    encoder = HistoryEncoder(input_dim=RECOVERY_OBSERVATION_DIM, hidden_dim=128,
                             output_dim=128, history_length=8)
    encoder.fit_normalization(history, mask)
    model = RecoveryWarmStart(skills, encoder=encoder, actor_hidden_dim=128)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(history))
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            losses.append(model.train_batch({"history": history[index], "mask": mask[index],
                                             "action": action[index], "skill_indices": skill_indices[index]}))
    model.encoder.freeze()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint(provenance), output)
    report = {"status": "written", "schema": "evotac.control_warmstart.v1",
              "checkpoint": str(output), "parents": len(provenance), "transitions": len(history),
              "skills": list(skills), "epochs": epochs, "batch_size": batch_size,
              "final_loss": float(losses[-1]), "loss_updates": len(losses),
              "split": metadata["split"]}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.output, epochs=args.epochs,
                           batch_size=args.batch_size, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
