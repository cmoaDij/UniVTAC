"""Fine-tune π0.5 on the audited train-only UniVTAC export.

This is supervised action-chunk training from accepted teacher targets. It
never reads dev/test data and records all frozen input hashes in the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from evotac.config import ROOT, output_path
from evotac.policy.pi05_runtime import input_batch, make_config, make_processors
from evotac.scripts.pi05_worker import load_policy_strict


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ChunkDataset:
    def __init__(self, root, horizon):
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema") != "evotac.pi05_dataset.v2" or manifest.get("source_split") != "train":
            raise ValueError("π0.5 training requires the audited train-only dataset")
        self.rows = []
        self.horizon = horizon
        for item in manifest["episodes"]:
            path = root / item["file"]
            with h5py.File(path, "r") as handle:
                length = len(handle["action"])
            self.rows.extend((path, index) for index in range(length))
        if not self.rows:
            raise ValueError("empty π0.5 train dataset")
        self.manifest = manifest

    def get(self, index):
        path, start = self.rows[index % len(self.rows)]
        with h5py.File(path, "r") as handle:
            end = min(start + self.horizon, len(handle["action"]))
            action = np.asarray(handle["action"][start:end], dtype=np.float32)
            if len(action) < self.horizon:
                action = np.concatenate([action, np.repeat(action[-1:], self.horizon - len(action), axis=0)])
            return {key: np.asarray(handle[key][start], dtype=np.uint8 if "images" in key else np.float32)
                    for key in ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb",
                                "observation.state")} | {"action": action}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/pi05_insert_hole_train_20260924_c")
    parser.add_argument("--base-checkpoint", type=Path, default=ROOT / "checkpoints/pi05_base_b211f3d")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if type(args.steps) is not int or args.steps < 1:
        raise ValueError("steps must be positive")
    output_arg = args.output
    if not output_arg.is_absolute() and output_arg.parts and output_arg.parts[0] == "evotac":
        output_arg = Path(*output_arg.parts[1:])
    output = output_path(output_arg)
    output.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("π0.5 training requires a working CUDA runtime; torch.cuda.is_available() is false")
    try:
        torch.empty(1, device=args.device)
    except RuntimeError as exc:
        raise RuntimeError(f"π0.5 training CUDA device probe failed: {exc}") from exc
    config = make_config(args.base_checkpoint, device="cpu", train_expert_only=True,
                         gradient_checkpointing=True)
    policy, _ = load_policy_strict(args.base_checkpoint, "cpu", train_expert_only=True,
                                   gradient_checkpointing=True)
    stats = json.loads((args.dataset / "stats.json").read_text())
    tokenizer = ROOT / "checkpoints/pi05_tokenizer_20260924/big_vision/paligemma_tokenizer.model"
    preprocessor, _ = make_processors(config, stats, tokenizer)
    dataset = ChunkDataset(args.dataset, config.chunk_size)
    policy.to(args.device).train()
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=2.5e-5)
    losses = []
    for step in range(args.steps):
        batch = preprocessor(input_batch(dataset.get(np.random.randint(len(dataset.rows))),
                                          "insert the stick to the hole."))
        loss, _ = policy(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1, "loss": losses[-1]}), flush=True)
    policy.eval()
    policy.save_pretrained(output)
    evidence = {"schema": "evotac.pi05_finetune.v1", "split": "train", "steps": args.steps,
                "seed": args.seed, "base_checkpoint": str(args.base_checkpoint.resolve()),
                "base_sha256": sha256(args.base_checkpoint / "model.safetensors"),
                "dataset": str(args.dataset.resolve()), "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
                "stats_sha256": sha256(args.dataset / "stats.json"), "final_loss": losses[-1],
                "loss_samples": losses[-10:]}
    (output / "evotac_training_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
