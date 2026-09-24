"""Optional LeRobot π0.5 inference worker; never imports LeRobot in Isaac."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from evotac.config import ROOT
from evotac.policy.pi05_runtime import input_batch, make_config, make_processors


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def load_policy_strict(checkpoint, device, *, steps=10, train_expert_only=False,
                       gradient_checkpointing=False):
    """Load safetensors without LeRobot's silent random-weight fallback."""
    from safetensors.torch import load_file
    from lerobot.policies.pi05 import PI05Policy

    weight_file = Path(checkpoint) / "model.safetensors"
    if not weight_file.is_file():
        raise FileNotFoundError(f"missing π0.5 weights: {weight_file}")
    config = make_config(checkpoint, device="cpu", steps=steps,
                        train_expert_only=train_expert_only,
                        gradient_checkpointing=gradient_checkpointing)
    policy = PI05Policy(config)
    original = load_file(str(weight_file), device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(original, policy.config)
    remapped = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}
    missing, unexpected = policy.load_state_dict(remapped, strict=False)
    # The released safetensors omits the tied PaliGemma input embedding and
    # stores the same tensor as lm_head. Every other parameter must match.
    allowed_missing = {"model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"strict π0.5 checkpoint load failed: missing={missing[:3]} unexpected={unexpected[:3]}")
    if missing:
        model = policy.model.paligemma_with_expert.paligemma
        model.model.language_model.embed_tokens.weight = model.lm_head.weight
    policy.config.device = device
    policy.to(device)
    policy.eval()
    return policy, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "checkpoints/pi05_tokenizer_20260924/big_vision/paligemma_tokenizer.model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        import torch
        if args.steps < 1:
            raise ValueError("inference steps must be positive")
        torch.set_num_threads(8)
        torch.manual_seed(args.seed)
        stats = json.loads(args.stats.read_text(encoding="utf-8"))
        # Reserve stdout for JSON transport, including model construction.
        with redirect_stdout(sys.stderr):
            config = make_config(args.checkpoint, device=args.device, steps=args.steps)
            preprocessor, postprocessor = make_processors(config, stats, args.tokenizer)
            policy, config = load_policy_strict(args.checkpoint, args.device, steps=args.steps)
        emit({"status": "ready", "backend": "pi05", "checkpoint": str(args.checkpoint),
              "state_dim": 8, "action_dim": 8, "tactile_inputs": False,
              "torch": torch.__version__, "seed": args.seed, "inference_steps": args.steps,
              "evidence_scope": "model interface; fine-tuning and physical competence unverified"})
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("operation") == "close":
                return
            if request.get("operation") != "infer":
                raise ValueError("unknown π0.5 worker operation")
            with np.load(request["input_path"], allow_pickle=False) as payload:
                with redirect_stdout(sys.stderr), torch.inference_mode():
                    batch = preprocessor(input_batch(dict(payload), request["prompt"]))
                    action = postprocessor(policy.predict_action_chunk(batch))
                chunk = action.detach().cpu().numpy()[0]
            if chunk.shape != (config.chunk_size, 8) or not np.isfinite(chunk).all():
                raise ValueError("invalid π0.5 action output")
            with Path(request["output_path"]).open("xb") as handle:
                np.save(handle, chunk, allow_pickle=False)
            emit({"status": "inferred", "request_id": request["request_id"], "output_shape": list(chunk.shape)})
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
