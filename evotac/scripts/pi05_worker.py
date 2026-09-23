"""Optional LeRobot π0.5 inference worker; never imports LeRobot in Isaac."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def load_policy_strict(checkpoint, device):
    """Load safetensors without LeRobot's silent random-weight fallback."""
    import torch
    from safetensors.torch import load_file
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy

    weight_file = checkpoint / "model.safetensors"
    if not weight_file.is_file() or weight_file.stat().st_size < 1024 * 1024:
        raise FileNotFoundError(f"missing/incomplete π0.5 weights: {weight_file}")
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    # Construct on CPU, load all tensors strictly, then transfer once.
    config.device = "cpu"
    policy = PI05Policy(config)
    original = load_file(str(weight_file), device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(original, policy.config)
    remapped = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}
    missing, unexpected = policy.load_state_dict(remapped, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict π0.5 checkpoint load failed: missing={missing[:3]} unexpected={unexpected[:3]}")
    policy.config.device = device
    policy.to(device)
    policy.eval()
    return policy, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--stats", type=Path,
                        help="train-only q01/q99 statistics used for state/action scaling")
    args = parser.parse_args()
    try:
        import torch
        from lerobot.policies import make_pre_post_processors
        from transformers import GemmaTokenizer
        policy, config = load_policy_strict(args.checkpoint, args.device)
        stats = None
        if args.stats:
            stats = json.loads(args.stats.read_text(encoding="utf-8"))
            for key in ("observation.state", "action"):
                if key not in stats or any(name not in stats[key] for name in ("q01", "q99")):
                    raise ValueError(f"π0.5 stats missing train-only q01/q99 for {key}")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=str(args.checkpoint),
            preprocessor_overrides={"device_processor": {"device": args.device}})
        # The public processor points at a gated HF tokenizer. The equivalent
        # published SentencePiece model is downloaded separately and loaded
        # explicitly, so inference does not depend on account permissions.
        tokenizer_path = Path(__file__).resolve().parents[2] / "checkpoints/pi05_tokenizer_20260924/big_vision/paligemma_tokenizer.model"
        if tokenizer_path.is_file():
            tokenizer = GemmaTokenizer(vocab_file=str(tokenizer_path))
            for step in preprocessor.steps:
                if hasattr(step, "input_tokenizer"):
                    step.input_tokenizer = tokenizer
        metadata = {"status": "ready", "backend": "pi05", "checkpoint": str(args.checkpoint),
                    "state_dim": 8, "action_dim": 8, "tactile_inputs": False,
                    "torch": torch.__version__}
        emit(metadata)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("operation") == "close":
                return
            if request.get("operation") != "infer":
                raise ValueError("unknown π0.5 worker operation")
            with np.load(request["input_path"], allow_pickle=False) as payload:
                raw_state = payload["observation.state"].astype(np.float32)
                if stats:
                    q01 = np.asarray(stats["observation.state"]["q01"], dtype=np.float32)
                    q99 = np.asarray(stats["observation.state"]["q99"], dtype=np.float32)
                    raw_state = 2.0 * (raw_state - q01) / np.maximum(q99 - q01, 1e-6) - 1.0
                batch = {
                    "observation.images.base_0_rgb": torch.from_numpy(payload["observation.images.base_0_rgb"]),
                    "observation.images.left_wrist_0_rgb": torch.from_numpy(payload["observation.images.left_wrist_0_rgb"]),
                    "observation.images.right_wrist_0_rgb": torch.from_numpy(payload["observation.images.right_wrist_0_rgb"]),
                    "observation.state": torch.from_numpy(raw_state),
                    "task": [request["prompt"]],
                }
                batch = preprocessor(batch)
                with torch.inference_mode():
                    action = postprocessor(policy.predict_action_chunk(batch))
                chunk = action.detach().cpu().numpy()
                if stats:
                    q01 = np.asarray(stats["action"]["q01"], dtype=np.float32)
                    q99 = np.asarray(stats["action"]["q99"], dtype=np.float32)
                    chunk[..., :len(q01)] = (chunk[..., :len(q01)] + 1.0) * (q99 - q01) / 2.0 + q01
            np.save(request["output_path"], chunk, allow_pickle=False)
            emit({"status": "inferred", "request_id": request["request_id"],
                  "output_shape": list(chunk.shape)})
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
