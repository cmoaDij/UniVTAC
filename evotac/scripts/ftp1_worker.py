"""Persistent FTP-1 inference worker, run only by the isolated model Python."""
import argparse
import ast
import contextlib
import json
import logging
import random
from pathlib import Path
import sys
import time
import traceback

import numpy as np


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    load_warnings = []
    class LoadAudit(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if any(s in message for s in ("Missing keys while loading checkpoint", "Unexpected keys while loading checkpoint", "skipping tokenizer loading")):
                load_warnings.append(message)
    handler = LoadAudit()
    logging.getLogger().addHandler(handler)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper
            torch.set_num_threads(8)
            train = json.loads((args.checkpoint/"train_config.json").read_text())
            if train.get("action_joint_rep") not in {"absolute", "relative", "mix"}:
                raise ValueError("Unknown checkpoint action representation")
            for name in ("GelSightMini_image_224_224_3.safetensors", "shared_image_chunk_encoder.safetensors"):
                if not (args.checkpoint/"hpt_tokenizer"/name).is_file():
                    raise FileNotFoundError(f"Missing trained tactile weights: {name}")
            model = FTP1InferenceWrapper(str(args.checkpoint), args.domain, device="cuda:0", num_inference_steps=args.steps)
            # safetensors saves the shared embedding through lm_head only. The
            # upstream loader deliberately omits the two unused expert heads.
            from safetensors import safe_open
            paligemma = model.model.paligemma_with_expert.paligemma
            with safe_open(args.checkpoint/"model.safetensors", framework="pt", device="cpu") as weights:
                tied_embedding_verified = (
                    "paligemma_with_expert.paligemma.lm_head.weight" in weights.keys()
                    and paligemma.get_input_embeddings().weight is paligemma.lm_head.weight)
            allowed_missing = {
                "paligemma_with_expert.gemma_expert.lm_head.weight",
                "paligemma_with_expert.gemma_tactile_expert.lm_head.weight",
            }
            if tied_embedding_verified:
                allowed_missing.add("paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight")
            for warning in load_warnings:
                if not warning.startswith("Missing keys while loading checkpoint"):
                    raise RuntimeError("Incomplete checkpoint loading: "+warning)
                missing = set(ast.literal_eval(warning.split(": ", 1)[1]))
                if missing - allowed_missing:
                    raise RuntimeError("Unexplained missing checkpoint keys: "+str(sorted(missing-allowed_missing)))
            if model.get_state_dim() != 120 or model.get_action_dim() != 120 or not model.model_config.disable_history:
                raise ValueError("Adapter requires the released stateless 120D UniVTAC checkpoint")
            torch.manual_seed(args.seed)
            metadata = {"status": "ready", "backend": "ftp1", "checkpoint": str(args.checkpoint),
                        "domain": args.domain, "state_dim": 120, "action_dim": 120,
                        "action_horizon": model.get_action_horizon(), "action_joint_rep": train["action_joint_rep"],
                        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "inference_seed": args.seed,
                        "weight_load_warnings": load_warnings, "tied_embedding_verified": tied_embedding_verified,
                        "normalization_enabled": not model.skip_normalization}
        emit(metadata)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("operation") == "close":
                return
            if request.get("operation") == "reset_seed":
                seed = request["seed"]
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise ValueError("Invalid continuation inference seed")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                emit({"status": "seed_reset", "seed": seed,
                      "request_id": request["request_id"]})
                continue
            start = time.perf_counter()
            with np.load(request["input_path"], allow_pickle=False) as payload:
                with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                    chunk = model.infer(
                        images={name: payload[name] for name in ("camera_ego_rgb_0", "right_wrist_camera_rgb_0")},
                        state=payload["state"], prompt=request["prompt"],
                        tactiles={"right_tactile_gripper": payload["right_tactile_gripper"]},
                        tactile_function_areas={"right_tactile_gripper": [0, 1]},
                        tactile_sensors={"right_tactile_gripper": "GelSightMini"})
                    torch.cuda.synchronize()
            chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.shape != (metadata["action_horizon"], 120) or not np.isfinite(chunk).all():
                raise RuntimeError("Model produced invalid actions")
            np.save(request["output_path"], chunk, allow_pickle=False)
            emit({"status": "inferred", "request_id": request["request_id"],
                  "wall_seconds": time.perf_counter()-start, "output_shape": list(chunk.shape)})
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
