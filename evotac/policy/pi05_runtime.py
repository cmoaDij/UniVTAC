"""Optional LeRobot runtime shared by inference and future fine-tuning.

Imports stay inside functions so the Isaac environment need not load LeRobot.
"""
from pathlib import Path

import numpy as np

IMAGE_KEYS = ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb")


def input_batch(inputs, prompt):
    """Unbatched HWC uint8 -> batched CHW float [0,1], with exactly two cameras."""
    import torch
    allowed = {*IMAGE_KEYS, "observation.state", "action"}
    if not set(inputs).issubset(allowed) or not {*IMAGE_KEYS, "observation.state"}.issubset(inputs):
        raise ValueError("π0.5 requires exactly two camera streams and proprioception")
    state = np.asarray(inputs["observation.state"])
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("π0.5 requires a finite 8D state")
    batch = {"observation.state": torch.as_tensor(state, dtype=torch.float32)[None], "task": [prompt]}
    for key in IMAGE_KEYS:
        rgb = np.asarray(inputs[key])
        if rgb.dtype != np.uint8 or rgb.shape != (224, 224, 3):
            raise ValueError("π0.5 requires adapter-resized 224x224 RGB uint8")
        batch[key] = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255.0
    if "action" in inputs:
        action = np.asarray(inputs["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 8 or not np.isfinite(action).all():
            raise ValueError("π0.5 training action must have shape [horizon, 8]")
        batch["action"] = torch.from_numpy(action.copy())[None]
    return batch


def make_config(checkpoint, *, device="cpu", steps=10, train_expert_only=False,
                gradient_checkpointing=False):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    if not isinstance(config, PI05Config):
        raise ValueError("checkpoint is not π0.5")
    config.device = device
    config.num_inference_steps = steps
    config.train_expert_only = train_expert_only
    config.gradient_checkpointing = gradient_checkpointing
    # These feature dimensions describe the environment; the internal model
    # still pads state/actions to the released 32D architecture.
    config.input_features = {key: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for key in IMAGE_KEYS}
    config.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (8,))
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (8,))}
    return config


def make_processors(config, stats, tokenizer_path):
    """Construct processors locally; normalize once, using audited train stats."""
    from transformers import GemmaTokenizer
    from lerobot.processor import (PolicyProcessorPipeline, NormalizerProcessorStep,
                                   UnnormalizerProcessorStep, TokenizerProcessorStep,
                                   DeviceProcessorStep)
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    for key in ("observation.state", "action"):
        for name in ("q01", "q99"):
            value = np.asarray(stats[key][name])
            if value.shape != (8,) or not np.isfinite(value).all():
                raise ValueError(f"invalid train normalization statistic: {key}.{name}")
        if np.any(np.asarray(stats[key]["q99"]) < stats[key]["q01"]):
            raise ValueError("reversed quantiles")
    if not Path(tokenizer_path).is_file():
        raise FileNotFoundError(tokenizer_path)
    tokenizer = GemmaTokenizer(vocab_file=str(tokenizer_path), add_bos_token=True, add_eos_token=False)
    pre = PolicyProcessorPipeline(steps=[
        NormalizerProcessorStep(features={**config.input_features, **config.output_features},
                                norm_map=config.normalization_mapping, stats=stats),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(tokenizer=tokenizer, max_length=config.tokenizer_max_length,
                               padding_side="right", padding="max_length"),
        DeviceProcessorStep(device=config.device)], name="evotac_pi05_pre")
    post = PolicyProcessorPipeline(steps=[
        UnnormalizerProcessorStep(features=config.output_features, norm_map=config.normalization_mapping, stats=stats),
        DeviceProcessorStep(device="cpu")], name="evotac_pi05_post",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return pre, post
