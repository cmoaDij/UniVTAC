"""Frozen UniVTAC image features and deployment-only recovery observations."""
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF

from encoder.network import Tactile


PREPROCESS_VERSION = "univtac_marked_rgb_float01_resize_hw320x240.v1"
SENSORS = ("left_tactile", "right_tactile")
PROPRIO_FIELDS = ("joint_position", "joint_velocity", "finger_position", "ee_position_world", "ee_quaternion_world")
CAMERAS = ("head", "wrist")


class CachedTactileEncoder:
    """Bounded content cache for immutable, frozen image features.

    Wrapper copies reference images at every step. Content keys work across
    those copies and invalidate changed references after reset. The cache
    retains small latent vectors rather than source images.
    """

    def __init__(self, encoder, capacity=32):
        if type(capacity) is not int or capacity < 6:
            raise ValueError("encoder cache needs capacity >= 6")
        self.encoder, self.capacity = encoder, capacity
        self.version = dict(encoder.version)
        self.cache = OrderedDict()
        self.encoded_images = 0
        self.cache_hits = 0

    def __call__(self, images):
        images = np.asarray(images)
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("Expected batched uint8 RGB HWC tactile images")
        keys = [(image.shape, hashlib.sha256(image.tobytes()).digest()) for image in images]
        missing = {}
        for key, image in zip(keys, images):
            if key not in self.cache:
                missing.setdefault(key, image)
        fresh = {}
        if missing:
            values = self.encoder(np.stack(list(missing.values())))
            fresh = dict(zip(missing, values))
            self.encoded_images += len(missing)
        self.cache_hits += len(keys) - len(missing)
        result = np.stack([self.cache[key] if key in self.cache else fresh[key] for key in keys])
        for key, value in zip(keys, result):
            self.cache[key] = value.copy()
            self.cache.move_to_end(key)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return result


class FrozenTactileEncoder(torch.nn.Module):
    def __init__(self, checkpoint, expected_sha256, device="cpu"):
        super().__init__()
        path = Path(checkpoint)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError("Tactile encoder SHA256 does not match pinned provenance")
        model = Tactile(backbone="resnet18", supervise=["marker", "rgb", "marked_rgb", "pose", "depth"])
        weights = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        self.backbone = model.backbone.to(device)
        self.requires_grad_(False)
        self.train(False)
        self.version = {"checkpoint_sha256": actual, "preprocessing": PREPROCESS_VERSION,
                        "feature_dimension": 512, "sensor_order": list(SENSORS),
                        "input": "RGB uint8 HWC rgb_marker; tensor resize H=320,W=240; float32 [0,1]",
                        "load": "full Tactile state_dict strict=True; backbone retained", "batchnorm": "eval frozen buffers"}

    def train(self, mode=True):
        # An enclosing SAC module calling train() must not unfreeze BatchNorm.
        return super().train(False)

    @torch.inference_mode()
    def forward(self, images):
        data = np.asarray(images)
        if data.dtype != np.uint8 or data.ndim != 4 or data.shape[-1] != 3:
            raise ValueError("Expected batched uint8 RGB HWC tactile images")
        tensor = torch.from_numpy(np.ascontiguousarray(data)).permute(0, 3, 1, 2).float()/255
        tensor = TF.resize(tensor, [320, 240], antialias=True).to(next(self.backbone.parameters()).device)
        value = self.backbone(tensor)
        if value.shape != (len(data), 512) or not torch.isfinite(value).all():
            raise RuntimeError("Invalid tactile encoder features")
        return value.cpu().numpy().copy()


def recovery_features(observation, encoder, *, recovery_remaining_steps):
    """Whitelisted features; references are encoded separately, never as 9 channels."""
    if type(recovery_remaining_steps) is not int or recovery_remaining_steps < 0:
        raise ValueError("Recovery budget must be a known nonnegative integer")
    current, ages = [], []
    for sensor in SENSORS:
        sample = observation["images"]["tactile"][sensor]["rgb_marker"]
        if (not sample["valid"] or sample["sample_physics_step"] > observation["physics_step"]
                or sample["age_steps"] != observation["physics_step"] - sample["sample_physics_step"]):
            raise ValueError("Invalid or future tactile sample")
        current.append(sample["data"])
        ages.append(sample["age_steps"])
    camera_ages = []
    camera_cues = []
    if "camera" in observation["images"]:
        for camera in CAMERAS:
            sample = observation["images"]["camera"][camera]["rgb"]
            age = observation["physics_step"] - sample["sample_physics_step"]
            if not sample["valid"] or age < 0 or sample["age_steps"] != age:
                raise ValueError("Invalid or future camera sample")
            camera_ages.append(age)
            image = np.asarray(sample["data"], dtype=np.float32) / 255.0
            camera_cues.append(np.concatenate([image.mean((0, 1)), image.std((0, 1))]))
    else:
        # Simulator-independent feature probes from v1 predate the camera
        # fields.  Real wrapper observations always include both cameras.
        camera_ages = [0, 0]
        camera_cues = [np.zeros(6, np.float32), np.zeros(6, np.float32)]
    if ("remaining_recovery_steps" in observation
            and observation["remaining_recovery_steps"] != recovery_remaining_steps):
        raise ValueError("Recovery budget disagrees with the captured observation")
    groups = [np.stack(current)]
    reference_steps = []
    for kind in ("empty", "grasp"):
        reference = observation["references"][kind]
        if reference is None or not reference["valid"] or reference["physics_step"] > observation["physics_step"]:
            raise ValueError("Valid earlier tactile references are required")
        groups.append(np.stack([reference["images"][s]["rgb_marker"] for s in SENSORS]))
        reference_steps.append(reference["physics_step"])
    if reference_steps[0] >= reference_steps[1]:
        raise ValueError("Empty reference must precede grasp reference")
    features = encoder(np.concatenate(groups)).reshape(3, 2, 512)
    # Signed per-channel means, absolute means and RMS, in normalized intensity.
    cues = []
    for reference in groups[1:]:
        delta = (groups[0].astype(np.float32)-reference.astype(np.float32))/255
        cues.append(np.concatenate([delta.mean((1, 2)), np.abs(delta).mean((1, 2)),
                                    np.sqrt((delta**2).mean((1, 2)))], axis=-1))
    p = observation["proprioception"]
    proprio = np.concatenate([p[k] for k in PROPRIO_FIELDS]).astype(np.float32)
    if not np.isfinite(proprio).all():
        raise ValueError("Nonfinite proprioception")
    previous = observation["previous_action"]
    accepted = np.zeros(9, np.float32) if previous is None else np.r_[previous["arm_position"], previous["finger_position"]].astype(np.float32)
    return {"schema": "evotac.recovery_features.v2", "tactile_embeddings": features,
            "embedding_order": ["current", "empty", "grasp"],
            "sensor_order": list(SENSORS), "proprioception_order": list(PROPRIO_FIELDS),
            "difference_order": ["current_minus_empty", "current_minus_grasp"],
            "difference_channels": ["signed_mean_R", "signed_mean_G", "signed_mean_B", "abs_mean_R", "abs_mean_G", "abs_mean_B", "rms_R", "rms_G", "rms_B"],
            "difference_cues": np.stack(cues), "proprioception": proprio,
            "previous_accepted_target": accepted, "previous_action_valid": previous is not None,
            "history_mask": np.asarray(observation["history_mask"], dtype=bool),
            "physics_step": observation["physics_step"], "tactile_age_steps": np.asarray(ages),
            "camera_age_steps": np.asarray(camera_ages),
            "camera_cues": np.stack(camera_cues).astype(np.float32),
            "reference_physics_steps": np.asarray(reference_steps),
            "remaining_task_steps": observation["remaining_physics_steps"],
            "remaining_recovery_steps": recovery_remaining_steps, "encoder_version": dict(encoder.version)}
