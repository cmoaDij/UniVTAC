"""Deployment-only FTP-1 mapping, following the official UniVTAC adapter.

Reference: michaelyuancb/ftp1-policy, 89fa681d6c014cce28300946b7526db808e0b1c1,
UniVTAC/policy/FTP1/deploy_policy.py (Apache-2.0).
"""
from collections import deque

import numpy as np

from evotac.data.schemas import Action


def encode_observation(observation):
    """Map named state and four valid RGB streams to the released 120D layout."""
    import cv2
    proprio = observation["proprioception"]
    arm = np.asarray(proprio["joint_position"], dtype=np.float32)
    fingers = np.asarray(proprio["finger_position"], dtype=np.float32)
    if arm.shape != (7,) or fingers.shape != (2,) or not np.isfinite(np.r_[arm, fingers]).all():
        raise ValueError("FTP-1 requires seven named arm joints and two finite finger states")
    # Published UniVTAC training uses the first finger, not the average width.
    qpos = np.r_[arm, fingers[0]].astype(np.float32)
    state = np.zeros((1, 120), dtype=np.float32)
    state[0, 9:16], state[0, 44] = arm, fingers[0]
    def image(category, name, field):
        modality = observation["images"][category][name][field]
        data = np.asarray(modality["data"])
        if not modality["valid"] or data.dtype != np.uint8 or data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError(f"Missing/invalid RGB observation: {category}/{name}/{field}")
        return cv2.resize(data, (224, 224), interpolation=cv2.INTER_LINEAR)
    return {
        "state": state,
        "camera_ego_rgb_0": image("camera", "head", "rgb"),
        "right_wrist_camera_rgb_0": image("camera", "wrist", "rgb"),
        "right_tactile_gripper": np.stack([
            image("tactile", "left_tactile", "rgb_marker"),
            image("tactile", "right_tactile", "rgb_marker")], axis=0)[None].astype(np.float32),
    }, qpos


def decode_chunk(chunk, base_qpos, representation, first_index=1, count=1, stride=1):
    """Resolve relative/mix outputs against the inference-time state once."""
    chunk = np.asarray(chunk, dtype=np.float32)
    base = np.asarray(base_qpos, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 120 or not np.isfinite(chunk).all():
        raise ValueError("Expected a finite FTP-1 action chunk with 120 columns")
    if base.shape != (8,) or not np.isfinite(base).all():
        raise ValueError("Invalid inference-time qpos")
    if representation not in {"absolute", "relative", "mix"}:
        raise ValueError("Checkpoint must explicitly declare action_joint_rep")
    if any(type(n) is not int for n in (first_index, count, stride)) or first_index < 1 or count < 1 or stride < 1:
        raise ValueError("Invalid chunk execution indices")
    indices = [first_index+i*stride for i in range(count)]
    if indices[-1] >= len(chunk):
        raise ValueError("Requested action prefix exceeds the predicted horizon")
    result = []
    for index in indices:
        target = np.r_[chunk[index, 9:16], chunk[index, 44]].copy()
        if representation in {"relative", "mix"}:
            target[:7] += base[:7]
        if representation == "relative":
            target[7] += base[7]
        result.append(Action("joint_target", target))
    return result


class FTP1Policy:
    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.pending = deque()
        self.inference_count = 0
        self.last_inference = None

    def reset(self):
        self.pending.clear()
        self.last_inference = None

    def switch_control(self):
        self.reset()

    def next_action(self, observation, prompt):
        if not self.pending:
            inputs, base = encode_observation(observation)
            chunk, metadata = self.backend.infer(inputs, prompt)
            self.pending.extend(decode_chunk(chunk, base, self.backend.metadata["action_joint_rep"],
                self.config["chunk_first_index"], self.config["execute_chunk_steps"], self.config["chunk_stride"]))
            self.inference_count += 1
            self.last_inference = metadata
        return self.pending.popleft()
