"""Pure observation/action mapping for a tactile-free π0.5 controller.

The adapter intentionally accepts only RGB, language and proprioception.  It
does not import LeRobot, so the contract can be tested in the UniVTAC runtime
before the optional π0.5 inference environment is installed.
"""
from __future__ import annotations

import numpy as np

from evotac.data.schemas import Action


STATE_DIM = 8
MODEL_STATE_DIM = 32
MODEL_ACTION_DIM = 32


def encode_observation(observation):
    """Return LeRobot-style RGB/state inputs and the 8D current qpos."""
    proprio = observation["proprioception"]
    arm = np.asarray(proprio["joint_position"], dtype=np.float32)
    fingers = np.asarray(proprio["finger_position"], dtype=np.float32)
    if arm.shape != (7,) or fingers.shape != (2,) or not np.isfinite(np.r_[arm, fingers]).all():
        raise ValueError("π0.5 requires seven arm joints and two finite finger states")
    qpos = np.r_[arm, fingers.mean()].astype(np.float32)

    def image(name):
        modality = observation["images"]["camera"][name]["rgb"]
        data = np.asarray(modality["data"])
        if not modality["valid"] or data.dtype != np.uint8 or data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError(f"Missing/invalid RGB observation: camera/{name}/rgb")
        # Match train/inference resizing exactly; preserve aspect ratio.
        import cv2
        h, w = data.shape[:2]
        scale = 224 / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        result = np.zeros((224, 224, 3), dtype=np.uint8)
        top, left = (224 - nh) // 2, (224 - nw) // 2
        result[top:top + nh, left:left + nw] = cv2.resize(data, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return result

    return {"observation.images.base_0_rgb": image("head"),
            "observation.images.left_wrist_0_rgb": image("wrist"),
            "observation.state": qpos.copy()}, qpos



def decode_chunk(chunk, base_qpos, *, representation="absolute", first_index=0,
                 count=1, stride=1):
    """Decode a π0.5 32D action chunk into UniVTAC joint-target actions."""
    chunk = np.asarray(chunk, dtype=np.float32)
    base = np.asarray(base_qpos, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] < STATE_DIM or not np.isfinite(chunk).all():
        raise ValueError("π0.5 must return a finite [horizon, >=8] action chunk")
    if base.shape != (STATE_DIM,) or not np.isfinite(base).all():
        raise ValueError("Invalid inference-time qpos")
    if representation not in {"absolute", "relative"}:
        raise ValueError("π0.5 action representation must be absolute or relative")
    if any(type(v) is not int for v in (first_index, count, stride)) or first_index < 0 or count < 1 or stride < 1:
        raise ValueError("Invalid π0.5 chunk execution indices")
    indices = [first_index + i * stride for i in range(count)]
    if indices[-1] >= len(chunk):
        raise ValueError("Requested action prefix exceeds π0.5 prediction horizon")
    result = []
    for index in indices:
        target = chunk[index, :STATE_DIM].copy()
        if representation == "relative":
            target += base
        result.append(Action("joint_target", target))
    return result
