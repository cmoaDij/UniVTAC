"""Complete deployment feature layout and causal windows for recovery models."""
from collections import deque

import numpy as np

RECOVERY_OBSERVATION_SCHEMA = "evotac.recovery_frame.v2"
RECOVERY_OBSERVATION_DIM = 3162
HISTORY_LENGTH = 8


def _array(features, key, shape):
    value = np.asarray(features[key], dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{key} must have shape {shape} and be finite")
    return value


def recovery_vector(features, *, task_budget_steps=1200, recovery_budget_steps=120,
                    previous_physics_step=None, physical_hz=120):
    """Ordered frame retaining bilateral embeddings and all robot fields.

    Layout: embeddings 3072, cues 36, proprioception 23, accepted targets 9,
    target-valid 1, tactile ages 2, camera ages 2, reference ages 2, camera
    RGB mean/std 12, sample interval 1, remaining budgets 2. Times are
    seconds, budgets fractions.
    """
    if any(type(value) is not int or value <= 0 for value in
           (task_budget_steps, recovery_budget_steps, physical_hz)):
        raise ValueError("budgets and physical_hz must be positive integers")
    if "schema" in features and features["schema"] != "evotac.recovery_features.v2":
        raise ValueError("unsupported recovery feature schema")
    # Keep old simulator-independent probes readable while making the live
    # wrapper contract strict. Production feature dictionaries always contain
    # the v2 schema fields below.
    if "physics_step" not in features:
        cues = np.asarray(features.get("difference_cues"), dtype=np.float32)
        if cues.shape == (2, 2, 9):
            cues = cues.ravel()
        elif cues.shape == (2, 9):
            cues = np.repeat(cues[:, None, :], 2, axis=1).ravel()
        else:
            raise ValueError("legacy difference_cues shape is invalid")
        proprio = np.asarray(features.get("proprioception"), dtype=np.float32).ravel()
        if proprio.size > 23 or not np.isfinite(proprio).all():
            raise ValueError("legacy proprioception is invalid")
        previous = np.asarray(features.get("previous_accepted_target", np.zeros(9)), dtype=np.float32).ravel()
        if previous.shape != (9,):
            raise ValueError("legacy previous target must have nine values")
        vector = np.zeros(RECOVERY_OBSERVATION_DIM, dtype=np.float32)
        vector[3072:3108] = cues
        vector[3108:3108 + proprio.size] = proprio
        vector[3131:3140] = previous
        vector[3140] = float(features.get("previous_action_valid", False))
        vector[3141:3143] = np.asarray(features.get("tactile_age_steps", [0, 0]), dtype=np.float32) / physical_hz
        vector[-2:] = [float(features.get("remaining_task_steps", 0)) / task_budget_steps,
                       float(features.get("remaining_recovery_steps", 0)) / recovery_budget_steps]
        return vector
    step = features["physics_step"]
    if type(step) is not int or step < 0:
        raise ValueError("physics_step must be a nonnegative integer")
    interval = 0 if previous_physics_step is None else step - previous_physics_step
    if interval < 0:
        raise ValueError("feature history must be chronological")
    tactile_ages = _array(features, "tactile_age_steps", (2,))
    camera_ages = _array(features, "camera_age_steps", (2,))
    reference_ages = step - _array(features, "reference_physics_steps", (2,))
    budgets = _array(features, "remaining_task_steps", ()), _array(features, "remaining_recovery_steps", ())
    if (np.any(tactile_ages < 0) or np.any(camera_ages < 0) or np.any(reference_ages < 0)
            or not 0 <= budgets[0] <= task_budget_steps or not 0 <= budgets[1] <= recovery_budget_steps):
        raise ValueError("invalid ages or remaining budgets")
    valid = features["previous_action_valid"]
    if not isinstance(valid, (bool, np.bool_)):
        raise ValueError("previous_action_valid must be boolean")
    vector = np.concatenate([
        _array(features, "tactile_embeddings", (3, 2, 512)).ravel(),
        _array(features, "difference_cues", (2, 2, 9)).ravel(),
        _array(features, "proprioception", (23,)),
        _array(features, "previous_accepted_target", (9,)),
        np.asarray([valid], dtype=np.float32),
        tactile_ages / physical_hz, camera_ages / physical_hz,
        reference_ages / physical_hz,
        _array(features, "camera_cues", (2, 6)).ravel(),
        np.asarray([interval / physical_hz, budgets[0] / task_budget_steps,
                    budgets[1] / recovery_budget_steps], dtype=np.float32),
    ]).astype(np.float32)
    if vector.shape != (RECOVERY_OBSERVATION_DIM,):
        raise RuntimeError("recovery frame layout mismatch")
    return vector


def recovery_summary(features):
    """Uncalibrated image-change proxy for development only.

    Intensity change is not force or calibrated slip probability. Robot
    coordinates with different units must not be pooled into a risk score.
    Formal trigger/handoff calibration remains a separate P2/P3 gate.
    """
    cues = _array(features, "difference_cues", (2, 2, 9))
    ages = _array(features, "tactile_age_steps", (2,))
    grasp_change = float(np.clip(np.mean(cues[1, :, 3:6]) * 4.0, 0, 1))
    remaining = features["remaining_task_steps"]
    if type(remaining) is not int or remaining < 0:
        raise ValueError("remaining_task_steps must be nonnegative integer")
    return {"object_lost_risk": grasp_change, "contact_blocked": 0.0,
            "remaining_physics_steps": remaining,
            "stable": bool(grasp_change < 0.28 and np.max(ages) <= 6),
            "calibrated": False}


class RecoveryHistory:
    """Window ending at the current observation, with masked left padding.

    Reset explicitly between episodes. Multiple consumers may request the
    same immutable feature object without advancing time. Replay starts must
    feed the historical prefix first. Wrapper history_mask counts past
    transitions and is not a mask for this current-inclusive window.
    """

    def __init__(self, *, length=HISTORY_LENGTH, **vector_kwargs):
        if type(length) is not int or length < 1:
            raise ValueError("history length must be a positive integer")
        self.length, self.vector_kwargs = length, vector_kwargs
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=self.length)
        self.last_features = None
        self.last_step = None

    def append(self, features):
        if features is not self.last_features:
            step = features["physics_step"]
            if self.last_step is not None and step <= self.last_step:
                raise ValueError("non-increasing history; reset on episode/replay boundary")
            frame = recovery_vector(features, previous_physics_step=self.last_step, **self.vector_kwargs)
            self.frames.append(frame)
            self.last_step, self.last_features = step, features
        data = np.zeros((self.length, RECOVERY_OBSERVATION_DIM), dtype=np.float32)
        mask = np.zeros(self.length, dtype=bool)
        data[-len(self.frames):] = np.stack(self.frames)
        mask[-len(self.frames):] = True
        return data, mask
