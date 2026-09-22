"""Simulator-free contracts shared by control, logging and replay."""
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

SCHEMA_VERSION = "evotac.phase1.v1"
ACTION_VERSION = "world_rotvec_panda_hand.v1"
PREFIX_VERSION = "accepted_targets.v2_standard_init"
ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_NAMES = ("panda_finger_joint1", "panda_finger_joint2")


class ActionKind(str, Enum):
    JOINT_TARGET = "joint_target"
    RECOVERY_DELTA = "recovery_delta"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    values: tuple[float, ...]
    version: str = ACTION_VERSION

    def __post_init__(self):
        kind = ActionKind(self.kind)
        values = np.asarray(self.values, dtype=np.float64)
        expected = 8 if kind is ActionKind.JOINT_TARGET else 7
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError(f"{kind.value} requires {expected} finite values")
        if self.version != ACTION_VERSION:
            raise ValueError(f"Unsupported action version: {self.version}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "values", tuple(map(float, values)))

    def as_dict(self):
        return {"kind": self.kind.value, "values": list(self.values), "version": self.version}


@dataclass
class RobotState:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    finger_position: np.ndarray
    ee_position_world: np.ndarray
    ee_quaternion_world: np.ndarray  # wxyz, panda_hand
    base_quaternion_world: np.ndarray
    jacobian_world: np.ndarray | None = None

    def as_dict(self):
        return {key: freeze(value) for key, value in vars(self).items() if key != "jacobian_world"}


def freeze(value: Any):
    """Detach simulator buffers before the next physics or render update."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {str(k): freeze(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [freeze(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_value(value):
    value = freeze(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_value(v) for v in value]
    return value
