"""A small, explicit replay buffer for real recovery transitions.

The buffer intentionally stores only transitions produced by the environment.
It has no synthetic success labels and does not accept privileged simulator
metadata as part of the model input.
"""
from dataclasses import dataclass
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool = False
    bootstrap_allowed: bool | None = None

    def __post_init__(self):
        observation = np.asarray(self.observation, dtype=np.float32)
        action = np.asarray(self.action, dtype=np.float32)
        next_observation = np.asarray(self.next_observation, dtype=np.float32)
        if observation.ndim != 1 or not observation.size or next_observation.shape != observation.shape:
            raise ValueError("observation and next_observation must have matching non-scalar shapes")
        if action.ndim != 1 or not action.size or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("action must be a finite normalized vector in [-1, 1]")
        if not np.isfinite(observation).all() or not np.isfinite(next_observation).all():
            raise ValueError("observations must be finite")
        if not np.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        for key in ("terminated", "truncated"):
            if not isinstance(getattr(self, key), (bool, np.bool_)):
                raise ValueError(f"{key} must be boolean")
        bootstrap = self.bootstrap_allowed
        if bootstrap is None:
            bootstrap = not (self.terminated or self.truncated)
        if not isinstance(bootstrap, (bool, np.bool_)) or (self.terminated and bootstrap):
            raise ValueError("bootstrap flag must be boolean and false at a terminal boundary")
        object.__setattr__(self, "bootstrap_allowed", bool(bootstrap))
        object.__setattr__(self, "observation", observation.copy())
        object.__setattr__(self, "action", action.copy())
        object.__setattr__(self, "next_observation", next_observation.copy())
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        for array in (self.observation, self.action, self.next_observation):
            array.setflags(write=False)


class ReplayBuffer:
    """FIFO replay storage with deterministic, serializable sampling."""

    def __init__(self, capacity: int, seed: int = 0):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._items = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self._items)

    def add(self, transition: Transition | dict[str, Any]):
        if isinstance(transition, dict):
            transition = Transition(**transition)
        if not isinstance(transition, Transition):
            raise TypeError("buffer.add requires a Transition or its field mapping")
        if self._items and transition.action.shape != self._items[0].action.shape:
            raise ValueError("all actions in a replay buffer must have the same shape")
        if self._items and transition.observation.shape != self._items[0].observation.shape:
            raise ValueError("all observations in a replay buffer must have the same shape")
        self._items.append(Transition(**vars(transition)))

    def sample(self, batch_size: int, *, rng=None) -> dict[str, np.ndarray]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if len(self) < batch_size:
            raise ValueError(f"need {batch_size} transitions, have {len(self)}")
        generator = self._rng if rng is None else rng
        indices = generator.choice(len(self._items), size=batch_size, replace=False)
        rows = [self._items[int(index)] for index in indices]
        return {
            "observation": np.stack([row.observation for row in rows]),
            "action": np.stack([row.action for row in rows]),
            "reward": np.asarray([row.reward for row in rows], dtype=np.float32),
            "next_observation": np.stack([row.next_observation for row in rows]),
            "terminated": np.asarray([row.terminated for row in rows], dtype=np.float32),
            "truncated": np.asarray([row.truncated for row in rows], dtype=np.float32),
            "bootstrap_allowed": np.asarray([row.bootstrap_allowed for row in rows], dtype=np.float32),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "transitions": [
                {"observation": row.observation, "action": row.action, "reward": row.reward,
                 "next_observation": row.next_observation, "terminated": row.terminated,
                 "truncated": row.truncated, "bootstrap_allowed": row.bootstrap_allowed}
                for row in self._items
            ],
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]):
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity does not match checkpoint")
        if len(state["transitions"]) > self.capacity:
            raise ValueError("checkpoint contains more transitions than capacity")
        restored = ReplayBuffer(self.capacity)
        for row in state["transitions"]:
            restored.add(Transition(**row))
        restored._rng.bit_generator.state = state["rng_state"]
        self._items, self._rng = restored._items, restored._rng

    def save(self, path: str | Path):
        rows = list(self._items)
        arrays = {key: np.asarray([getattr(row, key) for row in rows]) for key in (
            "observation", "action", "reward", "next_observation", "terminated", "truncated", "bootstrap_allowed")}
        metadata = {"schema": "evotac.replay_buffer.v2", "capacity": self.capacity,
                    "rng_state": self._rng.bit_generator.state}
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, metadata=json.dumps(metadata), **arrays)

    @classmethod
    def load(cls, path: str | Path):
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["schema"] != "evotac.replay_buffer.v2":
                raise ValueError("unsupported buffer checkpoint schema")
            fields = ("observation", "action", "reward", "next_observation", "terminated", "truncated", "bootstrap_allowed")
            rows = [{**{key: data[key][i] for key in fields},
                     "terminated": bool(data["terminated"][i]),
                     "truncated": bool(data["truncated"][i]),
                     "bootstrap_allowed": bool(data["bootstrap_allowed"][i])}
                    for i in range(len(data["reward"]))]
        buffer = cls(int(metadata["capacity"]))
        buffer.load_state_dict({**metadata, "transitions": rows})
        return buffer
