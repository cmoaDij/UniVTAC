"""Collect finite-horizon recovery fragments and actual continuation outcomes.

Keep the fragment pending until its final outcome is known. A handoff ends the
recovery MDP: add gamma times the fixed base policy's actual final reward only
to the last recovery action, and disable bootstrapping there. Base actions
never enter the recovery buffer.
"""
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from evotac.data.schemas import Action
from evotac.learning.recovery_buffer import ReplayBuffer, Transition


@dataclass
class CollectionResult:
    transition: Transition | None
    observation: dict | None
    info: dict
    admitted: int = 0


class RecoveryCollector:
    def __init__(self, wrapper, buffer: ReplayBuffer, feature_encoder: Callable, *, discount=0.99):
        if not 0 <= discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        self.wrapper, self.buffer, self.feature_encoder = wrapper, buffer, feature_encoder
        self.discount = discount
        self.pending = []
        self.awaiting_continuation = False

    def discard(self):
        count = len(self.pending)
        self.pending.clear()
        self.awaiting_continuation = False
        return count

    def _commit(self):
        if not self.pending:
            raise RuntimeError("no valid recovery fragment to commit")
        shapes = {(row.observation.shape, row.action.shape) for row in self.pending}
        if len(shapes) != 1:
            raise ValueError("feature shape changed within a recovery fragment")
        count = len(self.pending)
        for row in self.pending:
            self.buffer.add(row)
        self.discard()
        return count

    def step(self, observation, action_vector):
        if self.awaiting_continuation:
            raise RuntimeError("finish the fixed base continuation before another recovery action")
        values = np.asarray(action_vector, dtype=np.float32)
        if values.shape != (7,) or not np.isfinite(values).all() or np.any(np.abs(values) > 1):
            raise ValueError("recovery action must be seven finite values in [-1, 1]")
        state = np.asarray(self.feature_encoder(observation), dtype=np.float32).copy()
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("encoder must return a finite feature vector")
        try:
            next_obs, reward, terminated, truncated, info = self.wrapper.step(Action("recovery_delta", values))
            execution = info["execution_info"]
            valid = (execution.get("valid_trial") is True and execution.get("status") == "executed"
                     and execution.get("physics_steps") == 6 and next_obs is not None)
            if not valid:
                self.discard()
                return CollectionResult(None, next_obs, info)
            next_state = np.asarray(self.feature_encoder(next_obs), dtype=np.float32)
            next_state.setflags(write=False)
            transition = Transition(state, values, reward, next_state, terminated, truncated,
                                    execution["bootstrap_allowed"])
            self.pending.append(transition)
            admitted = self._commit() if terminated or truncated else 0
            return CollectionResult(transition, next_obs, info, admitted)
        except BaseException:
            self.discard()
            raise

    def handoff(self):
        if not self.pending or self.awaiting_continuation:
            raise RuntimeError("handoff requires a live nonempty recovery fragment")
        self.awaiting_continuation = True

    def finish_continuation(self, outcome):
        if not self.awaiting_continuation:
            raise RuntimeError("no handoff awaiting a continuation outcome")
        valid = (outcome.get("valid_trial") is True and not outcome.get("incomplete", False)
                 and outcome.get("reason") in {"success", "object_lost", "task_budget"}
                 and (outcome.get("terminated") is True or outcome.get("truncated") is True))
        if not valid:
            self.discard()
            return 0
        reward = float(outcome["reason"] == "success")
        self.pending[-1] = replace(self.pending[-1], reward=self.pending[-1].reward + self.discount * reward,
                                   terminated=bool(outcome.get("terminated", False)),
                                   truncated=bool(outcome.get("truncated", False)),
                                   bootstrap_allowed=False)
        return self._commit()
