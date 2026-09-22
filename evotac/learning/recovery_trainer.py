"""Simulator-independent SAC training loop for real recovery fragments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from evotac.learning.recovery_buffer import ReplayBuffer
from evotac.learning.recovery_sac import RecoverySAC, SACUpdate
from evotac.learning.recovery_training import RecoveryTrainingDriver, TrainingEpisodeResult


@dataclass(frozen=True)
class SACTrainingEpisode:
    """Auditable result of one collection episode and its learner updates."""

    episode: TrainingEpisodeResult
    updates: tuple[SACUpdate, ...]
    skipped_updates: int

    def as_dict(self):
        return {
            "status": self.episode.status,
            "reason": self.episode.reason,
            "recovery_actions": self.episode.recovery_actions,
            "admitted_transitions": self.episode.admitted_transitions,
            "terminal_outcome": self.episode.terminal_outcome,
            "updates": [vars(update) for update in self.updates],
            "skipped_updates": self.skipped_updates,
        }


class RecoverySACTrainer:
    """Connect a :class:`RecoveryTrainingDriver` to a SAC learner.

    The driver owns environment stepping and terminal attribution.  This class
    only samples transitions already admitted by the collector; it never adds
    synthetic labels or inserts base-policy actions into the recovery buffer.
    """

    def __init__(self, driver: RecoveryTrainingDriver, learner: RecoverySAC,
                 *, buffer: ReplayBuffer | None = None, batch_size=32,
                 updates_per_interaction=1, learning_starts=0):
        if not isinstance(driver, RecoveryTrainingDriver):
            raise TypeError("driver must be a RecoveryTrainingDriver")
        if not isinstance(learner, RecoverySAC):
            raise TypeError("learner must be a RecoverySAC")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(updates_per_interaction) is not int or updates_per_interaction < 1:
            raise ValueError("updates_per_interaction must be a positive integer")
        selected_buffer = driver.collector.buffer if buffer is None else buffer
        if not isinstance(selected_buffer, ReplayBuffer):
            raise TypeError("buffer must be a ReplayBuffer")
        if selected_buffer is not driver.collector.buffer:
            raise ValueError("trainer buffer must be the collector's recovery buffer")
        self.driver, self.learner, self.buffer = driver, learner, selected_buffer
        self.batch_size = batch_size
        self.updates_per_interaction = updates_per_interaction
        if type(learning_starts) is not int or learning_starts < 0:
            raise ValueError("learning_starts must be nonnegative")
        self.learning_starts = max(batch_size, learning_starts)

    def run_episode(self, seed: int, condition=None, **reset_kwargs) -> SACTrainingEpisode:
        result = self.driver.run(seed, condition=condition, **reset_kwargs)
        requested = result.admitted_transitions * self.updates_per_interaction
        updates = []
        skipped = 0
        for _ in range(requested):
            if len(self.buffer) < self.learning_starts:
                skipped += 1
                continue
            updates.append(self.learner.update(self.buffer.sample(self.batch_size)))
        return SACTrainingEpisode(result, tuple(updates), skipped)

    def fit(self, seeds, *, condition=None):
        return [self.run_episode(int(seed), condition=condition) for seed in seeds]

    def state_dict(self):
        return {
            "schema": "evotac.recovery_sac_trainer.v1",
            "batch_size": self.batch_size,
            "updates_per_interaction": self.updates_per_interaction,
            "learning_starts": self.learning_starts,
            "learner": self.learner.state_dict(),
            "buffer": self.buffer.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("schema") != "evotac.recovery_sac_trainer.v1":
            raise ValueError("unsupported recovery trainer checkpoint schema")
        if (int(state["batch_size"]) != self.batch_size
                or int(state["updates_per_interaction"]) != self.updates_per_interaction
                or int(state.get("learning_starts", state["batch_size"])) != self.learning_starts):
            raise ValueError("trainer configuration does not match checkpoint")
        self.learner.load_state_dict(state["learner"])
        self.buffer.load_state_dict(state["buffer"])

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path):
        state = torch.load(Path(path), map_location=self.learner.device, weights_only=False)
        self.load_state_dict(state)
