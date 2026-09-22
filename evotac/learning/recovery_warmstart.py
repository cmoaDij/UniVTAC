"""Shared control history warm-start from explicitly typed local demonstrations."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from evotac.learning.history_encoder import HistoryEncoder
from evotac.learning.recovery_observation import RECOVERY_OBSERVATION_SCHEMA
from evotac.learning.recovery_sac import _GaussianActor


class RecoveryWarmStart(nn.Module):
    def __init__(self, skill_names, *, encoder=None, actor_hidden_dim=128):
        super().__init__()
        if (not skill_names or len(set(skill_names)) != len(skill_names)
                or any(not isinstance(s, str) or not s or "." in s for s in skill_names)):
            raise ValueError("unique nonempty skill names required")
        self.skill_names = tuple(skill_names)
        self.encoder = encoder if encoder is not None else HistoryEncoder()
        self.actor_hidden_dim = actor_hidden_dim
        self.actors = nn.ModuleDict({name: _GaussianActor(self.encoder.config["output_dim"], 7, actor_hidden_dim)
                                    for name in self.skill_names})
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.steps = 0

    def loss(self, history, mask, action, skill_indices):
        state = self.encoder(history, mask)
        action = torch.as_tensor(action, dtype=state.dtype, device=state.device)
        skill_indices = torch.as_tensor(skill_indices, device=state.device)
        if action.shape != (len(state), 7) or not torch.isfinite(action).all() or (action.abs() > 1).any():
            raise ValueError("warm-start needs executed normalized seven-dimensional recovery actions")
        if (skill_indices.shape != (len(state),) or skill_indices.dtype not in (torch.int32, torch.int64)
                or ((skill_indices < 0) | (skill_indices >= len(self.skill_names))).any()):
            raise ValueError("invalid demonstration skill indices")
        loss = state.new_zeros(())
        for index, name in enumerate(self.skill_names):
            selected = skill_indices == index
            if selected.any():
                prediction, _ = self.actors[name].sample(state[selected], deterministic=True)
                loss = loss + (prediction - action[selected]).square().sum()
        return loss / action.numel()

    def train_batch(self, batch):
        required = {"history", "mask", "action", "skill_indices"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"warm-start batch is missing: {sorted(missing)}")
        loss = self.loss(batch["history"], batch["mask"], batch["action"], batch["skill_indices"])
        if not torch.isfinite(loss):
            raise ValueError("warm-start loss is not finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        return float(loss.detach().cpu())

    def checkpoint(self, provenance):
        if self.steps < 1:
            raise ValueError("cannot export an untrained control warm-start")
        if not provenance or any(row.get("split") != "train" for row in provenance):
            raise ValueError("control warm-start provenance must contain only training parents")
        if {row.get("skill") for row in provenance} != set(self.skill_names):
            raise ValueError("each warm-start skill requires demonstration provenance")
        return deepcopy({"schema": "evotac.control_warmstart.v1", "frame_schema": RECOVERY_OBSERVATION_SCHEMA,
                         "skill_names": self.skill_names, "actor_hidden_dim": self.actor_hidden_dim,
                         "encoder": self.encoder.checkpoint(),
                         "actors": self.actors.state_dict(), "steps": self.steps,
                         "provenance": provenance})

    @classmethod
    def from_checkpoint(cls, state, *, device="cpu"):
        if (state.get("schema") != "evotac.control_warmstart.v1"
                or state.get("frame_schema") != RECOVERY_OBSERVATION_SCHEMA or state.get("steps", 0) < 1):
            raise ValueError("a trained control checkpoint with matching frame schema is required")
        model = cls(state["skill_names"], encoder=HistoryEncoder.from_checkpoint(state["encoder"], device=device),
                    actor_hidden_dim=state["actor_hidden_dim"]).to(device)
        model.steps = state["steps"]
        model.actors.load_state_dict(state["actors"], strict=True)
        model.checkpoint(state["provenance"])
        model.encoder.freeze()
        return model
