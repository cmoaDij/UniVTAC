"""Version-aware recovery-effect predictor and masked training step."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn

from evotac.learning.recovery_observation import HISTORY_LENGTH, RECOVERY_OBSERVATION_DIM


class EffectHistoryEncoder(nn.Module):
    """Independent effect-history branch producing the planned 64D embedding.

    It deliberately does not share the control ``HistoryEncoder`` parameters;
    metric supervision and P4 updates can therefore never alter the frozen
    recovery controller representation.
    """

    def __init__(self, input_dim=RECOVERY_OBSERVATION_DIM, hidden_dim=128,
                 embedding_dim=64, history_length=HISTORY_LENGTH):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in
               (input_dim, hidden_dim, embedding_dim, history_length)):
            raise ValueError("effect history dimensions must be positive integers")
        self.config = {"input_dim": input_dim, "hidden_dim": hidden_dim,
                       "embedding_dim": embedding_dim, "history_length": history_length}
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.embedding = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, frames, mask):
        frames = torch.as_tensor(frames, dtype=torch.float32, device=self.embedding.weight.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=frames.device)
        if frames.ndim != 3 or tuple(frames.shape[1:]) != (self.config["history_length"], self.config["input_dim"]):
            raise ValueError("effect history frame shape does not match encoder")
        if mask.shape != frames.shape[:2] or not mask.any(-1).all() or not torch.isfinite(frames[mask]).all():
            raise ValueError("effect history mask or frames are invalid")
        projected = self.projection(frames)
        lengths = mask.sum(-1).to(torch.int64)
        # Left padding is the only accepted layout, as produced by
        # RecoveryHistory; packing prevents padding from influencing the final
        # embedding and keeps the time mask explicit.
        if mask[:, :-1].logical_and(~mask[:, 1:]).any():
            raise ValueError("effect history mask must be right-aligned")
        # Strip left padding before packing; pack_padded_sequence assumes valid
        # frames begin at column zero.
        aligned = nn.utils.rnn.pad_sequence(
            [projected[index, -int(length):] for index, length in enumerate(lengths)], batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(aligned, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.embedding(hidden[-1])

    def checkpoint(self):
        return {"schema": "evotac.effect_history_encoder.v1", "config": self.config,
                "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, state, *, device="cpu"):
        if state.get("schema") != "evotac.effect_history_encoder.v1":
            raise ValueError("unsupported effect history encoder schema")
        model = cls(**state["config"]).to(device)
        model.load_state_dict(state["state_dict"], strict=True)
        return model


@dataclass(frozen=True)
class EffectLabel:
    embedding: np.ndarray
    task_success: float
    recovery_cost: float
    violation: float
    skill_name: str
    version: str
    valid: bool = True

    def __post_init__(self):
        embedding = np.asarray(self.embedding, dtype=np.float32)
        if embedding.ndim != 1 or not np.isfinite(embedding).all():
            raise ValueError("embedding must be a finite vector")
        if any(not np.isfinite(float(x)) for x in (self.task_success, self.recovery_cost, self.violation)):
            raise ValueError("effect targets must be finite")
        if not 0.0 <= float(self.task_success) <= 1.0 or not 0.0 <= float(self.violation) <= 1.0:
            raise ValueError("task_success and violation must be probabilities in [0, 1]")
        if float(self.recovery_cost) < 0.0:
            raise ValueError("recovery_cost must be nonnegative")
        if not self.skill_name or not self.version:
            raise ValueError("skill_name and version are required")
        object.__setattr__(self, "embedding", embedding.copy())
        object.__setattr__(self, "task_success", float(self.task_success))
        object.__setattr__(self, "recovery_cost", float(self.recovery_cost))
        object.__setattr__(self, "violation", float(self.violation))
        object.__setattr__(self, "valid", bool(self.valid))


def collate_effect_labels(labels, skill_names, *, embedding_dim=None):
    """Pack sparse version-compatible labels into masked predictor targets.

    Each row is one parent-scene state. Missing skill branches remain masked;
    they are never turned into negative outcomes. Labels with a different
    effect version should be filtered by the caller before this function.
    """
    skill_names = tuple(skill_names)
    if not skill_names:
        raise ValueError("skill_names must be nonempty")
    rows = list(labels)
    if not rows:
        raise ValueError("at least one label row is required")
    if embedding_dim is None:
        first = rows[0]
        if isinstance(first, EffectLabel):
            embedding_dim = first.embedding.size
        elif isinstance(first, Mapping):
            first_embedding = np.asarray(first.get("embedding"), dtype=np.float32)
            if first_embedding.ndim != 1:
                raise ValueError("effect embedding must be a one-dimensional vector")
            embedding_dim = first_embedding.size
        else:
            raise TypeError("labels must contain EffectLabel or row mappings")
    if type(embedding_dim) is not int or embedding_dim < 1:
        raise ValueError("embedding_dim must be a positive integer")
    embeddings = []
    targets = {key: np.zeros((len(rows), len(skill_names)), dtype=np.float32)
               for key in ("success", "cost", "violation")}
    valid = np.zeros((len(rows), len(skill_names)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        if isinstance(row, EffectLabel):
            embedding = row.embedding
            row_labels = {row.skill_name: row}
        elif isinstance(row, Mapping):
            embedding = np.asarray(row.get("embedding"), dtype=np.float32)
            row_labels = row.get("skills", {})
        else:
            raise TypeError("labels must contain EffectLabel or row mappings")
        if embedding.shape != (embedding_dim,) or not np.isfinite(embedding).all():
            raise ValueError("effect embedding shape or values are invalid")
        embeddings.append(embedding)
        if not isinstance(row_labels, Mapping):
            raise ValueError("row skills must be a mapping")
        for skill_index, skill in enumerate(skill_names):
            value = row_labels.get(skill)
            if value is None:
                continue
            if isinstance(value, Mapping):
                value = EffectLabel(**value)
            if not isinstance(value, EffectLabel) or not value.valid:
                continue
            if value.embedding.shape != (embedding_dim,):
                raise ValueError("skill label embedding shape mismatch")
            targets["success"][row_index, skill_index] = value.task_success
            targets["cost"][row_index, skill_index] = value.recovery_cost
            targets["violation"][row_index, skill_index] = value.violation
            valid[row_index, skill_index] = 1.0
    if not valid.any():
        raise ValueError("no valid skill labels remain after masking")
    return np.stack(embeddings).astype(np.float32), targets, {"valid": valid}


class EffectPredictor(nn.Module):
    """One multi-head output per registered skill.

    Labels with ``valid=False`` are masked out instead of being treated as
    negative skill outcomes.  The model consumes only deployment-side effects
    embeddings, goal features and remaining budget.
    """

    def __init__(self, embedding_dim: int, goal_dim: int, skill_names: tuple[str, ...], hidden_dim=128):
        super().__init__()
        if min(embedding_dim, goal_dim, hidden_dim) < 1 or not skill_names:
            raise ValueError("invalid effect predictor dimensions")
        self.embedding_dim, self.goal_dim, self.hidden_dim = int(embedding_dim), int(goal_dim), int(hidden_dim)
        self.skill_names = tuple(skill_names)
        self.body = nn.Sequential(nn.Linear(embedding_dim + goal_dim + 1, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.head = nn.Linear(hidden_dim, len(skill_names) * 3)
        # Metric supervision must update a trainable representation.  Comparing
        # the caller's raw embeddings would otherwise produce a loss with no
        # path to this predictor's parameters.
        self.metric_projection = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
                                               nn.Linear(hidden_dim, embedding_dim))

    def forward(self, embedding, goal, remaining_budget):
        embedding = torch.as_tensor(embedding, dtype=torch.float32, device=self.head.weight.device)
        goal = torch.as_tensor(goal, dtype=torch.float32, device=self.head.weight.device)
        budget = torch.as_tensor(remaining_budget, dtype=torch.float32, device=self.head.weight.device)
        if embedding.shape[-1] != self.embedding_dim or goal.shape[-1] != self.goal_dim:
            raise ValueError("effect predictor input dimensions do not match")
        values = self.head(self.body(torch.cat([embedding, goal, budget.reshape(*budget.shape, 1)], dim=-1)))
        return values.reshape(*values.shape[:-1], len(self.skill_names), 3)

    @torch.no_grad()
    def predict(self, embedding, goal, remaining_budget):
        values = self(embedding, goal, remaining_budget)
        return {skill: {"success_probability": float(torch.sigmoid(values[..., index, 0]).reshape(-1)[0]),
                        "cost": float(torch.relu(values[..., index, 1]).reshape(-1)[0]),
                        "violation_probability": float(torch.sigmoid(values[..., index, 2]).reshape(-1)[0])}
                for index, skill in enumerate(self.skill_names)}

    def masked_loss(self, values, labels: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray]):
        """Compute masked BCE/Huber loss for a batch of skill heads."""
        targets = [torch.as_tensor(labels[key], dtype=torch.float32, device=values.device) for key in ("success", "cost", "violation")]
        mask = torch.as_tensor(masks["valid"], dtype=torch.float32, device=values.device)
        if values.shape[-1] != 3 or values.shape[1] != len(self.skill_names):
            raise ValueError("values must be [batch, skills, 3]")
        total = values.new_tensor(0.0)
        count = values.new_tensor(0.0)
        for index, target in enumerate(targets):
            target = target.reshape(values.shape[0], values.shape[1])
            if index in (0, 2):
                loss = nn.functional.binary_cross_entropy_with_logits(values[..., index], target, reduction="none")
            else:
                loss = nn.functional.huber_loss(torch.relu(values[..., index]), target, reduction="none")
            total = total + (loss * mask).sum()
            count = count + mask.sum()
        return total / count.clamp_min(1.0)

    @staticmethod
    def normalized_effect_distance(left, right, *, cost_weight=1.0, violation_weight=1.0):
        """Distance target from independently measured skill outcomes in [0, 1]."""
        left = torch.as_tensor(left, dtype=torch.float32)
        right = torch.as_tensor(right, dtype=torch.float32)
        if left.shape != right.shape or left.shape[-1] != 3:
            raise ValueError("effect outcomes must have matching [..., 3] shapes")
        delta = left - right
        value = delta[..., 0].square() + cost_weight * delta[..., 1].square() + violation_weight * delta[..., 2].square()
        return value / (1.0 + cost_weight + violation_weight)

    def metric_loss(self, left_embedding, right_embedding, target_distance, pair_mask=None):
        """Huber loss for outcome-distance supervision; pair labels are external."""
        left_embedding = torch.as_tensor(left_embedding, dtype=torch.float32, device=self.head.weight.device)
        right_embedding = torch.as_tensor(right_embedding, dtype=torch.float32, device=self.head.weight.device)
        target_distance = torch.as_tensor(target_distance, dtype=torch.float32, device=self.head.weight.device)
        if left_embedding.shape != right_embedding.shape or left_embedding.shape[-1] != self.embedding_dim:
            raise ValueError("embedding pair shape does not match predictor")
        if target_distance.shape != left_embedding.shape[:-1] or torch.any((target_distance < 0) | (target_distance > 1)):
            raise ValueError("target distance must match pair batch and be in [0, 1]")
        left_embedding = self.metric_projection(left_embedding)
        right_embedding = self.metric_projection(right_embedding)
        left_norm = left_embedding / left_embedding.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        right_norm = right_embedding / right_embedding.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        predicted = (left_norm - right_norm).square().sum(-1) / 4.0
        loss = nn.functional.huber_loss(predicted, target_distance, reduction="none")
        if pair_mask is not None:
            pair_mask = torch.as_tensor(pair_mask, dtype=torch.float32, device=loss.device)
            if pair_mask.shape != loss.shape:
                raise ValueError("pair mask shape mismatch")
            loss = loss * pair_mask
            return loss.sum() / pair_mask.sum().clamp_min(1.0)
        return loss.mean()

    def representation_loss(self, values, labels, masks, left_embedding, right_embedding,
                            target_distance, *, metric_weight=0.0, pair_mask=None):
        prediction = self.masked_loss(values, labels, masks)
        if metric_weight < 0 or not np.isfinite(metric_weight):
            raise ValueError("metric_weight must be nonnegative and finite")
        metric = self.metric_loss(left_embedding, right_embedding, target_distance, pair_mask)
        return prediction + float(metric_weight) * metric
