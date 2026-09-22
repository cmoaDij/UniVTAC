"""Masked effect-predictor training with versioned checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from evotac.learning.effect_model import EffectHistoryEncoder, EffectPredictor


@dataclass(frozen=True)
class EffectTrainingStep:
    loss: float
    prediction_loss: float
    metric_loss: float
    valid_labels: int


class EffectTrainer:
    """Train P4 output heads without treating missing branches as failures."""

    def __init__(self, predictor: EffectPredictor, *, learning_rate=3e-4,
                 metric_weight=0.0, device="cpu", effect_encoder=None):
        if not isinstance(predictor, EffectPredictor):
            raise TypeError("predictor must be an EffectPredictor")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(metric_weight) or metric_weight < 0:
            raise ValueError("metric_weight must be nonnegative and finite")
        self.predictor = predictor.to(device)
        if effect_encoder is not None and not isinstance(effect_encoder, EffectHistoryEncoder):
            raise TypeError("effect_encoder must be an EffectHistoryEncoder")
        if effect_encoder is not None and effect_encoder.config["embedding_dim"] != predictor.embedding_dim:
            raise ValueError("effect encoder and predictor embedding dimensions differ")
        self.effect_encoder = None if effect_encoder is None else effect_encoder.to(device)
        self.device = torch.device(device)
        self.learning_rate = float(learning_rate)
        self.metric_weight = float(metric_weight)
        parameters = list(self.predictor.parameters())
        if self.effect_encoder is not None:
            parameters += list(self.effect_encoder.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)

    def train_batch(self, batch):
        required = {"goal", "remaining_budget", "labels", "masks"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"effect batch is missing: {sorted(missing)}")
        if "effect_history" in batch:
            if self.effect_encoder is None:
                raise ValueError("effect_history requires an effect_encoder")
            history = batch["effect_history"]
            embedding = self.effect_encoder(history["frames"], history["mask"])
        elif "embedding" in batch:
            embedding = batch["embedding"]
        else:
            raise ValueError("effect batch requires embedding or effect_history")
        values = self.predictor(embedding, batch["goal"], batch["remaining_budget"])
        prediction_loss = self.predictor.masked_loss(values, batch["labels"], batch["masks"])
        metric_loss = values.new_tensor(0.0)
        if self.metric_weight:
            pair = batch.get("metric_pair")
            if not isinstance(pair, dict):
                raise ValueError("metric_pair is required when metric_weight is nonzero")
            if "left_history" in pair or "right_history" in pair:
                if self.effect_encoder is None or not {"left_history", "right_history"} <= pair.keys():
                    raise ValueError("paired effect histories require an effect_encoder")
                left = self.effect_encoder(pair["left_history"]["frames"], pair["left_history"]["mask"])
                right = self.effect_encoder(pair["right_history"]["frames"], pair["right_history"]["mask"])
            else:
                left, right = pair["left_embedding"], pair["right_embedding"]
            metric_loss = self.predictor.metric_loss(
                left, right, pair["target_distance"],
                pair.get("pair_mask"))
        loss = prediction_loss + self.metric_weight * metric_loss
        if not torch.isfinite(loss):
            raise ValueError("effect loss is not finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        valid = int(np.asarray(batch["masks"]["valid"]).sum())
        return EffectTrainingStep(float(loss.detach().cpu()), float(prediction_loss.detach().cpu()),
                                  float(metric_loss.detach().cpu()), valid)

    def state_dict(self):
        return {
            "schema": "evotac.effect_trainer.v1",
            "predictor": self.predictor.state_dict(),
            "predictor_config": {"embedding_dim": self.predictor.embedding_dim,
                                  "goal_dim": self.predictor.goal_dim,
                                  "skill_names": self.predictor.skill_names,
                                  "hidden_dim": self.predictor.hidden_dim},
            "learning_rate": self.learning_rate,
            "metric_weight": self.metric_weight,
            "effect_encoder": None if self.effect_encoder is None else self.effect_encoder.checkpoint(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("schema") != "evotac.effect_trainer.v1":
            raise ValueError("unsupported effect trainer checkpoint schema")
        expected = {"embedding_dim": self.predictor.embedding_dim,
                    "goal_dim": self.predictor.goal_dim,
                    "skill_names": self.predictor.skill_names,
                    "hidden_dim": self.predictor.hidden_dim}
        config = state.get("predictor_config", {})
        if config != expected or float(state["learning_rate"]) != self.learning_rate \
                or float(state["metric_weight"]) != self.metric_weight:
            raise ValueError("effect trainer configuration does not match checkpoint")
        self.predictor.load_state_dict(state["predictor"], strict=True)
        if (state.get("effect_encoder") is None) != (self.effect_encoder is None):
            raise ValueError("effect encoder presence does not match checkpoint")
        if self.effect_encoder is not None:
            self.effect_encoder.load_state_dict(state["effect_encoder"]["state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path):
        self.load_state_dict(torch.load(Path(path), map_location=self.device, weights_only=False))
