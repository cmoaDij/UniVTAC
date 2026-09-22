"""Separate trainable temporal encoders for control and recovery effects."""
from copy import deepcopy

import torch
from torch import nn

from evotac.learning.recovery_observation import HISTORY_LENGTH, RECOVERY_OBSERVATION_DIM


class HistoryEncoder(nn.Module):
    """Recompute an eight-frame causal GRU state with explicit padding masks.

    Each branch owns its own instance; no mutable recurrent state survives a
    call. Training, replay reconstruction and online inference consequently
    use the same window, including its observed sampling intervals.
    """

    def __init__(self, input_dim=RECOVERY_OBSERVATION_DIM, hidden_dim=128,
                 output_dim=128, history_length=HISTORY_LENGTH):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (input_dim, hidden_dim, output_dim, history_length)):
            raise ValueError("history dimensions must be positive integers")
        self.config = dict(input_dim=input_dim, hidden_dim=hidden_dim,
                           output_dim=output_dim, history_length=history_length)
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Identity() if output_dim == hidden_dim else nn.Linear(hidden_dim, output_dim)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))
        self._frozen = False

    def _inputs(self, frames, mask):
        frames = torch.as_tensor(frames, dtype=torch.float32, device=self.input_mean.device)
        mask = torch.as_tensor(mask, device=frames.device)
        expected = (self.config["history_length"], self.config["input_dim"])
        if frames.ndim != 3 or tuple(frames.shape[1:]) != expected or not len(frames):
            raise ValueError(f"history must have shape [batch, {expected[0]}, {expected[1]}]")
        if mask.dtype != torch.bool or mask.shape != frames.shape[:2] or not mask.any(-1).all():
            raise ValueError("history mask must be boolean with at least one valid frame per sample")
        if not torch.isfinite(frames[mask]).all():
            raise ValueError("valid history frames must be finite")
        return torch.where(mask[..., None], frames, 0.0), mask

    @torch.no_grad()
    def fit_normalization(self, frames, mask):
        """Call only on training parents before freezing the control branch."""
        if self._frozen:
            raise RuntimeError("cannot refit a frozen control representation")
        frames, mask = self._inputs(frames, mask)
        valid = frames[mask]
        self.input_mean.copy_(valid.mean(0))
        self.input_scale.copy_(valid.std(0, unbiased=False).clamp_min(1e-3))

    def forward(self, frames, mask):
        frames, mask = self._inputs(frames, mask)
        projected = self.projection((frames - self.input_mean) / self.input_scale)
        hidden = projected.new_zeros((len(frames), self.config["hidden_dim"]))
        for index in range(self.config["history_length"]):
            proposal = self.gru(projected[:, index], hidden)
            hidden = torch.where(mask[:, index, None], proposal, hidden)
        return self.output(hidden)

    def freeze(self):
        self._frozen = True
        self.requires_grad_(False)
        self.eval()
        return self

    def train(self, mode=True):
        return super().train(False if self._frozen else mode)

    def checkpoint(self):
        return deepcopy({"schema": "evotac.history_encoder.v1", "config": self.config,
                         "state_dict": self.state_dict()})

    @classmethod
    def from_checkpoint(cls, state, *, device="cpu", frozen=False):
        if state.get("schema") != "evotac.history_encoder.v1":
            raise ValueError("unsupported history encoder schema")
        model = cls(**state["config"]).to(device)
        model.load_state_dict(state["state_dict"], strict=True)
        return model.freeze() if frozen else model
