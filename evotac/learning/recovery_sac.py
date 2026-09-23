"""Minimal continuous-action SAC for one recovery skill.

This implementation is deliberately small: it provides one real interaction
and one gradient-update contract before any multi-skill or continual-learning
logic is introduced.
"""
from dataclasses import dataclass
from copy import deepcopy
import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def _mlp(input_dim, output_dim, hidden_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, output_dim))


class _GaussianActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim):
        super().__init__()
        self.body = _mlp(observation_dim, action_dim * 2, hidden_dim)
        self.action_dim = action_dim

    def distribution(self, observation):
        mean, log_std = self.body(observation).chunk(2, dim=-1)
        return Normal(mean, log_std.clamp(-5.0, 2.0).exp())

    def sample(self, observation, deterministic=False):
        distribution = self.distribution(observation)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = distribution.log_prob(pre_tanh).sum(-1)
        # Stable even when tanh saturates to exactly +/-1 in float32.
        log_prob -= (2 * (math.log(2) - pre_tanh - nn.functional.softplus(-2 * pre_tanh))).sum(-1)
        return action, log_prob


class _Critic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = _mlp(observation_dim + action_dim, 1, hidden_dim)

    def forward(self, observation, action):
        return self.net(torch.cat([observation, action], dim=-1)).squeeze(-1)


@dataclass
class SACUpdate:
    critic_loss: float
    actor_loss: float
    alpha: float
    batch_size: int


class RecoverySAC:
    """SAC actor/critics with normalized recovery actions in [-1, 1]."""

    def __init__(self, observation_dim: int, action_dim: int, *, hidden_dim=128,
                 discount=0.99, tau=0.005, alpha=0.2, learning_rate=3e-4,
                 device="cpu", seed=0):
        if any(type(x) is not int or x < 1 for x in (observation_dim, action_dim, hidden_dim)):
            raise ValueError("network dimensions must be positive")
        if (not 0 <= discount <= 1 or not 0 < tau <= 1
                or not np.isfinite([alpha, learning_rate]).all() or min(alpha, learning_rate) <= 0):
            raise ValueError("invalid SAC discount, tau, alpha or learning rate")
        torch.manual_seed(seed)
        self.network_config = {"observation_dim": observation_dim, "action_dim": action_dim,
                               "hidden_dim": hidden_dim, "discount": discount, "tau": tau,
                               "alpha": alpha, "learning_rate": learning_rate}
        self.observation_dim, self.action_dim = int(observation_dim), int(action_dim)
        self.discount, self.tau = float(discount), float(tau)
        self.device = torch.device(device)
        self.actor = _GaussianActor(observation_dim, action_dim, hidden_dim).to(self.device)
        self.critic_one = _Critic(observation_dim, action_dim, hidden_dim).to(self.device)
        self.critic_two = _Critic(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target_one = _Critic(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target_two = _Critic(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target_one.load_state_dict(self.critic_one.state_dict())
        self.target_two.load_state_dict(self.critic_two.state_dict())
        self.target_one.requires_grad_(False)
        self.target_two.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(list(self.critic_one.parameters()) + list(self.critic_two.parameters()), lr=learning_rate)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = -float(action_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _tensor(self, value):
        # HDF5-backed observations can be read-only.  Explicitly copy them so
        # downstream tensor operations never inherit an undefined writable
        # alias from ``torch.as_tensor``.
        array = np.array(value, dtype=np.float32, copy=True)
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (self.observation_dim,) or not np.isfinite(values).all():
            raise ValueError(f"observation must have shape ({self.observation_dim},) and be finite")
        action, _ = self.actor.sample(self._tensor(values).unsqueeze(0), deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Mapping[str, np.ndarray]) -> SACUpdate:
        required = {"observation", "action", "reward", "next_observation", "terminated", "truncated", "bootstrap_allowed"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"missing replay fields: {sorted(missing)}")
        observation = self._tensor(batch["observation"])
        action = self._tensor(batch["action"])
        reward = self._tensor(batch["reward"])
        next_observation = self._tensor(batch["next_observation"])
        bootstrap = self._tensor(batch["bootstrap_allowed"])
        terminated, truncated = self._tensor(batch["terminated"]), self._tensor(batch["truncated"])
        if observation.ndim != 2 or not observation.shape[0] or observation.shape[-1] != self.observation_dim:
            raise ValueError("batch observation shape does not match SAC")
        if action.shape != (observation.shape[0], self.action_dim):
            raise ValueError("batch action shape does not match SAC")
        if next_observation.shape != observation.shape or any(x.shape != (observation.shape[0],) for x in (reward, bootstrap, terminated, truncated)):
            raise ValueError("batch next observation or scalar field shape mismatch")
        if any(not torch.isfinite(x).all() for x in (observation, action, reward, next_observation, bootstrap, terminated, truncated)):
            raise ValueError("all batch values must be finite")
        if torch.any(action.abs() > 1):
            raise ValueError("critic must consume the exact normalized action, without clipping")
        if any(not ((x == 0) | (x == 1)).all() for x in (bootstrap, terminated, truncated)) or torch.any(bootstrap * terminated):
            raise ValueError("invalid terminal/bootstrap flags")
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_observation)
            target = torch.minimum(self.target_one(next_observation, next_action),
                                   self.target_two(next_observation, next_action))
            target = reward + bootstrap * self.discount * (target - self.alpha.detach() * next_log_prob)
        current_one = self.critic_one(observation, action)
        current_two = self.critic_two(observation, action)
        critic_loss = nn.functional.mse_loss(current_one, target) + nn.functional.mse_loss(current_two, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor update should not retain critic gradients or waste memory.
        self.critic_one.requires_grad_(False)
        self.critic_two.requires_grad_(False)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.critic_one.requires_grad_(False)
        self.critic_two.requires_grad_(False)
        policy_action, log_prob = self.actor.sample(observation)
        actor_value = torch.minimum(self.critic_one(observation, policy_action), self.critic_two(observation, policy_action))
        actor_loss = (self.alpha.detach() * log_prob - actor_value).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_one.requires_grad_(True)
        self.critic_two.requires_grad_(True)
        self.critic_one.requires_grad_(True)
        self.critic_two.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target, source in ((self.target_one, self.critic_one), (self.target_two, self.critic_two)):
                for target_param, source_param in zip(target.parameters(), source.parameters()):
                    target_param.mul_(1.0 - self.tau).add_(self.tau * source_param)
        return SACUpdate(float(critic_loss.item()), float(actor_loss.item()), float(self.alpha.item()), int(observation.shape[0]))

    def state_dict(self):
        return deepcopy({"schema": "evotac.sac.v2", "network_config": self.network_config,
                "actor": self.actor.state_dict(), "critic_one": self.critic_one.state_dict(),
                "critic_two": self.critic_two.state_dict(), "target_one": self.target_one.state_dict(),
                "target_two": self.target_two.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
                "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict()})

    def load_state_dict(self, state):
        if state.get("schema") != "evotac.sac.v2" or state.get("network_config") != self.network_config:
            raise ValueError("SAC checkpoint schema or network/training configuration mismatch")
        for name in ("actor", "critic_one", "critic_two", "target_one", "target_two"):
            getattr(self, name).load_state_dict(state[name], strict=True)
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
            getattr(self, name).load_state_dict(state[name])


class MultiSkillRecoverySAC:
    """Independent SAC learners for the three complementary recovery skills.

    Critics, actors, entropy temperatures and replay updates are intentionally
    separate per skill.  Sharing an untyped optimizer would let one skill's
    recovery outcomes change another skill's policy and would invalidate the
    paired effect labels used by P4.
    """

    def __init__(self, skill_names, observation_dim, action_dim=7, *, hidden_dim=128,
                 discount=0.99, tau=0.005, alpha=0.2, learning_rate=3e-4,
                 device="cpu", seed=0):
        names = tuple(skill_names)
        if not names or len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("skill_names must be unique nonempty strings")
        self.skill_names = names
        self.config = {"observation_dim": observation_dim, "action_dim": action_dim,
                       "hidden_dim": hidden_dim, "discount": discount, "tau": tau,
                       "alpha": alpha, "learning_rate": learning_rate, "device": str(device)}
        self.learners = {name: RecoverySAC(observation_dim, action_dim, hidden_dim=hidden_dim,
                                           discount=discount, tau=tau, alpha=alpha,
                                           learning_rate=learning_rate, device=device, seed=seed + index)
                         for index, name in enumerate(names)}

    def _get(self, skill_name):
        if skill_name not in self.learners:
            raise KeyError(f"unknown recovery skill: {skill_name}")
        return self.learners[skill_name]

    def act(self, skill_name, observation, deterministic=False):
        return self._get(skill_name).act(observation, deterministic=deterministic)

    def update(self, skill_name, batch):
        return self._get(skill_name).update(batch)

    def state_dict(self):
        return {"schema": "evotac.multi_skill_sac.v1", "skill_names": self.skill_names,
                "config": self.config,
                "learners": {name: learner.state_dict() for name, learner in self.learners.items()}}

    def load_state_dict(self, state):
        if state.get("schema") != "evotac.multi_skill_sac.v1" or tuple(state.get("skill_names", ())) != self.skill_names:
            raise ValueError("multi-skill checkpoint schema or skills mismatch")
        if state.get("config") != self.config:
            raise ValueError("multi-skill checkpoint configuration mismatch")
        for name in self.skill_names:
            self.learners[name].load_state_dict(state["learners"][name])
