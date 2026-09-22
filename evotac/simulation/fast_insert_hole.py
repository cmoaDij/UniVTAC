"""A deterministic surrogate for fast EvoTac control-loop validation.

The real task owns images, physics and contact state.  This surrogate keeps
the same public boundary: policies receive noisy visual, proprioceptive and
tactile signals, recovery actions are seven-dimensional normalized vectors,
each control advances six physics ticks, and only the terminal task outcome
is a reward.  It is useful for short CI checks and ablation sweeps while an
Isaac GPU is occupied; it is not evidence for TacEx physics or sim-to-real
performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FastSimulationConfig:
    physical_hz: int = 120
    control_hz: int = 20
    horizon_controls: int = 80
    recovery_horizon_controls: int = 10
    decimation: int = 6
    success_depth: float = 1.0
    success_error: float = 0.085
    loss_error: float = 0.42
    max_grip_error: float = 0.38

    def __post_init__(self):
        if (type(self.physical_hz) is not int or type(self.control_hz) is not int
                or self.physical_hz <= 0 or self.control_hz <= 0
                or self.physical_hz % self.control_hz):
            raise ValueError("physical_hz must be a positive multiple of control_hz")
        if self.decimation != self.physical_hz // self.control_hz:
            raise ValueError("decimation must match physical/control frequency")
        if (type(self.horizon_controls) is not int or type(self.recovery_horizon_controls) is not int
                or self.horizon_controls < 1 or self.recovery_horizon_controls < 1):
            raise ValueError("simulation horizons must be positive")
        thresholds = (self.success_depth, self.success_error, self.loss_error, self.max_grip_error)
        if not np.isfinite(thresholds).all() or self.success_depth <= 0 or self.success_error < 0:
            raise ValueError("simulation thresholds must be finite and nonnegative")
        if self.loss_error <= 0 or self.max_grip_error <= 0:
            raise ValueError("loss thresholds must be positive")


@dataclass
class FastEpisode:
    seed: int
    condition: str
    lateral: float
    angle: float
    depth: float
    grip_error: float
    controls: int = 0
    physics_steps: int = 0
    recovery_controls: int = 0
    recovered: bool = False
    done: bool = False
    reason: str = "running"
    selected_skill: str | None = None
    transitions: list[dict] = field(default_factory=list)


class FastInsertHoleEnv:
    """Small deterministic insert-hole model with observable-only sensors."""

    skill_names = ("reapproach", "lateral_adjust", "orientation_adjust")

    def __init__(self, config: FastSimulationConfig | None = None):
        self.config = config or FastSimulationConfig()
        self.rng = np.random.default_rng(0)
        self.episode: FastEpisode | None = None

    def reset(self, seed: int, condition: str = "nominal") -> dict:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if condition not in {"nominal", "C1", "C2", "C3"}:
            raise ValueError("condition must be nominal, C1, C2 or C3")
        self.rng = np.random.default_rng(seed)
        base = self.rng.normal(0.0, 0.028, size=3)
        offsets = {
            "nominal": (0.0, 0.0, 0.0),
            "C1": (0.16, 0.0, 0.0),
            "C2": (0.0, 0.18, 0.0),
            "C3": (0.15, 0.16, 0.12),
        }[condition]
        self.episode = FastEpisode(
            seed, condition, float(base[0] + offsets[0]), float(base[1] + offsets[1]),
            0.0, float(base[2] + offsets[2]),
        )
        return self.observe(tactile=True)

    @property
    def state(self) -> FastEpisode:
        if self.episode is None:
            raise RuntimeError("reset must be called before stepping")
        return self.episode

    def _error(self) -> float:
        episode = self.state
        return float(np.sqrt(episode.lateral * episode.lateral + episode.angle * episode.angle))

    def observe(self, *, tactile: bool = True) -> dict:
        episode = self.state
        visual = np.asarray(
            [episode.lateral, episode.angle, episode.depth, episode.grip_error],
            dtype=np.float32,
        )
        visual += self.rng.normal(0.0, [0.012, 0.012, 0.008, 0.01], size=4).astype(np.float32)
        touch = np.asarray([
            np.clip(episode.lateral / 0.35, -1, 1),
            np.clip(episode.angle / 0.35, -1, 1),
            np.clip(episode.grip_error / 0.35, -1, 1),
            float(episode.depth > 0.18),
        ], dtype=np.float32)
        if tactile:
            touch += self.rng.normal(0.0, 0.025, size=4).astype(np.float32)
        else:
            touch[:] = 0
        return {
            "vision": visual,
            "tactile": touch,
            "proprioception": np.asarray(
                [episode.depth, episode.grip_error,
                 episode.controls / self.config.horizon_controls],
                dtype=np.float32,
            ),
            "remaining_physics_steps": (self.config.horizon_controls - episode.controls)
            * self.config.decimation,
            "physics_step": episode.physics_steps,
        }

    def features(self, observation: dict, *, tactile: bool = True) -> dict:
        vision = np.asarray(observation["vision"], dtype=np.float32)
        touch = (np.asarray(observation["tactile"], dtype=np.float32)
                 if tactile else np.zeros(4, dtype=np.float32))
        risk = float(np.clip(max(abs(float(touch[0])), abs(float(touch[1])),
                               abs(float(vision[3])) * 0.8), 0, 1))
        blocked = float(np.clip(abs(float(touch[1])) * 1.15
                                + max(0.0, abs(float(vision[2])) - 0.45), 0, 1))
        return {
            "object_lost_risk": risk,
            "contact_blocked": blocked,
            "remaining_physics_steps": int(observation["remaining_physics_steps"]),
            "stable": risk < 0.28 and blocked < 0.28,
        }

    def _finish_if_terminal(self):
        episode, config = self.state, self.config
        if (self._error() < config.success_error
                and episode.grip_error < config.max_grip_error * 0.45
                and episode.depth >= config.success_depth):
            episode.done, episode.reason = True, "success"
        elif self._error() > config.loss_error or episode.grip_error > config.max_grip_error:
            episode.done, episode.reason = True, "object_lost"
        elif episode.controls >= config.horizon_controls:
            episode.done, episode.reason = True, "task_budget"

    def step_base(self, action: np.ndarray | None = None) -> dict:
        episode = self.state
        if episode.done:
            raise RuntimeError("episode already ended")
        observation = self.observe(tactile=True)
        visual = np.asarray(observation["vision"], dtype=np.float32)
        if action is None:
            action = np.asarray([
                -np.clip(visual[0] * 2.0, -1, 1),
                -np.clip(visual[1] * 1.8, -1, 1),
                0.0,
            ], dtype=np.float32)
        else:
            action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("base action must be three finite values in [-1, 1]")
        episode.lateral += float(action[0]) * 0.028 + self.rng.normal(0, 0.004)
        episode.angle += float(action[1]) * 0.024 + self.rng.normal(0, 0.004)
        episode.depth += 0.029 * (1.0 - min(1.0, self._error() / 0.5))
        episode.grip_error += 0.002 + self.rng.normal(0, 0.002)
        episode.controls += 1
        episode.physics_steps += self.config.decimation
        self._finish_if_terminal()
        return self.observe(tactile=True)

    def step_recovery(self, action: np.ndarray, skill_name: str | None = None) -> dict:
        episode = self.state
        if episode.done:
            raise RuntimeError("episode already ended")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("recovery action must be seven finite values in [-1, 1]")
        if episode.recovery_controls >= self.config.recovery_horizon_controls:
            raise RuntimeError("recovery budget exhausted")
        episode.lateral -= float(action[0]) * 0.055
        episode.angle -= float(action[1]) * 0.05
        episode.grip_error -= float(action[2]) * 0.035
        episode.depth += 0.008 * max(0.0, float(action[3]))
        episode.lateral += self.rng.normal(0, 0.002)
        episode.angle += self.rng.normal(0, 0.002)
        episode.grip_error = max(0.0, episode.grip_error + self.rng.normal(0, 0.002))
        episode.recovery_controls += 1
        episode.controls += 1
        episode.physics_steps += self.config.decimation
        episode.recovered = True
        episode.selected_skill = skill_name or episode.selected_skill
        self._finish_if_terminal()
        return self.observe(tactile=True)

    def outcome(self) -> dict:
        episode = self.state
        return {
            "seed": episode.seed,
            "condition": episode.condition,
            "reason": episode.reason,
            "success": episode.reason == "success",
            "controls": episode.controls,
            "physics_steps": episode.physics_steps,
            "recovery_controls": episode.recovery_controls,
            "recovered": episode.recovered,
            "selected_skill": episode.selected_skill,
            "terminal_error": self._error(),
            "terminal_grip_error": episode.grip_error,
        }


def run_episode(env: FastInsertHoleEnv, seed: int, condition: str,
                policy: Callable, *, tactile: bool = True) -> dict:
    """Run one surrogate episode with an auditable policy callback.

    ``policy`` receives ``(env, observation, tactile)`` and returns either
    ``("base", action_or_none)`` or ``("recovery", seven_vector, skill_name)``.
    """
    observation = env.reset(seed, condition)
    while not env.state.done:
        decision = policy(env, observation, tactile)
        if not isinstance(decision, tuple) or not decision:
            raise ValueError("policy must return a decision tuple")
        if decision[0] == "base":
            observation = env.step_base(decision[1] if len(decision) > 1 else None)
        elif decision[0] == "recovery":
            observation = env.step_recovery(
                decision[1], skill_name=decision[2] if len(decision) > 2 else None,
            )
        else:
            raise ValueError(f"unknown policy decision: {decision[0]}")
    return env.outcome()
